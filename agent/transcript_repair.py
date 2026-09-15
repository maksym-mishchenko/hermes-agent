"""Transcript repair for SessionDB batch appends: reconcile in-memory assistant rows with committed SQLite
rows (blank-row in-place update, concurrent-winner adoption, watermark-compaction clone lookup) and sync
markers after commit."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Callable, Dict, List

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_sanitization import (
    coalesce_tool_call_id,
    deterministic_call_id,
    tool_call_id_variants,
    tool_result_id_variants,
    uniquify_tool_call_ids,
)


_TRANSCRIPT_REPAIR_VERSION = 1
_TRANSCRIPT_REPAIR_CONFIG_KEY = "_transcript_repair_version"
_INTERRUPTED_PLACEHOLDER = "[response interrupted]"


def is_content_blank(content: Any) -> bool:
    """True when decoded message content is None, whitespace-only, or has no visible text parts."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return False


def _json_safe_db_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"encoding": "base64", "value": base64.b64encode(value).decode("ascii")}
    return value


def _write_repair_archive(
    db_path: Path,
    session_id: str,
    rows: List[Dict[str, Any]],
    *,
    model_config: Any,
) -> Path:
    """Write the exact pre-migration rows once, mode 0600, before the transaction mutates them."""
    archive_dir = db_path.parent / "transcript-repair-archives"
    archive_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(archive_dir, 0o700)
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    high_water = max((int(row["id"]) for row in rows), default=0)
    path = archive_dir / f"v{_TRANSCRIPT_REPAIR_VERSION}-{session_hash}-{high_water}.json"
    if path.exists():
        try:
            if isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                os.chmod(path, 0o600)
                return path
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    payload = {
        "version": _TRANSCRIPT_REPAIR_VERSION,
        "session_hash": session_hash,
        "created_at": time.time(),
        "model_config": _json_safe_db_value(model_config),
        "rows": [{key: _json_safe_db_value(value) for key, value in row.items()} for row in rows],
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=archive_dir)
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(archive_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    return path


def _parse_tool_calls(raw: Any) -> tuple[List[Dict[str, Any]], bool]:
    if raw in (None, ""):
        return [], False
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], True
    if not isinstance(parsed, list):
        return [], True
    return [item for item in parsed if isinstance(item, dict)], len(parsed) != len(
        [item for item in parsed if isinstance(item, dict)]
    )


def _safe_arguments(raw: Any) -> tuple[str, bool]:
    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return "{}", True
    try:
        json.loads(text)
        return text, False
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            return json.dumps(json.loads(text, strict=False), separators=(",", ":")), True
        except (json.JSONDecodeError, TypeError, ValueError):
            return "{}", True


def _normalize_tool_run(rows: List[sqlite3.Row], index: int) -> tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    """Normalize one assistant call batch and its immediately following tool results."""
    row = rows[index]
    calls, malformed = _parse_tool_calls(row["tool_calls"])
    updates: Dict[int, Dict[str, Any]] = {}
    counts = {"tool_call_rows": 0, "tool_result_rows": 0, "arguments": 0, "ids": 0}
    if not calls:
        if malformed or row["tool_calls"] not in (None, ""):
            updates[int(row["id"])] = {"tool_calls": None}
            counts["tool_call_rows"] = 1
        return updates, counts

    followers: List[sqlite3.Row] = []
    for candidate in rows[index + 1:]:
        if candidate["role"] != "tool":
            break
        followers.append(candidate)

    normalized_calls: List[Dict[str, Any]] = []
    changed = malformed
    for call_index, original in enumerate(calls):
        call = dict(original)
        raw_function = call.get("function")
        function = dict(raw_function) if isinstance(raw_function, dict) else {}
        name = function.get("name")
        normalized_name = (
            name.strip()
            if isinstance(name, str) and name.strip()
            else "_invalid_tool_call"
        )
        arguments, args_changed = _safe_arguments(function.get("arguments"))
        function.update({"name": normalized_name, "arguments": arguments})
        call["function"] = function
        if not tool_call_id_variants(call):
            follower_id = (
                followers[call_index]["tool_call_id"]
                if call_index < len(followers)
                else ""
            )
            follower_id = follower_id.strip() if isinstance(follower_id, str) else ""
            follower_variants = tool_result_id_variants(follower_id)
            other_variants = set().union(
                *(
                    tool_call_id_variants(candidate)
                    for candidate_index, candidate in enumerate(calls)
                    if candidate_index != call_index
                )
            )
            call["id"] = (
                follower_id
                if follower_variants and not follower_variants & other_variants
                else deterministic_call_id(
                    normalized_name, arguments, int(row["id"]) + call_index
                )
            )
        if (
            not isinstance(raw_function, dict)
            or normalized_name != name
            or args_changed
        ):
            changed = True
        if args_changed:
            counts["arguments"] += 1
        normalized_calls.append(call)

    original_identities = [
        (
            call.get("id"),
            call.get("call_id"),
            call.get("response_item_id"),
        )
        for call in normalized_calls
    ]
    original_variants = [tool_call_id_variants(call) for call in normalized_calls]
    uniquify_tool_call_ids(normalized_calls)
    for before, call in zip(original_identities, normalized_calls):
        after = (
            call.get("id"),
            call.get("call_id"),
            call.get("response_item_id"),
        )
        if after != before:
            changed = True
            counts["ids"] += 1

    unmatched = set(range(len(normalized_calls)))
    for follower_index, follower in enumerate(followers):
        result_variants = tool_result_id_variants(follower["tool_call_id"])
        candidates = [
            call_index
            for call_index in sorted(unmatched)
            if result_variants & original_variants[call_index]
        ]
        call_index = candidates[0] if candidates else None
        if call_index is None and not result_variants and follower_index in unmatched:
            call_index = follower_index
        if call_index is None:
            continue
        unmatched.remove(call_index)
        normalized_variants = tool_call_id_variants(normalized_calls[call_index])
        if result_variants & normalized_variants:
            continue
        call_id = coalesce_tool_call_id(normalized_calls[call_index])
        if call_id and follower["tool_call_id"] != call_id:
            updates[int(follower["id"])] = {"tool_call_id": call_id}
            counts["tool_result_rows"] += 1

    if changed:
        updates[int(row["id"])] = {
            **updates.get(int(row["id"]), {}),
            "tool_calls": json.dumps(normalized_calls, ensure_ascii=True, separators=(",", ":")),
        }
        counts["tool_call_rows"] = 1
    return updates, counts


def migrate_persisted_transcript(
    conn: sqlite3.Connection,
    db_path: Path,
    session_id: str,
    *,
    encode_content_fn: Callable[[Any], Any],
    decode_content_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Repair one legacy replay transcript once, in the caller's write transaction."""
    session = conn.execute(
        "SELECT model_config FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if session is None:
        return {"state": "missing", "rows_updated": 0}
    raw_model_config = session["model_config"]
    try:
        model_config = json.loads(raw_model_config or "{}")
    except (json.JSONDecodeError, TypeError):
        model_config = {}
    if not isinstance(model_config, dict):
        model_config = {}
    if int(model_config.get(_TRANSCRIPT_REPAIR_CONFIG_KEY, 0) or 0) >= _TRANSCRIPT_REPAIR_VERSION:
        return {"state": "already_current", "rows_updated": 0}

    rows = conn.execute(
        "SELECT * "
        "FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
        (session_id,),
    ).fetchall()
    updates: Dict[int, Dict[str, Any]] = {}
    counts = {
        "empty_rows": 0, "tool_call_rows": 0, "tool_result_rows": 0,
        "arguments": 0, "ids": 0,
    }
    for index, row in enumerate(rows):
        if row["role"] == "assistant":
            run_updates, run_counts = _normalize_tool_run(rows, index)
            for row_id, patch in run_updates.items():
                updates.setdefault(row_id, {}).update(patch)
            for key, value in run_counts.items():
                counts[key] += value
        decoded_content = decode_content_fn(row["content"])
        effective_tool_calls = updates.get(int(row["id"]), {}).get("tool_calls", row["tool_calls"])
        calls, _ = _parse_tool_calls(effective_tool_calls)
        has_sidecar_payload = any(
            row[key] not in (None, "", "[]", "{}")
            for key in (
                "reasoning", "reasoning_content", "reasoning_details",
                "codex_reasoning_items", "codex_message_items", "api_content",
            )
        )
        if (
            row["role"] in {"assistant", "user"}
            and is_content_blank(decoded_content)
            and not calls
            and not has_sidecar_payload
        ):
            updates.setdefault(int(row["id"]), {})["content"] = encode_content_fn(
                _INTERRUPTED_PLACEHOLDER
            )
            counts["empty_rows"] += 1

    archive_path = None
    if updates:
        archived = [dict(row) for row in rows if int(row["id"]) in updates]
        archive_path = _write_repair_archive(
            Path(db_path), session_id, archived, model_config=raw_model_config
        )
        for row_id, patch in updates.items():
            assignments = ", ".join(f"{column} = ?" for column in patch)
            conn.execute(
                f"UPDATE messages SET {assignments} WHERE id = ? AND session_id = ? AND active = 1",
                (*patch.values(), row_id, session_id),
            )

    model_config[_TRANSCRIPT_REPAIR_CONFIG_KEY] = _TRANSCRIPT_REPAIR_VERSION
    model_config["_transcript_repair_counts"] = counts
    conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        (json.dumps(model_config), session_id),
    )
    return {
        "state": "migrated" if updates else "clean",
        "rows_updated": len(updates),
        "archive_path": str(archive_path) if archive_path else None,
        **counts,
    }


def resolve_and_repair_transcript_batch(
    conn: sqlite3.Connection,
    session_id: str,
    messages: List[Dict[str, Any]],
    encode_content_fn: Callable[[Any], Any],
    decode_content_fn: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    """Partition a message batch within an active write transaction. An assistant message carrying an
    existing integer ``_row_id`` targets its active SQLite row (or the active clone a watermark compaction
    made of it): a blank row is updated in place; a non-blank one (concurrent winner) has its canonical
    content adopted without overwrite. Returns the messages that must be inserted as fresh rows."""
    inserted_rows: List[Dict[str, Any]] = []
    for msg in messages:
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        target_row = None
        if isinstance(existing_row_id, int) and msg.get("role", "unknown") == "assistant":
            target_row = _active_assistant_row(conn, session_id, existing_row_id)
        if target_row is None:
            inserted_rows.append(msg)
            continue
        target_id = int(target_row["id"])
        decoded = decode_content_fn(target_row["content"])
        msg["_row_id"] = target_id
        if is_content_blank(decoded):
            conn.execute(
                "UPDATE messages SET content = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (encode_content_fn(msg.get("content")), target_id, session_id),
            )
        else:
            msg["_canonical_content"] = decoded  # concurrent winner: adopt, don't overwrite
        if isinstance(msg.get("tool_calls"), list):
            # Repairs to arguments/IDs happen on a loaded assistant dict. The
            # row marker must not make that in-place repair process-only.
            conn.execute(
                "UPDATE messages SET tool_calls = ? "
                "WHERE id = ? AND session_id = ? AND active = 1",
                (json.dumps(msg["tool_calls"]), target_id, session_id),
            )
    return inserted_rows


def _active_assistant_row(conn: sqlite3.Connection, session_id: str, row_id: int):
    """The active assistant row for ``row_id``, or the active clone a watermark compaction made of it."""
    row = conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE id = ? AND session_id = ?",
        (row_id, session_id),
    ).fetchone()
    if row is None or row["role"] != "assistant":
        return None
    if int(row["active"] or 0) == 1:
        return row
    # Watermark compaction soft-archived the concurrent tail and cloned it.
    return conn.execute(
        "SELECT id, role, active, timestamp, content FROM messages "
        "WHERE session_id = ? AND active = 1 AND role = 'assistant' "
        "AND timestamp IS ? AND id != ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id, row["timestamp"], row["id"]),
    ).fetchone()


def sync_flushed_message_markers(batch_msgs: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> None:
    """Stamp _DB_PERSISTED_MARKER and sync canonical row ID / content onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        if "_canonical_content" in row:
            written["content"] = row["_canonical_content"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
