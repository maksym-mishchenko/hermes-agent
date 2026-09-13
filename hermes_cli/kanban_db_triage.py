"""Atomic domain operation for releasing a task from Kanban triage."""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Optional

from hermes_cli import kanban_db as kb


def server_actor() -> str:
    """Return the process identity used for native Kanban audit records."""
    return (
        (os.environ.get("HERMES_PROFILE") or "").strip()
        or (os.environ.get("USER") or "").strip()
        or "orchestrator"
    )


def release_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: str,
    actor: Optional[str] = None,
) -> dict[str, Any]:
    """Release exactly one ``triage`` task, preserving its graph and history.

    The status and audit event are committed in the same immediate transaction.
    All eligibility checks are repeated while that transaction is held, so a
    concurrent writer cannot release a task with a claim or open run.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is required")
    reason = reason.strip()
    actor = (actor if actor is not None else server_actor())
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor is required")
    actor = actor.strip()

    with kb.write_txn(conn):
        row = conn.execute(
            """
            SELECT status, claim_lock, claim_expires, worker_pid, current_run_id
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"task {task_id} not found")
        if row["status"] != "triage":
            raise ValueError("only triage tasks can be released")

        dangling_claim = any(
            row[column] is not None
            for column in ("claim_lock", "claim_expires", "worker_pid", "current_run_id")
        )
        if dangling_claim:
            raise ValueError(f"cannot release {task_id}: task has a dangling claim")

        open_run = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL LIMIT 1",
            (task_id,),
        ).fetchone()
        if open_run is not None:
            raise ValueError(f"cannot release {task_id}: task has a live run")

        new_status = "ready" if kb._parents_satisfied(conn, task_id) else "todo"
        updated = conn.execute(
            """
            UPDATE tasks
               SET status = ?
             WHERE id = ?
               AND status = 'triage'
               AND claim_lock IS NULL
               AND claim_expires IS NULL
               AND worker_pid IS NULL
               AND current_run_id IS NULL
            """,
            (new_status, task_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"task {task_id} changed before it could be released")

        kb._append_event(
            conn,
            task_id,
            "triage_released",
            {
                "actor": actor,
                "reason": reason,
                "prior_status": "triage",
                "new_status": new_status,
            },
        )

    return {
        "task_id": task_id,
        "prior_status": "triage",
        "status": new_status,
        "reason": reason,
        "actor": actor,
    }
