"""Tests for guarded native release of triage Kanban tasks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_triage as kbt
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def conn(tmp_path: Path):
    connection = connect(tmp_path / "kanban.db")
    try:
        yield connection
    finally:
        connection.close()


def _enable_orchestrator(home: Path) -> None:
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")


def _triage(conn, title: str = "triage", *, parents=(), assignee="reviewer") -> str:
    return kb.create_task(
        conn,
        title=title,
        body="original body",
        assignee=assignee,
        created_by="coordinator",
        parents=parents,
        triage=True,
        completion_contract="local-only",
    )


def test_release_triage_lands_ready_and_audits_server_actor(conn):
    task_id = _triage(conn)

    result = kbt.release_triage_task(
        conn, task_id, reason="reviewed and ready for dispatch", actor="orchestrator"
    )

    assert result == {
        "task_id": task_id,
        "prior_status": "triage",
        "status": "ready",
        "reason": "reviewed and ready for dispatch",
        "actor": "orchestrator",
    }
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "reviewer"
    assert task.completion_contract == "local-only"
    event = kb.list_events(conn, task_id)[-1]
    assert event.kind == "triage_released"
    assert event.payload == {
        "actor": "orchestrator",
        "reason": "reviewed and ready for dispatch",
        "prior_status": "triage",
        "new_status": "ready",
    }


def test_release_triage_lands_todo_when_parent_is_open_and_preserves_graph(conn):
    parent = kb.create_task(conn, title="open parent", assignee="worker")
    task_id = _triage(conn, parents=(parent,))
    before = kb.get_task(conn, task_id)
    assert before is not None

    result = kbt.release_triage_task(conn, task_id, reason="scope confirmed", actor="coord")

    assert result["status"] == "todo"
    after = kb.get_task(conn, task_id)
    assert after is not None
    assert after.status == "todo"
    assert after.assignee == before.assignee
    assert after.title == before.title
    assert after.body == before.body
    assert after.completion_contract == before.completion_contract
    assert kb.parent_ids(conn, task_id) == [parent]


def test_release_triage_rejects_blank_or_non_triage_without_mutation(conn):
    task_id = _triage(conn)
    before = list(conn.iterdump())

    with pytest.raises(ValueError, match="reason"):
        kbt.release_triage_task(conn, task_id, reason=" ", actor="operator")
    assert list(conn.iterdump()) == before

    for status in ("todo", "ready", "running", "blocked", "review", "done", "archived"):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()
        before = list(conn.iterdump())
        with pytest.raises(ValueError, match="triage"):
            kbt.release_triage_task(conn, task_id, reason="not allowed", actor="operator")
        assert list(conn.iterdump()) == before
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (task_id,))
        conn.commit()


def test_release_triage_rejects_any_live_or_dangling_claim_without_mutation(conn):
    task_id = _triage(conn)
    conn.execute(
        "UPDATE tasks SET claim_lock = 'host:claim', claim_expires = 999, worker_pid = 42 "
        "WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    before = list(conn.iterdump())

    with pytest.raises(ValueError, match="claim"):
        kbt.release_triage_task(conn, task_id, reason="unsafe", actor="operator")
    assert list(conn.iterdump()) == before


def test_release_triage_rejects_open_run_even_without_task_claim_columns(conn):
    task_id = _triage(conn)
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', 1)",
        (task_id,),
    )
    conn.commit()
    before = list(conn.iterdump())

    with pytest.raises(ValueError, match="live run"):
        kbt.release_triage_task(conn, task_id, reason="unsafe", actor="operator")
    assert list(conn.iterdump()) == before


def test_release_triage_preserves_failure_block_and_review_history(conn):
    task_id = _triage(conn)
    conn.execute(
        "UPDATE tasks SET consecutive_failures = 3, block_kind = 'needs_input', "
        "block_recurrences = 2, last_failure_error = 'retained' WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    kb._append_event(conn, task_id, "review_requested", {"reviewer": "reviewer"})
    conn.commit()
    before_events = len(kb.list_events(conn, task_id))

    kbt.release_triage_task(conn, task_id, reason="retain history", actor="coord")

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.consecutive_failures == 3
    assert task.block_kind == "needs_input"
    assert task.block_recurrences == 2
    assert task.last_failure_error == "retained"
    assert len(kb.list_events(conn, task_id)) == before_events + 1
    assert any(event.kind == "review_requested" for event in kb.list_events(conn, task_id))


def test_release_triage_tool_dispatches_through_registry(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _enable_orchestrator(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "coord")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with connect() as connection:
        task_id = _triage(connection)

    from tools import kanban_tools_triage  # noqa: F401
    from tools.registry import registry

    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    result = json.loads(str(registry.dispatch(
        "kanban_release_triage",
        {"task_id": task_id, "reason": "registry path"},
    )))
    assert result["ok"] is True
    assert result["status"] == "ready"


def test_release_triage_tool_is_visible_only_to_orchestrator_mode(monkeypatch):
    from tools import kanban_tools as base_tools
    from tools import kanban_tools_triage  # noqa: F401
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    monkeypatch.setattr(base_tools, "_profile_has_kanban_toolset", lambda: True)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    invalidate_check_fn_cache()
    orchestrator_names = {
        schema["function"]["name"]
        for schema in registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
        if "function" in schema
    }
    assert "kanban_release_triage" in orchestrator_names

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    invalidate_check_fn_cache()
    worker_names = {
        schema["function"]["name"]
        for schema in registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
        if "function" in schema
    }
    assert "kanban_release_triage" not in worker_names


def test_release_tool_is_orchestrator_only_and_actor_is_not_an_argument(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _enable_orchestrator(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "server-coordinator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with connect() as connection:
        task_id = _triage(connection)

    from tools import kanban_tools_triage as tool

    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    result = json.loads(tool._handle_release_triage({
        "task_id": task_id,
        "reason": "operator approved",
        "actor": "spoofed-caller",
    }))
    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["actor"] == "server-coordinator"

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    denied = json.loads(tool._handle_release_triage({
        "task_id": task_id,
        "reason": "worker must not release triage",
    }))
    assert "orchestrator-only" in denied["error"]


def test_release_triage_real_dispatch_denies_profile_without_orchestrator_access(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("toolsets: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "non-orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with connect() as connection:
        task_id = _triage(connection)
        before = list(connection.iterdump())

    from model_tools import handle_function_call
    from tools import kanban_tools_triage  # noqa: F401
    from tools.registry import registry

    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    direct = json.loads(str(registry.dispatch(
        "kanban_release_triage", {"task_id": task_id, "reason": "must deny"}
    )))
    via_model = json.loads(handle_function_call(
        "kanban_release_triage",
        {"task_id": task_id, "reason": "must deny"},
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
        enabled_toolsets=["hermes-cli"],
    ))

    assert "orchestrator" in direct["error"]
    assert "orchestrator" in via_model["error"]
    with connect() as connection:
        assert list(connection.iterdump()) == before


@pytest.mark.parametrize("bad_value", [True, 7, ["triage"], {"id": "triage"}, None, " "])
def test_release_triage_real_dispatch_rejects_non_string_arguments_without_mutation(
    tmp_path, monkeypatch, bad_value
):
    home = tmp_path / ".hermes"
    home.mkdir()
    _enable_orchestrator(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "coordinator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with connect() as connection:
        task_id = _triage(connection)
        before = list(connection.iterdump())

    from tools import kanban_tools_triage  # noqa: F401
    from tools.registry import registry

    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    bad_task = json.loads(str(registry.dispatch(
        "kanban_release_triage", {"task_id": bad_value, "reason": "valid reason"}
    )))
    bad_reason = json.loads(str(registry.dispatch(
        "kanban_release_triage", {"task_id": task_id, "reason": bad_value}
    )))

    assert "string" in bad_task["error"] or "required" in bad_task["error"]
    assert "string" in bad_reason["error"] or "required" in bad_reason["error"]
    with connect() as connection:
        assert list(connection.iterdump()) == before
