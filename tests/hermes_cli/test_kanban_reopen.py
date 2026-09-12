"""Tests for guarded native reopen of completed Kanban graph nodes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    connection = kb.connect(tmp_path / "kanban.db")
    try:
        yield connection
    finally:
        connection.close()


def _done_task(conn, title: str, assignee: str = "worker") -> str:
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.complete_task(conn, task_id, summary=f"completed {title}")
    return task_id


def test_reopen_done_task_preserves_owner_history_and_audits_reason(conn):
    task_id = _done_task(conn, "release", "programmer2")
    before_events = len(kb.list_events(conn, task_id))
    before_runs = len(kb.list_runs(conn, task_id))

    result = kb.reopen_task(conn, task_id, reason="recovery scope changed", author="operator")

    assert result["status"] == "ready"
    assert result["prior_status"] == "done"
    assert result["invalidated"] == []
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "programmer2"
    assert task.completed_at is None
    assert len(kb.list_events(conn, task_id)) > before_events
    assert len(kb.list_runs(conn, task_id)) == before_runs
    reopened = [e for e in kb.list_events(conn, task_id) if e.kind == "reopened"]
    assert len(reopened) == 1
    assert reopened[0].payload["reason"] == "recovery scope changed"
    assert any(
        c.author == "operator" and "recovery scope changed" in c.body
        for c in kb.list_comments(conn, task_id)
    )


def test_reopen_parent_invalidates_completed_descendants_and_respects_parent_gate(conn):
    parent = _done_task(conn, "parent")
    child = _done_task(conn, "child")
    with kb.write_txn(conn):
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (parent, child))
    grandchild = _done_task(conn, "grandchild")
    with kb.write_txn(conn):
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (child, grandchild))

    result = kb.reopen_task(conn, parent, reason="re-run verification", author="operator")

    assert result["status"] == "ready"
    assert {row["id"] for row in result["invalidated"]} == {child, grandchild}
    assert kb.get_task(conn, child).status == "todo"
    assert kb.get_task(conn, grandchild).status == "todo"
    assert all(
        e.payload["reason"] == "ancestor_reopened"
        for e in kb.list_events(conn, child)
        if e.kind == "status"
    )

    gated_parent = kb.create_task(conn, title="open parent", assignee="worker")
    gated = _done_task(conn, "already completed child")
    with kb.write_txn(conn):
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (gated_parent, gated))
    reopened = kb.reopen_task(conn, gated, reason="parent still open", author="operator")
    assert reopened["status"] == "todo"
    assert kb.get_task(conn, gated).status == "todo"


def test_reopen_rejects_live_target_or_descendant_without_mutation(conn):
    parent = _done_task(conn, "parent")
    child = kb.create_task(conn, title="live child", assignee="worker", parents=[parent])
    claimed = kb.claim_task(conn, child)
    assert claimed is not None
    parent_before = kb.get_task(conn, parent)
    child_before = kb.get_task(conn, child)
    event_count = len(kb.list_events(conn, parent))

    with pytest.raises(ValueError, match="live run"):
        kb.reopen_task(conn, parent, reason="unsafe", author="operator")

    assert kb.get_task(conn, parent) == parent_before
    assert kb.get_task(conn, child) == child_before
    assert len(kb.list_events(conn, parent)) == event_count


def test_reopen_rejects_unsupported_status_and_reason(conn):
    task_id = kb.create_task(conn, title="ready", assignee="worker")
    with pytest.raises(ValueError, match="only done"):
        kb.reopen_task(conn, task_id, reason="not done", author="operator")
    with pytest.raises(ValueError, match="reason"):
        kb.reopen_task(conn, task_id, reason=" ", author="operator")


def test_reopen_rejects_archived_without_mutation(conn):
    task_id = _done_task(conn, "archived", "worker")
    assert kb.archive_task(conn, task_id)
    before = list(conn.iterdump())

    with pytest.raises(ValueError, match="only done"):
        kb.reopen_task(conn, task_id, reason="archived work must stay terminal")

    assert list(conn.iterdump()) == before


def test_reopen_tool_is_orchestrator_only_and_returns_audit_result(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as connection:
        task_id = _done_task(connection, "tool target", "programmer2")

    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    result = json.loads(kt._handle_reopen({"task_id": task_id, "reason": "operator recovery"}))
    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["reason"] == "operator recovery"

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    denied = json.loads(kt._handle_reopen({"task_id": task_id, "reason": "worker must not route"}))
    assert "orchestrator-only" in denied["error"]


def test_reopen_tool_keeps_explicit_boards_isolated(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()

    with kb.connect(board="alpha") as alpha_conn:
        alpha_id = _done_task(alpha_conn, "alpha target")
    with kb.connect(board="beta") as beta_conn:
        beta_id = _done_task(beta_conn, "beta target")

    from tools import kanban_tools as kt

    result = json.loads(kt._handle_reopen({
        "task_id": alpha_id,
        "reason": "alpha-only recovery",
        "board": "alpha",
    }))
    assert result["ok"] is True
    with kb.connect(board="alpha") as alpha_conn:
        alpha = kb.get_task(alpha_conn, alpha_id)
        assert alpha is not None and alpha.status == "ready"
    with kb.connect(board="beta") as beta_conn:
        beta = kb.get_task(beta_conn, beta_id)
        assert beta is not None and beta.status == "done"


def test_reopen_tool_rejects_delegated_child(monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: True)
    denied = json.loads(kt._handle_reopen({
        "task_id": "t_foreign",
        "reason": "delegated child must not mutate",
    }))
    assert "delegate_task child agents" in denied["error"]
