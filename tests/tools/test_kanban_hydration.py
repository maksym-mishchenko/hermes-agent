import json
from pathlib import Path

import pytest


@pytest.fixture
def review_board(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "board.db"))
    monkeypatch.setattr(Path, "home", lambda: home)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    clock = [2_000_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    body = "Acceptance: implement addition. " + "required context " * 165
    paths = []
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Review the current candidate", body=body, assignee="programmer")
        for index in range(3):
            parent = kb.create_task(conn, title=f"Prerequisite {index}", assignee="planner")
            assert kb.complete_task(conn, parent, result=f"Parent evidence {index}: " + "p" * 2900)
            kb.link_tasks(conn, parent, tid)
        for index in range(30):
            path = tmp_path / (f"evidence-{index:02}-" + "x" * 95 + ".py")
            path.write_text("def add(a, b):\n    return a + b\n")
            paths.append(path)
            kb.add_attachment(conn, tid, filename=path.name, stored_path=str(path), size=path.stat().st_size)
        for index in range(68):
            kb.add_comment(conn, tid, "worker", f"Historical note {index}: " + "h" * 1460)
        for index in range(20):
            clock[0] += 10
            implementation = kb.claim_task(conn, tid)
            assert kb.request_review(conn, tid, reviewer="reviewer",
                                     summary=f"Historical candidate {index}: " + "i" * 1280,
                                     expected_run_id=implementation.current_run_id)
            clock[0] += 10
            reviewer = kb.claim_review_task(conn, tid)
            if index == 19:
                kb.add_comment(conn, tid, "reviewer", "Required correction: preserve negative operands.")
            ok, _ = kb.request_changes(conn, tid, reason=f"Historical finding {index}: " + "r" * 1280,
                                       expected_run_id=reviewer.current_run_id)
            assert ok
        # The latest formal finding stays authoritative through implementation
        # and later crashed/reclaimed reviewer attempts.
        finding_id = reviewer.current_run_id
        clock[0] += 10
        implementation = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, reviewer="reviewer", summary="Candidate ready for independent verification",
            metadata={"candidate": str(paths[0]), "acceptance_evidence": "e" * 1800},
            expected_run_id=implementation.current_run_id,
        )
        clock[0] += 10
        kb.add_comment(conn, tid, "worker", "Current direction: verify negative inputs.")
        reviewer = kb.claim_review_task(conn, tid)
        run_id = reviewer.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return kb, kt, tid, run_id, implementation.current_run_id, finding_id, body, paths


def test_worker_hydration_preserves_current_evidence_without_duplicate_history(
    review_board, monkeypatch, record_property,
):
    kb, kt, tid, run_id, handoff_id, finding_id, body, paths = review_board
    compact = kt._handle_show({})
    data = json.loads(compact)
    assert data["task"]["body"] == body
    assert compact.count(body) == 1
    assert data["worker_run"] == {
        "id": run_id, "is_current": True, "outcome": None, "ended_at": None,
    }
    assert {row["id"] for row in data["runs"]} == {run_id, handoff_id, finding_id}
    assert next(row for row in data["runs"] if row["id"] == handoff_id)["metadata"] == {
        "candidate": str(paths[0]), "acceptance_evidence": "e" * 1800,
    }
    assert "Historical finding 19:" in compact
    assert "Current direction: verify negative inputs." in compact
    assert "Required correction: preserve negative operands." in compact
    assert "Historical note 0:" not in compact
    for path in paths:
        assert str(path) in data["worker_context"]
    for index in range(3):
        assert f"Parent evidence {index}:" in data["worker_context"]
    assert data["history"]["omitted"]["comments"] > 0
    assert data["history"]["omitted"]["runs"] > 0
    assert "before_id" in data["history"]["retrieval"]

    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    full = kt._handle_show({})
    assert "Historical note 0:" in full
    assert len(full.encode()) > 200_000
    assert len(compact.encode()) < 30_000
    assert len(compact.encode()) < len(full.encode()) / 5
    record_property("full_bytes", len(full.encode()))
    record_property("compact_bytes", len(compact.encode()))
    print(f"HYDRATION_BYTES full={len(full.encode())} compact={len(compact.encode())}")
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).current_run_id == run_id


@pytest.mark.parametrize("history", ["comments", "runs", "events"])
def test_history_pages_are_bounded_complete_and_not_duplicated(review_board, history):
    kb, kt, tid, *_ = review_board
    with kb.connect() as conn:
        expected = {row.id for row in getattr(kb, "list_" + history)(conn, tid)}
    seen = []
    before = None
    for _ in range(len(expected) + 1):
        args = {"history": history, "limit": 7}
        if before is not None:
            args["before_id"] = before
        data = json.loads(kt._handle_show(args))
        assert "worker_context" not in data and "task" not in data
        assert len(data["items"]) <= 7
        ids = [row["id"] for row in data["items"]]
        assert ids == sorted(ids, reverse=True)
        seen.extend(ids)
        if not data["has_more"]:
            assert data["next_before_id"] is None
            break
        before = data["next_before_id"]
        assert before == ids[-1]
    else:
        pytest.fail("History traversal did not terminate")
    assert set(seen) == expected and len(seen) == len(expected)


def test_outgoing_reviewer_is_not_presented_as_successor(review_board):
    kb, kt, tid, run_id, *_ = review_board
    with kb.connect() as conn:
        assert kb.request_changes(conn, tid, reason="Fix negative inputs", expected_run_id=run_id)[0]
        successor = kb.claim_task(conn, tid)
        before = list(conn.iterdump())
    data = json.loads(kt._handle_show({}))
    assert data["worker_run"]["id"] == run_id
    assert data["worker_run"]["is_current"] is False
    assert data["worker_run"]["outcome"] == "changes_requested"
    assert data["task"]["current_run_id"] == successor.current_run_id
    with kb.connect() as conn:
        assert list(conn.iterdump()) == before


def test_review_receipts_survive_crash_and_reclaim(review_board, monkeypatch):
    kb, kt, tid, old_run_id, handoff_id, finding_id, *_ = review_board
    with kb.connect() as conn:
        kb._record_task_failure(conn, tid, "Synthetic worker exit", outcome="crashed",
                                release_claim=True, end_run=True)
        assert kb.get_task(conn, tid).status == "review"
        successor = kb.claim_review_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(successor.current_run_id))
    data = json.loads(kt._handle_show({}))
    ids = {row["id"] for row in data["runs"]}
    assert ids == {handoff_id, finding_id, successor.current_run_id}
    assert old_run_id not in ids
    assert "Required correction: preserve negative operands." in json.dumps(data)


@pytest.mark.parametrize("context", ["delegated", "cron", "child-process"])
def test_non_dispatcher_context_retains_full_view(review_board, monkeypatch, context):
    from contextlib import nullcontext

    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context

    _, kt, tid, *_ = review_board
    if context == "child-process":
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    scope = (delegated_child_context() if context == "delegated" else
             non_dispatcher_owned_context() if context == "cron" else nullcontext())
    with scope:
        data = json.loads(kt._handle_show({"task_id": tid}))
    assert "worker_run" not in data
    assert "Historical note 0:" in json.dumps(data)


@pytest.mark.parametrize("args", [
    {"history": "all"}, {"history": "runs", "limit": True},
    {"history": "comments", "limit": 0}, {"history": "events", "limit": 51},
    {"history": "runs", "before_id": -1}, {"limit": 2},
])
def test_invalid_history_requests_have_explicit_errors(review_board, args):
    _, kt, *_ = review_board
    assert "error" in json.loads(kt._handle_show(args))


def test_native_compression_rehydrates_current_evidence_without_history_spill(
    review_board, tmp_path, monkeypatch,
):
    import importlib.util
    from unittest.mock import MagicMock

    from hermes_state import SessionDB
    from tools.budget_config import budget_for_context_window
    from tools.file_tools import read_file_tool
    from tools.tool_result_storage import maybe_persist_tool_result

    _, kt, _, _, _, _, _, paths = review_board
    spec = importlib.util.spec_from_file_location(
        "hydration_runtime_fixtures",
        Path(__file__).parents[1] / "run_agent/test_tool_call_guardrail_runtime.py",
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    agent = fixtures._make_agent("read_file", max_iterations=4)
    db = SessionDB(db_path=tmp_path / "sessions.db")
    agent.session_db = db
    db.create_session(agent.session_id, source="cli")
    agent.compression_in_place = False
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "Compressed review: recover the current candidate and verify it."},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    task_id = "hydration-canary"
    assert "return a + b" in read_file_tool(str(paths[0]), task_id=task_id)
    with monkeypatch.context() as unscoped:
        unscoped.delenv("HERMES_KANBAN_RUN_ID")
        previous_history = kt._handle_show({})
    assert len(previous_history.encode()) > 200_000
    budget = budget_for_context_window(120_000)
    assert "<persisted-output>" in maybe_persist_tool_result(
        previous_history, "kanban_show", "legacy-hydration", config=budget,
    )
    try:
        agent._compress_context(
            [
                {"role": "user", "content": "Review the existing candidate."},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "previous-show", "type": "function",
                    "function": {"name": "kanban_show", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_name": "kanban_show",
                 "tool_call_id": "previous-show", "content": previous_history},
            ],
            "Preserve review requirements.", approx_tokens=60_000, task_id=task_id,
        )
        assert compressor.compress.call_count == 1
        hydrated = kt._handle_show({})
        assert len(hydrated.encode()) < 30_000
        assert maybe_persist_tool_result(
            hydrated, "kanban_show", "current-hydration", config=budget,
        ) == hydrated
        data = json.loads(hydrated)
        candidate = next(
            row["metadata"]["candidate"] for row in data["runs"]
            if row["outcome"] == "review_requested"
        )
        # The actual compression boundary reset the native file-dedup cache.
        # Recovery can read the candidate directly, not parse a huge history spill.
        assert "return a + b" in read_file_tool(candidate, task_id=task_id)
        assert "Historical note 0:" not in hydrated
    finally:
        db.close()


def test_bounded_native_implementation_review_correction_and_completion(
    review_board, monkeypatch, tmp_path,
):
    import runpy

    kb, kt, *_ = review_board
    candidate = tmp_path / "addition.py"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Implement addition", body="Add positive and negative integers.",
                             assignee="programmer")
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        for attempt, expression in enumerate(("a - b", "a + b")):
            implementation = kb.claim_task(conn, tid)
            candidate.write_text(f"def add(a, b):\n    return {expression}\n")
            assert kb.request_review(conn, tid, reviewer="reviewer", summary="Review the candidate",
                                     metadata={"candidate": str(candidate)},
                                     expected_run_id=implementation.current_run_id)
            reviewer = kb.claim_review_task(conn, tid)
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
            view = json.loads(kt._handle_show({}))
            path = next(row["metadata"]["candidate"] for row in view["runs"]
                        if row["outcome"] == "review_requested")
            add = runpy.run_path(path)["add"]
            accepted = add(2, 3) == 5 and add(-2, 5) == 3
            if not accepted:
                assert attempt == 0
                assert kb.request_changes(conn, tid, reason="Use addition, not subtraction",
                                          expected_run_id=reviewer.current_run_id)[0]
            else:
                assert attempt == 1
                assert kb.complete_task(conn, tid, summary="Independent arithmetic checks passed",
                                        expected_run_id=reviewer.current_run_id)
        assert kb.get_task(conn, tid).status == "done"
        runs = kb.list_runs(conn, tid)
        assert [run.outcome for run in runs] == [
            "review_requested", "changes_requested", "review_requested", "completed",
        ]
        assert all(run.ended_at is not None for run in runs)
