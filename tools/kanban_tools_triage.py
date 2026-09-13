"""Kanban triage-release tool registration."""
from __future__ import annotations

from hermes_cli import kanban_db_triage as kbt
from tools.kanban_tools import (
    _board,
    _check,
    _check_kanban_orchestrator_mode,
    _kanban_handler,
    _redact,
    _Reject,
    _reject_delegated_child_mutation,
    _require_orchestrator_tool,
    _ok,
)
from tools.kanban_tools_schemas import KANBAN_RELEASE_TRIAGE_SCHEMA
from tools.registry import registry


def _require_string_arg(args: dict, name: str, message: str) -> str:
    """Validate native tool arguments before redaction or domain coercion."""
    value = args.get(name)
    if not isinstance(value, str):
        raise _Reject(f"{message} (must be a string)")
    if not value.strip():
        raise _Reject(message)
    return value


@_kanban_handler("kanban_release_triage")
def _handle_release_triage(args: dict, **kw) -> str:
    """Release triage through the atomic native domain operation."""
    _reject_delegated_child_mutation("kanban_release_triage")
    _require_orchestrator_tool("kanban_release_triage")
    # Registry dispatch does not run check_fn; enforce the live profile gate here
    # so stale schemas or direct dispatch cannot turn visibility into authority.
    _check(
        _check_kanban_orchestrator_mode(),
        "kanban_release_triage refused: orchestrator Kanban access is not enabled",
    )
    task_id = _require_string_arg(args, "task_id", "task_id is required")
    reason = _redact(_require_string_arg(
        args, "reason", "reason is required — explain why triage is released"))
    with _board(args.get("board")) as (kb, conn):
        result = kbt.release_triage_task(
            conn,
            task_id,
            reason=reason,
            actor=kbt.server_actor(),
        )
    return _ok(**result)


registry.register(
    name="kanban_release_triage",
    toolset="kanban",
    schema=KANBAN_RELEASE_TRIAGE_SCHEMA,
    handler=_handle_release_triage,
    emoji="▶",
    check_fn=_check_kanban_orchestrator_mode,
)
