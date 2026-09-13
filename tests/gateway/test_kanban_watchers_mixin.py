"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import gateway.kanban_watchers as watchers
from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_dispatcher_retries_contended_lock_and_takes_over(monkeypatch, tmp_path, caplog):
    """A gateway waiting behind the singleton must dispatch after takeover."""
    from hermes_cli import kanban_db
    from hermes_cli import config as config_module

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None
    lock_handle = object()
    lock_attempts = iter(
        [(None, "contended"), (None, "unavailable"), (lock_handle, "held")]
    )
    monkeypatch.setattr(
        watchers,
        "_acquire_singleton_lock",
        lambda _path: next(lock_attempts),
    )
    released = []
    monkeypatch.setattr(
        watchers,
        "_release_singleton_lock",
        lambda handle: released.append(handle),
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(watchers.asyncio, "sleep", lambda _delay: _noop())
    monkeypatch.setattr(kanban_db, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(
        kanban_db,
        "list_boards",
        lambda include_archived=False: [{"slug": "default"}],
    )
    board_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda board=None: board_path)
    monkeypatch.setattr(kanban_db, "connect", lambda board=None: _Connection())
    monkeypatch.setattr(kanban_db, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kanban_db, "has_spawnable_ready", lambda _conn: False)
    monkeypatch.setattr(kanban_db, "has_spawnable_review", lambda _conn: False)
    monkeypatch.setattr(kanban_db, "review_dispatch_enabled", lambda: False)
    dispatch_calls = []

    def dispatch_once(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        runner._running = False
        return SimpleNamespace(
            spawned=[123],
            reclaimed=0,
            crashed=[],
            timed_out=[],
            promoted=0,
            auto_blocked=[],
        )

    monkeypatch.setattr(kanban_db, "dispatch_once", dispatch_once)

    with caplog.at_level("INFO", logger="gateway.run"):
        asyncio.run(watchers.GatewayKanbanWatchersMixin._kanban_dispatcher_watcher(runner))

    assert dispatch_calls, "takeover must enter the existing dispatch loop"
    assert released == [lock_handle]
    assert runner._kanban_dispatcher_lock_handle is None
    messages = [record.getMessage() for record in caplog.records]
    assert sum("waiting to take over" in message for message in messages) == 1
    assert sum("dispatch remains paused" in message for message in messages) == 1
    assert any("acquired singleton dispatcher lock after waiting" in message for message in messages)


async def _noop():
    return None


class _Connection:
    def close(self):
        pass
