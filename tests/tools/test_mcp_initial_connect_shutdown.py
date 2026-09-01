"""Regression tests for initial MCP failure ownership and teardown."""

import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest


def _reset_mcp_state(mcp_tool) -> None:
    mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


def _cleanup_mcp_state(mcp_tool, extra_servers=()) -> None:
    with mcp_tool._lock:
        loop = mcp_tool._mcp_loop
    if loop is not None and loop.is_running():
        for server in extra_servers:
            task = getattr(server, "_task", None)
            if task is not None and not task.done():
                mcp_tool._run_on_mcp_loop(server.shutdown, timeout=5)
    mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


def test_initial_connect_failure_is_registry_owned_and_reaped(monkeypatch, tmp_path):
    """Normal discovery must retain the parked task for clean shutdown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _FailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("deterministic initial failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)

    real_stop = mcp_tool._stop_mcp_loop
    pending_at_stop = []

    async def _pending_tasks():
        current = asyncio.current_task()
        return sorted(
            task.get_coro().__qualname__
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        )

    def _observed_stop(*, only_if_idle=False):
        pending_at_stop.extend(
            mcp_tool._run_on_mcp_loop(_pending_tasks, timeout=5)
        )
        return real_stop(only_if_idle=only_if_idle)

    monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", _observed_stop)

    try:
        assert mcp_tool.register_mcp_servers({
            "initial-failure": {"command": "unused", "connect_timeout": 5}
        }) == []

        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["initial-failure"] is server
            assert "deterministic initial failure" in (
                mcp_tool._server_connect_errors["initial-failure"]
            )
        assert server._task is not None
        assert not server._task.done(), "recoverable initial failure was not parked"

        mcp_tool.shutdown_mcp_servers()

        assert pending_at_stop == [], (
            "shutdown left MCP tasks pending at loop stop: "
            f"{pending_at_stop!r}"
        )
        assert server._task.done()
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is None
            assert mcp_tool._mcp_thread is None
    finally:
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", real_stop)
        _cleanup_mcp_state(mcp_tool, created)


def test_initial_connect_failure_revives_same_registered_server(monkeypatch, tmp_path):
    """A cached parked failure must revive through register_mcp_servers()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.registry import ToolRegistry
    import tools.registry as registry_module

    _reset_mcp_state(mcp_tool)
    created = []
    backend_up = threading.Event()
    revived = threading.Event()
    state = {"transport_calls": 0, "tool_calls": 0}
    mock_registry = ToolRegistry()

    class _Session:
        async def call_tool(self, name, arguments):
            state["tool_calls"] += 1
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=f"revived:{arguments['value']}")],
                structuredContent=None,
            )

    class _RecoveringServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            assert mcp_tool._connect_server_claim.get() is None
            state["transport_calls"] += 1
            if not backend_up.is_set():
                raise ConnectionError("backend still booting")

            self.session = _Session()
            self._tools = [SimpleNamespace(
                name="ping",
                description="Return a deterministic revival result",
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )]
            # Match the real transports: discovery runs before _ready is set.
            self._register_discovered_tools_if_needed()
            self._ready.set()
            revived.set()
            return await self._wait_for_lifecycle_event()

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _RecoveringServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    monkeypatch.setattr(registry_module, "registry", mock_registry)

    config = {
        "recovering": {"command": "unused", "connect_timeout": 5}
    }

    try:
        assert mcp_tool.register_mcp_servers(config) == []
        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "backend still booting" in (
                mcp_tool._server_connect_errors["recovering"]
            )
        assert not server._task.done()

        backend_up.set()
        mcp_tool.register_mcp_servers(config)

        assert revived.wait(timeout=5), "cached parked server did not revive"
        assert len(created) == 1, "revival created a duplicate server task"
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "recovering" not in mcp_tool._server_connect_errors
        assert state["transport_calls"] == 2
        assert server.session is not None
        assert server._error is None

        entry = mock_registry.get_entry("mcp__recovering__ping")
        assert entry is not None
        assert entry.check_fn() is True
        assert json.loads(entry.handler({"value": "ok"})) == {
            "result": "revived:ok"
        }
        assert state["tool_calls"] == 1
    finally:
        _cleanup_mcp_state(mcp_tool, created)


def test_initial_auth_failure_is_retained_and_reaped(monkeypatch, tmp_path):
    """An auth failure must stay parked (revivable) and reap on shutdown.

    A 401 used to end the run task outright, which dropped the only listener
    on ``_reconnect_event`` — the server could not come back even after the
    user re-authenticated. It is now retained like any other parked server,
    and must still tear down cleanly.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _AuthFailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise PermissionError("terminal authentication failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _AuthFailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    monkeypatch.setattr(mcp_tool, "_is_auth_error", lambda exc: True)

    try:
        assert mcp_tool.register_mcp_servers({
            "auth-failure": {"command": "unused", "connect_timeout": 5}
        }) == []
        assert len(created) == 1
        server = created[0]
        assert not server._task.done(), (
            "auth failure ended the run task — the server is unrevivable"
        )
        with mcp_tool._lock:
            assert mcp_tool._servers["auth-failure"] is server
            assert "terminal authentication failure" in (
                mcp_tool._server_connect_errors["auth-failure"]
            )

        mcp_tool.shutdown_mcp_servers()
        assert server._task.done()
    finally:
        _cleanup_mcp_state(mcp_tool, created)




def test_standalone_failed_connect_is_reaped_without_global_owner(monkeypatch, tmp_path):
    """Probe-only _connect_server failures must not publish parked servers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _ProbeServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("probe target unavailable")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _ProbeServerTask)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    mcp_tool._ensure_mcp_loop()

    try:
        with pytest.raises(ConnectionError, match="probe target unavailable"):
            mcp_tool._run_on_mcp_loop(
                lambda: mcp_tool._connect_server(
                    "probe-only", {"command": "unused"}
                ),
                timeout=5,
            )

        assert len(created) == 1
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert "probe-only" not in mcp_tool._servers
            assert "probe-only" not in mcp_tool._server_connect_errors
    finally:
        _cleanup_mcp_state(mcp_tool, created)


def test_registered_server_shutdown_is_idempotent(monkeypatch, tmp_path):
    """A normally registered task is drained, and repeated shutdown is safe."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    created = []

    class _HealthyServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            self.session = object()
            self._tools = []
            self._ready.set()
            return await self._wait_for_lifecycle_event()

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _HealthyServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    try:
        assert mcp_tool.register_mcp_servers({"healthy": {"command": "unused"}}) == []
        assert len(created) == 1
        mcp_tool.shutdown_mcp_servers()
        mcp_tool.shutdown_mcp_servers()
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert not mcp_tool._servers
            assert mcp_tool._mcp_loop is None
    finally:
        _cleanup_mcp_state(mcp_tool, created)


def test_shutdown_retains_loop_when_task_resists_cancellation(monkeypatch):
    """A cancellation-resistant task must outlive the first bounded attempt."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    released = threading.Event()
    started = threading.Event()

    async def resistant():
        started.set()
        while not released.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    mcp_tool._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop
    asyncio.run_coroutine_threadsafe(resistant(), loop)
    assert started.wait(1)
    try:
        begin = time.monotonic()
        mcp_tool.shutdown_mcp_servers()
        assert time.monotonic() - begin <= 3.2
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is loop
            assert mcp_tool._mcp_thread is not None
            assert mcp_tool._mcp_loop_shutting_down
        assert not loop.is_closed()
    finally:
        released.set()
        with mcp_tool._lock:
            future = mcp_tool._mcp_shutdown_future
        if future is not None and not future.done():
            future.result(timeout=5)
        # Once the coordinator has become terminal and the task quiesces,
        # this later caller is permitted to start the one retry.
        assert mcp_tool.shutdown_mcp_servers() is True
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is None


def test_blocked_callback_keeps_global_loop_owner_until_later_retry():
    """A callback blocking the loop cannot cause an untracked replacement."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    release = threading.Event()
    entered = threading.Event()

    def blocked_callback():
        entered.set()
        release.wait(12)

    mcp_tool._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop
    loop.call_soon_threadsafe(blocked_callback)
    assert entered.wait(1)
    try:
        begin = time.monotonic()
        mcp_tool.shutdown_mcp_servers()
        assert time.monotonic() - begin <= 3.2
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is loop
            assert mcp_tool._mcp_thread.is_alive()
        with pytest.raises(mcp_tool.MCPShutdownInProgressError):
            mcp_tool._ensure_mcp_loop()
        assert mcp_tool._mcp_loop is loop
    finally:
        release.set()
        time.sleep(0.05)
        mcp_tool.shutdown_mcp_servers()
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is None


def test_concurrent_shutdown_callers_share_one_coordinator(monkeypatch):
    """Ten callers wait on one coordinator and never create peer drain tasks."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    entered = threading.Event()
    release = threading.Event()
    coordinator_calls = 0
    coordinator_lock = threading.Lock()
    original = mcp_tool._mcp_shutdown_coordinator

    async def counted(*args, **kwargs):
        nonlocal coordinator_calls
        with coordinator_lock:
            coordinator_calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(mcp_tool, "_mcp_shutdown_coordinator", counted)
    mcp_tool._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop

    def blocked():
        entered.set()
        release.wait(1)

    loop.call_soon_threadsafe(blocked)
    assert entered.wait(1)
    barrier = threading.Barrier(10)
    results = []

    def caller():
        barrier.wait()
        results.append(mcp_tool.shutdown_mcp_servers())

    threads = [threading.Thread(target=caller) for _ in range(10)]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(3)

    assert len(results) == 10
    assert coordinator_calls == 1
    assert all(results)
    with mcp_tool._lock:
        assert mcp_tool._mcp_loop is None
        assert not mcp_tool._server_connecting


def test_registration_is_rejected_without_connecting_leak_during_teardown(monkeypatch):
    """A blocked teardown atomically denies new registration before mutation."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    entered = threading.Event()
    release = threading.Event()
    mcp_tool._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop

    def blocked():
        entered.set()
        release.wait(1)

    loop.call_soon_threadsafe(blocked)
    assert entered.wait(1)
    shutdown_thread = threading.Thread(target=mcp_tool.shutdown_mcp_servers)
    shutdown_thread.start()
    time.sleep(0.05)
    try:
        with pytest.raises(mcp_tool.MCPShutdownInProgressError):
            mcp_tool.register_mcp_servers({"replacement": {"command": "unused"}})
        with mcp_tool._lock:
            assert "replacement" not in mcp_tool._server_connecting
    finally:
        release.set()
        shutdown_thread.join(3)
        mcp_tool.shutdown_mcp_servers()


def test_registration_mutation_and_loop_admission_are_one_lock_transaction(monkeypatch):
    """Teardown cannot claim between connecting mutation and loop selection."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    entered = threading.Event()
    release = threading.Event()
    original_ensure = mcp_tool._ensure_mcp_loop

    def gated_ensure():
        entered.set()
        assert release.wait(2)
        original_ensure()

    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", gated_ensure)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", lambda *args, **kwargs: None)
    registration_error = []

    def run_registration():
        try:
            mcp_tool.register_mcp_servers({"atomic": {"command": "unused"}})
        except BaseException as exc:
            registration_error.append(exc)

    registration = threading.Thread(target=run_registration)
    registration.start()
    assert entered.wait(1)
    shutdown = threading.Thread(target=mcp_tool.shutdown_mcp_servers)
    shutdown.start()
    time.sleep(0.05)
    assert shutdown.is_alive(), "shutdown claimed while registration held admission"
    release.set()
    registration.join(2)
    shutdown.join(4)
    assert not registration.is_alive()
    assert not shutdown.is_alive()
    with mcp_tool._lock:
        assert "atomic" not in mcp_tool._server_connecting
    _cleanup_mcp_state(mcp_tool)


def test_registration_rejects_teardown_after_admission_without_scheduling(monkeypatch):
    """Teardown between admission and scheduling rejects without a leak."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    admitted = threading.Event()
    teardown_claimed = threading.Event()
    payload_scheduled = threading.Event()
    original_run = mcp_tool._run_on_mcp_loop
    original_coordinator = mcp_tool._mcp_shutdown_coordinator

    async def claims_then_shutdown(*args, **kwargs):
        teardown_claimed.set()
        return await original_coordinator(*args, **kwargs)

    def gated_run(coro_or_factory, timeout=30, **kwargs):
        if kwargs.get("admission_loop") is not None:
            admitted.set()
            assert teardown_claimed.wait(2)
        return original_run(coro_or_factory, timeout=timeout, **kwargs)

    monkeypatch.setattr(mcp_tool, "_mcp_shutdown_coordinator", claims_then_shutdown)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", gated_run)

    async def payload():
        payload_scheduled.set()

    async def discover_one(name, cfg):
        await payload()

    monkeypatch.setattr(mcp_tool, "_discover_and_register_server", discover_one)
    outcome = []

    def register():
        try:
            mcp_tool.register_mcp_servers({"race": {"command": "unused"}})
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=register)
    worker.start()
    assert admitted.wait(1)
    shutdown = threading.Thread(target=mcp_tool.shutdown_mcp_servers)
    shutdown.start()
    worker.join(3)
    shutdown.join(4)

    assert not worker.is_alive()
    assert not shutdown.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], mcp_tool.MCPShutdownInProgressError)
    assert not payload_scheduled.is_set()
    with mcp_tool._lock:
        assert "race" not in mcp_tool._server_connecting
    _cleanup_mcp_state(mcp_tool)


def test_schema_fingerprint_failure_rejects_before_reservation(monkeypatch):
    """Cache fingerprinting is preparation, never an admission mutation."""
    from tools import mcp_tool
    import tools.mcp_schema_cache as schema_cache

    _reset_mcp_state(mcp_tool)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)

    def fail_fingerprint(_config):
        raise ValueError("fingerprint barrier")

    monkeypatch.setattr(schema_cache, "config_fingerprint", fail_fingerprint)
    with pytest.raises(ValueError, match="fingerprint barrier"):
        mcp_tool.register_mcp_servers({
            "lazy": {"command": "unused", "lazy": True},
            "eager": {"command": "unused"},
        })

    with mcp_tool._lock:
        assert not mcp_tool._server_connecting
        assert "lazy" not in mcp_tool._lazy_server_configs
        assert "lazy" not in mcp_tool._lazy_server_fingerprints
        assert mcp_tool._mcp_loop is None


def test_scheduling_rejection_rolls_back_lazy_and_eager_admission(monkeypatch):
    """A rejected schedule leaves neither cache registration nor reservation."""
    from tools import mcp_tool
    import tools.mcp_schema_cache as schema_cache

    _reset_mcp_state(mcp_tool)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(schema_cache, "config_fingerprint", lambda _config: "fp")
    monkeypatch.setattr(schema_cache, "get_cached_entry", lambda _name, _fp: {
        "fingerprint": "fp", "tools": [], "utility_tools": [],
    })

    def register_then_fail(name, config, entry, *, fingerprint=None):
        mcp_tool._lazy_server_configs[name] = dict(config)
        if fingerprint is not None:
            mcp_tool._lazy_server_fingerprints[name] = fingerprint
        mcp_tool._lazy_server_tool_names[name] = []
        return []

    monkeypatch.setattr(mcp_tool, "_register_from_cache_sync", register_then_fail)
    monkeypatch.setattr(
        mcp_tool, "_run_on_mcp_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rejected")),
    )
    with pytest.raises(RuntimeError, match="rejected"):
        mcp_tool.register_mcp_servers({
            "lazy": {"command": "unused", "lazy": True},
            "eager": {"command": "unused"},
        })

    with mcp_tool._lock:
        assert not mcp_tool._server_connecting
        assert "lazy" not in mcp_tool._lazy_server_configs
        assert "lazy" not in mcp_tool._lazy_server_fingerprints
        assert "lazy" not in mcp_tool._lazy_server_tool_names
    _cleanup_mcp_state(mcp_tool)


def test_probe_check_and_schedule_share_admission_lock(monkeypatch):
    """A probe schedule cannot race teardown after its shutdown check."""
    from tools import mcp_tool
    from agent import async_utils

    _reset_mcp_state(mcp_tool)
    mcp_tool._ensure_mcp_loop()
    entered = threading.Event()
    release = threading.Event()
    original_schedule = async_utils.safe_schedule_threadsafe

    def gated_schedule(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_schedule(*args, **kwargs)

    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", gated_schedule)
    outcome = []

    async def noop():
        return None

    worker = threading.Thread(
        target=lambda: outcome.append(mcp_tool._run_on_mcp_loop(noop, timeout=2))
    )
    worker.start()
    assert entered.wait(1)
    shutdown = threading.Thread(target=mcp_tool.shutdown_mcp_servers)
    shutdown.start()
    time.sleep(0.05)
    assert shutdown.is_alive(), "teardown passed the schedule admission gate"
    release.set()
    worker.join(3)
    shutdown.join(4)
    assert outcome == [None]
    assert not worker.is_alive()
    assert not shutdown.is_alive()
    with mcp_tool._lock:
        assert mcp_tool._mcp_loop is None


def test_factory_runs_outside_admission_lock_and_payload_is_rejected(monkeypatch):
    """A blocking factory cannot delay teardown or run after shutdown claims."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    factory_entered = threading.Event()
    release_factory = threading.Event()
    payload_ran = threading.Event()
    outcome = []

    async def payload():
        payload_ran.set()

    def blocking_factory():
        factory_entered.set()
        assert release_factory.wait(10)
        return payload()

    def run_factory():
        try:
            mcp_tool._run_on_mcp_loop(blocking_factory, timeout=5)
        except BaseException as exc:
            outcome.append(exc)

    mcp_tool._ensure_mcp_loop()
    worker = threading.Thread(target=run_factory)
    worker.start()
    assert factory_entered.wait(1)

    started = time.monotonic()
    mcp_tool.shutdown_mcp_servers()
    assert time.monotonic() - started <= 3.2
    with mcp_tool._lock:
        assert mcp_tool._mcp_loop is None

    release_factory.set()
    worker.join(3)
    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], (mcp_tool.MCPShutdownInProgressError, RuntimeError))
    assert not payload_ran.is_set()


def test_shutdown_wave_of_fifty_resistant_callers_retries_once(monkeypatch):
    """One failed wave is shared by all callers; only quiescence permits retry."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    started = threading.Event()
    released = threading.Event()
    coordinator_calls = 0
    coordinator_lock = threading.Lock()
    original = mcp_tool._mcp_shutdown_coordinator

    async def counted(*args, **kwargs):
        nonlocal coordinator_calls
        with coordinator_lock:
            coordinator_calls += 1
        return await original(*args, **kwargs)

    async def resistant():
        started.set()
        while not released.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    monkeypatch.setattr(mcp_tool, "_mcp_shutdown_coordinator", counted)
    mcp_tool._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop
    asyncio.run_coroutine_threadsafe(resistant(), loop)
    assert started.wait(1)

    results = []
    barrier = threading.Barrier(50)

    def caller():
        barrier.wait()
        results.append(mcp_tool.shutdown_mcp_servers())

    callers = [threading.Thread(target=caller) for _ in range(50)]
    for caller_thread in callers:
        caller_thread.start()
    for caller_thread in callers:
        caller_thread.join(4)
    assert len(results) == 50
    assert results and not any(results)
    assert coordinator_calls == 1
    with mcp_tool._lock:
        assert mcp_tool._mcp_loop is loop
        assert not mcp_tool._mcp_shutdown_wave_active

    released.set()
    assert mcp_tool.shutdown_mcp_servers() is True
    assert coordinator_calls == 2
    with mcp_tool._lock:
        assert mcp_tool._mcp_loop is None


def test_pending_shutdown_wave_reuses_coordinator_before_retry(monkeypatch):
    """A timed-out waiter cannot release a still-running shutdown wave."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    started = threading.Event()
    released = threading.Event()
    coordinator_calls = []

    async def coordinator(*args, **kwargs):
        coordinator_calls.append(True)
        started.set()
        while not released.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue
        return len(coordinator_calls) > 1

    monkeypatch.setattr(mcp_tool, "_mcp_shutdown_coordinator", coordinator)
    monkeypatch.setattr(mcp_tool, "_MCP_LOOP_DRAIN_TIMEOUT", 0.1)
    mcp_tool._ensure_mcp_loop()
    try:
        assert mcp_tool.shutdown_mcp_servers() is False
        assert started.wait(1)
        assert mcp_tool.shutdown_mcp_servers() is False
        assert len(coordinator_calls) == 1
        with mcp_tool._lock:
            future = mcp_tool._mcp_shutdown_future
            assert mcp_tool._mcp_shutdown_wave_active
        released.set()
        assert future is not None
        assert future.result(timeout=1) is False
        assert mcp_tool.shutdown_mcp_servers() is True
        assert len(coordinator_calls) == 2
    finally:
        released.set()
        mcp_tool.shutdown_mcp_servers()


def test_stopped_loop_is_replaced_only_when_quiescent(monkeypatch):
    """A dead, empty loop is cleared; a loop with pending work is retained."""
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    stale = asyncio.new_event_loop()
    with mcp_tool._lock:
        mcp_tool._mcp_loop = stale
        mcp_tool._mcp_thread = None
    mcp_tool._ensure_mcp_loop()
    try:
        with mcp_tool._lock:
            assert mcp_tool._mcp_loop is not stale
            assert mcp_tool._mcp_loop is not None
            assert mcp_tool._mcp_loop.is_running()
    finally:
        mcp_tool.shutdown_mcp_servers()

    pending_loop = asyncio.new_event_loop()
    pending_loop.create_task(asyncio.sleep(60))
    with mcp_tool._lock:
        mcp_tool._mcp_loop = pending_loop
        mcp_tool._mcp_thread = None
    with pytest.raises(RuntimeError, match="stopped but still owned"):
        mcp_tool._ensure_mcp_loop()
    with mcp_tool._lock:
        pending = list(asyncio.all_tasks(pending_loop))
    for task in pending:
        task.cancel()
    pending_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    pending_loop.close()
    with mcp_tool._lock:
        mcp_tool._mcp_loop = None
        mcp_tool._mcp_thread = None


def test_mcp_teardown_closes_self_pipe_before_registry_logging(caplog):
    """An MCP callback followed by registry logging leaves the exact FD set."""
    from tools import mcp_tool
    from tools.registry import ToolRegistry

    def fd_snapshot():
        snapshot = {}
        for fd in os.listdir("/proc/self/fd"):
            try:
                snapshot[int(fd)] = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
        return snapshot

    _reset_mcp_state(mcp_tool)
    baseline = fd_snapshot()
    mcp_tool._ensure_mcp_loop()
    server = mcp_tool.MCPServerTask("logging-regression")
    callback = server._make_logging_callback()
    try:
        mcp_tool._run_on_mcp_loop(
            lambda: callback(SimpleNamespace(level="warning", data="diagnostic")),
            timeout=5,
        )
        registry = ToolRegistry()
        with caplog.at_level("ERROR", logger="tools.registry"):
            registry.register(
                name="mcp__logging_regression__tool",
                toolset="mcp-one",
                schema={},
                handler=lambda *_args, **_kwargs: "one",
            )
            registry.register(
                name="mcp__logging_regression__tool",
                toolset="mcp-two",
                schema={},
                handler=lambda *_args, **_kwargs: "two",
            )
        assert any("REJECTED" in record.message for record in caplog.records)
    finally:
        assert mcp_tool.shutdown_mcp_servers() is True

    assert fd_snapshot() == baseline
