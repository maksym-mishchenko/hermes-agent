import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    # Provide real exception classes so ``except (NetworkError, ...)`` in
    # connect() doesn't blow up with "catching classes that do not inherit
    # from BaseException" when another xdist worker pollutes sys.modules.
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)
    # Mock HTTPXRequest so the builder chain doesn't fail
    monkeypatch.setattr("plugins.platforms.telegram.adapter.HTTPXRequest", lambda **kwargs: MagicMock())


async def _cancel_heartbeat(adapter):
    """Cancel the lifetime heartbeat task connect() starts in polling mode.

    These tests call the real connect() but never disconnect(), so the
    _polling_heartbeat_loop task would otherwise outlive the test. With
    asyncio.sleep monkeypatched to instant, leaving it running busy-spins the
    event loop and starves the test (CI per-file timeout). disconnect() does
    this in production; tests that only connect() must do it themselves.
    """
    task = getattr(adapter, "_polling_heartbeat_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    adapter._polling_heartbeat_task = None


@pytest.mark.asyncio
async def test_connect_rejects_same_host_token_lock(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (False, {"pid": 4242}),
    )

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram-bot-token_lock"
    assert adapter.has_fatal_error is True
    assert "already in use" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_polling_conflict_retries_before_fatal(monkeypatch):
    """A single 409 should trigger a retry, not an immediate fatal error."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    assert callable(captured["error_callback"])

    conflict = type("Conflict", (Exception,), {})

    # First conflict: should retry, NOT be fatal
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates request"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Give the scheduled task a chance to run
    for _ in range(10):
        await asyncio.sleep(0)

    assert adapter.has_fatal_error is False, "First conflict should not be fatal"
    assert adapter._polling_conflict_count == 0, "Count should reset after successful retry"

    # connect() now starts a lifetime _polling_heartbeat_loop task. With
    # asyncio.sleep mocked to instant above, it must not be left running or it
    # busy-spins on the event loop and starves the test. Cancel it explicitly.
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_polling_conflict_becomes_fatal_after_retries(monkeypatch):
    """After exhausting retries, the conflict should become fatal."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    # Make start_polling fail on retries to exhaust retries
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (initial connect) succeeds
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry calls fail
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    conflict = type("Conflict", (Exception,), {})

    # Directly call _handle_polling_conflict to avoid event-loop scheduling
    # complexity.  Each call simulates one 409 from Telegram.
    for i in range(6):
        await adapter._handle_polling_conflict(
            conflict("Conflict: terminated by other getUpdates request")
        )

    # After 5 failed retries (count 1-5 each enter the retry branch but
    # start_polling raises), the 6th conflict pushes count to 6 which
    # exceeds MAX_CONFLICT_RETRIES (5), entering the fatal branch.
    assert adapter.fatal_error_code == "telegram_polling_conflict", (
        f"Expected fatal after 6 conflicts, got code={adapter.fatal_error_code}, "
        f"count={adapter._polling_conflict_count}"
    )
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_connect_marks_retryable_fatal_error_for_startup_network_failure(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    app = SimpleNamespace(
        bot=SimpleNamespace(delete_webhook=AsyncMock(), set_my_commands=AsyncMock()),
        updater=SimpleNamespace(),
        add_handler=MagicMock(),
        initialize=AsyncMock(side_effect=RuntimeError("Temporary failure in name resolution")),
        start=AsyncMock(),
    )
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_connect_error"
    assert adapter.fatal_error_retryable is True
    assert "Temporary failure in name resolution" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_connect_clears_webhook_before_polling(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        delete_webhook=AsyncMock(),
        set_my_commands=AsyncMock(),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_disconnect_skips_inactive_updater_and_app(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    updater = SimpleNamespace(running=False, stop=AsyncMock())
    app = SimpleNamespace(
        updater=updater,
        running=False,
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    adapter._app = app

    warning = MagicMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.logger.warning", warning)

    await adapter.disconnect()

    updater.stop.assert_not_awaited()
    app.stop.assert_not_awaited()
    app.shutdown.assert_awaited_once()
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_polling_conflict_reschedule_uses_running_loop(monkeypatch):
    """Regression for #19471.

    When a conflict-retry's start_polling raises and we are still below the
    retry ceiling, the handler reschedules itself via loop.create_task. The
    old code used the deprecated asyncio.get_event_loop(), which raises
    "RuntimeError: There is no current event loop in thread 'MainThread'" on
    Python 3.11+ when no loop is attached to the thread (as happens when PTB
    dispatches this error callback). That left the gateway alive but silent
    and drove the --replace crash loop. The fix uses get_running_loop(), which
    is always valid inside a coroutine. Force get_event_loop() to raise so a
    regression would surface as the original RuntimeError, not pass silently.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry attempt fails so the handler enters the reschedule branch.
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    # If the fix regresses to get_event_loop(), this makes it raise — the same
    # RuntimeError users hit in #19471. The running-loop path ignores it.
    def _boom():
        raise RuntimeError("There is no current event loop in thread 'MainThread'.")

    monkeypatch.setattr("asyncio.get_event_loop", _boom)

    conflict = type("Conflict", (Exception,), {})

    # One conflict: count goes to 1 (< MAX), retry's start_polling raises,
    # handler reschedules via loop.create_task — the previously-broken line.
    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )

    assert adapter.has_fatal_error is False
    assert adapter._polling_error_task is not None
    # The rescheduled task must be schedulable on the running loop.
    adapter._polling_error_task.cancel()
    try:
        await adapter._polling_error_task
    except (asyncio.CancelledError, Exception):
        pass
    await _cancel_heartbeat(adapter)


def _build_polling_app(monkeypatch):
    """Wire a mock PTB Application whose start_polling captures kwargs."""
    captured = {}

    async def fake_start_polling(**kwargs):
        captured.update(kwargs)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    return captured


@pytest.mark.asyncio
async def test_cold_connect_drops_pending_updates(monkeypatch):
    """A cold first boot (is_reconnect=False) drops the stale Bot API queue."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = _build_polling_app(monkeypatch)

    ok = await adapter.connect()  # default is_reconnect=False

    assert ok is True
    assert captured["drop_pending_updates"] is True
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_reconnect_preserves_pending_updates(monkeypatch):
    """A watcher reconnect (is_reconnect=True) preserves the queue Telegram
    accumulated during the outage — the core of #46621."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = _build_polling_app(monkeypatch)

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


def _bare_polling_adapter():
    """Build an adapter wired only for direct _handle_polling_conflict calls.

    Bypasses connect()/heartbeat/DoH/lock machinery — we only need _app with a
    succeeding updater and an error-callback ref so the conflict-recovery path
    runs in isolation. The mock bot deliberately lacks ``_request`` so
    ``_drain_polling_connections`` short-circuits to a safe no-op.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    updater = SimpleNamespace(
        start_polling=AsyncMock(),  # every restart "succeeds"
        stop=AsyncMock(),
        running=True,
    )
    adapter._app = SimpleNamespace(bot=SimpleNamespace(), updater=updater)
    adapter._polling_error_callback_ref = MagicMock()
    return adapter, updater


@pytest.mark.asyncio
async def test_polling_conflict_escalates_despite_successful_restarts(monkeypatch):
    """Regression: repeated async 409s must escalate to fatal even when every
    start_polling() restart *returns successfully*.

    The old code reset the conflict counter to 0 the instant start_polling()
    returned. But start_polling() returning only launches the background
    updater task — the 409 surfaces ASYNCHRONOUSLY via the error callback a few
    seconds later. So the counter oscillated 1->0->1 forever and the fatal
    branch was dead code (production logged "1/5" every ~21s indefinitely).

    The fix defers the reset to a confirmed-healthy window and cancels the
    pending reset on each fresh conflict, so genuinely repeated conflicts climb
    the counter and hit MAX_CONFLICT_RETRIES. This test fails on the old code
    (never becomes fatal) and passes on the fix.
    """
    adapter, _updater = _bare_polling_adapter()
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        # Yield to the loop (so cancelled reset tasks settle) but never wait —
        # keeps the test instant and deterministic.
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", fast_sleep)

    conflict = type("Conflict", (Exception,), {})

    # Six consecutive async 409 conflicts, each followed by a *successful*
    # start_polling() restart. With the bug the counter resets every cycle and
    # never reaches fatal; with the fix the next conflict cancels the pending
    # reset, so the counter climbs to 6 > MAX_CONFLICT_RETRIES (5).
    for _ in range(6):
        await adapter._handle_polling_conflict(
            conflict("Conflict: terminated by other getUpdates request")
        )

    assert adapter.fatal_error_code == "telegram_polling_conflict"
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()
    # No deferred reset should have survived to clear the counter.
    assert adapter._polling_conflict_count == 6

    # Let any cancelled reset tasks finalize to avoid pending-task warnings.
    for _ in range(3):
        await real_sleep(0)


@pytest.mark.asyncio
async def test_polling_conflict_counter_resets_only_after_healthy_window(monkeypatch):
    """The counter must clear ONLY after a conflict-free health window, not the
    instant start_polling() returns.

    This locks in the positive half of the fix: a single conflict followed by a
    successful restart leaves the counter at 1 (with a pending reset task) until
    the ~35s health window elapses without a new conflict, at which point it
    resets to 0.
    """
    adapter, _updater = _bare_polling_adapter()
    adapter.set_fatal_error_handler(AsyncMock())

    real_sleep = asyncio.sleep
    health_gate = asyncio.Event()

    async def gated_sleep(delay, *args, **kwargs):
        # 35s == CONFLICT_HEALTHY_RESET_DELAY — block it on the gate so we can
        # assert the "not yet healthy" state. All other sleeps pass instantly.
        if delay == 35:
            await health_gate.wait()
        else:
            await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", gated_sleep)

    conflict = type("Conflict", (Exception,), {})

    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )

    # start_polling() returned, but the health window has NOT elapsed: the
    # counter must still be 1 with a pending (not-done) reset task.
    await real_sleep(0)
    assert adapter._polling_conflict_count == 1
    assert adapter._polling_conflict_reset_task is not None
    assert not adapter._polling_conflict_reset_task.done()
    assert adapter.has_fatal_error is False

    # Open the health window: the reset task wakes and clears the counter.
    health_gate.set()
    await adapter._polling_conflict_reset_task
    assert adapter._polling_conflict_count == 0
