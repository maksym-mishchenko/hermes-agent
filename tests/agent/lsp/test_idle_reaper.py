"""Tests for the LSPService periodic idle reaper.

Verifies that:
- Idle clients are reaped by the background reaper without a subsequent
  diagnostic request.
- Clients used recently are NOT reaped.
- No reaper task leaks after shutdown (``_idle_reaper_task`` is done/cancelled).
- Reaper is a no-op when the service is disabled.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.lsp.manager import LSPService

# Upstream removed REAPER_INTERVAL constant; interval is now min(60, idle_timeout).
REAPER_INTERVAL = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(idle_timeout: float = 1.0) -> LSPService:
    """Create an LSPService with a very short idle_timeout for testing.

    We patch ``_BackgroundLoop.start`` to be a no-op so no real OS thread
    is spawned, and we inject a real asyncio event loop manually.
    """
    def _close_coro(coro, **kwargs):
        """Close the coroutine immediately so no 'never awaited' warning fires."""
        if hasattr(coro, "close"):
            coro.close()

    with (
        patch("agent.lsp.manager._BackgroundLoop.start"),
        patch("agent.lsp.manager._BackgroundLoop.run", side_effect=_close_coro),
        patch("agent.lsp.manager._BackgroundLoop.stop"),
    ):
        svc = LSPService(
            enabled=True,
            wait_mode="document",
            wait_timeout=5.0,
            install_strategy="none",
            idle_timeout=idle_timeout,
        )
    return svc


# ---------------------------------------------------------------------------
# Unit tests for _reap_idle_once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_idle_once_removes_stale_client():
    """_reap_idle_once must remove and shut down a client whose last-used
    timestamp is older than idle_timeout."""
    svc = _make_service(idle_timeout=10.0)

    mock_client = AsyncMock()
    key = ("pyright", "/workspace")
    with svc._state_lock:
        svc._clients[key] = mock_client
        svc._last_used[key] = time.time() - 20.0  # 20 s ago → stale

    await svc._reap_idle_once()

    # Client must be gone from internal maps.
    assert key not in svc._clients
    assert key not in svc._last_used
    # shutdown() must have been awaited.
    mock_client.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_reap_idle_once_preserves_recent_client():
    """_reap_idle_once must NOT remove a client used very recently."""
    svc = _make_service(idle_timeout=60.0)

    mock_client = AsyncMock()
    key = ("gopls", "/workspace")
    with svc._state_lock:
        svc._clients[key] = mock_client
        svc._last_used[key] = time.time() - 5.0  # 5 s ago → fresh

    await svc._reap_idle_once()

    assert key in svc._clients
    mock_client.shutdown.assert_not_awaited()


@pytest.mark.asyncio
async def test_reap_idle_once_handles_missing_last_used():
    """_reap_idle_once must reap a client with no last_used entry (treat as
    epoch 0, i.e. always stale)."""
    svc = _make_service(idle_timeout=1.0)

    mock_client = AsyncMock()
    key = ("tsserver", "/workspace")
    with svc._state_lock:
        svc._clients[key] = mock_client
        # Intentionally omit _last_used[key]

    await svc._reap_idle_once()

    assert key not in svc._clients
    mock_client.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration test: reaper loop actually triggers cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_loop_reaps_idle_client_without_diagnostic():
    """The _reaper_loop task must reap idle clients on its own schedule,
    without any subsequent get_diagnostics_sync call."""
    svc = _make_service(idle_timeout=0.05)  # 50 ms idle timeout
    # Upstream computes interval as min(60, idle_timeout) so 0.05 fires fast.

    mock_client = AsyncMock()
    key = ("pyright", "/workspace")
    with svc._state_lock:
        svc._clients[key] = mock_client
        svc._last_used[key] = time.time() - 1.0  # well past the 50 ms timeout

    # Run the reaper loop inside a real event loop for just a bit.
    loop = asyncio.get_running_loop()
    task = loop.create_task(svc._idle_reaper_loop())
    svc._idle_reaper_task = task
    await asyncio.sleep(0.15)  # give two reaper cycles time to run
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert key not in svc._clients
    mock_client.shutdown.assert_awaited()


# ---------------------------------------------------------------------------
# Shutdown cancels the reaper task cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_async_cancels_idle_reaper_task():
    """_shutdown_async must cancel _idle_reaper_task and await it so no task leak
    survives after shutdown."""
    svc = _make_service()

    # Simulate a running reaper task.
    loop = asyncio.get_running_loop()
    long_sleep = loop.create_task(asyncio.sleep(3600), name="fake-reaper")
    svc._idle_reaper_task = long_sleep

    await svc._shutdown_async()

    assert long_sleep.done()
    # Task was cancelled (not failed with an unrelated error).
    assert long_sleep.cancelled()


def test_disabled_service_has_no_reaper():
    """When enabled=False the reaper task must never be created."""
    with (
        patch("agent.lsp.manager._BackgroundLoop.start"),
        patch("agent.lsp.manager._BackgroundLoop.run"),
        patch("agent.lsp.manager._BackgroundLoop.stop"),
    ):
        svc = LSPService(
            enabled=False,
            wait_mode="document",
            wait_timeout=5.0,
            install_strategy="none",
        )
    assert svc._idle_reaper_task is None
