"""Behavior-contract tests for lazy MCP server startup (#56832).

A server configured with ``lazy: true`` whose config fingerprint matches an
on-disk schema-cache entry registers its tools WITHOUT spawning/connecting;
the first real call (raw tool OR resource/prompt utility) routes through the
existing connect path.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.mcp_tool as mcp


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    old_servers = dict(mcp._servers)
    old_lazy = dict(mcp._lazy_server_configs)
    old_fps = dict(mcp._lazy_server_fingerprints)
    old_names = dict(mcp._lazy_server_tool_names)
    old_connecting = set(mcp._server_connecting)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._lazy_server_configs.clear()
    mcp._lazy_server_configs.update(old_lazy)
    mcp._lazy_server_fingerprints.clear()
    mcp._lazy_server_fingerprints.update(old_fps)
    mcp._lazy_server_tool_names.clear()
    mcp._lazy_server_tool_names.update(old_names)
    mcp._server_connecting.clear()
    mcp._server_connecting.update(old_connecting)


def _fake_cache_entry():
    return {
        "fingerprint": "abc",
        "tools": [
            {
                "name": "browser_navigate",
                "description": "Navigate",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        "utility_tools": [],
    }


def _lazy_config():
    return {
        "playwright": {
            "command": "npx",
            "args": ["-y", "@playwright/mcp"],
            "lazy": True,
        }
    }


class TestLazyMcpRegistration:
    def test_registers_from_cache_without_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=_fake_cache_entry()), \
             patch(
                 "tools.mcp_tool._register_from_cache_sync",
                 return_value=["mcp_playwright_browser_navigate"],
             ) as mock_register, \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock) as mock_discover, \
             patch("tools.mcp_tool._ensure_mcp_loop") as mock_loop, \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            mcp.register_mcp_servers(config)

        mock_register.assert_called_once()
        mock_discover.assert_not_called()
        mock_run.assert_not_called()
        mock_loop.assert_not_called()

    def test_cache_miss_falls_back_to_eager_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=None), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            mcp.register_mcp_servers(config)

        mock_run.assert_called_once()

    def test_non_lazy_server_never_touches_cache(self):
        config = {"playwright": {"command": "npx", "args": []}}
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.get_cached_entry") as mock_get, \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            mcp.register_mcp_servers(config)

        mock_get.assert_not_called()
        mock_run.assert_called_once()

    def test_lazy_server_not_reregistered_on_second_pass(self):
        config = _lazy_config()
        mcp._lazy_server_configs["playwright"] = dict(config["playwright"])
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._register_from_cache_sync") as mock_register, \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            names = mcp.register_mcp_servers(config)

        mock_register.assert_not_called()
        mock_run.assert_not_called()
        assert "mcp_playwright_browser_navigate" in names


class TestLazyFirstUseConnect:
    def _connected_server(self):
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=False, content=[], structuredContent=None)
        )
        connected = SimpleNamespace(
            session=mock_session,
            _rpc_lock=MagicMock(),
            _pending_call_context=None,
        )
        connected._rpc_lock.__aenter__ = AsyncMock(return_value=None)
        connected._rpc_lock.__aexit__ = AsyncMock(return_value=None)
        return connected

    @staticmethod
    def _run_on_loop(coro_or_factory, timeout=120):
        import asyncio

        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_tool_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"

        connected = self._connected_server()

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(mcp, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = mcp._make_tool_handler("playwright", "browser_navigate", 5)
            out = handler({}, task_id="t1")

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload.get("result") == ""

    def test_list_resources_handler_lazy_connects_on_first_call(self):
        # Regression for the resource/prompt gap: utility handlers must also
        # route through the first-use connect path, or the first
        # list_resources/get_prompt on a lazy server fails.
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.list_resources = AsyncMock()

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        async def _fake_paginate(list_method, items_attr, server_name):
            return [SimpleNamespace(uri="file:///a", name="a", description="", mimeType="")]

        with patch.object(mcp, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_paginate_full_list", side_effect=_fake_paginate), \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = mcp._make_list_resources_handler("playwright", 5)
            out = handler({})

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload["resources"][0]["uri"] == "file:///a"

    def test_get_prompt_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[])
        )

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(mcp, "_ensure_lazy_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=self._run_on_loop):
            handler = mcp._make_get_prompt_handler("playwright", 5)
            out = handler({"name": "greeting"})

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload

    def test_check_fn_passes_for_lazy_registered_server(self):
        mcp._lazy_server_configs["playwright"] = {"lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        assert mcp._make_check_fn("playwright")() is True

    def test_check_fn_fails_for_unknown_server(self):
        assert mcp._make_check_fn("nope")() is False

    def test_lazy_connect_respects_connect_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        with patch.object(mcp, "_connect_cooldown_active", return_value=True), \
             patch.object(mcp, "_run_on_mcp_loop") as mock_run:
            assert mcp._ensure_lazy_server_connected("playwright") is False
        mock_run.assert_not_called()

    def test_lazy_connect_success_clears_lazy_state(self):
        config = {"command": "npx", "lazy": True}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_browser_navigate"],
        )

        def _fake_run(coro_or_factory, timeout=30):
            mcp._servers["playwright"] = connected
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            return ["mcp_playwright_browser_navigate"]

        with patch.object(mcp, "_ensure_mcp_loop"), \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=_fake_run):
            assert mcp._ensure_lazy_server_connected("playwright") is True

        assert "playwright" not in mcp._lazy_server_configs
        assert "playwright" not in mcp._lazy_server_fingerprints
        assert "playwright" not in mcp._lazy_server_tool_names

    def test_lazy_connect_deregisters_phantom_cached_tools(self):
        # Stale-cache reconciliation: the cached manifest advertised tool X,
        # but the live server only registers tool Y → X must be deregistered
        # after the first-use connect so the model stops seeing a phantom.
        from tools.registry import registry

        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "stale-fp"
        mcp._lazy_server_tool_names["playwright"] = [
            "mcp_playwright_tool_x",
            "mcp_playwright_tool_y",
        ]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_tool_y"],
        )

        def _fake_run(coro_or_factory, timeout=30):
            mcp._servers["playwright"] = connected
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            return ["mcp_playwright_tool_y"]

        with patch.object(mcp, "_ensure_mcp_loop"), \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=_fake_run), \
             patch.object(registry, "deregister") as mock_dereg:
            assert mcp._ensure_lazy_server_connected("playwright") is True

        mock_dereg.assert_called_once_with("mcp_playwright_tool_x")

    def test_lazy_connect_failure_records_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}

        def _fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            coro.close()
            raise RuntimeError("spawn failed")

        with patch.object(mcp, "_ensure_mcp_loop"), \
             patch.object(mcp, "_run_on_mcp_loop", side_effect=_fake_run), \
             patch.object(mcp, "_record_connect_failure") as mock_record:
            assert mcp._ensure_lazy_server_connected("playwright") is False

        mock_record.assert_called_once_with("playwright")
        # Config retained so a later call can retry after cooldown.
        assert "playwright" in mcp._lazy_server_configs


class TestCacheLoadDescriptionScan:
    def test_cache_registration_partial_exception_is_transactional(self, monkeypatch):
        from tools.registry import registry

        entry = {
            "fingerprint": "abc",
            "tools": [
                {"name": "first", "description": "", "inputSchema": {}},
                {"name": "second", "description": "", "inputSchema": {}},
            ],
            "utility_tools": [],
        }
        before_entries = {item.name: item for item in registry.get_all_entries()}
        before_aliases = registry.get_registered_toolset_aliases()
        before_provenance = dict(mcp._mcp_tool_server_names)
        original_convert = mcp._convert_mcp_schema
        calls = 0

        def convert(name, tool):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("partial cache failure")
            return original_convert(name, tool)

        monkeypatch.setattr(mcp, "_convert_mcp_schema", convert)
        try:
            with pytest.raises(RuntimeError, match="partial cache failure"):
                mcp._register_from_cache_sync(
                    "partial", {"command": "unused", "lazy": True}, entry,
                    fingerprint="abc",
                )
            assert {item.name: item for item in registry.get_all_entries()} == before_entries
            assert registry.get_registered_toolset_aliases() == before_aliases
            assert mcp._mcp_tool_server_names == before_provenance
            assert "partial" not in mcp._lazy_server_configs
            assert "partial" not in mcp._lazy_server_fingerprints
            assert "partial" not in mcp._lazy_server_tool_names
        finally:
            for name in ("mcp_partial_first", "mcp_partial_second"):
                if registry.snapshot_registration(name) is not None:
                    registry.deregister(name)
            mcp._mcp_tool_server_names.pop("mcp_partial_first", None)

    def test_scheduler_rejection_rolls_back_real_cached_registration(self, monkeypatch):
        from tools.registry import registry

        entry = {
            "fingerprint": "fp",
            "tools": [
                {"name": "browser_navigate", "description": "Navigate", "inputSchema": {}},
            ],
            "utility_tools": [],
        }
        before_entries = {item.name: item for item in registry.get_all_entries()}
        before_aliases = registry.get_registered_toolset_aliases()
        before_provenance = dict(mcp._mcp_tool_server_names)
        monkeypatch.setattr(mcp, "_MCP_AVAILABLE", True)
        monkeypatch.setattr("tools.mcp_schema_cache.config_fingerprint", lambda _cfg: "fp")
        monkeypatch.setattr("tools.mcp_schema_cache.get_cached_entry", lambda _name, _fp: entry)
        monkeypatch.setattr(mcp, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(
            mcp, "_run_on_mcp_loop",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rejected")),
        )
        try:
            with pytest.raises(RuntimeError, match="rejected"):
                mcp.register_mcp_servers({
                    "lazy": {"command": "unused", "lazy": True},
                    "eager": {"command": "unused"},
                })
            assert {item.name: item for item in registry.get_all_entries()} == before_entries
            assert registry.get_registered_toolset_aliases() == before_aliases
            assert mcp._mcp_tool_server_names == before_provenance
            assert "lazy" not in mcp._lazy_server_configs
            assert "lazy" not in mcp._lazy_server_fingerprints
            assert "lazy" not in mcp._lazy_server_tool_names
            assert not mcp._server_connecting
        finally:
            if registry.snapshot_registration("mcp_lazy_browser_navigate") is not None:
                registry.deregister("mcp_lazy_browser_navigate")
            mcp._mcp_tool_server_names.pop("mcp_lazy_browser_navigate", None)

    def test_scan_runs_on_cache_load_path(self):
        # Defense-in-depth: the cache file is user-writable JSON, so the
        # cache-load registration path must run the same injection scan as
        # eager discovery.
        entry = _fake_cache_entry()
        config = {"command": "npx", "args": [], "lazy": True}
        with patch.object(mcp, "_scan_mcp_description", return_value=[]) as mock_scan, \
             patch.object(mcp, "_convert_mcp_schema", side_effect=RuntimeError("stop")), \
             pytest.raises(RuntimeError):
            mcp._register_from_cache_sync("playwright", config, entry)

        mock_scan.assert_called_once_with("playwright", "browser_navigate", "Navigate")


    @pytest.mark.parametrize("failure_after", [1, 2, 3, 4])
    def test_utility_failure_after_each_write_restores_full_state(
        self, monkeypatch, failure_after
    ):
        from tools.registry import registry

        name = f"utility_failure_{failure_after}"
        entry = {"fingerprint": "fp", "tools": [], "utility_tools": mcp._build_utility_schemas(name)}
        before_entries = {item.name: item for item in registry.get_all_entries()}
        before_aliases = registry.get_registered_toolset_aliases()
        before_maps = {
            "provenance": dict(mcp._mcp_tool_server_names),
            "configs": dict(mcp._lazy_server_configs),
            "fingerprints": dict(mcp._lazy_server_fingerprints),
            "tool_names": dict(mcp._lazy_server_tool_names),
            "trust": dict(mcp._server_trust_levels),
            "hints": dict(mcp._tool_read_only_hints),
        }
        before_sets = {
            "connecting": set(mcp._server_connecting),
            "parallel": set(mcp._parallel_safe_servers),
        }
        original_register = registry.register
        writes = 0

        def register_then_fail(**kwargs):
            nonlocal writes
            result = original_register(**kwargs)
            if kwargs["name"].startswith(f"mcp__{name}__"):
                writes += 1
                if writes == failure_after:
                    raise RuntimeError("utility registration failure")
            return result

        monkeypatch.setattr(registry, "register", register_then_fail)
        with pytest.raises(RuntimeError, match="utility registration failure"):
            mcp._register_from_cache_sync(name, {"lazy": True}, entry, fingerprint="fp")

        assert {item.name: item for item in registry.get_all_entries()} == before_entries
        assert registry.get_registered_toolset_aliases() == before_aliases
        assert dict(mcp._mcp_tool_server_names) == before_maps["provenance"]
        assert dict(mcp._lazy_server_configs) == before_maps["configs"]
        assert dict(mcp._lazy_server_fingerprints) == before_maps["fingerprints"]
        assert dict(mcp._lazy_server_tool_names) == before_maps["tool_names"]
        assert dict(mcp._server_trust_levels) == before_maps["trust"]
        assert dict(mcp._tool_read_only_hints) == before_maps["hints"]
        assert set(mcp._server_connecting) == before_sets["connecting"]
        assert set(mcp._parallel_safe_servers) == before_sets["parallel"]

    @pytest.mark.parametrize("preexisting", [False, True])
    def test_alias_rollback_removes_absent_or_restores_preexisting(self, monkeypatch, preexisting):
        from tools.registry import registry

        name = f"alias_rollback_{preexisting}"
        old_target = "old-toolset"
        if preexisting:
            registry.register_toolset_alias(name, old_target)
        entry = {"fingerprint": "fp", "tools": [], "utility_tools": mcp._build_utility_schemas(name)}
        original_register = registry.register_toolset_alias

        def register_alias_then_fail(alias, toolset):
            original_register(alias, toolset)
            raise RuntimeError("alias failure")

        monkeypatch.setattr(registry, "register_toolset_alias", register_alias_then_fail)
        try:
            with pytest.raises(RuntimeError, match="alias failure"):
                mcp._register_from_cache_sync(name, {"lazy": True}, entry, fingerprint="fp")
            assert registry.get_toolset_alias_target(name) == (old_target if preexisting else None)
        finally:
            if preexisting:
                registry.restore_toolset_alias(name, old_target, None)


class TestResolveServerLazy:
    def test_default_off(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx"}) is False

    def test_explicit_true(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx", "lazy": True}) is True

    def test_explicit_false(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx", "lazy": False}) is False


def _cas_cache_entry(tool_name):
    return {"fingerprint": "fp", "tools": [{"name": tool_name, "description": "", "inputSchema": {}}], "utility_tools": []}


def test_public_cache_rollback_preserves_concurrent_registry_replacement(monkeypatch):
    from tools.registry import registry

    server, tool_name = "cas-registry-race", "cas_registry_race_tool"
    original_register = registry.register
    replaced = threading.Event()

    def register_and_replace(*args, **kwargs):
        written = original_register(*args, **kwargs)
        if kwargs.get("name") == "mcp__cas_registry_race__cas_registry_race_tool":
            def replace():
                original_register(name=kwargs["name"], toolset="foreign", schema={}, handler=lambda _: "foreign", override=True)
                replaced.set()
            worker = threading.Thread(target=replace)
            worker.start()
            worker.join(2)
            assert replaced.is_set()
        return written

    monkeypatch.setattr(registry, "register", register_and_replace)
    monkeypatch.setattr(mcp, "_MCP_AVAILABLE", True)
    monkeypatch.setattr("tools.mcp_schema_cache.config_fingerprint", lambda _cfg: "fp")
    monkeypatch.setattr("tools.mcp_schema_cache.get_cached_entry", lambda _n, _fp: _cas_cache_entry(tool_name))
    original_cached = mcp._register_from_cache_sync

    def register_then_fail(name, config, entry, *, fingerprint=None, journal=None):
        result = original_cached(name, config, entry, fingerprint=fingerprint, journal=journal)
        assert replaced.wait(2)
        raise RuntimeError("registry rollback barrier")

    monkeypatch.setattr(mcp, "_register_from_cache_sync", register_then_fail)
    try:
        with pytest.raises(RuntimeError, match="registry rollback barrier"):
            mcp.register_mcp_servers({server: {"command": "unused", "lazy": True}})
        assert registry.get_toolset_for_tool("mcp__cas_registry_race__cas_registry_race_tool") == "foreign"
    finally:
        registry_name = "mcp__cas_registry_race__cas_registry_race_tool"
        if registry.snapshot_registration(registry_name) is not None:
            registry.deregister(registry_name)


def test_public_cache_rollback_preserves_concurrent_alias_replacement(monkeypatch):
    from tools.registry import registry

    server, tool_name = "cas-alias-race", "cas_alias_race_tool"
    original_alias = registry.register_toolset_alias
    replaced = threading.Event()

    def alias_and_replace(alias, toolset):
        token = original_alias(alias, toolset)
        if alias == server:
            def replace():
                original_alias(alias, "foreign-alias")
                replaced.set()
            worker = threading.Thread(target=replace)
            worker.start()
            worker.join(2)
            assert replaced.is_set()
        return token

    monkeypatch.setattr(registry, "register_toolset_alias", alias_and_replace)
    monkeypatch.setattr(mcp, "_MCP_AVAILABLE", True)
    monkeypatch.setattr("tools.mcp_schema_cache.config_fingerprint", lambda _cfg: "fp")
    monkeypatch.setattr("tools.mcp_schema_cache.get_cached_entry", lambda _n, _fp: _cas_cache_entry(tool_name))
    original_cached = mcp._register_from_cache_sync

    def register_then_fail(name, config, entry, *, fingerprint=None, journal=None):
        result = original_cached(name, config, entry, fingerprint=fingerprint, journal=journal)
        assert replaced.wait(2)
        raise RuntimeError("alias rollback barrier")

    monkeypatch.setattr(mcp, "_register_from_cache_sync", register_then_fail)
    try:
        with pytest.raises(RuntimeError, match="alias rollback barrier"):
            mcp.register_mcp_servers({server: {"command": "unused", "lazy": True}})
        assert registry.get_toolset_alias_target(server) == "foreign-alias"
    finally:
        registry.restore_toolset_alias(server, "foreign-alias", None)


def test_public_cache_rollback_preserves_same_string_alias_aba(monkeypatch):
    """A stale rollback cannot remove a replacement with the same target."""
    from tools.registry import registry

    server, tool_name = "cas-alias-aba", "cas_alias_aba_tool"
    original_alias = registry.register_toolset_alias
    replacement_generation = []

    def alias_and_replace(alias, toolset):
        owned = original_alias(alias, toolset)
        if alias == server:
            replacement = original_alias(alias, toolset)
            replacement_generation.append(replacement.generation)
        return owned

    monkeypatch.setattr(registry, "register_toolset_alias", alias_and_replace)
    monkeypatch.setattr(mcp, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(
        "tools.mcp_schema_cache.config_fingerprint", lambda _cfg: "fp"
    )
    monkeypatch.setattr(
        "tools.mcp_schema_cache.get_cached_entry",
        lambda _name, _fp: _cas_cache_entry(tool_name),
    )
    original_cached = mcp._register_from_cache_sync

    def register_then_fail(name, config, entry, *, fingerprint=None, journal=None):
        original_cached(
            name, config, entry, fingerprint=fingerprint, journal=journal
        )
        raise RuntimeError("alias ABA rollback barrier")

    monkeypatch.setattr(mcp, "_register_from_cache_sync", register_then_fail)
    try:
        with pytest.raises(RuntimeError, match="alias ABA rollback barrier"):
            mcp.register_mcp_servers({server: {"command": "unused", "lazy": True}})
        assert registry.get_toolset_alias_target(server) == f"mcp-{server}"
        token = registry.snapshot_toolset_alias(server)
        assert token is not None
        assert token.generation == replacement_generation[-1]
    finally:
        registry.restore_toolset_alias(server, f"mcp-{server}", None)
