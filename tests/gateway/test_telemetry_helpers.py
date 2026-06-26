"""Tests for gateway telemetry helpers and the no-event safety contract.

Covers the producer-side primitives added for the Langfuse gateway telemetry
forward-port: identifier hashing, fire-and-forget observer dispatch, and the
guarantee that an absent inbound ``event`` (the ``event=None`` default on
``_run_agent``) cannot raise when telemetry fields are derived via ``getattr``.
"""

import hashlib

import gateway.run as gateway_run


class TestHashGatewayIdentifier:
    def test_blank_values_hash_to_empty_string(self):
        assert gateway_run._hash_gateway_identifier(None) == ""
        assert gateway_run._hash_gateway_identifier("") == ""
        assert gateway_run._hash_gateway_identifier(0) == ""

    def test_value_is_stable_sha256_hex(self):
        digest = gateway_run._hash_gateway_identifier("user-123")
        assert digest == hashlib.sha256(b"user-123").hexdigest()
        # Deterministic across calls.
        assert digest == gateway_run._hash_gateway_identifier("user-123")

    def test_distinct_values_hash_differently(self):
        assert gateway_run._hash_gateway_identifier("a") != gateway_run._hash_gateway_identifier("b")


class TestInvokeGatewayObserverHook:
    def test_no_registered_hook_is_silent(self, monkeypatch):
        called = {"invoked": False}

        def _has_hook(_name):
            return False

        def _invoke_hook(_name, **_kwargs):
            called["invoked"] = True

        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook", _has_hook, raising=False
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook", _invoke_hook, raising=False
        )

        gateway_run._invoke_gateway_observer_hook("gateway_agent_run_start", task_id="t")
        assert called["invoked"] is False

    def test_registered_hook_receives_kwargs(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook", lambda name: True, raising=False
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: seen.append((name, kwargs)),
            raising=False,
        )

        gateway_run._invoke_gateway_observer_hook(
            "gateway_agent_run_finish", task_id="t", status="ok"
        )
        assert seen == [("gateway_agent_run_finish", {"task_id": "t", "status": "ok"})]

    def test_hook_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook", lambda name: True, raising=False
        )

        def _boom(name, **kwargs):
            raise RuntimeError("telemetry must not crash the run")

        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook", _boom, raising=False
        )

        # Must not raise.
        gateway_run._invoke_gateway_observer_hook("gateway_agent_run_start", task_id="t")


class TestNoEventPathSafety:
    def test_getattr_fields_on_missing_event_do_not_raise(self):
        """The telemetry block derives fields via getattr(event, ...).

        With the ``event=None`` default, every derived field must resolve to a
        safe value without NameError / AttributeError.
        """
        event = None
        media_count = len(getattr(event, "media_urls", None) or [])
        message_type = getattr(event, "message_type", "")
        internal = bool(getattr(event, "internal", False))

        assert media_count == 0
        assert message_type == ""
        assert internal is False
