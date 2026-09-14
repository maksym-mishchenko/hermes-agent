from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.error_classifier import FailoverReason
from agent.turn_api_error import settle_unrecovered_error


def test_degraded_copilot_route_falls_back_without_re_resolving_credentials():
    """A provider-confirmed model 400 must not retry a known raw-token route."""
    entry = SimpleNamespace(id="copilot-entry", copilot_exchange_degraded=True)
    pool = SimpleNamespace(entries=lambda: [entry])
    agent = SimpleNamespace(
        provider="copilot",
        _credential_pool=pool,
        _credential_pool_entry_id=entry.id,
        _has_pending_fallback=Mock(return_value=True),
        _try_recover_stale_copilot_credential=Mock(return_value=True),
        _buffer_status=Mock(),
        _try_activate_fallback=Mock(return_value=True),
        _buffer_vprint=Mock(),
    )
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False)
    classified = SimpleNamespace(
        retryable=False,
        should_compress=False,
        reason=FailoverReason.model_not_found,
    )
    error = SimpleNamespace(status_code=400, message="model_not_available_for_integrator")

    with patch("agent.conversation_loop._is_copilot_provider", return_value=True), \
         patch("agent.conversation_loop._is_stale_copilot_credential_error", return_value=True), \
         patch("agent.conversation_loop._arm_fallback_restart", return_value="prompt"):
        verdict = settle_unrecovered_error(
            agent,
            api_error=error,
            classified=classified,
            _retry=retry,
            status_code=400,
            error_msg=str(error.message),
            is_context_length_error=False,
            is_rate_limited=False,
            _is_zai_coding_overload=False,
            _provider="copilot",
            _base="https://api.githubcopilot.com",
            _model="enterprise/model",
            messages=[],
            api_messages=[],
            api_kwargs={},
            active_system_prompt="prompt",
            conversation_history=[],
            approx_tokens=1,
            retry_count=0,
            max_retries=2,
            compression_attempts=0,
            api_call_count=1,
        )

    assert verdict.action == "break"
    agent._try_recover_stale_copilot_credential.assert_not_called()
    agent._try_activate_fallback.assert_called_once_with()
