# Telegram bot token incident remediation (GitGuardian)

This runbook is secret-safe and must never contain real token values.

## Incident
- Repository: `maksym-mishchenko/hermes-agent`
- Finding: valid Telegram bot token detected in `tests/test_env_sanitize_on_load.py`
- Severity: Critical

## Repository containment
1. Ensure test fixtures use only redacted/synthetic token values.
2. Never commit runtime `.env` with bot tokens.
3. Keep secret scanning enabled in CI.

## Required provider actions (manual)
1. Revoke the exposed token in BotFather.
2. Issue a new token.
3. Update runtime secret stores/environments with the new token.
4. Restart bot services and verify inbound/outbound messaging.

## Verification
- Secret scans pass on current branch.
- Bot health checks and message round-trip pass after rotation.
