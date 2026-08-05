"""Tests for session-retention observability in _run_state_db_auto_maintenance.

Regression tests for the audit finding where sessions.retention_days is
configured (e.g. 14 days) but auto_prune is false (the default), causing
the policy to be silently never enforced.

These tests verify:
  - No prune when auto_prune=False, no advisory when retention_days==default
  - Advisory info log when auto_prune=False but retention_days is non-default
  - Pruning runs when auto_prune=True
  - No advisory when auto_prune=True (it just runs)
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_session_db():
    db = MagicMock()
    db.get_meta.return_value = None
    db.maybe_auto_prune_and_vacuum.return_value = {"skipped": False, "pruned": 0, "vacuumed": False}
    db.prune_empty_ghost_sessions.return_value = 0
    db.finalize_orphaned_compression_sessions.return_value = 0
    return db


def _run_maintenance(session_cfg, db):
    """Import and call the function under test with a mocked config."""
    full_cfg = {"sessions": session_cfg}
    from hermes_constants import get_hermes_home
    with patch("cli.load_config", return_value=full_cfg, create=True), \
         patch("hermes_cli.config.load_config", return_value=full_cfg):
        from cli import _run_state_db_auto_maintenance
        _run_state_db_auto_maintenance(db)


class TestSessionRetentionObservability:
    def test_no_prune_and_no_advisory_when_default_config(self, mock_session_db, caplog):
        """With default config (auto_prune=False, retention_days=90), no prune, no warning."""
        with caplog.at_level(logging.INFO, logger="cli"):
            _run_maintenance({"auto_prune": False, "retention_days": 90}, mock_session_db)

        mock_session_db.maybe_auto_prune_and_vacuum.assert_not_called()
        assert "sessions.auto_prune=false" not in caplog.text

    def test_advisory_emitted_when_retention_days_nondefault_and_auto_prune_false(
        self, mock_session_db, caplog
    ):
        """When retention_days != 90 but auto_prune=False, log an advisory."""
        with caplog.at_level(logging.INFO, logger="cli"):
            _run_maintenance({"auto_prune": False, "retention_days": 14}, mock_session_db)

        mock_session_db.maybe_auto_prune_and_vacuum.assert_not_called()
        assert "auto_prune=false" in caplog.text
        assert "14" in caplog.text

    def test_prune_runs_when_auto_prune_true(self, mock_session_db, caplog):
        """When auto_prune=True, maybe_auto_prune_and_vacuum is called."""
        with caplog.at_level(logging.DEBUG, logger="cli"):
            _run_maintenance(
                {"auto_prune": True, "retention_days": 14, "min_interval_hours": 24,
                 "vacuum_after_prune": True},
                mock_session_db,
            )

        mock_session_db.maybe_auto_prune_and_vacuum.assert_called_once()
        call_kwargs = mock_session_db.maybe_auto_prune_and_vacuum.call_args
        assert call_kwargs.kwargs.get("retention_days") == 14 or call_kwargs.args[0] == 14

    def test_no_advisory_when_auto_prune_true(self, mock_session_db, caplog):
        """When auto_prune=True the advisory must not fire (it runs, not warns)."""
        with caplog.at_level(logging.INFO, logger="cli"):
            _run_maintenance({"auto_prune": True, "retention_days": 14}, mock_session_db)

        assert "auto_prune=false" not in caplog.text

    def test_none_session_db_returns_early(self, caplog):
        """Passing None for session_db must return without any error."""
        from cli import _run_state_db_auto_maintenance
        # Should not raise
        _run_state_db_auto_maintenance(None)
