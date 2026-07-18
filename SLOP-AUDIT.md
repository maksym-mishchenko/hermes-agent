# SLOP-AUDIT: hermes-agent

**Audited:** 2026-07-18
**Auditor:** hermes / repo-slop-audit (autopilot sweep)

## Debris file candidates (T1 auto-delete targets)
- ./tests/plugins/memory/test_mem0_v3.py

## Agent-scratchpad candidates

## TODO/FIXME/HACK count
- Python: 76 occurrences
- TS/TSX: 7 occurrences

## Silent-swallow patterns
- Python bare/silent except: 3451
- TS silent catches: 0

## God files (>500 lines)
- ./gateway/run.py (18683 lines)
- ./cli.py (15650 lines)
- ./hermes_cli/web_server.py (13806 lines)
- ./tui_gateway/server.py (13517 lines)
- ./hermes_cli/main.py (13382 lines)
- ./hermes_cli/kanban_db.py (8617 lines)
- ./tests/test_tui_gateway_server.py (8449 lines)
- ./hermes_cli/auth.py (8276 lines)
- ./plugins/platforms/telegram/adapter.py (7576 lines)
- ./hermes_cli/config.py (7402 lines)
