"""Doc-invariant tests for the hermes-agent skill's session-store guidance.

Agents were hallucinating a `missions` table with `created_at`/`finished_at`
columns. The real store is the `sessions` table in `~/.hermes/state.db` with
`started_at`/`ended_at`. These tests pin the SKILL.md guidance to the *actual*
schema in hermes_state.py (an invariant, not a snapshot), so the doc can't drift
back into describing columns that don't exist.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "autonomous-ai-agents" / "hermes-agent" / "SKILL.md"
STATE_PY = REPO_ROOT / "hermes_state.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sessions_columns() -> set:
    """Parse the column names from the CREATE TABLE sessions statement."""
    text = STATE_PY.read_text(encoding="utf-8")
    m = re.search(
        r"CREATE TABLE IF NOT EXISTS sessions \((.*?)\n\);",
        text,
        re.DOTALL,
    )
    assert m, "Could not locate the sessions table DDL in hermes_state.py"
    body = m.group(1)
    cols = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("FOREIGN KEY", "PRIMARY KEY", "UNIQUE", "CHECK", "--")):
            continue
        col = line.split()[0].strip(",")
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
            cols.add(col)
    return cols


def test_documents_real_timestamp_columns(skill_text, sessions_columns):
    # The doc names the real columns, and they really exist in the schema.
    for col in ("started_at", "ended_at"):
        assert col in skill_text, f"SKILL.md should document the `{col}` column"
        assert col in sessions_columns, f"`{col}` must exist in the sessions table"


def test_warns_no_missions_table(skill_text):
    assert "no `missions` table" in skill_text.lower() or "no missions table" in skill_text.lower()


def test_does_not_recommend_phantom_columns(skill_text, sessions_columns):
    # The hallucinated columns must not exist in the real schema...
    for phantom in ("created_at", "finished_at"):
        assert phantom not in sessions_columns
    # ...and the example SQL the agent might copy must not query them. (The
    # prose is allowed — and expected — to name them in a "not these" warning.)
    lower = skill_text.lower()
    start = lower.find("querying the session store")
    assert start != -1, "session-store subsection should exist"
    end = lower.find("\n### ", start + 1)
    section = lower[start:end if end != -1 else len(lower)]
    sql_blocks = re.findall(r"```sql(.*?)```", section, re.DOTALL)
    assert sql_blocks, "session-store subsection should include an example query"
    for block in sql_blocks:
        assert "created_at" not in block
        assert "finished_at" not in block


def test_references_state_db_and_sessions_table(skill_text):
    assert "state.db" in skill_text
    assert "sessions" in skill_text
