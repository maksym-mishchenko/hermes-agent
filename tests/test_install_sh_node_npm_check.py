"""Regression tests for install.sh Node/npm checks (#77003).

A stray `node` symlink without a sibling `npm` (leftover from a node
version manager) made the installer report "✓ Node.js found" and then fail
opaquely at the desktop stage. Node must only count as found when npm
resolves on the same PATH, and npm install stages must not report success
when the install actually failed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_check_node_requires_npm_alongside_node() -> None:
    """check_node must not report success when only `node` resolves.

    Before the fix, `command -v node` succeeding was enough — a stray node
    symlink (no sibling npm) passed the check, every later `npm install`
    failed silently, and the desktop build died with an opaque
    "Node.js / npm unavailable" (#77003).
    """
    text = INSTALL_SH.read_text()

    # The system-toolchain branch now gates on BOTH node and npm.
    assert (
        "if command -v node &> /dev/null && command -v npm &> /dev/null \\" in text
    )
    # The "node found but npm missing" case has its own explicit branch that
    # falls through to installing the Hermes-managed Node (which bundles npm).
    assert "node found but npm is not on PATH (stray node symlink?)" in text


def test_check_node_managed_requires_npm() -> None:
    """Managed Node must reject npm versions that cannot honor repo policy."""
    text = INSTALL_SH.read_text()
    assert (
        '[ -x "$HERMES_HOME/node/bin/node" ] && [ -x "$HERMES_HOME/node/bin/npm" ] \\'
        in text
    )
    managed = text.split(
        "# Prefer a Hermes-managed Node from a previous run", 1
    )[1].split("if command -v node", 1)[0]
    assert "npm_supports_npmrc" in managed
    assert "install_node" in managed


def test_fresh_managed_node_repairs_incompatible_bundled_npm() -> None:
    text = INSTALL_SH.read_text()
    install_node = text.split("install_node()", 1)[1].split(
        "check_network_prerequisites()", 1
    )[0]
    assert "npm_supports_npmrc" in install_node
    assert 'npm" install -g --prefix "$HERMES_HOME/node"' in install_node
    assert "npm@latest" in install_node


def test_managed_node_drops_caller_node_gyp_header_override() -> None:
    text = INSTALL_SH.read_text()
    install_deps = text.split("install_node_deps()", 1)[1].split(
        "install_browser_use_cli()", 1
    )[0]
    assert 'readlink -f "$(command -v node)"' in install_deps
    assert '"$HERMES_HOME/node/bin/node"' in install_deps
    assert "unset npm_config_nodedir npm_package_config_node_gyp_nodedir" in install_deps

