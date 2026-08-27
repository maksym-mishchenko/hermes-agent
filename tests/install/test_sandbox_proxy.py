from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from pathlib import Path


PROXY = Path(__file__).resolve().parents[2] / "scripts" / "sandbox" / "proxy.py"


def _load_proxy(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("sandbox_proxy_test", PROXY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [str(PROXY), str(tmp_path), str(tmp_path), str(tmp_path / "ca.pem")]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    setattr(module, "UPSTREAM_TIMEOUT_SECONDS", 1)
    return module


def test_relay_tunnel_is_bidirectional(tmp_path: Path) -> None:
    proxy = _load_proxy(tmp_path)
    client_proxy, client = socket.socketpair()
    upstream_proxy, upstream = socket.socketpair()
    thread = threading.Thread(
        target=proxy.relay_tunnel,
        args=(client_proxy, upstream_proxy),
        daemon=True,
    )
    thread.start()
    try:
        client.sendall(b"request")
        assert upstream.recv(7) == b"request"
        upstream.sendall(b"response")
        assert client.recv(8) == b"response"
    finally:
        client.close()
        upstream.close()
        client_proxy.close()
        upstream_proxy.close()
        thread.join(timeout=2)
