"""Integration project (next.md) Phase 1: exercises
NavdpUpstreamServerManager's path resolution and start()/stop() state
machine against mocked subprocess/socket/HTTP calls -- no real navdp_server
subprocess, checkpoint, or vendored checkout needed. See this module's
docstring for why start() is two-phase (poll a raw TCP connect, then exactly
one long-timeout /navigator_reset call) rather than repeatedly polling
/navigator_reset the way the socket-based server managers poll a ping route.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sam_vla.vlm.navdp_upstream_server_manager import (
    NavdpUpstreamServerManager,
    resolve_navdp_upstream_root,
)


def test_resolve_navdp_upstream_root_raises_when_not_found(tmp_path, monkeypatch):
    monkeypatch.delenv("NAVDP_UPSTREAM_ROOT", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_navdp_upstream_root(str(tmp_path))


def test_resolve_navdp_upstream_root_finds_raw_path(tmp_path, monkeypatch):
    monkeypatch.delenv("NAVDP_UPSTREAM_ROOT", raising=False)
    server_dir = tmp_path / "baselines" / "navdp"
    server_dir.mkdir(parents=True)
    (server_dir / "navdp_server.py").write_text("# stub")

    resolved = resolve_navdp_upstream_root(str(tmp_path))
    assert resolved == tmp_path.resolve()


def test_resolve_navdp_upstream_root_finds_env_var(tmp_path, monkeypatch):
    server_dir = tmp_path / "baselines" / "navdp"
    server_dir.mkdir(parents=True)
    (server_dir / "navdp_server.py").write_text("# stub")
    monkeypatch.setenv("NAVDP_UPSTREAM_ROOT", str(tmp_path))

    resolved = resolve_navdp_upstream_root(None)
    assert resolved == tmp_path.resolve()


def _make_manager(**kwargs) -> NavdpUpstreamServerManager:
    with patch(
        "sam_vla.vlm.navdp_upstream_server_manager.resolve_navdp_upstream_root",
        return_value=Path("/fake/navdp_upstream"),
    ):
        return NavdpUpstreamServerManager(checkpoint_path="/fake/ckpt.ckpt", **kwargs)


def test_start_adopts_already_running_server_without_spawning():
    manager = _make_manager()
    manager._port_open = MagicMock(return_value=True)
    manager._reset = MagicMock(return_value=True)

    with patch("sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen") as popen:
        manager.start()

    popen.assert_not_called()
    assert manager._owns_process is False
    assert manager.load_ms == 0.0


def test_start_spawns_and_waits_for_port_then_reset(monkeypatch):
    manager = _make_manager(start_timeout=5.0)

    # First health check (adopt-existing path) fails on _port_open alone --
    # short-circuits before _reset is ever called -- so we must spawn.
    port_open_results = iter([False, False, False, True])
    manager._port_open = MagicMock(side_effect=lambda timeout=1.0: next(port_open_results))
    # Only called once for real: the post-spawn /navigator_reset once the port opens.
    manager._reset = MagicMock(return_value=True)

    fake_process = MagicMock()
    fake_process.poll.return_value = None  # still running
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen",
        MagicMock(return_value=fake_process),
    )
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        lambda: "/fake/python",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    manager.start()

    assert manager._owns_process is True
    assert manager.load_ms is not None and manager.load_ms >= 0.0


def test_start_raises_if_subprocess_exits_early(monkeypatch):
    manager = _make_manager(start_timeout=5.0)
    manager._port_open = MagicMock(return_value=False)
    manager._reset = MagicMock(return_value=False)

    fake_process = MagicMock()
    fake_process.poll.return_value = 1  # exited already
    fake_process.returncode = 1
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen",
        MagicMock(return_value=fake_process),
    )
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        lambda: "/fake/python",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="exited early"):
        manager.start()


def test_stop_only_terminates_owned_process():
    manager = _make_manager()
    manager._owns_process = False
    manager._process = MagicMock()

    manager.stop()

    manager._process.terminate.assert_not_called()
