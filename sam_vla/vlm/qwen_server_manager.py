"""
Spawns and supervises the qwen_server subprocess, satisfying the lifecycle. Service protocol so it can be registered with ServiceRegistry.
"""

import json
import os
import signal
import socket
import struct
import subprocess
import time

from sam_vla.vlm.qwen_server import QWEN_SERVER_PORT

_HEADER_SIZE = 4
_HEALTH_CHECK_RETRY_INTERVAL = 1.0
_START_TIMEOUT = 30.0
_STOP_TIMEOUT = 5.0


class QwenServerManager:
    def __init__(self, port: int = None):
        self.port = port if port is not None else QWEN_SERVER_PORT
        self._process = None
        self._owns_process = False

    def _health_check(self, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as conn:
                conn.settimeout(timeout)
                payload = json.dumps({"mode": "ping"}).encode("utf-8")
                conn.sendall(struct.pack(">I", len(payload)) + payload)

                header = conn.recv(_HEADER_SIZE)
                if len(header) < _HEADER_SIZE:
                    return False
                (payload_len,) = struct.unpack(">I", header)
                response = conn.recv(payload_len)
                message = json.loads(response.decode("utf-8"))
                return message.get("status") == "ok"
        except (OSError, socket.timeout, json.JSONDecodeError):
            return False

    def start(self) -> None:
        if self._health_check():
            print(f"[QwenServerManager] server already running on port {self.port}, not spawning")
            self._owns_process = False
            return

        print(f"[QwenServerManager] no server on port {self.port}, spawning subprocess")
        self._process = subprocess.Popen(
            ["python", "-m", "sam_vla.vlm.qwen_server"],
        )
        self._owns_process = True

        deadline = time.time() + _START_TIMEOUT
        while time.time() < deadline:
            if self._health_check():
                print("[QwenServerManager] server is up")
                return
            time.sleep(_HEALTH_CHECK_RETRY_INTERVAL)

        raise RuntimeError(
            f"qwen_server did not respond to ping within {_START_TIMEOUT}s of spawning"
        )

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            print("[QwenServerManager] not owned, skipping shutdown")
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

        self._process = None
        self._owns_process = False


if __name__ == "__main__":
    manager = QwenServerManager()

    manager.start()
    print("health check after first start():", manager._health_check())

    owned_before = manager._owns_process
    manager.start()
    print("owns_process changed on second start():", owned_before != manager._owns_process)

    manager.stop()
    print("health check after stop():", manager._health_check())
