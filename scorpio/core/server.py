"""Run an SGLang server for the lifetime of a job."""

from __future__ import annotations

import contextlib
import random
import socket
import subprocess
import time

import requests

HEALTH_TIMEOUT = 1800
POLL_SECONDS = 5


def _free_port() -> int:
    """A free port, low enough that the server's port + 10000 stays legal."""
    for _ in range(100):
        port = random.randint(20000, 45000)
        with socket.socket() as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in range")


def _owns_port(port: int, pid: int) -> bool:
    """Whether our server holds the port.

    A healthy /health only proves something is listening. On a shared node
    that something can be another job's server, which would silently mix two
    models' generations together.
    """
    out = subprocess.run(["ss", "-lptn", f"sport = :{port}"],
                         capture_output=True, text=True).stdout
    return f"pid={pid}" in out


@contextlib.contextmanager
def serve(model_path: str, args: list[str]):
    """Start a server, yield its base URL, and stop it on the way out."""
    port = _free_port()
    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", model_path,
           "--host", "0.0.0.0", "--port", str(port), *args]
    process = subprocess.Popen(cmd)
    try:
        url = f"http://localhost:{port}"
        deadline = time.time() + HEALTH_TIMEOUT
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            try:
                if requests.get(f"{url}/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(POLL_SECONDS)
        else:
            raise TimeoutError(f"server not healthy after {HEALTH_TIMEOUT}s")

        if not _owns_port(port, process.pid):
            raise RuntimeError(f"port {port} is held by another process")

        yield f"{url}/v1"
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
