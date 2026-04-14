"""Start and stop an SGLang inference server as a managed subprocess."""

from __future__ import annotations

import atexit
import signal
import subprocess
import sys
from typing import Optional

from src.server.health import wait_for_server


class SGLangServer:
    """Context manager / standalone launcher for an SGLang server process."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 30000,
        extra_args: Optional[list] = None,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self._process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, wait: bool = True, retries: int = 60) -> None:
        """Launch the SGLang server subprocess."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ] + self.extra_args

        self._process = subprocess.Popen(cmd)
        atexit.register(self.stop)

        if wait:
            ok = wait_for_server(self.base_url, retries=retries)
            if not ok:
                self.stop()
                raise RuntimeError(
                    f"SGLang server did not become healthy at {self.base_url} "
                    f"after {retries} retries"
                )

    def stop(self) -> None:
        """Terminate the server process if running."""
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "SGLangServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
