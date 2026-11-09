"""Run the mock CRM on a local port so the client uses HTTP, not an in-process call."""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from supportdesk.crm_server import create_crm_app


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return None


@contextmanager
def running_crm(api_key: str = "test-key") -> Iterator[tuple[str, object]]:
    app = create_crm_app(api_key)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False)
    server = _Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("mock CRM did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        thread.join(timeout=5)
