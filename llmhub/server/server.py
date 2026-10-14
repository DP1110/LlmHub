"""Server runner launching the FastAPI application on host & port with PID tracking."""

import os
import signal
import sys
from pathlib import Path
from typing import Optional
import uvicorn

from llmhub.config import LLMHubConfig

def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config: Optional[LLMHubConfig] = None
) -> None:
    cfg = config or LLMHubConfig()
    server_cfg = cfg.settings.get("server", {})
    bind_host = host or server_cfg.get("host", "127.0.0.1")
    bind_port = port or server_cfg.get("port", 11434)

    pid_path = cfg.run_dir / "llmhub.pid"
    pid = os.getpid()

    # Write PID safely
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        if pid_path.exists():
            try:
                existing_pid = int(pid_path.read_text().strip())
                # Check if process is alive
                import psutil
                if psutil.pid_exists(existing_pid) and existing_pid != pid:
                    sys.stderr.write(f"[WARN] Existing LLMHub daemon process with PID {existing_pid} is active.\n")
            except Exception:
                pass
        pid_path.write_text(str(pid))
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to write PID file {pid_path}: {e}\n")

    sys.stdout.write(f"\n=================================================================\n")
    sys.stdout.write(f" LLMHub Server starting on http://{bind_host}:{bind_port}\n")
    sys.stdout.write(f" PID: {pid} | Storage: {cfg.base_dir}\n")
    sys.stdout.write(f" OpenAI Endpoint: http://{bind_host}:{bind_port}/v1/chat/completions\n")
    sys.stdout.write(f" Press Ctrl+C to terminate gracefully.\n")
    sys.stdout.write(f"=================================================================\n\n")

    def cleanup(signum=None, frame=None):
        if pid_path.exists():
            try:
                if pid_path.read_text().strip() == str(pid):
                    pid_path.unlink()
            except Exception:
                pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
    except Exception:
        pass

    try:
        uvicorn.run(
            "llmhub.server.app:app",
            host=bind_host,
            port=bind_port,
            log_level="info"
        )
    finally:
        cleanup()

if __name__ == "__main__":
    run_server()
