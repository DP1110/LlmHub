"""Server runner launching the FastAPI application on host & port with PID tracking."""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional
import psutil
import uvicorn

from llmhub.config import LLMHubConfig

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the given port is currently bound by a listening socket."""
    import socket
    target_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((target_host, port)) == 0
    except Exception:
        return False

def kill_process_on_port(port: int) -> bool:
    """Attempt to terminate any process currently listening on the specified port."""
    killed = False
    current_pid = os.getpid()
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.pid == current_pid:
                continue
            try:
                conns = proc.net_connections(kind='inet') if hasattr(proc, 'net_connections') else proc.connections(kind='inet')
                for conn in conns:
                    if conn.laddr.port == port:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2.0)
                        except Exception:
                            proc.kill()
                        killed = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except Exception:
                pass
    except Exception:
        pass
    return killed

def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config: Optional[LLMHubConfig] = None,
    force: bool = False,
    log_level: Optional[str] = None
) -> None:
    cfg = config or LLMHubConfig()
    server_cfg = cfg.settings.get("server", {})
    bind_host = host or server_cfg.get("host", "127.0.0.1")
    bind_port = port or server_cfg.get("port", 11434)
    probe_host = "127.0.0.1" if bind_host in ("0.0.0.0", "", "::") else bind_host

    from llmhub.core.lifecycle import PIDManager
    pid_mgr = PIDManager(cfg.run_dir)
    pid = os.getpid()

    # Check for port collisions before attempting to bind
    if is_port_in_use(bind_port, bind_host):
        if force:
            sys.stdout.write(f"[INFO] Port {bind_port} is occupied. Reclaiming port (--force)...\n")
            pid_info = pid_mgr.read_pid_info()
            if pid_info and pid_info.get("pid"):
                old_pid = pid_info["pid"]
                try:
                    if psutil.pid_exists(old_pid) and old_pid != pid:
                        p = psutil.Process(old_pid)
                        p.terminate()
                        try:
                            p.wait(timeout=1.0)
                        except psutil.TimeoutExpired:
                            p.kill()
                except Exception:
                    pass
                pid_mgr.release(force=True)

            kill_process_on_port(bind_port)
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline and is_port_in_use(bind_port, bind_host):
                time.sleep(0.15)

            if is_port_in_use(bind_port, bind_host):
                sys.stderr.write(f"[ERROR] Could not free port {bind_port}. Please kill the process manually.\n")
                sys.exit(1)
        else:
            # Check if existing server responds to healthz
            is_healthy = False
            try:
                import httpx
                with httpx.Client(timeout=1.5) as client:
                    resp = client.get(f"http://{probe_host}:{bind_port}/healthz")
                    is_healthy = (resp.status_code == 200)
            except Exception:
                pass

            if is_healthy:
                pid_info = pid_mgr.read_pid_info()
                existing_pid = pid_info.get("pid", "unknown") if pid_info else "unknown"
                sys.stdout.write(f"\n=================================================================\n")
                sys.stdout.write(f" LLMHub Server is ALREADY RUNNING on http://{bind_host}:{bind_port}\n")
                sys.stdout.write(f" PID: {existing_pid} | Storage: {cfg.base_dir}\n")
                sys.stdout.write(f" Web UI Dashboard: http://{probe_host}:{bind_port}/\n")
                sys.stdout.write(f" OpenAI Endpoint:  http://{probe_host}:{bind_port}/v1/chat/completions\n")
                sys.stdout.write(f"\n To restart the server: llmhub serve --force\n")
                sys.stdout.write(f" To stop the server:    llmhub stop\n")
                sys.stdout.write(f"=================================================================\n\n")
                sys.exit(0)
            else:
                sys.stdout.write(f"[WARN] Port {bind_port} is occupied by an unresponsive process. Cleaning up...\n")
                kill_process_on_port(bind_port)
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() < deadline and is_port_in_use(bind_port, bind_host):
                    time.sleep(0.15)

    # Acquire PID tracking
    try:
        from llmhub import __version__
        pid_mgr.acquire(host=bind_host, port=bind_port, version=__version__)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to write PID info via PIDManager: {e}\n")

    sys.stdout.write(f"\n=================================================================\n")
    sys.stdout.write(f" LLMHub Server starting on http://{bind_host}:{bind_port}\n")
    sys.stdout.write(f" PID: {pid} | Storage: {cfg.base_dir}\n")
    sys.stdout.write(f" OpenAI Endpoint: http://{probe_host}:{bind_port}/v1/chat/completions\n")
    sys.stdout.write(f" Press Ctrl+C to terminate gracefully.\n")
    sys.stdout.write(f"=================================================================\n\n")

    def cleanup(signum=None, frame=None):
        try:
            pid_mgr.release()
        except Exception:
            pass
        if signum is not None:
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
            log_level=(log_level or "info").lower()
        )
    finally:
        cleanup()

if __name__ == "__main__":
    run_server()
