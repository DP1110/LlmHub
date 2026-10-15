"""Process Lifecycle, PID Management, and Crash Recovery for LLMHub.

Manages:
- ~/.llmhub/run/llmhub.pid tracking process ID, host, port, and timestamp
- Detection and recovery from stale PID locks following hard kills or crashes
- Safe cleanup of orphaned run-directory sockets, temporary files, and locks
- Signal handling coordination for graceful server draining and shutdown
"""

import datetime
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("llmhub.lifecycle")


class ServerAlreadyRunningError(Exception):
    """Raised when an active LLMHub server instance is already running."""
    pass


def is_process_alive(pid: int) -> bool:
    """Check if process with given PID is currently active across platforms."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False

    # 1. Try psutil first if available
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass

    # 2. Windows fallback using Win32 API ctypes
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = kernel32.OpenProcess(0x1000, False, pid)
            if h:
                kernel32.CloseHandle(h)
                return True
            return False
        except Exception:
            pass

    # 3. Fallback to os.kill(pid, 0) for POSIX/Termux
    try:
        os.kill(pid, 0)
        return True
    except OSError as err:
        import errno
        # ESRCH means no such process; EPERM means exists but owned by other user
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            return True
        return False
    except (ValueError, Exception):
        return False


def is_llmhub_process(pid: int) -> bool:
    """Check if process is alive and plausibly an LLMHub/python process to protect against PID recycling."""
    if not is_process_alive(pid):
        return False
    try:
        import psutil
        proc = psutil.Process(pid)
        # On Android Termux, check if UID matches our user to rule out foreign system daemons
        if hasattr(os, "getuid"):
            try:
                if proc.uids().real != os.getuid():
                    return False
            except Exception:
                pass
        cmd = " ".join(proc.cmdline()).lower()
        if "python" in cmd or "llmhub" in cmd:
            return True
        return False
    except Exception:
        # If psutil cannot inspect cmdline (e.g. unprivileged Termux), assume alive to be safe
        return True


class PIDManager:
    """Manages PID file creation, verification, crash detection, and cleanup."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.pid_file = self.run_dir / "llmhub.pid"

    def read_pid_info(self) -> Optional[Dict[str, Any]]:
        """Read existing PID file data if present, supporting both JSON and plain-integer PID files."""
        if not self.pid_file.exists():
            return None
        try:
            with open(self.pid_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return None
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        pid_val = data.get("pid")
                        if isinstance(pid_val, int) and not isinstance(pid_val, bool) and pid_val > 0:
                            return data
                        elif isinstance(pid_val, str) and pid_val.isdigit() and int(pid_val) > 0:
                            data["pid"] = int(pid_val)
                            return data
                        return None
                    elif type(data) is int and data > 0:
                        return {"pid": data}
                except Exception:
                    pass
                if content.isdigit() and int(content) > 0:
                    return {"pid": int(content)}
                return None
        except Exception as e:
            logger.warning("Corrupt PID file at %s: %s", self.pid_file, e)
            return None

    def acquire(self, host: str, port: int, version: str) -> None:
        """Verify no active instance is running, cleaning up stale files if crashed.
        
        Raises ServerAlreadyRunningError if an active instance is running.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        existing = self.read_pid_info()

        if existing and isinstance(existing, dict):
            old_pid = existing.get("pid")
            if old_pid and is_llmhub_process(old_pid) and old_pid != os.getpid():
                raise ServerAlreadyRunningError(
                    f"LLMHub server is already running with PID {old_pid} "
                    f"listening on {existing.get('host', 'unknown')}:{existing.get('port', 'unknown')} "
                    f"(started at {existing.get('started_at', 'unknown')})."
                )
            else:
                # Process is dead or recycled; previous instance crashed without cleaning up
                logger.warning(
                    "Detected stale PID file for dead/recycled process %s. Recovering and cleaning up.",
                    old_pid,
                )
                self.release(force=True)

        # Write current process PID info atomically via temporary file rename
        pid_data = {
            "pid": os.getpid(),
            "host": host,
            "port": port,
            "version": version,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with tempfile.NamedTemporaryFile("w", dir=str(self.run_dir), delete=False, encoding="utf-8") as tmp:
            json.dump(pid_data, tmp, indent=2)
            tmp_path = Path(tmp.name)

        try:
            tmp_path.replace(self.pid_file)
        except Exception:
            # Fallback for Windows if target exists
            if self.pid_file.exists():
                try:
                    self.pid_file.unlink()
                except Exception:
                    pass
            tmp_path.replace(self.pid_file)

        logger.info("Acquired PID file %s for PID %d", self.pid_file, os.getpid())

    def release(self, force: bool = False) -> None:
        """Remove PID file safely, verifying ownership unless force=True."""
        try:
            if self.pid_file.exists():
                info = self.read_pid_info()
                target_pid = info.get("pid") if (info and isinstance(info, dict)) else None
                if force or (target_pid == os.getpid()):
                    try:
                        self.pid_file.unlink()
                    except FileNotFoundError:
                        pass
                    logger.info("Removed PID file %s for PID %s", self.pid_file, target_pid or os.getpid())
        except Exception as e:
            logger.warning("Failed to remove PID file %s: %s", self.pid_file, e)

    def cleanup_run_dir(self) -> List[str]:
        """Clean up orphaned sockets, lock files, or temporary files in ~/.llmhub/run/."""
        cleaned: List[str] = []
        if not self.run_dir.exists():
            return cleaned

        for item in self.run_dir.iterdir():
            # Don't delete active PID file
            if item.name == "llmhub.pid":
                continue
            if item.suffix in (".lock", ".sock", ".tmp") or item.name.startswith("orphan-") or item.name.startswith("tmp"):
                try:
                    if item.is_file() or item.is_socket():
                        item.unlink()
                        cleaned.append(item.name)
                except Exception as e:
                    logger.warning("Failed to clean file %s: %s", item, e)
        return cleaned
