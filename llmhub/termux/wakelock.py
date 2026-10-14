"""Termux Wake-Lock Management for Android background CPU persistence.

Android OS aggressively puts CPUs into low-power deep sleep / doze mode when
the display is off or the app is not in the foreground. This module wraps
`termux-wake-lock` and `termux-wake-unlock` from `termux-tools` to ensure continuous
token generation and scheduler execution without root permissions.
"""

import atexit
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("llmhub.termux.wakelock")

TERMUX_PREFIX = Path("/data/data/com.termux/files/usr/bin")

class WakelockManager:
    """Manages acquisition and release of Android CPU wake-locks via Termux."""

    def __init__(
        self,
        command_runner: Optional[Callable[[list[str]], int]] = None,
        force_supported: Optional[bool] = None,
    ) -> None:
        self._command_runner = command_runner or self._default_runner
        self._force_supported = force_supported
        self._is_held: bool = False
        self._lock_binary: Optional[str] = self._find_binary("termux-wake-lock")
        self._unlock_binary: Optional[str] = self._find_binary("termux-wake-unlock")
        atexit.register(self._cleanup_at_exit)

    def _cleanup_at_exit(self) -> None:
        if self._is_held:
            try:
                self.release()
            except Exception:
                pass

    @classmethod
    def _find_binary(cls, binary_name: str) -> Optional[str]:
        """Find binary on PATH or in standard Termux prefix."""
        # 1. Check standard PATH
        found = shutil.which(binary_name)
        if found:
            return found
        # 2. Check standard Termux prefix directly
        direct = TERMUX_PREFIX / binary_name
        if direct.exists() and os.access(direct, os.X_OK):
            return str(direct)
        return None

    @property
    def is_supported(self) -> bool:
        """Check if termux-wake-lock is supported on the current host."""
        if self._force_supported is not None:
            return self._force_supported
        return bool(self._lock_binary and self._unlock_binary)

    @property
    def is_held(self) -> bool:
        """Return True if wake-lock is currently active."""
        return self._is_held

    @staticmethod
    def _default_runner(cmd: list[str]) -> int:
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
                check=False,
            )
            return res.returncode
        except Exception as e:
            logger.debug("Failed to execute %s: %s", cmd, e)
            return 1

    def acquire(self) -> bool:
        """Acquire CPU wake-lock, preventing Android from throttling or suspending the CPU.
        
        Returns True if acquired (or already held), False if unsupported or execution failed.
        """
        if self._is_held:
            return True

        if not self.is_supported:
            logger.debug(
                "termux-wake-lock not available on this platform. Running without Android wake-lock."
            )
            return False

        cmd = [self._lock_binary or "termux-wake-lock"]
        ret = self._command_runner(cmd)
        if ret == 0:
            self._is_held = True
            logger.info("Acquired Termux CPU wake-lock (Android deep sleep disabled).")
            return True
        else:
            logger.warning("termux-wake-lock returned non-zero exit code: %d", ret)
            return False

    def release(self) -> bool:
        """Release CPU wake-lock, allowing system power management to resume.
        
        Returns True if released (or was not held), False if execution failed.
        """
        if not self._is_held:
            return True

        if not self.is_supported:
            self._is_held = False
            return True

        cmd = [self._unlock_binary or "termux-wake-unlock"]
        ret = self._command_runner(cmd)
        if ret == 0:
            self._is_held = False
            logger.info("Released Termux CPU wake-lock (Android power management restored).")
            return True
        else:
            logger.warning("termux-wake-unlock returned non-zero exit code: %d", ret)
            # Retain _is_held as True so caller can retry release
            return False

    def __enter__(self) -> "WakelockManager":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    async def __aenter__(self) -> "WakelockManager":
        self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
