"""API Key Authentication & Token Bucket Rate Limiting for LLMHub.

Provides:
- TokenBucket algorithm tracking requests (RPM) and token usage (TPM) with thread-safe refunds
- RateLimiter managing per-key or per-client IP buckets with actual token consumption
- APIKeyAuthenticator validating Bearer tokens with constant-time comparison and proxy awareness
"""

import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class TokenBucket:
    """Thread-safe Token Bucket for smooth rate limiting with burst support."""

    def __init__(self, capacity: float, refill_rate_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def refund(self, amount: float = 1.0) -> None:
        """Thread-safe refund of consumed tokens."""
        with self._lock:
            self.tokens = min(self.capacity, self.tokens + amount)

    def consume(self, cost: float = 1.0) -> Tuple[bool, float, int, int]:
        """Attempt to consume cost units from the bucket.
        
        Returns:
            (allowed: bool, retry_after_seconds: float, remaining: int, limit: int)
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            # Replenish tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            limit = int(self.capacity)
            if cost > self.capacity:
                # Request exceeds total bucket capacity; unserviceable without reconfiguring limits
                return False, -1.0, 0, limit

            if self.tokens >= cost:
                self.tokens -= cost
                remaining = int(self.tokens)
                return True, 0.0, remaining, limit
            else:
                shortfall = cost - self.tokens
                retry_after = shortfall / self.refill_rate if self.refill_rate > 0 else 60.0
                return False, retry_after, 0, limit


class RateLimiter:
    """Manages TokenBuckets for RPM (Requests Per Minute) and TPM (Tokens Per Minute)."""

    def __init__(
        self,
        default_rpm: int = 60,
        default_tpm: int = 30000,
        cleanup_interval_s: float = 300.0,
    ) -> None:
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self.cleanup_interval_s = cleanup_interval_s
        self._buckets: Dict[str, Tuple[TokenBucket, TokenBucket, float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def _get_buckets(
        self, client_id: str, custom_rpm: Optional[int] = None, custom_tpm: Optional[int] = None
    ) -> Tuple[TokenBucket, TokenBucket]:
        rpm = custom_rpm if custom_rpm is not None else self.default_rpm
        tpm = custom_tpm if custom_tpm is not None else self.default_tpm

        with self._lock:
            now = time.monotonic()
            # Periodic cleanup of idle client buckets
            if now - self._last_cleanup > self.cleanup_interval_s:
                cutoff = now - self.cleanup_interval_s
                stale = [k for k, (_, _, last_t) in self._buckets.items() if last_t < cutoff]
                for k in stale:
                    self._buckets.pop(k, None)
                self._last_cleanup = now

            if client_id not in self._buckets:
                # RPM: capacity = rpm, refill = rpm / 60
                rpm_bucket = TokenBucket(capacity=rpm, refill_rate_per_sec=rpm / 60.0)
                # TPM: capacity = tpm, refill = tpm / 60
                tpm_bucket = TokenBucket(capacity=tpm, refill_rate_per_sec=tpm / 60.0)
                self._buckets[client_id] = (rpm_bucket, tpm_bucket, now)
            else:
                rpm_b, tpm_b, _ = self._buckets[client_id]
                self._buckets[client_id] = (rpm_b, tpm_b, now)

            return self._buckets[client_id][0], self._buckets[client_id][1]

    def check_request(
        self,
        client_id: str,
        est_tokens: int = 1,
        custom_rpm: Optional[int] = None,
        custom_tpm: Optional[int] = None,
    ) -> Tuple[bool, float, int, int]:
        """Check if request passes both RPM and TPM buckets.
        
        Returns:
            (allowed: bool, retry_after_s: float, remaining: int, limit: int)
        """
        rpm_bucket, tpm_bucket = self._get_buckets(client_id, custom_rpm, custom_tpm)

        # 1. Check RPM (1 request)
        rpm_ok, rpm_retry, rpm_rem, rpm_lim = rpm_bucket.consume(1.0)
        if not rpm_ok:
            return False, rpm_retry, rpm_rem, rpm_lim

        # 2. Check TPM if estimated tokens specified
        if est_tokens > 0:
            tpm_ok, tpm_retry, _, _ = tpm_bucket.consume(float(est_tokens))
            if not tpm_ok:
                # Thread-safe refund of RPM token if TPM check fails
                rpm_bucket.refund(1.0)
                return False, tpm_retry, 0, rpm_lim

        return True, 0.0, rpm_rem, rpm_lim

    def consume_tokens(
        self, client_id: str, token_count: int, custom_tpm: Optional[int] = None
    ) -> None:
        """Deduct actual executed prompt and completion tokens from the TPM bucket."""
        if token_count <= 0:
            return
        _, tpm_bucket = self._get_buckets(client_id, custom_tpm=custom_tpm)
        tpm_bucket.consume(float(token_count))


class APIKeyAuthenticator:
    """Validates HTTP Bearer tokens and client credentials with constant-time comparison."""

    def __init__(
        self,
        enabled: bool = False,
        api_keys: Optional[List[str]] = None,
        key_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        exempt_localhost: bool = True,
        exempt_paths: Optional[Set[str]] = None,
    ) -> None:
        self.enabled = enabled
        self.valid_keys: Set[str] = set(api_keys or [])
        self.key_configs: Dict[str, Dict[str, Any]] = key_configs or {}
        if self.key_configs:
            self.valid_keys.update(self.key_configs.keys())

        self.exempt_localhost = exempt_localhost
        self.exempt_paths: Set[str] = exempt_paths or {
            "/healthz",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/api/cluster/status",
            "/api/cluster/nodes",
            "/api/cluster/heartbeat",
            "/api/cluster/join",
            "/api/cluster/plan",
        }

    def is_exempt_path(self, path: str) -> bool:
        return path in self.exempt_paths

    def is_localhost(self, client_host: Optional[str]) -> bool:
        if not client_host:
            return False
        return client_host in ("127.0.0.1", "::1", "localhost", "testclient")

    def authenticate_header(
        self,
        auth_header: Optional[str],
        client_host: Optional[str],
        path: str,
        is_forwarded: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate request authentication.
        
        Returns:
            (authenticated: bool, key_or_client_id: Optional[str], error_detail: Optional[str])
        """
        # If auth is disabled, allow all
        if not self.enabled:
            return True, client_host or "anonymous", None

        # Check exempt paths
        if self.is_exempt_path(path):
            return True, client_host or "exempt", None

        # Check localhost exemption (do not exempt if request was forwarded from external proxy)
        if self.exempt_localhost and not is_forwarded and self.is_localhost(client_host):
            return True, "localhost", None

        if not auth_header:
            return False, None, "Missing Authorization header"

        parts = auth_header.strip().split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False, None, "Invalid Authorization header format. Expected 'Bearer <token>'"

        token = parts[1]
        # Constant-time comparison to prevent timing attacks
        matched = False
        for valid_key in self.valid_keys:
            if secrets.compare_digest(token, valid_key):
                matched = True
                break

        if not matched:
            return False, None, "Invalid API key"

        return True, token, None

    def get_key_limits(self, token: str) -> Tuple[Optional[int], Optional[int]]:
        """Return custom (rpm, tpm) limits if configured for key."""
        cfg = self.key_configs.get(token, {})
        return cfg.get("rpm"), cfg.get("tpm")
