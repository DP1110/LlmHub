"""High-performance LRU Prefix and Prompt Cache for LLMHub.

Avoids redundant prompt evaluation by storing token sequence prefixes
and tracking reusable KV cache segments across requests.
"""

import array
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

@dataclass
class PrefixCacheEntry:
    """Represents a cached prompt prefix."""
    cache_key: str
    model_tag: str
    tokens: Tuple[int, ...]
    n_tokens: int
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 1
    custom_data: Optional[Any] = None

class PrefixCache:
    """Radix/LRU prefix cache for sharing prompt tokens across sessions."""

    def __init__(self, max_cached_tokens: int = 32768, min_prefix_tokens: int = 8):
        self.max_cached_tokens = max_cached_tokens
        self.min_prefix_tokens = min_prefix_tokens
        # O(1) LRU storage using OrderedDict: most recently used at the end
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self._total_tokens: int = 0
        self._known_lengths: Set[int] = set()

        # Telemetry
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.tokens_saved: int = 0

    @staticmethod
    def compute_hash(model_tag: str, tokens: Tuple[int, ...]) -> str:
        """Generate deterministic cryptographic hash for a token sequence prefix using C-speed array bytes."""
        h = hashlib.sha256()
        h.update(f"{len(model_tag)}:{model_tag}".encode("utf-8"))
        try:
            h.update(array.array("i", tokens).tobytes())
        except Exception:
            for t in tokens:
                h.update(int(t).to_bytes(4, byteorder="little", signed=True))
        return h.hexdigest()[:32]

    def find_longest_prefix(
        self,
        model_tag: str,
        prompt_tokens: List[int]
    ) -> Tuple[int, Optional[PrefixCacheEntry]]:
        """Search cache for longest matching prefix of prompt_tokens.
        
        Returns:
            (matched_token_count, cached_entry_or_None)
        """
        self.total_queries += 1
        n = len(prompt_tokens)
        if n < self.min_prefix_tokens:
            self.cache_misses += 1
            return 0, None

        # Build candidate check lengths from registered cache lengths plus boundaries
        candidate_lengths = sorted(
            list({
                n,
                *(l for l in self._known_lengths if self.min_prefix_tokens <= l <= n),
                *(l for l in range(64, n + 1, 64) if self.min_prefix_tokens <= l <= n),
                *(l for l in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if self.min_prefix_tokens <= l <= n)
            }),
            reverse=True
        )

        for l in candidate_lengths:
            prefix_tuple = tuple(prompt_tokens[:l])
            k = self.compute_hash(model_tag, prefix_tuple)
            entry = self._entries.get(k)
            if entry is not None and entry.tokens == prefix_tuple:
                # O(1) LRU promotion to end
                self._entries.move_to_end(k)
                entry.last_accessed = time.monotonic()
                entry.access_count += 1
                self.cache_hits += 1
                self.tokens_saved += entry.n_tokens
                return entry.n_tokens, entry

        self.cache_misses += 1
        return 0, None

    def _insert_single_entry(
        self,
        model_tag: str,
        tokens: List[int],
        custom_data: Optional[Any] = None
    ) -> Optional[PrefixCacheEntry]:
        """Insert or refresh a single prefix entry."""
        n = len(tokens)
        if n < self.min_prefix_tokens:
            return None

        prefix_tuple = tuple(tokens)
        key = self.compute_hash(model_tag, prefix_tuple)

        existing = self._entries.get(key)
        if existing:
            self._entries.move_to_end(key)
            existing.last_accessed = time.monotonic()
            existing.access_count += 1
            if custom_data is not None:
                existing.custom_data = custom_data
            return existing

        # Evict LRU entries if capacity exceeded
        while self._total_tokens + n > self.max_cached_tokens and self._entries:
            self._evict_oldest()

        entry = PrefixCacheEntry(
            cache_key=key,
            model_tag=model_tag,
            tokens=prefix_tuple,
            n_tokens=n,
            created_at=time.monotonic(),
            last_accessed=time.monotonic(),
            custom_data=custom_data
        )
        self._entries[key] = entry
        self._total_tokens += n
        self._known_lengths.add(n)
        return entry

    def insert_prefix(
        self,
        model_tag: str,
        tokens: List[int],
        custom_data: Optional[Any] = None
    ) -> Optional[PrefixCacheEntry]:
        """Insert or refresh a prefix entry in the cache with sub-prefix checkpoints."""
        n = len(tokens)
        if n < self.min_prefix_tokens:
            return None

        # WN-4: Guard against single prompt larger than max_cached_tokens
        if n > self.max_cached_tokens:
            tokens = tokens[:self.max_cached_tokens]
            n = len(tokens)

        # CR-4: Register sub-prefix checkpoints (every 16/64 tokens) to allow partial prefix matching
        checkpoints = sorted(list({
            *(cp for cp in range(64, n, 64) if cp >= self.min_prefix_tokens),
            *(cp for cp in (16, 32, 64, 128, 256, 512, 1024) if self.min_prefix_tokens <= cp < n)
        }))
        for cp in checkpoints:
            self._insert_single_entry(model_tag, tokens[:cp], custom_data)

        return self._insert_single_entry(model_tag, tokens, custom_data)

    def _evict_oldest(self) -> None:
        """Evict least-recently-accessed entry in O(1) time."""
        if not self._entries:
            return
        oldest_key, removed = self._entries.popitem(last=False)
        self._total_tokens = max(0, self._total_tokens - removed.n_tokens)
        # Recalculate known lengths if needed
        self._known_lengths = {e.n_tokens for e in self._entries.values()}

    def clear(self) -> None:
        """Clear all cached prefixes."""
        self._entries.clear()
        self._total_tokens = 0
        self._known_lengths.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for prefix caching."""
        hit_rate = (self.cache_hits / self.total_queries) * 100.0 if self.total_queries > 0 else 0.0
        return {
            "cached_entries": len(self._entries),
            "cached_tokens": self._total_tokens,
            "max_cached_tokens": self.max_cached_tokens,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(hit_rate, 2),
            "tokens_saved": self.tokens_saved,
        }
