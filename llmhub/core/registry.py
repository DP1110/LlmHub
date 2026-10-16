"""ACID SQLite Model Registry for LLMHub.

Stores metadata, paths, quantization levels, parameter configurations,
and usage timestamps for all models under ~/.llmhub/models/.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llmhub.core.aliases import resolve_model_alias

class ModelRecord:
    def __init__(
        self,
        id: int,
        org: str,
        name: str,
        quant: str,
        file_path: str,
        manifest_path: str,
        size_bytes: int,
        sha256: str,
        architecture: str,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_max: int,
        created_at: str,
        last_used_at: str
    ):
        self.id = id
        self.org = org
        self.name = name
        self.quant = quant
        self.file_path = file_path
        self.manifest_path = manifest_path
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.architecture = architecture
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.ctx_max = ctx_max
        self.created_at = created_at
        self.last_used_at = last_used_at

    @property
    def tag(self) -> str:
        return f"{self.org}/{self.name}:{self.quant}"

    @property
    def short_tag(self) -> str:
        return f"{self.name}:{self.quant}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "org": self.org,
            "name": self.name,
            "quant": self.quant,
            "file_path": self.file_path,
            "manifest_path": self.manifest_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "architecture": self.architecture,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "ctx_max": self.ctx_max,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

class ModelRegistry:
    """Thread-safe SQLite registry for model storage."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org TEXT NOT NULL,
                name TEXT NOT NULL,
                quant TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                manifest_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                architecture TEXT NOT NULL,
                n_layers INTEGER NOT NULL,
                n_kv_heads INTEGER NOT NULL,
                head_dim INTEGER NOT NULL,
                ctx_max INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                UNIQUE(org, name, quant)
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_lookup ON models(org COLLATE NOCASE, name COLLATE NOCASE, quant COLLATE NOCASE);")
            conn.commit()

    def register_model(
        self,
        org: str,
        name: str,
        quant: str,
        file_path: Path,
        manifest_path: Path,
        size_bytes: int,
        sha256: str,
        architecture: str,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_max: int
    ) -> ModelRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO models (
                org, name, quant, file_path, manifest_path, size_bytes,
                sha256, architecture, n_layers, n_kv_heads, head_dim,
                ctx_max, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(org, name, quant) DO UPDATE SET
                file_path=excluded.file_path,
                manifest_path=excluded.manifest_path,
                size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,
                architecture=excluded.architecture,
                n_layers=excluded.n_layers,
                n_kv_heads=excluded.n_kv_heads,
                head_dim=excluded.head_dim,
                ctx_max=excluded.ctx_max,
                last_used_at=excluded.last_used_at;
            """, (
                org, name, quant.lower(), str(file_path), str(manifest_path),
                size_bytes, sha256, architecture, n_layers, n_kv_heads,
                head_dim, ctx_max, now, now
            ))
            conn.commit()
            
            row = conn.execute(
                "SELECT * FROM models WHERE org=? AND name=? AND quant=?",
                (org, name, quant.lower())
            ).fetchone()
            return self._row_to_record(row)

    def list_models(self) -> List[ModelRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY org, name, quant").fetchall()
            return [self._row_to_record(row) for row in rows]

    def find_model(self, model_spec: str) -> Optional[ModelRecord]:
        """Find a model by tag: 'org/name:quant', 'name:quant', 'name', or alias."""
        spec = model_spec.strip()
        candidates_specs = [spec]
        resolved = resolve_model_alias(spec)
        if resolved != spec:
            candidates_specs.append(resolved)

        with self._get_connection() as conn:
            for s in candidates_specs:
                org = None
                name = s
                quant = None

                if "/" in s:
                    parts = s.split("/", 1)
                    org = parts[0]
                    rest = parts[1]
                else:
                    rest = s

                if ":" in rest:
                    name, quant = rest.split(":", 1)
                    quant = quant.lower()
                else:
                    name = rest

                if org and quant:
                    row = conn.execute(
                        "SELECT * FROM models WHERE org=? COLLATE NOCASE AND name=? COLLATE NOCASE AND quant=? COLLATE NOCASE",
                        (org, name, quant)
                    ).fetchone()
                elif org and not quant:
                    row = conn.execute(
                        "SELECT * FROM models WHERE org=? COLLATE NOCASE AND name=? COLLATE NOCASE ORDER BY id DESC LIMIT 1",
                        (org, name)
                    ).fetchone()
                elif quant and not org:
                    row = conn.execute(
                        "SELECT * FROM models WHERE name=? COLLATE NOCASE AND quant=? COLLATE NOCASE ORDER BY id DESC LIMIT 1",
                        (name, quant)
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM models WHERE name=? COLLATE NOCASE ORDER BY id DESC LIMIT 1",
                        (name,)
                    ).fetchone()

                if row:
                    return self._row_to_record(row)

            # Fallback 0: If alias resolved to a canonical repo org/name but requested quant
            # is not installed (e.g. user pulled custom quant like q8_0), match any installed quant
            if resolved != spec and "/" in resolved:
                r_org_repo = resolved.split(":", 1)[0]
                if "/" in r_org_repo:
                    r_org, r_name = r_org_repo.split("/", 1)
                    row = conn.execute(
                        "SELECT * FROM models WHERE org=? COLLATE NOCASE AND name=? COLLATE NOCASE ORDER BY id DESC LIMIT 1",
                        (r_org, r_name)
                    ).fetchone()
                    if row:
                        return self._row_to_record(row)

            # Fallback 1: Case-insensitive match across registered tags and names
            clean_spec = spec.lower().replace(":", "-").replace("/", "-").replace("\\", "-").replace("_", "-")
            clean_resolved = resolved.lower().replace(":", "-").replace("/", "-").replace("\\", "-").replace("_", "-")
            all_rows = conn.execute("SELECT * FROM models ORDER BY id DESC").fetchall()
            for r in all_rows:
                tag = f"{r['org']}/{r['name']}:{r['quant']}".lower()
                tag_short = f"{r['name']}:{r['quant']}".lower()
                stem_name = r['name'].lower()
                candidates = [
                    tag,
                    tag_short,
                    stem_name,
                    tag.replace(":", "-").replace("/", "-").replace("\\", "-"),
                    tag_short.replace(":", "-"),
                    tag.replace(":", "-").replace("/", "-").replace("\\", "-").replace("_", "-"),
                    tag_short.replace(":", "-").replace("_", "-"),
                    stem_name.replace(":", "-").replace("_", "-"),
                ]
                if (
                    spec.lower() in candidates
                    or clean_spec in candidates
                    or resolved.lower() in candidates
                    or clean_resolved in candidates
                ):
                    return self._row_to_record(r)

        # Fallback 2: Check if spec points directly to an existing GGUF file
        cand = Path(spec)
        if cand.is_file() and cand.suffix.lower() == ".gguf":
            try:
                from llmhub.core.gguf_parser import GGUFParser
                meta = GGUFParser.parse_file(cand)
                manifest_path = cand.with_suffix(".json")
                if not manifest_path.exists():
                    manifest_path = cand
                return self.register_model(
                    org="local",
                    name=cand.stem,
                    quant="gguf",
                    file_path=cand.resolve(),
                    manifest_path=manifest_path.resolve(),
                    size_bytes=cand.stat().st_size,
                    sha256="",
                    architecture=meta.architecture or "llama",
                    n_layers=meta.block_count or 0,
                    n_kv_heads=meta.head_count_kv or 0,
                    head_dim=meta.head_dim or 0,
                    ctx_max=meta.context_length or 2048,
                )
            except Exception:
                pass

        return None

    def scan_and_register_local_models(self, models_dir: Union[str, Path]) -> List[ModelRecord]:
        """Scan directory for any .gguf files not yet in the registry and register them."""
        p_dir = Path(models_dir)
        if not p_dir.is_dir():
            return []

        from llmhub.core.gguf_parser import GGUFParser

        registered: List[ModelRecord] = []
        try:
            candidates = list(p_dir.rglob("*.gguf"))
        except Exception:
            candidates = []

        seen_paths = set()
        deduped = []
        for c in candidates:
            resolved = str(c.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                deduped.append(c)

        for fpath in deduped:
            if not fpath.is_file() or fpath.suffix.lower() != ".gguf":
                continue

            resolved_str = str(fpath.resolve())
            with self._get_connection() as conn:
                existing = conn.execute("SELECT id FROM models WHERE file_path=?", (resolved_str,)).fetchone()
            if existing:
                continue

            try:
                meta = GGUFParser.parse_file(fpath)
                stem = fpath.stem
                org = "local"
                name = stem
                quant = "gguf"

                # Check if directory layout follows models/<org>/<name>/<quant>/model.gguf
                rel = fpath.relative_to(p_dir).parts
                if len(rel) >= 4:
                    org = rel[0]
                    name = rel[1]
                    quant = rel[2].lower()
                elif len(rel) == 3:
                    org = rel[0]
                    name = rel[1]
                    file_stem = Path(rel[2]).stem.lower()
                    parts = file_stem.replace(":", "-").replace("_", "-").split("-")
                    last_part = parts[-1]
                    if any(q in last_part for q in ["q4", "q5", "q8", "f16", "f32", "q2", "q3", "q6"]):
                        quant = last_part
                    elif any(q in file_stem for q in ["q4", "q5", "q8", "f16", "f32", "q2", "q3", "q6"]):
                        quant = file_stem
                    else:
                        quant = "gguf"
                elif len(rel) == 2:
                    org = "local"
                    name = rel[0]
                    file_stem = Path(rel[1]).stem.lower()
                    parts = file_stem.replace(":", "-").replace("_", "-").split("-")
                    last_part = parts[-1]
                    if any(q in last_part for q in ["q4", "q5", "q8", "f16", "f32", "q2", "q3", "q6"]):
                        quant = last_part
                    else:
                        quant = "gguf"
                elif "-" in stem or "_" in stem or ":" in stem:
                    parts = stem.replace(":", "-").replace("_", "-").split("-")
                    last_part = parts[-1].lower()
                    if any(q in last_part for q in ["q4", "q5", "q8", "f16", "f32", "q2", "q3", "q6"]):
                        quant = last_part
                        name = "-".join(parts[:-1])

                manifest_path = fpath.with_suffix(".json")
                if not manifest_path.exists():
                    manifest_path = fpath

                rec = self.register_model(
                    org=org,
                    name=name,
                    quant=quant,
                    file_path=fpath.resolve(),
                    manifest_path=manifest_path.resolve(),
                    size_bytes=fpath.stat().st_size,
                    sha256="",
                    architecture=meta.architecture or "llama",
                    n_layers=meta.block_count or 0,
                    n_kv_heads=meta.head_count_kv or 0,
                    head_dim=meta.head_dim or 0,
                    ctx_max=meta.context_length or 2048,
                )
                registered.append(rec)
            except Exception:
                continue

        return registered

    def remove_model(self, model_spec: str) -> Optional[ModelRecord]:
        record = self.find_model(model_spec)
        if not record:
            return None

        with self._get_connection() as conn:
            conn.execute("DELETE FROM models WHERE id=?", (record.id,))
            conn.commit()
        return record

    def touch_model(self, model_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("UPDATE models SET last_used_at=? WHERE id=?", (now, model_id))
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            id=row["id"],
            org=row["org"],
            name=row["name"],
            quant=row["quant"],
            file_path=row["file_path"],
            manifest_path=row["manifest_path"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            architecture=row["architecture"],
            n_layers=row["n_layers"],
            n_kv_heads=row["n_kv_heads"],
            head_dim=row["head_dim"],
            ctx_max=row["ctx_max"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )
