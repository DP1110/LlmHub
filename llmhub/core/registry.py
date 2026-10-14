"""ACID SQLite Model Registry for LLMHub.

Stores metadata, paths, quantization levels, parameter configurations,
and usage timestamps for all models under ~/.llmhub/models/.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

class ModelRegistry:
    """Thread-safe SQLite registry for model storage."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_lookup ON models(org, name, quant);")
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
        """Find a model by tag: 'org/name:quant', 'name:quant', or 'name'."""
        spec = model_spec.strip()
        org = None
        name = spec
        quant = None

        if "/" in spec:
            parts = spec.split("/", 1)
            org = parts[0]
            rest = parts[1]
        else:
            rest = spec

        if ":" in rest:
            name, quant = rest.split(":", 1)
            quant = quant.lower()
        else:
            name = rest

        with self._get_connection() as conn:
            if org and quant:
                row = conn.execute(
                    "SELECT * FROM models WHERE org=? AND name=? AND quant=?",
                    (org, name, quant)
                ).fetchone()
            elif org and not quant:
                row = conn.execute(
                    "SELECT * FROM models WHERE org=? AND name=? ORDER BY id DESC LIMIT 1",
                    (org, name)
                ).fetchone()
            elif quant and not org:
                row = conn.execute(
                    "SELECT * FROM models WHERE name=? AND quant=? ORDER BY id DESC LIMIT 1",
                    (name, quant)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM models WHERE name=? ORDER BY id DESC LIMIT 1",
                    (name,)
                ).fetchone()

            if row:
                return self._row_to_record(row)
        return None

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
