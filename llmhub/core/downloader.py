"""Hugging Face Model Downloader with HTTP Range resume and incremental SHA-256 verification.

Downloads GGUF files directly to ~/.llmhub/models/<org>/<name>/<quant>/model.gguf
without intermediate cloud or vendor proxies.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import httpx

from llmhub.config import LLMHubConfig
from llmhub.core.gguf_parser import GGUFParser
from llmhub.core.registry import ModelRegistry, ModelRecord

class DownloadError(Exception):
    pass

class HFDownloader:
    """Resumable streaming downloader for Hugging Face GGUF models."""

    def __init__(self, config: Optional[LLMHubConfig] = None):
        self.config = config or LLMHubConfig()
        self.registry = ModelRegistry(self.config.db_path)

    @staticmethod
    def parse_spec(spec: str) -> Tuple[str, str, str, Optional[str]]:
        """Parse model spec into (org, name, quant, filename).
        
        Formats supported:
        1. "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m"
        2. "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
        3. "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
        """
        spec = spec.strip()
        if "/" not in spec:
            raise DownloadError(f"Model spec must include repository org/name, e.g. 'Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m'. Got: {spec}")

        org_repo, quant_or_file = spec.split(":", 1) if ":" in spec else (spec, "q4_k_m")
        org, repo = org_repo.split("/", 1)
        
        quant_or_file = quant_or_file.strip()
        if quant_or_file.lower().endswith(".gguf"):
            filename = quant_or_file
            # extract quant from filename
            base = quant_or_file[:-5]
            extracted = base.split("-")[-1] if "-" in base else "default"
            quant = extracted.lower()
        else:
            quant = quant_or_file.lower()
            filename = None

        return org, repo, quant, filename

    def resolve_hf_download_url(self, org: str, repo: str, quant: str, filename: Optional[str] = None) -> Tuple[str, str, Optional[str], Optional[int]]:
        """Resolve the exact Hugging Face download URL, filename, expected SHA-256, and size."""
        api_url = f"https://huggingface.co/api/models/{org}/{repo}"
        expected_sha: Optional[str] = None
        expected_size: Optional[int] = None

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(api_url)
                if resp.status_code != 200:
                    if filename:
                        url = f"https://huggingface.co/{org}/{repo}/resolve/main/{filename}"
                        return url, filename, None, None
                    raise DownloadError(f"Failed to query Hugging Face repo {org}/{repo}: HTTP {resp.status_code}")
                
                data = resp.json()
                siblings = data.get("siblings", [])

                if filename:
                    for s in siblings:
                        if s.get("rfilename") == filename:
                            lfs = s.get("lfs", {})
                            expected_sha = lfs.get("oid")
                            expected_size = lfs.get("size")
                            break
                    url = f"https://huggingface.co/{org}/{repo}/resolve/main/{filename}"
                    return url, filename, expected_sha, expected_size

                # Match filenames containing the quant
                matched = []
                for s in siblings:
                    rfilename = s.get("rfilename", "")
                    if rfilename.lower().endswith(".gguf"):
                        if quant.lower() in rfilename.lower():
                            matched.append(s)

                if not matched:
                    # Fallback: look for any .gguf files
                    all_ggufs = [s.get("rfilename", "") for s in siblings if s.get("rfilename", "").lower().endswith(".gguf")]
                    if all_ggufs:
                        raise DownloadError(
                            f"No GGUF file matched quant '{quant}' in {org}/{repo}. Available files:\n  - " +
                            "\n  - ".join(all_ggufs[:15])
                        )
                    raise DownloadError(f"No GGUF files found in Hugging Face repository {org}/{repo}.")

                # Pick the best match (shortest filename)
                selected_item = min(matched, key=lambda item: len(item.get("rfilename", "")))
                selected_filename = selected_item.get("rfilename", "")
                lfs = selected_item.get("lfs", {})
                expected_sha = lfs.get("oid")
                expected_size = lfs.get("size")

                url = f"https://huggingface.co/{org}/{repo}/resolve/main/{selected_filename}"
                return url, selected_filename, expected_sha, expected_size
        except Exception as e:
            if isinstance(e, DownloadError):
                raise
            if filename:
                url = f"https://huggingface.co/{org}/{repo}/resolve/main/{filename}"
                return url, filename, None, None
            raise DownloadError(f"Error querying Hugging Face API: {e}")

    def download(
        self,
        spec: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> ModelRecord:
        """Download model GGUF file with HTTP Range resume and streaming SHA-256."""
        org, repo, quant, filename = self.parse_spec(spec)
        url, final_filename, expected_sha, expected_size = self.resolve_hf_download_url(org, repo, quant, filename)
        
        target_dir = self.config.get_model_dir(org, repo, quant)
        model_path = target_dir / "model.gguf"
        tmp_path = target_dir / "model.gguf.tmp"
        manifest_path = target_dir / "manifest.json"

        # Check if already fully downloaded
        if model_path.exists():
            existing_record = self.registry.find_model(f"{org}/{repo}:{quant}")
            if existing_record:
                sys.stdout.write(f"Model {org}/{repo}:{quant} is already downloaded and registered.\n")
                return existing_record

        # Check resume offset
        existing_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        sys.stdout.write(f"Connecting to: {url}\n")
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code not in (200, 206):
                    body = response.read().decode("utf-8", errors="replace")
                    raise DownloadError(f"Download failed with HTTP {response.status_code}: {body}")

                is_resume = response.status_code == 206
                total_content_length = int(response.headers.get("Content-Length", 0))
                total_bytes = (existing_bytes + total_content_length) if is_resume else total_content_length
                if expected_size and total_bytes == 0:
                    total_bytes = expected_size

                mode = "ab" if is_resume else "wb"
                if not is_resume and existing_bytes > 0:
                    existing_bytes = 0

                downloaded = existing_bytes
                start_time = time.time()
                last_update = start_time

                with open(tmp_path, mode) as f:
                    for chunk in response.iter_bytes(chunk_size=1024 * 512):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_update >= 0.2:
                            elapsed = now - start_time
                            speed = (downloaded - existing_bytes) / elapsed if elapsed > 0 else 0
                            if progress_callback:
                                progress_callback(downloaded, total_bytes, speed)
                            else:
                                percent = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                                sys.stdout.write(
                                    f"\rDownloading: {downloaded / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB "
                                    f"({percent:.1f}%) @ {speed / (1024*1024):.2f} MB/s   "
                                )
                                sys.stdout.flush()
                            last_update = now

        # Verify completed size
        actual_size = tmp_path.stat().st_size
        if total_bytes > 0 and actual_size < total_bytes:
            raise DownloadError(f"Download terminated prematurely: {actual_size} bytes received of {total_bytes} expected.")

        sys.stdout.write("\nDownload complete. Calculating SHA-256 and validating GGUF headers...\n")
        
        # Calculate SHA256 of downloaded file
        sha256_hash = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for b_chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(b_chunk)
        digest = sha256_hash.hexdigest()

        if expected_sha and digest.lower() != expected_sha.lower():
            raise DownloadError(
                f"SHA-256 integrity check failed!\n"
                f"Expected: {expected_sha.lower()}\n"
                f"Actual:   {digest.lower()}"
            )

        # Cross-platform atomic file replacement
        tmp_path.replace(model_path)

        # Parse GGUF metadata
        metadata = GGUFParser.parse_file(model_path)

        # Write manifest.json
        manifest = {
            "org": org,
            "repo": repo,
            "quant": quant,
            "filename": final_filename,
            "url": url,
            "sha256": digest,
            "size_bytes": model_path.stat().st_size,
            "architecture": metadata.architecture,
            "model_name": metadata.model_name,
            "block_count": metadata.block_count,
            "head_count": metadata.head_count,
            "head_count_kv": metadata.head_count_kv,
            "head_dim": metadata.head_dim,
            "context_length": metadata.context_length,
            "embedding_length": metadata.embedding_length,
            "has_chat_template": metadata.chat_template is not None,
            "chat_template": metadata.chat_template,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Register in SQLite DB
        record = self.registry.register_model(
            org=org,
            name=repo,
            quant=quant,
            file_path=model_path,
            manifest_path=manifest_path,
            size_bytes=model_path.stat().st_size,
            sha256=digest,
            architecture=metadata.architecture,
            n_layers=metadata.block_count,
            n_kv_heads=metadata.head_count_kv,
            head_dim=metadata.head_dim,
            ctx_max=metadata.context_length,
        )

        sys.stdout.write(f"Model successfully saved and registered as '{record.tag}'.\n")
        return record
