"""HTTP Client wrapper for LLMHub CLI.

The CLI must always talk to the server over HTTP, as specified by the system architecture.
"""

import json
from typing import Any, Dict, Generator, List, Optional
import httpx

class ServerUnavailableError(Exception):
    pass

class LLMHubClient:
    """Client for interacting with local LLMHub HTTP daemon."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_server_running(self) -> bool:
        """Check if local server daemon is reachable."""
        try:
            with httpx.Client(timeout=1.5, headers=self._get_headers()) as client:
                resp = client.get(f"{self.base_url}/healthz")
                return resp.status_code == 200
        except Exception:
            return False

    def check_connection(self) -> None:
        if not self.is_server_running():
            raise ServerUnavailableError(
                f"LLMHub server is not responding at {self.base_url}.\n"
                f"Start the server in another terminal using:\n"
                f"  llmhub serve\n"
                f"Or launch it in background mode."
            )

    def list_models(self) -> List[Dict[str, Any]]:
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.get(f"{self.base_url}/api/models")
            if resp.status_code != 200:
                raise RuntimeError(f"Error fetching models: HTTP {resp.status_code} - {resp.text}")
            return resp.json().get("models", [])

    def get_ps(self) -> Dict[str, Any]:
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.get(f"{self.base_url}/api/ps")
            if resp.status_code != 200:
                raise RuntimeError(f"Error fetching ps: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def get_metrics(self) -> str:
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.get(f"{self.base_url}/metrics")
            if resp.status_code != 200:
                raise RuntimeError(f"Error fetching metrics: HTTP {resp.status_code} - {resp.text}")
            return resp.text

    def pin_model(self, model: str) -> Dict[str, Any]:
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/models/{model}/pin")
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to pin model: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def unpin_model(self, model: str) -> Dict[str, Any]:
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/models/{model}/unpin")
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to unpin model: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def load_model(
        self,
        model: str,
        n_ctx: Optional[int] = None,
        pinned: bool = False,
        keep_alive: Optional[float] = None
    ) -> Dict[str, Any]:
        self.check_connection()
        params = {}
        if n_ctx is not None:
            params["n_ctx"] = n_ctx
        if pinned:
            params["pinned"] = "true"
        if keep_alive is not None:
            params["keep_alive"] = keep_alive

        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/models/{model}/load", params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to load model: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def unload_model(
        self,
        model: str,
        force: bool = False,
        drain: bool = False,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        self.check_connection()
        params: Dict[str, Any] = {}
        if force:
            params["force"] = "true"
        if drain:
            params["drain"] = "true"
        if timeout:
            params["timeout"] = str(timeout)

        with httpx.Client(timeout=self.timeout + timeout, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/models/{model}/unload", params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to unload model: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated chat tokens over HTTP SSE from /v1/chat/completions.

        Yields chunks with schema:
          {"token": str, "done": bool, "finish_reason": Optional[str], "usage": Dict[str, Any]}
        """
        self.check_connection()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stop": stop
        }

        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0), headers=self._get_headers()) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Inference failed with HTTP {response.status_code}: {body}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            usage = chunk.get("usage", {})
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                finish_reason = choices[0].get("finish_reason")
                                is_done = finish_reason is not None
                                yield {
                                    "token": content,
                                    "done": is_done,
                                    "finish_reason": finish_reason,
                                    "usage": usage
                                }
                            elif usage:
                                yield {
                                    "token": "",
                                    "done": True,
                                    "finish_reason": "stop",
                                    "usage": usage
                                }
                        except json.JSONDecodeError:
                            continue

    def get_cluster_status(self) -> Dict[str, Any]:
        """Fetch cluster status, connected nodes, and resident models."""
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.get(f"{self.base_url}/api/cluster/status")
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch cluster status: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Fetch list of active cluster nodes."""
        self.check_connection()
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.get(f"{self.base_url}/api/cluster/nodes")
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch cluster nodes: HTTP {resp.status_code} - {resp.text}")
            return resp.json().get("nodes", [])

    def join_cluster(self, peer_url: str) -> Dict[str, Any]:
        """Join an existing cluster by connecting to a peer node."""
        self.check_connection()
        with httpx.Client(timeout=self.timeout + 2.0, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/cluster/join", json={"peer_url": peer_url})
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to join cluster via {peer_url}: HTTP {resp.status_code} - {resp.text}")
            return resp.json()

    def get_cluster_pipeline_plan(
        self,
        model: str,
        total_layers: int = 32,
        hidden_dim: int = 4096,
        weights_mb: float = 4000.0,
    ) -> Dict[str, Any]:
        """Calculate distributed pipeline layer partitioning across cluster nodes."""
        self.check_connection()
        payload = {
            "model": model,
            "total_layers": total_layers,
            "hidden_dim": hidden_dim,
            "weights_mb": weights_mb,
        }
        with httpx.Client(timeout=self.timeout, headers=self._get_headers()) as client:
            resp = client.post(f"{self.base_url}/api/cluster/plan", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to compute pipeline plan: HTTP {resp.status_code} - {resp.text}")
            return resp.json()
