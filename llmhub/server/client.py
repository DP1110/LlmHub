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

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_server_running(self) -> bool:
        """Check if local server daemon is reachable."""
        try:
            with httpx.Client(timeout=1.5) as client:
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
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/api/models")
            if resp.status_code != 200:
                raise RuntimeError(f"Error fetching models: HTTP {resp.status_code} - {resp.text}")
            return resp.json().get("models", [])

    def get_ps(self) -> Dict[str, Any]:
        self.check_connection()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/api/ps")
            if resp.status_code != 200:
                raise RuntimeError(f"Error fetching ps: HTTP {resp.status_code} - {resp.text}")
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

        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
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
