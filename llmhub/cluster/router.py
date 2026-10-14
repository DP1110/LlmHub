"""Locality-aware distributed request router and remote dispatch proxy."""

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from llmhub.cluster.manager import ClusterManager
from llmhub.cluster.node import NodeInfo, NodeStatus

logger = logging.getLogger("llmhub.cluster.router")


@dataclass
class RoutingDecision:
    """Outcome of cluster routing resolution."""
    target: str  # "local" or peer node_id
    target_url: str  # API base URL
    reason: str  # "local_resident", "remote_resident", "local_capacity", "remote_capacity", "fallback"
    peer: Optional[NodeInfo] = None

    @property
    def is_local(self) -> bool:
        return self.target == "local"


class ClusterRouter:
    """Resolves whether requests execute locally or dispatch across the cluster."""

    def __init__(self, cluster_manager: ClusterManager) -> None:
        self.cluster = cluster_manager
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def route_model(self, model_tag: str, est_memory_mb: float = 0.0) -> RoutingDecision:
        """Determine optimal execution node for a model inference request.
        
        Priority order:
        1. Local Resident: Model already resident in local memory.
        2. Remote Resident: Model already resident on a healthy remote peer.
        3. Capacity Balancing: Cold model assigned to node with highest headroom.
        """
        local = self.cluster.local_node

        # If cluster is disabled, route locally
        if not self.cluster.enabled:
            return RoutingDecision(
                target="local",
                target_url=local.api_url,
                reason="local_fallback",
            )

        # 1. Local Resident Check: Model loaded locally
        if model_tag in local.resident_models and local.is_available:
            return RoutingDecision(
                target="local",
                target_url=local.api_url,
                reason="local_resident",
            )

        # If no peers connected in cluster, route locally
        if not self.cluster.peers:
            return RoutingDecision(
                target="local",
                target_url=local.api_url,
                reason="local_fallback",
            )

        # 2. Remote Resident Check: Find healthy peers where model is already in RAM
        resident_peers: List[NodeInfo] = []
        for peer in self.cluster.peers.values():
            if peer.is_available and model_tag in peer.resident_models:
                resident_peers.append(peer)

        if resident_peers:
            # Pick peer with the lowest load score (fewest active sessions, lowest latency)
            best_peer = min(resident_peers, key=lambda p: p.load_score(model_tag))
            return RoutingDecision(
                target=best_peer.node_id,
                target_url=best_peer.api_url,
                reason="remote_resident",
                peer=best_peer,
            )

        # 3. Cold Model Dispatch: Neither local nor peer has it loaded
        # Prefer local if free RAM is plentiful (>= 1.5GB or > est_memory_mb)
        if local.free_ram_mb >= max(1500.0, est_memory_mb):
            return RoutingDecision(
                target="local",
                target_url=local.api_url,
                reason="local_capacity",
            )

        # Check if any remote peer has substantially greater RAM headroom
        available_peers = [p for p in self.cluster.peers.values() if p.is_available]
        if available_peers:
            best_capacity_peer = max(available_peers, key=lambda p: p.free_ram_mb)
            # If peer has at least 1.0 GB more free RAM than local node
            if best_capacity_peer.free_ram_mb > (local.free_ram_mb + 1000.0):
                return RoutingDecision(
                    target=best_capacity_peer.node_id,
                    target_url=best_capacity_peer.api_url,
                    reason="remote_capacity",
                    peer=best_capacity_peer,
                )

        # Default back to local node
        return RoutingDecision(
            target="local",
            target_url=local.api_url,
            reason="local_fallback",
        )

    async def forward_chat_completion_stream(
        self,
        target_url: str,
        payload: Dict[str, Any],
        client_request: Request,
        headers: Optional[Dict[str, str]] = None,
    ) -> StreamingResponse:
        """Stream chat completion chunks from remote peer node directly to client."""
        client = self._get_client()
        fwd_headers = {"Content-Type": "application/json"}
        if headers:
            for k, v in headers.items():
                if k.lower() in ("authorization", "x-request-id", "x-cluster-secret"):
                    fwd_headers[k] = v

        url = f"{target_url.rstrip('/')}/v1/chat/completions"

    def _downgrade_peer(self, target_url: str) -> None:
        """Mark peer UNREACHABLE upon communication failure."""
        norm_url = target_url.rstrip("/")
        for peer in self.cluster.peers.values():
            if peer.api_url.rstrip("/") == norm_url:
                peer.status = NodeStatus.UNREACHABLE
                logger.warning("Peer %s marked UNREACHABLE due to connection failure", peer.node_id)
                break

    async def forward_chat_completion_stream(
        self,
        target_url: str,
        payload: Dict[str, Any],
        client_request: Request,
        headers: Optional[Dict[str, str]] = None,
    ) -> StreamingResponse:
        """Stream chat completion chunks from remote peer node directly to client."""
        client = self._get_client()
        fwd_headers = {"Content-Type": "application/json"}
        if headers:
            for k, v in headers.items():
                if k.lower() in ("authorization", "x-request-id", "x-cluster-secret"):
                    fwd_headers[k] = v

        url = f"{target_url.rstrip('/')}/v1/chat/completions"

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with client.stream("POST", url, json=payload, headers=fwd_headers) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        yield f"data: {json.dumps({'error': err_body.decode('utf-8', errors='ignore')})}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                        return

                    async for chunk in resp.aiter_bytes():
                        if await client_request.is_disconnected():
                            logger.info("Client disconnected during remote forwarded stream; closing")
                            break
                        if chunk:
                            yield chunk
            except Exception as e:
                logger.error("Error streaming from remote cluster node %s: %s", target_url, e)
                self._downgrade_peer(target_url)
                yield f"data: {json.dumps({'error': f'Remote cluster dispatch failed: {e}'})}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def forward_chat_completion_json(
        self,
        target_url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Response:
        """Forward non-streaming chat completion to remote peer node."""
        client = self._get_client()
        fwd_headers = {"Content-Type": "application/json"}
        if headers:
            for k, v in headers.items():
                if k.lower() in ("authorization", "x-request-id", "x-cluster-secret"):
                    fwd_headers[k] = v

        url = f"{target_url.rstrip('/')}/v1/chat/completions"
        try:
            resp = await client.post(url, json=payload, headers=fwd_headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except Exception as e:
            logger.error("Error forwarding JSON request to remote node %s: %s", target_url, e)
            self._downgrade_peer(target_url)
            status_code = 504 if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError)) else 502
            err_type = "cluster_gateway_timeout" if status_code == 504 else "cluster_gateway_error"
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "message": f"Cluster forward error to {target_url}: {e}",
                        "type": err_type,
                    }
                },
            )
