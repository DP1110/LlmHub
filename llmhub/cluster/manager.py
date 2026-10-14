"""Cluster Manager coordinating peer state, heartbeats, and cluster membership."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import httpx

from llmhub.cluster.beacon import DiscoveryBeacon
from llmhub.cluster.node import NodeInfo, NodeRole, NodeStatus

logger = logging.getLogger("llmhub.cluster.manager")


class ClusterManager:
    """Coordinates distributed cluster membership, peer heartbeats, and cluster state."""

    def __init__(
        self,
        local_node: NodeInfo,
        enabled: bool = False,
        static_peers: Optional[List[str]] = None,
        heartbeat_interval_s: float = 2.0,
        node_timeout_s: float = 6.0,
        dead_timeout_s: float = 30.0,
        beacon_enabled: bool = True,
        beacon_port: int = 11435,
        cluster_secret: Optional[str] = None,
    ) -> None:
        self.local_node = local_node
        self.enabled = enabled
        self.static_peers: List[str] = static_peers or []
        self.heartbeat_interval_s = heartbeat_interval_s
        self.node_timeout_s = node_timeout_s
        self.dead_timeout_s = dead_timeout_s
        self.beacon_enabled = beacon_enabled
        self.beacon_port = beacon_port
        self.cluster_secret = cluster_secret

        self.peers: Dict[str, NodeInfo] = {}
        self.lock = asyncio.Lock()
        self._running: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reaper_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._background_tasks: set[asyncio.Task] = set()

        # Beacon component
        self.beacon: Optional[DiscoveryBeacon] = None
        if self.enabled and self.beacon_enabled:
            self.beacon = DiscoveryBeacon(
                node_id=self.local_node.node_id,
                name=self.local_node.name,
                api_url=self.local_node.api_url,
                beacon_port=self.beacon_port,
                cluster_secret=self.cluster_secret,
                on_peer_discovered=self.on_beacon_discovered,
            )

        # External state callback (e.g. from ModelManager / ServerState)
        self._state_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def set_state_provider(self, fn: Callable[[], Dict[str, Any]]) -> None:
        """Register provider returning current free RAM, resident models, and active sessions."""
        self._state_provider = fn
        if self.beacon:
            self.beacon.set_dynamic_state_callback(fn)

    def _sync_local_node_state(self) -> None:
        """Update local node's resident models and dynamic metrics from provider."""
        if not self._state_provider:
            return
        try:
            st = self._state_provider()
            self.local_node.free_ram_mb = st.get("free_ram_mb", self.local_node.free_ram_mb)
            self.local_node.total_ram_mb = st.get("total_ram_mb", self.local_node.total_ram_mb)
            self.local_node.active_sessions = st.get("active_sessions", self.local_node.active_sessions)
            self.local_node.resident_models = st.get("resident_models", self.local_node.resident_models)
            self.local_node.last_heartbeat = time.time()
        except Exception as e:
            logger.debug("Failed to sync local node dynamic state: %s", e)

    def on_beacon_discovered(self, packet: Dict[str, Any], sender_ip: str) -> None:
        """Handle incoming beacon announcement packet from LAN."""
        node_id = packet.get("node_id")
        if not node_id or node_id == self.local_node.node_id:
            return

        api_url = packet.get("api_url") or f"http://{sender_ip}:11434"
        task = asyncio.create_task(self._process_discovered_node(node_id, api_url, packet))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_discovered_node(
        self, node_id: str, api_url: str, packet: Dict[str, Any]
    ) -> None:
        async with self.lock:
            existing = self.peers.get(node_id)
            if existing:
                existing.update_heartbeat(
                    free_ram_mb=packet.get("free_ram_mb"),
                    active_sessions=packet.get("active_sessions"),
                    resident_models=packet.get("resident_models"),
                    status=NodeStatus.HEALTHY,
                )
                existing.api_url = api_url
            else:
                from urllib.parse import urlsplit
                parsed = urlsplit(api_url if "://" in api_url else f"http://{api_url}")
                host = parsed.hostname or sender_ip if 'sender_ip' in locals() else (parsed.hostname or "127.0.0.1")
                port = parsed.port or 11434
                new_peer = NodeInfo(
                    node_id=node_id,
                    name=packet.get("name", node_id),
                    host=host,
                    port=port,
                    api_url=api_url,
                    role=NodeRole(packet.get("role", "hybrid")),
                    status=NodeStatus.HEALTHY,
                    free_ram_mb=packet.get("free_ram_mb", 0.0),
                    total_ram_mb=packet.get("total_ram_mb", 0.0),
                    active_sessions=packet.get("active_sessions", 0),
                    resident_models=packet.get("resident_models", {}),
                    is_termux=packet.get("is_termux", False),
                )
                self.peers[node_id] = new_peer
                logger.info("Discovered new cluster peer: %s (%s)", node_id, api_url)

    async def register_or_update_peer(self, peer: NodeInfo) -> None:
        """Explicitly add or update a peer in the registry."""
        if peer.node_id == self.local_node.node_id:
            return
        async with self.lock:
            self.peers[peer.node_id] = peer

    async def handle_heartbeat_received(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming HTTP heartbeat from a peer and return local node state."""
        self._sync_local_node_state()

        sender_id = data.get("node_id")
        if sender_id and sender_id != self.local_node.node_id:
            async with self.lock:
                peer = self.peers.get(sender_id)
                if peer:
                    peer.update_heartbeat(
                        free_ram_mb=data.get("free_ram_mb"),
                        active_sessions=data.get("active_sessions"),
                        resident_models=data.get("resident_models"),
                        status=NodeStatus.HEALTHY,
                    )
                else:
                    new_peer = NodeInfo.from_dict(data)
                    new_peer.status = NodeStatus.HEALTHY
                    new_peer.last_heartbeat = time.time()
                    self.peers[sender_id] = new_peer
                    logger.info("Registered cluster peer via heartbeat: %s", sender_id)

        return self.local_node.to_dict()

    async def join_peer(self, peer_url: str) -> Tuple[bool, str]:
        """Connect to an existing cluster node, ping it, and sync cluster state."""
        peer_url = peer_url.rstrip("/")
        if not peer_url.startswith("http://") and not peer_url.startswith("https://"):
            peer_url = f"http://{peer_url}"

        self._sync_local_node_state()
        headers = {}
        if self.cluster_secret:
            headers["X-Cluster-Secret"] = self.cluster_secret

        client_to_close = None
        if self._http_client and not self._http_client.is_closed:
            client = self._http_client
        else:
            client = httpx.AsyncClient(timeout=5.0)
            client_to_close = client

        start_t = time.perf_counter()
        try:
            # 1. Send our node info via heartbeat
            resp = await client.post(
                f"{peer_url}/api/cluster/heartbeat",
                json=self.local_node.to_dict(),
                headers=headers,
            )
            latency = (time.perf_counter() - start_t) * 1000.0

            if resp.status_code != 200:
                return False, f"Peer rejected heartbeat: HTTP {resp.status_code}"

            peer_data = resp.json()
            remote_node = NodeInfo.from_dict(peer_data)
            remote_node.latency_ms = latency
            remote_node.api_url = peer_url
            remote_node.status = NodeStatus.HEALTHY

            await self.register_or_update_peer(remote_node)

            # 2. Fetch full peer table from the peer
            try:
                nodes_resp = await client.get(f"{peer_url}/api/cluster/nodes", headers=headers)
                if nodes_resp.status_code == 200:
                    for n_data in nodes_resp.json().get("nodes", []):
                        if n_data.get("node_id") != self.local_node.node_id:
                            n_info = NodeInfo.from_dict(n_data)
                            await self.register_or_update_peer(n_info)
            except Exception as e:
                logger.debug("Failed to fetch extended nodes list from %s: %s", peer_url, e)

            return True, f"Successfully joined cluster via {peer_url} (latency: {latency:.1f}ms)"
        except Exception as e:
            return False, f"Failed to connect to peer at {peer_url}: {e}"
        finally:
            if client_to_close:
                await client_to_close.aclose()

    async def ping_peer(self, node_id_or_url: str) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """Measure roundtrip latency to a peer and fetch its latest health status."""
        target_url = node_id_or_url
        async with self.lock:
            if node_id_or_url in self.peers:
                target_url = self.peers[node_id_or_url].api_url

        target_url = target_url.rstrip("/")
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = f"http://{target_url}"

        client_to_close = None
        if self._http_client and not self._http_client.is_closed:
            client = self._http_client
        else:
            client = httpx.AsyncClient(timeout=4.0)
            client_to_close = client

        start_t = time.perf_counter()
        try:
            resp = await client.get(f"{target_url}/healthz")
            latency = (time.perf_counter() - start_t) * 1000.0
            if resp.status_code == 200:
                async with self.lock:
                    for peer in self.peers.values():
                        if peer.api_url == target_url:
                            peer.latency_ms = latency
                            peer.last_heartbeat = time.time()
                            peer.status = NodeStatus.HEALTHY
                return True, latency, resp.json()
            return False, latency, None
        except Exception:
            return False, 0.0, None
        finally:
            if client_to_close:
                await client_to_close.aclose()

    async def _heartbeat_loop(self) -> None:
        """Periodically broadcast heartbeat to all known peers."""
        while self._running:
            self._sync_local_node_state()
            active_peers = list(self.peers.values())

            client = self._http_client
            if client:
                headers = {}
                if self.cluster_secret:
                    headers["X-Cluster-Secret"] = self.cluster_secret

                payload = self.local_node.to_dict()

                for peer in active_peers:
                    if peer.status == NodeStatus.OFFLINE:
                        continue
                    try:
                        start_t = time.perf_counter()
                        resp = await client.post(
                            f"{peer.api_url}/api/cluster/heartbeat",
                            json=payload,
                            headers=headers,
                        )
                        dur_ms = (time.perf_counter() - start_t) * 1000.0
                        if resp.status_code == 200:
                            peer.last_heartbeat = time.time()
                            peer.latency_ms = dur_ms
                            if peer.status == NodeStatus.UNREACHABLE:
                                peer.status = NodeStatus.HEALTHY
                            # Update peer state from response
                            try:
                                ret_data = resp.json()
                                peer.free_ram_mb = ret_data.get("free_ram_mb", peer.free_ram_mb)
                                peer.active_sessions = ret_data.get("active_sessions", peer.active_sessions)
                                peer.resident_models = ret_data.get("resident_models", peer.resident_models)
                            except Exception:
                                pass
                        else:
                            logger.debug("Heartbeat to %s returned HTTP %s", peer.node_id, resp.status_code)
                    except Exception as e:
                        logger.debug("Heartbeat error to peer %s (%s): %s", peer.node_id, peer.api_url, e)

            try:
                await asyncio.sleep(self.heartbeat_interval_s)
            except asyncio.CancelledError:
                break

    async def _reap_peers_loop(self) -> None:
        """Periodically evaluate peer freshness and transition stale peers."""
        while self._running:
            now = time.time()
            async with self.lock:
                to_delete: List[str] = []
                for node_id, peer in self.peers.items():
                    elapsed = now - peer.last_heartbeat
                    if elapsed > self.dead_timeout_s:
                        to_delete.append(node_id)
                        logger.info("Reaping dead cluster peer %s (unresponsive for %.1fs)", node_id, elapsed)
                    elif elapsed > self.node_timeout_s:
                        if peer.status != NodeStatus.UNREACHABLE:
                            peer.status = NodeStatus.UNREACHABLE
                            logger.warning("Cluster peer %s timed out; marked UNREACHABLE", node_id)

                for nid in to_delete:
                    self.peers.pop(nid, None)

            try:
                await asyncio.sleep(self.heartbeat_interval_s)
            except asyncio.CancelledError:
                break

    def get_cluster_status(self) -> Dict[str, Any]:
        """Aggregate health, capacity, and model distribution across all cluster nodes."""
        self._sync_local_node_state()
        all_nodes = [self.local_node] + list(self.peers.values())
        healthy_nodes = [n for n in all_nodes if n.status == NodeStatus.HEALTHY]

        # Aggregate resident models across cluster
        cluster_models: Dict[str, Dict[str, Any]] = {}
        for node in healthy_nodes:
            for tag, meta in node.resident_models.items():
                if tag not in cluster_models:
                    cluster_models[tag] = {
                        "tag": tag,
                        "resident_nodes": [node.node_id],
                        "total_active_sessions": meta.get("active_sessions", 0),
                    }
                else:
                    cluster_models[tag]["resident_nodes"].append(node.node_id)
                    cluster_models[tag]["total_active_sessions"] += meta.get("active_sessions", 0)

        total_ram = sum(n.total_ram_mb for n in healthy_nodes)
        free_ram = sum(n.free_ram_mb for n in healthy_nodes)

        return {
            "cluster_enabled": self.enabled,
            "local_node_id": self.local_node.node_id,
            "total_nodes": len(all_nodes),
            "healthy_nodes_count": len(healthy_nodes),
            "total_cluster_ram_mb": round(total_ram, 1),
            "free_cluster_ram_mb": round(free_ram, 1),
            "resident_models": cluster_models,
            "nodes": [n.to_dict() for n in all_nodes],
        }

    async def start(self) -> None:
        """Start cluster networking, discovery beacon, and background loops."""
        if not self.enabled or self._running:
            return
        self._running = True
        self._http_client = httpx.AsyncClient(timeout=4.0)

        # 1. Start discovery beacon
        if self.beacon:
            await self.beacon.start()

        # 2. Connect to static peers from configuration
        for static_peer in self.static_peers:
            t = asyncio.create_task(self.join_peer(static_peer))
            self._background_tasks.add(t)
            t.add_done_callback(self._background_tasks.discard)

        # 3. Start heartbeat and reaper background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._reaper_task = asyncio.create_task(self._reap_peers_loop())

        logger.info(
            "ClusterManager started for node %s (%s). Peers: %s",
            self.local_node.node_id,
            self.local_node.api_url,
            len(self.static_peers),
        )

    async def stop(self) -> None:
        """Gracefully stop cluster operations and notify peers."""
        if not self._running:
            return
        self._running = False

        if self.beacon:
            await self.beacon.stop()

        for task in (self._heartbeat_task, self._reaper_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._heartbeat_task = None
        self._reaper_task = None

        # Cancel and await tracked background tasks
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        self._background_tasks.clear()

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("ClusterManager stopped")
