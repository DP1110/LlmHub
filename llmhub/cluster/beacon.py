"""Zero-cloud LAN UDP Discovery Beacon for automatic cluster peer detection."""

import asyncio
import json
import logging
import socket
import sys
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("llmhub.cluster.beacon")

MAGIC_BEACON = "LLMHUB_BEACON"
BEACON_VERSION = "1.0.0"
DEFAULT_BEACON_PORT = 11435


class DiscoveryBeacon:
    """Manages UDP broadcast announcements and incoming discovery packet listener."""

    def __init__(
        self,
        node_id: str,
        name: str,
        api_url: str,
        beacon_port: int = DEFAULT_BEACON_PORT,
        broadcast_interval_s: float = 3.0,
        cluster_secret: Optional[str] = None,
        on_peer_discovered: Optional[Callable[[Dict[str, Any], str], None]] = None,
    ) -> None:
        self.node_id = node_id
        self.name = name
        self.api_url = api_url
        self.beacon_port = beacon_port
        self.broadcast_interval_s = broadcast_interval_s
        self.cluster_secret = cluster_secret
        self.on_peer_discovered = on_peer_discovered

        self._running: bool = False
        self._broadcast_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._send_sock: Optional[socket.socket] = None
        self._recv_sock: Optional[socket.socket] = None

        # Dynamic state getters
        self._get_dynamic_state: Optional[Callable[[], Dict[str, Any]]] = None

    def set_dynamic_state_callback(self, fn: Callable[[], Dict[str, Any]]) -> None:
        """Register callback providing dynamic free RAM, resident models, and active sessions."""
        self._get_dynamic_state = fn

    def _create_send_socket(self) -> Optional[socket.socket]:
        """Create UDP broadcast sender socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # On Windows, bind explicitly to port 0 to ensure sock_sendto succeeds with ProactorEventLoop
            try:
                sock.bind(("", 0))
            except Exception:
                pass
            sock.setblocking(False)
            return sock
        except Exception as e:
            logger.warning("Could not create UDP broadcast sender socket: %s", e)
            return None

    def _create_recv_socket(self) -> Optional[socket.socket]:
        """Create UDP discovery listener socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            if sys.platform != "win32":
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except Exception:
                    pass
            if hasattr(socket, "SO_REUSEPORT") and sys.platform not in ("win32", "cygwin"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except Exception:
                    pass
            sock.bind(("", self.beacon_port))
            sock.setblocking(False)
            return sock
        except Exception as e:
            logger.warning("Could not bind UDP discovery receiver on port %s: %s", self.beacon_port, e)
            return None

    def build_packet(self) -> bytes:
        """Construct JSON announcement payload."""
        payload: Dict[str, Any] = {
            "magic": MAGIC_BEACON,
            "version": BEACON_VERSION,
            "node_id": self.node_id,
            "name": self.name,
            "api_url": self.api_url,
            "timestamp": time.time(),
        }
        if self.cluster_secret:
            payload["secret"] = self.cluster_secret

        if self._get_dynamic_state:
            try:
                dyn = self._get_dynamic_state()
                payload.update(dyn)
            except Exception as e:
                logger.debug("Failed to get dynamic state for beacon: %s", e)

        return json.dumps(payload).encode("utf-8")

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast presence on the local subnet."""
        broadcast_addr = ("255.255.255.255", self.beacon_port)
        loop = asyncio.get_running_loop()

        while self._running:
            if self._send_sock:
                try:
                    packet = self.build_packet()
                    await loop.sock_sendto(self._send_sock, packet, broadcast_addr)
                except Exception as e:
                    logger.debug("UDP beacon broadcast failed: %s", e)

            try:
                await asyncio.sleep(self.broadcast_interval_s)
            except asyncio.CancelledError:
                break

    async def _listen_loop(self) -> None:
        """Asynchronously receive and parse UDP discovery announcements."""
        loop = asyncio.get_running_loop()

        while self._running:
            if not self._recv_sock:
                break

            try:
                data, addr = await loop.sock_recvfrom(self._recv_sock, 4096)
                self._handle_raw_packet(data, addr[0])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("UDP beacon listen error: %s", e)
                await asyncio.sleep(0.5)

    def _handle_raw_packet(self, data: bytes, sender_ip: str) -> None:
        """Parse, validate, and dispatch received beacon packet."""
        try:
            packet = json.loads(data.decode("utf-8"))
        except Exception:
            return

        if not isinstance(packet, dict):
            return

        if packet.get("magic") != MAGIC_BEACON:
            return

        sender_node_id = packet.get("node_id")
        if not sender_node_id or sender_node_id == self.node_id:
            # Ignore self announcements
            return

        # Check cluster secret if configured
        if self.cluster_secret and packet.get("secret") != self.cluster_secret:
            logger.debug("Ignored discovery beacon from %s: secret mismatch", sender_node_id)
            return

        # Adjust API URL if host was 0.0.0.0 or 127.0.0.1
        api_url = packet.get("api_url", "")
        if "127.0.0.1" in api_url or "0.0.0.0" in api_url or "localhost" in api_url:
            from urllib.parse import urlsplit
            port = 11434
            try:
                parsed = urlsplit(api_url if "://" in api_url else f"http://{api_url}")
                port = parsed.port or 11434
            except Exception:
                port = 11434
            packet["api_url"] = f"http://{sender_ip}:{port}"
            packet["host"] = sender_ip

        if self.on_peer_discovered:
            try:
                self.on_peer_discovered(packet, sender_ip)
            except Exception as e:
                logger.warning("Error in on_peer_discovered callback for node %s: %s", sender_node_id, e)

    async def start(self) -> None:
        """Initialize sockets and start broadcast & listener tasks."""
        if self._running:
            return
        self._running = True

        self._send_sock = self._create_send_socket()
        self._recv_sock = self._create_recv_socket()

        if self._send_sock:
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        if self._recv_sock:
            self._listener_task = asyncio.create_task(self._listen_loop())

        logger.info("DiscoveryBeacon started on port %s (node: %s)", self.beacon_port, self.node_id)

    async def stop(self) -> None:
        """Stop background tasks and close sockets."""
        self._running = False

        for task in (self._broadcast_task, self._listener_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._broadcast_task = None
        self._listener_task = None

        if self._send_sock:
            try:
                self._send_sock.close()
            except Exception:
                pass
            self._send_sock = None

        if self._recv_sock:
            try:
                self._recv_sock.close()
            except Exception:
                pass
            self._recv_sock = None

        logger.info("DiscoveryBeacon stopped")
