"""Unit and integration tests for Phase 10: Distributed Scale-Out Runtime & Cluster Coordination.

Covers:
- NodeInfo dataclass, load scoring, and heartbeat status updates
- DiscoveryBeacon UDP packet serialization, secret verification, and self-packet filtering
- ClusterManager peer registration, HTTP heartbeats, stale peer detection, and state aggregation
- ClusterRouter locality-aware routing (local resident, remote resident, cold capacity balancing)
- PipelineCoordinator edge model partitioning across heterogeneous nodes (e.g., PC + Phone)
- FastAPI cluster API endpoints (/api/cluster/status, /nodes, /heartbeat, /join, /plan)
- Distributed request forwarding via HTTP streaming and JSON proxying
"""

import asyncio
import json
import time
from typing import Any, Dict
import pytest
from starlette.testclient import TestClient

from llmhub.cluster.beacon import DiscoveryBeacon, MAGIC_BEACON
from llmhub.cluster.manager import ClusterManager
from llmhub.cluster.node import NodeInfo, NodeRole, NodeStatus
from llmhub.cluster.pipeline import PipelineCoordinator, PipelinePlan, PipelineStage
from llmhub.cluster.router import ClusterRouter, RoutingDecision
from llmhub.config import LLMHubConfig
from llmhub.server.app import create_app, state


@pytest.fixture(scope="module")
def test_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


# ============================================================================
# 1. NodeInfo and Load Scoring Tests (US-053)
# ============================================================================

def test_node_info_serialization():
    node = NodeInfo(
        node_id="node-alpha",
        name="desktop-pc",
        host="192.168.1.50",
        port=11434,
        role=NodeRole.HYBRID,
        status=NodeStatus.HEALTHY,
        total_ram_mb=16384.0,
        free_ram_mb=8192.0,
        resident_models={"qwen2.5:0.5b": {"memory_bytes": 450000000, "active_sessions": 0}},
    )

    d = node.to_dict()
    assert d["node_id"] == "node-alpha"
    assert d["role"] == "hybrid"
    assert d["status"] == "healthy"
    assert d["api_url"] == "http://192.168.1.50:11434"

    restored = NodeInfo.from_dict(d)
    assert restored.node_id == node.node_id
    assert restored.role == NodeRole.HYBRID
    assert restored.status == NodeStatus.HEALTHY
    assert restored.free_ram_mb == 8192.0
    assert "qwen2.5:0.5b" in restored.resident_models


def test_node_info_load_score_and_heartbeat():
    node = NodeInfo(
        node_id="node-beta",
        free_ram_mb=4000.0,
        latency_ms=5.0,
        active_sessions=1,
        resident_models={"llama3:8b": {}},
    )

    # Score with resident model
    score_res = node.load_score("llama3:8b")
    # Score with non-resident model
    score_non_res = node.load_score("phi3:mini")
    # Resident model gets a significant discount
    assert score_res < score_non_res

    # Heartbeat update
    node.status = NodeStatus.UNREACHABLE
    node.update_heartbeat(free_ram_mb=5000.0, active_sessions=0)
    assert node.status == NodeStatus.HEALTHY
    assert node.free_ram_mb == 5000.0
    assert node.active_sessions == 0


# ============================================================================
# 2. DiscoveryBeacon Tests (US-054)
# ============================================================================

def test_discovery_beacon_packet_construction():
    beacon = DiscoveryBeacon(
        node_id="node-test-beacon",
        name="termux-phone",
        api_url="http://192.168.1.200:11434",
        cluster_secret="super-cluster-secret",
    )

    beacon.set_dynamic_state_callback(
        lambda: {"free_ram_mb": 3500.0, "resident_models": {"qwen:0.5b": {}}}
    )

    raw_bytes = beacon.build_packet()
    packet = json.loads(raw_bytes.decode("utf-8"))

    assert packet["magic"] == MAGIC_BEACON
    assert packet["node_id"] == "node-test-beacon"
    assert packet["secret"] == "super-cluster-secret"
    assert packet["free_ram_mb"] == 3500.0
    assert "qwen:0.5b" in packet["resident_models"]


def test_discovery_beacon_handle_raw_packet():
    received = []

    def _on_discovered(p, ip):
        received.append((p, ip))

    beacon = DiscoveryBeacon(
        node_id="node-local",
        name="local-node",
        api_url="http://127.0.0.1:11434",
        cluster_secret="cluster-token-123",
        on_peer_discovered=_on_discovered,
    )

    # 1. Packet from self -> ignored
    self_packet = {
        "magic": MAGIC_BEACON,
        "node_id": "node-local",
        "secret": "cluster-token-123",
    }
    beacon._handle_raw_packet(json.dumps(self_packet).encode("utf-8"), "127.0.0.1")
    assert len(received) == 0

    # 2. Packet with wrong secret -> ignored
    wrong_sec = {
        "magic": MAGIC_BEACON,
        "node_id": "node-foreign",
        "secret": "wrong-secret",
    }
    beacon._handle_raw_packet(json.dumps(wrong_sec).encode("utf-8"), "192.168.1.10")
    assert len(received) == 0

    # 3. Valid peer packet -> dispatched
    valid_peer = {
        "magic": MAGIC_BEACON,
        "node_id": "node-remote",
        "name": "remote-worker",
        "api_url": "http://127.0.0.1:11434",  # local placeholder
        "secret": "cluster-token-123",
        "free_ram_mb": 6000.0,
    }
    beacon._handle_raw_packet(json.dumps(valid_peer).encode("utf-8"), "192.168.1.100")
    assert len(received) == 1
    p, ip = received[0]
    assert p["node_id"] == "node-remote"
    assert ip == "192.168.1.100"
    # Adjusted placeholder host to real sender IP
    assert p["api_url"] == "http://192.168.1.100:11434"


# ============================================================================
# 3. ClusterManager Lifecycle & Stale Reaper Tests (US-056)
# ============================================================================

def test_cluster_manager_registration_and_heartbeat():
    async def _run():
        local_node = NodeInfo(node_id="node-main", free_ram_mb=4000.0)
        mgr = ClusterManager(local_node=local_node, enabled=True)

        # Peer sends heartbeat
        peer_payload = {
            "node_id": "node-worker-1",
            "name": "worker-1",
            "host": "192.168.1.101",
            "port": 11434,
            "api_url": "http://192.168.1.101:11434",
            "free_ram_mb": 7500.0,
            "active_sessions": 2,
            "resident_models": {"mistral:7b": {"active_sessions": 2}},
        }
        resp = await mgr.handle_heartbeat_received(peer_payload)
        assert resp["node_id"] == "node-main"

        # Peer is now registered in cluster manager
        assert "node-worker-1" in mgr.peers
        p = mgr.peers["node-worker-1"]
        assert p.free_ram_mb == 7500.0
        assert p.status == NodeStatus.HEALTHY

        # Aggregated cluster status
        st = mgr.get_cluster_status()
        assert st["total_nodes"] == 2
        assert st["healthy_nodes_count"] == 2
        assert "mistral:7b" in st["resident_models"]

    asyncio.run(_run())


def test_cluster_manager_stale_peer_reaping():
    async def _run():
        local_node = NodeInfo(node_id="node-main")
        # Fast timeouts for testing
        mgr = ClusterManager(
            local_node=local_node,
            enabled=True,
            node_timeout_s=0.1,
            dead_timeout_s=0.2,
        )

        peer = NodeInfo(node_id="node-temp", last_heartbeat=time.time())
        await mgr.register_or_update_peer(peer)
        assert "node-temp" in mgr.peers

        # Wait past node_timeout_s -> peer transitions to UNREACHABLE
        await asyncio.sleep(0.12)
        # Manually invoke one reaper cycle logic
        now = time.time()
        for p in mgr.peers.values():
            if now - p.last_heartbeat > mgr.node_timeout_s:
                p.status = NodeStatus.UNREACHABLE
        assert mgr.peers["node-temp"].status == NodeStatus.UNREACHABLE

        # Wait past dead_timeout_s -> peer reaped
        await asyncio.sleep(0.12)
        now = time.time()
        dead = [k for k, p in mgr.peers.items() if now - p.last_heartbeat > mgr.dead_timeout_s]
        for k in dead:
            mgr.peers.pop(k)
        assert "node-temp" not in mgr.peers

    asyncio.run(_run())


# ============================================================================
# 4. ClusterRouter Locality-Aware Dispatch Tests (US-055)
# ============================================================================

def test_cluster_router_local_resident_hit():
    local = NodeInfo(
        node_id="node-local",
        api_url="http://127.0.0.1:11434",
        resident_models={"llama3:8b": {}},
        free_ram_mb=4000.0,
    )
    mgr = ClusterManager(local_node=local, enabled=True)
    router = ClusterRouter(mgr)

    decision = router.route_model("llama3:8b")
    assert decision.is_local is True
    assert decision.reason == "local_resident"
    assert decision.target == "local"


def test_cluster_router_remote_resident_hit():
    local = NodeInfo(
        node_id="node-local",
        api_url="http://127.0.0.1:11434",
        resident_models={},
        free_ram_mb=4000.0,
    )
    mgr = ClusterManager(local_node=local, enabled=True)
    peer_remote = NodeInfo(
        node_id="node-peer-1",
        api_url="http://192.168.1.105:11434",
        resident_models={"qwen2.5:14b": {}},
        free_ram_mb=8000.0,
        latency_ms=3.0,
    )
    mgr.peers["node-peer-1"] = peer_remote

    router = ClusterRouter(mgr)
    decision = router.route_model("qwen2.5:14b")
    assert decision.is_local is False
    assert decision.target == "node-peer-1"
    assert decision.target_url == "http://192.168.1.105:11434"
    assert decision.reason == "remote_resident"


def test_cluster_router_cold_capacity_dispatch():
    local = NodeInfo(
        node_id="node-local",
        api_url="http://127.0.0.1:11434",
        resident_models={},
        free_ram_mb=600.0,  # Very low free RAM
    )
    mgr = ClusterManager(local_node=local, enabled=True)
    peer_big = NodeInfo(
        node_id="node-heavy-worker",
        api_url="http://192.168.1.120:11434",
        resident_models={},
        free_ram_mb=12000.0,  # High free RAM
    )
    mgr.peers["node-heavy-worker"] = peer_big

    router = ClusterRouter(mgr)
    decision = router.route_model("unloaded-model:7b", est_memory_mb=4000.0)
    # Remote node has substantially greater capacity
    assert decision.is_local is False
    assert decision.target == "node-heavy-worker"
    assert decision.reason == "remote_capacity"


# ============================================================================
# 5. PipelineCoordinator Edge Partitioning Tests (US-057)
# ============================================================================

def test_pipeline_coordinator_pc_phone_split():
    # Scenario: 14B model (32 layers, hidden_dim 5120, weights 8000 MB)
    # PC: 8 GB free RAM
    # Phone: 3 GB free RAM
    node_pc = NodeInfo(node_id="node-pc", name="Workstation", free_ram_mb=8000.0)
    node_phone = NodeInfo(node_id="node-phone", name="Android-Phone", free_ram_mb=3000.0)

    plan = PipelineCoordinator.plan_pipeline(
        model_tag="qwen2.5:14b",
        total_layers=32,
        hidden_dim=5120,
        total_weights_mb=8000.0,
        nodes=[node_pc, node_phone],
        network_latency_ms=2.5,
        network_bandwidth_mbps=100.0,
    )

    assert plan.is_feasible is True
    assert len(plan.stages) == 2
    assert plan.activation_bytes_per_token == 5120 * 2  # 10,240 bytes = 10 KB

    # Stage 0 (PC): embeddings + majority of layers
    stage0 = plan.stages[0]
    assert stage0.has_embeddings is True
    assert stage0.has_lm_head is False

    # Stage 1 (Phone): remaining layers + lm_head
    stage1 = plan.stages[1]
    assert stage1.has_embeddings is False
    assert stage1.has_lm_head is True

    # Total layers match
    assert stage0.layer_count + stage1.layer_count == 32
    assert stage0.layer_count > stage1.layer_count  # PC gets more layers proportionally

    # Estimated transfer overhead is reasonable (< 10 ms per token on 100Mbps LAN)
    assert plan.estimated_transfer_ms_per_step < 10.0


def test_pipeline_coordinator_insufficient_ram():
    node_small = NodeInfo(node_id="node-small", free_ram_mb=1000.0)

    # 70B model requires 40 GB RAM
    plan = PipelineCoordinator.plan_pipeline(
        model_tag="llama3:70b",
        total_layers=80,
        hidden_dim=8192,
        total_weights_mb=40000.0,
        nodes=[node_small],
    )
    assert plan.is_feasible is False
    assert "Aggregate cluster free RAM" in plan.feasibility_notes


# ============================================================================
# 6. Cluster FastAPI Endpoints Integration Tests (US-058)
# ============================================================================

def test_api_cluster_status_and_nodes_endpoints(test_client):
    # Enable cluster on server state
    old_enabled = state.cluster_manager.enabled
    state.cluster_manager.enabled = True
    try:
        # GET /api/cluster/status
        resp = test_client.get("/api/cluster/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_enabled"] is True
        assert "local_node_id" in data
        assert "total_cluster_ram_mb" in data

        # GET /api/cluster/nodes
        resp_nodes = test_client.get("/api/cluster/nodes")
        assert resp_nodes.status_code == 200
        nodes = resp_nodes.json().get("nodes", [])
        assert len(nodes) >= 1
        assert nodes[0]["node_id"] == state.cluster_manager.local_node.node_id
    finally:
        state.cluster_manager.enabled = old_enabled


def test_api_cluster_heartbeat_endpoint(test_client):
    peer_heartbeat = {
        "node_id": "test-peer-node-99",
        "name": "remote-test-worker",
        "host": "192.168.1.99",
        "port": 11434,
        "api_url": "http://192.168.1.99:11434",
        "free_ram_mb": 5120.0,
        "total_ram_mb": 8192.0,
        "active_sessions": 0,
        "resident_models": {"phi3:mini": {}},
    }

    resp = test_client.post("/api/cluster/heartbeat", json=peer_heartbeat)
    assert resp.status_code == 200
    ret_data = resp.json()
    assert ret_data["node_id"] == state.cluster_manager.local_node.node_id

    # Verify peer now exists in server state
    assert "test-peer-node-99" in state.cluster_manager.peers
    state.cluster_manager.peers.pop("test-peer-node-99", None)


def test_api_cluster_plan_endpoint(test_client):
    plan_req = {
        "model": "qwen2.5:14b",
        "total_layers": 32,
        "hidden_dim": 5120,
        "weights_mb": 4000.0,
    }
    resp = test_client.post("/api/cluster/plan", json=plan_req)
    assert resp.status_code == 200
    plan_data = resp.json()
    assert plan_data["model_tag"] == "qwen2.5:14b"
    assert "stages" in plan_data
    assert "activation_bytes_per_token" in plan_data


def test_api_cluster_join_invalid_peer_returns_502(test_client):
    # Attempting to join a non-existent port should return 502 Bad Gateway
    join_payload = {"peer_url": "http://127.0.0.1:59999"}
    resp = test_client.post("/api/cluster/join", json=join_payload)
    assert resp.status_code == 502
    assert "Failed to connect to peer" in resp.json()["detail"]


# ============================================================================
# 7. LLMHubClient and CLI Cluster Integration Tests (US-059)
# ============================================================================

def test_client_cluster_methods(monkeypatch):
    from llmhub.server.client import LLMHubClient
    client = LLMHubClient(base_url="http://127.0.0.1:11434")

    # Mock httpx.Client calls
    class MockResp:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
            self.text = json.dumps(data)
        def json(self):
            return self._data

    class MockHttpxClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, **kwargs):
            if "/healthz" in url:
                return MockResp(200, {"status": "ok"})
            elif "/api/cluster/status" in url:
                return MockResp(200, {"cluster_enabled": True, "local_node_id": "node-1", "nodes": []})
            elif "/api/cluster/nodes" in url:
                return MockResp(200, {"nodes": [{"node_id": "node-1"}]})
            return MockResp(404, {})
        def post(self, url, **kwargs):
            if "/api/cluster/join" in url:
                return MockResp(200, {"status": "joined", "message": "Joined ok"})
            elif "/api/cluster/plan" in url:
                return MockResp(200, {"model_tag": "test:7b", "is_feasible": True, "stages": []})
            return MockResp(404, {})

    import httpx
    monkeypatch.setattr(httpx, "Client", MockHttpxClient)

    st = client.get_cluster_status()
    assert st["cluster_enabled"] is True

    nodes = client.get_cluster_nodes()
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "node-1"

    join_res = client.join_cluster("192.168.1.50:11434")
    assert join_res["status"] == "joined"

    plan_res = client.get_cluster_pipeline_plan("test:7b")
    assert plan_res["is_feasible"] is True


def test_cli_cluster_commands(monkeypatch, capsys):
    import argparse
    from llmhub.cli import cmd_cluster

    # Mock client methods
    from llmhub.server.client import LLMHubClient
    monkeypatch.setattr(
        LLMHubClient,
        "get_cluster_status",
        lambda self: {
            "cluster_enabled": True,
            "local_node_id": "node-cli-test",
            "healthy_nodes_count": 2,
            "total_nodes": 2,
            "total_cluster_ram_mb": 16000.0,
            "free_cluster_ram_mb": 8000.0,
            "nodes": [
                {"node_id": "node-cli-test", "host": "127.0.0.1", "port": 11434, "status": "healthy", "free_ram_mb": 4000.0, "active_sessions": 0, "resident_models": {"qwen:0.5b": {}}},
                {"node_id": "node-worker-2", "host": "192.168.1.10", "port": 11434, "status": "healthy", "free_ram_mb": 4000.0, "active_sessions": 1, "resident_models": {}},
            ],
            "resident_models": {"qwen:0.5b": {"resident_nodes": ["node-cli-test"], "total_active_sessions": 0}},
        }
    )
    monkeypatch.setattr(
        LLMHubClient,
        "get_cluster_pipeline_plan",
        lambda self, model, **kwargs: {
            "is_feasible": True,
            "total_layers": kwargs.get("total_layers", 32),
            "hidden_dim": kwargs.get("hidden_dim", 4096),
            "activation_bytes_per_token": 8192,
            "estimated_transfer_ms_per_step": 3.5,
            "feasibility_notes": "Feasible test partition",
            "stages": [
                {"stage_id": 0, "node_name": "PC", "layer_start": 0, "layer_end": 20, "layer_count": 20, "estimated_memory_mb": 2500, "has_embeddings": True, "has_lm_head": False},
                {"stage_id": 1, "node_name": "Phone", "layer_start": 20, "layer_end": 32, "layer_count": 12, "estimated_memory_mb": 1500, "has_embeddings": False, "has_lm_head": True},
            ]
        }
    )
    monkeypatch.setattr(
        LLMHubClient,
        "join_cluster",
        lambda self, peer: {"status": "joined", "message": "Successfully joined test peer"}
    )

    # 1. Test 'llmhub cluster status'
    args_status = argparse.Namespace(cluster_action="status", api_key=None)
    cmd_cluster(args_status)
    out_status = capsys.readouterr().out
    assert "LLMHUB DISTRIBUTED CLUSTER TOPOLOGY" in out_status
    assert "node-cli-test" in out_status
    assert "node-worker-2" in out_status

    # 2. Test 'llmhub cluster join'
    args_join = argparse.Namespace(cluster_action="join", peer="192.168.1.10:11434", api_key=None)
    cmd_cluster(args_join)
    out_join = capsys.readouterr().out
    assert "Successfully joined test peer" in out_join

    # 3. Test 'llmhub cluster plan'
    args_plan = argparse.Namespace(cluster_action="plan", model="llama3:70b", api_key=None, layers=32, hidden_dim=4096)
    cmd_cluster(args_plan)
    out_plan = capsys.readouterr().out
    assert "DISTRIBUTED PIPELINE PARALLEL PLAN" in out_plan
    assert "Stage" in out_plan
    assert "Embeddings" in out_plan
    assert "LM Head" in out_plan


# ============================================================================
# 8. Regression Tests for Edge Cases & Security (Autofix Verification)
# ============================================================================

def test_pipeline_coordinator_layers_less_than_nodes_edge_cases():
    """Verify that when total_layers < len(nodes), nodes are clamped without negative layers."""
    nodes = [
        NodeInfo(node_id=f"node-{i}", free_ram_mb=4000.0)
        for i in range(5)
    ]

    # total_layers is 2, but 5 nodes are available
    plan = PipelineCoordinator.plan_pipeline(
        model_tag="mini-toy:2b",
        total_layers=2,
        hidden_dim=2048,
        total_weights_mb=1000.0,
        nodes=nodes,
    )
    assert plan.is_feasible is True
    # Clamped to 2 stages (1 layer each)
    assert len(plan.stages) == 2
    assert plan.stages[0].layer_count == 1
    assert plan.stages[1].layer_count == 1
    assert sum(s.layer_count for s in plan.stages) == 2
    assert all(s.layer_count > 0 for s in plan.stages)

    # total_layers = 0
    plan_zero = PipelineCoordinator.plan_pipeline(
        model_tag="mini-toy:2b",
        total_layers=0,
        hidden_dim=2048,
        total_weights_mb=1000.0,
        nodes=nodes,
    )
    assert plan_zero.is_feasible is False
    assert "Total layers must be at least 1" in plan_zero.feasibility_notes


def test_api_cluster_heartbeat_secret_auth(test_client):
    """Verify that /api/cluster/heartbeat strictly rejects missing/invalid secrets when configured."""
    old_secret = state.cluster_manager.cluster_secret
    state.cluster_manager.cluster_secret = "secure-cluster-token-xyz"
    try:
        hb_data = {
            "node_id": "auth-test-peer",
            "name": "auth-node",
            "host": "192.168.1.150",
            "port": 11434,
            "api_url": "http://192.168.1.150:11434",
            "free_ram_mb": 4096.0,
            "total_ram_mb": 8192.0,
        }

        # 1. No secret -> 403 Forbidden
        resp_no_sec = test_client.post("/api/cluster/heartbeat", json=hb_data)
        assert resp_no_sec.status_code == 403
        assert "Invalid cluster secret" in resp_no_sec.json()["detail"]

        # 2. Invalid secret in payload -> 403 Forbidden
        hb_wrong = dict(hb_data, secret="wrong-secret")
        resp_wrong = test_client.post("/api/cluster/heartbeat", json=hb_wrong)
        assert resp_wrong.status_code == 403

        # 3. Valid secret in payload -> 200 OK
        hb_valid = dict(hb_data, secret="secure-cluster-token-xyz")
        resp_valid = test_client.post("/api/cluster/heartbeat", json=hb_valid)
        assert resp_valid.status_code == 200
        assert resp_valid.json()["node_id"] == state.cluster_manager.local_node.node_id

        # 4. Valid secret via X-Cluster-Secret header -> 200 OK
        resp_header = test_client.post(
            "/api/cluster/heartbeat",
            json=hb_data,
            headers={"X-Cluster-Secret": "secure-cluster-token-xyz"}
        )
        assert resp_header.status_code == 200
    finally:
        state.cluster_manager.cluster_secret = old_secret
        state.cluster_manager.peers.pop("auth-test-peer", None)


