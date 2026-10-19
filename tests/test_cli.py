import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from llmhub.cli import build_parser, cmd_list
from llmhub.config import LLMHubConfig
from llmhub.core.registry import ModelRegistry

def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.subcommand == "list"

    args = parser.parse_args(["pull", "org/repo:q4_k_m"])
    assert args.subcommand == "pull"
    assert args.model == "org/repo:q4_k_m"

    args = parser.parse_args(["run", "qwen", "-p", "hello", "-t", "4", "-c", "1024"])
    assert args.subcommand == "run"
    assert args.model == "qwen"
    assert args.prompt == "hello"
    assert args.threads == 4
    assert args.ctx_size == 1024

    args = parser.parse_args(["show", "qwen"])
    assert args.subcommand == "show"
    assert args.model == "qwen"

    args = parser.parse_args(["rm", "qwen"])
    assert args.subcommand == "rm"
    assert args.model == "qwen"

    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "8000"])
    assert args.subcommand in ("serve", "server")
    assert args.host == "0.0.0.0"
    assert args.port == 8000

    args_server = parser.parse_args(["server", "--low-mem"])
    assert args_server.subcommand in ("serve", "server")
    assert args_server.low_mem is True

    args = parser.parse_args(["ps"])
    assert args.subcommand == "ps"

    args = parser.parse_args(["run", "qwen", "--local", "-p", "hi"])
    assert args.subcommand == "run"
    assert args.local is True

def test_cli_list_empty(capsys):
    parser = build_parser()
    args = parser.parse_args(["list"])
    args.func(args)
    captured = capsys.readouterr()
    assert "No models found" in captured.out or "TAG" in captured.out

def test_cli_models_and_ls_aliases(capsys):
    parser = build_parser()
    
    args_models = parser.parse_args(["models"])
    assert args_models.func == cmd_list
    args_models.func(args_models)
    captured = capsys.readouterr()
    assert "No models found" in captured.out or "TAG" in captured.out

    args_ls = parser.parse_args(["ls"])
    assert args_ls.func == cmd_list
    args_ls.func(args_ls)
    captured = capsys.readouterr()
    assert "No models found" in captured.out or "TAG" in captured.out

def test_cli_stop_and_diag_parser(capsys):
    from llmhub.cli import cmd_stop, cmd_diag
    parser = build_parser()

    # serve --force
    args_serve = parser.parse_args(["serve", "--force"])
    assert args_serve.force is True

    # stop
    args_stop = parser.parse_args(["stop"])
    assert args_stop.func == cmd_stop

    # diag & aliases
    for alias in ["diag", "debug", "test"]:
        args_d = parser.parse_args([alias])
        assert args_d.func == cmd_diag
        assert args_d.model is None

    args_dm = parser.parse_args(["diag", "smollm2:135m"])
    assert args_dm.model == "smollm2:135m"

def test_cli_stop_execution(capsys, monkeypatch, tmp_path):
    from llmhub.cli import cmd_stop
    from llmhub.config import LLMHubConfig
    from llmhub.core.lifecycle import PIDManager
    parser = build_parser()
    args = parser.parse_args(["stop"])

    cfg = LLMHubConfig(base_dir=tmp_path)
    monkeypatch.setattr("llmhub.cli.LLMHubConfig", lambda: cfg)
    pid_mgr = PIDManager(cfg.run_dir)
    pid_mgr.run_dir.mkdir(parents=True, exist_ok=True)
    import json
    pid_data = {
        "pid": 999999,
        "host": "127.0.0.1",
        "port": 11434,
        "version": "1.0.0",
        "started_at": "2026-10-20T00:00:00Z"
    }
    pid_mgr.pid_file.write_text(json.dumps(pid_data))
    assert pid_mgr.pid_file.exists()

    # Mock PIDManager and port check
    monkeypatch.setattr("llmhub.server.server.is_port_in_use", lambda port, host: False)
    cmd_stop(args)
    captured = capsys.readouterr()
    assert "stopped successfully" in captured.out
    assert not pid_mgr.pid_file.exists()

def test_cli_diag_execution(capsys, monkeypatch):
    from llmhub.cli import cmd_diag
    parser = build_parser()
    args = parser.parse_args(["diag"])

    cmd_diag(args)
    captured = capsys.readouterr()
    assert "LLMHUB SYSTEM & RUNTIME DIAGNOSTIC INSPECTOR" in captured.out
    assert "Operating System" in captured.out
    assert "Registered Models" in captured.out

def test_cmd_run_fallback_on_empty_server_response(capsys, monkeypatch):
    from llmhub.cli import cmd_run
    parser = build_parser()
    args = parser.parse_args(["run", "smollm2:135m", "-p", "hello"])

    # Mock client to pretend server is running
    monkeypatch.setattr("llmhub.cli.LLMHubClient.is_server_running", lambda self: True)
    # Mock _run_via_http to return False (0 tokens or error)
    monkeypatch.setattr("llmhub.cli._run_via_http", lambda client, args: False)
    standalone_called = []
    monkeypatch.setattr("llmhub.cli._run_standalone", lambda config, args: standalone_called.append(True))

    cmd_run(args)
    captured = capsys.readouterr()
    assert "Switching to direct standalone execution" in captured.out
    assert len(standalone_called) == 1

def test_run_via_http_returns_false_on_empty_tokens():
    from llmhub.cli import _run_via_http
    parser = build_parser()
    args = parser.parse_args(["run", "smollm2:135m", "-p", "hello"])

    mock_client = MagicMock()
    mock_client.stream_chat.return_value = []  # 0 tokens
    success = _run_via_http(mock_client, args)
    assert success is False

def test_run_via_http_keyboard_interrupt_does_not_trigger_fallback(capsys):
    from llmhub.cli import _run_via_http
    parser = build_parser()
    args = parser.parse_args(["run", "smollm2:135m", "-p", "hello"])

    def raise_interrupt(*a, **kw):
        raise KeyboardInterrupt()

    mock_client = MagicMock()
    mock_client.stream_chat.side_effect = raise_interrupt
    success = _run_via_http(mock_client, args)
    # Must return True so fallback is aborted upon explicit user Ctrl+C
    assert success is True
    captured = capsys.readouterr()
    assert "Interrupted" in captured.err or "Interrupted" in captured.out




