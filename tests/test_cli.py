import sys
from pathlib import Path
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
