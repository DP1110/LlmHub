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

def test_cli_list_empty(capsys):
    parser = build_parser()
    args = parser.parse_args(["list"])
    args.func(args)
    captured = capsys.readouterr()
    assert "No models found" in captured.out or "TAG" in captured.out
