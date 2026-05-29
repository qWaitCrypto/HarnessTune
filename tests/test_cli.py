from __future__ import annotations

from agent_tracegrad.cli import build_parser


def test_cli_analyze_parser_accepts_single_trace_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--trace",
            "trace.json",
            "--model",
            "/models/formal",
            "--target-node-id",
            "agent-1",
            "--output",
            "out.json",
            "--device",
            "cuda:0",
            "--devices",
            "cuda:0,cuda:1",
            "--dtype",
            "bfloat16",
        ]
    )

    assert args.command == "analyze"
    assert args.input_format == "json-fixture"
    assert args.target_node_id == ["agent-1"]
    assert args.device == "cuda:0"
    assert args.devices == "cuda:0,cuda:1"


def test_cli_analyze_parser_accepts_agentpi_marker_mode() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--trace",
            "trace.json",
            "--input-format",
            "agentpi-raw",
            "--model",
            "/models/formal",
            "--target-marker",
            "last-agent-output",
            "--output",
            "out.json",
        ]
    )

    assert args.command == "analyze"
    assert args.input_format == "agentpi-raw"
    assert args.target_node_id is None
    assert args.target_marker == "last-agent-output"
