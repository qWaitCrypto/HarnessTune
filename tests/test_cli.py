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
            "--dtype",
            "bfloat16",
        ]
    )

    assert args.command == "analyze"
    assert args.target_node_id == ["agent-1"]
    assert args.device == "cuda:0"
