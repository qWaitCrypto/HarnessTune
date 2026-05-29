"""Command-line entry point for Agent TraceGrad."""

from __future__ import annotations

import argparse
import json

from agent_tracegrad.analysis import analyze_normalized_trace, write_analysis_json
from agent_tracegrad.model import HuggingFaceCausalLMAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-tracegrad",
        description="Evaluate real-gradient attribution over structured agent traces.",
    )
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="Analyze one failed normalized JSON trace.")
    analyze.add_argument("--trace", required=True, help="Path to a normalized JSON trace file.")
    analyze.add_argument("--model", required=True, help="Local HuggingFace causal LM path or model name.")
    analyze.add_argument("--target-node-id", required=True, action="append", help="Agent node id to explain.")
    analyze.add_argument("--output", required=True, help="Path to write analysis JSON.")
    analyze.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    analyze.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    analyze.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    analyze.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    analyze.add_argument("--ig-steps", type=int, default=16, help="Integrated-gradient steps when selected.")
    analyze.add_argument("--topk-mean-k", type=int, default=5, help="k for the topk_mean aggregation view.")
    analyze.add_argument("--ranking-grain", choices=("node", "sub_block_kind"), default="node")
    analyze.add_argument(
        "--ranking-view",
        choices=("sum", "mean", "length_norm", "topk_mean"),
        default="sum",
    )
    analyze.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    subparsers.add_parser("evaluate", help="Run an attribution evaluation suite. Not implemented yet.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        _run_analyze(args)
        return 0
    if args.command == "evaluate":
        parser.error("`evaluate` is not implemented yet")
    return 0


def _run_analyze(args: argparse.Namespace) -> None:
    with open(args.trace, "r", encoding="utf-8") as handle:
        raw_trace = json.load(handle)
    dtype = _resolve_dtype(args.dtype)
    model_kwargs = {"trust_remote_code": True} if args.trust_remote_code else None
    tokenizer_kwargs = {"trust_remote_code": True} if args.trust_remote_code else None
    model = HuggingFaceCausalLMAdapter.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    result = analyze_normalized_trace(
        raw_trace,
        target_node_ids=tuple(args.target_node_id),
        model=model,
        method=args.method,
        execution_model_name=args.execution_model_name,
        topk_mean_k=args.topk_mean_k,
        ranking_grain=args.ranking_grain,
        ranking_view=args.ranking_view,
        integrated_gradients_steps=args.ig_steps,
        trace_metadata={"trace_path": args.trace},
    )
    write_analysis_json(result, args.output)


def _resolve_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


if __name__ == "__main__":
    raise SystemExit(main())
