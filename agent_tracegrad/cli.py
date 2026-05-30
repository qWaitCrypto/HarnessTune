"""Command-line entry point for Agent TraceGrad."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_tracegrad.analysis import analyze_trace, write_analysis_json
from agent_tracegrad.diagnosis import (
    run_diagnosis,
    run_drill,
    write_diagnosis_json,
    write_diagnosis_markdown,
    write_drill_json,
    write_drill_markdown,
)
from agent_tracegrad.evaluation import run_trace_level_evaluation, write_evaluation_artifacts
from agent_tracegrad.model import HuggingFaceCausalLMAdapter
from agent_tracegrad.target import ExpectedTarget, FailureTarget, TargetObjective, failure_target_marker_names
from agent_tracegrad.trace import trace_adapter_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-tracegrad",
        description="Evaluate real-gradient attribution over structured agent traces.",
    )
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="Analyze one failed trace.")
    analyze.add_argument("--trace", required=True, help="Path to a trace JSON file.")
    analyze.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help="Trace adapter to use before analysis.",
    )
    analyze.add_argument("--model", required=True, help="Local HuggingFace causal LM path or model name.")
    analyze.add_argument("--target-node-id", action="append", help="Agent node id to explain.")
    analyze.add_argument(
        "--target-marker",
        choices=failure_target_marker_names(),
        default=None,
        help="Failure target marker to use when --target-node-id is omitted.",
    )
    analyze.add_argument("--output", required=True, help="Path to write analysis JSON.")
    _add_objective_args(analyze)
    analyze.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    analyze.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    analyze.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    analyze.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for model sharding, for example cuda:0,cuda:1.",
    )
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
    diagnose = subparsers.add_parser("diagnose", help="Run a multi-objective diagnosis for one failed trace.")
    diagnose.add_argument("--trace", required=True, help="Path to a trace JSON file.")
    diagnose.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help="Trace adapter to use before diagnosis.",
    )
    diagnose.add_argument("--model", required=True, help="Local HuggingFace causal LM path or model name.")
    diagnose.add_argument("--target-node-id", action="append", help="Agent node id to explain.")
    diagnose.add_argument(
        "--target-marker",
        choices=failure_target_marker_names(),
        default=None,
        help="Failure target marker to use when --target-node-id is omitted.",
    )
    diagnose.add_argument("--output-dir", required=True, help="Directory to write diagnosis artifacts.")
    diagnose.add_argument("--output-prefix", default="tracegrad-diagnosis", help="Artifact filename prefix.")
    _add_objective_args(diagnose)
    diagnose.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    diagnose.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    diagnose.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    diagnose.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for model sharding, for example cuda:0,cuda:1.",
    )
    diagnose.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    diagnose.add_argument("--ig-steps", type=int, default=16, help="Integrated-gradient steps when selected.")
    diagnose.add_argument("--topk-mean-k", type=int, default=5, help="k for the topk_mean aggregation view.")
    diagnose.add_argument("--ranking-grain", choices=("node", "sub_block_kind"), default="node")
    diagnose.add_argument(
        "--ranking-view",
        choices=("sum", "mean", "length_norm", "topk_mean"),
        default="sum",
    )
    diagnose.add_argument("--ablation-k", action="append", type=int, default=None, help="k values for diagnosis ablation.")
    diagnose.add_argument("--control-ablation", action="store_true", help="Also ablate low-ranked control nodes.")
    diagnose.add_argument("--ablation-placeholder", default="[ABLATE]", help="Replacement text for ablated nodes.")
    diagnose.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    drill = subparsers.add_parser("drill", help="Run policy/tool atom drill-down for one failed trace.")
    drill.add_argument("--trace", required=True, help="Path to a trace JSON file.")
    drill.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help="Trace adapter to use before drill-down.",
    )
    drill.add_argument("--model", required=True, help="Local HuggingFace causal LM path or model name.")
    drill.add_argument("--target-node-id", action="append", help="Agent node id to explain.")
    drill.add_argument(
        "--target-marker",
        choices=failure_target_marker_names(),
        default=None,
        help="Failure target marker to use when --target-node-id is omitted.",
    )
    drill.add_argument("--output-dir", required=True, help="Directory to write drill artifacts.")
    drill.add_argument("--output-prefix", default="tracegrad-drill", help="Artifact filename prefix.")
    _add_objective_args(drill)
    drill.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    drill.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    drill.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    drill.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for model sharding, for example cuda:0,cuda:1.",
    )
    drill.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    drill.add_argument("--ig-steps", type=int, default=16, help="Integrated-gradient steps when selected.")
    drill.add_argument("--topk-mean-k", type=int, default=5, help="k for the topk_mean aggregation view.")
    drill.add_argument("--ranking-grain", choices=("node", "sub_block_kind"), default="node")
    drill.add_argument(
        "--ranking-view",
        choices=("sum", "mean", "length_norm", "topk_mean"),
        default="sum",
    )
    drill.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    evaluate = subparsers.add_parser("evaluate", help="Run an attribution evaluation suite.")
    evaluate.add_argument("--trace", required=True, help="Path to a trace JSON file.")
    evaluate.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help="Trace adapter to use before evaluation.",
    )
    evaluate.add_argument("--model", required=True, help="Local HuggingFace causal LM path or model name.")
    evaluate.add_argument("--target-node-id", action="append", help="Agent node id to explain.")
    evaluate.add_argument(
        "--target-marker",
        choices=failure_target_marker_names(),
        default=None,
        help="Failure target marker to use when --target-node-id is omitted.",
    )
    evaluate.add_argument("--output-dir", required=True, help="Directory to write evaluation artifacts.")
    evaluate.add_argument("--output-prefix", default="tracegrad-evaluation", help="Artifact filename prefix.")
    evaluate.add_argument(
        "--operator-config",
        action="append",
        default=[],
        help="Perturbation operator config as a JSON object. May be repeated.",
    )
    evaluate.add_argument(
        "--operator-config-file",
        default=None,
        help="JSON file containing an operator config object or list of config objects.",
    )
    evaluate.add_argument("--max-samples", type=int, default=None, help="Maximum generated perturbation samples.")
    evaluate.add_argument("--metric-k", action="append", type=int, default=None, help="k for recall@k and delta_ll@k.")
    evaluate.add_argument(
        "--ablation-k",
        action="append",
        type=int,
        default=None,
        help="k values for automatic baseline top-k ablation curve.",
    )
    evaluate.add_argument(
        "--ablation-placeholder",
        default="[ABLATE]",
        help="Replacement text for automatic ablation curve masks.",
    )
    _add_objective_args(evaluate)
    evaluate.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    evaluate.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    evaluate.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    evaluate.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for model sharding, for example cuda:0,cuda:1.",
    )
    evaluate.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    evaluate.add_argument("--ig-steps", type=int, default=16, help="Integrated-gradient steps when selected.")
    evaluate.add_argument("--topk-mean-k", type=int, default=5, help="k for the topk_mean aggregation view.")
    evaluate.add_argument("--ranking-grain", choices=("node", "sub_block_kind"), default="node")
    evaluate.add_argument(
        "--ranking-view",
        choices=("sum", "mean", "length_norm", "topk_mean"),
        default="sum",
    )
    evaluate.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        _run_analyze(args)
        return 0
    if args.command == "diagnose":
        _run_diagnose(args)
        return 0
    if args.command == "drill":
        _run_drill(args)
        return 0
    if args.command == "evaluate":
        _run_evaluate(args)
        return 0
    return 0


def _run_analyze(args: argparse.Namespace) -> None:
    with open(args.trace, "r", encoding="utf-8") as handle:
        raw_trace = json.load(handle)
    model = _load_model(args)
    target_node_ids = tuple(args.target_node_id) if args.target_node_id else None
    result = analyze_trace(
        raw_trace,
        input_format=args.input_format,
        target_node_ids=target_node_ids,
        target_marker=args.target_marker,
        objective=_build_objective(args, target_node_ids=target_node_ids),
        model=model,
        method=args.method,
        execution_model_name=args.execution_model_name,
        target_id=args.target_id,
        target_span=_target_span(args),
        topk_mean_k=args.topk_mean_k,
        ranking_grain=args.ranking_grain,
        ranking_view=args.ranking_view,
        integrated_gradients_steps=args.ig_steps,
        trace_metadata={"trace_path": args.trace},
    )
    write_analysis_json(result, args.output)


def _run_evaluate(args: argparse.Namespace) -> None:
    with open(args.trace, "r", encoding="utf-8") as handle:
        raw_trace = json.load(handle)
    operator_configs = _load_operator_configs(args)
    if not operator_configs:
        raise ValueError("evaluate requires at least one --operator-config or --operator-config-file entry")
    model = _load_model(args)
    target_node_ids = tuple(args.target_node_id) if args.target_node_id else None
    result = run_trace_level_evaluation(
        raw_trace,
        model=model,
        input_format=args.input_format,
        target_node_ids=target_node_ids,
        target_marker=args.target_marker,
        target_id=args.target_id,
        target_span=_target_span(args),
        objective=_build_objective(args, target_node_ids=target_node_ids),
        operator_configs=operator_configs,
        max_samples=args.max_samples,
        method=args.method,
        execution_model_name=args.execution_model_name,
        topk_mean_k=args.topk_mean_k,
        ranking_grain=args.ranking_grain,
        ranking_view=args.ranking_view,
        integrated_gradients_steps=args.ig_steps,
        trace_metadata={"trace_path": args.trace},
        metric_ks=tuple(args.metric_k) if args.metric_k else (1, 3, 5),
        ablation_ks=tuple(args.ablation_k) if args.ablation_k else (1, 3, 5),
        ablation_placeholder=args.ablation_placeholder,
    )
    write_evaluation_artifacts(result, output_dir=args.output_dir, prefix=args.output_prefix)


def _run_diagnose(args: argparse.Namespace) -> None:
    with open(args.trace, "r", encoding="utf-8") as handle:
        raw_trace = json.load(handle)
    model = _load_model(args)
    target_node_ids = tuple(args.target_node_id) if args.target_node_id else None
    result = run_diagnosis(
        raw_trace,
        input_format=args.input_format,
        target_node_ids=target_node_ids,
        target_marker=args.target_marker,
        model=model,
        method=args.method,
        execution_model_name=args.execution_model_name,
        target_id=args.target_id,
        target_span=_target_span(args),
        expected_target_text=_optional_expected_text(args),
        expected_target_id=args.expected_target_id,
        topk_mean_k=args.topk_mean_k,
        ranking_grain=args.ranking_grain,
        ranking_view=args.ranking_view,
        integrated_gradients_steps=args.ig_steps,
        trace_metadata={"trace_path": args.trace},
        ablation_ks=tuple(args.ablation_k) if args.ablation_k else (),
        control_ablation=args.control_ablation,
        ablation_placeholder=args.ablation_placeholder,
    )
    output_dir = Path(args.output_dir)
    write_diagnosis_json(result, output_dir / f"{args.output_prefix}.json")
    write_diagnosis_markdown(result, output_dir / f"{args.output_prefix}.md")


def _run_drill(args: argparse.Namespace) -> None:
    with open(args.trace, "r", encoding="utf-8") as handle:
        raw_trace = json.load(handle)
    model = _load_model(args)
    target_node_ids = tuple(args.target_node_id) if args.target_node_id else None
    diagnosis = run_diagnosis(
        raw_trace,
        input_format=args.input_format,
        target_node_ids=target_node_ids,
        target_marker=args.target_marker,
        model=model,
        method=args.method,
        execution_model_name=args.execution_model_name,
        target_id=args.target_id,
        target_span=_target_span(args),
        expected_target_text=_optional_expected_text(args),
        expected_target_id=args.expected_target_id,
        topk_mean_k=args.topk_mean_k,
        ranking_grain=args.ranking_grain,
        ranking_view=args.ranking_view,
        integrated_gradients_steps=args.ig_steps,
        trace_metadata={"trace_path": args.trace},
    )
    drill = run_drill(diagnosis)
    output_dir = Path(args.output_dir)
    write_drill_json(drill, output_dir / f"{args.output_prefix}.json")
    write_drill_markdown(drill, output_dir / f"{args.output_prefix}.md")


def _add_objective_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--objective-type",
        choices=("bad_action", "expected_action", "contrastive"),
        default="bad_action",
        help="Diagnostic target objective to explain.",
    )
    parser.add_argument("--objective-id", default=None, help="Optional stable objective id.")
    parser.add_argument(
        "--objective-source",
        choices=("trace", "human", "benchmark", "synthetic"),
        default=None,
        help="Source label for expected or contrastive objectives.",
    )
    parser.add_argument("--target-id", default="target-1", help="Stable id for the trace failure target.")
    parser.add_argument(
        "--expected-target-id",
        default="expected-1",
        help="Stable id for expected-action or contrastive target text.",
    )
    parser.add_argument("--expected-target-text", default=None, help="Expected target text.")
    parser.add_argument("--expected-target-file", default=None, help="Path to expected target text.")
    parser.add_argument("--target-span-start", type=int, default=None, help="Optional target token span start.")
    parser.add_argument("--target-span-end", type=int, default=None, help="Optional target token span end.")


def _build_objective(args: argparse.Namespace, *, target_node_ids: tuple[str, ...] | None) -> TargetObjective | None:
    target_span = _target_span(args)
    bad_target = (
        FailureTarget(target_id=args.target_id, node_ids=target_node_ids, span=target_span)
        if target_node_ids
        else None
    )
    if args.objective_type == "bad_action":
        return TargetObjective.bad_action(
            bad_target,
            objective_id=args.objective_id,
            source=args.objective_source or "trace",
        ) if bad_target is not None else None
    expected = _expected_target(args)
    if args.objective_type == "expected_action":
        return TargetObjective.expected_action(
            expected,
            objective_id=args.objective_id,
            source=args.objective_source,
        )
    if bad_target is None:
        return TargetObjective(
            objective_id=args.objective_id or f"{args.target_id}:vs:{expected.target_id}",
            objective_type="contrastive",
            bad_target=None,
            expected_target=expected,
            source=args.objective_source or expected.source,
            metadata={"requires_resolved_bad_target": True},
        )
    return TargetObjective.contrastive(
        bad_target,
        expected,
        objective_id=args.objective_id,
        source=args.objective_source or expected.source,
    )


def _expected_target(args: argparse.Namespace) -> ExpectedTarget:
    content = args.expected_target_text
    if args.expected_target_file:
        content = Path(args.expected_target_file).read_text(encoding="utf-8")
    if not content:
        raise ValueError(f"{args.objective_type} requires --expected-target-text or --expected-target-file")
    return ExpectedTarget(
        target_id=args.expected_target_id,
        content=content.strip(),
        source=args.objective_source or "human",
    )


def _optional_expected_text(args: argparse.Namespace) -> str | None:
    content = args.expected_target_text
    if args.expected_target_file:
        content = Path(args.expected_target_file).read_text(encoding="utf-8")
    return content.strip() if content else None


def _target_span(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.target_span_start is None and args.target_span_end is None:
        return None
    if args.target_span_start is None or args.target_span_end is None:
        raise ValueError("--target-span-start and --target-span-end must be provided together")
    return (args.target_span_start, args.target_span_end)


def _load_operator_configs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    configs: list[dict[str, Any]] = []
    if args.operator_config_file:
        loaded = json.loads(Path(args.operator_config_file).read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            configs.extend(_coerce_operator_config(item) for item in loaded)
        else:
            configs.append(_coerce_operator_config(loaded))
    configs.extend(_coerce_operator_config(json.loads(raw)) for raw in args.operator_config)
    return tuple(configs)


def _coerce_operator_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("operator config must be a JSON object")
    return value


def _load_model(args: argparse.Namespace) -> HuggingFaceCausalLMAdapter:
    dtype = _resolve_dtype(args.dtype)
    model_kwargs = {"trust_remote_code": True} if args.trust_remote_code else None
    tokenizer_kwargs = {"trust_remote_code": True} if args.trust_remote_code else None
    return HuggingFaceCausalLMAdapter.from_pretrained(
        args.model,
        device=args.device,
        devices=args.devices,
        dtype=dtype,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )


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
