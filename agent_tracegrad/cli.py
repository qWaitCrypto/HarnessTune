"""Command-line entry point for Agent TraceGrad."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_tracegrad.analysis import analyze_trace, write_analysis_json
from agent_tracegrad.diagnosis import (
    CandidateAction,
    read_diagnosis_json,
    run_diagnosis,
    run_drill,
    run_influence_matrix,
    run_landscape,
    write_diagnosis_json,
    write_diagnosis_markdown,
    write_diagnosis_html,
    write_drill_json,
    write_drill_markdown,
    write_influence_matrix_json,
    write_influence_matrix_markdown,
    write_landscape_json,
    write_landscape_markdown,
    write_landscape_html,
    load_diagnosis_inputs,
    load_trace_inputs,
    run_landscape_from_diagnoses,
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
    _add_trace_args(analyze, trace_arg="--trace", input_help="Trace adapter to use before analysis.")
    _add_model_args(analyze)
    _add_target_args(analyze)
    analyze.add_argument("--output", required=True, help="Path to write analysis JSON.")
    _add_objective_args(analyze)
    _add_attribution_args(analyze)
    diagnose = subparsers.add_parser("diagnose", help="Run a multi-objective diagnosis for one failed trace.")
    _add_trace_args(diagnose, trace_arg="--trace", input_help="Trace adapter to use before diagnosis.")
    _add_model_args(diagnose)
    _add_target_args(diagnose)
    _add_output_dir_args(diagnose, default_prefix="tracegrad-diagnosis", noun="diagnosis")
    _add_objective_args(diagnose)
    _add_attribution_args(diagnose)
    diagnose.add_argument("--ablation-k", action="append", type=int, default=None, help="k values for diagnosis ablation.")
    diagnose.add_argument("--control-ablation", action="store_true", help="Also ablate low-ranked control nodes.")
    diagnose.add_argument("--ablation-placeholder", default="[ABLATE]", help="Replacement text for ablated nodes.")
    drill = subparsers.add_parser("drill", help="Run policy/tool atom drill-down for one failed trace.")
    _add_trace_args(drill, trace_arg="--trace", input_help="Trace adapter to use before drill-down.")
    _add_model_args(drill)
    _add_target_args(drill)
    _add_output_dir_args(drill, default_prefix="tracegrad-drill", noun="drill")
    drill.add_argument("--diagnose-result", default=None, help="Existing diagnose JSON artifact to drill without rerunning attribution.")
    _add_objective_args(drill)
    _add_attribution_args(drill)
    drill.add_argument(
        "--candidate-action",
        action="append",
        default=[],
        help="Candidate action as id=text. May be repeated to emit an influence matrix.",
    )
    drill.add_argument(
        "--candidate-action-file",
        default=None,
        help="JSON file containing candidate actions as objects with action_id/text or an id-to-text map.",
    )
    landscape = subparsers.add_parser("landscape", help="Run harness-only landscape analysis over failed traces.")
    landscape.add_argument("--traces", default=None, help="Trace JSON file or directory of trace JSON files.")
    landscape.add_argument("--diagnose-results", default=None, help="Diagnose JSON file or directory of diagnose JSON artifacts.")
    landscape.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help="Trace adapter to use before landscape analysis.",
    )
    _add_model_args(landscape, required=False)
    _add_target_args(landscape, target_node_help="Agent node id to explain for every trace.")
    _add_output_dir_args(landscape, default_prefix="tracegrad-landscape", noun="landscape")
    _add_objective_args(landscape)
    _add_attribution_args(landscape)
    landscape.add_argument("--top-k", type=int, default=3, help="Top harness components per trace.")
    evaluate = subparsers.add_parser("evaluate", help="Run an attribution evaluation suite.")
    _add_trace_args(evaluate, trace_arg="--trace", input_help="Trace adapter to use before evaluation.")
    _add_model_args(evaluate)
    _add_target_args(evaluate)
    _add_output_dir_args(evaluate, default_prefix="tracegrad-evaluation", noun="evaluation")
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
    _add_attribution_args(evaluate)
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
    if args.command == "landscape":
        _run_landscape(args)
        return 0
    if args.command == "evaluate":
        _run_evaluate(args)
        return 0
    return 0


def _run_analyze(args: argparse.Namespace) -> None:
    raw_trace = _load_json(args.trace)
    model = _load_model(args)
    target_node_ids = _target_node_ids(args)
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
    raw_trace = _load_json(args.trace)
    operator_configs = _load_operator_configs(args)
    if not operator_configs:
        raise ValueError("evaluate requires at least one --operator-config or --operator-config-file entry")
    model = _load_model(args)
    target_node_ids = _target_node_ids(args)
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
    raw_trace = _load_json(args.trace)
    model = _load_model(args)
    result = run_diagnosis(
        raw_trace,
        model=model,
        **_diagnosis_kwargs(args, trace_path=args.trace),
        ablation_ks=tuple(args.ablation_k) if args.ablation_k else (),
        control_ablation=args.control_ablation,
        ablation_placeholder=args.ablation_placeholder,
    )
    output_dir = Path(args.output_dir)
    write_diagnosis_json(result, output_dir / f"{args.output_prefix}.json")
    write_diagnosis_markdown(result, output_dir / f"{args.output_prefix}.md")
    write_diagnosis_html(result, output_dir / f"{args.output_prefix}.html")


def _run_drill(args: argparse.Namespace) -> None:
    raw_trace = _load_json(args.trace)
    model = None if args.diagnose_result else _load_model(args)
    diagnosis = read_diagnosis_json(args.diagnose_result) if args.diagnose_result else run_diagnosis(
        raw_trace,
        model=model,
        **_diagnosis_kwargs(args, trace_path=args.trace),
    )
    drill = run_drill(diagnosis)
    output_dir = Path(args.output_dir)
    write_drill_json(drill, output_dir / f"{args.output_prefix}.json")
    write_drill_markdown(drill, output_dir / f"{args.output_prefix}.md")
    candidates = _load_candidate_actions(args)
    if candidates:
        if model is None:
            model = _load_model(args)
        matrix = run_influence_matrix(
            raw_trace,
            model=model,
            candidates=candidates,
            **_matrix_kwargs(args, trace_path=args.trace),
        )
        write_influence_matrix_json(matrix, output_dir / f"{args.output_prefix}-matrix.json")
        write_influence_matrix_markdown(matrix, output_dir / f"{args.output_prefix}-matrix.md")


def _run_landscape(args: argparse.Namespace) -> None:
    if bool(args.traces) == bool(args.diagnose_results):
        raise ValueError("landscape requires exactly one of --traces or --diagnose-results")
    if args.diagnose_results:
        result = run_landscape_from_diagnoses(
            load_diagnosis_inputs(args.diagnose_results),
            ranking_view=args.ranking_view,
            top_k=args.top_k,
        )
    else:
        if not args.model:
            raise ValueError("landscape requires --model when using --traces")
        model = _load_model(args)
        result = run_landscape(
            load_trace_inputs(args.traces),
            model=model,
            **_landscape_kwargs(args),
            top_k=args.top_k,
        )
    output_dir = Path(args.output_dir)
    write_landscape_json(result, output_dir / f"{args.output_prefix}.json")
    write_landscape_markdown(result, output_dir / f"{args.output_prefix}.md")
    write_landscape_html(result, output_dir / f"{args.output_prefix}.html")


def _add_trace_args(
    parser: argparse.ArgumentParser,
    *,
    trace_arg: str,
    trace_help: str = "Path to a trace JSON file.",
    input_help: str,
) -> None:
    parser.add_argument(trace_arg, required=True, help=trace_help)
    parser.add_argument(
        "--input-format",
        choices=trace_adapter_names(),
        default="json-fixture",
        help=input_help,
    )


def _add_model_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--model", required=required, help="Local HuggingFace causal LM path or model name.")
    parser.add_argument("--device", default=None, help="Torch device passed to the HF adapter, for example cuda:0.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for model sharding, for example cuda:0,cuda:1.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")


def _add_target_args(
    parser: argparse.ArgumentParser,
    *,
    target_node_help: str = "Agent node id to explain.",
) -> None:
    parser.add_argument("--target-node-id", action="append", help=target_node_help)
    parser.add_argument(
        "--target-marker",
        choices=failure_target_marker_names(),
        default=None,
        help="Failure target marker to use when --target-node-id is omitted.",
    )


def _add_output_dir_args(
    parser: argparse.ArgumentParser,
    *,
    default_prefix: str,
    noun: str,
) -> None:
    parser.add_argument("--output-dir", required=True, help=f"Directory to write {noun} artifacts.")
    parser.add_argument("--output-prefix", default=default_prefix, help="Artifact filename prefix.")


def _add_attribution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=("gradient_saliency", "gradient_times_input", "integrated_gradients"),
        default="gradient_saliency",
        help="Attribution method to run.",
    )
    parser.add_argument("--execution-model-name", default=None, help="Optional execution model identity.")
    parser.add_argument("--ig-steps", type=int, default=16, help="Integrated-gradient steps when selected.")
    parser.add_argument("--topk-mean-k", type=int, default=5, help="k for the topk_mean aggregation view.")
    parser.add_argument("--ranking-grain", choices=("node", "sub_block_kind"), default="node")
    parser.add_argument(
        "--ranking-view",
        choices=("sum", "mean", "length_norm", "topk_mean"),
        default="sum",
    )


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


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _target_node_ids(args: argparse.Namespace) -> tuple[str, ...] | None:
    return tuple(args.target_node_id) if args.target_node_id else None


def _diagnosis_kwargs(args: argparse.Namespace, *, trace_path: str | Path) -> dict[str, Any]:
    return {
        "input_format": args.input_format,
        "target_node_ids": _target_node_ids(args),
        "target_marker": args.target_marker,
        "target_id": args.target_id,
        "target_span": _target_span(args),
        "expected_target_text": _optional_expected_text(args),
        "expected_target_id": args.expected_target_id,
        "method": args.method,
        "execution_model_name": args.execution_model_name,
        "topk_mean_k": args.topk_mean_k,
        "ranking_grain": args.ranking_grain,
        "ranking_view": args.ranking_view,
        "integrated_gradients_steps": args.ig_steps,
        "trace_metadata": {"trace_path": str(trace_path)},
    }


def _matrix_kwargs(args: argparse.Namespace, *, trace_path: str | Path) -> dict[str, Any]:
    kwargs = _diagnosis_kwargs(args, trace_path=trace_path)
    kwargs.pop("expected_target_text")
    kwargs.pop("expected_target_id")
    return kwargs


def _landscape_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = _diagnosis_kwargs(args, trace_path=args.traces)
    kwargs.pop("trace_metadata")
    kwargs.pop("expected_target_id")
    return kwargs


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


def _load_candidate_actions(args: argparse.Namespace) -> tuple[CandidateAction, ...]:
    candidates: list[CandidateAction] = []
    if args.candidate_action_file:
        loaded = json.loads(Path(args.candidate_action_file).read_text(encoding="utf-8"))
        candidates.extend(_coerce_candidate_actions(loaded))
    for raw in args.candidate_action:
        if "=" not in raw:
            raise ValueError("--candidate-action must use id=text format")
        action_id, text = raw.split("=", 1)
        candidates.append(CandidateAction(action_id=action_id.strip(), text=text.strip()))
    return tuple(candidates)


def _coerce_candidate_actions(value: Any) -> tuple[CandidateAction, ...]:
    if isinstance(value, dict):
        if "action_id" in value and "text" in value:
            return (
                CandidateAction(
                    action_id=str(value["action_id"]),
                    text=str(value["text"]),
                    metadata=value.get("metadata") or {},
                ),
            )
        return tuple(CandidateAction(action_id=str(key), text=str(text)) for key, text in value.items())
    if isinstance(value, list):
        candidates: list[CandidateAction] = []
        for item in value:
            candidates.extend(_coerce_candidate_actions(item))
        return tuple(candidates)
    raise ValueError("candidate action file must contain an object, map, or list")


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
