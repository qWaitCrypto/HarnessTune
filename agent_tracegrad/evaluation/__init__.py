"""Evaluation data structures and replay extension surfaces."""

from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel
from agent_tracegrad.evaluation.export import (
    ablation_curve_point_to_dict,
    evaluation_run_to_markdown,
    evaluation_run_to_dict,
    evaluation_run_to_jsonl_records,
    evaluation_sample_result_to_dict,
    metric_to_dict,
    trace_level_sample_to_dict,
    write_evaluation_artifacts,
)
from agent_tracegrad.evaluation.evidence import (
    EvidenceReport,
    TokenEvidence,
    WindowEvidence,
    build_evidence_report,
    evidence_report_to_dict,
)
from agent_tracegrad.evaluation.counterfactual import DeltaLLPoint, delta_ll_curve, objective_loss
from agent_tracegrad.evaluation.annotations import FailureAnnotation, labels_for_trace, load_failure_annotations
from agent_tracegrad.evaluation.metrics import (
    MetricResult,
    delta_ll_at_k,
    method_consistency,
    metrics_for_distribution,
    rank_correlation,
    recall_at_k,
    summarize_metric_results,
)
from agent_tracegrad.evaluation.orchestration import TraceEvaluationContext, generate_evaluation_context
from agent_tracegrad.evaluation.perturbation import TraceLevelPerturbation, apply_trace_level_perturbation
from agent_tracegrad.evaluation.replay import ReplayHook
from agent_tracegrad.evaluation.runner import (
    AblationCurvePoint,
    EvaluationRunResult,
    EvaluationSampleResult,
    run_trace_level_evaluation,
)
from agent_tracegrad.evaluation.sample_generation import TraceLevelSample, generate_trace_level_samples
from agent_tracegrad.evaluation.spec import PerturbationSpec

__all__ = [
    "AblationCurvePoint",
    "DeltaLLPoint",
    "EvaluationRunResult",
    "EvaluationSampleResult",
    "EvidenceReport",
    "FailureAnnotation",
    "GroundTruthLabel",
    "MetricResult",
    "PerturbationSpec",
    "ReplayHook",
    "TraceEvaluationContext",
    "TraceLevelPerturbation",
    "TraceLevelSample",
    "TokenEvidence",
    "WindowEvidence",
    "ablation_curve_point_to_dict",
    "apply_trace_level_perturbation",
    "build_evidence_report",
    "delta_ll_at_k",
    "delta_ll_curve",
    "evaluation_run_to_dict",
    "evaluation_run_to_jsonl_records",
    "evaluation_run_to_markdown",
    "evaluation_sample_result_to_dict",
    "evidence_report_to_dict",
    "generate_evaluation_context",
    "generate_trace_level_samples",
    "labels_for_trace",
    "load_failure_annotations",
    "metric_to_dict",
    "method_consistency",
    "objective_loss",
    "metrics_for_distribution",
    "rank_correlation",
    "recall_at_k",
    "run_trace_level_evaluation",
    "summarize_metric_results",
    "trace_level_sample_to_dict",
    "write_evaluation_artifacts",
]
