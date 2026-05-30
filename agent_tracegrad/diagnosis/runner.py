"""Multi-objective diagnosis runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_tracegrad.analysis.single_trace import (
    SingleTraceAnalysisResult,
    analysis_from_artifact_dict,
    analysis_to_artifact_dict,
    analysis_to_dict,
    analyze_trace,
)
from agent_tracegrad.diagnosis.patterns import detect_diagnostic_labels, diagnostic_label_to_dict
from agent_tracegrad.diagnosis.types import (
    ComponentClassification,
    DiagnosisAblation,
    DiagnosisEvidence,
    DiagnosisResult,
    MarginContribution,
    MarginDistribution,
)
from agent_tracegrad.evaluation.evidence import build_evidence_report, evidence_report_to_dict
from agent_tracegrad.evaluation.perturbation.trace_level import apply_trace_level_perturbation
from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.objective import ExpectedTarget, TargetObjective
from agent_tracegrad.trace.serializer import TraceSerializer


def run_diagnosis(
    raw_trace: Any,
    *,
    model: ModelAdapter,
    expected_target_text: str | None = None,
    expected_target_id: str = "expected-1",
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    trace_metadata: Mapping[str, Any] | None = None,
    strengthen_threshold: float = 0.1,
    evidence_top_tokens: int = 8,
    evidence_top_windows: int = 5,
    evidence_window_radius: int = 2,
    ablation_ks: Sequence[int] = (),
    control_ablation: bool = False,
    ablation_placeholder: str = "[ABLATE]",
) -> DiagnosisResult:
    common = dict(
        input_format=input_format,
        target_node_ids=target_node_ids,
        target_marker=target_marker,
        model=model,
        method=method,
        execution_model_name=execution_model_name,
        target_id=target_id,
        target_span=target_span,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata=trace_metadata,
    )

    bad_result = analyze_trace(raw_trace, **common)

    if expected_target_text is None:
        return DiagnosisResult(
            bad_result=bad_result,
            expected_result=None,
            contrastive_result=None,
            margin_distributions=(),
            evidence=_build_evidence(
                (("bad_action", bad_result),),
                top_tokens=evidence_top_tokens,
                top_windows=evidence_top_windows,
                window_radius=evidence_window_radius,
            ),
            ablations=(),
            diagnostic_labels=(),
            confidence_level="weak",
            metadata={"mode": "bad_action_only"},
        )

    expected = ExpectedTarget(
        target_id=expected_target_id,
        content=expected_target_text,
        source="human",
    )
    expected_objective = TargetObjective.expected_action(expected)
    expected_result = analyze_trace(raw_trace, objective=expected_objective, **common)

    contrastive_objective = TargetObjective(
        objective_id=f"{target_id}:vs:{expected_target_id}",
        objective_type="contrastive",
        bad_target=None,
        expected_target=expected,
        source="human",
        metadata={"requires_resolved_bad_target": True},
    )
    contrastive_result = analyze_trace(raw_trace, objective=contrastive_objective, **common)

    margin_distributions = _compute_margin_distributions(
        bad_result,
        expected_result,
        contrastive_result,
        strengthen_threshold=strengthen_threshold,
    )
    ablations = _run_diagnosis_ablations(
        contrastive_result,
        model=model,
        method=method,
        execution_model_name=execution_model_name,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata=trace_metadata,
        ablation_ks=ablation_ks,
        control_ablation=control_ablation,
        ablation_placeholder=ablation_placeholder,
    )
    evidence = _build_evidence(
        (
            ("bad_action", bad_result),
            ("expected_action", expected_result),
            ("contrastive", contrastive_result),
        ),
        top_tokens=evidence_top_tokens,
        top_windows=evidence_top_windows,
        window_radius=evidence_window_radius,
    )

    result = DiagnosisResult(
        bad_result=bad_result,
        expected_result=expected_result,
        contrastive_result=contrastive_result,
        margin_distributions=margin_distributions,
        evidence=evidence,
        ablations=ablations,
        confidence_level=_confidence_level(ablations),
        metadata={"mode": "full_diagnosis"},
    )
    return DiagnosisResult(
        bad_result=result.bad_result,
        expected_result=result.expected_result,
        contrastive_result=result.contrastive_result,
        margin_distributions=result.margin_distributions,
        evidence=result.evidence,
        ablations=result.ablations,
        diagnostic_labels=detect_diagnostic_labels(result),
        confidence_level=result.confidence_level,
        metadata=result.metadata,
    )


def _compute_margin_distributions(
    bad_result: SingleTraceAnalysisResult,
    expected_result: SingleTraceAnalysisResult,
    contrastive_result: SingleTraceAnalysisResult,
    *,
    strengthen_threshold: float,
) -> tuple[MarginDistribution, ...]:
    total_margin = _extract_loss(contrastive_result)
    bad_by_key = {(d.grain, d.view_name): d for d in bad_result.distributions}
    expected_by_key = {(d.grain, d.view_name): d for d in expected_result.distributions}
    contrastive_by_key = {(d.grain, d.view_name): d for d in contrastive_result.distributions}
    distributions: list[MarginDistribution] = []
    for key in contrastive_by_key:
        if key not in bad_by_key or key not in expected_by_key:
            continue
        bad_dist = bad_by_key[key]
        expected_dist = expected_by_key[key]
        contrastive_dist = contrastive_by_key[key]
        contributions = _align_and_compute(
            bad_dist,
            expected_dist,
            contrastive_dist,
            strengthen_threshold=strengthen_threshold,
        )
        distributions.append(
            MarginDistribution(
                grain=key[0],
                view_name=key[1],
                contributions=contributions,
                total_margin=total_margin,
            )
        )
    return tuple(distributions)


def _align_and_compute(
    bad_dist,
    expected_dist,
    contrastive_dist,
    *,
    strengthen_threshold: float,
) -> tuple[MarginContribution, ...]:
    bad_by_id = {inst.instance_id: inst for inst in bad_dist.instances}
    expected_by_id = {inst.instance_id: inst for inst in expected_dist.instances}
    contrastive_by_id = {inst.instance_id: inst for inst in contrastive_dist.instances}
    all_ids = list(dict.fromkeys(list(contrastive_by_id) + list(bad_by_id) + list(expected_by_id)))

    raw: list[tuple[str, float, float, float]] = []
    for iid in all_ids:
        bad_inst = bad_by_id.get(iid)
        exp_inst = expected_by_id.get(iid)
        margin_inst = contrastive_by_id.get(iid)
        bad_score = bad_inst.views[bad_dist.view_name] if bad_inst else 0.0
        exp_score = exp_inst.views[expected_dist.view_name] if exp_inst else 0.0
        margin = margin_inst.views[contrastive_dist.view_name] if margin_inst else 0.0
        raw.append((iid, bad_score, exp_score, margin))

    max_abs_margin = max((abs(margin) for _, _, _, margin in raw), default=0.0)

    contributions: list[MarginContribution] = []
    for iid, bad_score, exp_score, margin in raw:
        classification, classification_reason = _classify_component(
            margin, exp_score, max_abs_margin, strengthen_threshold,
        )
        inst = contrastive_by_id.get(iid) or bad_by_id.get(iid) or expected_by_id[iid]
        contributions.append(
            MarginContribution(
                instance_id=iid,
                block_role=inst.block_role,
                sub_block_kind=inst.sub_block_kind,
                node_ids=inst.node_ids,
                bad_score=bad_score,
                expected_score=exp_score,
                margin=margin,
                classification=classification,
                classification_reason=classification_reason,
            )
        )
    return tuple(contributions)


def _build_evidence(
    analyses: Sequence[tuple[str, SingleTraceAnalysisResult]],
    *,
    top_tokens: int,
    top_windows: int,
    window_radius: int,
) -> tuple[DiagnosisEvidence, ...]:
    return tuple(
        DiagnosisEvidence(
            objective_name=name,
            report=build_evidence_report(
                analysis,
                top_tokens=top_tokens,
                top_windows=top_windows,
                window_radius=window_radius,
            ),
        )
        for name, analysis in analyses
    )


def _run_diagnosis_ablations(
    baseline: SingleTraceAnalysisResult,
    *,
    model: ModelAdapter,
    method: str,
    execution_model_name: str | None,
    topk_mean_k: int,
    ranking_grain: str,
    ranking_view: str,
    integrated_gradients_steps: int,
    trace_metadata: Mapping[str, Any] | None,
    ablation_ks: Sequence[int],
    control_ablation: bool,
    ablation_placeholder: str,
) -> tuple[DiagnosisAblation, ...]:
    baseline_loss = _analysis_loss(baseline)
    if baseline_loss is None:
        return ()
    ranking = _select_ranking(baseline, ranking_grain, ranking_view)
    points: list[DiagnosisAblation] = []
    for k in _normalize_positive_ks(ablation_ks):
        top_ids = _top_ranked_node_ids(ranking, k=k, reverse=False)
        if top_ids:
            point = _run_single_ablation(
                baseline,
                target_node_ids=top_ids,
                ablation_type="top",
                k=k,
                baseline_loss=baseline_loss,
                model=model,
                method=method,
                execution_model_name=execution_model_name,
                topk_mean_k=topk_mean_k,
                ranking_grain=ranking_grain,
                ranking_view=ranking_view,
                integrated_gradients_steps=integrated_gradients_steps,
                trace_metadata=trace_metadata,
                ablation_placeholder=ablation_placeholder,
            )
            if point is not None:
                points.append(point)
        if control_ablation:
            control_ids = _top_ranked_node_ids(ranking, k=k, reverse=True)
            if control_ids:
                point = _run_single_ablation(
                    baseline,
                    target_node_ids=control_ids,
                    ablation_type="control",
                    k=k,
                    baseline_loss=baseline_loss,
                    model=model,
                    method=method,
                    execution_model_name=execution_model_name,
                    topk_mean_k=topk_mean_k,
                    ranking_grain=ranking_grain,
                    ranking_view=ranking_view,
                    integrated_gradients_steps=integrated_gradients_steps,
                    trace_metadata=trace_metadata,
                    ablation_placeholder=ablation_placeholder,
                )
                if point is not None:
                    points.append(point)
    return tuple(points)


def _run_single_ablation(
    baseline: SingleTraceAnalysisResult,
    *,
    target_node_ids: Sequence[str],
    ablation_type: str,
    k: int,
    baseline_loss: float,
    model: ModelAdapter,
    method: str,
    execution_model_name: str | None,
    topk_mean_k: int,
    ranking_grain: str,
    ranking_view: str,
    integrated_gradients_steps: int,
    trace_metadata: Mapping[str, Any] | None,
    ablation_placeholder: str,
) -> DiagnosisAblation | None:
    serializer = TraceSerializer(model.tokenizer)
    perturbation = apply_trace_level_perturbation(
        baseline.trace,
        _perturbation_spec(target_node_ids, ablation_placeholder),
        serializer,
    )
    analysis = analyze_trace(
        _trace_to_raw_payload(perturbation.trace),
        input_format="json-fixture",
        target_node_ids=baseline.target.node_ids,
        model=model,
        method=method,
        execution_model_name=execution_model_name,
        target_id=baseline.target.target_id,
        target_span=baseline.target.span,
        objective=baseline.objective,
        topk_mean_k=topk_mean_k,
        ranking_grain=ranking_grain,
        ranking_view=ranking_view,
        integrated_gradients_steps=integrated_gradients_steps,
        trace_metadata={
            **dict(trace_metadata or {}),
            "diagnosis_ablation_type": ablation_type,
            "diagnosis_ablation_k": k,
        },
    )
    ablated_loss = _analysis_loss(analysis)
    if ablated_loss is None:
        return None
    return DiagnosisAblation(
        ablation_type=ablation_type,
        k=k,
        target_node_ids=tuple(target_node_ids),
        baseline_loss=baseline_loss,
        ablated_loss=ablated_loss,
        delta_loss=ablated_loss - baseline_loss,
    )


def _classify_component(
    margin: float,
    expected_score: float,
    max_abs_margin: float,
    threshold: float,
) -> tuple[ComponentClassification, str]:
    if max_abs_margin > 0.0 and abs(margin) < threshold * max_abs_margin and expected_score > 0.0:
        return "strengthen", "expected_support_present_but_margin_near_zero"
    if margin < 0.0:
        return "preserve", "expected_support_exceeds_bad_support"
    return "narrow", "bad_support_exceeds_expected_support"


def _perturbation_spec(target_node_ids: Sequence[str], placeholder: str) -> PerturbationSpec:
    return PerturbationSpec(
        operator="replace_with_placeholder",
        target_node_ids=tuple(target_node_ids),
        parameters={"placeholder": placeholder},
    )


def _trace_to_raw_payload(trace) -> Mapping[str, Any]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "block_role": node.block_role,
                "sub_block_kind": node.sub_block_kind,
                "content": node.content,
                "metadata": dict(node.metadata),
                "sequence_index": node.sequence_index,
                "timestamp": node.timestamp,
                "parents": list(node.parents),
            }
            for node in sorted(trace.nodes.values(), key=lambda item: (item.sequence_index or 0, item.node_id))
        ]
    }


def _select_ranking(
    analysis: SingleTraceAnalysisResult,
    grain: str,
    view_name: str,
):
    for ranking in analysis.rankings:
        if ranking.grain == grain and ranking.view_name == view_name:
            return ranking.items
    raise ValueError(f"missing ranking for grain={grain!r}, view_name={view_name!r}")


def _top_ranked_node_ids(ranking, *, k: int, reverse: bool) -> tuple[str, ...]:
    items = tuple(reversed(ranking)) if reverse else tuple(ranking)
    node_ids: list[str] = []
    for item in items[:k]:
        for node_id in item.instance.node_ids:
            if node_id not in node_ids:
                node_ids.append(node_id)
    return tuple(node_ids)


def _normalize_positive_ks(ks: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(k) for k in ks if int(k) > 0}))


def _analysis_loss(result: SingleTraceAnalysisResult) -> float | None:
    loss = result.attribution.metadata.get("loss")
    if loss is None:
        return None
    return float(loss)


def _confidence_level(ablations: Sequence[DiagnosisAblation]) -> str:
    if not ablations:
        return "weak"
    top = [point for point in ablations if point.ablation_type == "top"]
    control = [point for point in ablations if point.ablation_type == "control"]
    if top and control and max(abs(point.delta_loss) for point in top) > max(abs(point.delta_loss) for point in control):
        return "strong"
    return "medium"


def _extract_loss(result: SingleTraceAnalysisResult) -> float:
    loss = result.attribution.metadata.get("loss")
    if loss is None:
        return 0.0
    return float(loss)


def diagnosis_to_dict(result: DiagnosisResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bad_result": analysis_to_artifact_dict(result.bad_result),
        "confidence_level": result.confidence_level,
        "metadata": dict(result.metadata),
    }
    if result.expected_result is not None:
        payload["expected_result"] = analysis_to_artifact_dict(result.expected_result)
    if result.contrastive_result is not None:
        payload["contrastive_result"] = analysis_to_artifact_dict(result.contrastive_result)
    if result.margin_distributions:
        payload["margin_distributions"] = [
            _margin_distribution_to_dict(md) for md in result.margin_distributions
        ]
    if result.evidence:
        payload["evidence"] = [
            {
                "objective_name": evidence.objective_name,
                "report": evidence_report_to_dict(evidence.report),
            }
            for evidence in result.evidence
        ]
    if result.ablations:
        payload["ablations"] = [
            {
                "ablation_type": ablation.ablation_type,
                "k": ablation.k,
                "target_node_ids": list(ablation.target_node_ids),
                "baseline_loss": ablation.baseline_loss,
                "ablated_loss": ablation.ablated_loss,
                "delta_loss": ablation.delta_loss,
            }
            for ablation in result.ablations
        ]
    if result.diagnostic_labels:
        payload["diagnostic_labels"] = [
            diagnostic_label_to_dict(label)
            for label in result.diagnostic_labels
        ]
    return payload


def diagnosis_from_dict(payload: Mapping[str, Any]) -> DiagnosisResult:
    return DiagnosisResult(
        bad_result=analysis_from_artifact_dict(payload["bad_result"]),
        expected_result=analysis_from_artifact_dict(payload["expected_result"]) if "expected_result" in payload else None,
        contrastive_result=analysis_from_artifact_dict(payload["contrastive_result"]) if "contrastive_result" in payload else None,
        margin_distributions=tuple(
            _margin_distribution_from_dict(item)
            for item in payload.get("margin_distributions", ())
        ),
        evidence=(),
        ablations=tuple(
            DiagnosisAblation(
                ablation_type=item["ablation_type"],
                k=item["k"],
                target_node_ids=tuple(item["target_node_ids"]),
                baseline_loss=item["baseline_loss"],
                ablated_loss=item["ablated_loss"],
                delta_loss=item["delta_loss"],
            )
            for item in payload.get("ablations", ())
        ),
        diagnostic_labels=(),
        confidence_level=payload.get("confidence_level", "weak"),
        metadata=payload.get("metadata") or {},
    )


def read_diagnosis_json(path: str | Path) -> DiagnosisResult:
    return diagnosis_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_diagnosis_json(result: DiagnosisResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnosis_to_dict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _margin_distribution_to_dict(md: MarginDistribution) -> dict[str, Any]:
    return {
        "grain": md.grain,
        "view_name": md.view_name,
        "total_margin": md.total_margin,
        "contributions": [
            {
                "instance_id": c.instance_id,
                "block_role": c.block_role,
                "sub_block_kind": c.sub_block_kind,
                "node_ids": list(c.node_ids),
                "bad_score": c.bad_score,
                "expected_score": c.expected_score,
                "margin": c.margin,
                "classification": c.classification,
                "classification_reason": c.classification_reason,
            }
            for c in md.contributions
        ],
    }


def _margin_distribution_from_dict(payload: Mapping[str, Any]) -> MarginDistribution:
    return MarginDistribution(
        grain=payload["grain"],
        view_name=payload["view_name"],
        total_margin=payload["total_margin"],
        contributions=tuple(
            MarginContribution(
                instance_id=item["instance_id"],
                block_role=item["block_role"],
                sub_block_kind=item["sub_block_kind"],
                node_ids=tuple(item["node_ids"]),
                bad_score=item["bad_score"],
                expected_score=item["expected_score"],
                margin=item["margin"],
                classification=item["classification"],
                classification_reason=item.get("classification_reason", ""),
            )
            for item in payload.get("contributions", ())
        ),
    )
