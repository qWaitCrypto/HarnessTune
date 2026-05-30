"""Markdown report rendering for diagnosis results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agent_tracegrad.diagnosis.types import DiagnosisResult, MarginContribution, MarginDistribution
from agent_tracegrad.target.objective import target_objective_to_dict


def diagnosis_to_markdown(result: DiagnosisResult) -> str:
    lines = [
        "# Agent TraceGrad Diagnosis Report",
        "",
        "## Target Summary",
        "",
        f"- mode: `{result.metadata.get('mode', 'unknown')}`",
        f"- bad_target: `{result.bad_result.target.target_id}` nodes=`{', '.join(result.bad_result.target.node_ids)}`",
        f"- confidence: `{result.confidence_level}`",
    ]
    if result.expected_result is not None:
        lines.append(f"- expected_target: `{result.expected_result.objective.expected_target.target_id}`")
    if result.contrastive_result is not None and result.contrastive_result.objective is not None:
        objective = target_objective_to_dict(result.contrastive_result.objective)
        lines.append(f"- objective: `{objective['objective_formula']}`")

    node_sum = _select_margin(result.margin_distributions, grain="node", view_name="sum")
    if node_sum is not None:
        lines.extend(["", "## Decision Boundary", ""])
        lines.append(f"- total_margin: {node_sum.total_margin:.6g}")
        positive = [item for item in node_sum.contributions if item.margin > 0.0]
        negative = [item for item in node_sum.contributions if item.margin < 0.0]
        if positive:
            lines.append(f"- top_bad_push: `{positive[0].instance_id}` margin={positive[0].margin:.6g}")
        if negative:
            lines.append(f"- top_good_support: `{negative[0].instance_id}` margin={negative[0].margin:.6g}")
        lines.extend(["", "## Component Ranking", ""])
        lines.append("| Rank | Component | Kind | Margin | Bad | Expected | Class |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for rank, item in enumerate(_rank_margin(node_sum.contributions)[:12], start=1):
            lines.append(
                f"| {rank} | `{item.instance_id}` | `{item.sub_block_kind}` | "
                f"{item.margin:.6g} | {item.bad_score:.6g} | {item.expected_score:.6g} | "
                f"`{item.classification}` |"
            )

    if result.ablations:
        lines.extend(["", "## Ablation Evidence", ""])
        lines.append("| Type | k | Nodes | Delta Loss | Baseline | Ablated |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
        for ablation in result.ablations:
            nodes = ", ".join(ablation.target_node_ids)
            lines.append(
                f"| `{ablation.ablation_type}` | {ablation.k} | `{nodes}` | "
                f"{ablation.delta_loss:.6g} | {ablation.baseline_loss:.6g} | {ablation.ablated_loss:.6g} |"
            )

    if result.diagnostic_labels:
        lines.extend(["", "## Diagnostic Labels", ""])
        for label in result.diagnostic_labels:
            lines.append(f"### {label.label_name}")
            lines.append("")
            lines.append(f"- confidence: `{label.confidence}`")
            lines.append(f"- summary: {label.summary}")
            lines.append(f"- recommendation: {label.recommendation}")
            if label.evidence:
                lines.append("- evidence:")
                for item in label.evidence:
                    lines.append(f"  - {item}")
            lines.append("")

    if result.evidence:
        lines.extend(["", "## Evidence Windows", ""])
        for evidence in result.evidence:
            if not evidence.report.top_windows:
                continue
            lines.append(f"### {evidence.objective_name}")
            lines.append("")
            for window in evidence.report.top_windows:
                text = window.text.replace("|", "\\|")
                lines.append(f"- `{window.node_id}` score={window.score:.6g}: {text}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_diagnosis_markdown(result: DiagnosisResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diagnosis_to_markdown(result), encoding="utf-8")


def _select_margin(
    distributions: Sequence[MarginDistribution],
    *,
    grain: str,
    view_name: str,
) -> MarginDistribution | None:
    for distribution in distributions:
        if distribution.grain == grain and distribution.view_name == view_name:
            return distribution
    return None


def _rank_margin(contributions: Sequence[MarginContribution]) -> tuple[MarginContribution, ...]:
    return tuple(sorted(contributions, key=lambda item: (-abs(item.margin), item.instance_id)))
