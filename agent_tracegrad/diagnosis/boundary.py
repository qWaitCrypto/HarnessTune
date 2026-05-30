"""Decision-boundary artifact derived from diagnosis results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from agent_tracegrad.diagnosis.types import DiagnosisResult, MarginContribution, MarginDistribution
from agent_tracegrad.target.objective import target_objective_to_dict

BoundaryDirection = Literal["pushes_bad", "pushes_expected", "neutral"]


@dataclass(frozen=True)
class BoundaryComponent:
    component_id: str
    sub_block_kind: str
    node_ids: Sequence[str]
    direction: BoundaryDirection
    classification: str
    margin: float
    bad_score: float
    expected_score: float
    classification_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))


@dataclass(frozen=True)
class DecisionBoundaryArtifact:
    target_id: str
    target_node_ids: Sequence[str]
    objective_formula: str
    total_margin: float
    components: Sequence[BoundaryComponent]
    bad_push_components: Sequence[BoundaryComponent]
    expected_support_components: Sequence[BoundaryComponent]
    strengthen_components: Sequence[BoundaryComponent]
    diagnostic_labels: Sequence[str]
    confidence_level: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_node_ids", tuple(self.target_node_ids))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "bad_push_components", tuple(self.bad_push_components))
        object.__setattr__(self, "expected_support_components", tuple(self.expected_support_components))
        object.__setattr__(self, "strengthen_components", tuple(self.strengthen_components))
        object.__setattr__(self, "diagnostic_labels", tuple(self.diagnostic_labels))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def build_decision_boundary_artifact(
    diagnosis: DiagnosisResult,
    *,
    grain: str = "node",
    view_name: str = "sum",
    top_k: int = 8,
) -> DecisionBoundaryArtifact:
    distribution = _select_margin(diagnosis.margin_distributions, grain=grain, view_name=view_name)
    if distribution is None:
        raise ValueError(f"decision boundary artifact requires margin distribution {grain}/{view_name}")
    ranked = tuple(_component_from_margin(item) for item in _rank_by_abs_margin(distribution.contributions))
    bad_push = tuple(item for item in ranked if item.direction == "pushes_bad")[:top_k]
    expected_support = tuple(item for item in ranked if item.direction == "pushes_expected")[:top_k]
    strengthen = tuple(
        sorted(
            (item for item in ranked if item.classification == "strengthen"),
            key=lambda item: (-item.expected_score, item.component_id),
        )[:top_k]
    )
    return DecisionBoundaryArtifact(
        target_id=diagnosis.bad_result.target.target_id,
        target_node_ids=diagnosis.bad_result.target.node_ids,
        objective_formula=_objective_formula(diagnosis),
        total_margin=distribution.total_margin,
        components=ranked[:top_k],
        bad_push_components=bad_push,
        expected_support_components=expected_support,
        strengthen_components=strengthen,
        diagnostic_labels=tuple(label.label_name for label in diagnosis.diagnostic_labels),
        confidence_level=diagnosis.confidence_level,
        metadata={
            "source": "diagnosis-decision-boundary",
            "diagnosis_mode": diagnosis.metadata.get("mode", "unknown"),
            "grain": grain,
            "view_name": view_name,
            "top_k": top_k,
        },
    )


def decision_boundary_to_dict(artifact: DecisionBoundaryArtifact) -> dict[str, Any]:
    return {
        "metadata": dict(artifact.metadata),
        "target": {
            "target_id": artifact.target_id,
            "node_ids": list(artifact.target_node_ids),
        },
        "objective_formula": artifact.objective_formula,
        "total_margin": artifact.total_margin,
        "confidence_level": artifact.confidence_level,
        "components": [_component_to_dict(item) for item in artifact.components],
        "bad_push_components": [_component_to_dict(item) for item in artifact.bad_push_components],
        "expected_support_components": [_component_to_dict(item) for item in artifact.expected_support_components],
        "strengthen_components": [_component_to_dict(item) for item in artifact.strengthen_components],
        "diagnostic_labels": list(artifact.diagnostic_labels),
    }


def decision_boundary_to_markdown(artifact: DecisionBoundaryArtifact) -> str:
    lines = [
        "# Agent TraceGrad Decision Boundary",
        "",
        "## Summary",
        "",
        f"- target: `{artifact.target_id}` nodes=`{', '.join(artifact.target_node_ids)}`",
        f"- objective: `{artifact.objective_formula}`",
        f"- total_margin: {artifact.total_margin:.6g}",
        f"- confidence: `{artifact.confidence_level}`",
        "",
        "## Components Pushing Toward Bad",
        "",
    ]
    _append_component_table(lines, artifact.bad_push_components)
    lines.extend(["", "## Components Supporting Expected", ""])
    _append_component_table(lines, artifact.expected_support_components)
    lines.extend(["", "## Components To Strengthen", ""])
    _append_component_table(lines, artifact.strengthen_components)
    if artifact.diagnostic_labels:
        lines.extend(["", "## Diagnostic Labels", ""])
        for label in artifact.diagnostic_labels:
            lines.append(f"- `{label}`")
    return "\n".join(lines).rstrip() + "\n"


def write_decision_boundary_json(artifact: DecisionBoundaryArtifact, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision_boundary_to_dict(artifact), indent=2, ensure_ascii=False), encoding="utf-8")


def write_decision_boundary_markdown(artifact: DecisionBoundaryArtifact, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decision_boundary_to_markdown(artifact), encoding="utf-8")


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


def _rank_by_abs_margin(contributions: Sequence[MarginContribution]) -> tuple[MarginContribution, ...]:
    return tuple(sorted(contributions, key=lambda item: (-abs(item.margin), item.instance_id)))


def _component_from_margin(contribution: MarginContribution) -> BoundaryComponent:
    direction: BoundaryDirection = "neutral"
    if contribution.margin > 0.0:
        direction = "pushes_bad"
    elif contribution.margin < 0.0:
        direction = "pushes_expected"
    return BoundaryComponent(
        component_id=contribution.instance_id,
        sub_block_kind=contribution.sub_block_kind,
        node_ids=contribution.node_ids,
        direction=direction,
        classification=contribution.classification,
        margin=contribution.margin,
        bad_score=contribution.bad_score,
        expected_score=contribution.expected_score,
        classification_reason=contribution.classification_reason,
    )


def _objective_formula(diagnosis: DiagnosisResult) -> str:
    if diagnosis.contrastive_result is not None and diagnosis.contrastive_result.objective is not None:
        return str(target_objective_to_dict(diagnosis.contrastive_result.objective)["objective_formula"])
    if diagnosis.expected_result is not None:
        return "logp(expected_action)"
    return "logp(bad_action)"


def _component_to_dict(component: BoundaryComponent) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "sub_block_kind": component.sub_block_kind,
        "node_ids": list(component.node_ids),
        "direction": component.direction,
        "classification": component.classification,
        "margin": component.margin,
        "bad_score": component.bad_score,
        "expected_score": component.expected_score,
        "classification_reason": component.classification_reason,
    }


def _append_component_table(lines: list[str], components: Sequence[BoundaryComponent]) -> None:
    if not components:
        lines.append("_None._")
        return
    lines.append("| Component | Kind | Direction | Margin | Bad | Expected | Class | Reason |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | --- |")
    for item in components:
        lines.append(
            f"| `{item.component_id}` | `{item.sub_block_kind}` | `{item.direction}` | "
            f"{item.margin:.6g} | {item.bad_score:.6g} | {item.expected_score:.6g} | "
            f"`{item.classification}` | `{item.classification_reason}` |"
        )
