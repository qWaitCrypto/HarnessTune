"""Diagnosis result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from agent_tracegrad.analysis.single_trace import SingleTraceAnalysisResult

ComponentClassification = Literal["preserve", "narrow", "strengthen"]
ConfidenceLevel = Literal["weak", "medium", "strong"]
DiagnosticLabelName = Literal[
    "action_affordance_gap",
    "schema_magnetism",
    "dangerous_action_magnet",
    "weak_refusal_affordance",
    "precedence_inversion",
]


@dataclass(frozen=True)
class MarginContribution:
    instance_id: str
    block_role: str
    sub_block_kind: str
    node_ids: Sequence[str]
    bad_score: float
    expected_score: float
    margin: float
    classification: ComponentClassification
    classification_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))


@dataclass(frozen=True)
class MarginDistribution:
    grain: str
    view_name: str
    contributions: Sequence[MarginContribution]
    total_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "contributions", tuple(self.contributions))


@dataclass(frozen=True)
class DiagnosisEvidence:
    objective_name: str
    report: Any


@dataclass(frozen=True)
class DiagnosisAblation:
    ablation_type: str
    k: int
    target_node_ids: Sequence[str]
    baseline_loss: float
    ablated_loss: float
    delta_loss: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_node_ids", tuple(self.target_node_ids))


@dataclass(frozen=True)
class DiagnosticLabel:
    label_name: DiagnosticLabelName
    confidence: ConfidenceLevel
    summary: str
    evidence: Sequence[str]
    recommendation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class DiagnosisResult:
    bad_result: SingleTraceAnalysisResult
    expected_result: SingleTraceAnalysisResult | None
    contrastive_result: SingleTraceAnalysisResult | None
    margin_distributions: Sequence[MarginDistribution]
    evidence: Sequence[DiagnosisEvidence] = ()
    ablations: Sequence[DiagnosisAblation] = ()
    diagnostic_labels: Sequence[DiagnosticLabel] = ()
    confidence_level: ConfidenceLevel = "weak"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "margin_distributions", tuple(self.margin_distributions))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "ablations", tuple(self.ablations))
        object.__setattr__(self, "diagnostic_labels", tuple(self.diagnostic_labels))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
