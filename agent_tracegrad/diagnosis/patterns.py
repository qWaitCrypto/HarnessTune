"""Diagnostic label detectors over structured diagnosis results."""

from __future__ import annotations

from typing import Protocol, Sequence

from agent_tracegrad.diagnosis.types import DiagnosisResult, DiagnosticLabel, MarginContribution


class DiagnosticPattern(Protocol):
    name: str

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        """Return a diagnostic label when the pattern is present."""


class ActionAffordanceGapDetector:
    name = "action_affordance_gap"

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        node_sum = _node_sum(result)
        if node_sum is None:
            return None
        positive = [item for item in node_sum.contributions if item.margin > 0.0]
        negative = [item for item in node_sum.contributions if item.margin < 0.0]
        if not positive:
            return None
        pos_total = sum(item.margin for item in positive)
        neg_total = abs(sum(item.margin for item in negative))
        if neg_total > 0.0 and pos_total < neg_total * 1.5:
            return None
        top = _rank_abs(positive)[:3]
        return DiagnosticLabel(
            label_name="action_affordance_gap",
            confidence=_label_confidence(result),
            summary="Bad-action path has stronger positive margin pressure than the expected-action path.",
            evidence=[
                f"{item.instance_id} margin={item.margin:.6g} kind={item.sub_block_kind}"
                for item in top
            ],
            recommendation="Narrow the bad-action path or add a concrete expected-action path with explicit precedence.",
            metadata={"positive_margin_total": pos_total, "negative_margin_total": neg_total},
        )


class SchemaMagnetismDetector:
    name = "schema_magnetism"

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        node_sum = _node_sum(result)
        if node_sum is None:
            return None
        schema = [
            item
            for item in node_sum.contributions
            if item.sub_block_kind == "system.tool_schema" and item.margin > 0.0
        ]
        if not schema:
            return None
        ranked = _rank_abs(schema)
        top = ranked[0]
        if top.margin < _margin_threshold(node_sum.contributions):
            return None
        return DiagnosticLabel(
            label_name="schema_magnetism",
            confidence=_label_confidence(result),
            summary="A tool schema contributes positive bad-vs-good margin and may be acting as an action magnet.",
            evidence=[
                f"{item.instance_id} margin={item.margin:.6g}"
                for item in ranked[:3]
            ],
            recommendation="Review tool name, description, examples, and negative constraints for overly broad action affordance.",
            metadata={"top_schema_node": top.instance_id, "top_schema_margin": top.margin},
        )


class DangerousActionMagnetDetector:
    name = "dangerous_action_magnet"

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        node_sum = _node_sum(result)
        if node_sum is None:
            return None
        candidates = [
            item
            for item in node_sum.contributions
            if item.margin > 0.0 and _has_action_affordance(result, item)
        ]
        if not candidates:
            return None
        top = _rank_abs(candidates)[0]
        if top.margin < _margin_threshold(node_sum.contributions):
            return None
        return DiagnosticLabel(
            label_name="dangerous_action_magnet",
            confidence=_label_confidence(result),
            summary="A concrete action path has both positive margin pressure and static action affordance features.",
            evidence=[
                f"{top.instance_id} margin={top.margin:.6g} kind={top.sub_block_kind}",
                "matched action affordance tokens in component text",
            ],
            recommendation="Constrain this action path with explicit activation conditions and negative constraints.",
            metadata={"component": top.instance_id, "margin": top.margin},
        )


class WeakRefusalAffordanceDetector:
    name = "weak_refusal_affordance"

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        if result.expected_result is None:
            return None
        expected = result.expected_result.objective.expected_target
        if expected is None:
            return None
        text = expected.content.lower()
        refusal_like = any(term in text for term in ("can't", "cannot", "unable", "sorry", "refuse", "not able"))
        executable = any(term in text for term in ("call_", "use ", "tool", "first ", "then "))
        if not refusal_like or executable:
            return None
        node_sum = _node_sum(result)
        if node_sum is None:
            return None
        expected_support = sum(abs(item.margin) for item in node_sum.contributions if item.margin < 0.0)
        bad_pressure = sum(item.margin for item in node_sum.contributions if item.margin > 0.0)
        if expected_support > bad_pressure:
            return None
        return DiagnosticLabel(
            label_name="weak_refusal_affordance",
            confidence="weak",
            summary="Expected refusal is present as text but lacks a concrete executable action path.",
            evidence=[
                "expected target is refusal-like text",
                f"expected-support margin={expected_support:.6g}, bad-pressure margin={bad_pressure:.6g}",
            ],
            recommendation="Add an explicit refusal template, prohibited tools, and precedence over fallback/transfer actions.",
            metadata={"expected_target_id": expected.target_id},
        )


class PrecedenceInversionDetector:
    name = "precedence_inversion"

    def detect(self, result: DiagnosisResult) -> DiagnosticLabel | None:
        node_sum = _node_sum(result)
        if node_sum is None:
            return None
        system_instruction = [
            item for item in node_sum.contributions if item.sub_block_kind == "system.instruction"
        ]
        positive = _rank_abs([item for item in system_instruction if item.margin > 0.0])
        negative = _rank_abs([item for item in system_instruction if item.margin < 0.0])
        if not positive or not negative:
            return None
        if positive[0].margin < abs(negative[0].margin):
            return None
        return DiagnosticLabel(
            label_name="precedence_inversion",
            confidence=_label_confidence(result),
            summary="System instruction margin suggests a bad-action rule is dominating an expected-action rule.",
            evidence=[
                f"bad pressure: {positive[0].instance_id} margin={positive[0].margin:.6g}",
                f"good support: {negative[0].instance_id} margin={negative[0].margin:.6g}",
            ],
            recommendation="Make the expected-action rule explicitly override the fallback or bad-action rule.",
            metadata={
                "positive_instruction_margin": positive[0].margin,
                "negative_instruction_margin": negative[0].margin,
            },
        )


DEFAULT_DIAGNOSTIC_PATTERNS: tuple[DiagnosticPattern, ...] = (
    ActionAffordanceGapDetector(),
    SchemaMagnetismDetector(),
    DangerousActionMagnetDetector(),
    WeakRefusalAffordanceDetector(),
    PrecedenceInversionDetector(),
)


def detect_diagnostic_labels(
    result: DiagnosisResult,
    *,
    patterns: Sequence[DiagnosticPattern] = DEFAULT_DIAGNOSTIC_PATTERNS,
) -> tuple[DiagnosticLabel, ...]:
    labels: list[DiagnosticLabel] = []
    for pattern in patterns:
        label = pattern.detect(result)
        if label is not None:
            labels.append(label)
    return tuple(labels)


def diagnostic_label_to_dict(label: DiagnosticLabel) -> dict[str, object]:
    return {
        "label_name": label.label_name,
        "confidence": label.confidence,
        "summary": label.summary,
        "evidence": list(label.evidence),
        "recommendation": label.recommendation,
        "metadata": dict(label.metadata),
    }


def _node_sum(result: DiagnosisResult):
    for distribution in result.margin_distributions:
        if distribution.grain == "node" and distribution.view_name == "sum":
            return distribution
    return None


def _rank_abs(contributions: Sequence[MarginContribution]) -> tuple[MarginContribution, ...]:
    return tuple(sorted(contributions, key=lambda item: (-abs(item.margin), item.instance_id)))


def _margin_threshold(contributions: Sequence[MarginContribution]) -> float:
    max_abs = max((abs(item.margin) for item in contributions), default=0.0)
    return max_abs * 0.2


def _label_confidence(result: DiagnosisResult):
    if result.confidence_level == "strong":
        return "strong"
    if result.ablations:
        return "medium"
    return "weak"


def _has_action_affordance(result: DiagnosisResult, contribution: MarginContribution) -> bool:
    action_terms = (
        "tool",
        "call",
        "use",
        "transfer",
        "first",
        "then",
        "must",
        "should",
        "template",
    )
    for node_id in contribution.node_ids:
        node = result.contrastive_result.trace.nodes.get(node_id) if result.contrastive_result is not None else None
        if node is None:
            continue
        text = node.content.lower()
        if any(term in text for term in action_terms):
            return True
    return False
