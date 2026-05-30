"""Tests for diagnostic label detectors."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from agent_tracegrad.diagnosis.patterns import detect_diagnostic_labels
from agent_tracegrad.diagnosis.types import DiagnosisResult, MarginContribution, MarginDistribution
from agent_tracegrad.target.objective import ExpectedTarget, TargetObjective
from agent_tracegrad.trace.schema import SerializedTrace, TraceNode


@dataclass(frozen=True)
class _Analysis:
    trace: SerializedTrace
    objective: TargetObjective | None = None


def _trace() -> SerializedTrace:
    nodes = {
        "policy": TraceNode(
            node_id="policy",
            block_role="system",
            sub_block_kind="system.instruction",
            content="To transfer, first call transfer_to_human_agents, then send the template.",
        ),
        "schema": TraceNode(
            node_id="schema",
            block_role="system",
            sub_block_kind="system.tool_schema",
            content="tool name transfer_to_human_agents description transfer user when needed",
        ),
    }
    return SerializedTrace(
        nodes=nodes,
        serialized_text="",
        spans=(),
        tokenizer_name="test-tokenizer",
    )


def _result(*, expected_text: str = "I am sorry, I cannot do that.") -> DiagnosisResult:
    trace = _trace()
    expected = ExpectedTarget(target_id="expected", content=expected_text)
    objective = TargetObjective.expected_action(expected)
    return DiagnosisResult(
        bad_result=SimpleNamespace(),
        expected_result=_Analysis(trace=trace, objective=objective),
        contrastive_result=_Analysis(trace=trace),
        margin_distributions=(
            MarginDistribution(
                grain="node",
                view_name="sum",
                total_margin=3.0,
                contributions=(
                    MarginContribution(
                        instance_id="policy",
                        block_role="system",
                        sub_block_kind="system.instruction",
                        node_ids=("policy",),
                        bad_score=3.0,
                        expected_score=0.2,
                        margin=2.0,
                        classification="narrow",
                    ),
                    MarginContribution(
                        instance_id="schema",
                        block_role="system",
                        sub_block_kind="system.tool_schema",
                        node_ids=("schema",),
                        bad_score=2.0,
                        expected_score=0.1,
                        margin=1.0,
                        classification="narrow",
                    ),
                ),
            ),
        ),
        confidence_level="medium",
    )


def test_detects_initial_diagnostic_labels() -> None:
    labels = detect_diagnostic_labels(_result())
    names = {label.label_name for label in labels}

    assert "action_affordance_gap" in names
    assert "schema_magnetism" in names
    assert "dangerous_action_magnet" in names
    assert "weak_refusal_affordance" in names


def test_expected_executable_action_does_not_trigger_weak_refusal_label() -> None:
    labels = detect_diagnostic_labels(_result(expected_text="First call the approved refusal tool."))
    names = {label.label_name for label in labels}

    assert "weak_refusal_affordance" not in names
