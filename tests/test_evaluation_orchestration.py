from __future__ import annotations

from agent_tracegrad.evaluation import generate_evaluation_context
from agent_tracegrad.target import ExpectedTarget, TargetObjective

from tests.test_single_trace_analysis import WhitespaceOffsetTokenizer, make_raw_trace
from tests.test_trace_ingestion import make_agentpi_raw


def test_generate_evaluation_context_for_normalized_trace_with_manual_target() -> None:
    context = generate_evaluation_context(
        make_raw_trace(),
        tokenizer=WhitespaceOffsetTokenizer(),
        input_format="json-fixture",
        target_node_ids=("agent-1",),
        operator_configs=(
            {
                "operator": "replace_with_placeholder",
                "parameters": {"placeholder": "masked"},
            },
        ),
    )

    assert context.trace.metadata["trace_adapter"] == "json-fixture"
    assert context.targets[0].node_ids == ("agent-1",)
    assert context.objective.objective_type == "bad_action"
    assert context.objective.bad_target == context.targets[0]
    assert len(context.samples) == 2
    assert [sample.spec.target_node_ids for sample in context.samples] == [("sys-1",), ("user-1",)]


def test_generate_evaluation_context_for_agentpi_raw_with_marker() -> None:
    context = generate_evaluation_context(
        make_agentpi_raw(),
        tokenizer=WhitespaceOffsetTokenizer(),
        input_format="agentpi-raw",
        target_marker="last-agent-output",
        target_id="agentpi-target",
        operator_configs=(
            {
                "operator": "inject_unrelated_content",
                "parameters": {"content": "noise"},
            },
        ),
        max_samples=3,
    )

    assert context.trace.metadata["trace_adapter"] == "agentpi-raw"
    assert context.objective.objective_id == "agentpi-target"
    assert context.targets[0].target_id == "agentpi-target"
    assert context.targets[0].node_ids == ("agentpi:message-3:assistant-content",)
    assert len(context.samples) == 3
    assert all(
        context.trace.nodes[node_id].block_role in {"system", "user"}
        for sample in context.samples
        for node_id in sample.spec.target_node_ids
    )


def test_generate_evaluation_context_resolves_contrastive_marker_bad_target() -> None:
    objective = TargetObjective(
        objective_id="bad-vs-good",
        objective_type="contrastive",
        expected_target=ExpectedTarget(target_id="gold", content="cannot proceed"),
        metadata={"requires_resolved_bad_target": True},
    )

    context = generate_evaluation_context(
        make_agentpi_raw(),
        tokenizer=WhitespaceOffsetTokenizer(),
        input_format="agentpi-raw",
        target_marker="last-agent-output",
        target_id="resolved-bad",
        objective=objective,
        operator_configs=(
            {
                "operator": "replace_with_placeholder",
                "parameters": {"placeholder": "masked"},
            },
        ),
        max_samples=1,
    )

    assert context.objective.objective_type == "contrastive"
    assert context.objective.bad_target == context.targets[0]
    assert context.objective.bad_target.target_id == "resolved-bad"
    assert context.objective.expected_target.target_id == "gold"  # type: ignore[union-attr]


def test_generate_evaluation_context_rejects_marker_without_targets() -> None:
    raw_trace = make_agentpi_raw()
    messages = raw_trace["simulation"]["messages"]  # type: ignore[index]
    messages[1]["content"] = None  # type: ignore[index]
    messages[1]["tool_calls"] = None  # type: ignore[index]
    messages[3]["role"] = "user"  # type: ignore[index]

    try:
        generate_evaluation_context(
            raw_trace,
            tokenizer=WhitespaceOffsetTokenizer(),
            input_format="agentpi-raw",
            target_marker="last-agent-output",
            operator_configs=(
                {
                    "operator": "replace_with_placeholder",
                    "parameters": {"placeholder": "masked"},
                },
            ),
        )
    except ValueError as exc:
        assert "did not produce any targets" in str(exc)
    else:
        raise AssertionError("expected marker resolution to fail when no agent output exists")
