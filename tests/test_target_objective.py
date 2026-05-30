from __future__ import annotations

import pytest

from agent_tracegrad.target import ExpectedTarget, FailureTarget, TargetObjective, target_objective_to_dict
from tests.test_core_contracts import make_trace


def test_bad_action_objective_wraps_failure_target() -> None:
    target = FailureTarget(target_id="bad-transfer", node_ids=("agent-1",))
    objective = TargetObjective.bad_action(target)

    assert objective.objective_id == "bad-transfer"
    assert objective.objective_type == "bad_action"
    objective.validate_against_trace(make_trace())


def test_expected_action_objective_wraps_expected_content() -> None:
    expected = ExpectedTarget(target_id="gold-refusal", content="I cannot cancel this reservation.", source="human")

    objective = TargetObjective.expected_action(expected)

    assert objective.objective_type == "expected_action"
    assert objective.expected_target == expected
    objective.validate_against_trace(make_trace())


def test_contrastive_objective_requires_bad_and_expected_targets() -> None:
    bad = FailureTarget(target_id="bad-transfer", node_ids=("agent-1",))
    expected = ExpectedTarget(target_id="gold-refusal", content="I cannot cancel this reservation.")

    objective = TargetObjective.contrastive(bad, expected)

    assert objective.objective_type == "contrastive"
    assert objective.bad_target == bad
    assert objective.expected_target == expected
    payload = target_objective_to_dict(objective)
    assert payload["objective_formula"] == "log P(bad_target | context) - log P(expected_target | context)"


def test_target_objective_rejects_missing_parts() -> None:
    with pytest.raises(ValueError, match="requires expected_target"):
        TargetObjective(objective_id="bad", objective_type="expected_action")
