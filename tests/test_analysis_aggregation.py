from __future__ import annotations

import math

import pytest

from agent_tracegrad.analysis import aggregate_attribution, rank_distribution
from agent_tracegrad.attribution import AttributionResult
from agent_tracegrad.trace import SpanMetadata, SerializedTrace, TraceNode


def make_trace() -> SerializedTrace:
    nodes = {
        "sys-1": TraceNode("sys-1", "system", "system.instruction", "a b"),
        "sys-2": TraceNode("sys-2", "system", "system.instruction", "c"),
        "user-1": TraceNode("user-1", "user", "user.content", "d e"),
        "agent-1": TraceNode("agent-1", "agent", "agent.content", "f g"),
    }
    return SerializedTrace(
        nodes=nodes,
        serialized_text="a b c d e f g",
        spans=(
            SpanMetadata("span-sys-1", "sys-1", "system", "system.instruction", 0, 2),
            SpanMetadata("span-sys-2", "sys-2", "system", "system.instruction", 2, 3),
            SpanMetadata("span-user-1", "user-1", "user", "user.content", 3, 5),
            SpanMetadata("span-agent-1", "agent-1", "agent", "agent.content", 5, 7),
        ),
        tokenizer_name="test-tokenizer",
    )


def make_result() -> AttributionResult:
    return AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(1.0, 2.0, 4.0, 8.0, 16.0, 0.0, 0.0),
    )


def by_key(distributions, *, grain: str, view_name: str):
    return next(item for item in distributions if item.grain == grain and item.view_name == view_name)


def test_aggregate_attribution_builds_node_and_kind_distributions_without_agent_instances() -> None:
    distributions = aggregate_attribution(make_trace(), make_result(), topk=2)

    assert len(distributions) == 18
    node_sum = by_key(distributions, grain="node", view_name="sum")
    assert [item.instance_id for item in node_sum.instances] == ["sys-1", "sys-2", "user-1"]
    assert all(item.block_role in {"system", "user"} for item in node_sum.instances)
    assert all(not item.sub_block_kind.startswith("agent.") for item in node_sum.instances)
    assert {item.instance_id: item.views["sum"] for item in node_sum.instances} == {
        "sys-1": 3.0,
        "sys-2": 4.0,
        "user-1": 24.0,
    }


def test_kind_sum_equals_member_node_sums_for_each_sub_block_kind() -> None:
    distributions = aggregate_attribution(make_trace(), make_result())
    node_sum = by_key(distributions, grain="node", view_name="sum")
    kind_sum = by_key(distributions, grain="sub_block_kind", view_name="sum")
    node_total_by_kind: dict[str, float] = {}
    for instance in node_sum.instances:
        node_total_by_kind[instance.sub_block_kind] = node_total_by_kind.get(instance.sub_block_kind, 0.0) + instance.views[
            "sum"
        ]

    assert {item.instance_id: item.views["sum"] for item in kind_sum.instances} == node_total_by_kind
    assert {item.instance_id: item.node_ids for item in kind_sum.instances} == {
        "system.instruction": ("sys-1", "sys-2"),
        "user.content": ("user-1",),
    }


def test_aggregation_views_and_distribution_stats_are_computed() -> None:
    distributions = aggregate_attribution(make_trace(), make_result(), topk=1)
    node_mean = by_key(distributions, grain="node", view_name="mean")
    sys_1 = next(item for item in node_mean.instances if item.instance_id == "sys-1")

    assert sys_1.token_count == 2
    assert sys_1.views["mean"] == 1.5
    assert sys_1.views["topk_mean"] == 2.0
    assert sys_1.views["length_norm"] == pytest.approx(3.0 / math.log(3.0))
    assert node_mean.distribution_stats["top1_mass"] == pytest.approx(12.0 / 17.5)
    assert node_mean.distribution_stats["top3_mass"] == pytest.approx(1.0)


def test_rank_distribution_sorts_by_selected_view_then_instance_id() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0),
    )
    distributions = aggregate_attribution(trace, result)
    node_sum = by_key(distributions, grain="node", view_name="sum")

    ranked = rank_distribution(node_sum)

    assert [(item.rank, item.instance.instance_id, item.score) for item in ranked] == [
        (1, "sys-1", 3.0),
        (2, "sys-2", 3.0),
        (3, "user-1", 3.0),
    ]


def test_signed_aggregation_views_preserve_direction_and_abs_strength() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(-5.0, -2.0, 4.0, 1.0, -3.0, 0.0, 0.0),
    )

    distributions = aggregate_attribution(trace, result, topk=1)
    node_sum = by_key(distributions, grain="node", view_name="sum")
    sys_1 = next(item for item in node_sum.instances if item.instance_id == "sys-1")
    user_1 = next(item for item in node_sum.instances if item.instance_id == "user-1")

    assert sys_1.views["net_sum"] == -7.0
    assert sys_1.views["positive_sum"] == 0.0
    assert sys_1.views["negative_sum"] == -7.0
    assert sys_1.views["abs_sum"] == 7.0
    assert sys_1.views["topk_abs_mean"] == 5.0
    assert user_1.views["net_sum"] == -2.0
    assert user_1.views["positive_sum"] == 1.0
    assert user_1.views["negative_sum"] == -3.0


def test_signed_distribution_stats_use_abs_mass_without_nan() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(-5.0, -2.0, 4.0, 1.0, -3.0, 0.0, 0.0),
    )

    node_sum = by_key(aggregate_attribution(trace, result), grain="node", view_name="sum")

    assert not math.isnan(node_sum.distribution_stats["entropy"])
    assert node_sum.distribution_stats["positive_mass"] == pytest.approx(4.0 / 13.0)
    assert node_sum.distribution_stats["negative_mass"] == pytest.approx(9.0 / 13.0)
    assert node_sum.distribution_stats["net_direction"] == pytest.approx(-5.0 / 13.0)


def test_rank_distribution_can_rank_by_abs_positive_or_negative() -> None:
    trace = make_trace()
    result = AttributionResult(
        method_name="gradient",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(-5.0, -2.0, 4.0, 1.0, -3.0, 0.0, 0.0),
    )
    node_sum = by_key(aggregate_attribution(trace, result), grain="node", view_name="sum")

    assert [item.instance.instance_id for item in rank_distribution(node_sum, rank_by="abs")] == [
        "sys-1",
        "sys-2",
        "user-1",
    ]
    assert [item.instance.instance_id for item in rank_distribution(node_sum, rank_by="positive")] == [
        "sys-2",
        "sys-1",
        "user-1",
    ]
    assert [item.instance.instance_id for item in rank_distribution(node_sum, rank_by="negative")] == [
        "sys-1",
        "user-1",
        "sys-2",
    ]
