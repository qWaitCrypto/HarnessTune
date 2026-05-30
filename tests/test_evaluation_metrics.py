from __future__ import annotations

from agent_tracegrad.analysis import aggregate_attribution
from agent_tracegrad.attribution import AttributionResult
from agent_tracegrad.evaluation import (
    GroundTruthLabel,
    delta_ll_at_k,
    method_consistency,
    metrics_for_distribution,
    rank_correlation,
    recall_at_k,
)
from agent_tracegrad.trace import JsonTraceAdapter, TraceSerializer


class WhitespaceOffsetTokenizer:
    name_or_path = "whitespace-offset-tokenizer"

    def __call__(self, text: str, *, return_offsets_mapping: bool, add_special_tokens: bool) -> dict[str, object]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        offsets: list[tuple[int, int]] = []
        position = 0
        for part in text.split():
            start = text.index(part, position)
            end = start + len(part)
            offsets.append((start, end))
            position = end
        return {"offset_mapping": offsets}


def make_distribution():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys-1",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "policy text",
                "sequence_index": 0,
            },
            {
                "node_id": "user-1",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "user clue",
                "sequence_index": 1,
            },
            {
                "node_id": "agent-1",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "wrong answer",
                "sequence_index": 2,
            },
        ]
    )
    trace = TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes)
    result = AttributionResult(
        method_name="gradient_saliency",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="target-1",
        token_scores=(10.0, 9.0, 1.0, 1.0, 0.0, 0.0),
    )
    distributions = aggregate_attribution(trace, result)
    distribution = next(item for item in distributions if item.grain == "node" and item.view_name == "sum")
    return distribution


def test_recall_at_k_hits_labeled_node() -> None:
    distribution = make_distribution()
    ranking = tuple(sorted(distribution.instances, key=lambda item: (-item.views["sum"], item.instance_id)))
    from agent_tracegrad.analysis.ranking import rank_distribution

    ranked = rank_distribution(distribution)
    label = GroundTruthLabel(label_id="label-1", target_node_ids=("sys-1",), source="test")

    metric = recall_at_k(ranked, label, k=1)

    assert metric.value == 1.0


def test_rank_correlation_prefers_top_labeled_node() -> None:
    distribution = make_distribution()
    from agent_tracegrad.analysis.ranking import rank_distribution

    ranked = rank_distribution(distribution)
    label = GroundTruthLabel(label_id="label-1", target_node_ids=("sys-1",), source="test")

    metric = rank_correlation(ranked, label)

    assert metric.value > 0.0


def test_metrics_for_distribution_returns_recall_and_correlation() -> None:
    distribution = make_distribution()
    label = GroundTruthLabel(label_id="label-1", target_node_ids=("sys-1",), source="test")

    metrics = metrics_for_distribution(distribution, label, ks=(1, 2))

    assert [metric.metric_name for metric in metrics] == ["recall@k", "recall@k", "rank_correlation"]


def test_method_consistency_compares_shared_instance_ranks() -> None:
    distribution = make_distribution()
    from agent_tracegrad.analysis.ranking import rank_distribution

    ranked = rank_distribution(distribution)
    metric = method_consistency(
        ranked,
        ranked,
        left_method_name="gradient_saliency",
        right_method_name="gradient_times_input",
    )

    assert metric.value == 1.0


def test_delta_ll_at_k_reports_objective_loss_change() -> None:
    distribution = make_distribution()
    from agent_tracegrad.analysis.ranking import rank_distribution

    ranked = rank_distribution(distribution)

    metric = delta_ll_at_k(
        baseline_loss=-3.0,
        perturbed_loss=-1.5,
        ranking=ranked,
        k=1,
        label_id="label-1",
        objective_id="objective-1",
    )

    assert metric.metric_name == "delta_ll@k"
    assert metric.value == 1.5
    assert metric.metadata["selected_count"] == 1
    assert metric.metadata["objective_id"] == "objective-1"
