"""Trace-level perturbation source for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel
from agent_tracegrad.evaluation.perturbation.operators import get_operator
from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.trace.schema import SerializedTrace, TraceNode
from agent_tracegrad.trace.serializer import TraceSerializer


@dataclass(frozen=True)
class TraceLevelPerturbation:
    trace: SerializedTrace
    label: GroundTruthLabel


def apply_trace_level_perturbation(
    trace: SerializedTrace,
    spec: PerturbationSpec,
    serializer: TraceSerializer,
) -> TraceLevelPerturbation:
    spec.validate_against_trace(trace)
    operator = get_operator(spec.operator)
    target_node_ids = set(spec.target_node_ids)
    perturbed_nodes: list[TraceNode] = []

    for node in sorted(trace.nodes.values(), key=_node_sort_key):
        if node.node_id not in target_node_ids:
            perturbed_nodes.append(node)
            continue
        perturbed_nodes.append(_replace_content(node, operator(node, spec.parameters, serializer.tokenizer), spec))

    perturbed_trace = serializer.serialize(
        perturbed_nodes,
        metadata={
            **trace.metadata,
            "perturbation": {
                "operator": spec.operator,
                "target_node_ids": tuple(spec.target_node_ids),
                "parameters": dict(spec.parameters),
                "source": "trace-level-perturbation",
            },
        },
    )
    return TraceLevelPerturbation(
        trace=perturbed_trace,
        label=GroundTruthLabel(
            label_id=f"trace-level:{spec.operator}:{','.join(spec.target_node_ids)}",
            target_node_ids=spec.target_node_ids,
            source="trace-level-perturbation",
            metadata={
                "operator": spec.operator,
                "parameters": dict(spec.parameters),
            },
        ),
    )


def _replace_content(node: TraceNode, content: str, spec: PerturbationSpec) -> TraceNode:
    metadata = {
        **node.metadata,
        "perturbation": {
            "operator": spec.operator,
            "original_content": node.content,
            "parameters": dict(spec.parameters),
        },
    }
    return TraceNode(
        node_id=node.node_id,
        block_role=node.block_role,
        sub_block_kind=node.sub_block_kind,
        content=content,
        metadata=metadata,
        sequence_index=node.sequence_index,
        timestamp=node.timestamp,
        parents=node.parents,
    )


def _node_sort_key(node: TraceNode) -> tuple[int, str]:
    sequence_index = node.sequence_index if node.sequence_index is not None else 0
    return sequence_index, node.node_id
