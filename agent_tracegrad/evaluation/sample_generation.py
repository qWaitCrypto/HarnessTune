"""Generate trace-level evaluation samples from canonical traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.evaluation.perturbation.trace_level import (
    TraceLevelPerturbation,
    apply_trace_level_perturbation,
)
from agent_tracegrad.evaluation.spec import PerturbationSpec
from agent_tracegrad.trace.schema import SerializedTrace, TraceNode
from agent_tracegrad.trace.serializer import TraceSerializer


@dataclass(frozen=True)
class TraceLevelSample:
    spec: PerturbationSpec
    perturbation: TraceLevelPerturbation
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def generate_trace_level_samples(
    trace: SerializedTrace,
    serializer: TraceSerializer,
    *,
    operator_configs: Sequence[Mapping[str, Any]],
    max_samples: int | None = None,
) -> tuple[TraceLevelSample, ...]:
    """Generate perturbation samples over system/user nodes in a canonical trace."""

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    samples: list[TraceLevelSample] = []
    candidate_nodes = tuple(_perturbable_nodes(trace))

    for config in operator_configs:
        operator = _operator_name(config)
        for target_node_ids, parameters in _specs_for_operator(operator, config, candidate_nodes):
            spec = PerturbationSpec(
                operator=operator,
                target_node_ids=target_node_ids,
                parameters=parameters,
            )
            samples.append(
                TraceLevelSample(
                    spec=spec,
                    perturbation=apply_trace_level_perturbation(trace, spec, serializer),
                    metadata={
                        "source": "trace-level-sample-generation",
                        "operator": operator,
                    },
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                return tuple(samples)
    return tuple(samples)


def _perturbable_nodes(trace: SerializedTrace) -> list[TraceNode]:
    return [
        node
        for node in sorted(trace.nodes.values(), key=_node_sort_key)
        if node.block_role in {"system", "user"} and node.content
    ]


def _specs_for_operator(
    operator: str,
    config: Mapping[str, Any],
    candidate_nodes: Sequence[TraceNode],
) -> tuple[tuple[tuple[str, ...], Mapping[str, Any]], ...]:
    if operator == "swap_between_instances":
        return _swap_specs(config, candidate_nodes)
    parameters = dict(config.get("parameters") or {})
    target_node_ids = config.get("target_node_ids")
    if target_node_ids is not None:
        return ((tuple(str(node_id) for node_id in target_node_ids), parameters),)
    return tuple(((node.node_id,), parameters) for node in candidate_nodes)


def _swap_specs(
    config: Mapping[str, Any],
    candidate_nodes: Sequence[TraceNode],
) -> tuple[tuple[tuple[str, ...], Mapping[str, Any]], ...]:
    target_node_ids = config.get("target_node_ids")
    pairs: list[tuple[TraceNode, TraceNode]] = []
    if target_node_ids is not None:
        node_ids = tuple(str(node_id) for node_id in target_node_ids)
        if len(node_ids) != 2:
            raise ValueError("swap_between_instances target_node_ids must contain exactly two node ids")
        nodes_by_id = {node.node_id: node for node in candidate_nodes}
        pairs.append((nodes_by_id[node_ids[0]], nodes_by_id[node_ids[1]]))
    else:
        by_kind: dict[str, list[TraceNode]] = {}
        for node in candidate_nodes:
            by_kind.setdefault(node.sub_block_kind, []).append(node)
        for nodes in by_kind.values():
            if len(nodes) >= 2:
                pairs.append((nodes[0], nodes[1]))

    specs: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    for first, second in pairs:
        specs.append(
            (
                (first.node_id, second.node_id),
                {
                    **dict(config.get("parameters") or {}),
                    "replacements": {
                        first.node_id: second.content,
                        second.node_id: first.content,
                    },
                },
            )
        )
    return tuple(specs)


def _operator_name(config: Mapping[str, Any]) -> str:
    operator = config.get("operator")
    if not isinstance(operator, str) or not operator:
        raise ValueError("operator config requires non-empty string field 'operator'")
    return operator


def _node_sort_key(node: TraceNode) -> tuple[int, str]:
    sequence_index = node.sequence_index if node.sequence_index is not None else 0
    return sequence_index, node.node_id
