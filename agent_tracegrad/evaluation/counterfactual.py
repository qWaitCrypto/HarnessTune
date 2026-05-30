"""Counterfactual likelihood-drop metrics for attribution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.objective import TargetObjective
from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class DeltaLLPoint:
    k: int
    selected_token_indexes: Sequence[int]
    baseline_loss: float
    masked_loss: float
    delta_loss: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_token_indexes", tuple(self.selected_token_indexes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def delta_ll_curve(
    trace: SerializedTrace,
    attribution: AttributionResult,
    objective: TargetObjective,
    model: ModelAdapter,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> tuple[DeltaLLPoint, ...]:
    """Mask top-k attribution input embeddings and re-score the same objective."""

    baseline = objective_loss(trace, objective, model)
    points: list[DeltaLLPoint] = []
    for k in _normalize_positive_ks(ks):
        selected = _top_token_indexes(trace, attribution, k=k)
        if not selected:
            continue
        masked = objective_loss(trace, objective, model, masked_token_indexes=selected)
        points.append(
            DeltaLLPoint(
                k=k,
                selected_token_indexes=selected,
                baseline_loss=baseline,
                masked_loss=masked,
                delta_loss=masked - baseline,
                metadata={
                    "objective_id": objective.objective_id,
                    "objective_type": objective.objective_type,
                },
            )
        )
    return tuple(points)


def objective_loss(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    *,
    masked_token_indexes: Sequence[int] = (),
) -> float:
    """Return the objective log-likelihood score used by attribution."""

    objective.validate_against_trace(trace)
    masked = tuple(masked_token_indexes)
    if objective.objective_type in {"expected_action", "contrastive"} and objective.bad_target is not None:
        return _anchored_objective_loss(trace, objective, model, masked)
    return _plain_objective_loss(trace, objective, model, masked)


def _plain_objective_loss(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    masked_token_indexes: Sequence[int],
) -> float:
    from agent_tracegrad.attribution.gradient import _build_objective_input, _objective_loss

    objective_input = _build_objective_input(trace, objective)
    tokenized = model.tokenize(objective_input.text)
    inputs_embeds = model.input_embeddings(tokenized.input_ids, requires_grad=False)
    _mask_positions(inputs_embeds, masked_token_indexes)
    output = model.forward(inputs_embeds, tokenized.attention_mask)
    loss = _objective_loss(output.logits, tokenized.input_ids, objective_input)
    return float(loss.detach().cpu())


def _anchored_objective_loss(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    masked_token_indexes: Sequence[int],
) -> float:
    from agent_tracegrad.attribution.gradient import _build_anchored_objective_input, _positions_logprob_loss

    objective_input = _build_anchored_objective_input(trace, objective, model)
    if objective.expected_target is None:
        raise ValueError("anchored objective requires expected target")
    masked = tuple(index for index in masked_token_indexes if index < objective_input.prefix_token_count)
    expected_score = _branch_score(
        model,
        objective_input.expected_text,
        start_token=objective_input.prefix_token_count,
        masked_token_indexes=masked,
        score_fn=_positions_logprob_loss,
    )
    if objective.objective_type == "expected_action":
        return expected_score
    if objective.objective_type != "contrastive":
        raise ValueError("anchored objective only supports expected_action or contrastive")
    bad_score = _branch_score(
        model,
        objective_input.bad_text,
        start_token=objective_input.prefix_token_count,
        masked_token_indexes=masked,
        score_fn=_positions_logprob_loss,
    )
    return bad_score - expected_score


def _branch_score(
    model: ModelAdapter,
    text: str,
    *,
    start_token: int,
    masked_token_indexes: Sequence[int],
    score_fn,
) -> float:
    tokenized = model.tokenize(text)
    inputs_embeds = model.input_embeddings(tokenized.input_ids, requires_grad=False)
    _mask_positions(inputs_embeds, masked_token_indexes)
    output = model.forward(inputs_embeds, tokenized.attention_mask)
    score = score_fn(output.logits, tokenized.input_ids, tuple(range(start_token, tokenized.input_ids.shape[1])))
    if score is None:
        raise ValueError("objective branch must contain at least one scored token")
    return float(score.detach().cpu())


def _mask_positions(inputs_embeds, indexes: Sequence[int]) -> None:
    for index in indexes:
        if 0 <= index < inputs_embeds.shape[1]:
            inputs_embeds[:, index, :] = 0.0


def _top_token_indexes(
    trace: SerializedTrace,
    attribution: AttributionResult,
    *,
    k: int,
) -> tuple[int, ...]:
    allowed = _allowed_token_indexes(trace)
    ranked = sorted(
        ((index, abs(score)) for index, score in enumerate(attribution.token_scores) if index in allowed),
        key=lambda item: (-item[1], item[0]),
    )
    selected = tuple(index for index, _score in ranked[:k])
    _assert_non_agent_positions(trace, selected)
    return selected


def _allowed_token_indexes(trace: SerializedTrace) -> set[int]:
    allowed: set[int] = set()
    for span in trace.spans:
        if span.block_role in {"system", "user"}:
            allowed.update(range(span.start_token, span.end_token))
    return allowed


def _assert_non_agent_positions(trace: SerializedTrace, indexes: Sequence[int]) -> None:
    agent_positions: set[int] = set()
    for span in trace.spans:
        if span.block_role == "agent":
            agent_positions.update(range(span.start_token, span.end_token))
    if agent_positions.intersection(indexes):
        raise ValueError("counterfactual mask selected an agent token")


def _normalize_positive_ks(ks: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(k) for k in ks if int(k) > 0}))
