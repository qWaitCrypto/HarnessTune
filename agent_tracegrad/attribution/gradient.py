"""Real gradient attribution methods over input embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.schema import FailureTarget
from agent_tracegrad.trace.schema import SerializedTrace


@dataclass(frozen=True)
class GradientSaliencyAttribution:
    execution_model_name: str | None = None

    name: str = "gradient_saliency"

    def attribute(
        self,
        trace: SerializedTrace,
        target: FailureTarget,
        model: ModelAdapter,
        *,
        contrastive_target: FailureTarget | None = None,
    ) -> AttributionResult:
        if contrastive_target is not None:
            raise NotImplementedError("contrastive attribution is not implemented yet")
        return _attribute_with_gradients(trace, target, model, self.name, self.execution_model_name, mode="gradient")


@dataclass(frozen=True)
class GradientTimesInputAttribution:
    execution_model_name: str | None = None

    name: str = "gradient_times_input"

    def attribute(
        self,
        trace: SerializedTrace,
        target: FailureTarget,
        model: ModelAdapter,
        *,
        contrastive_target: FailureTarget | None = None,
    ) -> AttributionResult:
        if contrastive_target is not None:
            raise NotImplementedError("contrastive attribution is not implemented yet")
        return _attribute_with_gradients(
            trace,
            target,
            model,
            self.name,
            self.execution_model_name,
            mode="gradient_times_input",
        )


@dataclass(frozen=True)
class IntegratedGradientsAttribution:
    execution_model_name: str | None = None
    steps: int = 16

    name: str = "integrated_gradients"

    def attribute(
        self,
        trace: SerializedTrace,
        target: FailureTarget,
        model: ModelAdapter,
        *,
        contrastive_target: FailureTarget | None = None,
    ) -> AttributionResult:
        if contrastive_target is not None:
            raise NotImplementedError("contrastive attribution is not implemented yet")
        if self.steps < 1:
            raise ValueError("integrated gradients steps must be positive")
        return _attribute_integrated_gradients(trace, target, model, self.execution_model_name, self.steps)


def _attribute_with_gradients(
    trace: SerializedTrace,
    target: FailureTarget,
    model: ModelAdapter,
    method_name: str,
    execution_model_name: str | None,
    *,
    mode: Literal["gradient", "gradient_times_input"],
) -> AttributionResult:
    import torch

    target.validate_against_trace(trace)
    tokenized = model.tokenize(trace.serialized_text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    inputs_embeds = model.input_embeddings(input_ids, requires_grad=True)
    output = model.forward(inputs_embeds, attention_mask)
    loss = _target_loss(output.logits, input_ids, target, trace)
    loss.backward()
    gradients = inputs_embeds.grad
    if gradients is None:
        raise RuntimeError("input embedding gradients were not populated")
    scores = gradients if mode == "gradient" else gradients * inputs_embeds.detach()
    token_scores = torch.linalg.vector_norm(scores, dim=-1).squeeze(0)
    token_scores = _zero_agent_scores(token_scores, trace)
    result = AttributionResult(
        method_name=method_name,
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=target.target_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata={
            "loss": float(loss.detach().cpu()),
            "target_node_ids": tuple(target.node_ids),
            "target_span": target.span,
        },
    )
    result.validate_against_trace(trace)
    return result


def _attribute_integrated_gradients(
    trace: SerializedTrace,
    target: FailureTarget,
    model: ModelAdapter,
    execution_model_name: str | None,
    steps: int,
) -> AttributionResult:
    import torch

    target.validate_against_trace(trace)
    tokenized = model.tokenize(trace.serialized_text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    actual_embeds = model.input_embeddings(input_ids, requires_grad=False).detach()
    baseline = torch.zeros_like(actual_embeds)
    accumulated_gradients = torch.zeros_like(actual_embeds)
    for step in range(1, steps + 1):
        alpha = step / steps
        scaled = (baseline + alpha * (actual_embeds - baseline)).detach().requires_grad_(True)
        output = model.forward(scaled, attention_mask)
        loss = _target_loss(output.logits, input_ids, target, trace)
        gradients = torch.autograd.grad(loss, scaled)[0]
        accumulated_gradients = accumulated_gradients + gradients
    average_gradients = accumulated_gradients / steps
    attributions = (actual_embeds - baseline) * average_gradients
    token_scores = torch.linalg.vector_norm(attributions, dim=-1).squeeze(0)
    token_scores = _zero_agent_scores(token_scores, trace)
    result = AttributionResult(
        method_name="integrated_gradients",
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=target.target_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata={
            "steps": steps,
            "target_node_ids": tuple(target.node_ids),
            "target_span": target.span,
        },
    )
    result.validate_against_trace(trace)
    return result


def _target_loss(logits, input_ids, target: FailureTarget, trace: SerializedTrace):
    import torch.nn.functional as F

    token_count = input_ids.shape[1]
    if token_count != max((span.end_token for span in trace.spans), default=0):
        raise ValueError("model token count must match SerializedTrace span token count")
    positions = _target_token_positions(target, trace)
    losses = []
    for position in positions:
        if position == 0:
            continue
        target_ids = input_ids[:, position].to(logits.device)
        losses.append(F.cross_entropy(logits[:, position - 1, :], target_ids, reduction="none"))
    if not losses:
        raise ValueError("failure target must contain at least one token position with predecessor context")
    return -sum(losses).mean()


def _target_token_positions(target: FailureTarget, trace: SerializedTrace) -> tuple[int, ...]:
    positions: set[int] = set()
    if target.span is not None:
        start, end = target.span
        return tuple(range(start, end))
    for span in trace.spans:
        if span.node_id in target.node_ids:
            positions.update(range(span.start_token, span.end_token))
    return tuple(sorted(positions))


def _zero_agent_scores(token_scores, trace: SerializedTrace):
    token_scores = token_scores.clone()
    for span in trace.spans:
        if span.block_role == "agent":
            token_scores[span.start_token : span.end_token] = 0.0
    return token_scores
