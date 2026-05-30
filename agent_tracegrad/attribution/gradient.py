"""Real gradient attribution methods over input embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_tracegrad.attribution.result import AttributionResult
from agent_tracegrad.model.adapter import ModelAdapter
from agent_tracegrad.target.objective import TargetObjective, target_objective_to_dict
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

    def attribute_objective(
        self,
        trace: SerializedTrace,
        objective: TargetObjective,
        model: ModelAdapter,
    ) -> AttributionResult:
        return _attribute_objective_with_gradients(
            trace,
            objective,
            model,
            self.name,
            self.execution_model_name,
            mode="gradient",
        )


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

    def attribute_objective(
        self,
        trace: SerializedTrace,
        objective: TargetObjective,
        model: ModelAdapter,
    ) -> AttributionResult:
        return _attribute_objective_with_gradients(
            trace,
            objective,
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

    def attribute_objective(
        self,
        trace: SerializedTrace,
        objective: TargetObjective,
        model: ModelAdapter,
    ) -> AttributionResult:
        if self.steps < 1:
            raise ValueError("integrated gradients steps must be positive")
        return _attribute_objective_integrated_gradients(
            trace,
            objective,
            model,
            self.execution_model_name,
            self.steps,
        )


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


def _attribute_objective_with_gradients(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    method_name: str,
    execution_model_name: str | None,
    *,
    mode: Literal["gradient", "gradient_times_input"],
) -> AttributionResult:
    import torch

    objective.validate_against_trace(trace)
    if objective.objective_type in {"expected_action", "contrastive"} and objective.bad_target is not None:
        return _attribute_anchored_objective_with_gradients(
            trace,
            objective,
            model,
            method_name,
            execution_model_name,
            mode=mode,
        )
    objective_input = _build_objective_input(trace, objective)
    inputs_embeds, loss = _objective_forward_loss(model, objective_input)
    loss.backward()
    gradients = inputs_embeds.grad
    if gradients is None:
        raise RuntimeError("input embedding gradients were not populated")
    scores = gradients if mode == "gradient" else gradients * inputs_embeds.detach()
    token_scores = torch.linalg.vector_norm(scores, dim=-1).squeeze(0)
    token_scores = token_scores[: objective_input.trace_token_count]
    token_scores = _zero_agent_scores(token_scores, trace)
    result = AttributionResult(
        method_name=method_name,
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=objective.objective_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata=_objective_result_metadata(objective, loss),
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


def _attribute_objective_integrated_gradients(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    execution_model_name: str | None,
    steps: int,
) -> AttributionResult:
    import torch

    objective.validate_against_trace(trace)
    if objective.objective_type in {"expected_action", "contrastive"} and objective.bad_target is not None:
        return _attribute_anchored_objective_integrated_gradients(
            trace,
            objective,
            model,
            execution_model_name,
            steps,
        )
    objective_input = _build_objective_input(trace, objective)
    tokenized = model.tokenize(objective_input.text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    actual_embeds = model.input_embeddings(input_ids, requires_grad=False).detach()
    baseline = torch.zeros_like(actual_embeds)
    accumulated_gradients = torch.zeros_like(actual_embeds)
    for step in range(1, steps + 1):
        alpha = step / steps
        scaled = (baseline + alpha * (actual_embeds - baseline)).detach().requires_grad_(True)
        output = model.forward(scaled, attention_mask)
        loss = _objective_loss(output.logits, input_ids, objective_input)
        gradients = torch.autograd.grad(loss, scaled)[0]
        accumulated_gradients = accumulated_gradients + gradients
    average_gradients = accumulated_gradients / steps
    attributions = (actual_embeds - baseline) * average_gradients
    token_scores = torch.linalg.vector_norm(attributions, dim=-1).squeeze(0)
    token_scores = token_scores[: objective_input.trace_token_count]
    token_scores = _zero_agent_scores(token_scores, trace)
    result = AttributionResult(
        method_name="integrated_gradients",
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=objective.objective_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata={
            **_objective_result_metadata(objective, loss),
            "steps": steps,
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


@dataclass(frozen=True)
class _ObjectiveInput:
    text: str
    objective_type: str
    trace_token_count: int
    bad_positions: tuple[int, ...]
    expected_start_token: int | None = None


@dataclass(frozen=True)
class _AnchoredObjectiveInput:
    prefix_text: str
    bad_text: str
    expected_text: str
    trace_token_count: int
    prefix_token_count: int
    anchor_start_token: int
    anchor_end_token: int
    anchor_start_char: int
    anchor_end_char: int
    exact_anchor: bool


@dataclass(frozen=True)
class _ObjectiveBranch:
    input_ids: object
    inputs_embeds: object
    score: object


def _build_objective_input(trace: SerializedTrace, objective: TargetObjective) -> _ObjectiveInput:
    if objective.objective_type == "bad_action":
        if objective.bad_target is None:
            raise ValueError("bad_action objective requires bad_target")
        return _ObjectiveInput(
            text=trace.serialized_text,
            objective_type=objective.objective_type,
            trace_token_count=max((span.end_token for span in trace.spans), default=0),
            bad_positions=_target_token_positions(objective.bad_target, trace),
        )
    if objective.expected_target is None:
        raise ValueError(f"{objective.objective_type} objective requires expected_target")
    text = trace.serialized_text + "\n" + objective.expected_target.content
    return _ObjectiveInput(
        text=text,
        objective_type=objective.objective_type,
        trace_token_count=max((span.end_token for span in trace.spans), default=0),
        bad_positions=_target_token_positions(objective.bad_target, trace) if objective.bad_target is not None else (),
        expected_start_token=max((span.end_token for span in trace.spans), default=0),
    )


def _build_anchored_objective_input(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
) -> _AnchoredObjectiveInput:
    if objective.bad_target is None or objective.expected_target is None:
        raise ValueError("anchored objective requires bad_target and expected_target")
    anchor = _target_anchor(trace, objective.bad_target)
    prefix_text = trace.serialized_text[: anchor["start_char"]]
    bad_continuation = trace.serialized_text[anchor["start_char"] : anchor["end_char"]]
    if not bad_continuation:
        raise ValueError("bad target continuation must not be empty")
    prefix_token_count = model.tokenize(prefix_text).input_ids.shape[1] if prefix_text else 0
    return _AnchoredObjectiveInput(
        prefix_text=prefix_text,
        bad_text=f"{prefix_text}{bad_continuation}",
        expected_text=f"{prefix_text}{objective.expected_target.content}",
        trace_token_count=max((span.end_token for span in trace.spans), default=0),
        prefix_token_count=prefix_token_count,
        anchor_start_token=anchor["start_token"],
        anchor_end_token=anchor["end_token"],
        anchor_start_char=anchor["start_char"],
        anchor_end_char=anchor["end_char"],
        exact_anchor=anchor["exact"],
    )


def _objective_forward_loss(model: ModelAdapter, objective_input: _ObjectiveInput):
    tokenized = model.tokenize(objective_input.text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    objective_input = _resolve_expected_start_token(objective_input, input_ids)
    inputs_embeds = model.input_embeddings(input_ids, requires_grad=True)
    output = model.forward(inputs_embeds, attention_mask)
    return inputs_embeds, _objective_loss(output.logits, input_ids, objective_input)


def _objective_branch_forward(model: ModelAdapter, text: str, *, start_token: int) -> _ObjectiveBranch:
    tokenized = model.tokenize(text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    inputs_embeds = model.input_embeddings(input_ids, requires_grad=True)
    output = model.forward(inputs_embeds, attention_mask)
    score = _positions_logprob_loss(output.logits, input_ids, tuple(range(start_token, input_ids.shape[1])))
    if score is None:
        raise ValueError("objective branch must contain at least one scored token")
    return _ObjectiveBranch(input_ids=input_ids, inputs_embeds=inputs_embeds, score=score)


def _objective_branch_score(model: ModelAdapter, inputs_embeds, text: str, *, start_token: int):
    tokenized = model.tokenize(text)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    output = model.forward(inputs_embeds, attention_mask)
    score = _positions_logprob_loss(output.logits, input_ids, tuple(range(start_token, input_ids.shape[1])))
    if score is None:
        raise ValueError("objective branch must contain at least one scored token")
    return score


def _integrated_branch_state(model: ModelAdapter, text: str):
    import torch

    tokenized = model.tokenize(text)
    actual_embeds = model.input_embeddings(tokenized.input_ids, requires_grad=False).detach()
    baseline = torch.zeros_like(actual_embeds)
    accumulated = torch.zeros_like(actual_embeds)
    return actual_embeds, baseline, accumulated


def _attribute_anchored_objective_with_gradients(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    method_name: str,
    execution_model_name: str | None,
    *,
    mode: Literal["gradient", "gradient_times_input"],
) -> AttributionResult:
    import torch

    if objective.expected_target is None or objective.bad_target is None:
        raise ValueError("anchored objective requires bad_target and expected_target")
    objective_input = _build_anchored_objective_input(trace, objective, model)
    expected_branch = _objective_branch_forward(
        model,
        objective_input.expected_text,
        start_token=objective_input.prefix_token_count,
    )
    branches = [expected_branch]
    if objective.objective_type == "contrastive":
        bad_branch = _objective_branch_forward(
            model,
            objective_input.bad_text,
            start_token=objective_input.prefix_token_count,
        )
        loss = bad_branch.score - expected_branch.score
        branches.insert(0, bad_branch)
    elif objective.objective_type == "expected_action":
        loss = expected_branch.score
    else:
        raise ValueError("anchored objective only supports expected_action or contrastive")

    loss.backward()
    prefix_vectors = []
    for branch in branches:
        gradients = branch.inputs_embeds.grad
        if gradients is None:
            raise RuntimeError("input embedding gradients were not populated")
        vectors = gradients if mode == "gradient" else gradients * branch.inputs_embeds.detach()
        prefix_vectors.append(vectors[:, : objective_input.prefix_token_count, :])
    combined = sum(prefix_vectors)
    prefix_scores = torch.linalg.vector_norm(combined, dim=-1).squeeze(0)
    token_scores = _pad_prefix_scores(prefix_scores, objective_input.trace_token_count)
    token_scores = _zero_agent_scores(token_scores, trace)
    result = AttributionResult(
        method_name=method_name,
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=objective.objective_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata={
            **_objective_result_metadata(objective, loss),
            **_anchored_objective_metadata(objective_input),
        },
    )
    result.validate_against_trace(trace)
    return result


def _attribute_anchored_objective_integrated_gradients(
    trace: SerializedTrace,
    objective: TargetObjective,
    model: ModelAdapter,
    execution_model_name: str | None,
    steps: int,
) -> AttributionResult:
    import torch

    if objective.expected_target is None or objective.bad_target is None:
        raise ValueError("anchored objective requires bad_target and expected_target")
    objective_input = _build_anchored_objective_input(trace, objective, model)
    expected_actual, expected_baseline, expected_accumulated = _integrated_branch_state(
        model,
        objective_input.expected_text,
    )
    bad_state = None
    if objective.objective_type == "contrastive":
        bad_state = _integrated_branch_state(model, objective_input.bad_text)
    elif objective.objective_type != "expected_action":
        raise ValueError("anchored objective only supports expected_action or contrastive")

    final_loss = None
    for step in range(1, steps + 1):
        alpha = step / steps
        expected_scaled = (
            expected_baseline + alpha * (expected_actual - expected_baseline)
        ).detach().requires_grad_(True)
        expected_score = _objective_branch_score(
            model,
            expected_scaled,
            objective_input.expected_text,
            start_token=objective_input.prefix_token_count,
        )
        if bad_state is None:
            loss = expected_score
            expected_grad = torch.autograd.grad(loss, expected_scaled)[0]
            expected_accumulated = expected_accumulated + expected_grad
        else:
            bad_actual, bad_baseline, bad_accumulated = bad_state
            bad_scaled = (bad_baseline + alpha * (bad_actual - bad_baseline)).detach().requires_grad_(True)
            bad_score = _objective_branch_score(
                model,
                bad_scaled,
                objective_input.bad_text,
                start_token=objective_input.prefix_token_count,
            )
            loss = bad_score - expected_score
            bad_grad, expected_grad = torch.autograd.grad(loss, (bad_scaled, expected_scaled))
            bad_state = (bad_actual, bad_baseline, bad_accumulated + bad_grad)
            expected_accumulated = expected_accumulated + expected_grad
        final_loss = loss
    expected_attributions = (expected_actual - expected_baseline) * (expected_accumulated / steps)
    prefix_vectors = [expected_attributions[:, : objective_input.prefix_token_count, :]]
    if bad_state is not None:
        bad_actual, bad_baseline, bad_accumulated = bad_state
        bad_attributions = (bad_actual - bad_baseline) * (bad_accumulated / steps)
        prefix_vectors.insert(0, bad_attributions[:, : objective_input.prefix_token_count, :])
    combined = sum(prefix_vectors)
    prefix_scores = torch.linalg.vector_norm(combined, dim=-1).squeeze(0)
    token_scores = _pad_prefix_scores(prefix_scores, objective_input.trace_token_count)
    token_scores = _zero_agent_scores(token_scores, trace)
    if final_loss is None:
        raise RuntimeError("integrated gradients did not run any steps")
    result = AttributionResult(
        method_name="integrated_gradients",
        attribution_model_name=model.name,
        execution_model_name=execution_model_name,
        same_model=execution_model_name is not None and model.name == execution_model_name,
        target_id=objective.objective_id,
        token_scores=tuple(token_scores.detach().cpu().tolist()),
        metadata={
            **_objective_result_metadata(objective, final_loss),
            **_anchored_objective_metadata(objective_input),
            "steps": steps,
        },
    )
    result.validate_against_trace(trace)
    return result


def _resolve_expected_start_token(objective_input: _ObjectiveInput, input_ids) -> _ObjectiveInput:
    if objective_input.expected_start_token is not None:
        return objective_input
    if not objective_input.bad_positions:
        return _ObjectiveInput(
            text=objective_input.text,
            objective_type=objective_input.objective_type,
            trace_token_count=objective_input.trace_token_count,
            bad_positions=objective_input.bad_positions,
            expected_start_token=objective_input.trace_token_count,
        )
    return objective_input


def _objective_loss(logits, input_ids, objective_input: _ObjectiveInput):
    import torch.nn.functional as F

    token_count = input_ids.shape[1]
    if objective_input.trace_token_count > token_count:
        raise ValueError("objective input token count is shorter than SerializedTrace token count")
    bad_loss = _positions_logprob_loss(logits, input_ids, objective_input.bad_positions)
    expected_positions = _expected_token_positions(objective_input, token_count)
    expected_loss = _positions_logprob_loss(logits, input_ids, expected_positions)
    if objective_input.objective_type == "contrastive":
        if bad_loss is None or expected_loss is None:
            raise ValueError("contrastive objective requires scored bad and expected tokens")
        return bad_loss - expected_loss
    if objective_input.objective_type == "bad_action" and bad_loss is not None:
        return bad_loss
    if objective_input.objective_type == "expected_action" and expected_loss is not None:
        return expected_loss
    raise ValueError("target objective must contain at least one scored token position")


def _positions_logprob_loss(logits, input_ids, positions: tuple[int, ...] | range | None):
    import torch.nn.functional as F

    if not positions:
        return None
    losses = []
    for position in positions:
        if position == 0:
            continue
        target_ids = input_ids[:, position].to(logits.device)
        losses.append(F.cross_entropy(logits[:, position - 1, :], target_ids, reduction="none"))
    if not losses:
        return None
    return -sum(losses).mean()


def _expected_token_positions(objective_input: _ObjectiveInput, token_count: int) -> tuple[int, ...]:
    if objective_input.expected_start_token is None:
        return ()
    if token_count <= objective_input.expected_start_token:
        raise ValueError("expected target content must add at least one token")
    return tuple(range(objective_input.expected_start_token, token_count))


def _objective_result_metadata(objective: TargetObjective, loss) -> dict:
    payload = target_objective_to_dict(objective)
    return {
        "loss": float(loss.detach().cpu()),
        "objective": payload,
        "objective_type": objective.objective_type,
        "objective_formula": payload["objective_formula"],
    }


def _anchored_objective_metadata(objective_input: _AnchoredObjectiveInput) -> dict:
    return {
        "objective_anchor": {
            "mode": "failure_target_prefix",
            "prefix_token_count": objective_input.prefix_token_count,
            "anchor_start_token": objective_input.anchor_start_token,
            "anchor_end_token": objective_input.anchor_end_token,
            "anchor_start_char": objective_input.anchor_start_char,
            "anchor_end_char": objective_input.anchor_end_char,
            "exact_anchor": objective_input.exact_anchor,
        }
    }


def _target_token_positions(target: FailureTarget, trace: SerializedTrace) -> tuple[int, ...]:
    positions: set[int] = set()
    if target.span is not None:
        start, end = target.span
        return tuple(range(start, end))
    for span in trace.spans:
        if span.node_id in target.node_ids:
            positions.update(range(span.start_token, span.end_token))
    return tuple(sorted(positions))


def _target_anchor(trace: SerializedTrace, target: FailureTarget) -> dict:
    positions = _target_token_positions(target, trace)
    if not positions:
        raise ValueError("failure target must contain at least one token")
    start_token = min(positions)
    end_token = max(positions) + 1
    selected_spans = [span for span in trace.spans if span.node_id in target.node_ids]
    if not selected_spans:
        raise ValueError("failure target references no serialized spans")
    start_span = next((span for span in selected_spans if span.start_token <= start_token < span.end_token), None)
    end_span = next((span for span in selected_spans if span.start_token < end_token <= span.end_token), None)
    if start_span is None:
        start_span = min(selected_spans, key=lambda span: span.start_token)
    if end_span is None:
        end_span = max(selected_spans, key=lambda span: span.end_token)
    if start_span.text_start_char is None or end_span.text_end_char is None:
        raise ValueError("anchored objectives require trace span character offsets")
    exact = start_token == start_span.start_token and end_token == end_span.end_token
    start_char = start_span.text_start_char
    end_char = end_span.text_end_char
    if end_char <= start_char:
        raise ValueError("failure target character anchor must be non-empty")
    return {
        "start_token": start_token,
        "end_token": end_token,
        "start_char": start_char,
        "end_char": end_char,
        "exact": exact,
    }


def _pad_prefix_scores(prefix_scores, trace_token_count: int):
    import torch

    token_scores = torch.zeros(trace_token_count, dtype=prefix_scores.dtype, device=prefix_scores.device)
    copy_count = min(prefix_scores.shape[0], trace_token_count)
    token_scores[:copy_count] = prefix_scores[:copy_count]
    return token_scores


def _zero_agent_scores(token_scores, trace: SerializedTrace):
    token_scores = token_scores.clone()
    for span in trace.spans:
        if span.block_role == "agent":
            token_scores[span.start_token : span.end_token] = 0.0
    return token_scores
