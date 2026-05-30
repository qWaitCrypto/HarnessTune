from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_tracegrad.attribution import (
    GradientSaliencyAttribution,
    GradientTimesInputAttribution,
    IntegratedGradientsAttribution,
)
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.target import ExpectedTarget, FailureTarget, TargetObjective
from agent_tracegrad.trace import JsonTraceAdapter, TraceSerializer

torch = pytest.importorskip("torch")


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


@dataclass
class FakeGradientModel:
    name: str = "fake-gradient-model"

    def tokenize(self, text: str) -> TokenizedOutput:
        token_count = len(text.split())
        return TokenizedOutput(
            input_ids=torch.arange(token_count, dtype=torch.long).unsqueeze(0),
            attention_mask=torch.ones((1, token_count), dtype=torch.long),
        )

    def input_embeddings(self, input_ids, *, requires_grad: bool):
        vocab_size = 16
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).to(torch.float32)
        embeddings = one_hot.detach().clone()
        if requires_grad:
            embeddings.requires_grad_(True)
        return embeddings

    def forward(self, inputs_embeds, attention_mask):
        del attention_mask
        logits = inputs_embeds * 2.0
        return ModelForwardOutput(logits=logits)

    def chat_template_supported(self) -> bool:
        return False


class ContextualFakeGradientModel(FakeGradientModel):
    def forward(self, inputs_embeds, attention_mask):
        del attention_mask
        return ModelForwardOutput(logits=inputs_embeds.cumsum(dim=1))


def make_trace():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys-1",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "zero one",
                "sequence_index": 0,
            },
            {
                "node_id": "user-1",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "two",
                "sequence_index": 1,
            },
            {
                "node_id": "agent-1",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "three four",
                "sequence_index": 2,
            },
        ]
    )
    return TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes)


def test_gradient_saliency_produces_scores_and_zeros_agent_span() -> None:
    trace = make_trace()
    target = FailureTarget("target-1", ("agent-1",))

    result = GradientSaliencyAttribution(execution_model_name="fake-gradient-model").attribute(
        trace,
        target,
        FakeGradientModel(),
    )

    assert result.method_name == "gradient_saliency"
    assert result.same_model is True
    assert len(result.token_scores) == 5
    assert result.token_scores[2] > 0
    assert result.token_scores[3:] == (0.0, 0.0)


def test_gradient_saliency_supports_expected_action_objective() -> None:
    trace = make_trace()
    objective = TargetObjective.expected_action(
        ExpectedTarget(target_id="gold-refusal", content="five six", source="human")
    )

    result = GradientSaliencyAttribution().attribute_objective(trace, objective, ContextualFakeGradientModel())

    assert result.target_id == "gold-refusal"
    assert result.metadata["objective_type"] == "expected_action"
    assert result.metadata["score_method"] == "norm_saliency"
    assert result.metadata["score_family"] == "sensitivity"
    assert result.metadata["objective"]["expected_target"]["content"] == "five six"
    assert len(result.token_scores) == 5
    assert result.token_scores[0] > 0
    assert result.token_scores[3:] == (0.0, 0.0)


def test_gradient_saliency_supports_contrastive_objective() -> None:
    trace = make_trace()
    bad = FailureTarget("bad-transfer", ("agent-1",))
    expected = ExpectedTarget(target_id="gold-refusal", content="five six")
    objective = TargetObjective.contrastive(bad, expected)

    result = GradientSaliencyAttribution().attribute_objective(trace, objective, ContextualFakeGradientModel())

    assert result.target_id == "bad-transfer:vs:gold-refusal"
    assert result.metadata["objective_type"] == "contrastive"
    assert result.metadata["score_method"] == "branch_difference_norm_saliency"
    assert result.metadata["score_family"] == "sensitivity"
    assert result.metadata["objective_formula"] == "log P(bad_target | context) - log P(expected_target | context)"
    assert result.metadata["objective_anchor"]["mode"] == "failure_target_prefix"
    assert result.metadata["objective_anchor"]["prefix_token_count"] == 3
    assert len(result.token_scores) == 5
    assert result.token_scores[3:] == (0.0, 0.0)


def test_contrastive_objective_can_produce_signed_scores() -> None:
    class SignedContrastiveModel(ContextualFakeGradientModel):
        def tokenize(self, text: str) -> TokenizedOutput:
            vocab = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            ids = [vocab.get(token, 15) for token in text.split()]
            return TokenizedOutput(
                input_ids=torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            )

        def forward(self, inputs_embeds, attention_mask):
            del attention_mask
            logits = torch.zeros_like(inputs_embeds)
            if inputs_embeds.shape[1] > 3:
                expected_signal = inputs_embeds[:, 0, 0]
                bad_signal = inputs_embeds[:, 1, 1]
                bad_branch = inputs_embeds[:, 3, 3]
                expected_branch = inputs_embeds[:, 3, 5]
                logits[:, 2, 3] = 4.0 * bad_signal * bad_branch
                logits[:, 2, 5] = 4.0 * expected_signal * expected_branch
            return ModelForwardOutput(logits=logits)

    trace = make_trace()
    bad = FailureTarget("bad-transfer", ("agent-1",))
    expected = ExpectedTarget(target_id="gold-refusal", content="five six")
    objective = TargetObjective.contrastive(bad, expected)

    result = GradientSaliencyAttribution().attribute_objective(trace, objective, SignedContrastiveModel())

    components = result.metadata["contrastive_components"]
    bad_scores = components["bad_token_scores"]
    expected_scores = components["expected_token_scores"]
    margin_scores = components["margin_token_scores"]

    assert result.metadata["score_semantics"] == "bad_support_minus_expected_support"
    assert result.token_scores == margin_scores
    assert any(bad > expected for bad, expected in zip(bad_scores, expected_scores, strict=True))
    assert any(expected > bad for bad, expected in zip(bad_scores, expected_scores, strict=True))
    assert any(score < 0.0 for score in result.token_scores)
    assert any(score > 0.0 for score in result.token_scores)
    for bad_score, expected_score, margin_score in zip(bad_scores, expected_scores, margin_scores, strict=True):
        assert margin_score == pytest.approx(bad_score - expected_score)
    assert result.token_scores[3:] == (0.0, 0.0)


def test_gradient_times_input_and_integrated_gradients_validate_against_trace() -> None:
    trace = make_trace()
    target = FailureTarget("target-1", ("agent-1",))
    model = FakeGradientModel()

    gxi = GradientTimesInputAttribution(execution_model_name="fake-gradient-model").attribute(trace, target, model)
    ig = IntegratedGradientsAttribution(execution_model_name="fake-gradient-model", steps=2).attribute(
        trace,
        target,
        model,
    )

    gxi.validate_against_trace(trace)
    ig.validate_against_trace(trace)
    assert gxi.token_scores[3:] == (0.0, 0.0)
    assert ig.token_scores[3:] == (0.0, 0.0)


def test_gradient_attribution_rejects_tokenization_mismatch() -> None:
    trace = make_trace()
    target = FailureTarget("target-1", ("agent-1",))

    class MismatchedModel(FakeGradientModel):
        def tokenize(self, text: str) -> TokenizedOutput:
            del text
            return TokenizedOutput(input_ids=torch.arange(4, dtype=torch.long).unsqueeze(0))

    with pytest.raises(ValueError, match="token count"):
        GradientSaliencyAttribution().attribute(trace, target, MismatchedModel())


def test_integrated_gradients_rejects_nonpositive_steps() -> None:
    trace = make_trace()
    target = FailureTarget("target-1", ("agent-1",))

    with pytest.raises(ValueError, match="steps"):
        IntegratedGradientsAttribution(steps=0).attribute(trace, target, FakeGradientModel())
