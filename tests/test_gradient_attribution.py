from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_tracegrad.attribution import (
    GradientSaliencyAttribution,
    GradientTimesInputAttribution,
    IntegratedGradientsAttribution,
)
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.target import FailureTarget
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
