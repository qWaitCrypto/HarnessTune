from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_tracegrad.attribution import AttributionResult
from agent_tracegrad.evaluation import delta_ll_curve, objective_loss
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
class TinyModel:
    name: str = "tiny-model"

    @property
    def tokenizer(self):
        return WhitespaceOffsetTokenizer()

    def tokenize(self, text: str) -> TokenizedOutput:
        token_count = len(text.split())
        return TokenizedOutput(
            input_ids=torch.arange(token_count, dtype=torch.long).unsqueeze(0),
            attention_mask=torch.ones((1, token_count), dtype=torch.long),
        )

    def input_embeddings(self, input_ids, *, requires_grad: bool):
        embeddings = torch.nn.functional.one_hot(input_ids, num_classes=128).to(torch.float32)
        embeddings = embeddings.detach().clone()
        if requires_grad:
            embeddings.requires_grad_(True)
        return embeddings

    def forward(self, inputs_embeds, attention_mask):
        del attention_mask
        return ModelForwardOutput(logits=inputs_embeds * 2.0)

    def chat_template_supported(self) -> bool:
        return False


def _trace():
    nodes = JsonTraceAdapter().adapt(
        [
            {
                "node_id": "sys",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": "policy text",
                "sequence_index": 0,
            },
            {
                "node_id": "user",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "user clue",
                "sequence_index": 1,
            },
            {
                "node_id": "agent",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": "wrong answer",
                "sequence_index": 2,
            },
        ]
    )
    return TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes)


def test_delta_ll_curve_masks_top_non_agent_tokens_only() -> None:
    trace = _trace()
    objective = TargetObjective.bad_action(FailureTarget(target_id="bad", node_ids=("agent",)))
    attribution = AttributionResult(
        method_name="gradient_saliency",
        attribution_model_name="tiny-model",
        execution_model_name="tiny-model",
        same_model=True,
        target_id="bad",
        token_scores=(1.0, 8.0, 5.0, 2.0, 100.0, 99.0),
    )

    points = delta_ll_curve(trace, attribution, objective, TinyModel(), ks=(1, 3))

    assert [point.k for point in points] == [1, 3]
    assert points[0].selected_token_indexes == (1,)
    assert all(index < 4 for point in points for index in point.selected_token_indexes)
    assert points[0].delta_loss == points[0].masked_loss - points[0].baseline_loss


def test_objective_loss_supports_anchored_contrastive_objective() -> None:
    trace = _trace()
    objective = TargetObjective.contrastive(
        FailureTarget(target_id="bad", node_ids=("agent",)),
        ExpectedTarget(target_id="good", content="right answer"),
    )

    score = objective_loss(trace, objective, TinyModel(), masked_token_indexes=(0,))

    assert isinstance(score, float)
