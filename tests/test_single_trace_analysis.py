from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.analysis import analysis_to_dict, analyze_normalized_trace
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput

import pytest

torch = pytest.importorskip("torch")


@dataclass
class TinyBackwardModel:
    name: str = "tiny-backward-model"

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
        embeddings = torch.nn.functional.one_hot(input_ids, num_classes=16).to(torch.float32)
        embeddings = embeddings.detach().clone()
        if requires_grad:
            embeddings.requires_grad_(True)
        return embeddings

    def forward(self, inputs_embeds, attention_mask):
        del attention_mask
        return ModelForwardOutput(logits=inputs_embeds * 2.0)

    def chat_template_supported(self) -> bool:
        return False


class WhitespaceOffsetTokenizer:
    name_or_path = "whitespace-offset-tokenizer"

    def __call__(self, text: str, *, return_offsets_mapping: bool, add_special_tokens: bool):
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


def make_raw_trace():
    return {
        "nodes": [
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
    }


def test_analyze_normalized_trace_runs_backward_and_returns_rankings() -> None:
    result = analyze_normalized_trace(
        make_raw_trace(),
        target_node_ids=("agent-1",),
        model=TinyBackwardModel(),
        execution_model_name="tiny-backward-model",
    )

    assert result.attribution.method_name == "gradient_saliency"
    assert result.attribution.same_model is True
    assert len(result.attribution.token_scores) == 5
    assert result.attribution.token_scores[3:] == (0.0, 0.0)
    node_sum = next(item for item in result.rankings if item.grain == "node" and item.view_name == "sum")
    assert [item.instance.instance_id for item in node_sum.items] == ["user-1", "sys-1"]


def test_analysis_to_dict_is_json_ready() -> None:
    result = analyze_normalized_trace(
        make_raw_trace(),
        target_node_ids=("agent-1",),
        model=TinyBackwardModel(),
    )

    payload = analysis_to_dict(result)

    assert payload["trace"]["token_count"] == 5
    assert payload["target"]["node_ids"] == ["agent-1"]
    assert payload["rankings"]
    assert payload["distributions"]
