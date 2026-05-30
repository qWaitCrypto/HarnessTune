"""Tests for batch failure landscape analysis."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.diagnosis import run_landscape
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
        embeddings = torch.nn.functional.one_hot(input_ids, num_classes=256).to(torch.float32)
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


def _trace(policy_text: str, agent_text: str):
    return {
        "nodes": [
            {
                "node_id": "policy",
                "block_role": "system",
                "sub_block_kind": "system.instruction",
                "content": policy_text,
                "sequence_index": 0,
            },
            {
                "node_id": "user",
                "block_role": "user",
                "sub_block_kind": "user.content",
                "content": "user request",
                "sequence_index": 1,
            },
            {
                "node_id": "agent",
                "block_role": "agent",
                "sub_block_kind": "agent.content",
                "content": agent_text,
                "sequence_index": 2,
            },
        ]
    }


def test_run_landscape_builds_harness_stats_and_clusters() -> None:
    result = run_landscape(
        (
            ("trace-a", _trace("transfer policy", "wrong transfer"), None),
            ("trace-b", _trace("refusal policy", "wrong answer"), None),
        ),
        model=TinyBackwardModel(),
        target_node_ids=("agent",),
        expected_target_text="I cannot do that.",
    )

    assert len(result.traces) == 2
    assert result.component_stats
    assert result.clusters
    assert all("user" not in component for trace in result.traces for component in trace.fingerprint)
