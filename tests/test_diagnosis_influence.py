"""Tests for policy-action influence matrix."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.diagnosis import CandidateAction, run_influence_matrix
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


def test_run_influence_matrix_builds_candidate_columns() -> None:
    result = run_influence_matrix(
        {
            "nodes": [
                {
                    "node_id": "policy",
                    "block_role": "system",
                    "sub_block_kind": "system.instruction",
                    "content": "# Policy\n- First call transfer_to_human_agents.\n- Refuse after deadline.",
                    "sequence_index": 0,
                },
                {
                    "node_id": "agent",
                    "block_role": "agent",
                    "sub_block_kind": "agent.content",
                    "content": "wrong answer",
                    "sequence_index": 1,
                },
            ]
        },
        model=TinyBackwardModel(),
        target_node_ids=("agent",),
        candidates=(
            CandidateAction(action_id="refuse", text="I cannot do that."),
            CandidateAction(action_id="transfer", text="I will transfer you."),
        ),
    )

    assert len(result.candidates) == 2
    assert result.rows
    assert all(len(row.cells) == 2 for row in result.rows)
