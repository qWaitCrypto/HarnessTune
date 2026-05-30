"""Tests for policy/tool atom drill-down."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.diagnosis import atomize_policy_text, atomize_tool_schema, run_diagnosis, run_drill
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.trace.schema import TraceNode

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


def test_atomize_policy_text_splits_markdown_blocks() -> None:
    node = TraceNode(
        node_id="policy",
        block_role="system",
        sub_block_kind="system.instruction",
        content="# Transfer\n- First call transfer_to_human_agents.\n\nRefuse when deadline passed.",
    )

    atoms = atomize_policy_text(node)
    kinds = [atom.atom_kind for atom in atoms]

    assert "policy.heading" in kinds
    assert "policy.bullet" in kinds
    assert "policy.paragraph" in kinds


def test_atomize_tool_schema_extracts_json_fields() -> None:
    node = TraceNode(
        node_id="schema",
        block_role="system",
        sub_block_kind="system.tool_schema",
        content='{"name":"transfer_to_human_agents","description":"Transfer user when needed"}',
    )

    atoms = atomize_tool_schema(node)
    kinds = {atom.atom_kind for atom in atoms}

    assert "tool_schema.name" in kinds
    assert "tool_schema.description" in kinds


def test_run_drill_produces_atom_scores() -> None:
    diagnosis = run_diagnosis(
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
        expected_target_text="I cannot do that.",
    )

    result = run_drill(diagnosis)

    assert result.atoms
    assert all(atom.token_count >= 1 for atom in result.atoms)
    assert {atom.classification for atom in result.atoms} <= {"preserve", "narrow", "strengthen"}
