"""Tests for policy/tool atom drill-down."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.attribution import AttributionResult
from agent_tracegrad.analysis.single_trace import SingleTraceAnalysisResult
from agent_tracegrad.diagnosis import atomize_policy_text, atomize_tool_schema, run_diagnosis, run_drill
from agent_tracegrad.diagnosis.types import DiagnosisResult
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.target import FailureTarget
from agent_tracegrad.trace import SpanMetadata, SerializedTrace
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


def test_atomize_tool_schema_uses_jsonpath_source_span_for_duplicate_values() -> None:
    node = TraceNode(
        node_id="schema",
        block_role="system",
        sub_block_kind="system.tool_schema",
        content='{"first":"same","second":"same"}',
    )

    atoms = {atom.metadata["jsonpath"]: atom for atom in atomize_tool_schema(node)}

    assert atoms["$.first"].text == '"same"'
    assert atoms["$.second"].text == '"same"'
    assert atoms["$.second"].char_start > atoms["$.first"].char_start


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


def test_run_drill_classification_is_not_atom_order_sensitive() -> None:
    first = _make_order_sensitive_diagnosis(("small", "large"))
    second = _make_order_sensitive_diagnosis(("large", "small"))

    first_classes = {
        atom.atom.text.removeprefix("- "): atom.classification
        for atom in run_drill(first).atoms
    }
    second_classes = {
        atom.atom.text.removeprefix("- "): atom.classification
        for atom in run_drill(second).atoms
    }

    assert first_classes == second_classes
    assert first_classes["small"] == "strengthen"
    assert first_classes["large"] == "narrow"


def _make_order_sensitive_diagnosis(order: tuple[str, str]) -> DiagnosisResult:
    content = "\n".join(f"- {name}" for name in order)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in content.split():
        start = content.index(token, cursor)
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    trace = SerializedTrace(
        nodes={
            "policy": TraceNode(
                "policy",
                "system",
                "system.instruction",
                content,
                sequence_index=0,
            ),
            "agent": TraceNode(
                "agent",
                "agent",
                "agent.content",
                "bad",
                sequence_index=1,
            ),
        },
        serialized_text=content,
        spans=(
            SpanMetadata(
                "span-policy",
                "policy",
                "system",
                "system.instruction",
                0,
                len(offsets),
                text_start_char=0,
                text_end_char=len(content),
            ),
            SpanMetadata(
                "span-agent",
                "agent",
                "agent",
                "agent.content",
                len(offsets),
                len(offsets),
            ),
        ),
        tokenizer_name="test-tokenizer",
        token_offsets=tuple(offsets),
    )
    margin_by_name = {"small": 0.5, "large": 10.0}
    expected_by_name = {"small": 1.0, "large": 1.0}
    margin_scores = tuple(
        margin_by_name.get(content[start:end].lstrip("- "), 0.0)
        for start, end in offsets
    )
    expected_scores = tuple(
        expected_by_name.get(content[start:end].lstrip("- "), 0.0)
        for start, end in offsets
    )
    zero_scores = tuple(0.0 for _ in offsets)
    target = FailureTarget("target", ("agent",))
    return DiagnosisResult(
        bad_result=_analysis(trace, target, zero_scores),
        expected_result=_analysis(trace, target, expected_scores),
        contrastive_result=_analysis(trace, target, margin_scores),
        margin_distributions=(),
    )


def _analysis(
    trace: SerializedTrace,
    target: FailureTarget,
    token_scores: tuple[float, ...],
) -> SingleTraceAnalysisResult:
    return SingleTraceAnalysisResult(
        trace=trace,
        target=target,
        attribution=AttributionResult(
            method_name="gradient_saliency",
            attribution_model_name="test-model",
            execution_model_name=None,
            same_model=False,
            target_id=target.target_id,
            token_scores=token_scores,
        ),
        distributions=(),
        rankings=(),
        metadata={},
    )
