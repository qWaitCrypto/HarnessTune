"""Tests for the multi-objective diagnosis runner."""

from __future__ import annotations

from dataclasses import dataclass

from agent_tracegrad.diagnosis import (
    DiagnosisResult,
    build_decision_boundary_artifact,
    decision_boundary_to_dict,
    decision_boundary_to_markdown,
    diagnosis_from_dict,
    diagnosis_to_dict,
    run_diagnosis,
)
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


class SignedContrastiveModel(TinyBackwardModel):
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
            prefix_signal = inputs_embeds[:, 0, 0]
            logits[:, 2, 3] = -4.0 * prefix_signal
            logits[:, 2, 5] = 4.0 * prefix_signal
        return ModelForwardOutput(logits=logits)


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


def _make_raw_trace():
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


class TestBadActionOnly:
    def test_returns_bad_result_only(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
        )

        assert isinstance(result, DiagnosisResult)
        assert result.bad_result is not None
        assert result.expected_result is None
        assert result.contrastive_result is None
        assert result.margin_distributions == ()
        assert result.metadata["mode"] == "bad_action_only"

    def test_bad_result_has_attribution_scores(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
        )

        assert len(result.bad_result.attribution.token_scores) == 5
        assert result.bad_result.attribution.token_scores[3:] == (0.0, 0.0)


class TestFullDiagnosis:
    def test_all_three_results_present(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        assert result.bad_result is not None
        assert result.expected_result is not None
        assert result.contrastive_result is not None
        assert result.metadata["mode"] == "full_diagnosis"

    def test_margin_distributions_cover_all_grain_view_pairs(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        keys = {(md.grain, md.view_name) for md in result.margin_distributions}
        assert ("node", "sum") in keys
        assert ("sub_block_kind", "sum") in keys

    def test_margin_instance_ids_align_with_bad_result(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        node_sum = next(
            md for md in result.margin_distributions
            if md.grain == "node" and md.view_name == "sum"
        )
        margin_ids = {c.instance_id for c in node_sum.contributions}
        bad_ids = {
            inst.instance_id
            for dist in result.bad_result.distributions
            if dist.grain == "node"
            for inst in dist.instances
        }
        assert margin_ids >= bad_ids

    def test_margin_uses_contrastive_result_scores(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        node_sum = next(
            md for md in result.margin_distributions
            if md.grain == "node" and md.view_name == "sum"
        )
        contrastive_sum = next(
            dist for dist in result.contrastive_result.distributions
            if dist.grain == "node" and dist.view_name == "sum"
        )
        contrastive_by_id = {
            inst.instance_id: inst.views["sum"]
            for inst in contrastive_sum.instances
        }
        for contribution in node_sum.contributions:
            assert contribution.margin == contrastive_by_id[contribution.instance_id]

    def test_contrastive_margin_can_be_negative_and_preserved(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=SignedContrastiveModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        node_sum = next(
            md for md in result.margin_distributions
            if md.grain == "node" and md.view_name == "sum"
        )

        assert any(score < 0.0 for score in result.contrastive_result.attribution.token_scores)
        assert any(contribution.margin < 0.0 for contribution in node_sum.contributions)
        assert any(contribution.classification == "preserve" for contribution in node_sum.contributions)


class TestComponentClassification:
    def test_classifications_are_valid_values(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        node_sum = next(
            md for md in result.margin_distributions
            if md.grain == "node" and md.view_name == "sum"
        )
        for contribution in node_sum.contributions:
            assert contribution.classification in ("preserve", "narrow", "strengthen")

    def test_positive_margin_is_narrow(self) -> None:
        from agent_tracegrad.diagnosis.runner import _classify_component

        assert _classify_component(margin=1.0, expected_score=0.5, max_abs_margin=2.0, threshold=0.1) == "narrow"

    def test_negative_margin_is_preserve(self) -> None:
        from agent_tracegrad.diagnosis.runner import _classify_component

        assert _classify_component(margin=-1.0, expected_score=0.5, max_abs_margin=2.0, threshold=0.1) == "preserve"

    def test_near_zero_margin_with_expected_score_is_strengthen(self) -> None:
        from agent_tracegrad.diagnosis.runner import _classify_component

        assert _classify_component(margin=0.05, expected_score=0.5, max_abs_margin=2.0, threshold=0.1) == "strengthen"

    def test_near_zero_margin_without_expected_score_is_narrow(self) -> None:
        from agent_tracegrad.diagnosis.runner import _classify_component

        assert _classify_component(margin=0.05, expected_score=0.0, max_abs_margin=2.0, threshold=0.1) == "narrow"


class TestSerialization:
    def test_bad_only_serializes(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
        )

        payload = diagnosis_to_dict(result)
        assert "bad_result" in payload
        assert "expected_result" not in payload
        assert "contrastive_result" not in payload
        assert "margin_distributions" not in payload

    def test_full_diagnosis_serializes(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        payload = diagnosis_to_dict(result)
        assert "bad_result" in payload
        assert "expected_result" in payload
        assert "contrastive_result" in payload
        assert "margin_distributions" in payload
        for md in payload["margin_distributions"]:
            assert "grain" in md
            assert "contributions" in md
            for c in md["contributions"]:
                assert "classification" in c
                assert "margin" in c
        assert payload["bad_result"]["trace"]["nodes"]
        assert payload["bad_result"]["trace"]["spans"]
        assert payload["bad_result"]["trace"]["serialized_text"]

    def test_full_diagnosis_round_trips_for_drill(self) -> None:
        from agent_tracegrad.diagnosis import run_drill

        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        restored = diagnosis_from_dict(diagnosis_to_dict(result))
        drill = run_drill(restored)

        assert restored.bad_result.trace.serialized_text == result.bad_result.trace.serialized_text
        assert restored.contrastive_result.attribution.token_scores == result.contrastive_result.attribution.token_scores
        assert drill.atoms

    def test_write_diagnosis_json(self, tmp_path) -> None:
        from agent_tracegrad.diagnosis import write_diagnosis_json

        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        output = tmp_path / "diagnosis.json"
        write_diagnosis_json(result, output)
        assert output.exists()

        import json

        loaded = json.loads(output.read_text())
        assert loaded["metadata"]["mode"] == "full_diagnosis"

    def test_write_diagnosis_markdown(self, tmp_path) -> None:
        from agent_tracegrad.diagnosis import write_diagnosis_markdown

        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        output = tmp_path / "diagnosis.md"
        write_diagnosis_markdown(result, output)
        text = output.read_text()
        assert "# Agent TraceGrad Diagnosis Report" in text
        assert "Component Ranking" in text


class TestDecisionBoundaryArtifact:
    def test_builds_directional_boundary_from_full_diagnosis(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        artifact = build_decision_boundary_artifact(result)
        payload = decision_boundary_to_dict(artifact)

        assert payload["target"]["target_id"] == result.bad_result.target.target_id
        assert payload["objective_formula"] == "log P(bad_target | context) - log P(expected_target | context)"
        assert payload["components"]
        assert all(item["direction"] in ("pushes_bad", "pushes_expected", "neutral") for item in payload["components"])

    def test_boundary_markdown_has_decision_sections(self) -> None:
        result = run_diagnosis(
            _make_raw_trace(),
            model=TinyBackwardModel(),
            target_node_ids=("agent-1",),
            expected_target_text="five six",
        )

        markdown = decision_boundary_to_markdown(build_decision_boundary_artifact(result))

        assert "# Agent TraceGrad Decision Boundary" in markdown
        assert "Components Pushing Toward Bad" in markdown
        assert "Components Supporting Expected" in markdown
