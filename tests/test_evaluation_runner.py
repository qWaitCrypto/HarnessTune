from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_tracegrad.evaluation import run_trace_level_evaluation
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.target import ExpectedTarget, TargetObjective

from tests.test_trace_ingestion import make_agentpi_raw


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
class LargeTinyBackwardModel:
    name: str = "tiny-backward-model"

    @property
    def tokenizer(self):
        return WhitespaceOffsetTokenizer()

    def tokenize(self, text: str) -> TokenizedOutput:
        import torch

        token_count = len(text.split())
        return TokenizedOutput(
            input_ids=torch.arange(token_count, dtype=torch.long).unsqueeze(0),
            attention_mask=torch.ones((1, token_count), dtype=torch.long),
        )

    def input_embeddings(self, input_ids, *, requires_grad: bool):
        import torch

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


def test_run_trace_level_evaluation_for_agentpi_raw() -> None:
    pytest.importorskip("torch")

    result = run_trace_level_evaluation(
        make_agentpi_raw(),
        model=LargeTinyBackwardModel(),
        input_format="agentpi-raw",
        target_marker="last-agent-output",
        operator_configs=(
            {
                "operator": "replace_with_placeholder",
                "parameters": {"placeholder": "masked"},
            },
        ),
        max_samples=2,
        metric_ks=(1, 2),
    )

    assert result.context.targets[0].node_ids == ("agentpi:message-3:assistant-content",)
    assert len(result.sample_results) == 2
    assert result.summary
    assert all(sample_result.metrics for sample_result in result.sample_results)
    assert "delta_ll@k" in {metric.metric_name for metric in result.sample_results[0].metrics}
    assert result.baseline_analysis.attribution.metadata["loss"] is not None
    assert [point.k for point in result.ablation_curve] == [1, 3, 5]
    assert all(point.target_node_ids for point in result.ablation_curve)
    assert "ablation_delta_ll@k" in result.summary


def test_run_trace_level_evaluation_uses_expected_action_objective() -> None:
    pytest.importorskip("torch")
    objective = TargetObjective.expected_action(
        ExpectedTarget(target_id="gold-refusal", content="cannot cancel", source="human")
    )

    result = run_trace_level_evaluation(
        make_agentpi_raw(),
        model=LargeTinyBackwardModel(),
        input_format="agentpi-raw",
        target_marker="last-agent-output",
        objective=objective,
        operator_configs=(
            {
                "operator": "replace_with_placeholder",
                "parameters": {"placeholder": "masked"},
            },
        ),
        max_samples=1,
        ablation_ks=(1,),
    )

    assert result.context.objective.objective_type == "expected_action"
    assert result.sample_results[0].analysis.objective.expected_target == objective.expected_target
    assert result.sample_results[0].analysis.objective.bad_target is not None
    assert result.sample_results[0].analysis.attribution.target_id == "gold-refusal"
    assert result.sample_results[0].analysis.attribution.metadata["objective_anchor"]["mode"] == "failure_target_prefix"
    assert result.ablation_curve
