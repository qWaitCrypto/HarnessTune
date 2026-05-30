from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agent_tracegrad.evaluation import (
    evaluation_run_to_dict,
    evaluation_run_to_jsonl_records,
    evaluation_run_to_markdown,
    run_trace_level_evaluation,
    write_evaluation_artifacts,
)
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput

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


def make_run_result():
    pytest.importorskip("torch")
    return run_trace_level_evaluation(
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
        max_samples=1,
        metric_ks=(1,),
    )


def test_evaluation_run_to_dict_is_json_ready() -> None:
    payload = evaluation_run_to_dict(make_run_result())

    json.dumps(payload)
    assert payload["context"]["objective"]["objective_type"] == "bad_action"
    assert payload["baseline_analysis"]["attribution"]["metadata"]["loss"] is not None
    assert payload["ablation_curve"]
    assert payload["ablation_curve"][0]["evidence"]["top_tokens"]
    assert payload["context"]["targets"][0]["node_ids"] == ["agentpi:message-3:assistant-content"]
    assert payload["sample_results"][0]["sample"]["label"]["target_node_ids"]
    assert payload["sample_results"][0]["analysis"]["rankings"]
    assert payload["sample_results"][0]["evidence"]["top_tokens"]
    assert payload["sample_results"][0]["evidence"]["top_windows"]
    assert payload["sample_results"][0]["metrics"]


def test_evaluation_run_to_jsonl_records_returns_one_record_per_sample() -> None:
    result = make_run_result()

    records = evaluation_run_to_jsonl_records(result)

    assert len(records) == 1
    assert records[0]["record_type"] == "evaluation_sample"
    assert records[0]["objective"]["objective_type"] == "bad_action"
    assert records[0]["ablation_curve"]
    assert records[0]["sample"]["spec"]["operator"] == "replace_with_placeholder"


def test_write_evaluation_artifacts_writes_json_and_jsonl(tmp_path) -> None:
    paths = write_evaluation_artifacts(make_run_result(), output_dir=tmp_path, prefix="eval")

    aggregate = json.loads(paths["aggregate_json"].read_text(encoding="utf-8"))
    jsonl_lines = paths["samples_jsonl"].read_text(encoding="utf-8").splitlines()
    markdown = paths["markdown_report"].read_text(encoding="utf-8")

    assert aggregate["sample_results"]
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["record_type"] == "evaluation_sample"
    assert "# TraceGrad Evaluation Report" in markdown
    assert "## Ablation Curve" in markdown
    assert "Top windows:" in markdown
    assert "Top tokens:" in markdown


def test_evaluation_run_to_markdown_includes_objective_and_delta_metrics() -> None:
    markdown = evaluation_run_to_markdown(make_run_result())

    assert "objective_type" in markdown
    assert "bad_action" in markdown
    assert "delta_ll@" in markdown
    assert "Ablation Curve" in markdown
    assert "Top windows:" in markdown
