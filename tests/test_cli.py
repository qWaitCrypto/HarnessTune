from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agent_tracegrad.cli import build_parser, main
from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput


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
class FakeCliModel:
    name: str = "fake-cli-model"

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
        return ModelForwardOutput(logits=inputs_embeds.cumsum(dim=1))

    def chat_template_supported(self) -> bool:
        return False


def test_cli_analyze_parser_accepts_single_trace_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--trace",
            "trace.json",
            "--model",
            "/models/formal",
            "--target-node-id",
            "agent-1",
            "--output",
            "out.json",
            "--device",
            "cuda:0",
            "--devices",
            "cuda:0,cuda:1",
            "--dtype",
            "bfloat16",
        ]
    )

    assert args.command == "analyze"
    assert args.input_format == "json-fixture"
    assert args.target_node_id == ["agent-1"]
    assert args.device == "cuda:0"
    assert args.devices == "cuda:0,cuda:1"


def test_cli_analyze_parser_accepts_agentpi_marker_mode() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--trace",
            "trace.json",
            "--input-format",
            "agentpi-raw",
            "--model",
            "/models/formal",
            "--target-marker",
            "last-agent-output",
            "--output",
            "out.json",
        ]
    )

    assert args.command == "analyze"
    assert args.input_format == "agentpi-raw"
    assert args.target_node_id is None
    assert args.target_marker == "last-agent-output"


def test_cli_evaluate_parser_accepts_objective_and_operator_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate",
            "--trace",
            "trace.json",
            "--input-format",
            "agentpi-raw",
            "--model",
            "/models/formal",
            "--target-marker",
            "last-agent-output",
            "--objective-type",
            "contrastive",
            "--expected-target-text",
            "I cannot do that.",
            "--operator-config",
            '{"operator":"replace_with_placeholder","parameters":{"placeholder":"masked"}}',
            "--output-dir",
            "out",
            "--metric-k",
            "1",
            "--metric-k",
            "3",
            "--ablation-k",
            "1",
            "--ablation-k",
            "2",
        ]
    )

    assert args.command == "evaluate"
    assert args.objective_type == "contrastive"
    assert args.expected_target_text == "I cannot do that."
    assert args.metric_k == [1, 3]
    assert args.ablation_k == [1, 2]


def test_cli_evaluate_writes_artifacts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    output_dir = tmp_path / "out"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "policy text",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "user-1",
                        "block_role": "user",
                        "sub_block_kind": "user.content",
                        "content": "user clue",
                        "sequence_index": 1,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained",
        lambda *args, **kwargs: FakeCliModel(),
    )

    exit_code = main(
        [
            "evaluate",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--objective-type",
            "expected_action",
            "--expected-target-text",
            "right answer",
            "--operator-config",
            '{"operator":"replace_with_placeholder","parameters":{"placeholder":"masked"}}',
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "eval",
            "--max-samples",
            "1",
            "--metric-k",
            "1",
            "--ablation-k",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "eval.json").exists()
    assert (output_dir / "eval.jsonl").exists()
    assert (output_dir / "eval.md").exists()
    payload = json.loads((output_dir / "eval.json").read_text(encoding="utf-8"))
    assert payload["context"]["objective"]["objective_type"] == "expected_action"
    assert payload["ablation_curve"]


def test_cli_diagnose_writes_artifacts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    output_dir = tmp_path / "diagnose"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "policy text",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "user-1",
                        "block_role": "user",
                        "sub_block_kind": "user.content",
                        "content": "user clue",
                        "sequence_index": 1,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained",
        lambda *args, **kwargs: FakeCliModel(),
    )

    exit_code = main(
        [
            "diagnose",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "diag",
            "--ablation-k",
            "1",
            "--control-ablation",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "diag.json").exists()
    assert (output_dir / "diag.md").exists()
    assert (output_dir / "diag.html").exists()
    payload = json.loads((output_dir / "diag.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["mode"] == "full_diagnosis"
    assert payload["confidence_level"] in ("medium", "strong")
    assert payload["margin_distributions"]
    assert payload["evidence"]
    assert payload["ablations"]


def test_cli_drill_writes_artifacts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    output_dir = tmp_path / "drill"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "# Policy\n- First call transfer_to_human_agents.\n- Refuse after deadline.",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained",
        lambda *args, **kwargs: FakeCliModel(),
    )

    exit_code = main(
        [
            "drill",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "drill",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "drill.json").exists()
    assert (output_dir / "drill.md").exists()
    payload = json.loads((output_dir / "drill.json").read_text(encoding="utf-8"))
    assert payload["atoms"]


def test_cli_drill_can_read_existing_diagnose_artifact(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    diagnose_dir = tmp_path / "diagnose"
    drill_dir = tmp_path / "drill"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "# Policy\n- First call transfer_to_human_agents.\n- Refuse after deadline.",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    load_count = 0

    def fake_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return FakeCliModel()

    monkeypatch.setattr("agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained", fake_load)

    assert main(
        [
            "diagnose",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(diagnose_dir),
            "--output-prefix",
            "diag",
        ]
    ) == 0
    assert load_count == 1

    assert main(
        [
            "drill",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--diagnose-result",
            str(diagnose_dir / "diag.json"),
            "--output-dir",
            str(drill_dir),
            "--output-prefix",
            "drill",
        ]
    ) == 0
    assert load_count == 1
    assert (drill_dir / "drill.json").exists()


def test_cli_drill_writes_influence_matrix_when_candidates_are_provided(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    output_dir = tmp_path / "drill"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "# Policy\n- First call transfer_to_human_agents.\n- Refuse after deadline.",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained",
        lambda *args, **kwargs: FakeCliModel(),
    )

    exit_code = main(
        [
            "drill",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "drill",
            "--candidate-action",
            "refuse=right answer",
            "--candidate-action",
            "transfer=wrong answer",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "drill-matrix.json").exists()
    assert (output_dir / "drill-matrix.md").exists()
    payload = json.loads((output_dir / "drill-matrix.json").read_text(encoding="utf-8"))
    assert [candidate["action_id"] for candidate in payload["candidates"]] == ["refuse", "transfer"]
    assert payload["rows"]


def test_cli_landscape_writes_artifacts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_dir = tmp_path / "traces"
    output_dir = tmp_path / "landscape"
    trace_dir.mkdir()
    for index in range(2):
        (trace_dir / f"trace-{index}.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "node_id": "sys-1",
                            "block_role": "system",
                            "sub_block_kind": "system.instruction",
                            "content": f"policy text {index}",
                            "sequence_index": 0,
                        },
                        {
                            "node_id": "agent-1",
                            "block_role": "agent",
                            "sub_block_kind": "agent.content",
                            "content": "wrong answer",
                            "sequence_index": 1,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained",
        lambda *args, **kwargs: FakeCliModel(),
    )

    exit_code = main(
        [
            "landscape",
            "--traces",
            str(trace_dir),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "landscape",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "landscape.json").exists()
    assert (output_dir / "landscape.md").exists()
    assert (output_dir / "landscape.html").exists()
    payload = json.loads((output_dir / "landscape.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["trace_count"] == 2
    assert payload["component_stats"]
    assert payload["clusters"]


def test_cli_landscape_can_read_diagnose_artifacts_without_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    trace_path = tmp_path / "trace.json"
    diagnose_dir = tmp_path / "diagnose"
    landscape_dir = tmp_path / "landscape"
    trace_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node_id": "sys-1",
                        "block_role": "system",
                        "sub_block_kind": "system.instruction",
                        "content": "policy text",
                        "sequence_index": 0,
                    },
                    {
                        "node_id": "agent-1",
                        "block_role": "agent",
                        "sub_block_kind": "agent.content",
                        "content": "wrong answer",
                        "sequence_index": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    load_count = 0

    def fake_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return FakeCliModel()

    monkeypatch.setattr("agent_tracegrad.cli.HuggingFaceCausalLMAdapter.from_pretrained", fake_load)

    assert main(
        [
            "diagnose",
            "--trace",
            str(trace_path),
            "--model",
            "fake-model",
            "--target-node-id",
            "agent-1",
            "--expected-target-text",
            "right answer",
            "--output-dir",
            str(diagnose_dir),
            "--output-prefix",
            "diag",
        ]
    ) == 0
    assert load_count == 1

    assert main(
        [
            "landscape",
            "--diagnose-results",
            str(diagnose_dir),
            "--output-dir",
            str(landscape_dir),
            "--output-prefix",
            "landscape",
        ]
    ) == 0
    assert load_count == 1
    assert (landscape_dir / "landscape.json").exists()
