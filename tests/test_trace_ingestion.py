from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import pytest

from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput
from agent_tracegrad.target import LastAgentOutputMarker
from agent_tracegrad.trace import AgentPIRawTraceAdapter, TraceSerializer, get_trace_adapter


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


def make_agentpi_raw() -> dict[str, object]:
    return {
        "source_repo": "AgentPI",
        "source_file": "/tmp/results.json",
        "simulation_index": 3,
        "task": {"id": "task-1"},
        "simulation": {
            "id": "sim-1",
            "task_id": "task-1",
            "start_time": "2026-05-29T00:00:00",
            "termination_reason": "user_stop",
            "reward_info": {"reward": 0.0},
            "review": {"note": "failed"},
            "hallucination_check": {"ok": False},
            "policy": "Follow policy.",
            "messages": [
                {
                    "role": "user",
                    "content": "Please do the task.",
                    "turn_idx": 0,
                    "timestamp": "2026-05-29T00:00:01",
                },
                {
                    "role": "assistant",
                    "content": "I will call a tool.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "lookup",
                            "arguments": {"id": 123},
                            "requestor": "assistant",
                        }
                    ],
                    "turn_idx": 1,
                    "timestamp": "2026-05-29T00:00:02",
                },
                {
                    "id": "call-1",
                    "role": "tool",
                    "content": '{"ok": true}',
                    "requestor": "assistant",
                    "turn_idx": 2,
                    "timestamp": "2026-05-29T00:00:03",
                },
                {
                    "role": "assistant",
                    "content": "Final answer.",
                    "turn_idx": 3,
                    "timestamp": "2026-05-29T00:00:04",
                },
            ],
        },
    }


def test_agentpi_raw_adapter_maps_messages_to_canonical_nodes() -> None:
    nodes = AgentPIRawTraceAdapter().adapt(make_agentpi_raw())

    assert [(node.node_id, node.block_role, node.sub_block_kind) for node in nodes] == [
        ("agentpi:simulation:policy", "system", "system.instruction"),
        ("agentpi:message-0:user-content", "user", "user.content"),
        ("agentpi:message-1:assistant-content", "agent", "agent.content"),
        ("agentpi:message-1:assistant-tool-call", "agent", "agent.tool_call"),
        ("agentpi:message-2:tool-result", "user", "user.tool_result"),
        ("agentpi:message-3:assistant-content", "agent", "agent.content"),
    ]
    tool_call = nodes[3]
    assert json.loads(tool_call.content) == [
        {"arguments": {"id": 123}, "id": "call-1", "name": "lookup", "requestor": "assistant"}
    ]
    assert tool_call.metadata["tool_call_ids"] == ("call-1",)
    assert nodes[4].parents == ("call-1",)
    assert nodes[5].metadata["source_repo"] == "AgentPI"
    assert [node.sequence_index for node in nodes] == list(range(len(nodes)))


def test_agentpi_raw_adapter_ingests_repository_samples() -> None:
    adapter = get_trace_adapter("agentpi-raw")
    for path in sorted(Path("samples/agentpi_raw").glob("*.json")):
        raw_trace = json.loads(path.read_text(encoding="utf-8"))
        nodes = adapter.adapt(raw_trace)
        assert nodes
        assert all(node.content for node in nodes)
        assert all(node.sequence_index == index for index, node in enumerate(nodes))


def test_last_agent_output_marker_prefers_last_agent_content() -> None:
    nodes = AgentPIRawTraceAdapter().adapt(make_agentpi_raw())
    trace = TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes)

    targets = LastAgentOutputMarker().mark(trace)

    assert len(targets) == 1
    assert targets[0].node_ids == ("agentpi:message-3:assistant-content",)


def test_last_agent_output_marker_falls_back_to_tool_call() -> None:
    raw_trace = make_agentpi_raw()
    messages = raw_trace["simulation"]["messages"]  # type: ignore[index]
    messages[1]["content"] = None  # type: ignore[index]
    messages[-1]["role"] = "user"  # type: ignore[index]
    nodes = AgentPIRawTraceAdapter().adapt(raw_trace)
    trace = TraceSerializer(WhitespaceOffsetTokenizer()).serialize(nodes)

    targets = LastAgentOutputMarker().mark(trace)

    assert len(targets) == 1
    assert targets[0].node_ids == ("agentpi:message-1:assistant-tool-call",)


def test_agentpi_raw_can_enter_analysis_pipeline_with_default_marker() -> None:
    torch = pytest.importorskip("torch")
    from agent_tracegrad.analysis import analyze_trace

    @dataclass
    class LargeTinyBackwardModel:
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

    result = analyze_trace(
        make_agentpi_raw(),
        input_format="agentpi-raw",
        model=LargeTinyBackwardModel(),
        execution_model_name="tiny-backward-model",
    )

    assert result.trace.metadata["trace_adapter"] == "agentpi-raw"
    assert result.target.node_ids == ("agentpi:message-3:assistant-content",)
    assert result.rankings


def test_agentpi_node_ids_use_message_index_when_turn_idx_repeats() -> None:
    raw_trace = make_agentpi_raw()
    messages = raw_trace["simulation"]["messages"]  # type: ignore[index]
    for message in messages:
        message["turn_idx"] = 7

    nodes = AgentPIRawTraceAdapter().adapt(raw_trace)

    node_ids = [node.node_id for node in nodes]
    assert len(node_ids) == len(set(node_ids))
    assert "agentpi:message-1:assistant-content" in node_ids
    assert "agentpi:message-1:assistant-tool-call" in node_ids
