from __future__ import annotations

from agent_tracegrad.analysis import analyze_normalized_trace
from agent_tracegrad.evaluation import build_evidence_report, evidence_report_to_dict

from tests.test_single_trace_analysis import TinyBackwardModel, WhitespaceOffsetTokenizer, make_raw_trace


def test_build_evidence_report_returns_context_tokens_and_windows() -> None:
    result = analyze_normalized_trace(
        make_raw_trace(),
        target_node_ids=("agent-1",),
        model=TinyBackwardModel(),
        tokenizer=WhitespaceOffsetTokenizer(),
    )

    report = build_evidence_report(result, top_tokens=3, top_windows=2)
    payload = evidence_report_to_dict(report)

    assert len(report.top_tokens) <= 3
    assert len(report.top_windows) <= 2
    assert all(token.block_role in {"system", "user"} for token in report.top_tokens)
    assert all(window.block_role in {"system", "user"} for window in report.top_windows)
    assert payload["top_tokens"][0]["node_id"] in {"sys-1", "user-1"}
    assert payload["top_windows"][0]["text"]
