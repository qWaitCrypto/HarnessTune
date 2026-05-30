"""Human-readable evidence extracted from token-level attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.analysis import SingleTraceAnalysisResult


@dataclass(frozen=True)
class TokenEvidence:
    token_index: int
    score: float
    text: str
    node_id: str
    block_role: str
    sub_block_kind: str
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class WindowEvidence:
    window_id: str
    score: float
    text: str
    node_id: str
    block_role: str
    sub_block_kind: str
    start_token: int
    end_token: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class EvidenceReport:
    top_tokens: Sequence[TokenEvidence]
    top_windows: Sequence[WindowEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_tokens", tuple(self.top_tokens))
        object.__setattr__(self, "top_windows", tuple(self.top_windows))


def build_evidence_report(
    analysis: SingleTraceAnalysisResult,
    *,
    top_tokens: int = 8,
    top_windows: int = 5,
    window_radius: int = 2,
) -> EvidenceReport:
    if top_tokens < 0:
        raise ValueError("top_tokens must be non-negative")
    if top_windows < 0:
        raise ValueError("top_windows must be non-negative")
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    token_evidence = _top_token_evidence(analysis, limit=top_tokens)
    window_evidence = _window_evidence(
        analysis,
        token_evidence,
        limit=top_windows,
        window_radius=window_radius,
    )
    return EvidenceReport(top_tokens=token_evidence, top_windows=window_evidence)


def evidence_report_to_dict(report: EvidenceReport) -> dict[str, Any]:
    return {
        "top_tokens": [
            {
                "token_index": token.token_index,
                "score": token.score,
                "text": token.text,
                "node_id": token.node_id,
                "block_role": token.block_role,
                "sub_block_kind": token.sub_block_kind,
                "char_start": token.char_start,
                "char_end": token.char_end,
            }
            for token in report.top_tokens
        ],
        "top_windows": [
            {
                "window_id": window.window_id,
                "score": window.score,
                "text": window.text,
                "node_id": window.node_id,
                "block_role": window.block_role,
                "sub_block_kind": window.sub_block_kind,
                "start_token": window.start_token,
                "end_token": window.end_token,
                "metadata": dict(window.metadata),
            }
            for window in report.top_windows
        ],
    }


def _top_token_evidence(analysis: SingleTraceAnalysisResult, *, limit: int) -> tuple[TokenEvidence, ...]:
    if limit == 0:
        return ()
    candidates: list[TokenEvidence] = []
    for span in analysis.trace.spans:
        if span.block_role == "agent":
            continue
        token_count = span.end_token - span.start_token
        char_offsets = _span_token_offsets(analysis, span.node_id)
        if len(char_offsets) != token_count:
            raise ValueError("SerializedTrace token_offsets must align with span token ranges")
        for offset_index, token_index in enumerate(range(span.start_token, span.end_token)):
            score = analysis.attribution.token_scores[token_index]
            if score <= 0.0:
                continue
            char_start, char_end = char_offsets[offset_index]
            candidates.append(
                TokenEvidence(
                    token_index=token_index,
                    score=score,
                    text=analysis.trace.serialized_text[char_start:char_end],
                    node_id=span.node_id,
                    block_role=span.block_role,
                    sub_block_kind=span.sub_block_kind,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    candidates.sort(key=lambda item: (-item.score, item.token_index))
    return tuple(candidates[:limit])


def _window_evidence(
    analysis: SingleTraceAnalysisResult,
    tokens: Sequence[TokenEvidence],
    *,
    limit: int,
    window_radius: int,
) -> tuple[WindowEvidence, ...]:
    if limit == 0:
        return ()
    windows: dict[tuple[str, int, int], WindowEvidence] = {}
    span_by_node = {span.node_id: span for span in analysis.trace.spans}
    for token in tokens:
        span = span_by_node[token.node_id]
        start_token = max(span.start_token, token.token_index - window_radius)
        end_token = min(span.end_token, token.token_index + window_radius + 1)
        key = (span.node_id, start_token, end_token)
        if key in windows:
            continue
        score = sum(analysis.attribution.token_scores[start_token:end_token])
        char_start, char_end = _window_char_range(analysis, span.node_id, start_token, end_token)
        windows[key] = WindowEvidence(
            window_id=f"{span.node_id}:{start_token}-{end_token}",
            score=score,
            text=_compact_text(analysis.trace.serialized_text[char_start:char_end]),
            node_id=span.node_id,
            block_role=span.block_role,
            sub_block_kind=span.sub_block_kind,
            start_token=start_token,
            end_token=end_token,
            metadata={"center_token": token.token_index},
        )
    ranked = sorted(windows.values(), key=lambda item: (-item.score, item.window_id))
    return tuple(ranked[:limit])


def _window_char_range(
    analysis: SingleTraceAnalysisResult,
    node_id: str,
    start_token: int,
    end_token: int,
) -> tuple[int, int]:
    span = next(item for item in analysis.trace.spans if item.node_id == node_id)
    char_offsets = _span_token_offsets(analysis, node_id)
    start_index = start_token - span.start_token
    end_index = end_token - span.start_token - 1
    return char_offsets[start_index][0], char_offsets[end_index][1]


def _span_token_offsets(
    analysis: SingleTraceAnalysisResult,
    node_id: str,
) -> tuple[tuple[int, int], ...]:
    span = next(item for item in analysis.trace.spans if item.node_id == node_id)
    if not analysis.trace.token_offsets:
        raise ValueError("SerializedTrace.token_offsets are required for exact evidence extraction")
    return tuple(analysis.trace.token_offsets[span.start_token:span.end_token])


def _compact_text(text: str, *, max_length: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."
