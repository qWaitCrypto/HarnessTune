"""Rule/tool-part drill-down over diagnosis attribution results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.analysis.single_trace import SingleTraceAnalysisResult
from agent_tracegrad.diagnosis.atomizer import ComponentAtom, atomize_node
from agent_tracegrad.diagnosis.types import ComponentClassification, DiagnosisResult


@dataclass(frozen=True)
class AtomAttribution:
    atom: ComponentAtom
    sub_block_kind: str
    token_count: int
    bad_score: float
    expected_score: float
    margin: float
    classification: ComponentClassification


@dataclass(frozen=True)
class DrillResult:
    atoms: Sequence[AtomAttribution]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def run_drill(
    diagnosis: DiagnosisResult,
    *,
    include_sub_block_kinds: Sequence[str] = ("system.instruction", "system.tool_schema"),
) -> DrillResult:
    if diagnosis.expected_result is None or diagnosis.contrastive_result is None:
        raise ValueError("drill requires a full diagnosis with expected and contrastive results")
    allowed_kinds = set(include_sub_block_kinds)
    atoms: list[AtomAttribution] = []
    for node in sorted(diagnosis.contrastive_result.trace.nodes.values(), key=lambda item: (item.sequence_index or 0, item.node_id)):
        if node.sub_block_kind not in allowed_kinds:
            continue
        for atom in atomize_node(node):
            bad_score, token_count = _score_atom(diagnosis.bad_result, atom)
            expected_score, _ = _score_atom(diagnosis.expected_result, atom)
            margin, _ = _score_atom(diagnosis.contrastive_result, atom)
            atoms.append(
                AtomAttribution(
                    atom=atom,
                    sub_block_kind=node.sub_block_kind,
                    token_count=token_count,
                    bad_score=bad_score,
                    expected_score=expected_score,
                    margin=margin,
                    classification=_classify_atom(margin, expected_score, atoms),
                )
            )
    ranked = tuple(sorted(atoms, key=lambda item: (-abs(item.margin), item.atom.atom_id)))
    return DrillResult(
        atoms=ranked,
        metadata={
            "atom_count": len(ranked),
            "source": "diagnosis-drill",
        },
    )


def drill_result_to_dict(result: DrillResult) -> dict[str, Any]:
    return {
        "metadata": dict(result.metadata),
        "atoms": [
            {
                "rank": index + 1,
                "atom_id": item.atom.atom_id,
                "source_node_id": item.atom.source_node_id,
                "atom_kind": item.atom.atom_kind,
                "sub_block_kind": item.sub_block_kind,
                "text": item.atom.text,
                "char_start": item.atom.char_start,
                "char_end": item.atom.char_end,
                "token_count": item.token_count,
                "bad_score": item.bad_score,
                "expected_score": item.expected_score,
                "margin": item.margin,
                "classification": item.classification,
                "metadata": dict(item.atom.metadata),
            }
            for index, item in enumerate(result.atoms)
        ],
    }


def drill_result_to_markdown(result: DrillResult) -> str:
    lines = [
        "# Agent TraceGrad Drill Report",
        "",
        "## Atom Ranking",
        "",
        "| Rank | Atom | Kind | Margin | Bad | Expected | Class | Evidence |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, item in enumerate(result.atoms[:30], start=1):
        text = " ".join(item.atom.text.split())
        if len(text) > 120:
            text = text[:117].rstrip() + "..."
        text = text.replace("|", "\\|")
        lines.append(
            f"| {rank} | `{item.atom.atom_id}` | `{item.atom.atom_kind}` | "
            f"{item.margin:.6g} | {item.bad_score:.6g} | {item.expected_score:.6g} | "
            f"`{item.classification}` | {text} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_drill_json(result: DrillResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(drill_result_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def write_drill_markdown(result: DrillResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(drill_result_to_markdown(result), encoding="utf-8")


def _score_atom(analysis: SingleTraceAnalysisResult, atom: ComponentAtom) -> tuple[float, int]:
    span = next((item for item in analysis.trace.spans if item.node_id == atom.source_node_id), None)
    if span is None:
        return 0.0, 0
    token_indexes = _atom_token_indexes(
        analysis.trace.serialized_text,
        span.text_start_char,
        span.text_end_char,
        span.start_token,
        span.end_token,
        atom.char_start,
        atom.char_end,
    )
    return sum(analysis.attribution.token_scores[index] for index in token_indexes), len(token_indexes)


def _atom_token_indexes(
    serialized_text: str,
    node_char_start: int | None,
    node_char_end: int | None,
    node_token_start: int,
    node_token_end: int,
    atom_char_start: int,
    atom_char_end: int,
) -> tuple[int, ...]:
    if node_char_start is None or node_char_end is None or node_char_end <= node_char_start:
        return ()
    token_count = node_token_end - node_token_start
    if token_count <= 0:
        return ()
    abs_start = node_char_start + atom_char_start
    abs_end = node_char_start + atom_char_end
    offsets = _token_char_offsets(
        serialized_text,
        node_char_start,
        node_char_end,
        token_count=token_count,
    )
    indexes: list[int] = []
    for offset_index, (start, end) in enumerate(offsets):
        if start < abs_end and end > abs_start:
            indexes.append(node_token_start + offset_index)
    return tuple(indexes)


def _token_char_offsets(
    text: str,
    char_start: int,
    char_end: int,
    *,
    token_count: int,
) -> tuple[tuple[int, int], ...]:
    words = list(_word_offsets(text, char_start, char_end))
    if len(words) == token_count:
        return tuple(words)
    width = max(1, char_end - char_start)
    return tuple(
        (
            char_start + int(width * index / token_count),
            char_start + int(width * (index + 1) / token_count),
        )
        for index in range(token_count)
    )


def _word_offsets(text: str, start: int, end: int):
    cursor = start
    while cursor < end:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        token_start = cursor
        while cursor < end and not text[cursor].isspace():
            cursor += 1
        yield token_start, cursor


def _classify_atom(
    margin: float,
    expected_score: float,
    existing_atoms: Sequence[AtomAttribution],
) -> ComponentClassification:
    max_abs = max([abs(item.margin) for item in existing_atoms] + [abs(margin)], default=0.0)
    if max_abs > 0.0 and abs(margin) < 0.1 * max_abs and expected_score > 0.0:
        return "strengthen"
    if margin < 0.0:
        return "preserve"
    return "narrow"
