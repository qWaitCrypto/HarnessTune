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
    pending_atoms: list[tuple[ComponentAtom, str, int, float, float, float]] = []
    for node in sorted(diagnosis.contrastive_result.trace.nodes.values(), key=lambda item: (item.sequence_index or 0, item.node_id)):
        if node.sub_block_kind not in allowed_kinds:
            continue
        for atom in atomize_node(node):
            bad_score, token_count = _score_atom(diagnosis.bad_result, atom)
            expected_score, _ = _score_atom(diagnosis.expected_result, atom)
            margin, _ = _score_atom(diagnosis.contrastive_result, atom)
            pending_atoms.append((atom, node.sub_block_kind, token_count, bad_score, expected_score, margin))
    max_abs_margin = max((abs(margin) for *_, margin in pending_atoms), default=0.0)
    atoms: list[AtomAttribution] = []
    for atom, sub_block_kind, token_count, bad_score, expected_score, margin in pending_atoms:
        classification, _ = _classify_atom(
            margin,
            expected_score,
            max_abs_margin=max_abs_margin,
        )
        atoms.append(
            AtomAttribution(
                atom=atom,
                sub_block_kind=sub_block_kind,
                token_count=token_count,
                bad_score=bad_score,
                expected_score=expected_score,
                margin=margin,
                classification=classification,
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
        analysis.trace.token_offsets,
        span.text_start_char,
        span.start_token,
        span.end_token,
        atom.char_start,
        atom.char_end,
    )
    return sum(analysis.attribution.token_scores[index] for index in token_indexes), len(token_indexes)


def _atom_token_indexes(
    token_offsets: Sequence[tuple[int, int]],
    node_char_start: int | None,
    node_token_start: int,
    node_token_end: int,
    atom_char_start: int,
    atom_char_end: int,
) -> tuple[int, ...]:
    if not token_offsets:
        raise ValueError("SerializedTrace.token_offsets are required for exact drill attribution")
    if node_char_start is None:
        return ()
    if node_token_end <= node_token_start:
        return ()
    abs_start = node_char_start + atom_char_start
    abs_end = node_char_start + atom_char_end
    indexes: list[int] = []
    for token_index in range(node_token_start, node_token_end):
        start, end = token_offsets[token_index]
        if start < abs_end and end > abs_start:
            indexes.append(token_index)
    return tuple(indexes)


def _classify_atom(
    margin: float,
    expected_score: float,
    max_abs_margin: float,
) -> tuple[ComponentClassification, str]:
    if max_abs_margin > 0.0 and abs(margin) < 0.1 * max_abs_margin and expected_score > 0.0:
        return "strengthen", "expected_support_present_but_margin_near_zero"
    if margin < 0.0:
        return "preserve", "expected_support_exceeds_bad_support"
    return "narrow", "bad_support_exceeds_expected_support"
