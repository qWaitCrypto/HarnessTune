"""Policy-action influence matrix built from repeated contrastive diagnoses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.diagnosis.drill import DrillResult, run_drill
from agent_tracegrad.diagnosis.runner import run_diagnosis
from agent_tracegrad.model.adapter import ModelAdapter


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("candidate action_id is required")
        if not self.text:
            raise ValueError("candidate text is required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class InfluenceCell:
    action_id: str
    bad_score: float
    expected_score: float
    margin: float
    classification: str


@dataclass(frozen=True)
class InfluenceRow:
    atom_id: str
    source_node_id: str
    atom_kind: str
    sub_block_kind: str
    text: str
    cells: Sequence[InfluenceCell]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class InfluenceMatrixResult:
    candidates: Sequence[CandidateAction]
    rows: Sequence[InfluenceRow]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def run_influence_matrix(
    raw_trace: Any,
    *,
    model: ModelAdapter,
    candidates: Sequence[CandidateAction],
    input_format: str = "json-fixture",
    target_node_ids: Sequence[str] | None = None,
    target_marker: str | None = None,
    target_id: str = "target-1",
    target_span: tuple[int, int] | None = None,
    method: str = "gradient_saliency",
    execution_model_name: str | None = None,
    topk_mean_k: int = 5,
    ranking_grain: str = "node",
    ranking_view: str = "sum",
    integrated_gradients_steps: int = 16,
    trace_metadata: Mapping[str, Any] | None = None,
) -> InfluenceMatrixResult:
    if not candidates:
        raise ValueError("influence matrix requires at least one candidate")
    drills: list[tuple[CandidateAction, DrillResult]] = []
    for candidate in candidates:
        diagnosis = run_diagnosis(
            raw_trace,
            model=model,
            expected_target_text=candidate.text,
            expected_target_id=candidate.action_id,
            input_format=input_format,
            target_node_ids=target_node_ids,
            target_marker=target_marker,
            target_id=target_id,
            target_span=target_span,
            method=method,
            execution_model_name=execution_model_name,
            topk_mean_k=topk_mean_k,
            ranking_grain=ranking_grain,
            ranking_view=ranking_view,
            integrated_gradients_steps=integrated_gradients_steps,
            trace_metadata={
                **dict(trace_metadata or {}),
                "influence_candidate_action_id": candidate.action_id,
            },
        )
        drills.append((candidate, run_drill(diagnosis)))
    return _merge_drills(tuple(drills))


def influence_matrix_to_dict(result: InfluenceMatrixResult) -> dict[str, Any]:
    return {
        "metadata": dict(result.metadata),
        "candidates": [
            {
                "action_id": candidate.action_id,
                "text": candidate.text,
                "metadata": dict(candidate.metadata),
            }
            for candidate in result.candidates
        ],
        "rows": [
            {
                "atom_id": row.atom_id,
                "source_node_id": row.source_node_id,
                "atom_kind": row.atom_kind,
                "sub_block_kind": row.sub_block_kind,
                "text": row.text,
                "metadata": dict(row.metadata),
                "cells": [
                    {
                        "action_id": cell.action_id,
                        "bad_score": cell.bad_score,
                        "expected_score": cell.expected_score,
                        "margin": cell.margin,
                        "classification": cell.classification,
                    }
                    for cell in row.cells
                ],
            }
            for row in result.rows
        ],
    }


def influence_matrix_to_markdown(result: InfluenceMatrixResult) -> str:
    lines = ["# Agent TraceGrad Influence Matrix", ""]
    headers = ["Atom", "Kind", *[candidate.action_id for candidate in result.candidates]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---", "---", *[("---:" ) for _ in result.candidates]]) + " |")
    for row in result.rows[:40]:
        text = " ".join(row.text.split())
        if len(text) > 80:
            text = text[:77].rstrip() + "..."
        text = text.replace("|", "\\|")
        cells = [f"{cell.margin:.6g}" for cell in row.cells]
        lines.append(f"| `{row.atom_id}`<br>{text} | `{row.atom_kind}` | " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"


def write_influence_matrix_json(result: InfluenceMatrixResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(influence_matrix_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def write_influence_matrix_markdown(result: InfluenceMatrixResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(influence_matrix_to_markdown(result), encoding="utf-8")


def _merge_drills(drills: Sequence[tuple[CandidateAction, DrillResult]]) -> InfluenceMatrixResult:
    row_order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for candidate, drill in drills:
        for atom in drill.atoms:
            atom_id = atom.atom.atom_id
            if atom_id not in rows:
                row_order.append(atom_id)
                rows[atom_id] = {
                    "atom": atom,
                    "cells": {},
                }
            rows[atom_id]["cells"][candidate.action_id] = InfluenceCell(
                action_id=candidate.action_id,
                bad_score=atom.bad_score,
                expected_score=atom.expected_score,
                margin=atom.margin,
                classification=atom.classification,
            )
    merged_rows: list[InfluenceRow] = []
    candidates = tuple(candidate for candidate, _ in drills)
    for atom_id in row_order:
        record = rows[atom_id]
        atom = record["atom"]
        cells = tuple(
            record["cells"].get(
                candidate.action_id,
                InfluenceCell(
                    action_id=candidate.action_id,
                    bad_score=0.0,
                    expected_score=0.0,
                    margin=0.0,
                    classification="strengthen",
                ),
            )
            for candidate in candidates
        )
        merged_rows.append(
            InfluenceRow(
                atom_id=atom_id,
                source_node_id=atom.atom.source_node_id,
                atom_kind=atom.atom.atom_kind,
                sub_block_kind=atom.sub_block_kind,
                text=atom.atom.text,
                cells=cells,
                metadata=dict(atom.atom.metadata),
            )
        )
    merged_rows.sort(key=lambda row: (-max(abs(cell.margin) for cell in row.cells), row.atom_id))
    return InfluenceMatrixResult(
        candidates=candidates,
        rows=tuple(merged_rows),
        metadata={
            "row_count": len(merged_rows),
            "candidate_count": len(candidates),
            "source": "diagnosis-influence-matrix",
        },
    )
