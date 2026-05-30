"""Split editable harness components into attribution atoms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_tracegrad.trace.schema import TraceNode


@dataclass(frozen=True)
class ComponentAtom:
    atom_id: str
    source_node_id: str
    atom_kind: str
    text: str
    char_start: int
    char_end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ValueError("atom_id is required")
        if not self.source_node_id:
            raise ValueError("source_node_id is required")
        if not self.atom_kind:
            raise ValueError("atom_kind is required")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("atom char range must be a valid half-open range")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


def atomize_node(node: TraceNode) -> tuple[ComponentAtom, ...]:
    if node.sub_block_kind == "system.instruction":
        return atomize_policy_text(node)
    if node.sub_block_kind == "system.tool_schema":
        return atomize_tool_schema(node)
    return ()


def atomize_policy_text(node: TraceNode) -> tuple[ComponentAtom, ...]:
    """Split policy text into markdown-ish headings, list items, and paragraphs."""

    atoms: list[ComponentAtom] = []
    pending_paragraph: list[tuple[int, int, str]] = []
    for match in re.finditer(r".*(?:\n|$)", node.content):
        line = match.group(0)
        if not line:
            continue
        line_start = match.start()
        line_text = line.rstrip("\n")
        stripped = line_text.strip()
        if not stripped:
            _flush_paragraph(node, atoms, pending_paragraph)
            pending_paragraph = []
            continue
        atom_kind = _policy_line_kind(stripped)
        if atom_kind is None:
            pending_paragraph.append((line_start, line_start + len(line_text), stripped))
            continue
        _flush_paragraph(node, atoms, pending_paragraph)
        pending_paragraph = []
        start = line_start + len(line_text) - len(line_text.lstrip())
        end = line_start + len(line_text.rstrip())
        atoms.append(_atom(node, len(atoms), atom_kind, start, end, node.content[start:end]))
    _flush_paragraph(node, atoms, pending_paragraph)
    if not atoms and node.content:
        atoms.append(_atom(node, 0, "policy.paragraph", 0, len(node.content), node.content))
    return tuple(atoms)


def atomize_tool_schema(node: TraceNode) -> tuple[ComponentAtom, ...]:
    """Split tool schema text into stable top-level JSON path atoms when possible."""

    parsed = _try_parse_json(node.content)
    if parsed is None:
        return _fallback_tool_schema_atoms(node)
    atoms: list[ComponentAtom] = []
    _collect_json_atoms(node, parsed, "$", atoms)
    return tuple(atoms) if atoms else _fallback_tool_schema_atoms(node)


def _flush_paragraph(
    node: TraceNode,
    atoms: list[ComponentAtom],
    pending: Sequence[tuple[int, int, str]],
) -> None:
    if not pending:
        return
    start = pending[0][0]
    end = pending[-1][1]
    text = node.content[start:end].strip()
    if text:
        leading = len(node.content[start:end]) - len(node.content[start:end].lstrip())
        trailing = len(node.content[start:end].rstrip())
        atoms.append(_atom(node, len(atoms), "policy.paragraph", start + leading, start + trailing, text))


def _policy_line_kind(stripped: str) -> str | None:
    if stripped.startswith("#"):
        return "policy.heading"
    if re.match(r"^[-*+]\s+", stripped):
        return "policy.bullet"
    if re.match(r"^\d+[.)]\s+", stripped):
        return "policy.step"
    return None


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _collect_json_atoms(node: TraceNode, value: Any, path: str, atoms: list[ComponentAtom]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
            if _is_scalar_like(child):
                _append_json_atom(node, atoms, child_path, child)
            else:
                _collect_json_atoms(node, child, child_path, atoms)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if _is_scalar_like(child):
                _append_json_atom(node, atoms, child_path, child)
            else:
                _collect_json_atoms(node, child, child_path, atoms)
        return
    _append_json_atom(node, atoms, path, value)


def _append_json_atom(node: TraceNode, atoms: list[ComponentAtom], path: str, value: Any) -> None:
    text = str(value)
    if not text:
        return
    match = _find_text(node.content, text)
    if match is None:
        return
    start, end = match
    atoms.append(
        _atom(
            node,
            len(atoms),
            _json_atom_kind(path),
            start,
            end,
            node.content[start:end],
            metadata={"jsonpath": path},
        )
    )


def _fallback_tool_schema_atoms(node: TraceNode) -> tuple[ComponentAtom, ...]:
    atoms = atomize_policy_text(node)
    if atoms:
        return tuple(
            ComponentAtom(
                atom_id=atom.atom_id.replace(":policy.", ":tool_schema."),
                source_node_id=atom.source_node_id,
                atom_kind=atom.atom_kind.replace("policy.", "tool_schema."),
                text=atom.text,
                char_start=atom.char_start,
                char_end=atom.char_end,
                metadata=atom.metadata,
            )
            for atom in atoms
        )
    return ()


def _json_atom_kind(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".name") or lower == "$.name":
        return "tool_schema.name"
    if "description" in lower:
        return "tool_schema.description"
    if "parameter" in lower or "properties" in lower:
        return "tool_schema.parameter"
    if "required" in lower:
        return "tool_schema.required"
    if "example" in lower:
        return "tool_schema.example"
    return "tool_schema.field"


def _is_scalar_like(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _find_text(haystack: str, needle: str) -> tuple[int, int] | None:
    start = haystack.find(needle)
    if start < 0:
        escaped = json.dumps(needle)[1:-1]
        start = haystack.find(escaped)
        if start < 0:
            return None
        return start, start + len(escaped)
    return start, start + len(needle)


def _atom(
    node: TraceNode,
    index: int,
    atom_kind: str,
    char_start: int,
    char_end: int,
    text: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ComponentAtom:
    return ComponentAtom(
        atom_id=f"{node.node_id}:atom-{index}:{atom_kind}",
        source_node_id=node.node_id,
        atom_kind=atom_kind,
        text=text,
        char_start=char_start,
        char_end=char_end,
        metadata=metadata or {},
    )
