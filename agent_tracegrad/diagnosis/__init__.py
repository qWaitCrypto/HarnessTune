"""Diagnostic product modules."""

from agent_tracegrad.diagnosis.atomizer import ComponentAtom, atomize_node, atomize_policy_text, atomize_tool_schema
from agent_tracegrad.diagnosis.drill import (
    AtomAttribution,
    DrillResult,
    drill_result_to_dict,
    drill_result_to_markdown,
    run_drill,
    write_drill_json,
    write_drill_markdown,
)
from agent_tracegrad.diagnosis.patterns import detect_diagnostic_labels, diagnostic_label_to_dict
from agent_tracegrad.diagnosis.report import diagnosis_to_markdown, write_diagnosis_markdown
from agent_tracegrad.diagnosis.runner import (
    diagnosis_to_dict,
    run_diagnosis,
    write_diagnosis_json,
)
from agent_tracegrad.diagnosis.types import (
    ComponentClassification,
    DiagnosisAblation,
    DiagnosisEvidence,
    DiagnosisResult,
    DiagnosticLabel,
    DiagnosticLabelName,
    MarginContribution,
    MarginDistribution,
)

__all__ = [
    "AtomAttribution",
    "ComponentAtom",
    "ComponentClassification",
    "DiagnosisAblation",
    "DiagnosisEvidence",
    "DiagnosisResult",
    "DiagnosticLabel",
    "DiagnosticLabelName",
    "DrillResult",
    "MarginContribution",
    "MarginDistribution",
    "atomize_node",
    "atomize_policy_text",
    "atomize_tool_schema",
    "detect_diagnostic_labels",
    "diagnostic_label_to_dict",
    "diagnosis_to_markdown",
    "diagnosis_to_dict",
    "drill_result_to_dict",
    "drill_result_to_markdown",
    "run_diagnosis",
    "run_drill",
    "write_diagnosis_markdown",
    "write_diagnosis_json",
    "write_drill_json",
    "write_drill_markdown",
]
