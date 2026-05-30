"""Diagnostic product modules."""

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
    "ComponentClassification",
    "DiagnosisAblation",
    "DiagnosisEvidence",
    "DiagnosisResult",
    "DiagnosticLabel",
    "DiagnosticLabelName",
    "MarginContribution",
    "MarginDistribution",
    "detect_diagnostic_labels",
    "diagnostic_label_to_dict",
    "diagnosis_to_markdown",
    "diagnosis_to_dict",
    "run_diagnosis",
    "write_diagnosis_markdown",
    "write_diagnosis_json",
]
