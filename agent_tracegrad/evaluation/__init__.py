"""Evaluation data structures and replay extension surfaces."""

from agent_tracegrad.evaluation.ground_truth import GroundTruthLabel
from agent_tracegrad.evaluation.perturbation import TraceLevelPerturbation, apply_trace_level_perturbation
from agent_tracegrad.evaluation.replay import ReplayHook
from agent_tracegrad.evaluation.spec import PerturbationSpec

__all__ = [
    "GroundTruthLabel",
    "PerturbationSpec",
    "ReplayHook",
    "TraceLevelPerturbation",
    "apply_trace_level_perturbation",
]
