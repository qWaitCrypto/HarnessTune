"""Vendored Inseq feature attribution subset."""

from agent_tracegrad._vendor.inseq.attr.feat.attribution_utils import extract_args, join_token_ids
from agent_tracegrad._vendor.inseq.attr.feat.feature_attribution import FeatureAttribution

__all__ = ["FeatureAttribution", "extract_args", "join_token_ids"]
