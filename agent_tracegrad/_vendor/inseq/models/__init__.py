"""Vendored Inseq model wrapper subset."""

from agent_tracegrad._vendor.inseq.models.attribution_model import AttributionModel, InputFormatter
from agent_tracegrad._vendor.inseq.models.decoder_only import DecoderOnlyAttributionModel
from agent_tracegrad._vendor.inseq.models.huggingface_model import HuggingfaceDecoderOnlyModel, HuggingfaceModel

__all__ = ["AttributionModel", "DecoderOnlyAttributionModel", "HuggingfaceDecoderOnlyModel", "HuggingfaceModel", "InputFormatter"]
