"""HuggingFace decoder-only model adapter."""

from __future__ import annotations

from typing import Any

from agent_tracegrad.model.adapter import ModelForwardOutput, TokenizedOutput


class HuggingFaceCausalLMAdapter:
    """Reference adapter for local HuggingFace decoder-only causal LMs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        name: str | None = None,
        device: str | None = None,
        devices: str | list[str] | tuple[str, ...] | None = None,
        dtype: Any | None = None,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.name = name or getattr(getattr(model, "config", None), "name_or_path", None) or model.__class__.__name__
        self.devices = _normalize_devices(device=device, devices=devices)
        self.device = self.devices[0] if self.devices else _infer_device(model)
        self.is_dispatched = _is_dispatched_model(model)
        self.dtype = dtype
        if dtype is not None and not self.is_dispatched:
            self.model.to(dtype=dtype)
        if not self.is_dispatched:
            self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if getattr(getattr(self.model, "config", None), "is_encoder_decoder", False):
            raise ValueError("HuggingFaceCausalLMAdapter only supports decoder-only causal LMs")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        device: str | None = None,
        devices: str | list[str] | tuple[str, ...] | None = None,
        dtype: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> "HuggingFaceCausalLMAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_devices = _normalize_devices(device=device, devices=devices)
        model_kwargs = dict(model_kwargs or {})
        if len(resolved_devices) > 1:
            _configure_multi_gpu_load(model_kwargs, resolved_devices, dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, **(tokenizer_kwargs or {}))
        return cls(model, tokenizer, name=model_name_or_path, device=device, devices=devices, dtype=dtype)

    def tokenize(self, text: str) -> TokenizedOutput:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return TokenizedOutput(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded.get("attention_mask", None).to(self.device)
            if encoded.get("attention_mask", None) is not None
            else None,
            metadata={"tokenizer_name": getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__)},
        )

    def input_embeddings(self, input_ids: Any, *, requires_grad: bool) -> Any:
        embedding_device = _module_device(self.model.get_input_embeddings(), fallback=self.device)
        embeddings = self.model.get_input_embeddings()(input_ids.to(embedding_device))
        if requires_grad:
            embeddings = embeddings.detach().clone().requires_grad_(True)
        return embeddings

    def forward(self, inputs_embeds: Any, attention_mask: Any | None) -> ModelForwardOutput:
        if attention_mask is not None and getattr(attention_mask, "device", None) != inputs_embeds.device:
            attention_mask = attention_mask.to(inputs_embeds.device)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return ModelForwardOutput(logits=output.logits, hidden_states=getattr(output, "hidden_states", None))

    def chat_template_supported(self) -> bool:
        return getattr(self.tokenizer, "chat_template", None) is not None


def _infer_device(model: Any):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_devices(
    *,
    device: str | None = None,
    devices: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[Any, ...]:
    import torch

    if devices is None:
        if device is None:
            return ()
        devices = (device,)
    if isinstance(devices, str):
        raw_devices = tuple(item.strip() for item in devices.split(",") if item.strip())
    else:
        raw_devices = tuple(devices)
    if not raw_devices:
        if device is None:
            return ()
        raw_devices = (device,)
    normalized = tuple(torch.device(item) for item in raw_devices)
    if len(normalized) > 8:
        raise ValueError("at most 8 CUDA devices are supported")
    if device is not None and len(normalized) > 1 and torch.device(device) != normalized[0]:
        raise ValueError("--device must match the first --devices entry when both are provided")
    if len(set(str(item) for item in normalized)) != len(normalized):
        raise ValueError("devices must not contain duplicates")
    return normalized


def _configure_multi_gpu_load(model_kwargs: dict[str, Any], devices: tuple[Any, ...], *, dtype: Any | None) -> None:
    if any(device.type != "cuda" for device in devices):
        raise ValueError("multi-device loading currently supports CUDA devices only")
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "balanced"
    if "max_memory" not in model_kwargs:
        model_kwargs["max_memory"] = {_device_index(device): _default_max_memory() for device in devices}
    if dtype is not None and "torch_dtype" not in model_kwargs and "dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype


def _device_index(device: Any) -> int:
    if device.index is None:
        raise ValueError("CUDA devices used for model sharding must include an explicit index, for example cuda:0")
    return int(device.index)


def _default_max_memory() -> str:
    return "22GiB"


def _is_dispatched_model(model: Any) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    return isinstance(device_map, dict) and bool(device_map)


def _module_device(module: Any, *, fallback: Any):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback
