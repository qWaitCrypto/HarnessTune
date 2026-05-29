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
        dtype: Any | None = None,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.name = name or getattr(getattr(model, "config", None), "name_or_path", None) or model.__class__.__name__
        self.device = torch.device(device) if device is not None else _infer_device(model)
        self.dtype = dtype
        if dtype is not None:
            self.model.to(dtype=dtype)
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
        dtype: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> "HuggingFaceCausalLMAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **(model_kwargs or {}))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, **(tokenizer_kwargs or {}))
        return cls(model, tokenizer, name=model_name_or_path, device=device, dtype=dtype)

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
        embeddings = self.model.get_input_embeddings()(input_ids.to(self.device))
        if requires_grad:
            embeddings = embeddings.detach().clone().requires_grad_(True)
        return embeddings

    def forward(self, inputs_embeds: Any, attention_mask: Any | None) -> ModelForwardOutput:
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
