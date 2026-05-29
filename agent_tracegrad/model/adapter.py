"""Model adapter Protocol and light framework-owned output containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenizedOutput:
    input_ids: Any
    attention_mask: Any | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelForwardOutput:
    logits: Any
    hidden_states: Any | None = None
    metadata: dict[str, Any] | None = None


class ModelAdapter(Protocol):
    name: str

    def tokenize(self, text: str) -> TokenizedOutput: ...

    def input_embeddings(self, input_ids: Any, *, requires_grad: bool) -> Any: ...

    def forward(self, inputs_embeds: Any, attention_mask: Any | None) -> ModelForwardOutput: ...

    def chat_template_supported(self) -> bool: ...
