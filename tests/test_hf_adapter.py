from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from agent_tracegrad.model import HuggingFaceCausalLMAdapter
from agent_tracegrad.model.hf_adapter import _configure_multi_gpu_load, _normalize_devices


def test_hf_adapter_freezes_weights_and_forwards_embeddings() -> None:
    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = transformers.GPT2LMHeadModel(config)

    class TinyTokenizer:
        name_or_path = "tiny-tokenizer"
        chat_template = None

        def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
            del text
            assert return_tensors == "pt"
            assert add_special_tokens is False
            return {
                "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    adapter = HuggingFaceCausalLMAdapter(model, TinyTokenizer(), name="tiny-gpt2")
    tokenized = adapter.tokenize("ignored")
    embeddings = adapter.input_embeddings(tokenized.input_ids, requires_grad=True)
    output = adapter.forward(embeddings, tokenized.attention_mask)

    assert adapter.name == "tiny-gpt2"
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert embeddings.requires_grad is True
    assert output.logits.shape[:2] == (1, 3)
    assert adapter.chat_template_supported() is False


def test_normalize_devices_accepts_comma_separated_cuda_list() -> None:
    devices = _normalize_devices(devices="cuda:0,cuda:1")

    assert [str(device) for device in devices] == ["cuda:0", "cuda:1"]


def test_normalize_devices_rejects_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _normalize_devices(devices="cuda:0,cuda:0")


def test_configure_multi_gpu_load_sets_balanced_device_map() -> None:
    kwargs: dict[str, object] = {}
    devices = _normalize_devices(devices="cuda:0,cuda:1")

    _configure_multi_gpu_load(kwargs, devices, dtype=torch.float16)

    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {0: "22GiB", 1: "22GiB"}
    assert kwargs["torch_dtype"] is torch.float16
