"""Span extractor: pretrained encoder (BERT / DeBERTa-v3 via AutoModel) + linear start/end head."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()  # hide the (expected) report about unused MLM/pooler weights at load time


class SpanExtractor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        try:
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        except TypeError:  # encoders without a pooler argument (e.g. DeBERTa)
            self.encoder = AutoModel.from_pretrained(model_name)
        # transformers>=5 loads weights in the dtype stored in the checkpoint (fp16 for microsoft/deberta-v3-base);
        # keep fp32 master weights so the head matches and AMP/GradScaler behave as expected
        self.encoder.float()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden, 2)
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        nn.init.zeros_(self.qa_outputs.bias)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        h = self.encoder(**kwargs).last_hidden_state
        logits = self.qa_outputs(self.dropout(h))
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        out = {"start_logits": start_logits, "end_logits": end_logits}
        if return_hidden:
            out["hidden"] = h
        return out


def tokenizer_uses_token_type_ids(tokenizer) -> bool:
    return "token_type_ids" in getattr(tokenizer, "model_input_names", [])


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def save_checkpoint(model: SpanExtractor, path: str, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": "g17qa-extractor-v1",
        "model_name": model.model_name,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device, model_name_override: Optional[str] = None):
    """Returns (model, tokenizer, config). Accepts both the new format and the older
    {'model_state_dict', 'model_name'} layout (with 'encoder.'/'qa_outputs.' keys)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_name = model_name_override or ckpt.get("model_name") or ckpt.get("config", {}).get("model")
    if not model_name:
        raise ValueError(f"checkpoint {path} does not record its model name; pass --model_name")
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None:
        raise ValueError(f"checkpoint {path} has no state dict")
    model = SpanExtractor(model_name)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.pooler")]
    if missing or unexpected:
        raise ValueError(f"checkpoint/model mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer, ckpt.get("config", {})
