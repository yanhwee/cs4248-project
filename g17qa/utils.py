"""Small shared helpers: seeding, device selection, JSON/JSONL I/O, timing."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device string. 'auto' picks cuda > mps > cpu."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(pref)


def amp_dtype_for(device: torch.device, mode: str) -> Optional[torch.dtype]:
    """Map --precision {fp32,fp16,bf16} to an autocast dtype valid for the device (or None = fp32).
    Warns loudly when the requested precision is not available on the device."""
    import warnings

    mode = (mode or "fp32").lower()
    if mode == "fp32":
        return None
    if mode == "fp16":
        if device.type == "cuda":
            return torch.float16
        warnings.warn(f"--precision fp16 is only supported on CUDA; running {device.type} in fp32", stacklevel=2)
        return None
    if mode == "bf16":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            warnings.warn("--precision bf16 requested but this GPU has no bf16 support (e.g. T4/V100); running in fp32. Use --precision fp16 instead", stacklevel=2)
            return None
        if device.type == "cpu":
            return torch.bfloat16
        warnings.warn(f"--precision bf16 is not supported on {device.type}; running in fp32", stacklevel=2)
        return None
    raise ValueError(f"unknown precision mode {mode!r}")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def batched(items: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
