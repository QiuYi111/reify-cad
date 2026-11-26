"""Unified compute device backend for Reify (CUDA -> Apple Metal MPS -> CPU)."""

from __future__ import annotations

import os
import warnings
from typing import Any

import torch


def get_compute_device(
    dtype: torch.dtype = torch.float32,
    prefer: str | None = None,
) -> torch.device:
    """Select the fastest available compute device compatible with `dtype`.

    Priority for ``prefer="auto"`` (or ``REIFY_DEVICE=auto``):
      1. CUDA (discrete NVIDIA GPU — accelerates both float32 and float64)
      2. CPU (Apple Silicon / x86 SIMD — avoids Metal command-buffer dispatch overhead on small AST kernels;
         pass ``REIFY_DEVICE=mps`` or ``device="mps"`` to explicitly run float32 kernels on Apple Metal MPS)
    """
    choice = (prefer or os.environ.get("REIFY_DEVICE", "auto")).strip().lower()
    if choice in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if choice == "cpu":
        return torch.device("cpu")

    if choice.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(choice)
        warnings.warn(f"REIFY_DEVICE={choice!r} requested but CUDA is unavailable; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")

    if choice == "mps":
        if dtype != torch.float32:
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("REIFY_DEVICE='mps' requested but MPS is unavailable; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")

    warnings.warn(f"Unknown REIFY_DEVICE={choice!r}; falling back to CPU.", stacklevel=2)
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize asynchronous GPU kernels (CUDA or MPS) before host timing/readback."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_info() -> dict[str, Any]:
    """Return runtime GPU/CPU backend capabilities for diagnostics and `reify doctor`."""
    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(torch.backends.mps.is_available())
    cuda_name = torch.cuda.get_device_name(0) if cuda_ok else None
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_ok,
        "cuda_device_name": cuda_name,
        "mps_available": mps_ok,
        "env_reify_device": os.environ.get("REIFY_DEVICE", "auto"),
        "active_float32_device": str(get_compute_device(torch.float32)),
        "active_float64_device": str(get_compute_device(torch.float64)),
    }
