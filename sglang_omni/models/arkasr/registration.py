# SPDX-License-Identifier: Apache-2.0
"""Hugging Face configuration registration for ARK-ASR."""

from __future__ import annotations

_arkasr_hf_config_registered = False


def register_arkasr_hf_config() -> None:
    """Register the local config before architecture discovery."""
    global _arkasr_hf_config_registered
    if _arkasr_hf_config_registered:
        return

    from transformers import AutoConfig

    from .hf_config import ArkasrConfig

    AutoConfig.register("arkasr", ArkasrConfig, exist_ok=True)
    _arkasr_hf_config_registered = True


__all__ = ["register_arkasr_hf_config"]
