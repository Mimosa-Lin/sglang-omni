# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR-3B model support for sglang-omni."""

from . import config
from .registration import register_arkasr_hf_config

register_arkasr_hf_config()

__all__ = ["config", "register_arkasr_hf_config"]
