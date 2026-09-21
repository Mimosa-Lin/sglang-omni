# SPDX-License-Identifier: Apache-2.0
"""Local ARK-ASR configuration and processor support."""

from __future__ import annotations

from typing import ClassVar

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    ProcessorMixin,
)

from .audio_lengths import arkasr_audio_token_lengths
from .hf_config import ArkasrConfig


class ArkasrProcessor(ProcessorMixin):
    """Whisper feature extractor and tokenizer for ARK-ASR."""

    attributes: ClassVar[list[str]] = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


# Register the processor without changing Transformers' process-global config mapping.
register_customized_processor(ArkasrProcessor)(ArkasrConfig)


__all__ = ["ArkasrConfig", "ArkasrProcessor", "arkasr_audio_token_lengths"]
