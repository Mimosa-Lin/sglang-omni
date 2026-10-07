# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR audio prefill on SGLang's native MLX worker."""

import logging
import time

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)


class FunASRMlxModelRunner(AudioMlxModelRunner):
    model_name = "Fun-ASR"

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model, quantize_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import FunASRModel

        path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(path, self.trust_remote_code)
        logger.info(f"Loading native MLX Fun-ASR model: {path}")
        started = time.perf_counter()
        self.model, config = load_model(
            path, get_model_classes=lambda config: (FunASRModel, ModelConfig)
        )
        presets = {"mlx_q4": (4, 64), "mlx_q8": (8, 64)}
        if (
            self._quantization in presets  # noqa: leading-underscore
            and "quantization" not in config
        ):
            bits, group_size = presets[
                self._quantization  # noqa: leading-underscore
            ]
            logger.info(
                f"Quantizing native MLX Fun-ASR text stack: "
                f"bits={bits}, group_size={group_size}"
            )
            self.model, _config = quantize_model(
                self.model,
                config,
                group_size=group_size,
                bits=bits,
            )
        else:
            pass
        mx.eval(self.model.parameters())
        logger.info(
            f"Loaded native MLX Fun-ASR model in "
            f"{time.perf_counter() - started:.2f}s"
        )


def make_fun_asr_mlx_runner_class():
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class FunASRMlxRunner(FunASRMlxModelRunner, MlxModelRunner):
        pass

    return FunASRMlxRunner
