# ARK-ASR MLX Review Findings

> Review date: 2026-09-18  
> Scope: ARK-ASR MLX runner, engine builder, request builder, scheduler integration  
> Baseline: Qwen3-ASR MLX implementation in the same repository  
> Status: Qwen3-ASR comparison P1 fixes implemented; documentation follow-ups resolved

## Summary

The ARK-ASR MLX path now selects the native MLX scheduler runner, disables
radix cache and chunked prefill, and avoids initializing the Torch/CUDA audio
encoder resources. The Qwen3-ASR comparison is documented below. Remaining
functional gaps are deliberately scoped as future features rather than
unverified MLX integration requirements.

## P1 Checklist

- [x] Reject `mlx_enable_sampling=True` during startup.
- [x] Apply static `logit_bias` during MLX greedy prefill and decode.
- [x] Use the effective tokenizer vocabulary size including added tokens.
- [x] Reject audio beyond the native encoder window instead of truncating it.
- [x] Validate request prompt and output tokens against context length.
- [x] Add an opt-in end-to-end MLX server test.
- [x] Reuse ARK pre-LM cache before mel extraction.
- [x] Use queue-aware synchronous/asynchronous pre-LM admission.
- [x] Preserve MLX backend overrides through builder finalization.

## Findings

### P1: MLX sampling mode is not rejected during startup

**Status:** Resolved

ARK does not implement the same startup validation as Qwen3-ASR:

- Qwen3-ASR: `sglang_omni/models/qwen3_asr/engine_builder.py:259`
- ARK: `sglang_omni/models/arkasr/engine_builder.py:102`

When `mlx_enable_sampling=True`, the MLX worker can construct a
`logit_edit_row`, but the ARK audio prefill explicitly rejects it at:

- `sglang_omni/model_runner/audio_mlx.py:122`

The failure previously occurred on the first request instead of during server
startup. ARK now rejects `mlx_enable_sampling=True` in
`validate_before_infrastructure()`, matching Qwen3-ASR. Regression coverage was
added to `tests/unit_test/arkasr/test_pipeline.py`.

### P1: Greedy MLX decoding bypasses `logit_bias`

**Status:** Resolved

The ARK request builder creates a `logit_bias` for reserved marker tokens:

- `sglang_omni/models/arkasr/request_builders.py:202`

However, the native MLX audio path directly performs `argmax`:

- `sglang_omni/model_runner/audio_mlx.py:133`
- `sglang_omni/model_runner/audio_mlx.py:175`

With the current default `mlx_enable_sampling=False`, the generic worker does
not apply the request's `logit_bias`. ARK now records the static bias in the
audio runner and applies it before greedy selection during audio prefill,
ordinary decode, and chained decode. Output filtering remains a
post-processing fallback.

Regression coverage was added for the greedy logit-edit operation in
`tests/unit_test/arkasr/test_mlx_model.py`.

### P1: `vocab_size` excludes tokenizer added tokens

**Status:** Resolved

ARK uses:

```python
vocab_size = int(tokenizer.vocab_size)
```

at `sglang_omni/models/arkasr/request_builders.py:112`.

For the local checkpoint, the observed values are:

```text
tokenizer.vocab_size = 151643
len(tokenizer)       = 151670
added token ids      = 151643 ... 151669
```

SGLang checks generated tokens against `Req.vocab_size` in
`sglang/srt/managers/schedule_batch.py:1691`. Added tokens can therefore be
treated as out-of-vocabulary and replaced with EOS.

ARK now uses `len(tokenizer)`, matching Qwen3-ASR. A regression test covers
the effective tokenizer size separately from the base vocabulary size.

### P1: Long audio is silently truncated

**Status:** Resolved

ARK extracts features with:

- `sglang_omni/models/arkasr/request_builders.py:151`
- `truncation=True` at line 157

The checkpoint configuration reports:

```text
nb_max_frames = 3000
max_source_positions = 1500
```

ARK now uses `truncation=False` and rejects audio beyond the feature extractor
window before feature extraction. The error instructs callers to split the
audio before submitting it.

Regression coverage was added for rejection before feature extraction.

### P1: No request-level context budget validation

**Status:** Resolved

ARK computes a default context length in:

- `sglang_omni/models/arkasr/engine_builder.py:111-112`

But the request builder accepts an arbitrary `max_new_tokens` value at:

- `sglang_omni/models/arkasr/request_builders.py:196`

ARK now checks that:

```text
prompt_tokens + max_new_tokens <= context_length - 1
```

This prevents a request from passing admission and failing later in scheduler
or decode code. Regression coverage was added for an overflowing
`max_new_tokens` request.

### P2: Torch path does not translate audio decode errors

**Status:** Resolved

Qwen3-ASR converts `AudioDecodeError` into a clear client-facing
`ValueError`:

- `sglang_omni/models/qwen3_asr/request_builders.py:245-253`

ARK now translates the same exception at:

- `sglang_omni/models/arkasr/request_builders.py:141`

Invalid uploads now return a clear bad-request message instead of surfacing
the internal decoder exception.

### P1: Torch path did not reuse Qwen's pre-LM embedding cache admission

**Status:** Resolved

Qwen3-ASR checks the audio fingerprint cache before doing feature extraction
and before submitting work to the pre-LM encoder service:

- `sglang_omni/models/qwen3_asr/request_builders.py:267-310`

ARK always extracts features first and submits the item later:

- `sglang_omni/models/arkasr/request_builders.py:141-181`
- `sglang_omni/models/arkasr/request_builders.py:232-237`

ARK now estimates the audio token count from the feature extractor hop length,
checks the cache before mel extraction, and attaches a validated cached
embedding directly to the multimodal item. Cache misses continue through the
existing single-flight encoder service. This does not affect the native MLX
path because MLX deliberately skips the Torch/CUDA encoder service.

### P1: ARK always deferred pre-LM encoding

**Status:** Resolved

Qwen3-ASR uses the scheduler's `request_build_queue_fits_workers()` signal to
encode immediately on the request-build worker when capacity is available, and
only returns `DeferredAdmission` when the request-build pool is busy. ARK
previously always returned `DeferredAdmission`, adding avoidable admission
latency at low concurrency.

ARK now wires `post_scheduler_setup()` to the same scheduler signal and passes
`should_wait_for_encode` into the request builder. The request builder uses
`encode_item()` when the queue has capacity and preserves deferred admission
under load.

### P1: MLX backend overrides could be re-enabled after profile selection

**Status:** Resolved

Qwen3-ASR explicitly reapplies `enable_torch_compile=False` in
`adjust_overrides()` because typed pipeline defaults are merged after backend
profile construction. ARK now has the same guard and synchronizes the resolved
server context length in `customize_server_args()`.

### P1: No complete ARK MLX server integration test

**Status:** Resolved

An opt-in server test was added at:

- `tests/test_model/test_arkasr_mlx_server.py`

The test launches the real CLI server with `SGLANG_USE_MLX=1`, submits a local
audio file to `/v1/audio/transcriptions`, and validates the JSON transcript.
It is skipped unless `ARKASR_MLX_SERVER_CHECKPOINT` is set, so normal CI does
not download a checkpoint or require Apple Metal.

Run it with:

```bash
ARKASR_MLX_SERVER_CHECKPOINT=checkpoints/ark-asr-3b \
SGLANG_USE_MLX=1 \
pytest -s -x tests/test_model/test_arkasr_mlx_server.py
```

## Verification Completed

The following tests passed before the latest Qwen-comparison fixes:

```text
tests/unit_test/arkasr/test_pipeline.py   31 passed
tests/unit_test/arkasr/test_mlx_model.py  25 passed
```

The actual ARK MLX runner was also exercised with both BF16 and Q4 model
loading. The Q4 runner completed successfully, but the benchmark harness is
not a substitute for a full HTTP server test. The new server test was
collected and skipped locally because
`ARKASR_MLX_SERVER_CHECKPOINT` was not set; it must be run on Apple Silicon
with the local checkpoint to validate the full HTTP path.

After the latest changes, `compileall` and `git diff --check` pass. The unit
test command is currently blocked during SGLang import because the local
`.venv-apple` does not contain the `triton` package:

```text
ModuleNotFoundError: No module named 'triton'
```

## Remaining Follow-ups

1. Add transcript-prefix conditioning for long chunks or realtime refreshes
   only after validating ARK's checkpoint behavior. Current long-audio
   requests are independent chunks and realtime refreshes replace the current
   hypothesis.
2. Add a Torch/MPS runner only if ARK Torch/MPS support is in scope.
3. Add an Apple-compatible SGLang test environment or conditional import fix
   so the opt-in MLX server test can run without the unavailable macOS arm64
   Triton package.
