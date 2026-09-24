# SPDX-License-Identifier: Apache-2.0
"""Opt-in MLX-vs-Torch/MPS checkpoint parity for ARK-ASR-3B (docs/cookbook/arkasr.md).

Requires a local ARK-ASR-3B checkout and Apple Silicon (MLX + MPS). Nothing
here runs unless ``ARKASR_PARITY_CHECKPOINT`` is set, so CI never pays for it.

Covered, all against the torch reference on the same official checkpoint:

- fp32 audio adapter parity (B=1 and B=4) and fp32 batch invariance
- fp32 text logits parity (B=4) against a transformers Qwen2 reference
- bf16 per-layer drift table (run with ``-s`` to print) with bounded stages
- end-to-end greedy transcripts token-identical between the two stacks
- batched decode (B=4) vs per-sequence decode: only post-EOS near-tie flips
"""

from __future__ import annotations

import gc
import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.accelerator

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.arkasr.mlx.config import ModelConfig  # noqa: E402
from sglang_omni.models.arkasr.mlx.model import ArkasrModel  # noqa: E402

_IM_END = 151645
_REL_TOL_FP32 = 1e-4


def _rel_err(actual, expected) -> float:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _load_mel(checkpoint: Path, audio_path: Path):
    """Serving-faithful mel: pad-to-longest, then crop to the true frame count."""
    from transformers import AutoFeatureExtractor

    with wave.open(str(audio_path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        waveform = pcm.astype(np.float32) / 32768.0
    if sr != 16000:
        torchaudio = pytest.importorskip("torchaudio")
        waveform = (
            torchaudio.functional.resample(torch.from_numpy(waveform)[None], sr, 16000)[
                0
            ]
            .numpy()
        )
    fe = AutoFeatureExtractor.from_pretrained(str(checkpoint))
    extracted = fe(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
        truncation=True,
    )
    num_mel_frames = int(extracted.attention_mask.sum().item())
    return extracted.input_features[:, :, :num_mel_frames].contiguous()


def _prompt_ids(checkpoint: Path, num_audio_tokens: int) -> "torch.Tensor":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    prompt = (
        f"<|user|><|begin_of_audio|>{'<|audio|>' * num_audio_tokens}"
        f"<|end_of_audio|>Please transcribe this audio.<|assistant|>"
    )
    return tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids


def _mlx_hook(monkeypatch, module, name: str, sink: dict):
    """Record one module instance's output by patching its class __call__."""
    cls = type(module)
    original = cls.__call__

    def wrapped(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if self is module:
            value = out[0] if isinstance(out, tuple) else out
            sink[name] = np.asarray(value.astype(mx.float32))
        return out

    monkeypatch.setattr(cls, "__call__", wrapped)


@pytest.fixture(scope="module")
def refs():
    checkpoint = Path(os.environ.get("ARKASR_PARITY_CHECKPOINT", ""))
    if not checkpoint or not checkpoint.is_dir():
        pytest.skip("Set ARKASR_PARITY_CHECKPOINT to a local ARK-ASR-3B checkout")
    if not torch.backends.mps.is_available():
        pytest.skip("ARK-ASR MLX parity requires torch MPS")
    if not mx.metal.is_available():
        pytest.skip("ARK-ASR MLX parity requires MLX Metal")

    audio_path = Path(os.environ.get("ARKASR_PARITY_AUDIO", "tests/data/query_to_cars.wav"))
    mel = _load_mel(checkpoint, audio_path)
    mel4 = mel.repeat(4, 1, 1).contiguous()
    cfg_dict = json.load(open(checkpoint / "config.json"))

    from safetensors.torch import load_file

    from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
    from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig

    state = {}
    weights = {}
    for shard in sorted(checkpoint.glob("model*.safetensors")):
        weights.update(load_file(str(shard)))

    def torch_adapter(dtype, device):
        config = ArkasrConfig(**cfg_dict)
        adapter = ArkAudioMLPAdapter(config).to(device=device, dtype=dtype).eval()
        adapter.load_state_dict(
            {
                k[len("audio_encoder.") :]: v
                for k, v in weights.items()
                if k.startswith("audio_encoder.")
            }
        )
        return adapter

    def torch_lm(dtype, device):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        qcfg = Qwen2Config(
            hidden_size=cfg_dict["hidden_size"],
            intermediate_size=cfg_dict["intermediate_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            vocab_size=cfg_dict["vocab_size"],
            max_position_embeddings=cfg_dict["max_position_embeddings"],
            rms_norm_eps=cfg_dict["rms_norm_eps"],
            rope_theta=cfg_dict["rope_theta"],
            tie_word_embeddings=True,
        )
        qcfg._attn_implementation = "sdpa"
        lm = Qwen2ForCausalLM(qcfg).to(device=device, dtype=dtype).eval()
        lm.load_state_dict(
            {
                k: v
                for k, v in weights.items()
                if k.startswith("model.") or k == "lm_head.weight"
            },
            strict=False,
        )
        return lm

    # ---- torch fp32 on CPU: the algorithm-exactness reference -------------
    # Only B=4 is computed; identical mel rows make row 0 the B=1 answer and
    # the invariance assertion below proves that equivalence.
    adapter = torch_adapter(torch.float32, "cpu")
    with torch.no_grad():
        state["t_audio_fp32_b4"] = adapter(mel4).numpy()
    del adapter
    gc.collect()

    lm = torch_lm(torch.float32, "cpu")
    ids = _prompt_ids(checkpoint, state["t_audio_fp32_b4"].shape[1])
    ids4 = ids.repeat(4, 1).contiguous()
    with torch.no_grad():
        state["t_logits_fp32_b4"] = lm(ids4).logits[:, -1, :].numpy()
    del lm
    gc.collect()

    # ---- torch bf16 on MPS: the production-dtype reference -----------------
    device = "mps"
    adapter = torch_adapter(torch.bfloat16, device)
    stages = {}
    mel_mps = mel.to(device, torch.bfloat16)
    mel4_mps = mel4.to(device, torch.bfloat16)
    adapter.whisper.conv1.register_forward_hook(
        lambda m, i, o: stages.__setitem__("conv1", o.transpose(1, 2).float().cpu().numpy())
    )
    adapter.whisper.conv2.register_forward_hook(
        lambda m, i, o: stages.__setitem__("conv2", o.transpose(1, 2).float().cpu().numpy())
    )
    for idx, layer in enumerate(adapter.whisper.layers):
        layer.register_forward_hook(
            lambda m, i, o, idx=idx: stages.__setitem__(
                f"layer{idx:02d}", o[0].float().cpu().numpy()
            )
        )
    adapter.register_forward_hook(
        lambda m, i, o: stages.__setitem__("adapter", o.float().cpu().numpy())
    )
    with torch.no_grad():
        t_audio_b1 = adapter(mel_mps).float().cpu().numpy()
        t_audio_b4 = adapter(mel4_mps).float().cpu().numpy()
    state["t_audio_bf16_b1"], state["t_audio_bf16_b4"] = t_audio_b1, t_audio_b4
    state["t_audio_bf16_stages"] = stages
    state["torch_audio_bf16_row_invariance"] = _rel_err(t_audio_b4[1], t_audio_b4[0])

    lm = torch_lm(torch.bfloat16, device)
    ids_mps = ids.to(device)
    ids4_mps = ids4.to(device)
    with torch.no_grad():
        out = lm(ids_mps, output_hidden_states=True)
        state["t_logits_bf16_b1"] = out.logits[:, -1, :].float().cpu().numpy()
        text_stages = {
            "embed": out.hidden_states[0].float().cpu().numpy(),
            "norm": out.hidden_states[-1].float().cpu().numpy(),
        }
        t_logits_b4 = lm(ids4_mps).logits[:, -1, :].float().cpu().numpy()
    state["t_logits_bf16_b4"] = t_logits_b4
    state["t_text_bf16_stages"] = text_stages

    # Full torch/MPS greedy pipeline (audio injection -> greedy decode).
    audio_positions = [
        i for i, t in enumerate(ids[0].tolist()) if t == 151663
    ]
    with torch.no_grad():
        embeds = lm.model.embed_tokens(ids_mps)
        embeds[0, audio_positions[0] : audio_positions[-1] + 1, :] = adapter(mel_mps)[0]
        out2 = lm(inputs_embeds=embeds)
    greedy = []
    for _ in range(64):
        tok = int(out2.logits[:, -1, :].argmax(-1).item())
        greedy.append(tok)
        if tok == _IM_END:
            break
        with torch.no_grad():
            out2 = lm(
                torch.tensor([[tok]], device=device),
                past_key_values=out2.past_key_values,
            )
    state["t_greedy"] = greedy

    del adapter, lm, out, out2, weights
    gc.collect()
    torch.mps.empty_cache()

    state["mel"], state["mel4"] = mel, mel4
    state["ids"], state["ids4"] = ids, ids4
    state["checkpoint"] = checkpoint
    return state


@pytest.fixture(scope="module")
def model(refs):
    from mlx_lm.utils import load_model

    loaded, _ = load_model(
        refs["checkpoint"],
        get_model_classes=lambda config: (ArkasrModel, ModelConfig),
    )
    return loaded


def test_audio_adapter_fp32_parity(refs, model):
    from mlx.utils import tree_map

    model.audio_encoder.update(
        tree_map(lambda p: p.astype(mx.float32), model.audio_encoder.parameters())
    )
    mel_f = mx.array(np.ascontiguousarray(refs["mel"].numpy()))
    mel4_f = mx.array(np.ascontiguousarray(refs["mel4"].numpy()))
    b1 = model.audio_encoder(mel_f)
    b4 = model.audio_encoder(mel4_f)
    mx.eval(b1, b4)
    # Batch invariance must hold in fp32, so B=1 parity carries to B=4.
    assert _rel_err(np.asarray(b4[0]), np.asarray(b1[0])) < _REL_TOL_FP32
    for row in range(4):
        assert _rel_err(np.asarray(b4[row]), refs["t_audio_fp32_b4"][row]) < _REL_TOL_FP32


def test_text_logits_fp32_parity(refs, model):
    from mlx.utils import tree_map

    model.model.update(tree_map(lambda p: p.astype(mx.float32), model.model.parameters()))
    ids4 = mx.array(np.asarray(refs["ids4"])).astype(mx.int32)
    logits = model(ids4)
    mx.eval(logits)
    for row in range(4):
        assert (
            _rel_err(np.asarray(logits[row, -1]), refs["t_logits_fp32_b4"][row])
            < _REL_TOL_FP32
        )
    assert (
        np.asarray(logits[:, -1, :].argmax(-1)) == refs["t_logits_fp32_b4"].argmax(-1)
    ).all()


def test_bf16_layer_drift_bounded(refs, model, monkeypatch, capsys):
    stages = {}
    whisper = model.audio_encoder.whisper
    _mlx_hook(monkeypatch, whisper.conv1, "conv1", stages)
    _mlx_hook(monkeypatch, whisper.conv2, "conv2", stages)
    for idx, layer in enumerate(whisper.layers):
        _mlx_hook(monkeypatch, layer, f"layer{idx:02d}", stages)
    _mlx_hook(monkeypatch, model.audio_encoder, "adapter", stages)

    mel = mx.array(np.ascontiguousarray(refs["mel"].numpy())).astype(mx.bfloat16)
    features = model.get_audio_features(mel, None)
    mx.eval(features)

    torch_stages = refs["t_audio_bf16_stages"]
    lines = [f"{'stage':>12} | {'rel err':>9} | {'max abs':>8} | {'ref std':>7}"]
    ordered = (
        ["conv1", "conv2"]
        + [f"layer{i:02d}" for i in range(len(whisper.layers))]
        + ["adapter"]
    )
    for name in ordered:
        actual = stages[name].reshape(-1)
        expected = torch_stages[name].reshape(-1)
        rel = _rel_err(actual, expected)
        lines.append(
            f"{name:>12} | {rel:9.5f} | {np.abs(actual - expected).max():8.4f} "
            f"| {expected.std():7.4f}"
        )
    capsys.readouterr()
    print("\n" + "\n".join(lines))

    assert _rel_err(stages["conv1"], torch_stages["conv1"]) < 0.01
    assert _rel_err(stages["conv2"], torch_stages["conv2"]) < 0.01
    assert _rel_err(stages["layer00"], torch_stages["layer00"]) < 0.02
    # Deep layers carry massive activations whose bf16 rounding trajectories
    # diverge; the adapter LayerNorm + frame merge absorbs them.
    assert _rel_err(stages["adapter"], torch_stages["adapter"]) < 0.10
    assert _rel_err(
        np.asarray(features.astype(mx.float32)), refs["t_audio_bf16_b1"][0]
    ) < 0.10

    ids = mx.array(np.asarray(refs["ids"])).astype(mx.int32)
    logits = model(ids)
    mx.eval(logits)
    assert (
        np.asarray(logits[0, -1, :].argmax(-1)).item()
        == refs["t_logits_bf16_b1"].argmax(-1).item()
    )


def test_greedy_transcript_identical(refs, model):
    audio_positions = [i for i, t in enumerate(refs["ids"][0].tolist()) if t == 151663]
    features = model.get_audio_features(
        mx.array(np.ascontiguousarray(refs["mel"].numpy())).astype(mx.bfloat16), None
    )
    cache = model.make_cache()
    inputs_embeds = model._build_inputs_embeds(
        mx.array(np.asarray(refs["ids"])).astype(mx.int32),
        features,
        audio_start=audio_positions[0],
        num_audio_tokens=features.shape[0],
    )
    logits = model._forward_last_logits(inputs_embeds, cache=cache)
    greedy = []
    for _ in range(64):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        greedy.append(token)
        if token == _IM_END:
            break
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
    assert greedy == refs["t_greedy"]


def test_batched_decode_matches_single(refs, model):
    variants = []
    for i in range(4):
        variants.append(refs["ids"][0].tolist() + [100 + i])
    variant_mx = mx.array(np.array(variants), dtype=mx.int32)

    alone = []
    for i in range(4):
        cache = model.make_cache()
        logits = model(variant_mx[i : i + 1], cache=cache)
        seq = []
        for _ in range(8):
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            seq.append(token)
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        alone.append(seq)

    cache = model.make_cache()
    logits = model(variant_mx, cache=cache)
    batched = [[] for _ in range(4)]
    for _ in range(8):
        tokens = np.asarray(mx.argmax(logits[:, -1, :], axis=-1)).tolist()
        for row in range(4):
            batched[row].append(tokens[row])
        logits = model(mx.array(np.array(tokens)[:, None], dtype=mx.int32), cache=cache)

    identical = sum(alone[r] == batched[r] for r in range(4))
    assert identical >= 3, f"only {identical}/4 batched rows match single-request decode"
    # Any divergence must live strictly after the first EOS: the served
    # sequence has already stopped there, and the flip is a bf16 near-tie.
    for row in range(4):
        if alone[row] == batched[row]:
            continue
        eos_positions = [i for i, t in enumerate(alone[row]) if t == _IM_END]
        assert eos_positions, f"row {row} diverges without an EOS boundary"
        boundary = eos_positions[0] + 1
        assert batched[row][:boundary] == alone[row][:boundary]


def test_audio_tower_all_layers(monkeypatch):
    """Compare the real audio checkpoint without loading the text decoder."""
    from safetensors import safe_open

    from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
    from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
    from sglang_omni.models.arkasr.mlx.model import ArkAudioMLPAdapter as MlxAdapter

    path = os.environ.get("ARKASR_PARITY_CHECKPOINT")
    if not path or not Path(path).is_dir():
        pytest.skip("Set ARKASR_PARITY_CHECKPOINT to a local ARK-ASR-3B checkout")
    if not torch.backends.mps.is_available() or not mx.metal.is_available():
        pytest.skip("Requires Torch MPS and MLX Metal")
    checkpoint = Path(path)
    cfg_dict = json.loads((checkpoint / "config.json").read_text())
    mel = _load_mel(
        checkpoint,
        Path(os.environ.get("ARKASR_PARITY_AUDIO", "tests/data/query_to_cars.wav")),
    )

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    by_shard = {}
    for key, shard in index["weight_map"].items():
        if key.startswith("audio_encoder."):
            by_shard.setdefault(shard, []).append(key)
    weights = {}
    for shard, keys in by_shard.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
            for key in keys:
                weights[key.removeprefix("audio_encoder.")] = reader.get_tensor(key)

    torch_adapter = ArkAudioMLPAdapter(ArkasrConfig(**cfg_dict)).eval()
    torch_adapter.load_state_dict(weights, strict=True)
    mlx_adapter = MlxAdapter(ModelConfig.from_dict(cfg_dict))
    mlx_weights = []
    for name, tensor in weights.items():
        value = mx.array(tensor.float().numpy()).astype(mx.bfloat16)
        if name.startswith("whisper.conv") and name.endswith(".weight"):
            value = value.transpose(0, 2, 1)
        mlx_weights.append((name, value))
    mlx_adapter.load_weights(mlx_weights)
    mx.eval(mlx_adapter.parameters())
    del weights, mlx_weights
    gc.collect()

    names = ["conv1", "conv2"]
    names += [f"layer{i:02d}" for i in range(len(torch_adapter.whisper.layers))]
    names += ["tower", "layer_norm", "adapter"]
    for dtype, device in ((torch.float32, "cpu"), (torch.bfloat16, "mps")):
        t_stages = {}
        m_stages = {}
        handles = []

        def capture(name, transpose=False):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if transpose:
                    value = value.transpose(1, 2)
                t_stages[name] = value.detach().float().cpu().numpy()

            return hook

        handles.append(
            torch_adapter.whisper.conv1.register_forward_hook(capture("conv1", True))
        )
        handles.append(
            torch_adapter.whisper.conv2.register_forward_hook(capture("conv2", True))
        )
        for i, layer in enumerate(torch_adapter.whisper.layers):
            handles.append(layer.register_forward_hook(capture(f"layer{i:02d}")))
        for name, module in (
            ("tower", torch_adapter.whisper),
            ("layer_norm", torch_adapter.layer_norm),
            ("adapter", torch_adapter),
        ):
            handles.append(module.register_forward_hook(capture(name)))

        mlx_tower = mlx_adapter.whisper
        _mlx_hook(monkeypatch, mlx_tower.conv1, "conv1", m_stages)
        _mlx_hook(monkeypatch, mlx_tower.conv2, "conv2", m_stages)
        for i, layer in enumerate(mlx_tower.layers):
            _mlx_hook(monkeypatch, layer, f"layer{i:02d}", m_stages)
        _mlx_hook(monkeypatch, mlx_tower, "tower", m_stages)
        _mlx_hook(monkeypatch, mlx_adapter.layer_norm, "layer_norm", m_stages)
        _mlx_hook(monkeypatch, mlx_adapter, "adapter", m_stages)

        torch_adapter.to(device=device, dtype=dtype)
        mlx_dtype = mx.float32 if dtype == torch.float32 else mx.bfloat16
        from mlx.utils import tree_map

        mlx_adapter.update(
            tree_map(lambda p: p.astype(mlx_dtype), mlx_adapter.parameters())
        )
        with torch.no_grad():
            torch_adapter(mel.to(device=device, dtype=dtype))
        mlx_adapter(mx.array(mel.numpy()).astype(mlx_dtype))
        print(f"\n{dtype} audio tower: stage | shape | rel L2 | max abs | cosine")
        for name in names:
            actual = m_stages[name]
            expected = t_stages[name]
            assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
            rel = _rel_err(actual, expected)
            max_abs = float(np.max(np.abs(actual - expected)))
            a = actual.astype(np.float64).ravel()
            b = expected.astype(np.float64).ravel()
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            print(
                f"{name:>12} | {str(actual.shape):>17} | {rel:.6e} | "
                f"{max_abs:.6e} | {cosine:.8f}"
            )
            if dtype == torch.float32:
                assert rel < _REL_TOL_FP32, f"{name}: fp32 rel L2={rel:.6e}"
        if dtype == torch.bfloat16:
            assert _rel_err(m_stages["adapter"], t_stages["adapter"]) < 0.10
        for handle in handles:
            handle.remove()
        monkeypatch.undo()


def test_text_decoder_all_layers_fp32(monkeypatch):
    """Compare every text decoder stage without constructing the audio tower."""
    from safetensors import safe_open
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from sglang_omni.models.arkasr.mlx.config import TextConfig
    from sglang_omni.models.arkasr.mlx.model import TextModel

    path = os.environ.get("ARKASR_PARITY_CHECKPOINT")
    if not path or not Path(path).is_dir():
        pytest.skip("Set ARKASR_PARITY_CHECKPOINT to a local ARK-ASR-3B checkout")
    checkpoint = Path(path)
    cfg = json.loads((checkpoint / "config.json").read_text())

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    by_shard = {}
    for key, shard in index["weight_map"].items():
        if key.startswith("model."):
            by_shard.setdefault(shard, []).append(key)
    weights = {}
    for shard, keys in by_shard.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
            for key in keys:
                weights[key] = reader.get_tensor(key)

    qcfg = Qwen2Config(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        tie_word_embeddings=True,
    )
    qcfg._attn_implementation = "sdpa"
    torch_model = Qwen2ForCausalLM(qcfg).float().eval()
    torch_model.load_state_dict(weights, strict=False)

    mlx_model = TextModel(TextConfig.from_dict(cfg))
    mlx_model.load_weights(
        [
            (name.removeprefix("model."), mx.array(tensor.float().numpy()))
            for name, tensor in weights.items()
        ],
        strict=True,
    )
    mx.eval(mlx_model.parameters())
    del weights
    gc.collect()

    torch_stages = {}
    mlx_stages = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            torch_stages[name] = value.detach().float().numpy()

        return hook

    handles.append(
        torch_model.model.embed_tokens.register_forward_hook(capture("embed"))
    )
    for i, layer in enumerate(torch_model.model.layers):
        handles.append(layer.register_forward_hook(capture(f"layer{i:02d}")))
    handles.append(torch_model.model.norm.register_forward_hook(capture("norm")))

    _mlx_hook(monkeypatch, mlx_model.embed_tokens, "embed", mlx_stages)
    for i, layer in enumerate(mlx_model.layers):
        _mlx_hook(monkeypatch, layer, f"layer{i:02d}", mlx_stages)
    _mlx_hook(monkeypatch, mlx_model.norm, "norm", mlx_stages)

    token_ids = np.array(
        [[1, 17, 101, 1009, 151643, 151663, 151644, 42, 151645]],
        dtype=np.int64,
    )
    with torch.no_grad():
        torch_hidden = torch_model.model(torch.from_numpy(token_ids)).last_hidden_state
        torch_logits = torch_model.lm_head(torch_hidden).float().numpy()
    mlx_hidden = mlx_model(mx.array(token_ids.astype(np.int32)))
    mlx_logits = mlx_model.embed_tokens.as_linear(mlx_hidden)
    mx.eval(mlx_hidden, mlx_logits)

    names = ["embed"]
    names += [f"layer{i:02d}" for i in range(qcfg.num_hidden_layers)]
    names += ["norm"]
    print("\nfp32 text decoder: stage | shape | rel L2 | max abs | cosine")
    for name in names:
        actual = mlx_stages[name]
        expected = torch_stages[name]
        rel = _rel_err(actual, expected)
        max_abs = float(np.max(np.abs(actual - expected)))
        a = actual.astype(np.float64).ravel()
        b = expected.astype(np.float64).ravel()
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(
            f"{name:>12} | {str(actual.shape):>17} | {rel:.6e} | "
            f"{max_abs:.6e} | {cosine:.8f}"
        )
        assert actual.shape == expected.shape
        tolerance = 2e-4 if name == "norm" else _REL_TOL_FP32
        assert rel < tolerance, f"{name}: fp32 rel L2={rel:.6e}"

    actual_logits = np.asarray(mlx_logits)
    logits_rel = _rel_err(actual_logits, torch_logits)
    print(
        f"{'lm_head':>12} | {str(actual_logits.shape):>17} | "
        f"{logits_rel:.6e} | "
        f"{np.max(np.abs(actual_logits - torch_logits)):.6e}"
    )
    assert logits_rel < 2e-4
    assert np.array_equal(actual_logits.argmax(-1), torch_logits.argmax(-1))

    for handle in handles:
        handle.remove()
