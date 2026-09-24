# ARK-ASR MLX 数值验证记录

> 工作笔记，用于 PR #2198 证据整理。所有数据实测于 2026-09-17，环境见下。
> 全部对照实验已固化为 opt-in 测试 `tests/test_model/test_arkasr_mlx_parity.py`
> （仿 `test_auk_parity.py` 先例，`accelerator` 标记，`ARKASR_PARITY_CHECKPOINT`
> 环境变量激活，CI 默认跳过），运行：
> `ARKASR_PARITY_CHECKPOINT=checkpoints/ark-asr-3b pytest tests/test_model/test_arkasr_mlx_parity.py -q`

## 环境

- 硬件：MacBook Air M4，16 GB，macOS 15.7.3，arm64
- 版本：mlx 0.32.2，mlx-lm 0.31.3，torch 2.13.0，transformers 5.12.1
- 代码：sglang-omni 分支 `apple/ark-asr-3b`，基线 commit `89227913`（+ 工作区未提交的 `sglang_omni/models/arkasr/mlx/`）
- 模型：官方 `AutoArk-AI/ARK-ASR-3B`（本地 `checkpoints/ark-asr-3b`，BF16，926 个 tensor，3.75B 参数 = LM 3.09B + 音频塔 0.66B）
- 测试音频：`tests/data/query_to_cars.wav`（4.6 s，48 kHz 重采样到 16 kHz，462 mel 帧 → 57 audio tokens）

## 证据链总表

| # | 检查项 | 结果 | 方式 |
|---|---|---|---|
| 1 | mlx_lm 原生加载 | 失败，`ValueError: Model type arkasr not supported`（`mlx_lm/utils.py:191`） | 直接调用 |
| 2 | `get_model_classes` 注入加载 | 成功，926 key 严格对齐（strict load 零缺失），峰值内存 6.99 GB | 探针 |
| 3 | 手写 Qwen2 vs mlx_lm qwen2 | **logits 逐位相同（diff = 0.0）**，argmax 全同 | 同权重双加载 |
| 4 | 音频塔 fp32 跨栈（B=1） | rel err **1.3e-5**（torch fp32 CPU 参考） | 真实 mel |
| 5 | 音频塔 fp32 跨栈（B=4） | rel err **1.03e-5**（4 行逐一） | 真实 mel |
| 6 | 文本栈 fp32 跨栈（B=4，首测） | rel err **7.1e-6**，argmax 4/4 一致 | 真实 prompt |
| 7 | fp32 栈内 batch 不变性 | torch 音频 4.8e-7 / 文本 0.0；MLX 音频 1.3e-6 / 文本 9.3e-6 | B1 vs B4 |
| 8 | 音频塔 bf16 跨栈逐层（B=1） | conv 0.27% → layer0-19 缓增 0.7%~7.3% → layer20-31 ~9.7% → adapter 输出 **5.3%** | 钩子逐层 |
| 9 | 文本栈 bf16 跨栈 | embed 0.0（逐位）→ layer0-26 稳定 0.2%~0.3% → norm 1.9% → logits **1.7%** | hidden_states |
| 10 | 端到端贪心转写 | torch/MPS 管线 vs MLX 管线 **token 逐位相同**（10/10，含 im_end EOS） | 完整管线 |
| 11 | bf16 栈内 batch 不变性 | torch 0.0；MLX 音频 3.6e-4（kernel 调度差异），文本 0.0 | B1 vs B4 |
| 12 | bf16 跨栈 B=4 | 音频 5.34%（B=1 为 5.35%），文本 1.8%（B=1 为 1.7%）——gap 不随 batch 恶化 | 同 #8/#9 口径 |
| 13 | MLX 批量解码（B=4 合批 vs 各自 B=1） | 3/4 行逐 token 相同；唯一差异 = EOS 后死区单 token near-tie 翻转（72 vs 40，下一步立即重收敛） | 8 步贪心 |

## bf16 逐层误差表（torch/MPS bf16 vs MLX bf16，真实 mel，relative L2）

```
           stage |   rel err |  max abs | ref std
     audio.conv1 |   0.00274 |   0.0312 |  0.3019
     audio.conv2 |   0.00279 |   0.0312 |  1.6316
   audio.layer00 |   0.00667 |   0.0625 |  0.4355
   audio.layer01 |   0.01245 |   0.0781 |  0.4099
        ...（layer02-18 在 1.9%~3.6% 之间缓慢波动）
   audio.layer19 |   0.07289 |   4.1250 |  0.3905
   audio.layer20 |   0.09724 | 200.0000 |  7.4404   ← massive activation 出现（std 20 倍跳升）
   audio.layer21~30 | 0.0972~0.0988 | 200.0 | 7.44~7.48
   audio.layer31 |   0.07870 | 200.0000 |  9.4752
        audio.ln |   0.08647 |   3.7812 |  0.2982   ← LayerNorm 收敛
   audio.adapter |   0.05345 |   1.1348 |  1.1850   ← 最终音频特征
      text.embed |   0.00000 |   0.0000 |  0.0111   ← 逐位相同
    text.layer00 |   0.00287 |   0.0312 |  0.7330
    text.layer09 |   0.00205 |   8.0000 | 53.9303
    text.layer17 |   0.00201 |   8.0000 | 57.3988
    text.layer26 |   0.00202 |   8.0000 | 57.6496
       text.norm |   0.01862 |   2.0000 |  1.8716
     text.logits |   0.01716 |   0.1094 |  1.2159   ← cosine ≈ 0.9998
```

**Outlier 取证**：layer20 的 squared error **84.1% 来自 top-8 维度**；torch 端 36 个维度 |v|>100
（最大 -848，MLX 侧 -648，相对差 ~24%）。这是该 checkpoint 深层的 massive activation
（attention sink 现象），bf16 下两栈舍入轨迹被逐层放大；fp32 下同一输入误差回到 1e-5，
证明与算法无关。LayerNorm/4 帧 merge 平均把 outlier 稀释回 5.3%。

## 端到端结果

- prompt 模板（与 serving `request_builders.py` 一致）：
  `<|user|><|begin_of_audio|>{57×<|audio|>}<|end_of_audio|>Please transcribe this audio.<|assistant|>`
- 贪心解码 10 token：`[5158, 1657, 9331, 525, 1052, 304, 279, 6802, 13, 151645]` —
  解码为 `So, uh.` + `<|im_end|>`，torch/MPS 与 MLX **完全一致**

## M4 Pro 性能基准

以下数据来自独立的 M4 Pro 性能测试，测试日期为 **2026-09-18**。与上面的
M4 Air 数值 parity 实验相比，硬件、测试分工和延迟口径不同；这里的 prefill
包含音频编码和 72-token prompt 前向，decode 为逐 token 自回归生成。

### 环境

- 硬件：MacBook Pro，Apple M4 Pro，48 GB RAM
- 分支：`apple/ark-asr-3b-backup`
- 模型：`AutoArk-AI/ARK-ASR-3B`，3.75B 参数
- 音频：`tests/data/query_to_cars.wav`，4.6 s，462 mel frames，57 audio tokens

### 分阶段延迟

| 配置 | Prefill | Decode 均值 | Decode P50 | Decode P95 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| MLX bf16 | 169.7 ms | 29.7 ms/token | 29.9 ms/token | 30.9 ms/token | 33.7 tok/s |
| MLX q4 | 152.0 ms | 10.6 ms/token | 10.5 ms/token | 11.2 ms/token | 94.6 tok/s |
| q4 相对 bf16 | 1.12x | 2.81x | - | - | 2.81x |

主要结论：

- q4 prefill 仅比 bf16 快约 1.12x，因为较长的矩阵乘法可以较充分地利用 GPU。
- q4 decode 达到 94.6 tok/s，相比 bf16 提升 2.81x；单 token decode 主要受权重访存限制，
  4-bit 权重显著降低了内存带宽压力。
- 以生成 10 个 token 估算，bf16 为约 `170 + 10 x 30 = 470 ms`，q4 为约
  `152 + 10 x 11 = 262 ms`。

### 完整推理循环

| 配置 | 端到端延迟 | GPU 峰值内存 | 转写结果 |
|---|---:|---:|---|
| MLX bf16 | 635 ms | 7.31 GB | `how many cars are there in the picture` |
| MLX q8 | 500 ms | 4.62 GB | `how many cars are there in the picture` |
| MLX q4 | **465 ms** | **3.18 GB** | `how many cars are there in the picture?` |
| Torch/MPS bf16 | 406 ms（仅 prefill） | 6.99 GB | 未完成 decode |

Torch/MPS 的 406 ms 只包含音频编码和 prefill，不能直接与 MLX 的完整推理循环
比较。MLX q4 的完整推理延迟为 465 ms，且峰值内存最低。

### 与当前 Q4 优化后的本地复测

在后续接入 compiled SwiGLU 和 headless text trunk 后，M4 环境上的同音频 Q4
复测结果为：

| 指标 | 优化前基线 | 优化后复测 |
|---|---:|---:|
| Decode P50 | 21.33 ms/token | 21.21 ms/token |
| Decode 吞吐 | 47.00 tok/s | 47.33 tok/s |
| Text prefill | 236.08 ms | 233.53 ms |
| Full prefill | 382.81 ms | 374.42 ms |

这组结果确认没有性能回退；decode 提升约 0.6%，接近单次 benchmark 的测量波动，
更明显的收益出现在 prefill。compiled SwiGLU 使用 `mlx_lm` 的
`mx.compile(shapeless=True)` 实现，headless text trunk 则在分块 prefill 的
非末块跳过不需要的词表投影。

## 当前干净 PR 分支的 M4 Air 复测

以下结果实测于 **2026-09-21**，用于验证基于最新 `main` 整理后的单提交 PR：

- 硬件：MacBook Air，Apple M4，16 GB
- 分支：`pr/ark-asr-3b-clean`
- Commit：`31d12e50 feat(arkasr): add native MLX serving`
- 模型：`checkpoints/ark-asr-3b`
- 音频：`tests/data/query_to_cars.wav`
- 测量方法：每种配置先 warmup 1 次，再执行 5 次完整音频编码、prefill 和 greedy decode
- Q4：`bits=4`、`group_size=64`，仅量化文本栈，音频塔保持未量化

### BF16 与 Q4 延迟、吞吐和内存

| 指标 | BF16 | Q4 | Q4 相对 BF16 |
|---|---:|---:|---:|
| 模型加载 | 3.693 s | 4.542 s | 即时量化增加 0.849 s |
| Prefill 平均 | 404.7 ms | 375.8 ms | 降低 7.1% |
| Prefill P50 | 403.4 ms | 376.0 ms | 降低 6.8% |
| Decode 平均 | 63.7 ms/token | 20.6 ms/token | **3.08x** |
| Decode P50 | 70.5 ms/token | 22.6 ms/token | **3.12x** |
| Decode P95 | 73.1 ms/token | 25.1 ms/token | **2.91x** |
| Decode 吞吐 | 15.7 tok/s | 48.4 tok/s | **3.08x** |
| 完整推理平均 | 1041.3 ms | 582.3 ms | **1.79x** |
| 完整推理 P50 | 1041.1 ms | 582.2 ms | **1.79x** |
| MLX 峰值内存 | 7.124 GB | 2.993 GB | 减少 **58.0%** |
| MLX active memory | 6.990 GB | 2.859 GB | 减少 **59.1%** |

5 次 BF16 输出完全一致：

```text
how many cars are there in the picture.
```

5 次 Q4 输出完全一致：

```text
how many cars are there in the picture?
```

两种模式的正文 token 和 EOS 相同，仅最后标点不同：

```text
BF16: token 13  (.)
Q4:   token 30  (?)
EOS:  token 151645
```

### 阶段 Profile

以下结果通过在阶段边界显式 `mx.eval()` 得到，用于说明耗时归属；它是阶段级
profile，不是 Metal kernel trace。

| 阶段 | BF16 | Q4 | 结论 |
|---|---:|---:|---|
| Audio encode | 143.2 ms | 144.4 ms | 基本不变 |
| Audio embedding merge | 0.35 ms | 0.49 ms | 可忽略 |
| Text prefill | 263.7 ms | 235.0 ms | Q4 快 1.12x |
| Decode 总耗时 | 638.1 ms | 203.9 ms | Q4 快 3.13x |

音频编码耗时不变，符合“只量化文本栈”的实现。Q4 的主要收益来自文本 prefill
和访存受限的单 token decode。

### MLX 配置和 Torch/CUDA 资源隔离

使用临时 pytest probe 直接调用 `ArkasrEngineBuilder`，结果为 `1 passed`，验证：

```text
disable_radix_cache=True
chunked_prefill_size=-1
enable_torch_compile=False
disable_cuda_graph=True
audio_encoder_service=None
```

probe 使用一个在 `set_encoder_max_batch_size()` 被调用时立即失败的模型对象。
测试通过说明 MLX 路径提前返回，没有初始化 Torch audio encoder、pre-LM encoder
service 或 CUDA encoder graph。临时 probe 已在测试完成后删除，没有进入 PR。

### 当前代码的 parity 复测

运行：

```bash
ARKASR_PARITY_CHECKPOINT=checkpoints/ark-asr-3b \
.venv-apple/bin/python -m pytest \
  tests/test_model/test_arkasr_mlx_parity.py -q -s
```

在当前代码上完成的前三项结果为 `3 passed`：

- 音频 adapter FP32 parity 通过；
- 文本 logits FP32 parity 通过；
- BF16 逐层漂移边界和首 token argmax 通过。

本次 BF16 音频塔逐层 relative L2 结果：

| 阶段 | Relative L2 |
|---|---:|
| Conv1 | 0.168% |
| Conv2 | 0.163% |
| Layer 0 | 0.426% |
| Layer 19 | 2.29% |
| Layer 20-30 | 约 8.4% |
| Layer 31 | 6.78% |
| 最终 adapter | **1.68%** |

深层误差仍集中在 massive activation 区域，最终 LayerNorm、frame merge 和 adapter
将输出误差收敛到 1.68%。

端到端 parity 用例没有完成自动断言。未提交测试仍调用旧私有方法：

```python
model._build_inputs_embeds(...)
model._forward_last_logits(...)
```

当前正式 API 已重命名为：

```python
model.build_inputs_embeds(...)
model.forward_last_logits(...)
```

因此测试结果为 `3 passed, 1 failed`，失败是测试文件 API 陈旧导致的
`AttributeError`，不是模型数值断言失败。该 opt-in parity 文件目前未跟踪，
不能在 PR 描述中写成“当前完整 parity 测试通过”，除非先更新测试并重新运行。

## 复现要点（关键操作，实现 parity 测试时的依据）

1. 加载：`mlx_lm.utils.load_model(Path(path), get_model_classes=lambda config: (ArkasrModel, ModelConfig))`
2. mel：WhisperFeatureExtractor 必须 `padding="longest"` + 按 attention_mask 裁到真实帧数
   （默认会 pad 到 30 s/3000 帧，带着 padding 跑会输出退化的 `100.0.0.0…`）
3. torch 文本参考：transformers `Qwen2ForCausalLM`（显式 Qwen2Config + `_attn_implementation="sdpa"`），
   权重过滤 `model.*` + `lm_head.weight`，`strict=False`（missing 仅 inv_freq）
4. torch hidden_states 采样注意 transformers v5 惯例：末位是 norm 后输出，与 norm 前 layer 输出对比会假报警
5. MLX 钩子：按实例身份补丁 `type(module).__call__`（MLX 无原生 hook）
6. fp32 测量：MLX 侧 `module.update(tree_map(lambda p: p.astype(mx.float32), module.parameters()))`；
   torch fp32 文本在 CPU 跑（12 GB RAM，避开与 MLX 统一内存抢）；先 torch 后 MLX，顺序不可换

## 单测覆盖（已入库，tests/unit_test/arkasr/test_mlx_model.py，21 个）

config 解析（扁平布局/默认值）、LM 前向/cache/末位 logits/untied、`_rope_safe` B3 对齐 B1、
音频塔形状/奇数帧截断/短序列补零/all-ones mask ≡ no-mask、B=2 变长 mask 与 torch 严格同权重
parity（atol=1e-5）、注入 span 替换/计数校验/B>1 拒绝、sanitize（conv 转置/tied 丢弃）、
量化 predicate、模块树 key 契约、单请求音频 prefill 端到端。整个 arkasr 目录 88 过 4 跳过。

## PR 描述建议引用口径

> - fp32 parity: audio adapter rel err 1.0e-5 / text stack 7.1e-6 vs the torch reference
>   (official checkpoint, real mel + real prompt, B=1 and B=4, argmax identical)
> - bf16 end-to-end: greedy transcripts token-identical to the torch/MPS pipeline (10/10 incl. EOS)
> - bf16 per-layer drift follows the expected sqrt(layers) random walk; deep-layer divergence
>   attributed to massive activations (top-8 dims = 84% of squared error) and eliminated in fp32;
>   batch size does not affect either the fp32 parity or the bf16 cross-stack gap

## 过程中的坑（实现笔记）

1. `mx.fast.scaled_dot_product_attention` 在 mlx 0.32 里 `scale` 是**必传关键字**，对应 torch SDPA
   默认缩放传 `head_dim**-0.5`
2. mlx_lm 的 `Qwen2Model` 是内层模块名，外层类叫 `Model`（组合时容易拿错）
3. WhisperFeatureExtractor 默认 30 s padding（见上"复现要点 2"）
4. MLX Conv1d 输入 channels-last `(B,T,C)`、权重 `(O,k,I)`；torch `(B,C,T)`/`(O,I,k)`，
   sanitize 里 `transpose(0,2,1)`
5. `.git/info/exclude` 曾有未锚定的 `models/`，静默忽略 `sglang_omni/models/` 下所有新文件
   （已改 `/checkpoints/`；本地 checkpoint 目录约定为 `checkpoints/`，与源码隔离）
6. 手写文本栈后参数路径与 checkpoint 同名（`model.*`），sanitize 只剩 conv 转置 + tied lm_head 丢弃
7. transformers v5 `Qwen2Model` hidden_states 末位 = norm 后输出（对比时注意语义对齐）

## 与仓库先例的对比

qwen3_asr (#1730) 等 Apple 模型 PR 只报告延迟和定性 smoke，无跨后端数值证据；
本套验证（fp32 精确性 + bf16 逐层曲线 + outlier 归因 + batch 行为 + 转写一致性）
已固化为 `tests/test_model/test_arkasr_mlx_parity.py`，是仓库内第一个完整的
MLX/torch 跨后端数值 parity 测试（先例 `test_auk_parity.py` 是 CUDA 单栈对上游
推理代码，非跨后端数值对比）。qwen3_asr 的 MLX 代码继承自 mlx-audio（上游已验证），
ARK 为从零手写移植，故需自证。
