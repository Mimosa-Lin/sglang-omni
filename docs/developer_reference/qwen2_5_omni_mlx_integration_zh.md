# Qwen2.5-Omni-3B 的 SGLang-Omni / MLX 集成设计

> 状态：本地设计草案，2026-09-24。本文描述建议实现路径，不代表仓库已经支持
> Qwen2.5-Omni-3B。

## 1. 目标与范围

目标是在 Apple Silicon 上为 `Qwen/Qwen2.5-Omni-3B` 增加原生 MLX
推理能力，并尽量复用 SGLang-Omni 已有的多阶段调度、请求生命周期、流式输出和
量化基础设施。

建议将工作拆成三个明确层级：

1. **第一个可合并里程碑**：先验证官方 Transformers 的 Torch/MPS 路径，再实现
   单请求、greedy 的 MLX Thinker，只支持文本输入和文本输出。
2. **第一阶段完整目标**：增加短音频输入到文本输出，并完成 MLX Q4、HTTP 服务、
   请求取消和内存回收验证。
3. **完整目标**：文本、图像、视频和音频输入，同时输出文本和流式语音，覆盖
   Thinker、Talker、Token2Wav 全链路。

第一阶段不应承诺语音输出、视频输入、并发批处理或 radix cache。这些能力需要在
基础 parity、内存和生命周期正确之后逐项增加。Torch/MPS 是本机参考和可行性验证
后端，不替代最终的原生 MLX 实现。

## 2. 当前仓库基础

当前仓库还没有独立的 Qwen2.5-Omni pipeline，但已经具备四类可复用基础。

### 2.1 Qwen3-Omni 的完整 pipeline 骨架

`sglang_omni/models/qwen3_omni/` 已经实现：

- preprocessing、image encoder、audio encoder、Thinker、Talker、Code2Wav；
- 多输入 fan-out 和 encoder 结果 fan-in；
- Thinker hidden state 向 Talker 的流式传递；
- 文本与音频两个终端输出；
- abort、stream done、Talker backpressure 和请求级状态回收；
- 多阶段 placement 和统一显存预算。

Qwen2.5-Omni 应复用这套 stage topology 和生命周期设计，但不能直接复用
Qwen3-Omni 的模型类、position 规则或 Code2Wav 实现。

### 2.2 Qwen3-ASR 的 MLX 音频 prefill

`sglang_omni/model_runner/audio_mlx.py` 和
`sglang_omni/models/qwen3_asr/mlx/` 已经提供：

- 音频占位 token 到连续 embedding 的替换；
- MLX KV cache 获取、prefill 和单步 decode；
- SGLang scheduler 与 MLX lazy execution 的衔接；
- 官方或转换后 checkpoint 的 MLX 加载入口。

第一阶段的 Qwen2.5-Omni audio-to-text 可以从这里扩展。不过 Qwen2.5-Omni
还需要 M-RoPE、混合模态位置和非连续媒体 span，不能直接继承当前只处理一个连续
音频 span 的实现。

### 2.3 AuK 中的 Qwen2.5-Omni Thinker 参考

`sglang_omni/models/auk/reference_encode.py` 已使用
`Qwen2_5OmniProcessor` 和 `Qwen2_5OmniThinkerForConditionalGeneration`。
它可以作为以下内容的 PyTorch 参考：

- chat template 和 processor 输出；
- 音频特征输入格式；
- Thinker 权重名称；
- hidden states parity。

AuK 会关闭视觉模块，并明确忽略 Talker 和 Token2Wav，因此它不是完整集成，也不能
直接作为 MLX runner。

### 2.4 Fun-CosyVoice3 的原生 MLX Flow/DiT/HiFT

`sglang_omni/models/fun_cosyvoice3/mlx/vocoder/` 已有 MLX 版本的：

- Flow matching；
- DiT；
- vocoder；
- 权重清洗和映射；
- MLX graph 预物化与 NumPy 波形边界。

这些代码适合复用算子写法和工程模式。Qwen2.5-Omni 使用自己的 Token2Wav 权重、
block attention、ODE 求解器和 BigVGAN，不能直接加载到 CosyVoice3 模型中。

## 3. 模型数据流

Qwen2.5-Omni 的完整推理可抽象为：

```text
text/image/video/audio
          |
          v
      Processor
          |
     +----+----+
     |         |
 Vision     Audio
 Encoder    Encoder
     |         |
     +----+----+
          |
       Thinker -----------------> text tokens
          |
          | per-token hidden states + token embeddings
          v
        Talker -----------------> codec tokens
          |
          v
   Token2Wav DiT --RK4/CFG-----> mel
          |
       BigVGAN -----------------> waveform
```

需要特别注意三个契约。

### 3.1 Thinker 契约

Thinker 是一个多模态 causal LM。媒体 encoder 输出必须替换 prompt 中对应的
audio/image/video placeholder embedding，并使用官方一致的三轴 M-RoPE position。

当需要语音输出时，Thinker 还必须返回每个生成步的 hidden states。只保留最终文本
token 不足以驱动 Talker。

### 3.2 Talker 契约

Talker 不是一个独立的普通 TTS LM。官方 Transformers 实现会组合：

- Thinker prompt 的 token embedding 和 hidden state；
- Thinker 回复部分每一步的 token embedding 和末层 hidden state；
- speaker 对应的 BOS token；
- codec BOS、PAD、MASK token；
- 图像、视频、音频长度以及 M-RoPE 状态。

Talker 每步消费 Thinker 回复信息并生成 codec token。实现时必须保留 request-local
FIFO、两个 KV cache 和严格的结束状态，不能只在 Thinker 全部完成后拼一个文本字符串
重新做 TTS。

### 3.3 Token2Wav 契约

当前 Transformers 5.12.1 的 Qwen2.5-Omni 实现显示：

- DiT hidden size 为 1024，22 层，16 个 attention heads；
- codec embedding 会按 `repeats=2` 扩成 mel 时间轴；
- 默认执行 10 个 flow-matching 时间点；
- ODE solver 是 RK4，因此一个时间步不等于一次 DiT forward；
- CFG 开启时会同时计算 conditional 和 unconditional 分支；
- 输出 80 维 mel，再由 BigVGAN 生成波形；
- 官方实现强制 Token2Wav 使用 FP32；
- audio output 当前只支持 batch size 1。

因此 Token2Wav 很可能是 16GB Mac 上的主要延迟和峰值内存来源。第一版不应把它和
Thinker MLX 支持绑在同一个 PR 中。

## 4. 推荐 pipeline

建议新增目录：

```text
sglang_omni/models/qwen2_5_omni/
|-- __init__.py
|-- config.py
|-- bootstrap.py
|-- payload_types.py
|-- request_builders.py
|-- routing.py
|-- stages.py
|-- mrope_positions.py
|-- thinker_model_runner.py
|-- talker_model_runner.py
|-- talker_scheduler.py
|-- components/
|   |-- preprocessor.py
|   |-- audio_encoder.py
|   |-- image_encoder.py
|   |-- sglang_thinker.py
|   |-- talker.py
|   `-- token2wav.py
`-- mlx/
    |-- config.py
    |-- model.py
    |-- runner.py
    |-- audio_encoder.py
    |-- vision_encoder.py
    |-- thinker.py
    |-- talker.py
    |-- token2wav_dit.py
    `-- bigvgan.py
```

不要一次创建所有文件。每个阶段只添加当前实现真正需要的模块。

### 4.1 文本输出 pipeline

```text
preprocessing
  |-- image_encoder --+
  |-- audio_encoder --+--> mm_aggregate --> thinker --> decode
  `-------------------+
```

第一版只有文本和音频时，可以先简化为：

```text
preprocessing --> thinker --> decode
```

音频 encoder 可暂时在 Thinker 的 MLX prefill 内执行，等功能稳定后再拆成独立 stage。
这种实现范围最接近现有 Qwen3-ASR MLX runner。

### 4.2 语音输出 pipeline

```text
preprocessing
  |-- image_encoder --+
  |-- audio_encoder --+--> thinker --> decode
  |                   |       |
  `-------------------+       +--> talker_ar --> token2wav
```

建议沿用 Qwen3-Omni 的 stage 名称，使 shared runtime、profiling 和 placement 工具保持
一致。模型差异放在 model-local request builder、runner 和 component 中。

## 5. 分阶段实施

### Phase 0：MPS 可行性验证和官方参考基线

先验证官方 Transformers checkpoint 在当前 Apple Silicon 环境中的实际行为，再开始
MLX 移植。这样可以提前区分官方模型、Torch/MPS 算子和 MLX 实现各自的问题。

环境检查至少记录：

- Mac 型号和统一内存容量；
- macOS、Python、PyTorch、Transformers、MLX 和 MLX-LM 版本；
- `torch.backends.mps.is_available()` 和 MPS fallback 设置；
- checkpoint revision、配置中的 architecture 和实际权重前缀；
- 模型加载时间、进程 RSS 和 MPS 峰值内存。

按以下顺序验证：

1. 只加载 `Qwen2_5OmniThinkerForConditionalGeneration`，避免一开始加载 Talker 和
   Token2Wav。
2. 在 CPU 上运行一个最小纯文本样例，作为数值参考。
3. 在 MPS 上运行同一纯文本样例，记录不支持的算子、首 token logits top-k、greedy
   token 序列、TTFT 和 tokens/s。
4. 若纯文本可用，再运行一段短音频到文本，记录 processor 输出、audio encoder
   shape、placeholder 数量、position IDs、rope delta 和生成 token。
5. MPS 某条路径不可用时，保留具体失败算子和堆栈，并使用 CPU Transformers 结果
   作为该模块的 parity 基线，不阻塞 MLX 开发。

MPS 验证结果分为三类：

- **完整可用**：作为本机 MLX parity 的主要参考后端；
- **仅纯文本可用**：文本使用 MPS，音频和其他模态使用 CPU 参考；
- **无法稳定运行**：全部使用 CPU 生成参考值，MPS 失败作为已知限制记录。

后续图像、视频和语音输出阶段继续增加官方参考样例。每个样例只保存可重算的输入、
token IDs、媒体 grid、position IDs、shape、logits 和统计量，不提交大型二进制 tensor。

### Phase 1：独立的 MLX 纯文本 Thinker

先验证模型实现本身，不在这一阶段同时引入完整 pipeline。

实现目标：

- 加载 `Qwen/Qwen2.5-Omni-3B` 的官方 BF16 权重；
- 单请求、greedy prefill 和 decode；
- 文本输入和文本输出；
- 与 CPU/MPS Transformers Thinker 做 BF16 parity；
- 验证 KV cache prefill/decode 一致性。

最小目录：

```text
sglang_omni/models/qwen2_5_omni/
|-- __init__.py
`-- mlx/
    |-- __init__.py
    |-- config.py
    |-- thinker.py
    |-- model.py
    `-- runner.py
```

主要改动：

1. 在 `mlx/config.py` 中解析 root config 的 `thinker_config.text_config`。
2. 在 `mlx/thinker.py` 中实现 embedding、RMSNorm、GQA、RoPE、gated MLP 和 KV
   cache。
3. 在 `mlx/model.py` 中提供 prefill、单步 decode、cache 创建和权重 sanitize 接口。
4. 在 `mlx/runner.py` 中仿照 Qwen3-ASR 调用 `mlx_lm.utils.load_model`。
5. 为 tied/untied LM head、权重转置和 checkpoint 前缀编写单元测试。

不要直接复用 Qwen3-Omni 的 MoE Thinker。Qwen2.5-Omni-3B 是不同的 dense
architecture，权重前缀和配置层级也不同。

### Phase 2：接入 SGLang-Omni 文本服务

模型 parity 通过后，再接入当前仓库的正式调度和服务生命周期：

1. 新增 `config.py`、`engine_builder.py`、`request_builders.py` 和 `stages.py`。
2. 通过带 `EntryClass` 的 pipeline config 注册模型。
3. 在 `model_runner/mlx_model_worker.py` 增加 architecture dispatch。
4. 使用现有 `MlxSchedulerModelRunner` 管理 MLX lazy decode、cache pool 和输出。
5. 打通 CLI、HTTP、流式文本、abort 和客户端断连。

第一版配置限制：

- `max_running_requests=1`；
- `disable_radix_cache=True`；
- `chunked_prefill_size=-1`；
- greedy decode；
- 不接受图像、视频、音频和语音输出参数。

阶段验收是通过真实 HTTP 请求完成纯文本生成，并在正常结束、取消和断连后释放
request state 与 KV cache。

### Phase 3：音频输入到文本

实现音频 encoder、embedding 注入和多模态位置：

1. 复用官方 processor 生成 128-bin log-mel 和 attention mask。
2. 用 MLX 重写 Qwen2.5-Omni audio encoder。
3. 精确复现 encoder 的卷积降采样和输出长度公式。
4. 将 audio embedding 写入所有 audio placeholder span。
5. 计算官方一致的三轴 M-RoPE position IDs。
6. 将媒体 span 和 position metadata 传入 MLX prefill。

可以参考 `AudioMlxModelRunner`，但不能直接继承它当前的全部假设：

- 只有一个媒体 item；
- audio placeholder 必须是唯一连续 span；
- position 只有普通一维 RoPE；
- prefill 不需要额外的 multimodal metadata。

第一版继续保持单请求、greedy、禁用 radix cache 和 chunked prefill，并限制最大音频
长度。短音频到文本稳定后，再决定哪些通用逻辑可以下沉到共享 runner。

### Phase 4：Q4、性能和生命周期稳定

1. 支持官方 checkpoint 和转换后的 MLX Q4 artifact。
2. 比较 BF16 与 Q4 的首 token logits、greedy token 序列和短音频识别结果。
3. 测量模型加载、TTFT、prefill tokens/s、decode tokens/s 和统一内存峰值。
4. 验证 MLX lazy graph 不会保留已经无消费者的中间 tensor。
5. 验证正常结束、abort、断连和连续请求后的 cache 与内存回收。
6. 在这些结果稳定后，再单独评估 radix cache、chunked prefill 和并发请求。

### Phase 5：图像和视频输入到文本

图像支持需要：

- Qwen2.5-VL 风格 patch embedding；
- window/full attention 的层级切换；
- patch merger；
- image grid 到 placeholder 数量的严格校验；
- 三轴 M-RoPE。

视频还需要：

- temporal patch；
- `video_grid_thw`；
- `video_second_per_grid`；
- 音频与视频按真实时间交错；
- `use_audio_in_video` 的一致语义。

建议先做单图，再做无音轨短视频，最后做音视频联合输入。不要把三种情况放在第一个
视觉 PR 中。

### Phase 6：Talker 生成 codec token

Talker 阶段应先只输出 codec token，不接 Token2Wav：

1. 从 Thinker 捕获官方所需 hidden states；
2. 构造 Talker prefill 的 text/codec token 和 embedding；
3. 实现 Talker 自己的 M-RoPE；
4. 维护独立 KV cache；
5. 实现 top-k、top-p、temperature 和 repetition penalty；
6. 对齐固定 seed 下的 codec token 序列。

这里应优先复用 Qwen3-Omni 的：

- stream payload；
- pending text FIFO；
- Talker scheduler 生命周期；
- cancel 和 stream-done 处理。

不要直接复用其 Talker model，因为 Qwen2.5 的输入 embedding 构造、vocab、位置规则和
hidden-state 消费方式不同。

### Phase 7：Token2Wav 和语音输出

Token2Wav 分成混合后端和原生 MLX 两个子阶段。

#### Phase 7A：混合后端 Token2Wav

完整 MLX DiT 之前，先实现：

```text
MLX Thinker + MLX Talker + Torch/MPS Token2Wav
```

这个阶段可以验证：

- codec token 到最终音频的端到端正确性；
- speaker 参数加载；
- 请求结束和波形返回；
- 16GB Mac 的实际峰值内存；
- Token2Wav 是否需要按请求加载和卸载。

混合后端边界只传 codec token、speaker conditioning 和 reference mel，避免传递大型
hidden states。Torch/MPS 和 MLX 在同一进程共享 Metal 资源时要显式控制执行顺序，
不要并行提交两个大型 graph。

#### Phase 7B：原生 MLX Token2Wav

按以下顺序移植：

1. speaker encoder 和 conditioning；
2. codec embedding 与 input embedding；
3. block-wise DiT attention mask；
4. adaLN 和 rotary embedding；
5. RK4 solver 与 sway schedule；
6. CFG 双分支；
7. BigVGAN；
8. 窗口化和流式波形输出。

首先对齐 FP32。确认 mel 和音频质量后，再单独评估哪些线性层可使用 BF16 或 Q4。
不要默认整个 Token2Wav 都能安全量化。

## 6. 权重加载与量化

### 6.1 Artifact 设计

推荐保留官方 checkpoint 作为 tokenizer、processor、speaker map 和配置来源，并允许
MLX stage 指向独立转换 artifact：

```text
official checkpoint
|-- config.json
|-- tokenizer / processor
|-- spk_dict.pt
`-- original safetensors

MLX artifact
|-- config.json
|-- model.safetensors
`-- quantization metadata
```

配置应分别表达：

- `model_path`：官方模型目录；
- `mlx_model_path`：转换后的 MLX 权重；
- `mlx_model_revision`：MLX artifact revision。

这与 Fun-CosyVoice3 将官方 bundle 和 MLX vocoder artifact 分开的方式一致。

### 6.2 权重映射

转换器至少要识别：

- `thinker.audio_tower.*`；
- `thinker.visual.*`；
- `thinker.model.*` 和 LM head；
- `talker.*`；
- `token2wav.code2wav_dit_model.*`；
- `token2wav.code2wav_bigvgan_model.*`。

转换时应结构化读取 safetensors，不要通过字符串替换猜测 tensor shape。每个映射规则
都要有：

- 源名称；
- 目标名称；
- 是否 transpose；
- 是否拆分 fused QKV；
- 是否折叠 weight normalization；
- 目标 dtype；
- 是否允许量化。

### 6.3 量化边界

16GB 目标建议：

- Thinker linear layers：Q4；
- Talker linear layers：Q4；
- embedding、norm、卷积、encoder projection：BF16/FP16；
- audio/vision encoder：先 BF16，再逐层评估 Q4；
- Token2Wav DiT：先 FP32 parity，再评估 BF16；
- BigVGAN：优先 FP16/BF16，不优先 Q4。

量化验收不能只看模型能否加载。至少比较文本正确性、ASR WER/CER、视觉问答结果、
codec token 一致率、speaker similarity 和音频可懂度。

## 7. 16GB Apple Silicon 策略

16GB 是统一内存，不应同时为每个 stage 复制完整 checkpoint。建议：

1. `max_running_requests=1`。
2. 初始 `max_seq_len` 使用 4096 或更小的实测值，而不是直接沿用 32768。
3. 限制图片尺寸、视频帧数和音频时长。
4. Thinker 和 Talker 使用 Q4，KV cache 保持较小。
5. Token2Wav 先采用串行执行。
6. 禁止在完整链路初版中同时保留不再使用的 encoder activation。
7. 每个阶段完成后调用 `mx.eval`，并在所有消费者完成后释放 request-local tensor。
8. 记录 `mx.get_active_memory()`、`mx.get_peak_memory()` 和进程 RSS。
9. 分别报告权重加载峰值、prefill 峰值、decode 峰值和 Token2Wav 峰值。

如果完整链路仍超过预算，可以评估两种方案：

- **顺序驻留**：文本/codec 完成后卸载 Thinker，再加载 Token2Wav；
- **混合后端/进程**：MLX 负责 AR，Torch/MPS 或 CPU 负责 Token2Wav。

Apple 统一内存下，多进程并不会自动避免权重重复；必须用实际峰值数据选择方案。

## 8. 测试计划

### 8.1 GPU-free 单元测试

- root config 到 MLX dataclass 的解析；
- 权重名称映射和 shape 校验；
- audio/image/video placeholder 数量；
- M-RoPE position IDs；
- Talker prefill token 与 embedding 布局；
- payload 序列化；
- abort、disconnect 和 stream-done 后的状态清理；
- 不支持参数返回明确错误。

### 8.2 Apple 实机 parity

每个阶段先比较 BF16，再比较 Q4：

| 阶段 | 最低验收 |
|---|---|
| Torch/MPS 预检 | 模型加载、算子支持、CPU/MPS greedy 结果和峰值内存 |
| Text Thinker | 首 token logits、greedy token 序列 |
| MLX KV cache | 完整 forward 与 prefill + decode 的 logits 一致 |
| Text HTTP E2E | 流式文本、取消、断连和重复请求 |
| Audio encoder | 输出 shape、有效长度、选定位置误差 |
| Audio-to-text | 固定样例文本和小型 ASR 集 |
| Vision encoder | 输出 shape、单图问答 |
| M-RoPE | 官方 position IDs 完全一致 |
| Talker | 固定 seed codec token 序列 |
| DiT | 固定噪声下 mel 统计和逐步误差 |
| BigVGAN | 波形长度、频谱和音频质量 |
| Speech HTTP E2E | 文本、音频、取消、断连和重复请求 |

浮点误差阈值应按模块和 dtype 定义，不能给所有模块使用同一个容差。

### 8.3 性能与稳定性

在 M4 16GB 上至少记录：

- 冷启动和 warm-cache 启动时间；
- TTFT；
- text tokens/s；
- 首个 codec token 延迟；
- 首包音频延迟；
- 整体 RTF；
- 各阶段 active/peak MLX memory；
- 50 次连续请求后的稳定内存；
- 请求取消和客户端断连后的内存恢复。

## 9. 建议 PR 拆分

推荐按以下顺序提交，避免一个 PR 同时改变模型、运行时和声码器。

1. **PR 1：MPS 预检和 reference parity**
   - 固定环境和 checkpoint，增加 CPU/MPS 文本参考脚本、processor fixture 和失败
     算子记录。
2. **PR 2：MLX text-only Thinker**
   - BF16 dense decoder、KV cache、权重加载和单元测试，不接完整服务。
3. **PR 3：SGLang-Omni text serving**
   - pipeline 注册、MLX worker dispatch、单请求 greedy 文本 API 和生命周期测试。
4. **PR 4：audio-to-text**
   - audio encoder、embedding 注入、Apple 实机准确率。
5. **PR 5：Q4 和运行时稳定**
   - Q4 质量、性能、峰值内存、abort 和连续请求验证。
6. **PR 6：image-to-text**
   - 单图 encoder 和 M-RoPE。
7. **PR 7：video/audio interleave**
   - 短视频和联合时间轴。
8. **PR 8：MLX Talker**
   - codec token 输出，不包含波形。
9. **PR 9：混合后端 Token2Wav**
   - 完整语音输出和内存报告。
10. **PR 10：原生 MLX Token2Wav**
   - DiT、RK4、CFG、BigVGAN 和流式优化。

第一份实现 PR 的标题和范围可以是：

```text
[Apple][Qwen2.5-Omni] Add native MLX Thinker text inference
```

不要在第一份 PR 中宣称完整 Qwen2.5-Omni 支持。

## 10. 第一阶段完成标准

第一个可合并里程碑可以视为完成，需要同时满足：

- 已记录 Torch/MPS 预检结果；若 MPS 不可用，已有具体失败原因和 CPU 参考结果；
- 能从官方配置和 BF16 checkpoint 加载 MLX Thinker；
- 在 M4 16GB 上通过真实 HTTP 请求；
- 支持文本输入和文本输出；
- 单请求 greedy 结果与 CPU/MPS Transformers 基线一致；
- 不支持的音频、图像、视频和语音输出明确报错，不能静默降级；
- abort 和断连后 KV cache、请求状态和媒体 tensor 被释放；
- 文档记录 checkpoint、commit、macOS、Python、MLX、Transformers 和 SGLang
  版本；
- 报告启动时间、TTFT、tokens/s、active/peak memory；
- GPU-free 单元测试可以在普通 CI 收集和运行；
- Apple 实机测试通过环境变量显式启用，不影响非 Apple CI。

第一阶段完整目标还需要：

- 支持短音频输入和文本输出；
- audio encoder、placeholder 注入和 M-RoPE 与 Transformers 基线对齐；
- 能从 MLX Q4 artifact 启动；
- Q4 差异、短音频识别结果和内存收益有明确报告；
- 音频请求完成、取消或断连后，媒体 tensor 和 request-local state 被释放。

## 11. 开始实现前必须确认的未知项

实现前应使用实际 `Qwen/Qwen2.5-Omni-3B` checkpoint 核对：

1. 3B checkpoint 的真实 Thinker、Talker 和 Token2Wav 配置，不能使用 7B 默认值代替。
2. safetensors 的完整权重前缀和分片索引。
3. `spk_dict.pt` 的 speaker、conditioning 和 reference mel shape。
4. 官方 processor 对音频、图片、视频和联合输入生成的准确字段。
5. 当前 Transformers 版本与模型发布时 reference code 的行为差异。
6. 社区 MLX Q4 artifact 是否包含完整 Thinker/Talker/Token2Wav，还是只有部分权重。
7. 量化 metadata 是否符合当前 `mlx_lm.utils.load_model` 的加载契约。
8. 16GB 主机上 BF16 reference 和 Q4 runtime 是否能分别完成最小样例。

以上信息未确认前，不应固定模型尺寸、token ID、采样参数或内存预算到代码中；这些值
应从 checkpoint config 自顶向下传递。
