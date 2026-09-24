# ARK-ASR 五次提交完整改动说明

本文记录分支 `apple/ark-asr-3b` 上以下五个提交的全部文件改动：

| 顺序 | Commit | 标题 | 文件数 | 代码统计 |
|---|---|---|---:|---:|
| 1 | `68f77376` | `feat(arkasr): add native MLX backend` | 7 | +1715 / -0 |
| 2 | `34971921` | `feat(arkasr): optimize MLX quantized inference` | 8 | +339 / -9 |
| 3 | `1c757d01` | `feat(arkasr): complete MLX serving integration` | 10 | +995 / -30 |
| 4 | `c1e5b5bf` | `refactor(arkasr): narrow dynamic types` | 6 | +137 / -65 |
| 5 | `03655b45` | `test(arkasr): streamline MLX coverage` | 3 | +3 / -724 |

五次提交的演进关系如下：

1. 第一次提交完成 ARK-ASR 的原生 MLX 模型移植，并建立 Torch/MLX 数值一致性测试。
2. 第二次提交把 MLX runner 接入统一 worker，加入 Q4/Q8 文本栈量化，并优化解码路径。
3. 第三次提交补齐正式 server 接入、请求安全检查、缓存、长音频、实时流和端到端测试。
4. 第四次提交收窄 ARK/MLX 代码中的动态类型，使用具体类型和 Protocol 表达真实接口。
5. 第五次提交按 Qwen3-ASR 的测试层级精简实验型 parity 测试，保留常规单元测试和 server 集成测试。

---

## Commit 1：新增原生 MLX 后端

Commit：`68f77376cfdac829d9f8fb936d63426402615555`

标题：`feat(arkasr): add native MLX backend`

### 1. `arkasr_mlx_validation_notes.md`

新增 ARK-ASR MLX 数值验证工作记录，主要包含：

- 记录测试环境：Apple M4、MLX、MLX-LM、Torch、Transformers 和模型版本。
- 记录官方 ARK-ASR-3B checkpoint 的参数规模和测试音频信息。
- 说明原生 `mlx_lm` 无法直接识别 `arkasr` 模型类型，需要通过 `get_model_classes` 注入自定义模型类。
- 记录模型严格加载结果，确认 checkpoint 参数键与 MLX 模块树能够对齐。
- 记录音频塔 FP32 跨后端误差。
- 记录文本 decoder FP32 跨后端误差。
- 记录 BF16 音频塔和文本 decoder 的逐层误差。
- 分析音频塔深层 massive activation 对 BF16 误差的影响。
- 记录 Torch/MPS 和 MLX 端到端贪心解码 token 一致性。
- 记录 batch=1 和 batch=4 的批处理稳定性。
- 记录 batched decode 与逐请求 decode 的对比结果。
- 总结 WhisperFeatureExtractor 不能保留默认 30 秒 padding 的原因。
- 总结 MLX Conv1d 和 Torch Conv1d 权重布局差异。
- 总结 MLX RoPE、SDPA、hidden states 和 checkpoint key 对齐过程中的问题。
- 给出 parity 测试运行命令和 PR 描述建议。

该文件是实验记录，不参与模型运行。

### 2. `sglang_omni/models/arkasr/mlx/__init__.py`

新增 ARK-ASR MLX 子包入口：

- 声明该目录为 ARK-ASR 原生 MLX 后端。
- 为后续 `config.py`、`model.py` 和 `runner.py` 提供包级组织。

### 3. `sglang_omni/models/arkasr/mlx/config.py`

新增 MLX 模型配置解析，实现 checkpoint 配置到 MLX 模块配置的转换。

#### `AudioEncoderConfig`

- 定义 Whisper 风格音频 encoder 的维度、层数、attention head 数、FFN 维度和 mel 维度。
- 定义最大音频位置长度、激活函数和 dropout 等参数。
- `from_dict()` 只读取 dataclass 支持的字段，忽略 checkpoint 中无关配置。
- `head_dim` 根据 `d_model / encoder_attention_heads` 计算每个 attention head 的维度。

#### `TextConfig`

- 定义 Qwen2 文本 decoder 的词表大小、hidden size、FFN 维度和层数。
- 定义 attention head、KV head、最大位置长度和 RoPE 配置。
- 定义 RMSNorm epsilon、embedding 是否绑定、激活函数和 cache 配置。
- `from_dict()` 负责过滤并加载 Qwen2 文本配置。
- `head_dim` 根据 `hidden_size / num_attention_heads` 计算。

#### `ModelConfig`

- 将 ARK checkpoint 顶层的 Qwen2 参数组装为 `TextConfig`。
- 将 checkpoint 中的 `whisper_config` 转换为 `AudioEncoderConfig`。
- 保存 `audio_token_id`、`merge_factor`、adapter 类型、adapter 激活函数和 `use_rope`。
- `__post_init__()` 支持配置对象和原始字典两种输入。
- `from_dict()` 兼容 ARK checkpoint 的“文本参数平铺、音频配置嵌套”布局。
- 导出 `ModelArgs = ModelConfig`，满足 MLX-LM 模型加载接口。

### 4. `sglang_omni/models/arkasr/mlx/model.py`

新增完整 ARK-ASR MLX 模型，包括音频塔、音频 adapter、Qwen2 decoder 和多模态 embedding 替换。

#### `_rope_safe`

- 包装 MLX `nn.RoPE`。
- 规避 batch 大于 1、序列长度为 1 时 `mx.fast.rope` 破坏后续 batch 行的问题。
- 对这种特殊输入临时补到长度 2，执行 RoPE 后再裁回长度 1。
- 保证 batched single-token decode 与逐请求 decode 一致。

#### `ArkRotaryEmbedding`

- 实现 ARK 音频塔自己的 RoPE cache。
- 按 `rope_ratio` 调整 RoPE base。
- 生成交错排列的 cosine/sine cache。
- 该实现对应 Torch 音频塔中的 `ArkRotaryEmbedding`，不是直接使用文本 decoder 的 RoPE。

#### `apply_rotary_pos_emb`

- 将音频 attention 的 query/key 拆成旋转部分和透传部分。
- 对成对维度应用 cosine/sine 旋转。
- 将旋转结果和未旋转维度重新拼接。

#### `WhisperRoPESdpaAttention`

- 实现带 RoPE 的 Whisper self-attention。
- 对齐 checkpoint 中 q/v 有 bias、k 无 bias 的权重结构。
- 将输入转换为多头 attention 布局。
- 对 query 和 key 应用音频 RoPE。
- 调用 MLX 原生 scaled dot-product attention。
- 显式使用 `head_dim**-0.5` 作为 attention scale。

#### `WhisperSpecialEncoderLayer`

- 实现音频 encoder 的 pre-norm self-attention 残差块。
- 实现 GELU FFN 残差块。
- 对 FP16 路径增加有限值 clamp，避免大激活溢出。
- BF16 路径保留原始数值范围。

#### `ArkAudioTower`

- 实现 mel 输入的两层 Conv1d frontend。
- 处理 Torch `(B, mel, T)` 和 MLX `(B, T, C)` 的布局转换。
- `conv2` 使用 stride=2 对时间维降采样。
- 支持右侧 padding mask。
- 将输入 mask 下采样成 attention mask。
- 在 convolution 和每层 encoder 后清除 padding 位置。
- `use_rope=True` 时使用自定义音频 RoPE。
- `use_rope=False` 时退回 learned position embedding。
- 保留 checkpoint 中的 `embed_positions` 参数，使权重能够严格加载。

#### `_Gelu`

- 提供单独的 GELU module。
- 保持 adapter 的 module index 与 checkpoint 的 `adapting.0`、`adapting.2` 参数布局一致。

#### `ArkAudioMLPAdapter`

- 组合 `ArkAudioTower`、LayerNorm 和两层 MLP adapter。
- 按 `merge_factor` 合并连续音频帧。
- 对不能整除 merge factor 的尾部进行裁剪。
- 对比 merge factor 更短的极短输入进行补零。
- 将音频 encoder hidden size 投影到语言模型 hidden size。
- 只允许 checkpoint 使用的 GELU adapter。

#### `TextAttention`

- 手写 Qwen2 grouped-query attention。
- 对齐 q/k/v bias 和无 bias 的 output projection。
- 支持不同数量的 query heads 和 KV heads。
- 使用 MLX-LM cache 更新 key/value。
- 支持 causal mask 和增量 decode。
- 文本 RoPE 通过 `_rope_safe` 处理批量单 token 场景。

#### `TextMLP`

- 实现 Qwen2 SwiGLU MLP。
- 包含 gate、up 和 down 三个 projection。

#### `TextDecoderLayer`

- 实现 Qwen2 的 RMSNorm、self-attention、MLP 和两次残差连接。

#### `TextModel`

- 实现 token embedding、decoder layer 列表和最终 RMSNorm。
- 支持 `input_ids` 或预先构造的 `inputs_embeds`。
- 支持每层 KV cache。
- 使用 MLX-LM 的 causal attention mask。

#### `ArkasrModel`

- 将 Qwen2 文本模型和 ARK 音频 encoder 组合为完整模型。
- 支持 tied 和 untied LM head。
- `get_audio_features()` 将 mel 编码成语言模型可消费的音频 embedding。
- 当前音频 prefill 明确限制为单请求。
- `_build_inputs_embeds()` 将 `<|audio|>` placeholder 对应位置替换为音频 embedding。
- 校验 placeholder 数量、音频 feature 数量和 audio span 边界。
- `_forward_last_logits()` 只对最后一个位置执行 LM head，减少 prefill 投影计算。
- `__call__()` 支持普通文本前向和带预构造 embedding 的前向。
- `make_cache()` 为每层创建 MLX-LM KV cache。
- `sanitize()` 将 Torch Conv1d 权重从 `(out, in, kernel)` 转成 MLX `(out, kernel, in)`。
- tied embedding 时丢弃单独的 `lm_head.weight`。
- `model_quant_predicate()` 声明量化时跳过音频塔，只允许量化文本栈。

### 5. `sglang_omni/models/arkasr/mlx/runner.py`

新增 ARK-ASR MLX runner mixin：

- 继承共享 `AudioMlxModelRunner`。
- 使用 SGLang MLX remote-code gate 解析模型目录和校验 `trust_remote_code`。
- 通过 `mlx_lm.utils.load_model()` 加载官方 checkpoint。
- 使用 `get_model_classes` 注入 `ArkasrModel` 和 `ModelConfig`。
- 记录模型加载开始和耗时。
- `make_arkasr_mlx_runner_class()` 在 MLX 后端确定后，动态组合 `ArkasrMlxModelRunner` 与 SGLang `MlxModelRunner`。
- 复用 SGLang 原有 cache、token pool、radix 状态和批量 decode 实现。

### 6. `tests/test_model/test_arkasr_mlx_parity.py`

新增 opt-in 的真实 checkpoint Torch/MLX 一致性测试。

#### 测试基础设施

- 使用 `ARKASR_PARITY_CHECKPOINT` 指定本地 checkpoint。
- 使用 `ARKASR_PARITY_AUDIO` 指定测试音频。
- 没有配置 checkpoint 时自动跳过，不增加常规 CI 成本。
- 读取真实 WAV 并重采样到 16 kHz。
- 使用 WhisperFeatureExtractor 构造真实 mel。
- 构造与 serving 相同的 ARK audio prompt。
- 实现 MLX module hook，用于记录指定层输出。
- 分别加载 Torch/MPS 和 MLX 的真实权重。

#### `test_audio_adapter_fp32_parity`

- 比较 Torch 和 MLX 音频 adapter 的 FP32 输出。
- 同时覆盖 batch=1 和 batch=4。
- 检查 relative L2 error 和 batch invariance。

#### `test_text_logits_fp32_parity`

- 比较 Torch Qwen2 和 MLX 文本 decoder 的 FP32 logits。
- 检查 argmax 是否一致。
- 检查 batch=1 和 batch=4 的输出稳定性。

#### `test_bf16_layer_drift_bounded`

- 对音频塔 conv、encoder layer、LayerNorm、adapter 逐层记录 BF16 输出。
- 对文本 embedding、decoder layer、norm 和 logits 逐层记录 BF16 输出。
- 打印逐层 relative error、max absolute error 和参考标准差。
- 为最终音频 feature 和文本 logits 设置误差上限。
- 验证 batch 增大不会导致跨栈误差失控。

#### `test_greedy_transcript_identical`

- 分别通过 Torch/MPS 和 MLX 完成音频 prefill 与贪心 decode。
- 比较每一步生成 token。
- 包括 EOS token 在内要求完整 token 序列一致。

#### `test_batched_decode_matches_single`

- 比较 batch=4 合并 decode 与四个单请求 decode。
- 允许 EOS 之后无语义区域出现 near-tie token 差异。
- 要求有效转写区域保持一致。

### 7. `tests/unit_test/arkasr/test_mlx_model.py`

新增不依赖完整 checkpoint 的 MLX 单元测试。

#### 配置测试

- 验证 ARK 平铺文本配置和嵌套 `whisper_config` 能正确解析。
- 验证缺失字段时使用预期默认值。

#### 文本模型测试

- 验证文本 LM 前向输出 shape。
- 验证带 KV cache 的增量前向。
- 验证 untied LM head。
- 验证 `_rope_safe` 的 batch 单 token 结果与逐请求 RoPE 一致。
- 验证 `_forward_last_logits()` 只返回最后一个 token 的 logits。

#### 多模态 embedding 测试

- 验证音频 feature 正确替换 audio placeholder span。
- 验证 placeholder 数量与 feature 数量不一致时抛错。
- 验证音频 span 越界时抛错。
- 验证多请求音频 prefill 被拒绝。

#### checkpoint 布局测试

- 验证 MLX module tree 与官方 checkpoint 参数名匹配。
- 验证 Conv1d 权重转置。
- 验证 tied LM head 被移除。
- 验证 untied LM head 被保留。
- 验证量化 predicate 跳过音频塔。

#### 音频塔测试

- 验证正常输入的输出 shape。
- 验证奇数帧在 merge 前正确裁剪。
- 验证极短输入会补齐到一个 merge group。
- 验证全 1 attention mask 与不传 mask 输出一致。
- 验证变长 batch mask 与 Torch 同权重输出一致。
- 验证完整音频 prefill 能完成 placeholder 替换和文本前向。
- 验证 MLX 音频塔与 Torch 音频塔的数值 parity。
- 验证音频 prefill 的单请求限制。

---

## Commit 2：量化与 MLX 推理优化

Commit：`34971921bbcacf9888daf6010ece34a36e35cef9`

标题：`feat(arkasr): optimize MLX quantized inference`

### 1. `sglang_omni/model_runner/audio_mlx.py`

新增 `_finalize_model_load()`：

- 在模型加载后调用 `mx.eval(parameters())`，立即物化 lazy weights。
- 避免首次请求才触发大规模权重求值。
- 从完整多模态模型中提取可调用的文本 trunk。
- 将文本 trunk 保存为 `_trunk`，供共享 MLX decode 路径直接使用。
- 为 ARK-ASR 和 Qwen3-ASR 提供统一的模型加载收尾逻辑。

### 2. `sglang_omni/model_runner/mlx_model_worker.py`

将 ARK-ASR 接入 MLX worker architecture dispatch：

- 识别 `ArkasrForConditionalGeneration`。
- 延迟导入 `make_arkasr_mlx_runner_class()`。
- 让 SGLang 的 MLX model worker 能真正创建 ARK-ASR runner。
- 解决“模型代码存在，但 engine worker 没有选择该 runner”的接入缺口。

### 3. `sglang_omni/models/arkasr/mlx/model.py`

优化文本 MLP 和量化接口：

- 使用 MLX-LM 提供的 fused `swiglu()` 替代分离的 `silu(gate) * up`。
- 减少中间 tensor 和 kernel 调度开销。
- 将量化选择方法正式命名为 `quant_predicate()`，匹配 MLX-LM `quantize_model()` 接口。
- 保留 `model_quant_predicate` 别名，兼容已有本地测试和旧调用方式。
- 量化规则保持不变：只量化文本栈，不量化 `audio_encoder`。

### 4. `sglang_omni/models/arkasr/mlx/runner.py`

加入运行时 MLX Q4/Q8 量化：

- 同时导入 `load_model` 和 `quantize_model`。
- 支持 `mlx_q4`，配置为 4 bit、group size 64。
- 支持 `mlx_q8`，配置为 8 bit、group size 64。
- checkpoint 已经包含量化配置时不重复量化。
- 量化时调用模型的 `quant_predicate()`，因此音频塔保持 BF16/全精度。
- 记录量化 bits 和 group size。
- 量化完成后调用 `_finalize_model_load()`。
- 提前物化量化权重并绑定文本 trunk。

### 5. `sglang_omni/models/qwen3_asr/mlx/model.py`

同步优化 Qwen3-ASR 文本 MLP：

- 将手动 `silu(gate) * up` 替换成 MLX-LM fused `swiglu()`。
- 与 ARK-ASR 使用同一条优化过的文本 MLP 路径。

### 6. `sglang_omni/models/qwen3_asr/mlx/runner.py`

在 Qwen3-ASR 模型加载后调用 `_finalize_model_load()`：

- 提前物化模型参数。
- 暴露 headless text trunk。
- 让 ARK-ASR 引入的共享加载优化同时覆盖 Qwen3-ASR。

### 7. `tests/test_model/test_arkasr_mlx_parity.py`

扩展真实 checkpoint 的逐层 parity 测试。

#### 调整原有 BF16 阈值

- 根据真实 deep-layer massive activation 行为调整音频塔最终误差上限。
- 继续要求 FP32 精确对齐。
- 将 BF16 差异限定为可解释且不会影响最终贪心转写的范围。

#### `test_audio_tower_all_layers`

- 只加载真实音频塔和 adapter 权重，不加载文本 decoder。
- 对比 Torch 和 MLX 的 conv1、conv2。
- 对比每一层 Whisper encoder。
- 对比 tower 输出、LayerNorm 和 adapter projection。
- 输出每层 relative error、max absolute error 和统计量。
- 验证自定义音频 RoPE、mask、merge 和 adapter 实现。

#### `test_text_decoder_all_layers_fp32`

- 只加载真实文本 decoder 权重，不构造音频塔。
- 将 Torch 和 MLX 文本栈转为 FP32。
- 对比 token embedding。
- 对比每一层 decoder hidden state。
- 对比最终 norm 和 logits。
- 避免统一内存同时持有两个完整多模态模型。
- 提供逐层定位文本栈误差的独立测试入口。

### 8. `tests/unit_test/arkasr/test_mlx_model.py`

新增 runner 和量化测试。

#### `test_runner_finalizes_headless_text_trunk`

- 验证模型加载收尾会调用参数物化。
- 验证 `_trunk` 指向可调用的文本模型。

#### `test_runner_chains_native_single_request_decode`

- 验证 ARK runner 能完成 native prefill、cache 保存和后续 decode。
- 覆盖共享 MLX runner 与 ARK model mixin 的组合行为。

#### `test_native_mlx_quantizes_text_only`

- 对小模型实际调用 MLX 量化。
- 验证文本 Linear 被替换为量化模块。
- 验证音频 encoder Linear 保持未量化。
- 验证模型的量化 predicate 与 runner 的量化行为一致。

#### 既有量化 predicate 测试调整

- 同时验证新名称 `quant_predicate()` 和兼容别名。
- 固定“文本栈量化、音频塔不量化”的接口契约。

---

## Commit 3：补齐正式 Serving、长音频与实时流

Commit：`1c757d01c32915477e97fece41c50e056a381760`

标题：`feat(arkasr): complete MLX serving integration`

### 1. `sglang_omni/model_runner/audio_mlx.py`

为音频 MLX runner 补齐 request-level logit bias：

#### `_remember_audio_logit_bias`

- 从 scheduler request 的 `sampling_params.logit_bias` 读取静态 bias。
- 将 token id 转成整数，将 bias 转成浮点数。
- 按 request id 保存 bias。
- 没有 bias 时保存空字典。

#### `_apply_audio_logit_bias`

- 在 greedy argmax 前将 bias 加到对应 token logits。
- 支持 batch 中每一行使用自己的 request bias。
- 忽略超出词表范围的 token id。
- 保留无 bias 请求的原始 logits。

#### `remove_request`

- request 结束时删除对应 bias。
- 再调用父类清理 KV cache 和 token 状态。
- 防止 request id 生命周期结束后遗留 bias。

#### prefill/decode 接入

- prefill 开始时记录 request bias。
- prefill 最后一个位置的 logits 在 argmax 前应用 bias。
- 第一次 decode 在 argmax 前应用 bias。
- 后续 batched decode 在 argmax 前按行应用 bias。
- 使 ARK request builder 对特殊 token 的 suppression 在 MLX greedy 路径真正生效。

### 2. `sglang_omni/models/arkasr/config.py`

声明 ARK 的长音频和实时转写能力：

- `allow_audio_chunking=True`，允许共享 transcription endpoint 自动切分长音频。
- `max_native_clip_s=30.0`，声明 ARK 单次音频 encoder 的原生上限。
- 默认 `audio_chunking.max_audio_clip_s=30.0`。
- 配置 `ArkASRStreamingStrategy`。
- 开启 realtime server VAD。
- realtime 单 segment 最大长度设为 30 秒。
- 长音频仍由上层拆成多个单音频请求，不改变 MLX 单请求音频约束。

### 3. `sglang_omni/models/arkasr/encoder_service.py`

新增 `lookup_cached_embedding()`：

- 根据 cache namespace 和音频 fingerprint 构造 cache key。
- 在 request builder 做 mel feature extraction 前查询 embedding cache。
- 校验缓存 embedding 的 token 数、shape 和 dtype。
- 命中有效缓存时增加 hit 计数并返回 embedding。
- 无缓存时返回 `None`。
- 缓存无效时记录 warning。
- 仅在当前对象仍是同一缓存值时删除，避免并发覆盖错误。
- 让缓存命中请求跳过 mel 提取和音频 encoder。

### 4. `sglang_omni/models/arkasr/engine_builder.py`

补齐 ARK MLX 与 SGLang engine builder 的正式集成。

#### MLX generation profile

- 检查 MLX 只能运行在 Apple Metal/MPS 平台。
- 关闭 CUDA graph。
- 关闭 overlap schedule。
- 关闭 radix cache。
- 关闭 Torch compile。
- 将 `max_prefill_tokens` 设置为完整 context length。
- 将 `chunked_prefill_size` 设置为 `-1`，关闭 chunked prefill。
- 保留 dtype、内存比例和最大运行请求数。
- 原因是音频 embedding 在 native MLX prefill 内构造，不能由 token-only radix/chunked prefill 重放。

#### `make_model_runner`

- MLX 模式下创建 `MlxSchedulerModelRunner`。
- 非 MLX 模式保持父类 Torch/CUDA runner。
- 解决 ARK MLX runner 没有真正进入 scheduler 的问题。

#### `adjust_overrides`

- pipeline typed defaults 合并后再次强制关闭 `enable_torch_compile`。
- 防止用户配置或默认配置把 MLX 不支持的 Torch compile 重新打开。

#### `customize_server_args`

- 从最终解析后的 server args 同步真实 context length。
- 后续 request builder 使用该值做 prompt/output budget 检查。

#### `validate_before_infrastructure`

- MLX 模式下拒绝 `mlx_enable_sampling=True`。
- ARK MLX 当前仅支持自己的 greedy decode 路径。
- 在创建重量级资源前提前报错。

#### `setup_model_resources`

- MLX 模式直接返回。
- 不初始化 Torch/CUDA 音频 encoder service。
- 不创建 encoder CUDA graph。
- 避免 MLX server 同时占用一套无用 Torch 音频 encoder 资源。
- CUDA 模式继续保留原有 pre-LM encoder 和 CUDA graph 行为。

#### `make_adapters`

- 向 request builder 传入最终 context length。
- 传入 scheduler queue 状态回调。
- MLX 模式开启 `greedy_only`。
- 继续传入 tokenizer、feature extractor、merge factor、audio token id 和 encoder service。

#### encode admission

- 新增 `should_wait_for_encode()`。
- 新增 `post_scheduler_setup()`。
- 将 scheduler 的 `request_build_queue_fits_workers` 信号接入 request builder。
- request-build worker 空闲时同步执行音频 encode。
- worker 忙时使用 DeferredAdmission 异步提交。

### 5. `sglang_omni/models/arkasr/request_builders.py`

增强 ARK 请求构造的正确性、缓存和错误处理。

#### prompt

- 新增 `_build_instruction()`。
- 请求提供非空 `prompt` 时，将其作为 ARK user instruction。
- 未提供或只提供空白 prompt 时继续使用官方默认指令 `Please transcribe this audio.`。
- `language` 保持原作者逻辑，只记录为请求/响应元数据，不加入模型 prompt。

#### tokenizer 词表范围

- 将 `tokenizer.vocab_size` 改为 `len(tokenizer)`。
- `vocab_size` 往往不包含 added tokens，而 ARK audio token 和其他 marker 位于 added vocabulary。
- 确保 SGLang `Req` 的 token id 校验覆盖完整有效词表。

#### greedy 限制

- 新增 `greedy_only` 参数。
- MLX 请求使用非零 temperature 时提前抛出明确错误。
- CUDA/Torch 路径不受该限制。

#### 音频解码错误

- 捕获 `AudioDecodeError`。
- 转换为面向 API 用户的 `ValueError`。
- 避免无效音频表现为内部 server error。

#### 原生音频窗口

- 根据 feature extractor 的 `nb_max_frames * hop_length` 计算最大采样点。
- 在 feature extraction 前拒绝超过原生窗口的单个 chunk。
- 错误信息包含支持的秒数。
- 服务层长音频会先自动切分，因此正常长音频不会直接进入该错误。

#### pre-LM cache 提前查询

- 在 mel extraction 前根据 waveform fingerprint 查询 embedding cache。
- 使用 hop length 估算 mel frame 和 audio token 数。
- 缓存命中时不再调用 feature extractor。
- 将缓存 embedding 直接附加到 multimodal item。
- 降低重复音频请求的 CPU mel 和 encoder 成本。

#### feature extraction

- 将 `truncation=True` 改为 `truncation=False`。
- 不再静默截断超长音频。
- 超长输入由前置窗口检查显式拒绝。
- 保持 `padding="longest"`，短音频不支付完整 30 秒 FFT 成本。

#### context budget

- 新增 `context_length` 参数。
- 校验 `prompt/audio tokens + max_new_tokens <= context_length - 1`。
- 超限时提示减少输出 token 或切分音频。
- 防止请求进入 engine 后才发生 context overflow。

#### encode admission

- 没有 encoder service 时直接返回请求。
- cache 命中时附加 embedding 后直接返回。
- scheduler build queue 有容量时同步 `encode_item()`。
- queue 忙时返回 `DeferredAdmission`，通过 `submit_item()` 异步编码。
- 避免所有请求无条件 deferred，减少低并发额外调度延迟。

### 6. `sglang_omni/models/arkasr/streaming.py`

新增 ARK realtime transcription strategy。

#### `ArkASRStreamingState`

- 保存模型名。
- 保存 language 元数据。
- 保存当前 segment 的最新 transcript。

#### `create_state`

- 为每个 realtime segment 创建独立状态。

#### `_state`

- 校验 realtime session 传入的状态类型。
- 防止不同模型的 streaming state 混用。

#### `build_decode_request`

- 将 realtime PCM segment 包装为 WAV transcription request。
- 使用 ARK 模型名和 language 元数据。
- 固定 temperature 为 0。
- 每次 refresh 都提交当前 segment 的完整音频。
- 不启用 Qwen3-ASR 的 transcript prefix 和 rollback token 机制。

#### `update_hypothesis`

- 更新 language 元数据。
- 用最新生成文本替换当前 interim hypothesis。
- final 时返回同一个完整文本。

### 7. `tests/test_model/test_arkasr_mlx_server.py`

新增 opt-in 的完整 MLX server 集成测试：

- 使用 `ARKASR_MLX_SERVER_CHECKPOINT` 指定本地 checkpoint。
- 使用 `ARKASR_MLX_SERVER_AUDIO` 覆盖默认测试音频。
- 未配置 checkpoint 时跳过。
- 查找可用本地端口。
- 通过真实 CLI 启动 `sglang_omni.cli serve`。
- 设置 `SGLANG_USE_MLX=1`。
- 等待 server health ready。
- 向 `/v1/audio/transcriptions` 上传真实 WAV。
- 使用 greedy temperature=0。
- 要求 HTTP 200。
- 要求 JSON 中存在非空文本。
- 测试结束后始终关闭 server。

### 8. `tests/unit_test/arkasr/test_mlx_model.py`

新增 `test_audio_runner_applies_static_logit_bias_before_greedy`：

- 构造多行 logits。
- 为不同 request 配置不同 token bias。
- 验证 bias 在 argmax 前生效。
- 验证超出词表的 token id 被忽略。
- 验证无 bias 请求保持原始结果。
- 固定特殊 token suppression 在 MLX greedy 路径中的行为。

### 9. `tests/unit_test/arkasr/test_pipeline.py`

大幅扩展 ARK pipeline 和 request builder 测试。

#### pipeline capability

- 验证长音频切分已开启。
- 验证原生单 chunk 上限为 30 秒。
- 验证默认切分长度为 30 秒。
- 验证 realtime server VAD 已开启。
- 验证 realtime segment 上限为 30 秒。

#### engine builder

- 新增统一 builder fixture helper。
- 验证 MLX generation defaults 关闭 CUDA graph、radix cache、chunked prefill、overlap schedule 和 Torch compile。
- 验证 MLX override 合并后 Torch compile 仍保持关闭。
- 验证最终 server context length 能同步到 builder。
- 验证 scheduler queue 状态接入 encode admission。
- 验证 MLX 模式选择 `MlxSchedulerModelRunner`。
- 验证 `mlx_enable_sampling=True` 在启动前被拒绝。
- 验证 MLX 模式不初始化 Torch encoder service。
- 验证 factory 将 context length 传给 request builder。

#### request builder

- 验证 MLX greedy-only 模式拒绝非零 temperature。
- 验证有效词表大小使用 `len(tokenizer)`，而不是不包含 added tokens 的 `vocab_size`。
- 验证调用方 `prompt` 真正进入 ARK prompt。
- 验证 `language` 只记录为元数据，不进入模型 prompt。
- 验证超过 encoder 原生窗口的音频在 feature extraction 前被拒绝。
- 验证 prompt/audio/output 超过 context budget 时被拒绝。
- 验证 pre-LM cache 命中会跳过 feature extraction。
- 验证 cache embedding 正确附加到 multimodal item。
- 验证 build queue 空闲时同步 encode，不创建 deferred submission。

### 10. `tests/unit_test/arkasr/test_streaming.py`

新增 ARK realtime strategy 单元测试：

#### `test_arkasr_streaming_request_records_language`

- 验证 state 保存 language 元数据。
- 验证构造出的 transcription request 包含 language。
- 验证 task 为 `transcribe`。
- 验证音频字节被放入请求。

#### `test_arkasr_streaming_replaces_interim_hypothesis`

- 验证新 hypothesis 替换旧 transcript。
- 验证 update 返回最新可见文本。
- 验证后端返回的 language 能更新 state。

---

## Commit 4：收窄 ARK/MLX 动态类型

Commit：`c1e5b5bf7c42a4d08d99cbe6aa55eb7f87690d61`

标题：`refactor(arkasr): narrow dynamic types`

### 1. `sglang_omni/models/arkasr/encoder_service.py`

- 将 pre-LM encoder 的 item 泛型从 `Any` 改为 `MultimodalDataItem`。
- 将 enqueue、submit、同步 encode、embedding 附加、cache key 和 batch 回调统一改为具体 item 类型。
- 将 `QueueEntry[Any]` 收窄为 `QueueEntry[MultimodalDataItem]`。
- 将待校验 embedding 从 `Any` 改为 `object`，继续通过运行时类型检查确认它是 Torch tensor。
- 保留 Torch 模型对象的动态类型，因为它通过运行时属性提供 config、audio encoder 和 `get_audio_feature()`。

### 2. `sglang_omni/models/arkasr/engine_builder.py`

- tokenizer 字段改为 `PreTrainedTokenizerBase | None`。
- feature extractor 字段改为 `WhisperFeatureExtractor | None`。
- 音频 encoder service 字段改为 `AudioEncoderService | None`。
- 为 `make_adapters()` 写出具体 request adapter 和 result adapter 返回类型。
- 保留 server args、model worker 和 scheduler 等 SGLang 动态框架对象的 `Any`。

### 3. `sglang_omni/models/arkasr/mlx/config.py`

- 将配置字典中的 value 类型从 `Any` 改为 `object`。
- 收窄 audio、text 和顶层 model config 的 `from_dict()` 参数。
- 将 `rope_scaling` 和 audio/text 子配置字典改为 `dict[str, object]`。
- 配置字段过滤和运行行为保持不变。

### 4. `sglang_omni/models/arkasr/mlx/model.py`

- 导入并使用具体的 MLX-LM `KVCache` 类型。
- 将 attention、decoder layer、text model 和完整模型中的 cache 从 `Any` 改为 `KVCache`。
- 将 `make_cache()` 返回值改为 `List[KVCache]`。
- 移除方法内部的局部 `KVCache` import。

### 5. `sglang_omni/models/arkasr/request_builders.py`

- tokenizer 参数改为 `PreTrainedTokenizerBase`。
- feature extractor 参数改为 `WhisperFeatureExtractor | None`。
- 新增 `AudioEncoderService` Protocol，描述 cache 查询、embedding 附加、同步 encode 和异步 submit 接口。
- result adapter 明确接收 `ArkASRRequestData` 并返回 `StagePayload`。
- instruction 参数字典改为 `dict[str, object]`。
- stream output builder 使用共享 token streaming Protocol。

### 6. `sglang_omni/scheduling/token_text_streaming.py`

- 新增 `TokenStreamRequest` Protocol，声明 chunk 状态和 `finished()`。
- 新增 `TokenStreamRequestData` Protocol，声明 request 和 `StagePayload`。
- 新增 `TokenStreamRequestOutput` Protocol，声明流式 token data。
- 将 stream builder 的请求和输出参数从 `Any` 改为最小 Protocol 接口。
- 将新 Protocol 导出，供 ARK-ASR 等具体 builder 复用。

---

## Commit 5：精简 MLX 测试覆盖

Commit：`03655b459fbefc55c9d0a54a6d0d1360c378a40b`

标题：`test(arkasr): streamline MLX coverage`

### 1. `sglang_omni/models/arkasr/mlx/model.py`

- 在 `_Gelu` 前增加注释，说明该无参数 module 用于占据 `adapting.1`。
- 明确第二个 Linear 必须保持 `adapting.2`，才能直接匹配官方 checkpoint 权重。
- 精简 `ArkAudioTower`、`TextAttention` 和 `sanitize()` 的长 docstring。
- 没有修改模型计算、权重加载、RoPE 或量化行为。

### 2. `tests/test_model/test_arkasr_mlx_parity.py`

- 删除整个 648 行 opt-in parity 文件。
- 删除真实 checkpoint 的 Torch/MPS 与 MLX 音频、文本和 logits 对比。
- 删除 BF16 逐层误差表、greedy token 一致性和 batch decode 实验。
- 删除音频塔和文本 decoder 的逐层 hook 测试。
- 原因是这些测试依赖本地 checkpoint、Apple MPS 和 MLX，属于开发阶段实验验证。
- Qwen3-ASR 的常规测试结构中也没有对应的 checkpoint parity 文件。
- 实验结论继续保存在验证记录和 benchmark 文档中。

### 3. `tests/unit_test/arkasr/test_mlx_model.py`

- 删除与 parity 文件重复的 `test_native_mlx_audio_parity_with_torch`。
- 删除该测试独占的 NumPy import。
- 保留 24 个 MLX 单元测试。
- 继续覆盖配置、文本前向、KV cache、RoPE workaround、音频 placeholder、module tree、sanitize、q4 量化范围和完整 prefill。
- 精简后 MLX 单元测试结果为 `24 passed`。
- 保留的真实 MLX server 集成测试结果为 `1 passed`。

---

## 跨提交文件演进

### `sglang_omni/models/arkasr/mlx/model.py`

- Commit 1：从零实现完整 ARK-ASR MLX 模型。
- Commit 2：使用 fused SwiGLU，并将量化 predicate 接入 MLX-LM 正式接口。
- Commit 3：没有继续修改该文件，说明模型计算主体在前两次提交后已经稳定。
- Commit 4：将 cache 类型从 `Any` 收窄为 `KVCache`。
- Commit 5：补充 `_Gelu` checkpoint index 注释并精简 docstring，不改变计算行为。

### `sglang_omni/models/arkasr/mlx/runner.py`

- Commit 1：实现 checkpoint 加载和动态 runner 组合。
- Commit 2：加入 Q4/Q8 文本栈量化和模型加载收尾。
- Commit 3：通过 engine builder 和 MLX worker 完成 server 侧正式接入。

### `sglang_omni/model_runner/audio_mlx.py`

- Commit 2：增加模型参数物化和文本 trunk 提取。
- Commit 3：增加 request-level logit bias，补齐特殊 token suppression。

### `tests/test_model/test_arkasr_mlx_parity.py`

- Commit 1：建立端到端 Torch/MLX parity 基线。
- Commit 2：增加音频塔和文本 decoder 的完整逐层对比。
- Commit 3：不再修改 parity 测试，转而新增正式 server 集成测试。
- Commit 5：删除实验型 parity 文件，最终保留常规 MLX 单元测试和 server 集成测试。

### `tests/unit_test/arkasr/test_mlx_model.py`

- Commit 1：覆盖模型结构、音频塔、文本栈、cache、sanitize 和基础量化范围。
- Commit 2：覆盖 runner decode 链和实际文本栈量化。
- Commit 3：覆盖 MLX greedy 前的 logit bias。
- Commit 5：删除重复的 Torch/MLX 小模型 parity 用例，保留 24 个常规 MLX 单元测试。

---

## 五次提交最终提供的能力

- ARK-ASR-3B 可以在 Apple Silicon 上使用原生 MLX 推理。
- 音频塔和 Qwen2 文本 decoder 都有 MLX 实现。
- 音频塔使用与 Torch 对齐的自定义 RoPE。
- 支持 MLX BF16、`mlx_q4` 和 `mlx_q8`。
- Q4/Q8 只量化文本栈，音频塔保持未量化。
- prefill 只投影最后一个 token 的 logits。
- 支持 MLX KV cache 和批量 decode。
- 修复 MLX batched single-token RoPE 问题。
- ARK architecture 已接入统一 MLX worker。
- ARK engine builder 会选择 MLX scheduler runner。
- MLX 模式不会初始化无用的 Torch/CUDA 音频 encoder。
- MLX 模式关闭不兼容的 radix cache 和 chunked prefill。
- 支持特殊 token logit bias。
- 支持 prompt 注入。
- `language` 保持原作者的元数据语义，不注入模型 prompt。
- 支持 pre-LM embedding cache 提前命中。
- 支持同步或 deferred 音频 encode admission。
- 支持非流式长音频自动切分。
- 支持 `/v1/realtime`、server VAD、interim 和 final hypothesis。
- 包含常规 MLX 单元测试和完整 server 集成测试。
- Torch/MLX 逐层 parity 作为开发阶段实验记录保留，不再进入常规测试目录。
