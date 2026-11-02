# Qwen3.8-Flash-Next (qwen4_exp) architecture contract

Qwen3.8-Flash-Next の text backbone が「正確に何を計算するか」を固定する。
この文書はまだ executor を持つ contract ではない。runtime 実装が追加されるまで、
config / tensor / state / 演算の正本としてここを参照する。

参照実装は Hugging Face Transformers の `qwen4_exp` である。
固定した参照 revision と検証記録は
[research/20261024/qwen4exp/source_audit.md](../research/20261024/qwen4exp/source_audit.md)
を参照。

namespace は `qwen4exp` とし、既存 `qwen35` を rename しない。
Qwen3.8-Flash-Next を `Qwen35TextConfig` の差分として押し込まない。

---

## 1. config

`config.json` は `architectures: ["Qwen4ExpForConditionalGeneration"]`,
`model_type: "qwen4_exp"`、text 側は `text_config` に入る (`model_type: "qwen4_exp_text"`)。

### 1.1 text_config

| field | value |
| --- | --- |
| `vocab_size` | 248320 |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| `rope_theta` | 10000000 |
| `rms_norm_eps` | 1e-6 |
| `hidden_act` | `silu` |
| `output_gate_type` | `sigmoid` |
| `attention_bias` | false |
| `tie_word_embeddings` | false |
| `max_position_embeddings` | 262144 |
| `full_attention_interval` | 4 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 48 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| `num_experts` | 512 |
| `num_experts_per_tok` | 10 |
| `moe_intermediate_size` | 640 |
| `shared_expert_intermediate_size` | 640 |
| `router_aux_loss_coef` | 0.001 |
| `norm_topk_prob` | true (config 未指定時の default) |
| `hc_count` | 4 |
| `hc_lowrank` | 320 |
| `ngram_size` | 3 |
| `heads_per_ngram` | 8 |
| `ngram_vocab_size_base` | 20000000 |
| `make_ngram_vocab_size_divisible_by` | 128 |
| `seed` | 1234（`config.json` に**存在しない**既定値） |
| `split_ngram_parts` | 128 |
| `ple_layer_ids` | `[2]` (1-indexed)。`ple_slot` は配列内 0-indexed |
| `ple_embed_dim` | 2560 |
| `ple_conv_kernel_size` | 4 |
| `indexer_n_heads` | 4 |
| `indexer_kv_heads` | 1 |
| `indexer_head_dim` | 128 |
| `indexer_budget` | 2048 |
| `indexer_compress_ratio` | 4 |
| `mamba_ssm_dtype` | `float32`（参照実装の値。PhaseShift の実装は GDN を FP16 とする、5.0 参照） |

`indexer_*` が 5 つとも設定されている時だけ QSA が有効になる。
`indexer_kv_heads` は 1 でなければならない。
`indexer_budget` は `indexer_compress_ratio` で割り切れなければならない
(`block_topk = 2048 / 4 = 512`)。
`partial_rotary_factor` から `rotary_dim = int(head_dim * factor) = 64` が決まる。
`rotary_dim <= indexer_head_dim` が要求される (64 <= 128)。

### 1.2 MTP config

| field | value |
| --- | --- |
| `mtp_num_hidden_layers` | 1 |
| `mtp.hybrid` | true |
| `mtp.layer_types` | `["full_attention"]` |
| `mtp.rope_theta` | 10000000 |
| `mtp_use_hidden_state_from_layer` | null |
| `mtp_use_dedicated_embeddings` | false |

### 1.3 vision_config

vision tower は text support とは別 Gate であり、この contract の対象外とする。
`vision_config`: depth 27 / hidden 1152 / heads 16 / patch 16 /
spatial_merge 2 / temporal_patch 2 / out_hidden_size 2560 / num_position_embeddings 2304。

`layer_types` は checkpoint では `full_attention` と `linear_attention` の列で保存される。
runtime では `full_attention` を `indexed_attention` として扱う (QSA 層)。

---

## 2. layer pattern

`layer_types` を `(i + 1) % 4` で決めると:

- index `3, 7, 11, ..., 47` (12 層): QSA (`indexed_attention`)
- それ以外 (36 層): GDN (`linear_attention`)

各 decoder layer は `attention/GDN -> MoE` の 2 サブ層を持ち、
それぞれに Gated Residual (HyperConnection) が付く。
PLE は 1-indexed layer 2 (0-indexed layer 1) のみに入る。

1 層の構造 (`Qwen4ExpTextDecoderLayer.forward`):

```text
x4                         (hyper stream, feature = hc_count * hidden_size = 10240)
if ple: x4 = x4 + ple(x4, input_ids)
mixed, hyper, inj = attn_hyper_connection(x4)
attn_out = attention_or_gdn(mixed)
x4 = hyper + (attn_out[..., None, :] * inj[..., None]).flatten(-2)
mixed, hyper, inj = mlp_hyper_connection(x4)
moe_out = moe(mixed)
x4 = hyper + (moe_out[..., None, :] * inj[..., None]).flatten(-2)
```

model 全体 (`Qwen4ExpTextModel.forward`):

```text
emb = embed_tokens(input_ids)                      # [B, T, 2560]
x4  = emb.repeat(1, 1, hc_count)                   # [B, T, 10240]
for layer in layers:
    x4 = layer(x4, ...)
h = hyper_connection_mixer(x4)                     # [B, T, 2560] (use_combine=False)
logits = lm_head(h)                                # final norm は存在しない
```

`model.language_model.norm.weight` は checkpoint に存在しない。
最終 collapse は `hyper_connection_mixer` が行う。

---

## 3. RoPE / position

`Qwen4ExpTextRotaryEmbedding` は mRoPE である。

- `inv_freq` の次元は `rotary_dim / 2 = 32`。
- `mrope_section = [11, 11, 10]`、`mrope_interleaved = true`。
- `forward` 後半の `recomposition_frequencies` は T/H/W の 3 grid を合成し、
  最後に `cat((freqs, freqs), dim=-1)` で長さ 64 に倍化する。

text-only 入力では 3 grid の position が同一になるため、合成は no-op となり、
実質は `rotary_dim = 64` の標準 partial RoPE (theta = 1e7) である。
multimodal 時のみ grid 合成が意味を持つ。
`apply_rotary_pos_emb` は head_dim 前半 `rotary_dim = cos.shape[-1] = 64` のみ回転し、
残りを非回転で連結する。QSA indexer の q/key にも同じ関数を `rotary_dim = 64` で適用する。

`position_ids` は 2D (`[B, T]`) を渡すと内部で `(4, B, T)` (text pos + T/H/W) に
拡張される。rotary はこの 3 grid を入力に取る。

---

## 4. norm

### 4.1 Qwen4ExpTextRMSNorm

```text
y = (1 + weight) * (x_f32 * rsqrt(mean(x_f32^2) + eps))
```

weight は 0 初期化であり、`(1 + weight)` の 1-centered 表現である。
PhaseShift の `RmsNormWeightMode::ONE_PLUS` に対応する。

`group_size` が指定された場合、最終次元を `(-1, group_size)` に reshape し、
`group_size` 単位で RMS を取る (stream ごとの norm)。
QSA の `q_norm` / `k_norm` は `group_size` なし (`head_dim = 256` ごと)。
GDN の `linear_attn.norm` は `RMSNormGated` を使う (下記)。

### 4.2 Qwen4ExpTextRMSNormGated

```text
y = weight * (x_f32 * rsqrt(mean(x_f32^2) + eps)).to(dtype)
y = y * act(gate_f32)          # act = output_gate_type or hidden_act
```

Qwen3.8-Flash-Next では `output_gate_type = "sigmoid"` のため
GDN 出力ゲートは **sigmoid** である (silu ではない)。
weight は 1 初期化である。

---

## 5. GDN (linear_attention)

`Qwen4ExpTextGatedDeltaNet`。36 層に存在する。

### 5.0 精度方針

Qwen4Exp の GDN は **FP16 ベース**で実装し、lossy な計算を明示的に許容する。
recurrent state も FP16 を基本とする。config の `mamba_ssm_dtype: float32`
（上記 text_config）は参照実装の値であり、PhaseShift の実装方針との差分である。

差分の理由と根拠:

- Qwen3.8-27B の GDN は recurrent state が FP16 で、既存の最適化済み HIP
  kernel と state 管理機構が FP16 前提で実装・検証されている。
- Qwen4Exp を FP32 で実装すると GDN kernel / state pool の二重実装が必要に
  なり、Qwen35 と Qwen4Exp で同一の最適化済み kernel を共有できない。
- 本番の lossy 許容は品質検証（`docs/developer` の quality contract）で評価する。

実装上の制約:

- Qwen35 と Qwen4Exp の差は head 数 / head_dim / conv kernel 等の geometry で
  吸収する。geometry が既存 kernel の対応範囲を超える場合は、共通 API の内部に
  静的 specialization を追加する。
- FP32 / FP16 切り替えのためだけの抽象化や runtime branch を導入しない。
- 既存 Qwen35 の GDN 精度・計算経路は変更しない。

### 5.1 geometry

| field | value |
| --- | --- |
| `num_k_heads` | 16 |
| `num_v_heads` | 48 |
| `head_k_dim` | 128 |
| `head_v_dim` | 128 |
| `key_dim` | 2048 |
| `value_dim` | 6144 |
| `conv_dim` | `key_dim * 2 + value_dim = 10240` |
| `conv_kernel_size` | 4 |

### 5.2 projection

```text
mixed_qkv = in_proj_qkv(x)         # [.., 10240]
mixed_qkv = depthwise_causal_conv1d(mixed_qkv, conv1d.weight, silu)
q, k, v = split(mixed_qkv, [2048, 2048, 6144])
z = in_proj_z(x)                   # [.., 6144]
b = in_proj_b(x)                   # [.., 48]
a = in_proj_a(x)                   # [.., 48]
```

conv1d の activation は `hidden_act` (silu) である (`activation=self.activation`)。
`conv1d.weight` の shape は `[10240, 1, 4]` (depthwise) である。

### 5.3 recurrence

```text
q = q.reshape(.., 16, 128); k = k.reshape(.., 16, 128); v = v.reshape(.., 48, 128)
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)          # [.., 48], <= 0
q = q.repeat_interleave(3, dim=2)                # 16 -> 48
k = k.repeat_interleave(3, dim=2)
```

gated delta rule (`use_qk_l2norm_in_kernel=True`):

```text
q = l2norm(q); k = l2norm(k)
q = q * head_k_dim^-0.5
state: S [B, 48, 128, 128] fp16（Qwen35 と同じ。lossy 許容、5.0 参照）
per step:
  S = S * exp(g)
  kv_mem = sum_k(S * k)          # [B, 48, 128]
  delta  = (v - kv_mem) * beta
  S = S + k ⊗ delta
  out = sum_k(S * q)
```

prefill は chunk_size=64 の chunk 版、decode (seq_len=1) は recurrent 版を使う。
どちらも数学的には同じ gated delta rule である。

### 5.4 output

```text
out = RMSNormGated(out, z, activation=sigmoid)
out = out_proj(out)              # [2560, 6144] @ [.., 6144]
```

state: conv state (`[B, 10240, 4]`) と recurrent state (`[B, 48, 128, 128]` fp16、5.0 参照)。

---

## 6. QSA (indexed_attention)

`Qwen4ExpTextAttention` + `Qwen4ExpTextQSAIndexer`。12 層に存在する。

### 6.1 main attention geometry

| field | value |
| --- | --- |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `rotary_dim` | 64 |
| `scaling` | `256^-0.5` |
| `q_proj` out | `24 * 256 * 2 = 12288` (query + gate) |
| `k_proj` out | `2 * 256 = 512` |
| `v_proj` out | `2 * 256 = 512` |
| `o_proj` in | `24 * 256 = 6144` |

```text
q_and_gate = q_proj(x)                                  # [.., 12288]
query, gate = chunk(q_and_gate.view(.., -1, 512), 2, dim=-1)
gate = gate.reshape(.., -1)                             # [.., 6144]
query = q_norm(query.view(.., 24, 256))
key   = k_norm(k_proj(x).view(.., 2, 256))
value = v_proj(x).view(.., 2, 256)
query, key = rope(query, key, cos, sin)                 # rotary_dim 64
key, value = kv_cache.update(key, value)
attn = softmax(query @ key^T * scale + mask) @ value    # GQA 24/2 = 12
attn = attn.reshape(.., -1)
attn = attn * sigmoid(gate)
out = o_proj(attn)
```

### 6.2 QSA indexer

```text
qk = index_qk_proj(x)                                   # [.., 640]
q, token_k = split(qk, [4 * 128, 1 * 128])
q = q.view(.., 4, 128)
raw_keys = token_k.view(.., 1, 128).squeeze(1)          # [.., 128]
q = q_layernorm(q)
q = rope(q, cos_current, sin_current)                   # rotary_dim 64, head 軸
raw_keys = indexer_key_cache.update(raw_keys)           # 全系列の raw key
```

block selection (`R = indexer_compress_ratio = 4`, `block_topk = 512`):

```text
for each query position i:
    visible = causal visible token indices (0..i)
    num_complete_blocks = len(visible) // R
    if num_complete_blocks > 0:
        blocks = visible[: num_complete_blocks * R].view(num_complete_blocks, R)
        pooled = mean(raw_keys[blocks], dim=1_over_R)         # fp32 mean
        pooled = k_layernorm(pooled)
        block_keys = rope(pooled, cos[block_start], sin[block_start])
        scores = sum_over_4_heads( relu(q_i @ block_keys^T) ) / sqrt(128)
        selected_blocks = topk(scores, min(block_topk, num_complete_blocks))
        selected = blocks[selected_blocks].flatten()
    tail = visible[num_complete_blocks * R :]                 # 未完成 block
    selected = concat(selected, tail)
```

block は系列先頭から `R = 4` トークンごとに固定整列する
(`visible` が prefix であるため)。micro-block 単位の選択である。

indexer は選択結果を boolean / float の additive mask に変換し、
通常の causal mask に AND (bool) または加算 (float) して main attention に渡す。
main attention / KV cache 自体は通常の causal attention である。

state: QSA 用 KV cache (kv_heads 2, head_dim 256) に加えて、
indexer key cache (1 head, head_dim 128) が必要である。

備考: `indexer_budget = 2048` はトークン数予算、`block_topk = 512` は block 数予算である。

---

## 7. MoE

`Qwen4ExpTextSparseMoeBlock`。全 48 層に存在する。

### 7.1 router

`Qwen4ExpTextTopKRouter`, weight shape `[512, 2560]`。

```text
logits = x @ weight^T                                  # [tokens, 512]
probs  = softmax(logits, dtype=float32)
top_w, top_i = topk(probs, 10)
if norm_topk_prob: top_w = top_w / sum(top_w)          # 10 個で再正規化
top_w = top_w.to(logits.dtype)
```

softmax は fp32、top-k 後の再正規化は `norm_topk_prob = true` で有効である。

### 7.2 routed experts

`Qwen4ExpTextExperts` は fused 3D tensor を保持する。

| tensor | shape |
| --- | --- |
| `mlp.experts.gate_up_proj` | `[512, 1280, 2560]` = `[E, 2*I, H]` |
| `mlp.experts.down_proj` | `[512, 2560, 640]` = `[E, H, I]` |

`I = moe_intermediate_size = 640`。

```text
for each (token, slot) with expert e = top_i[token, slot]:
    gate, up = chunk(x_token @ gate_up_proj[e]^T, 2)   # each [I]
    hidden = silu(gate) * up
    out_e = hidden @ down_proj[e]^T                    # [H]
    acc[token] += top_w[token, slot] * out_e
```

activation は `hidden_act = silu` である。
`gate_up_proj[e]` は `[1280, 2560]` を `x` に右から掛ける形 (`nn.functional.linear`) である。

### 7.3 shared expert

`shared_expert` は通常の MLP (intermediate 640) である。

```text
shared = down_proj(silu(gate_proj(x)) * up_proj(x))    # [.., 2560]
shared = sigmoid(shared_expert_gate(x)) * shared       # scalar gate
```

`mlp.shared_expert_gate.weight` の shape は `[1, 2560]` である。

### 7.4 合成

```text
moe_out = routed_acc + shared
```

MoE 層に dense FFN は存在しない。routed + shared のみである。

### 7.5 tensor partition（EP 分散の表現）

routed expert の fused tensor は、将来の EP 分散のために
`TensorPartitionDesc` の axis=0 single-range partition で表現できる
（例: `gate_up_proj` `[512,1280,2560]` を EP=2 で
`ranges = {{0,256}}` / `{{256,256}}`、local `[256,1280,2560]`）。
descriptor と N-D canonical slicing は generic weights layer にあり、
Qwen4Exp 専用ではない。詳細は
[tensor_partition.md](tensor_partition.md) を参照。
現時点の contract は表現のみであり、EP / multi-GPU execution は未実装である。

---

## 8. Gated Residual / HyperConnection

`Qwen4ExpTextGatedResidual`。`hc_count = 4`, `hc_lowrank = 320`,
`hc_hidden_size = hc_count * hidden_size = 10240`。

### 8.1 read (use_combine=True)

per-layer の `attn_hyper_connection` / `mlp_hyper_connection`。

```text
n   = hc_norm(x4)                                   # grouped RMSNorm (group = 2560), (1+w)
d   = input_mix_weight_down(n)                      # [.., 320]
u   = input_mix_weight_up(silu(d / hc_count))       # [.., 10240]
w   = sigmoid(u).unflatten(-1, (4, 2560))           # read gate
mixed = mean_over_4( w * n.unflatten(-1, (4, 2560)) )   # [.., 2560]
inj = 2 * sigmoid(block_inject_weight(n) / hc_count)    # [.., 4]
```

`hc_norm` の weight shape は `[10240]`、`group_size = 2560` である。

### 8.2 write

```text
x4 = x4 + (sub_out[..., None, :] * inj[..., None]).flatten(-2)   # [.., 10240]
```

write gate `inj` は branch (stream) ごとのスカラーである。

### 8.3 final collapse (use_combine=False)

`hyper_connection_mixer`。`block_inject_weight` を持たず、write はない。

```text
mixed = mean_over_4( sigmoid(up(silu(down(hc_norm(x4)) / 4))) * hc_norm(x4) )
```

read gate は element-wise、write は branch ごとのスカラー、
`mean` は stream 方向の平均である。

---

## 9. PLE (Per-Layer Embedding)

0-indexed layer 1 のみ。`Qwen4ExpTextPLELayer` + `Qwen4ExpTextNGramEmbedding`。

### 9.1 n-gram hash

`ngram_size = 3`, `heads_per_ngram = 8`。
n-gram 次数 2, 3 ごとに 8 head ずつ、合計 `ngram_heads = 16`。

各 head の語彙サイズは `ngram_vocab_size_base = 20_000_000` 以上の素数である:

```text
for head_idx in 0..15:
    global = ple_slot * 16 + head_idx
    size[head_idx] = nth_prime_after(19_999_999, global + 1)
    offset[head_idx] = cumulative_sum
```

`ple_slot` は `ple_layer_ids` 内の 0-indexed 位置である（現在の checkpoint は
`ple_layer_ids = [2]` 1 件のため `ple_slot = 0`）。decoder layer index ではない。

`layer_multipliers` は seed と `ple_slot` から splitmix64 で決まる
`ngram_size = 3` 個の奇数である:

```text
base_seed = seed + 10007 * ple_slot
multiplier[k] = 2 * (splitmix64((base_seed + 0x9E3779B97F4A7C15 * (k+1)) & mask64) % half_bound) + 1
half_bound = max(1, (max_int63 // unigram_vocab_size) // 2)
unigram_vocab_size = text_config.vocab_size            # 248320
splitmix64(x): x += 0x9E3779B97F4A7C15 から始まる標準形
half_bound = 18571544855136   (max multiplier = 37143089710273)
```

`unigram_vocab_size` は `ngram_vocab_size_base` ではなく `vocab_size` である。
multiplier の上限が token id の上限 (248319) と掛かっても `int63` に収まるように
`half_bound` が決まる。

実 checkpoint（revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`）の実値:

```text
layer_multipliers  = [23703573157769, 20109073645365, 8052911324071]
ngram_heads_vocab_sizes = [20000003, 20000023, 20000033, 20000047, 20000059,
    20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
    20000153, 20000159, 20000161, 20000171]
ngram_heads_offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165,
    120000228, 140000297, 160000374, 180000455, 200000548, 220000655,
    240000802, 260000955, 280001114, 300001275]
```

これら 3 tensor は checkpoint に I64 で実在する。
`seed` は `config.json` に**存在しない**（既定値 1234 を使う）。

ID 生成:

```text
token_history = concat(previous_context (2 tokens), input_ids)
s[k] = shift_right_ignore_eos(token_history, k)      # k = 0,1,2, eos を跨がない
for ngram in {2, 3}:
    h = s[0] * multiplier[0]
    for pos in 1..ngram-1:
        h = h XOR (s[pos] * multiplier[pos])
    for head in that ngram's 8 heads:
        id[head] = (h mod size[head]) + offset[head]
embed = ngram_embedding(concat(id[0..15]))           # [.., 16, 160] -> [.., 2560]
```

`head_dim_per_ngram = ple_embed_dim // ngram_heads = 2560 // 16 = 160`。
`ngram_embedding.weight` の行数は全 head 合計を 128 で割り切り上げた
`320_001_536` (= 128 shard * 2_500_012) 行である。
checkpoint では `shard_0..shard_127` に分かれ、各行数 `2_500_012` を持つ。

`shift_right_ignore_eos` は eos を segment 境界として扱い、
shift 分が segment を跨ぐ場合は eos を埋める。

### 9.2 projection / gate / conv

`hidden_states` は 4-stream (`[.., 10240]`) である。

```text
key = norm_key(key_proj(embed)).unflatten(-1, (4, 2560))     # key_proj: 2560 -> 10240
value = value_proj(embed)                                     # 2560 -> 2560
query = norm_query(hidden_states).unflatten(-1, (4, 2560))
gate = sum_over_2560(key * query) / sqrt(2560)               # [.., 4, 1]
gate = sqrt(abs(gate).clamp_min(1e-6)) * sign(gate)
gated = sigmoid(gate) * value.unsqueeze(-2)                  # [.., 4, 2560]
gated_normed = norm_conv(gated.flatten(-2))                  # [.., 10240]
conv = silu( depthwise_conv1d(gated_normed, conv1d.weight, dilation=ngram_size=3) )
out = gated.flatten(-2) + conv                               # [.., 10240]
```

`ple.conv1d.weight` shape は `[10240, 1, 4]`、dilation = `ngram_size = 3`、
depthwise である。`short_conv_state_len = (4 - 1) * 3 = 9`。

`ple.norm_*` は grouped RMSNorm (`group_size = 2560`、`(1+w)`) である。

PLE の n-gram table は約 51.2B パラメータ (BF16 で約 102 GB) であり、
GPU 常駐を前提にしない。1 token あたりの参照は 16 row (5.0 KiB) のみであり、
必要 row を外部 memory / storage から gather する。
state は 3 つに分かれる (`number_of_conv_states = 3`):
`conv_states[0]` = GDN conv、`conv_states[1]` = PLE short conv、
`conv_states[2]` = n-gram context (トークン履歴) である。

---

## 10. lm_head

`lm_head.weight` は `[248320, 2560]` で `embed_tokens` とは tie しない。
final norm は存在しない。`hyper_connection_mixer` の出力を直接 `lm_head` に通す。

---

## 11. MTP

MTP は Transformers 参照実装に存在せず、`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`
として読み込み時に無視される。したがって現時点で正しい forward の正本がない。

checkpoint から読み取れる構造:

| tensor | shape |
| --- | --- |
| `mtp.fc_embedding.weight` | `[2560, 2560]` |
| `mtp.fc_hidden.weight` | `[2560, 2560]` |
| `mtp.pre_fc_norm_embedding.weight` | `[2560]` |
| `mtp.pre_fc_norm_hidden.weight` | `[10240]` |
| `mtp.hyper_connection_mixer.*` | GatedResidual (use_combine=False) と同形 |
| `mtp.layers.0.*` | QSA + MoE + HyperConnection を持つ 1 decoder layer |

`mtp.layers.0` は `self_attn` (QSA) を持ち、`linear_attn` を持たない。
`mtp_num_hidden_layers = 1`、`mtp.layer_types = ["full_attention"]`。

`pre_fc_norm_hidden` が `[10240]` (= 4-stream) である一方、
`fc_hidden` / `fc_embedding` が `[2560, 2560]` である点は、
MTP が 4-stream 表現を入力に取ることを示唆するが、
正確な forward は Gate 10 で Qwen 実装 / tech report を追跡して確定する。
この contract は MTP を未確定として扱う。

---

## 12. safetensors tensor manifest

text backbone の全 tensor pattern は以下で尽くされる
(`N` は layer index、`shard_N` は 0..127)。
dtype は特別な 3 tensor を除き BF16 である。

| category | disposition | tensor pattern | shape |
| --- | --- | --- | --- |
| token_embedding | REUSE | `model.language_model.embed_tokens.weight` | `[248320, 2560]` |
| lm_head | REUSE | `lm_head.weight` | `[248320, 2560]` |
| hyper_connection_final | NEW | `model.language_model.hyper_connection_mixer.hc_norm.weight` | `[10240]` |
| hyper_connection_final | NEW | `...hyper_connection_mixer.input_mix_weight_down.weight` | `[320, 10240]` |
| hyper_connection_final | NEW | `...hyper_connection_mixer.input_mix_weight_up.weight` | `[10240, 320]` |
| hyper_connection_layer | NEW | `layers.N.attn_hyper_connection.hc_norm.weight` | `[10240]` |
| hyper_connection_layer | NEW | `layers.N.attn_hyper_connection.input_mix_weight_down.weight` | `[320, 10240]` |
| hyper_connection_layer | NEW | `layers.N.attn_hyper_connection.input_mix_weight_up.weight` | `[10240, 320]` |
| hyper_connection_layer | NEW | `layers.N.attn_hyper_connection.block_inject_weight.weight` | `[4, 10240]` |
| hyper_connection_layer | NEW | `layers.N.mlp_hyper_connection.*` (同 4 本) | 同上 |
| gdn | ADAPT | `layers.N.linear_attn.in_proj_qkv.weight` | `[10240, 2560]` |
| gdn | ADAPT | `layers.N.linear_attn.in_proj_z.weight` | `[6144, 2560]` |
| gdn | ADAPT | `layers.N.linear_attn.in_proj_a.weight` | `[48, 2560]` |
| gdn | ADAPT | `layers.N.linear_attn.in_proj_b.weight` | `[48, 2560]` |
| gdn | ADAPT | `layers.N.linear_attn.out_proj.weight` | `[2560, 6144]` |
| gdn | ADAPT | `layers.N.linear_attn.conv1d.weight` | `[10240, 1, 4]` |
| gdn | ADAPT | `layers.N.linear_attn.A_log` | `[48]` |
| gdn | ADAPT | `layers.N.linear_attn.dt_bias` | `[48]` |
| gdn | ADAPT | `layers.N.linear_attn.norm.weight` | `[128]` |
| qsa_attention | ADAPT | `layers.N.self_attn.q_proj.weight` | `[12288, 2560]` |
| qsa_attention | ADAPT | `layers.N.self_attn.k_proj.weight` | `[512, 2560]` |
| qsa_attention | ADAPT | `layers.N.self_attn.v_proj.weight` | `[512, 2560]` |
| qsa_attention | ADAPT | `layers.N.self_attn.o_proj.weight` | `[2560, 6144]` |
| qsa_attention | ADAPT | `layers.N.self_attn.q_norm.weight` | `[256]` |
| qsa_attention | ADAPT | `layers.N.self_attn.k_norm.weight` | `[256]` |
| qsa_indexer | NEW | `layers.N.self_attn.indexer.index_qk_proj.weight` | `[640, 2560]` |
| qsa_indexer | NEW | `layers.N.self_attn.indexer.q_layernorm.weight` | `[128]` |
| qsa_indexer | NEW | `layers.N.self_attn.indexer.k_layernorm.weight` | `[128]` |
| routed_experts | NEW | `layers.N.mlp.experts.gate_up_proj` | `[512, 1280, 2560]` |
| routed_experts | NEW | `layers.N.mlp.experts.down_proj` | `[512, 2560, 640]` |
| router | NEW | `layers.N.mlp.gate.weight` | `[512, 2560]` |
| shared_expert | ADAPT | `layers.N.mlp.shared_expert.gate_proj.weight` | `[640, 2560]` |
| shared_expert | ADAPT | `layers.N.mlp.shared_expert.up_proj.weight` | `[640, 2560]` |
| shared_expert | ADAPT | `layers.N.mlp.shared_expert.down_proj.weight` | `[2560, 640]` |
| shared_expert | ADAPT | `layers.N.mlp.shared_expert_gate.weight` | `[1, 2560]` |
| ple_other | NEW | `layers.N.ple.key_proj.weight` | `[10240, 2560]` |
| ple_other | NEW | `layers.N.ple.value_proj.weight` | `[2560, 2560]` |
| ple_other | NEW | `layers.N.ple.conv1d.weight` | `[10240, 1, 4]` |
| ple_other | NEW | `layers.N.ple.norm_key.weight` | `[10240]` |
| ple_other | NEW | `layers.N.ple.norm_query.weight` | `[10240]` |
| ple_other | NEW | `layers.N.ple.norm_conv.weight` | `[10240]` |
| ple_other | NEW | `layers.N.ple.ple_embedding.layer_multipliers` | `[3]` I64 |
| ple_other | NEW | `layers.N.ple.ple_embedding.ngram_heads_vocab_sizes` | `[16]` I64 |
| ple_other | NEW | `layers.N.ple.ple_embedding.ngram_heads_offsets` | `[16]` I64 |
| ple_ngram_table | NEW | `layers.N.ple.ple_embedding.ngram_embedding.shard_N.weight` | `[2500012, 160]` |
| mtp | NEW | `mtp.*` (31 tensor) | section 11 参照 |
| vision | IGNORE | `model.visual.*` (333 tensor) | vision_config 参照 |

`layers.N.linear_attn.*` は 36 層、`layers.N.self_attn.*` は 12 層に存在する。
`layers.N.ple.*` は layer 1 のみに存在する。
tensor 総数は 1658 (text backbone 1294 required + MTP 31 optional + vision 333 ignored)。
shape / shard / dtype の全列挙は実 checkpoint の safetensors ヘッダから取得できる
（記録は
[research/20261024/qwen4exp/model_inventory.md](../research/20261024/qwen4exp/model_inventory.md)）。

---

## 13. runtime state 一覧

| state | 対象層 | 内容 |
| --- | --- | --- |
| QSA KV cache | 12 | K/V、kv_heads 2、head_dim 256 |
| QSA indexer key cache | 12 | raw key 1 head、head_dim 128 |
| GDN conv state | 36 | `[B, 10240, 4]` |
| GDN recurrent state | 36 | `[B, 48, 128, 128]` fp16（5.0 の精度方針） |
| PLE short conv state | 1 | `[B, 10240, 9]` |
| PLE n-gram context state | 1 | トークン履歴 2 token (integer) |

既存 `PrimitiveStateKind` は `KV_CACHE` / `GDN_CONV_STATE` / `GDN_RECURRENCE_STATE`
のみである。QSA indexer key cache は新規 state kind が必要である
（`KV_CACHE` は paged pool と `kv_heads * head_dim` の feature 検証が前提で、
`[tokens, 128]` の連続 raw key を保持できない）。

### owner

| state | owner | 根拠 |
| --- | --- | --- |
| QSA KV cache | 既存 `PagedKVPool`（app 側） | `emit_gated_attention` が `KV_APPEND` / `PAGED_ATTENTION` を emit |
| GDN conv state | 既存 `GdnStatePool`（app 側） | qwen4exp の GDN は `conv_dim = 10240` / `conv_history = 4` で現行 layout と一致 |
| GDN recurrent state | 同上 | `num_v_heads = 48` / `head_k = head_v = 128` を現行 layout が保持 |
| QSA indexer key cache | qwen4exp-local の独立 pool | 新 state kind が必要。既存3種の所有者・レイアウトは変更しない |
| PLE short conv state | qwen4exp-local | `GdnStatePoolLayout` は `conv_history` を pool 全体で単一の値（4）として持つため、`conv_history = 9` の PLE を保持できない |
| PLE n-gram context state | qwen4exp-local（host integer） | 既存の state kind はいずれも GPU tensor 前提 |

`GDN_CONV_STATE` の `state_index` で PLE short conv を表現することは
上記の `conv_history` 制約により**できない**。

---

## 14. gap (current PhaseShift)

| # | gap | 影響範囲 |
| --- | --- | --- |
| 1 | MoE の Primitive（`MOE_ROUTER` / `MOE_EXPERTS`）、correctness / optimized kernel、`lower_qwen4exp_to_primitives` からの emit（`connect_moe`）、`connect_moe=true` の full-graph 数値検証はいずれも実装済み。shared expert は既存ノードで表現。**optimized dispatch 接続が未**（現状は correctness fallback。`kernels/optimized/moe_ops.hip` は router 出力が row_stride 契約に未対応） | MoE runtime |
| 2 | fused 3D expert weight (`[E, 2I, H]` / `[E, H, I]`) の axis-0 slice ロードと、expert ごとの rank-2 `WeightSlot` 連続展開による MoE 実行経路は実装済み。**実モデル 512 expert の常駐・キャッシュは未**（G6 の課題） | MoE runtime |
| 3 | selected-token attention の kernel（`kernels/optimized/attention/qsa_sparse_attention`）は実装済みだが、`PagedAttentionNode` への mask 入力が未接続（optional な2番目入力で表現可能なことが G4c の判定） | QSA runtime |
| 4 | QSA indexer / sparse attention の kernel と qwen4exp-local の `IndexerKeyPool`（`models/qwen4exp/runtime/`）は実装済み。**`PrimitiveStateKind` への indexer key 登録と lowering 接続が未** | QSA runtime |
| 5 | Gated Residual は host 参照実装に加え、**PrimitiveGraph へ載る経路が実装済み**（`GROUP_MEAN` / `GROUP_BROADCAST_MUL` の2種 + 既存ノードへの分解、`models/common/qwen_hybrid/hyper_lowering.h` の emit）。**層構造への接続は下記 8** | HC runtime |
| 6 | PLE の SSD row gather（`ple/ple_row_gather`）、n-gram hash（`ple/ngram_hash`）、dilated conv（`ple/ple_conv`）、graph 接続（`ple/ple_forward` + `connect_ple` による prefix graph 分割、新規 Node なし）は実装済み。**projection / gate / conv の GPU 実行経路は未**（現状 host F32）。実モデル 95.4 GiB table の SSD 経路 E2E は G6 | PLE runtime |
| 8 | qwen4exp の config / weights 読み込み、qwen4exp-local の state pool、`lower_qwen4exp_to_primitives`（層結線、`connect_moe` / `connect_ple` は既定 false）、PLE の graph 接続（prefix graph 分割）はいずれも実装済み。**entrypoint（`phaseshift-compute` / `phaseshift-bench`）からの起動経路、`ContinuousBatcher` / `PrefixCache` 統合、optimized dispatch 接続が未** | qwen4exp runtime |
| 9 | MTP の正本 forward が存在しない | Gate 10 |
| 10 | vision tower 未対応 | Gate 11 |

`lower_to_primitives` は QSA attention の gated query (`q_proj` 2x + sigmoid gate) と
partial rotary をすでに扱う。GDN の conv / recurrence / grouped RMSNorm (ONE_PLUS) も
primitive として存在する。これらは geometry と activation の差分で再利用する。
GDN の output gate は `GdnLoweringGeometry::output_gate_silu` で `SiLUNode` / `SigmoidNode` を
切り替えられる。`output_gate_type: "sigmoid"` はこの経路で sigmoid を選ぶ。

## 15. tensor partition / TP execution

routed expert を将来 EP で分散する場合の weight storage geometry の正本は
[tensor_partition.md](tensor_partition.md) である
（`TensorPartitionDesc` による axis=0 single-range partition で表現できる）。
Qwen3.8 Dense 向けの TP execution runtime は
[tensor_parallel_execution.md](tensor_parallel_execution.md) を参照。
現時点で Qwen3.8-Flash-Next / MoE / EP の execution は未実装である。

