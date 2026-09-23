# M2 Worker A — Qwen3 forward semantics (frozen spec)

Sources (authoritative, both read directly — no assumptions from Qwen2):

| Source | Path |
| --- | --- |
| Real GGUF metadata + inventory | `models/Qwen3-32B-Q4_K_M.gguf` (707 tensors, GGUF v3, via `inspect_gguf.py` / struct dump) |
| Official HF implementation | `/tmp/qwen-fp8-venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py` (transformers 5.13.0) |
| Official HF config defaults | `.../transformers/models/qwen3/configuration_qwen3.py` |
| Official Qwen3-32B `config.json` | `https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json` (architectures `Qwen3ForCausalLM`) |
| llama.cpp reference (GGUF consumer) | `_reference/llama.cpp/src/models/qwen3.cpp`, `_reference/llama.cpp/src/llama-model.cpp` |

Below, `M:NN` = line `NN` of `modeling_qwen3.py`, `C:NN` = line `NN` of `configuration_qwen3.py`.

---

## 1. Frozen numbers

| Quantity | Value | Source |
| --- | --- | --- |
| `hidden_size` (`n_embd`) | **5120** | GGUF `qwen3.embedding_length=5120`; config `hidden_size: 5120` |
| `num_hidden_layers` | **64** | GGUF `qwen3.block_count=64`; config `num_hidden_layers: 64` |
| `intermediate_size` (`n_ff`) | **25600** | GGUF `qwen3.feed_forward_length=25600`; config `intermediate_size: 25600` |
| `num_attention_heads` (Q heads) | **64** | GGUF `qwen3.attention.head_count=64` |
| `num_key_value_heads` (KV heads) | **8** | GGUF `qwen3.attention.head_count_kv=8` |
| `head_dim` | **128** (`qwen3.attention.key_length=128`, `value_length=128`) | config `head_dim: 128` |
| **`hidden_size / num_heads`** | **80** (= 5120/64) | arithmetic |
| `rms_norm_eps` | **1e-6** — GGUF stores f32 `9.999999974752427e-07` (f32 rounding of 1e-6); config `rms_norm_eps: 1e-06` | `qwen3.attention.layer_norm_rms_epsilon` |
| RoPE base (`theta`) | **1_000_000.0** | GGUF `qwen3.rope.freq_base=1000000.0`; config `rope_theta: 1000000` |
| RoPE type | **default (no scaling)** | config `rope_scaling: null`; `attention_scaling = 1.0` (M:126) |
| RoPE dims rotated | **all 128** (`n_rot == head_dim`, no partial rotary) | M:125 uses `dim = head_dim = 128`; llama.cpp `GGML_ASSERT(n_embd_head == n_rot)` (`qwen3.cpp` L~73) |
| attention scale | **128\*\*-0.5 = 0.08838834764831845** | M:232 `self.scaling = self.head_dim**-0.5` |
| `query_pre_attn_scalar` | **does not exist** in Qwen3 (`grep query_pre_attn_scalar models/qwen3/*.py` → no match, rc=1) | — |
| context length | **40960** | GGUF `qwen3.context_length=40960`; config `max_position_embeddings: 40960` |
| `vocab_size` | **151936** | GGUF `token_embd.weight` ne[1]=151936 |
| attention bias | **false everywhere** (no `*.bias` tensors in 707-tensor inventory) | config `attention_bias: false` |
| MLP activation | **silu** (SiLU/Swish) | config `hidden_act: "silu"`; M:79 `ACT2FN[config.hidden_act]` |
| sliding window | **NOT used** (`use_sliding_window: false`, `sliding_window: null`, `max_window_layers: 64`); all 64 layers are `full_attention` | C:77-96 (`layer_types` all `full_attention`); M:370 `has_sliding_layers=False` |
| tie embeddings | **NOT tied** — separate `output.weight` present (`q6_k`, 151936×5120) | config `tie_word_embeddings: false`; GGUF has both `token_embd.weight` and `output.weight` |
| attention dropout | 0.0 (inference) | config `attention_dropout: 0.0` |

### 1.1 head_dim is NOT hidden/heads (CRITICAL)

`5120 / 64 = 80`, but `head_dim = 128`. Qwen3-32B must be driven with `head_dim = 128`, not 80.

Confirmed three ways:
1. GGUF `qwen3.attention.key_length=128` / `value_length=128`.
2. `blk.0.attn_q.weight` shape ne = `[5120, 8192]` where **8192 = 64 heads × 128**, and `blk.0.attn_k.weight`/`attn_v.weight` ne = `[5120, 1024]` where **1024 = 8 KV heads × 128**.
3. M:230 `self.head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)` — `config.head_dim` exists (=128) so the fallback is never taken; M:125 same pattern for RoPE.

Consequences: `q_proj` out = 8192, `k_proj`/`v_proj` out = 1024, `o_proj` in = 8192. `num_key_value_groups = 64/8 = 8` (M:231).

---

## 2. Exact RMSNorm

M:50-66, identical to T5LayerNorm:

```
rmsnorm(x; w, eps) = w * ( x / sqrt( mean(x^2, over last dim) + eps ) )
```

- computed in **fp32** internally (`hidden_states.to(torch.float32)`), weight multiplied in fp32, result cast back to input dtype (M:60-64);
- `eps = 1e-6` (f32: 9.999999974752427e-07) for **all four** norms per layer plus final norm:
  `attn_norm`, `post_attention_layernorm`, `attn_q_norm`, `attn_k_norm` (M:248-249 for q/k norm, M:321-322 for the two layer norms);
- **no mean subtraction** (RMS, not layer norm), **no bias**;
- `attn_norm`/`ffn_norm` reduce over `hidden_size`=5120; `q_norm`/`k_norm` reduce over `head_dim`=128.

---

## 3. Exact RoPE

**Layout = standard HF `rotate_half` (NeoX / "half-split" convention)**, not interleaved pairs.

```
inv_freq[i] = 1 / theta^(2i/dim)        for i in [0, dim/2), dim = 128, theta = 1e6   (M:123-133)

freqs[s, i] = pos[s] * inv_freq[i]                                  (M:142)
emb         = cat(freqs, freqs)      # length dim = 128              (M:144)
cos = cos(emb) * attention_scaling   # attention_scaling = 1.0      (M:145)
sin = sin(emb) * attention_scaling                                  (M:146)

rotate_half(x) = cat(-x[..., half:], x[..., :half])                 (M:151-155), half = 64
q' = q*cos + rotate_half(q)*sin
k' = k*cos + rotate_half(k)*sin                                     (M:171-173)
```

So for a head vector of 128, RoPE pairs element `i` with element `i+64` (`i ∈ [0,64)`): the half-split layout, each pair rotated by angle `pos * inv_freq[i]`. That equals llama.cpp `LLAMA_ROPE_TYPE_NEOX`, and llama.cpp does select NEOX for Qwen3 (`llama-model.cpp` L2994-2995 `case LLM_ARCH_QWEN3:` inside the NEOX group; the NORM group ends at L2975).

- Not interleaved (`LLAMA_ROPE_TYPE_NORM` would pair 2i/2i+1) → **do not** use the LLaMA-1-style pair rotation.
- No `partial_rotary_factor` (all 128 dims rotated), no YaRN/longrope scaling.
- RoPE is applied **after** q_norm/k_norm and **before** the attention matmul (M:263-268), and **only to q/k, not v**.
- `position_ids = arange(seq_len) + past_seen_tokens` (M:392-395) → pure 1-D positions.

---

## 4. Ordering facts (per layer)

| Item | Ordering | Evidence |
| --- | --- | --- |
| Q norm ordering | `a → q_proj → reshape to (…, n_head=64, head_dim=128) → **q_norm over head_dim** → transpose(1,2) → RoPE` | M:263 `self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)` |
| K norm ordering | `a → k_proj → reshape to (…, n_kv=8, 128) → **k_norm over head_dim** → transpose(1,2) → RoPE` | M:264 |
| V | `a → v_proj → reshape → transpose`; **no norm, no RoPE** | M:265 |
| **q_norm/k_norm are PER HEAD over head_dim=128** | Yes. Weight shape is `[128]` (GGUF `blk.0.attn_q_norm.weight` dims=128, `blk.0.attn_k_norm.weight` dims=128); applied on the last axis of the `(…, n_head, 128)` view → one 128-element scale vector reused for every head; **not** per-q-vector over 8192, **not** over 80 | M:248-249 (`Qwen3RMSNorm(self.head_dim, …)`), M:263-264; llama.cpp `build_norm(Qcur, attn_q_norm …)` then rope (`qwen3.cpp` L86-102) |
| residual ordering | two **pre-norm** residuals, attention then MLP: `x = x0 + attn(...)`, then `x = r + mlp(...)` where `r = x` | M:315-333 |
| MLP ordering | `norm → gate_proj → silu → × up_proj → down_proj` (gated, PAR/SiLU-GLU) | M:81-83, `silu(gate)*up`; llama.cpp `LLM_FFN_SILU, LLM_FFN_PAR` |
| final norm | one `Qwen3RMSNorm(5120, eps=1e-6)` after layer 63, then `lm_head` | M:367, M:434, M:505 |
| embedding/output weight | **separate tensors, untied**: `token_embd.weight` (q4_k) and `output.weight` (q6_k) both exist | GGUF inventory; config `tie_word_embeddings: false`; llama.cpp `create_tensor(..., LLM_TENSOR_OUTPUT, ..., TENSOR_NOT_REQUIRED)` |

No biases exist anywhere in this GGUF (inventory has exactly 3 non-layer tensors + 11 per layer, all `.weight`).

---

## 5. Layer pseudocode (as required by `docs/M2.md`, verbatim)

```text
x0 = x

a = rmsnorm(x)

q = q_proj(a)
k = k_proj(a)
v = v_proj(a)

q = q_norm(q)
k = k_norm(k)

q,k = rope(q,k,pos)

attn = causal_gqa(q,k,v)

aout = o_proj(attn)

x = x0 + aout

r = x
m = rmsnorm(x)

gate = gate_proj(m)
up   = up_proj(m)

m = silu(gate) * up
m = down_proj(m)

x = r + m
```

### 5.1 Verified against official implementation — deviations/precisions only

The pseudocode is a faithful match. Precisions needed to turn it into executable code:

1. `rmsnorm` = §2 formula with `eps = 1e-6`, weight-only, fp32 accumulate.
2. `a = rmsnorm(x)` is `self.input_layernorm` → GGUF **`blk.{i}.attn_norm.weight`**.
3. `m = rmsnorm(x)` is `self.post_attention_layernorm` → GGUF **`blk.{i}.ffn_norm.weight`** (the second residual's norm, taken on `r`, i.e. after the attention residual add — M:330-331).
4. `q`, `k`, `v` are per-head views `(n_head, head_dim)`; `q_norm`/`k_norm` operate on that 128-wide last axis → **per head**, before RoPE.
5. `rope(q,k,pos)` = §3, NeoX half-split, over all 128 dims, `theta=1e6`; applies to q and k only.
6. `causal_gqa` = causal softmax attention with `scaling = head_dim**-0.5` (see §6); no sliding window, no `query_pre_attn_scalar`, no dropout at inference, `is_causal=True`.
7. Attention output is reshaped `(n_head * head_dim) = 8192` before `o_proj` (M:289-290).
8. `q_proj/k_proj/v_proj/o_proj` are **bias-free**.
9. Whole-model: `x = embed(token_embd) → 64 × layer → final rmsnorm(5120) → lm_head(output.weight)`, logits in fp32 not required (M:505).
10. No RoPE on `v`, no norm on `v`, **no norm inside MLP other than `ffn_norm`**.

---

## 6. Attention semantics

```
# shapes: B batch, S seq, H=64, Hkv=8, D=128, S_kv = S + past
q: [B,H,S,D]  (after q_norm + rope)
k: [B,Hkv,S_kv,D] (after k_norm + rope)
v: [B,Hkv,S_kv,D]
k,g/v: repeat_kv n_rep = H/Hkv = 8   -> KV head j feeds Q heads [8j, 8j+8)   (M:184-193, torch.repeat_interleave order)
scores = (q @ k^T) * 0.08838834764831845      (M:232, M:209)
scores += additive_causal_mask                 # 0 on j<=i, -inf on j>i      (M:210-211, M:414)
p = softmax(scores, dim=-1, dtype=float32).to(q.dtype)                       (M:213)
attn = p @ v   -> transpose(1,2) -> reshape to (H*D = 8192)                  (M:215, M:289)
out  = o_proj(attn)                                                          (M:290)
```

- **scaling**: `head_dim**-0.5` = `128**-0.5` ≈ `0.08838834764831845`. No `query_pre_attn_scalar` in Qwen3 (that is a Qwen2-VL/Qwen2.5-VL concept); grep over `models/qwen3/*.py` returns nothing.
- **causal masking**: strict lower-triangular inclusive (token `i` attends to `j ≤ i`), additive `-inf` before softmax; prefill uses `create_causal_mask` (M:414), decode uses the same cache-length masking with a 1-token query.
- **GQA**: 64 Q heads, 8 KV heads, contiguous group of 8 (repeat_interleave, not strided).
- **softmax is computed in float32** then cast back to the q dtype.
- **sliding window**: not used for this config — `layer_types[i] = "full_attention"` for all 64 layers (C:91-96 with `sliding_window=None` because `use_sliding_window=False`, C:87), and `has_sliding_layers=False` (M:370). Also note `max_window_layers=64` means even with sliding enabled no layer would switch.
- Masking for the cached suffix: mask width = full KV length; single query attends to all cached positions.

---

## 7. GGUF metadata actually present (verbatim keys)

```
general.architecture = qwen3
general.type = model
general.name = Qwen3 32B Awq Compatible Instruct
general.finetune = awq-compatible-Instruct
general.basename = Qwen3
general.size_label = 32B
qwen3.block_count = 64
qwen3.context_length = 40960
qwen3.embedding_length = 5120
qwen3.feed_forward_length = 25600
qwen3.attention.head_count = 64
qwen3.attention.head_count_kv = 8
qwen3.rope.freq_base = 1000000.0
qwen3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
qwen3.attention.key_length = 128
qwen3.attention.value_length = 128
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = <str>
tokenizer.ggml.tokens / token_type / merges = <lists>
tokenizer.ggml.eos_token_id = 151645
tokenizer.ggml.bos_token_id = 151643
tokenizer.ggml.padding_token_id = 151643
tokenizer.ggml.add_bos_token = <bool>
tokenizer.chat_template = <str>
general.quantization_version = 2
general.file_type = 15            # Q4_K_M
```

Notes / decisions:

- **There is NO `qwen3.rope.scaling.*`, NO `qwen3.attention.sliding_window`, NO `qwen3.attention.max_window_layers`, NO `qwen3.rope.scale_linear`, NO `qwen3.attention.causal` key.** Only 28 metadata KVs total. Anything not listed above must be defaulted, not read from the file.
- The architecture prefix is `qwen3.` (not `qwen2.`); the parser must key on `general.architecture`.
- `general.file_type = 15` = Q4_K_M per llama.cpp `llama_ftype` (F32=0 … Q4_K_M=15), consistent with the observed tensor mix.
- Header: GGUF v3, `n_tensors = 707`, `n_kv = 28`, `alignment = 32`, tensor info ends at file offset `5974661`, tensor data starts at `5974688`. All 707 tensor offsets+bytes fit within the 19762149024-byte file (`max_end == file_size`) → tight packing, no gaps/padding beyond 32-byte alignment.

---

## 8. Exact tensor names for one layer (for Worker B)

Layer count = 64. `blk.0` and `blk.63` inventories below are byte-exact from the real GGUF.

Per layer, **exactly 11 tensors** (order as stored in `blk.0`):

| GGUF name | qtype | GGUF ne (ne[0] = contiguous row width) | log tensor | role |
| --- | --- | --- | --- | --- |
| `blk.0.attn_k.weight` | **q4_k** | `[5120, 1024]` | k_proj | 8 KV heads × 128 |
| `blk.0.attn_k_norm.weight` | **f32** | `[128]` | k_norm | per-head RMSNorm over head_dim |
| `blk.0.attn_norm.weight` | **f32** | `[5120]` | input_layernorm | pre-attention RMSNorm |
| `blk.0.attn_output.weight` | **q4_k** | `[8192, 5120]` | o_proj | 64×128 in → 5120 |
| `blk.0.attn_q.weight` | **q4_k** | `[5120, 8192]` | q_proj | 5120 → 64×128 |
| `blk.0.attn_q_norm.weight` | **f32** | `[128]` | q_norm | per-head RMSNorm over head_dim |
| `blk.0.attn_v.weight` | **q6_k** on layer 0 (see §8.1) | `[5120, 1024]` | v_proj | 8 KV heads × 128 |
| `blk.0.ffn_down.weight` | **q6_k** on layer 0 (see §8.1) | `[25600, 5120]` | down_proj | 25600 → 5120 |
| `blk.0.ffn_gate.weight` | **q4_k** | `[5120, 25600]` | gate_proj | 5120 → 25600 |
| `blk.0.ffn_norm.weight` | **f32** | `[5120]` | post_attention_layernorm | pre-MLP RMSNorm |
| `blk.0.ffn_up.weight` | **q4_k** | `[5120, 25600]` | up_proj | 5120 → 25600 |

Global (non-layer) tensors:

| GGUF name | qtype | ne | role |
| --- | --- | --- | --- |
| `token_embd.weight` | **q4_k** | `[5120, 151936]` | embedding, 20 Q4_K blocks per row |
| `output_norm.weight` | **f32** | `[5120]` | final norm |
| `output.weight` | **q6_k** | `[5120, 151936]` | lm_head, **separate/untied** (42 q6_k blocks per row) |

Raw inventory lines (verbatim tool output, `ne[0]xne[1]`):
`blk.0.attn_k.weight q4_k dims=1024x5120`, `blk.0.attn_k_norm.weight f32 dims=128`, `blk.0.attn_norm.weight f32 dims=5120`,
`blk.0.attn_output.weight q4_k dims=5120x8192`, `blk.0.attn_q.weight q4_k dims=8192x5120`, `blk.0.attn_q_norm.weight f32 dims=128`,
`blk.0.attn_v.weight q6_k dims=1024x5120`, `blk.0.ffn_down.weight q6_k dims=5120x25600`, `blk.0.ffn_gate.weight q4_k dims=25600x5120`,
`blk.0.ffn_norm.weight f32 dims=5120`, `blk.0.ffn_up.weight q4_k dims=25600x5120`, `blk.63.*` identical shapes.

Blocks per row (row width = ne[0], 256 elems / Q4_K or Q6_K block):
`attn_q`/`attn_k`/`attn_v` 5120 → 20; `attn_output` 8192 → 32; `ffn_gate`/`ffn_up` 5120 → 20; `ffn_down` 25600 → 100; `token_embd`/`output` 5120 → 20. All divisible by 256, no padding rows.

### 8.1 qtype census — **attn_v and ffn_down are NOT uniform across layers**

| tensor | qtype |
| --- | --- |
| `blk.*.attn_k`, `blk.*.attn_q`, `blk.*.attn_output`, `blk.*.ffn_gate`, `blk.*.ffn_up` | **q4_k on all 64 layers** |
| `blk.*.attn_norm`, `blk.*.ffn_norm`, `blk.*.attn_q_norm`, `blk.*.attn_k_norm` | **f32 on all 64 layers** |
| `blk.*.attn_v`, `blk.*.ffn_down` | **q6_k on 32 layers, q4_k on 32 layers** |

q6_k layers for `attn_v` and `ffn_down`:
`0,1,2,3,4,5,6,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,56,57,58,59,60,61,62,63`
q4_k layers: `8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54`
This is the canonical Q4_K_M rule: `i < n_layer/8 (=8) || i >= 7*n_layer/8 (=56) || (i - n_layer/8) % 3 == 2`.

Totals (matches GGUF exactly): q4_k 385, q6_k 65, f32 257, unknown 0. Per-layer q4_k = 5, plus `token_embd`; f32 = 4/layer plus `output_norm`; q6_k = 64 mixed + `output.weight`.

**M1 impact (flag for Worker B / M1 scope):** the M1 Q4_K path (`_reference/q38.c`, Q4_K block = 144 B) covers `attn_k/q/output`, `ffn_gate/up`, `token_embd`, and the 32 q4_k `attn_v`/`ffn_down` layers. It does **not** cover the **q6_k** tensors (`output.weight`, and `attn_v`/`ffn_down` on 32 layers, Q6_K block = 210 B). Either a Q6_K decode path is required in M2, or those 65 tensors need an up-front dequant fallback — they are load-bearing (lm_head especially; `output.weight` is 638 MB and must be read).

---

## 9. One-line frozen summary

`hidden=5120, layers=64, ffn=25600, H=64, Hkv=8, head_dim=128 (≠ 80), eps=1e-6, theta=1e6, rope=NeoX half-split over all 128 dims, scale=128^-0.5=0.08838834764831845, per-head q_norm/k_norm (128-wide, pre-RoPE), pre-norm dual residual (attn then MLP), MLP = down(silu(gate)·up), final RMSNorm(5120) then untied lm_head, full causal attention, no sliding window, no biases, no q/k/v on v beyond proj.`