# M2 / Worker C — Attention + RoPE + KV-cache donor analysis

Scope: minimal dependency closure for **Qwen3 GQA causal attention**, **Qwen3 RoPE**, and the
**KV cache** (prefill path + cached suffix path) for Qwen3-32B on NVIDIA GB10 / SM121
(`compute_121a`, CUDA 13.0, aarch64).

This document is **analysis only**. No source is copied here. No code is written.
It ends with the three mandatory sections: `COPY`, `EDIT`, `DELETE`.

---

## TL;DR

1. **No donor has a Qwen3-compatible causal GQA attention kernel.** q38.c is QSA/sparse-attention
   (top-k indexer + gather), q38-main is DeepSeek-V4-flavoured (sliding-window / compressed /
   MLA / per-head sinks, `head_dim == 512` hard assumptions). Attention must be **written**, using
   the donors only as *structural* reference.
2. **q38.c's `rope_kernel` is the correct RoPE layout (NeoX / half-split)** and is the closest thing
   to copyable, but it hardcodes `theta = 1e7` and ignores its `sections[4]` argument. Qwen3-32B needs
   `theta = 1e6`. → copy the *loop structure*, fix 2 constants, drop the sections plumbing.
3. **Neither donor materializes KV repetition.** Both map `kv_head = q_head / (n_head / n_kv)`
   directly (q38.c `attention_kernel:519`, `q38_qsa_ref.c`; llama.cpp uses the `ncols2` GQA tile
   broadcast). M2 must do the same: direct mapping, **no repeat-interleave copy**.
4. **FlashAttention is NOT reasonably extractable for M2.** llama.cpp's FA kernels drag in
   `ggml` tensor types, `ggml_cuda_flash_attn_ext_f16_extra_data` / `get_alloc_size` plumbing,
   `KQ_STRIDE = 256` padding, and a template/arch dispatch maze (`fattn.cu:172/357/760`).
   Correctness first → M2 uses a **simple two-pass max-softmax device kernel, written fresh**.
5. **KV cache: F16, layout `[layer][position][kv_head][head_dim]` = `[64][T][8][128]`**, FP32
   accumulation in attention. Append = device-to-device row copy *after* `k_norm` + RoPE.
   FP8 KV quantization (q38-main) is explicitly **out of scope** for M2.
6. **RMSNorm / SiLU / residual-add / RoPE / attention / KV cache do not exist in-tree.** M1 only
   landed `q3_cuda_dequantize_row` (Q4_K) and the Q4_K MMQ linear. Everything in this doc is new.
7. **SM121 is not a blocker.** q38.c/q38-main only use `__CUDA_ARCH__ >= 700/750/800` guards;
   llama.cpp has an explicit `GGML_CUDA_CC_DGX_SPARK = 1210` branch (`fattn.cu:357`). No SM90+
   assumption breaks under `compute_121a`. Only `-gencode arch=compute_121a,code=sm_121a` matters.
8. Estimated port: **~600–900 LOC across 3 new files** (+ 3 edited files, +1 Makefile rule).

---

## Verified target geometry (Qwen3-32B-Q4_K_M.gguf)

Parsed directly from `models/Qwen3-32B-Q4_K_M.gguf`; cross-checked against
`docs/research/m2/.tensors.txt`.

| Param | Value |
|---|---|
| architecture | `qwen3` |
| block_count (layers) | 64 |
| context_length (train) | 40960 |
| embedding_length (hidden) | 5120 |
| feed_forward_length | 25600 |
| attention.head_count (Q) | 64 |
| attention.head_count_kv (KV) | 8 |
| GQA ratio | 8 |
| key_length / value_length (head_dim) | 128 / 128 |
| rope.freq_base | **1000000.0** (1e6) |
| rms_eps | 1e-6 (`9.99999997e-07`) |
| attn scale | `1/sqrt(128)` |

Real tensor names / shapes / types (from `.tensors.txt`):

```
blk.N.attn_norm.weight      f32          5120      (RMSNorm, weight-only)
blk.N.attn_q.weight         q4_k   5120 x 8192
blk.N.attn_k.weight         q4_k   5120 x 1024
blk.N.attn_v.weight         q6_k   5120 x 1024
blk.N.attn_output.weight    q4_k   8192 x 5120
blk.N.attn_q_norm.weight    f32           128     (per-head RMSNorm, head_dim)
blk.N.attn_k_norm.weight    f32           128     (per-head RMSNorm, head_dim)
blk.N.ffn_norm.weight       f32          5120
```

Note `k`/`v` are only `1024 = 8 * 128` wide — GQA is baked into the projection, not the cache.

---

## Qwen3 attention + RoPE semantics (authoritative ordering)

Source of truth: `_reference/llama.cpp/src/models/qwen3.cpp` (159 lines).

Per layer:

```
1. attn_norm  : RMSNorm(hidden, attn_norm.weight)                  # LLM_NORM_RMS, weight-only
2. q = attn_q @ x   -> [64 heads x 128]
   k = attn_k @ x   -> [ 8 heads x 128]
   v = attn_v @ x   -> [ 8 heads x 128]
3. q = attn_q_norm : RMSNorm per head over head_dim=128  (qwen3.cpp:88)
   k = attn_k_norm : RMSNorm per head over head_dim=128  (qwen3.cpp:95)
4. q = rope(q, pos); k = rope(k, pos)                    # Q and K only, NeoX
5. out = causal_gqa_attention(q, k_cache, v_cache) * (1/sqrt(128))   (qwen3.cpp:108)
6. hidden += attn_output @ out                           # residual add
7. ffn_norm -> gate/up -> silu(gate)*up -> down -> residual add
```

**Norm semantics (critical Qwen3 vs Qwen3.8 difference):** llama.cpp uses `LLM_NORM_RMS` with
`weight` used **directly** (no `1 + weight`). q38.c's `rms()` (`q38_forward.c:80`) takes a
`one_plus` flag and calls `rms(q, w->q_norm, head_dim, true)` (`:347`) — i.e. **`(1 + weight)`**.
That is the Qwen3.8/gemma convention and is **wrong for Qwen3**. M2 must use `weight` directly.

**RoPE scale of application:** `rope` is applied to `q`/`k` **after** the per-head norm and **before**
attention. RoPE is **not** applied to `v`.

### RoPE layout — NEOX (half-split), exactly

- `llama-model.cpp` rope-type switch maps `LLM_ARCH_QWEN3` → **`LLAMA_ROPE_TYPE_NEOX`**.
- NEOX = HF `rotate_half`: pair `(x[i], x[i + rotary_dims/2])` for `i in [0, rotary_dims/2)`:
  ```
  a = base + i                    # first half
  b = base + rotary_dims/2 + i    # second half
  out[a] = x[a]*c - x[b]*s
  out[b] = x[a]*s + x[b]*c
  ```
- `rotary_dims = head_dim = 128` (full rotation, `n_rot = 128`; no partial-rotary in this GGUF).
- `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1`, no YARN, no mscale.
  Frequency: `theta_i = 1e6 ^ (-2i / 128)`, angle `= position * theta_i`.
- **This matches q38.c `cuda/q38_qsa_cuda.cu:328 rope_kernel` exactly in layout**, and does
  **not** match q38-main's `rope_tail_kernel`/`rope_yarn_ramp_dev` (YARN ramp) nor the generic
  llama.cpp `rope.cu` yarn form (carries `corr_dims`/`beta_fast`/`beta_slow` plumbing that
  degenerates to nothing at `ext_factor=0` but is dead weight).

---

## Donor inventory (file, symbol, line, verdict)

### q38.c — branch `qwen38-spark-proto` (M0 donor, model = Qwen3.8-Flash-Next / QSA)

| File : symbol : line | What it is | Verdict |
|---|---|---|
| `cuda/q38_qsa_cuda.cu:328` `rope_kernel` | NeoX half-split, `theta=1e7` hardcoded, `sections[4]` ignored, per-element | **COPY structure**, fix theta→1e6, drop sections |
| `cuda/q38_qsa_cuda.cu:506` `attention_kernel` | per-(token,head) 2-pass max→softmax, **direct** `kv_head = head/(q/kv)`, no mask, `sqrtf(head_dim)` inline | **COPY structure**, add causal mask, remove `gather` pre-step |
| `cuda/q38_qsa_cuda.cu:489` `gather_kernel` | top-k index gather into dense scratch | **DELETE** (QSA-only) |
| `cuda/q38_qsa.cu:118` `chain_append_kernel` | coarse row append into state_k/state_v | reference only — layout too coarse |
| `q38_qsa.c:74` `q38_qsa_state_append` | host-side row append on `main_k`/`main_v`/`index_k` | **EDIT idea only** (over-complex: 3 streams) |
| `q38_qsa.h` `q38_qsa_cache {uint8_t*; row_bytes; capacity; count}` | untyped row cache | **DELETE** (untyped, 3-cache, QSA) |
| `q38_forward.c:80` `rms(x, weight, n, one_plus)` | RMSNorm with `(1+weight)` | **EDIT**: drop `one_plus`, weight directly |
| `cuda/q38_cuda_primitives.cu:94` `rms_norm_kernel` | block-reduction RMSNorm | **COPY** as per-head RMSNorm reference |
| `cuda/q38_cuda_primitives.cu:120` `silu_kernel` | SiLU | **COPY** (needed for FFN, already adjacent to M1) |
| `q38_forward.c:92` `rope()` (CPU) | NeoX half-split CPU reference | **COPY** (oracle semantics) |
| `q38_forward.c:348/354` | norm→rope ordering for q/k | **COPY semantics** |
| `q38_forward.c:392+` GQA loop | direct kv_head mapping | **COPY semantics** |
| `q38_qsa_ref.c` | CPU QSA reference (top-k block scoring) | **DELETE** (QSA) |

### q38-main — branch `main` (Q4_K MMQ donor, model = DeepSeek-V4-style `ds4`)

| File : symbol : line | What it is | Verdict |
|---|---|---|
| `ds4_cuda.cu:7047` `attention_prefill_raw_kernel` | raw prefill scores | **DELETE** (windowed/MLA) |
| `ds4_cuda.cu:7182` `attention_prefill_raw_softmax_kernel` | causal (`k <= t`) softmax + out | **DELETE**, but *cite* as causal-mask pattern |
| `ds4_cuda.cu:7350` `attention_decode_mixed_kernel` | decode w/ `__shared__ scores[8192]`, sliding window, `use_comp_mask`, ratio pooling | **DELETE** (windowed + O(T) shared cap) |
| `ds4_cuda.cu:10116` `attention_static_mixed_heads8_online_kernel` | **flash-style online softmax**, warp-per-head, float4 staging, but **hard-returns unless `head_dim == 512`** | **DELETE for M2**, *keep as future FA reference* |
| `ds4_cuda.cu:6727` `rope_yarn_ramp_dev` | YARN ramp | **DELETE** (YARN not used) |
| `ds4_cuda.cu:6732` `rope_tail_kernel` | YARN-positioned rope | **DELETE** (wrong layout/variant) |
| fp8 KV quant kernels | quantized KV | **DELETE** (out of scope M2) |
| `cuda/mmq/*` | Q4_K MMQ | already absorbed in M1 — untouched |
| `ds4_cuda.cu:11800, 13200, 14500` caches | variable-resolution / compressed caches | **DELETE** |

### llama.cpp — upstream reference (commit `709fe755dfa810d77e2ac386292b29648b536864`)

| File : symbol : line | What it is | Verdict |
|---|---|---|
| `src/models/qwen3.cpp:60-159` | **authoritative Qwen3 graph** (order, norms, rope params, scale) | **COPY semantics** (the spec) |
| `src/llama-model.cpp` rope switch | `QWEN3 -> LLAMA_ROPE_TYPE_NEOX` | **COPY fact** |
| `src/llama-kv-cache.cpp:232-237` | `cache_k_l%d` / `cache_v_l%d`, shape `(n_embd_k_gqa, kv_size, n_stream)` | **COPY layout fact** |
| `ggml-cuda/fattn-vec.cuh` (609 lines) | FA vector kernel, `FATTN_KQ_STRIDE=256` | **DELETE** (runtime coupling) |
| `ggml-cuda/fattn-tile.cu` | tiled FA | **DELETE** (runtime coupling) |
| `ggml-cuda/fattn-mma-f16.cuh` | MMA FA, `__CUDA_ARCH__ == CC_TURING` special case at `:1863` | **DELETE** (runtime coupling) |
| `ggml-cuda/fattn.cu:172/357/760` | dispatch; `cc >= GGML_CUDA_CC_DGX_SPARK (1210)` branch at `:357` | **DELETE**, *cite* arch-dispatch evidence |
| `ggml-cuda/rope.cu` `rope_yarn` machinery | generic yarn rope | **DELETE** (plumbing; layout is what matters) |
| `ggml-cuda/common.cuh:60-61` | `GGML_CUDA_CC_BLACKWELL 1200`, `GGML_CUDA_CC_DGX_SPARK 1210` | **COPY fact** |

---

## Prefill path

**Inputs:** hidden states `[T, 5120]` for `T` tokens; `position` = `0..T-1` (no cache offset).
**Outputs:** attn out `[T, 5120]` + appended KV rows.

```
per layer:
  x      = RMSNorm(hidden, attn_norm.w)                 # T x 5120
  q      = Q4_K_linear(q_w, x)                          # T x 8192  (M1 MMQ, exists)
  k      = Q4_K_linear(k_w, x)                          # T x 1024
  v      = Q6_K_linear(v_w, x)                          # T x 1024  <-- see EDIT notes
  q      = per_head_rmsnorm(q, q_norm.w, 64 heads, 128)  # NEW k3 kernel
  k      = per_head_rmsnorm(k, k_norm.w,  8 heads, 128)  # NEW k3 kernel
  rope(q, positions[0..T)); rope(k, positions[0..T))     # NEW k3 kernel, NeoX, theta 1e6
  kv_cache_k[layer][0..T) = k ; kv_cache_v[layer][0..T) = v   # device row copy
  o      = causal_gqa_prefill(q, k, v, T)                # NEW k3 kernel
  hidden = hidden + Q4_K_linear(o_w, o.reshape(T,5120))
  hidden = hidden + ffn(hidden)                          # ffn_norm/gate/up/silu/down
```

**Prefill attention kernel (chosen design):**
- grid = `(T, 64)` — one CTA per `(query_token t, q_head h)`.
- block = 128 threads (one per `head_dim` lane).
- `kv_head = h >> 3`  (`h / 8`).
- Pass 1: for `j in [0, t]` (causal) compute `score_j = dot(q, k_j)/sqrt(128)`, track `max`.
  `j > t` contributes `-INFINITY` (mask by loop bound — no score buffer needed).
- Pass 2: `den = sum_j exp(score_j - max)`, `num[d] += exp_j * v_j[d]`.
- `out = num/den`.
- **No global scores tensor.** Scratch is `head_dim`-sized per-thread accumulator + broadcast `max`
  via `__shfl_xor_sync` (or a 128-float `__shared__` reduce buffer = 512 B/CTA).
- This is q38.c `attention_kernel:506`'s structure **plus** the causal `k <= t` bound, **minus**
  the `gather_kernel` pre-step, and per-lane over `d` instead of per-element.

**Complexity:** O(T² · 128) FLOPs, O(1) extra memory. For `T = 4096` at 64 layers this is the
M2 correctness baseline, not the final perf target.

---

## Cached suffix path (decode / continuation)

**Inputs:** `Q` tokens appended after `P` cached positions; cache already holds rows `0..P`.
**Outputs:** attn out for the new tokens + append of rows `P..P+Q`.

Two kernel variants, both NEW:

1. **Prefill-over-suffix** (`Q > 1`, chunked prompt / continuation):
   identical to the prefill kernel except the visible range is `j in [0, P + t]` (causal against the
   *global* position) and `rope` is applied at `position = P + t`.
2. **Decode** (`Q == 1`): grid = `(64,)`, one CTA per q_head; threads stream over the `P` cached
   rows. Same two-pass max→softmax; `max`/`den` reduced across the block. Same direct
   `kv_head = h >> 3`. No score buffer: full stream-and-reduce.

**KV cache layout (decision):**

```
dtype   : F16   (k and v).  FP32 accumulators for dot/softmax.
layout  : [layer][position][kv_head][head_dim]  =  [64][T][8][128]
row size: 8 * 128 = 1024 half = 2048 bytes   (16 KB per position, 64 layers)
```

This matches llama.cpp's `cache_k_l%d` shape `(n_embd_k_gqa = 8*128, kv_size)`
(`llama-kv-cache.cpp:232`) flattened per layer, and is the natural `[pos][kv_head][head_dim]`
row you get by appending the post-norm, post-RoPE `k` row directly.

**Append path:**

```
h2d:  none — k/v are produced on device by MMQ + norm + rope
d2d :  cudaMemcpyAsync(kv_k + layer*T*1024 + P*1024, k_new, Q*1024*2, D2D, stream)
       cudaMemcpyAsync(kv_v + layer*T*1024 + P*1024, v_new, Q*1024*2, D2D, stream)
```

A single `cudaMemcpyAsync` per tensor per layer per step — no kernel needed for the append
(the layout is exactly row-contiguous). `P` advances by `Q` after all 64 layers have appended.
No paged/block table, no slot mapping, no `llama_kv_cache_view` — a flat arena with a `count`.

**Why F16 and not F32:** halves cache footprint (`[64][40960][8][128]` = 20 GB F32 → 10 GB F16)
and F16 is what the FA/vec kernels consume; accuracy impact for M2 is acceptable and is validated
against the CPU/Python oracle. FP8 KV (q38-main) is deferred to a later milestone.

---

## GQA decision

- **Direct mapping, no KV repetition.** `kv_head = q_head * n_kv / n_q = q_head / 8`.
  All three donors agree: q38.c `attention_kernel:519` (`group = query_heads / kv_heads;
  kv_head = head / group`), q38.c `q38_qsa_ref.c`, and llama.cpp's `ncols2` GQA tile broadcast.
- **Do NOT materialize** a repeated `K`/`V` of `[T, 64, 128]`. The 8× memory traffic and the extra
  kernel are pure loss; the direct index is one integer divide (`>> 3` since 8 is a power of two).
- Prefill CTA indexing `(t, h)` with `h in [0,64)` already broadcasts naturally: heads
  `8k..8k+7` share the same `kv_head = k`.

---

## Workspace / scratch

| Item | Size | Notes |
|---|---|---|
| prefill per-CTA reduce buffer | 128 floats = 512 B | `__shared__` for max/sum across `head_dim` lanes |
| decode per-CTA reduce buffer | 128 floats = 512 B | same |
| persistent score tensor | **0** | causal loop bound + streaming → no `T × T` buffer |
| KV cache | `2 * 64 * T * 8 * 128 * 2 B` | F16, grow-by-`reserve` (double) or cap at ctx |
| per-step scratch | `head_dim` accumulators only | reused across layers |

**O(T²) persistent allocation: avoided.** q38-main *does* allocate `__shared__ float
scores[DS4_CUDA_ATTENTION_SCORE_CAP = 8192]` and caps visible rows (raw cap 256) — i.e. it avoids
O(T²) by **truncating context**, which M2 must not do (Qwen3-32B needs full 40960 ctx).
The M2 kernel avoids O(T²) by never materializing scores at all. A future flash-style online-softmax
formulation also avoids it and is the recommended optimization target once the oracle passes.

---

## CUDA arch / SM121 / GB10 dispatch notes

- Build flags in-tree: `Makefile:18 NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a`,
  `NVCCFLAGS := -O3 -g -lineinfo --use_fast_math -Isrc $(NVCC_ARCH_FLAGS)`.
  SM121 = compute capability 1210 = `GGML_CUDA_CC_DGX_SPARK` (`common.cuh:61`); `BLACKWELL` = 1200.
- **q38.c**: only `__CUDA_ARCH__ >= 700 / 750 / 800` guards. Nothing SM90+-specific, nothing that
  fails under `compute_121a`. Its kernels are plain SIMT — safe.
- **q38-main**: same story, `>= 700/750/800` only. Its flash-style heads8 kernel's only gate is a
  *runtime* `head_dim == 512` return, not an arch gate. Safe to build, but semantically wrong for us.
- **llama.cpp FA**: not a blocker architecturally — `fattn.cu:357` has an explicit
  `cc >= GGML_CUDA_CC_DGX_SPARK` branch, and `fattn-mma-f16.cuh:1863` only special-cases
  `CC_TURING`. The blocker is **runtime coupling** (`ggml` tensor types, `KQ_STRIDE=256` padding,
  `extra_data` alloc plumbing, `BEST_FATTN_KERNEL_{VEC,TILE,MMA_F16}` template dispatch), not arch.
- **M2 requirement:** the new attention/RoPE kernels must be plain SIMT with no arch guards, so they
  compile unconditionally under `compute_121a`. If a `cp.async`/`TMA` path is later added, guard it
  with `#if __CUDA_ARCH__ >= 900` and keep a SIMT fallback.
- `--use_fast_math` is already on: `expf`/`cosf`/`sinf` become fast intrinsics. Acceptable for M2
  given the oracle tolerance, but note it makes the softmax slightly non-reproducible across flags —
  keep the oracle tolerance explicit.

---

## Why FlashAttention is rejected for M2 (explicit answer)

**No, a FlashAttention-compatible kernel is not reasonably extractable.**

To lift `fattn-vec.cuh` / `fattn-tile.cu` / `fattn-mma-f16.cuh` into the q3 tree you would need to
port: `ggml_tensor` + `ggml_type` type system, `ggml_cuda_flash_attn_ext_f16_extra_data`,
`get_alloc_size` / `f16_extra` allocation negotiation, `KQ_STRIDE = 256` K-padding and its fixup,
the `BEST_FATTN_KERNEL_*` template selection and its `cc`/`gqa_ratio`/`can_use_vector_kernel`
predicate (`fattn.cu:760`), plus the `ncols2` GQA tile layout. That is materially more than "half a
runtime" — it is the ggml tensor/allocator/graph boundary.

**Decision: correct first.** M2 uses the simple two-pass max-softmax device kernel above. Once the
Python/CPU oracle proves token-exact agreement, a later milestone can introduce an online-softmax
flash variant (pattern available at `ds4_cuda.cu:10116`) without changing the cache layout or the
RoPE/norm kernels.

---

## LOC & file-count estimate

| File (new) | Contents | ~LOC |
|---|---|---|
| `cuda/q3_attention.cu` | `q3_rope_neox_kernel`, `q3_rmsnorm_heads_kernel`, `q3_attention_prefill_kernel`, `q3_attention_decode_kernel`, `q3_residual_add_kernel`, `q3_silu_kernel`, `q3_kv_append` (memcpy wrapper) | 380–520 |
| `src/q3_attention.h` | host-side entry points, geometry struct, cache descriptor | 90–130 |
| `cuda/q3_attention_kernels.h` | kernel decls / launch config constants | 40–60 |
| **subtotal new** | | **~510–710** |

| File (edited) | Change | ~LOC delta |
|---|---|---|
| `src/q3_cuda.h` | declare attention/rope/norm/cache entry points | +20 |
| `cuda/q3_cuda.cu` | Q6_K path for `attn_v` (or route to MMQ), wiring | +30 |
| `Makefile` | object rule + test target | +6 |
| `tests/test_q3_attention.cu` | CPU-vs-GPU sanity on a small GEMM/GQA case | +80 new |
| **subtotal edits** | | **~136** |

**Total ≈ 650–850 LOC across 3 new files + 3 edited files (+1 new test).** No other file is touched.

---

## Explicit answers to the required questions

1. **Does q38.c / q38-main already have Qwen3-compatible attention + RoPE?**
   **No attention.** q38.c = QSA sparse (top-k indexer + gather + dense-kernel attention, no causal
   mask). q38-main = windowed/compressed/MLA/sinks, `head_dim == 512` gated.
   **RoPE:** q38.c's layout **is** Qwen3-compatible (NeoX half-split) but hardcodes `theta = 1e7`;
   q38-main's is YARN, **not** compatible.
2. **Attention/rope/kv files in the donors?** q38.c: `cuda/q38_qsa_cuda.cu`,
   `q38_qsa.c`, `q38_qsa.h`, `q38_kvstore.h`, `q38_rope_ref.c/.h`, `q38_forward.c`.
   q38-main: `ds4_cuda.cu` (single ~28k-line file), no separate rope/attn files.
3. **Is a FA kernel extractable?** No — see the dedicated section. M2 picks the simple correct
   device-only kernel instead.
4. **RoPE layout required.** NeoX / `rotate_half`, `rotary_dims = head_dim = 128`, `theta = 1e6`,
   `freq_scale = 1`, no YARN, applied to Q and K after per-head RMSNorm, before attention.
   Matching donor: q38.c (fix `1e7 -> 1e6`, drop `sections`).
5. **Causal GQA — repeat or map?** **Map directly.** `kv_head = q_head / 8`; no KV repetition.
6. **KV cache.** **F16**, `[layer][position][kv_head][head_dim]` = `[64][T][8][128]`;
   append via one `cudaMemcpyAsync` D2D per tensor per layer, after `k_norm` + RoPE.
7. **Workspace.** No O(T²) persistent allocation; 512 B `__shared__` reduce buffer per CTA;
   streaming causal loop bound. Flash-style online softmax also avoids O(T²) and is the future path.
8. **SM121 dispatch.** No breaking SM90+ assumptions in q38.c/q38-main (only `>=700/750/800`);
   llama.cpp FA has an explicit `DGX_SPARK = 1210` branch. Plain SIMT, no arch guards needed.
9. **LOC:** ~650–850 LOC, 3 new files + 3 edited files (+1 test).

---

## COPY these files/symbols

Copy (adapting constants; no verbatim source is copied in this analysis):

```
COPY  _reference/q38.c/cuda/q38_qsa_cuda.cu : rope_kernel (:328)
        -> new q3_rope_neox_kernel ; CHANGE theta 1e7 -> 1e6 ; DROP sections[4] ;
           keep exact NeoX half-split indexing (a = base+p, b = base+rotary_dims/2+p)

COPY  _reference/q38.c/cuda/q38_qsa_cuda.cu : attention_kernel (:506)
        -> new q3_attention_prefill_kernel / q3_attention_decode_kernel ;
           keep 2-pass max -> exp -> divide, keep direct kv_head = head/(q/kv) ;
           ADD causal bound j <= t ; REMOVE gather_kernel dependency

COPY  _reference/q38.c/cuda/q38_cuda_primitives.cu : rms_norm_kernel (:94)
        -> new q3_rmsnorm_heads_kernel (per-head, over head_dim=128)

COPY  _reference/q38.c/cuda/q38_cuda_primitives.cu : silu_kernel (:120)
        -> q3_silu_kernel (FFN path; same style as M1 primitives)

COPY  _reference/q38.c/q38_forward.c : rope() (:92)
        -> SEMANTICS ONLY (CPU oracle reference for NeoX half-split)

COPY  _reference/q38.c/q38_forward.c : ordering at :248-260 and :348-355
        -> SEMANTICS ONLY (norm(q,k) per head -> rope(q,k) -> attention -> o_proj -> residual)

COPY  _reference/llama.cpp/src/models/qwen3.cpp : 60-159
        -> SEMANTICS ONLY (authoritative Qwen3 layer graph, scale 1/sqrt(128))

COPY  _reference/llama.cpp/src/llama-model.cpp : rope-type switch
        -> FACT ONLY (QWEN3 -> LLAMA_ROPE_TYPE_NEOX)

COPY  _reference/llama.cpp/src/llama-kv-cache.cpp : 232-237
        -> FACT ONLY (KV layout ([n_head_kv*head_dim], kv_size) per layer)

COPY  _reference/llama.cpp/ggml/src/ggml-cuda/common.cuh : 60-61 (as a comment/fact)
        -> GGML_CUDA_CC_BLACKWELL=1200, GGML_CUDA_CC_DGX_SPARK=1210
```

## EDIT these dependencies

```
EDIT  _reference/q38.c/q38_forward.c :80 rms(..., bool one_plus)
      -> when porting: DROP one_plus ; use weight directly (Qwen3 semantics).
      q38.c calls rms(q, w->q_norm, head_dim, true) at :347 -> (1+weight) is Qwen3.8-only.

EDIT  _reference/q38.c/cuda/q38_qsa_cuda.cu : rope_kernel theta
      -> 10000000.0f  ==>  1000000.0f  (from GGUF rope.freq_base); make theta an argument.

EDIT  _reference/q38.c attention kernel sqrtf((float)head_dim) inline scaling
      -> hoist to a `scale` argument = 1/sqrtf(128) to match llama.cpp exactly.

EDIT  in-tree src/q3_cuda.h / cuda/q3_cuda.cu
      -> add entry points: q3_rope_neox, q3_rmsnorm_heads, q3_attention_prefill,
         q3_attention_decode, q3_kv_cache_{alloc,append,reset}. Keep existing API untouched.

EDIT  in-tree Makefile
      -> add cuda/q3_attention.o to objects; new test_q3_attention target under
         NVCC_ARCH_FLAGS (-gencode arch=compute_121a,code=sm_121a).

EDIT  cuda/q3_cuda.cu (Q6_K path)
      -> attn_v.weight is q6_k, not q4_k: either extend the dequant path or route v through
         the MMQ Q4_K-family entry; choose at M2 wiring time. Flagged, not solved here.

EDIT  proposed new cuda/q3_attention.cu
      -> replace F32 KV with F16 KV + F32 accumulate; replace per-element grid with
         per-(token,head) grid + per-lane head_dim; add causal bound; add D2D append helper.
```

## DELETE these paths

```
DELETE  QSA sparse-attention machinery (all of it):
          _reference/q38.c/cuda/q38_qsa_cuda.cu : gather_kernel, index_scores_kernel,
                                                    chain_select_kernel, chain_append_kernel
          _reference/q38.c/q38_qsa_ref.c         : top-k block indexer / candidate selection
          _reference/q38.c/q38_qsa.h             : q38_qsa_cache (untyped double/triple cache)
          _reference/q38.c/q38_kvstore.h         : row-bytes cache abstraction
          _reference/q38.c/q38_rope_ref.c/.h     : QSA sections/yarn rope config
        Rationale: Qwen3 has no indexer, no budget, no ratio, no sections.

DELETE  q38-main attention variants (wrong algorithm for Qwen3):
          ds4_cuda.cu : attention_prefill_raw_kernel, attention_prefill_raw_softmax_kernel,
                        attention_decode_mixed_kernel, attention_static_mixed_heads8_online_kernel
                        (KEEP ONLY AS A FUTURE-OPTIMIZATION REFERENCE, do not port),
                        all attention_indexed_*, attention_decode_splitkv*, attention_tokentile_*,
                        per-head sink / compressed-KV / ratio-pooling / use_comp_mask paths.
          Rationale: windowed/compressed/MLA + head_dim==512 hard gate; truncates context.

DELETE  q38-main RoPE + quantized KV:
          ds4_cuda.cu : rope_yarn_ramp_dev, rope_tail_kernel, fp8 KV quant/dequant kernels,
                        variable-resolution and compressed cache tables.
          Rationale: Qwen3 uses plain NeoX RoPE at theta 1e6; FP8 KV is out of M2 scope.

DELETE  llama.cpp FlashAttention runtime coupling (do NOT vendor):
          ggml/src/ggml-cuda/fattn-vec.cuh, fattn-tile.cu, fattn-mma-f16.cuh,
          fattn.cu dispatch (BEST_FATTN_KERNEL_VEC/TILE/MMA_F16, fattn.cu:172/357/760),
          ggml_cuda_flash_attn_ext_f16_extra_data, get_alloc_size / f16_extra plumbing,
          FATTN_KQ_STRIDE = 256 padding, ncols2 GQA tile machinery,
          ggml-cuda/rope.cu yarn/corr_dims plumbing.
          Rationale: drags the ggml tensor/type/allocator/graph boundary = more than half a runtime.

DELETE  general runtime (never in scope for M2 attention work):
          paged/slot KV block tables and llama_kv_cache_view,
          graph scheduler, multi-GPU / pipeline split, server/batch/cont-batching,
          tokenizer, sampling, any model-arch other than qwen3.
```

**Chosen M2 attention kernel:** a **simple, device-only, correct causal GQA kernel** —
one CTA per `(query_token, q_head)` for the prefill path with the causal loop bound `j <= t`,
one CTA per `q_head` streaming the KV cache for the cached-suffix/decode path; two-pass
max→exp→normalize softmax in FP32; direct `kv_head = q_head / 8` mapping with **no KV repetition**;
zero O(T²) allocation. A FlashAttention-style online-softmax variant is deferred until the
correctness oracle passes.
