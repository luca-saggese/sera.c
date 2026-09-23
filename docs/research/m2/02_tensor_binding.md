# M2 Worker B — Tensor binding (Qwen3-32B Q4_K_M)

Source of truth: `models/Qwen3-32B-Q4_K_M.gguf`, parsed with the M0/M1 binary
(`./q3 --list-tensors ... --json`), 707 tensors. **No tensor data was loaded.**

- GGUF: v3, arch `qwen3`, 28 metadata keys, 707 tensors, alignment 32
- tensor data starts at abs offset **5974688**; file 19762149024 bytes
- total tensor bytes: **19756174336**
- logical parameters: 32762123264
- geometry from metadata: block_count 64, embedding_length 5120,
  feed_forward_length 25600, head_count 64, head_count_kv 8, key/value_length 128,
  rope.freq_base 1e6, rms eps 1e-6, context 40960, vocab 151936

## 1. Resident tensor ID vs resident descriptor index

Two distinct things must not be conflated:

- **`tensor_id`** (`q3_resident_tensor.tensor_id`, `q3_model_loader_cuda.h:84`)
  is set in `cuda/q3_model_loader_cuda.cu:227` as
  `entry->tensor_id = (uint32_t)i;` where `i` is the **GGUF tensor index**,
  i.e. the position in `q3_gguf.tensors[]`, which is the on-disk order.
  The descriptor table is `q3_loader_tensor(ctx, index)` indexed by the same
  GGUF index. So in M2, **resident tensor ID == GGUF tensor index**.
- `q3_residency_plan_entry.tensor_index` is also that GGUF index, but the plan
  is the offset-ordered *allocation* order; the JSON confirms
  `offset-sorted order == index order`, so plan order and index order coincide
  for this file (707 planned tensors, 88 spans, 0 excluded PLE bytes).

GGUF index table (identical in every layer, verified for all 64 layers):

| GGUF idx | slot suffix |
|---|---|
| 3 + 11*N + 0 | `attn_k.weight` |
| 3 + 11*N + 1 | `attn_k_norm.weight` |
| 3 + 11*N + 2 | `attn_norm.weight` |
| 3 + 11*N + 3 | `attn_output.weight` |
| 3 + 11*N + 4 | `attn_q.weight` |
| 3 + 11*N + 5 | `attn_q_norm.weight` |
| 3 + 11*N + 6 | `attn_v.weight` |
| 3 + 11*N + 7 | `ffn_down.weight` |
| 3 + 11*N + 8 | `ffn_gate.weight` |
| 3 + 11*N + 9 | `ffn_norm.weight` |
| 3 + 11*N + 10 | `ffn_up.weight` |

Global slots: `output.weight` = 0, `output_norm.weight` = 1,
`token_embd.weight` = 2, `blk.63.ffn_up.weight` = 706 (last).

The binder must still resolve by **name → index** at startup (one linear scan of
`model->tensors[i].name_buf`), then freeze the index. Hot path uses the index only.

## 2. Naming convention — confirmed against the real inventory

This GGUF uses the canonical llama.cpp qwen3 names. There are exactly 19 distinct
tensor names (11 per-block suffixes + 3 global), and every layer 0..63 carries
exactly the same 11 suffixes (verified: the per-layer suffix set is identical
across all 64 layers). Full names for layer N:

| logical weight | GGUF tensor name (N = layer) |
|---|---|
| embedding | `token_embd.weight` |
| input_layernorm | `blk.N.attn_norm.weight` |
| q_proj | `blk.N.attn_q.weight` |
| q_norm | `blk.N.attn_q_norm.weight` |
| k_proj | `blk.N.attn_k.weight` |
| k_norm | `blk.N.attn_k_norm.weight` |
| v_proj | `blk.N.attn_v.weight` |
| o_proj | `blk.N.attn_output.weight` |
| post_attention_layernorm | `blk.N.ffn_norm.weight` |
| gate_proj | `blk.N.ffn_gate.weight` |
| up_proj | `blk.N.ffn_up.weight` |
| down_proj | `blk.N.ffn_down.weight` |
| final_norm | `output_norm.weight` |
| output / lm_head | `output.weight` |

No `blk.N.attn_*_bias` and no `blk.N.*.bias` exist anywhere (Qwen3 has no biases).
No MoE tensors (`ffn_gate_inp`, `ffn_gate_exps`, ...) exist: dense FFN only.
No `rope_freqs.weight`; RoPE uses `qwen3.rope.freq_base = 1000000.0` metadata only.

## 3. qtype histogram (full)

| qtype | tensors | bytes |
|---|---|---|
| `q4_k` | 385 | 15537070080 |
| `q6_k` | 65  | 4216396800 |
| `f32`  | 257 | 2707456 |
| **total** | **707** | **19756174336** |

f32 (257 tensors) = 64*`blk.N.attn_norm` + 64*`blk.N.ffn_norm` +
64*`blk.N.attn_q_norm` + 64*`blk.N.attn_k_norm` + 1*`output_norm.weight`
= 256 + 1. All of them are 1-D; there is no f32 2-D tensor at all
(verified: every name containing `norm` is `f32`, and every `f32` tensor is a norm).

### Non-Q4_K tensors (complete list)

`q6_k` (65):
- `output.weight`
- `blk.N.attn_v.weight` for N in **{0,1,2,3,4,5,6,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,56,57,58,59,60,61,62,63}** (32 layers)
- `blk.N.ffn_down.weight` for **exactly the same 32 layers** (v_proj and down_proj
  carry the identical type in every layer — verified for all 64)

`f32` (257): the 257 1-D norm tensors listed above (`blk.N.attn_norm.weight`,
`blk.N.ffn_norm.weight`, `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight`,
`output_norm.weight`).

There are **no** F16, BF16, Q8_0, Q5_K, Q4_0/Q4_1, IQ*, MXFP4/NVFP4, Q8_1-weight
or Q2_K tensors. `q3_gguf.native_nvfp4` is false for this file.

## 4. UNSUPPORTED qtypes — flag to main agent

M1's production backend is **Q4_K dense only** (`src/q3_q4_linear.h`,
`cuda/q3_mmq.h`: "dense only, Q4_K weights only"). `cuda/q3_mmq.h` explicitly
states there are *no* `Q2_K/IQ2_XXS/Q8_0` weight paths; `cuda/q3_cuda_primitives.cu`
only has `dequant_q4_k_kernel`, and `src/q3_quant.h` says it is "trimmed to the
Q4_K layout".

| qtype | needed by | status in M1 | action |
|---|---|---|---|
| `q4_k` | 385 tensors (all q/k/o/gate/up, embedding, and v/down on 32 layers) | supported | production `q3_cuda_q4k_linear` (MMVQ small M / MMQ larger M) |
| `q6_k` | **65 tensors**: `output.weight` (638131200 B, largest tensor), 32x `attn_v`, 32x `ffn_down` | **UNSUPPORTED** | **M2 must add a Q6_K quantized-matmul primitive** |
| `f32` | 257 1-D norms | N/A — no quantized matmul | plain elementwise RMSNorm input; needs no new matmul primitive |

**Immediate blocker:** Q6_K is not a corner case here. It covers 4216396800 bytes
(21.3% of the model) and it is unavoidable in the decode path:
- `output.weight` (lm_head) is Q6_K, and it is **not** tied to the Q4_K embedding,
  so the final projection is a genuine Q6_K matmul `[151936, 5120]`.
- 32 of 64 layers use Q6_K for both `attn_v.proj` and `down_proj`
  ({0..7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55..63}),
  i.e. Q4_K cannot even be assumed for half the layers.

Per `docs/M2.md` §8 ("Se alcuni piccoli tensor sono F32/BF16/Q6_K/Q8_0, portare
soltanto il primitive minimo necessario a quel tensor reale") the required
minimal addition is **one Q6_K x FP32 → FP32 dense primitive** (donor:
llama.cpp/donor `dequantize_row_q6_K` + the Q6_K MMVQ/MMQ variant), dispatched on
`qtype` per weight. `f32` needs nothing new. A silent scalar fallback is
forbidden. No other unsupported qtype exists in this file.

Dispatch keyed by qtype must therefore be per-tensor, not per-model:
`attn_v` and `ffn_down` are Q6_K in some layers and Q4_K in others, so the layer
descriptor cannot assume a fixed type for those two slots.

## 5. Shapes of every needed tensor

GGML order is `dim[0] = ne0 = in_features` (contiguous), `dim[1] = ne1 = out_features`
(row count). Logical order is `(out, in)`.
Note `q3_resident_tensor.rows/cols` (from `tensor_shape()` in
`q3_model_loader_cuda.cu:89`) collapse `dim[0..ndim-2]` into `rows` and take the
**last** dim as `cols`; for these 2-D tensors that means `rows = ne0 = K` and
`cols = ne1 = N`. `q3_q4_linear_bind_geometry(qtype, gguf_rows_dim0, gguf_cols_dim1, ...)`
is the single sanctioned inversion point — pass the descriptor's `rows` as
`gguf_rows_dim0` (really K) and `cols` as `gguf_cols_dim1` (really N).

| logical weight | name pattern | qtype | GGML shape | logical (out=N, in=K) | bytes |
|---|---|---|---|---|---|
| embedding (lookup) | `token_embd.weight` | q4_k | `[5120, 151936]` | `(151936, 5120)` | 437575680 |
| output / lm_head | `output.weight` | **q6_k** | `[5120, 151936]` | `(151936, 5120)` | 638131200 |
| final_norm | `output_norm.weight` | f32 | `[5120]` | 1-D, 5120 | 20480 |
| input_layernorm | `blk.N.attn_norm.weight` | f32 | `[5120]` | 1-D, 5120 | 20480 |
| q_proj | `blk.N.attn_q.weight` | q4_k | `[5120, 8192]` | `(8192, 5120)` | 23592960 |
| q_norm | `blk.N.attn_q_norm.weight` | f32 | `[128]` | 1-D, 128 (per head) | 512 |
| k_proj | `blk.N.attn_k.weight` | q4_k | `[5120, 1024]` | `(1024, 5120)` | 2949120 |
| k_norm | `blk.N.attn_k_norm.weight` | f32 | `[128]` | 1-D, 128 (per KV head) | 512 |
| v_proj | `blk.N.attn_v.weight` | q4_k/q6_k | `[5120, 1024]` | `(1024, 5120)` | 2949120 / 4300800 |
| o_proj | `blk.N.attn_output.weight` | q4_k | `[8192, 5120]` | `(5120, 8192)` | 23592960 |
| post_attention_layernorm | `blk.N.ffn_norm.weight` | f32 | `[5120]` | 1-D, 5120 | 20480 |
| gate_proj | `blk.N.ffn_gate.weight` | q4_k | `[5120, 25600]` | `(25600, 5120)` | 73728000 |
| up_proj | `blk.N.ffn_up.weight` | q4_k | `[5120, 25600]` | `(25600, 5120)` | 73728000 |
| down_proj | `blk.N.ffn_down.weight` | q4_k/q6_k | `[25600, 5120]` | `(5120, 25600)` | 73728000 / 107520000 |

Shape/type uniformity verified across all 64 layers: exactly one distinct
`(qtype, shape, bytes)` per suffix except `attn_v` and `ffn_down` which have two
(q4_k and q6_k) and always the same choice within a layer. All K extents
(5120, 8192, 25600) are multiples of 256, and the Q4_K block is 256 elems /
144 bytes; every 2-D tensor size is exactly `elems/256*144` (Q4_K) or `elems/256*210`
(Q6_K) — confirmed against the byte counts above.

Per-layer geometry summary (identical for every layer):

- attn: q_proj (8192, 5120), k_proj (1024, 5120), v_proj (1024, 5120),
  o_proj (5120, 8192) — 64 heads x 128, 8 KV heads x 128 = GQA 8:1
- mlp: gate (25600, 5120), up (25600, 5120), down (5120, 25600)
- norms: attn_norm 5120, ffn_norm 5120, q_norm 128, k_norm 128 (all f32, 1-D)

## 6. Embedding / lm_head tie check

`token_embd.weight` (index 2, q4_k) and `output.weight` (index 0, **q6_k**) are
**both present and distinct tensors**: different names, different GGUF indices,
different abs offsets (644126368 vs 5974688), different qtypes, and different
byte sizes (437575680 vs 638131200). **Embeddings are NOT tied** in this GGUF;
the binder must bind two independent weights and must not alias them.

## 7. Concrete mapping table

Resident tensor ID = GGUF tensor index. `qtype`/`shape` are the real per-file values.
Rows are collapsed per layer (all 64 layers are structurally identical); the
`id` shown uses the formula from §1 (N from 0 to 63). Bytes are stated once.

Global:

| logical | GGUF name | qtype | GGML shape | logical (out,in) | resident ID |
|---|---|---|---|---|---|
| embedding | `token_embd.weight` | q4_k | [5120, 151936] | (151936, 5120) | 2 |
| final_norm | `output_norm.weight` | f32 | [5120] | 1-D | 1 |
| output / lm_head | `output.weight` | q6_k | [5120, 151936] | (151936, 5120) | 0 |

Per layer N (0..63):

| logical | GGUF name | qtype | GGML shape | logical (out,in) | resident ID |
|---|---|---|---|---|---|
| input_layernorm | `blk.N.attn_norm.weight` | f32 | [5120] | 1-D | 3 + 11N + 2 |
| q_proj | `blk.N.attn_q.weight` | q4_k | [5120, 8192] | (8192, 5120) | 3 + 11N + 4 |
| q_norm | `blk.N.attn_q_norm.weight` | f32 | [128] | 1-D | 3 + 11N + 5 |
| k_proj | `blk.N.attn_k.weight` | q4_k | [5120, 1024] | (1024, 5120) | 3 + 11N + 0 |
| k_norm | `blk.N.attn_k_norm.weight` | f32 | [128] | 1-D | 3 + 11N + 1 |
| v_proj | `blk.N.attn_v.weight` | q4_k or q6_k | [5120, 1024] | (1024, 5120) | 3 + 11N + 6 |
| o_proj | `blk.N.attn_output.weight` | q4_k | [8192, 5120] | (5120, 8192) | 3 + 11N + 3 |
| post_attention_layernorm | `blk.N.ffn_norm.weight` | f32 | [5120] | 1-D | 3 + 11N + 9 |
| gate_proj | `blk.N.ffn_gate.weight` | q4_k | [5120, 25600] | (25600, 5120) | 3 + 11N + 8 |
| up_proj | `blk.N.ffn_up.weight` | q4_k | [5120, 25600] | (25600, 5120) | 3 + 11N + 10 |
| down_proj | `blk.N.ffn_down.weight` | q4_k or q6_k | [25600, 5120] | (5120, 25600) | 3 + 11N + 7 |

Spot checks against the raw inventory (all 64 layers verified programmatically):

| N | v_proj qtype | down_proj qtype | id(v_proj) | id(down_proj) |
|---|---|---|---|---|
| 0  | q6_k | q6_k | 9  | 10 |
| 1  | q6_k | q6_k | 20 | 21 |
| 2  | q6_k | q6_k | 31 | 32 |
| 7  | q6_k | q6_k | 86 | 87 |
| 8  | q4_k | q4_k | 97 | 98 |
| 10 | q6_k | q6_k | 119 | 120 |
| 32 | q4_k | q4_k | 361 | 362 |
| 55 | q6_k | q6_k | 614 | 615 |
| 56 | q6_k | q6_k | 625 | 626 |
| 63 | q6_k | q6_k | 702 | 703 |

## 8. Recommended C struct layout — covers every needed tensor

The layout proposed in `docs/M2.md` §7 is sufficient **provided** each
`q3_exec_tensor` carries its own `qtype` (needed because `v_proj` / `down_proj`
vary by layer) and, for the f32 norms, a host/device pointer plus length. Sketch
(types only, no implementation):

```c
typedef struct {
    void    *data;      /* resident device pointer (NULL if not resident) */
    uint64_t bytes;
    uint32_t qtype;     /* real GGUF qtype: q4_k / q6_k / f32 */
    uint32_t tensor_id; /* GGUF index == q3_resident_tensor.tensor_id */
    int32_t  N, K;      /* logical matmul geometry; N=K=0 for 1-D */
} q3_exec_tensor;

typedef struct {
    q3_exec_tensor input_norm;      /* blk.N.attn_norm.weight          f32 [5120] */
    q3_exec_tensor q_proj;          /* blk.N.attn_q.weight             q4_k 8192x5120 */
    q3_exec_tensor q_norm;          /* blk.N.attn_q_norm.weight        f32 [128]  */
    q3_exec_tensor k_proj;          /* blk.N.attn_k.weight             q4_k 1024x5120 */
    q3_exec_tensor k_norm;          /* blk.N.attn_k_norm.weight        f32 [128]  */
    q3_exec_tensor v_proj;          /* blk.N.attn_v.weight             q4_k|q6_k 1024x5120 */
    q3_exec_tensor o_proj;          /* blk.N.attn_output.weight        q4_k 5120x8192 */
    q3_exec_tensor post_attn_norm;  /* blk.N.ffn_norm.weight           f32 [5120] */
    q3_exec_tensor gate_proj;       /* blk.N.ffn_gate.weight           q4_k 25600x5120 */
    q3_exec_tensor up_proj;         /* blk.N.ffn_up.weight             q4_k 25600x5120 */
    q3_exec_tensor down_proj;       /* blk.N.ffn_down.weight           q4_k|q6_k 5120x25600 */
} q3_layer_weights;

typedef struct {
    q3_exec_tensor embedding;       /* token_embd.weight   q4_k 151936x5120 (lookup) */
    q3_layer_weights layer[64];
    q3_exec_tensor final_norm;      /* output_norm.weight  f32 [5120] */
    q3_exec_tensor output;          /* output.weight       q6_k 151936x5120 */
} q3_weights;
```

Coverage check: 1 + 64*11 + 2 = **707** slots, i.e. exactly one slot per GGUF
tensor, with no leftovers. No bias, no MoE, no rope-freqs, no per-head table is
needed. Every tensor in this file is bound; anything unbound at startup is a
hard error, and any `qtype != q4_k && qtype != q6_k` for a 2-D weight (or any
non-f32 norm) must fail loudly.

Notes for the binder implementation (not done here):
- `q3_exec_tensor.N/K` must come from the sanctioned inversion in
  `q3_q4_linear_bind_geometry()` — GGUF `dim[0]` is K, `dim[1]` is N; the M0
  descriptor exposes them as `rows = K`, `cols = N`.
- The f32 norms are elementwise; they need a device pointer and length only.
- `q6_k` slots require the missing primitive from §4 before the binder can run
  end-to-end; the binding table itself is independent of that primitive.
