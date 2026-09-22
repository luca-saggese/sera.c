# Worker C — Real Q4_K Probe (M1)

Source of truth: `cd /home/lvx/sera.c && ./q3 --list-tensors models/Qwen3-32B-Q4_K_M.gguf --json`
(707 entries: name, type, ndim, shape, elements, bytes, rel_offset, abs_offset).
File: `models/Qwen3-32B-Q4_K_M.gguf`, 19,762,149,024 B, GGUF v3, alignment 32, `qwen3`, 64 blocks, 5120 embd.

QK_K = 256 (`_reference/llama.cpp/ggml/src/ggml-common.h:89`).
`sizeof(block_q4_K) = 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128 = 144`
(`ggml-common.h:326-338`; `K_SCALE_SIZE=12` at `:90`). Q6_K = 210 B.
**GGUF dim convention (verified):** `dim[0] = ne[0] = K` (contiguous row length), `dim[1] = ne[1] = N` (rows).
Determined by pairing `ffn_gate` shape `[5120,25600]` (hidden 5120 -> intermediate 25600 => N=25600, K=5120)
against `ffn_down` `[25600,5120]` (intermediate -> hidden => N=5120, K=25600). Note the loader's
`q3_resident_tensor.rows = dim[0] = K` and `.cols = dim[1] = N` (see task 5 caveat).

## 1. Tensor name patterns

| pattern | qtype | GGUF shape | elements | bytes | layout comment |
|---|---|---|---|---|---|
| `token_embd.weight` | q4_k | [5120,151936] | 777,912,320 | 437,575,680 | vocab rows N=151936, K=5120; q4_k |
| `output.weight` | **q6_k** | [5120,151936] | 777,912,320 | 638,131,200 | LM head; **NOT f32 — q6_k**, largest tensor |
| `output_norm.weight` | f32 | [5120] | 5,120 | 20,480 | 1-D, n/a |
| `blk.N.attn_norm.weight` | f32 | [5120] | 5,120 | 20,480 | 1-D, all 64 layers |
| `blk.N.attn_q.weight` | q4_k | [5120,8192] | 41,943,040 | 23,592,960 | K=5120, N=8192 (64 heads x128) |
| `blk.N.attn_q_norm.weight` | f32 | [128] | 128 | 512 | per-head norm |
| `blk.N.attn_k.weight` | q4_k | [5120,1024] | 5,242,880 | 2,949,120 | K=5120, N=1024 (8 KV heads x128) |
| `blk.N.attn_k_norm.weight` | f32 | [128] | 128 | 512 | per-head norm |
| `blk.N.attn_v.weight` | **MIXED q6_k/q4_k** | [5120,1024] | 5,242,880 | 4,300,800 (q6) / 2,949,120 (q4) | 32 layers each |
| `blk.N.attn_output.weight` | q4_k | [8192,5120] | 41,943,040 | 23,592,960 | o-proj; K=8192, N=5120 |
| `blk.N.ffn_gate.weight` | q4_k | [5120,25600] | 131,072,000 | 73,728,000 | K=5120, N=25600; all 64 layers q4_k |
| `blk.N.ffn_up.weight` | q4_k | [5120,25600] | 131,072,000 | 73,728,000 | same shape as gate; all 64 q4_k |
| `blk.N.ffn_down.weight` | **MIXED q6_k/q4_k** | [25600,5120] | 131,072,000 | 107,520,000 (q6) / 73,728,000 (q4) | 32 layers each |
| `blk.N.ffn_norm.weight` | f32 | [5120] | 5,120 | 20,480 | 1-D |

Per-layer counts: f32 = 257, q4_k = 385, q6_k = 65 (matches M0 inventory).

## 2. Q4_K families (physical, summed over all layers)

| family | qtype(s) | logical N,K | rows (layers) | cols (per layer) | elements | physical bytes |
|---|---|---|---|---|---|---|
| `blk.N.attn_q.weight` | q4_k only | N=8192, K=5120 | 64 | 8192 | 2,684,354,560 | 1,509,949,440 |
| `blk.N.attn_output.weight` | q4_k only | N=5120, K=8192 | 64 | 5120 | 2,684,354,560 | 1,509,949,440 |
| `blk.N.attn_k.weight` | q4_k only | N=1024, K=5120 | 64 | 1024 | 335,544,320 | 188,743,680 |
| `blk.N.attn_v.weight` | **MIXED** | N=1024, K=5120 | q4_k=32 / q6_k=32 | 1024 | q4:167,772,160 / q6:167,772,160 | q4:94,371,840 / q6:129,024,000 |
| `blk.N.ffn_gate.weight` | q4_k only | N=25600, K=5120 | 64 | 25600 | 8,388,608,000 | 4,718,592,000 |
| `blk.N.ffn_up.weight` | q4_k only | N=25600, K=5120 | 64 | 25600 | 8,388,608,000 | 4,718,592,000 |
| `blk.N.ffn_down.weight` | **MIXED** | N=5120, K=25600 | q4_k=32 / q6_k=32 | 5120 | q4:2,097,152,000 / q6:2,097,152,000 | q4:1,179,648,000 / q6:1,610,612,736 |
| `token_embd.weight` | q4_k | N=151936, K=5120 | 1 | 151936 | 777,912,320 | 437,575,680 |

**MIXED verification (self-checked against JSON, matches M0):**
- `attn_v.weight` and `ffn_down.weight` are the *only* mixed families.
- q4_k layers = `[8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54]`
- q6_k layers = `[0,1,2,3,4,5,6,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,56,57,58,59,60,61,62,63]`
  - pattern: layer 0-7 q6_k, then q6_k every 3rd from 10..55, plus 56-63 q6_k; 32 q4_k / 32 q6_k in both families.

## 3. Acceptance-target selection

All Q4_K candidates have `K % 256 == 0`, so no awkward K. Ranking by size + representativeness:

| candidate | qtype | GGUF shape | N,K | elements | bytes | K | K%256 |
|---|---|---|---|---|---|---|---|
| **`blk.0.ffn_gate.weight` (CHOSEN)** | q4_k | [5120,25600] | N=25600, K=5120 | 131,072,000 | 73,728,000 | 5120 | 0 |
| `ffn_up.weight` (backup 1) | q4_k | [5120,25600] | N=25600, K=5120 | 131,072,000 | 73,728,000 | 5120 | 0 |
| `attn_output.weight` (backup 2) | q4_k | [8192,5120] | N=5120, K=8192 | 41,943,040 | 23,592,960 | 8192 | 0 |
| `attn_q.weight` | q4_k | [5120,8192] | N=8192, K=5120 | 41,943,040 | 23,592,960 | 5120 | 0 |
| `ffn_down.weight` (blk.8+) | q4_k | [25600,5120] | N=5120, K=25600 | 131,072,000 | 73,728,000 | 25600 | 0 |
| `token_embd.weight` | q4_k | [5120,151936] | N=151936, K=5120 | 777,912,320 | 437,575,680 | 5120 | 0 |
| `attn_k.weight` | q4_k | [5120,1024] | N=1024, K=5120 | 5,242,880 | 2,949,120 | 5120 | 0 |
| `output.weight` | q6_k (not q4_k!) | [5120,151936] | — | — | — | — | — |
| `attn_v.weight` (blk.8+) | q4_k | [5120,1024] | N=1024, K=5120 | 5,242,880 | 2,949,120 | 5120 | 0 |

**Why `blk.0.ffn_gate.weight`:** genuinely q4_k in **all 64 layers** (no mixed-family exception), the largest
q4_k projection family (4.72 GB), representative FFN matrix, K=5120=20*256 gives a clean block count, and it
shares shape with `ffn_up` (independent cross-check). `output.weight` was initially suspected f32 — it is
**q6_k** (638,131,200 B = 3,038,720 blocks * 210) and therefore excluded.

## 4. Chosen tensor — exact numbers

| field | value |
|---|---|
| tensor name | `blk.0.ffn_gate.weight` |
| qtype | `q4_k` (GGUF type id 12) |
| GGUF physical dims | `ne[0]=5120`, `ne[1]=25600` (shape `[5120,25600]`) |
| logical rows/cols | weight=[N,K] => **N=25600, K=5120** (output=[M,N]) |
| elements | 131,072,000 |
| physical bytes | 73,728,000 |
| K % 256 | 0 (5120 = 20*256) |
| row stride bytes | 20 * 144 = **2880** |
| Q4_K blocks per row | K/QK_K = 5120/256 = **20** |
| total blocks | 131,072,000/256 = 512,000 |
| N determination | `dim[0]=5120=K` (contiguous), `dim[1]=25600=N`; validated vs `ffn_down [25600,5120]` and known hidden=5120 / intermediate=25600 |

## 5. Resident device pointer availability (static confirmation)

Declared in `src/q3_model_loader_cuda.h`:
- `typedef struct { const void *host; const void *ptr; uint64_t bytes; uint32_t rows; uint32_t cols; uint32_t qtype; uint32_t tensor_id; uint64_t gguf_offset; const char *name; bool resident; } q3_resident_tensor;`
- accessors `size_t q3_loader_tensor_count(...)`, `const q3_resident_tensor *q3_loader_tensor(ctx, index)` (index = GGUF tensor index).

`cuda/q3_model_loader_cuda.cu`:
- descriptor table (lines ~209-230) fills `host=gguf_tensor_data`, `bytes`, `rows`/`cols` via `tensor_shape()`,
  `qtype=tensor->type`, `tensor_id=i`, `gguf_offset=tensor->abs_offset`, `name` for **all 707** tensors.
- resident loop (lines ~232-261) does `cudaMalloc(&device, tensor->bytes)`, sets
  `context->tensors[tensor_index].ptr = device` and `.resident = true` for every planned tensor.
- `tensor_shape()` (lines 89-104): `rows = prod(dim[0..ndim-2]) = dim[0]`, `cols = dim[ndim-1] = dim[1]`
  => **for this tensor `rows=5120` (=K) and `cols=25600` (=N); the field names are K/N, not N/K.**
- `--memory-plan` reports `planned_tensors=707`, `plan_span_bytes == resident_bytes == 19,756,174,336`,
  `coverage_ok` enforced in `src/q3_main.c:482-500` -> every tensor including `blk.0.ffn_gate.weight` is resident.

For `blk.0.ffn_gate.weight`: `resident == true`, `ptr != NULL` (cudaMalloc of 73,728,000 B),
`gguf_offset == abs_offset` (1,243,679,392), `rows == 5120`, `cols == 25600`, `qtype == 12 (q4_k)`, `bytes == 73728000`.

## 6. Deterministic small-input recipe

Reference (FP32 accumulate, no RNG):
```
for k in 0..K-1:
    x[k] = sinf(0.017f * k)      // input vector, K=5120
    w[k] = cosf(0.017f * k)      // column of W to dot against
dot = sum_k x[k]*w[k]            // split-sum in FP32
```
Closed form: `dot = 0.5 * sum_{k=0}^{K-1} sin(0.034*k)`, bounded by `1/(2*sin(0.017)) ≈ 29.42`;
for K=5120 it is O(1) and exactly reproducible, so a tolerance of `1e-3` relative is meaningful
(FP32 accumulation error grows ~ `eps*sqrt(K)*|sum|`).
Broader reference scale: for two unit-RMS length-K vectors the dot-product RMS is `sqrt(K) = sqrt(5120) ≈ 71.55`,
so any dequantized-GEMM sanity check should expect result magnitude ~70 for generic data and ~O(1) for the
deterministic sin/cos recipe above.

## 7. Q4_K block-byte math verification (against the real file)

Assumed per-block size `144 B = 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 = 4+12+128`
(`_reference/llama.cpp/ggml/src/ggml-common.h:326-338`; `QK_K=256` at `:89`; `K_SCALE_SIZE=12` at `:90`;
donor `q38_quant.h:19` has the matching `Q38_QUANT_QK_K 256`).
Checked **every** q4_k tensor: `bytes == (elements/256)*144` for all 385 q4_k tensors — **zero mismatches**.
Also checked all 65 q6_k: `bytes == (elements/256)*210` — zero mismatches.
No tensor in the file fails the block-byte math; any M1 layout mismatch would come from code, not the file.

## M1 acceptance tensor

**`blk.0.ffn_gate.weight` — q4_k, N=25600, K=5120, 131,072,000 elements, 73,728,000 physical bytes,
row stride 2880 B, 20 Q4_K blocks/row, 512,000 total blocks, K%256=0.**

## Backups

1. `blk.0.ffn_up.weight` — q4_k, N=25600, K=5120, same 73,728,000 B; identical geometry, independent weights.
2. `blk.0.attn_output.weight` — q4_k, N=5120, K=8192 (32 blocks/row, stride 4608 B, 163,840 blocks, K%256=0);
   smaller (23,592,960 B) but exercises a non-5120 K.
(Alternative: `blk.8.ffn_down.weight` is q4_k N=5120,K=25600 — but omit if mixing with q6_k layers is unwanted.)

## Open risks

- Loader `rows`/`cols` semantics are **swapped** relative to the [N,K] convention (rows=K, cols=N); a test
  harness must not assume `rows==N`.
- `output.weight` is **q6_k**, not f32 — any M1 code path assuming an f32 LM head is wrong.
- `attn_v` and `ffn_down` are mixed q4_k/q6_k; selecting `blk.0` (q6_k for both) would test q6_k, not q4_k.
  For a q4_k `ffn_down`/`attn_v` test use layers 8,9,11,... (list in §2).
- Resident `ptr` for **all** 707 tensors is set by code path; needs one live `--load-only` run to confirm
  actual non-NULL device pointers on this GPU (static inspection only here).
