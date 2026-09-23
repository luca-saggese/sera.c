# M3 W3 — Branch-aware two-segment causal GQA attention kernel

Scope: attention only. No KV storage (W1), no packing/side-array production (W2), no CLI (W4).
Baseline = M2 `attention_kernel` (`src/q3_forward.cu` L218-296) + `q3_cuda_attention` (L467-499).
No paged attention / page tables / LRU / refcounted pages (§66-§68).

## Proposed kernel signature

Per-layer base pointers are pre-offset by the host wrapper exactly like M2 L484-490
(`kv->k + layer*per_layer`); the kernel sees only the layer's slab.

```cuda
__global__ static void attention_segmented_kernel(
    const float *q, float *out,
    uint32_t tokens, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, float scale,
    /* segment 0 — shared prefix, ONE slab, no branch index */
    const float *prefix_k, const float *prefix_v, uint32_t prefix_len,
    /* segment 1 — per-branch suffix, [branch][pos][kv_head][head_dim] */
    const float *suffix_k, const float *suffix_v,
    uint32_t suffix_capacity,          /* rows reserved per branch (= stride/branch) */
    /* per-query side arrays, packed order == q token order (W2 §20) */
    const uint32_t *token_branch,      /* [tokens] branch id */
    const uint32_t *token_pos);        /* [tokens] LOCAL suffix position (0-based) */
```

Host wrapper (`q3_cuda_attention_segmented`, new decl in `src/q3_forward.h`):

```c
bool q3_cuda_attention_segmented(
    const float *q, float *out, uint32_t tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale,
    const float *prefix_k, const float *prefix_v, uint32_t prefix_len,
    const float *suffix_k, const float *suffix_v, uint32_t suffix_capacity,
    const uint32_t *token_branch, const uint32_t *token_pos,
    void *stream, char *error, size_t error_len);
```

Launch identical to M2 L491-493: `dim3 grid(tokens, n_heads); <<<grid, 32>>>`.
`position_base` and the scalar `kv_length` from M2 are DELETED — replaced by
`prefix_len` + per-token `token_pos` (equivalent information, no batch-wide window).

## Two-segment read (exact addressing)

Row stride inside a segment slab = `n_kv_heads * head_dim` (same as M2 L250/266).
`group = n_heads / n_kv_heads`, `kv_head = head / group` (M2 L229-230).

```
prefix row j:  prefix_k + (size_t)j * n_kv_heads*head_dim + kv_head*head_dim
suffix row j (branch b): suffix_k
             + ((size_t)b * suffix_capacity + j) * n_kv_heads*head_dim
             + kv_head*head_dim
```

For query token `t`: `b = token_branch[t]`, `lp = token_pos[t]`.
The two-pass loops of M2 L249-281 become, in BOTH passes:

```cuda
/* segment 0: shared prefix, independent of branch */
for (uint32_t j = 0; j < prefix_len; j++) {
    const float *k_row = prefix_k + (size_t)j * n_kv_heads*head_dim + (size_t)kv_head*head_dim;
    /* same warp-shuffle dot as M2 L251-258 */
}
/* segment 1: own branch suffix only, local rows 0..lp inclusive */
const float *sb_k = suffix_k + (size_t)b * suffix_capacity * n_kv_heads*head_dim;
const float *sb_v = suffix_v + (size_t)b * suffix_capacity * n_kv_heads*head_dim;
for (uint32_t j = 0; j <= lp; j++) {
    const float *k_row = sb_k + (size_t)j * n_kv_heads*head_dim + (size_t)kv_head*head_dim;
    const float *v_row = sb_v + (size_t)j * n_kv_heads*head_dim + (size_t)kv_head*head_dim;
}
```

This is a direct read of two discontiguous slabs — NO concatenation, NO temp contiguous
KV, NO memcpy (§14). The only new "index" is the scalar `b * suffix_capacity` offset
(donor pattern: q38 `gather_kernel` id-decoupled layout, `q38_qsa_cuda.cu` L489-506,
where row id replaces linear `j`).

## Causal range proof

Logical key index space is `0 .. position` with `position = prefix_len + lp` (§13).
The two loops cover exactly:
- indices `0 .. prefix_len-1`  → prefix rows `0 .. prefix_len-1`;
- indices `prefix_len .. prefix_len+lp` → own suffix rows `0 .. lp`.

Count = `prefix_len + (lp + 1) = position + 1` → contiguous `0..position`, no gap, no
overlap, no key beyond the query's own position. Mask is exact without a mask tensor.

- **Decision case** (prefix exists + branch suffix): two loops as above.
- **Prefill case** (query itself is a prefix token, no suffix yet): `token_branch` unused,
  suffix loop skipped, and the prefix loop runs `0 .. q_index` where `q_index = position`.
  This is *identical* to the M2 single-segment loop — so prefill keeps using
  `attention_kernel` UNCHANGED (see below). No degradation, no second code path in the
  prefill kernel.

## Branch isolation gating (§21, §32)

The gate is the branch term in the suffix row address: `suffix_k + b*suffix_capacity*row`.
Because `b = token_branch[token]` selects the *slab*, the local loop `j in 0..lp` cannot
address any other branch's rows — branch 4's suffix is physically in a different slab and
no code path computes its base. There is no mask/boolean to get wrong. The prefix slab
carries no branch index at all, so it is visible to every branch.
Canary §32 ("The code is ALPHA." vs "The code is OMEGA."): if the gate degrades to
`b == 0` or an address bug aliases slabs, branch B's suffix KV leaks and logits diverge —
this is exactly the first boundary to check in the §64 order (`attention suffix access`).

## GQA grouping (verified against model config)

`src/q3_model.c` L69-72 reads `attention.head_count` → `num_attention_heads` (Qwen3-32B: 64)
and `attention.head_count_kv` → `num_kv_heads` (Qwen3-32B: 8); `head_dim` from
`attention.key_length`, else `hidden/num_attention_heads` (L74-87); divisibility asserted
L124 (`num_attention_heads % num_kv_heads == 0` → group = 8).
Qwen3 maps q head `i` to kv head `i / group` (contiguous grouping, not interleaved).
M2's `kv_head = head / group` (L230) is therefore CORRECT and must be preserved verbatim.
Do not touch the grouping math in the refactor.

## M2 kernel lines to change (COPY→RENAME→EDIT)

Preserve the M2 body: same `dims_per_lane = head_dim/32` (L241), lane dims `d = lane + i*32`,
`__shfl_down_sync` reduction (L255-256, 272-273), two-pass max-softmax (pass1 L247-259,
pass2 L261-283), `acc[4]`, `__expf`, `out_row = out + ((token*n_heads+head)*head_dim)` (L287-295).

| Lines (q3_forward.cu) | Action |
|---|---|
| L218-224 kernel signature | EDIT: new params (above); drop `position_base`, `kv_length` |
| L226-227 blockIdx | keep |
| L229-235 group/kv_head/q_row | keep; split `k_rows/v_rows` into `prefix_*` + per-branch `sb_*` |
| L244-245 `visible`/gate | DELETE: replaced by `prefix_len` and `lp` |
| L249-259 pass1 loop | EDIT: wrap body in two loops (prefix, suffix) |
| L261-283 pass2 loop | EDIT: same two-loop split |
| L287-295 output | keep |
| L467-499 `q3_cuda_attention` | keep for prefill; add `q3_cuda_attention_segmented` wrapper |
| L937-939 call site | EDIT: decision path calls segmented wrapper with W2 side arrays |

Optional: extract the per-row dot into `__device__ __forceinline__ float seg_dot(...)` to
avoid duplicating the shuffle block twice; behavior-identical to M2.

## Prefill vs decision path

- **Prefill** (§9): keep `attention_kernel` + `q3_cuda_attention` UNCHANGED. `run_layer`
  already appends K/V into the prefix storage via D2D copies (L925-932) before attention
  (L937); the kernel's single contiguous segment + `position_base`/`kv_length` is correct.
- **Decision**: new `attention_segmented_kernel`; prefix slab = the layer slice of the
  already-built prefix, suffix slabs = W1 per-branch storage, side arrays from W2.
- Selection is by call site (prefill driver vs packed decision driver), not a runtime flag
  in the kernel. No `kv == NULL` branch is added to the segmented kernel.

## Performance sanity (opinion, no optimization work)

B=32, ~8 suffix tokens (incl. generated pos 0) → tokens≈256, heads=64 → 16384 blocks of
32 threads = 512K lanes. Work per block: (prefix_len≈129 + lp+1≈9) × 2 passes ×
(4 FMA + 5 shuffles) ≈ 1.4K ops. This is latency-bound but massively parallel; 16384
independent warps is ample to fill SM121 without split-K or tiling. Prefix K/V per layer
≈ 129 rows × 8 kv_heads × 128 × 4B × 2 ≈ 1 MiB, re-read by 64×tokens blocks → L2-resident.
Verdict: one warp per (token, head) is ACCEPTABLE for M3 correctness and the §34 benchmark.
§69-§72 ("B32 ~ 32×B1, no real batching") concerns the Q4-linear/projection forward, not
attention; attention is not the dominant cost. Explicitly: add NO flash tiling, NO
split-K, NO CUDA Graphs, NO multi-warp CTA in M3.

## Files to touch

- `src/q3_forward.cu` — rename/edit kernel L218-296; add `q3_cuda_attention_segmented`
  wrapper near L467-499; edit decision call site L937-939.
- `src/q3_forward.h` — add the `q3_cuda_attention_segmented` prototype (primitive surface).
- (W2 owns the side-array production; W1 owns the suffix slab layout/`suffix_capacity`.)
No other source file.

