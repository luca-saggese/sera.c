# 05 — Packed suffix batching (W2)

Scope: ONE batched forward for all B branch suffixes (M3 §18-21, §46). §66-68: static
batch only, no paged/continuous batching. KV *storage* is W1, kernel math W3, CLI W4.
All anchors: `src/q3_forward.cu` unless noted.

## Packed token layout

Branches b=0..B-1 with suffix lengths S_b (ragged, §16). Total active tokens
`T = Σ_b S_b`. Activations are one `[T, hidden]` row-major buffer (same shape M2 already
uses: `rt->hidden/normed/q/k/v/attn/gate/up/down`, `q3_forward_create` L742-757).

```
branch_offsets[b] = Σ_{i<b} S_i          (uint32, B+1 entries; branch_offsets[B]=T)
flat token index for (b, j), j in [0,S_b):
    t = branch_offsets[b] + j
```

So branch 0's rows first, then branch 1's, etc. — exactly the conceptual layout in §20.
Global absolute position of t: `token_positions[t] = prefix_len + token_local_position[t]`
where `token_local_position[t] = t - branch_offsets[token_to_branch[t]] = j` (§17). Every
downstream op (norm/Q/K/V/qk-norm/RoPE/o_proj/MLP) is the unchanged M2 op run once on the
whole T rows: `run_layer(..., token_count=T)` (L815) already supports arbitrary T and only
needs the position plumbing below. There is no per-branch kernel launch: one RMSNorm
(L839), one Q/K/V projection (L848/L856/L864), one RoPE (L916), one attention (L922),
one gate/up/down set per layer — that is the §46 shape.

## Padding decision: PACK TIGHT (no padding), side arrays

Rejected: pad every branch to `S_max = max_b S_b` → `T_pad = B*S_max` rows. Cost is not
just wasted projection rows: the padding tokens enter every `q3_cuda_q4k_linear` call
(§19) and are rows the MMQ path stages as Q8_1 (`q3_mmq.cu` L230-236 sizes the arena by
`max_tokens*max_k`), and, worse, the padding tokens have no real position. If they share
row-shape they must be *excluded* from KV append, from attention, and from candidate
output — exactly the three things §16 forbids them from touching. Excluding them adds a
per-token predicate to append (`L899-905`) and to attention anyway, so tight packing with
side arrays is strictly less work and less memory.

Tight packing: `T = Σ S_b ≤ B*S_max`. Raggedness is handled by side arrays, not shapes.
Concrete padding cost avoided: `B*S_max - ΣS_b` rows × (hidden + inter + n_heads*D +
n_kv_heads*D)·4 bytes × 64 layers of Q4 work per packed forward. For the §16 worst case
(all S_b = S_max) tight == padded; for the real workload (different question lengths) the
saving is the difference.

"Must not participate in attention" cost with tight packing = 0: there are no fake tokens,
so no mask is needed. The only mask is branch isolation (§21), which is required for real
tokens too.

## Side arrays (concrete)

Built on host per decision batch, uploaded once (`cudaMemcpyAsync`, H2D, stream) into a
small device scratch slice of the workspace (add `T*4 + 4*sizeof(uint32)` to L742-757;
T ≤ 1024 for B32×S32 so it is negligible).

```
uint32 *token_to_branch;       /* [T]   token_to_branch[t] = b                */
uint32 *token_local_position;  /* [T]   local suffix position j of t          */
uint32 *token_positions;       /* [T]   prefix_len + token_local_position[t]  */
uint32 *branch_offsets;        /* [B+1] branch_offsets[0]=0, [B]=T            */
uint32 *branch_lengths = S_b;  /* [B]   suffix length per branch              */
```

`token_to_branch` and `branch_lengths` are also needed host-side to slice the final logits
per branch (W4). Host copies can live in a `q3_decision_batch` shaped after
`llama-batch.h` L21-52 (donor doc §3): `n_tokens=T`, per-token `pos[]`/`seq_id[]`,
`seq_pos_min/max`. No device constant memory needed; plain device pointers passed to
kernels.

## position_base → token_positions (KEY DELIVERABLE)

M2 `rope_kernel` L200-201 and `attention_kernel` L245 take ONE scalar `position_base` and
compute `position = position_base + token`. In a packed batch prefix_len is shared but the
suffix offset differs per branch, so the token index is not the position. Minimal change:

```c
/* rope_kernel: replace the scalar with a per-token array */
__global__ static void rope_kernel(float *q, float *k, uint32_t tokens,
                                   uint32_t n_heads, uint32_t n_kv_heads,
                                   uint32_t head_dim,
                                   const uint32_t *token_positions,   /* NEW */
                                   const float *inv_freq) {
    ...
    const uint32_t token = index / (pairs * (n_heads + n_kv_heads));
    const uint32_t position = token_positions[token];   /* was position_base+token */
```

```c
/* attention_kernel: per-query absolute position replaces the scalar bound */
__global__ static void attention_kernel(const float *q, const float *k, ...,
                                        const uint32_t *token_positions,   /* NEW */
                                        const uint32_t *token_to_branch,   /* NEW */
                                        uint32_t kv_length, float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t b     = token_to_branch[token];
    const uint32_t pos   = token_positions[token];
    /* causal bound = pos+1 keys, but restricted to this branch's segment */
    ...
```

Everything else in the two kernels is untouched. `q3_cuda_rope` (L448-464) and
`q3_cuda_attention` (L467-499) change only by swapping `uint32_t position_base` for
`const uint32_t *token_positions` and adding `const uint32_t *token_to_branch` to the
attention wrapper. The M2 B1 path is preserved by passing a length-1 / identity mapping
(or by keeping a `position_base` fast path); Test 1 (§33) uses that equivalence.
`run_layer` (L815) gains the two array args and forwards them at L916/L922.

## Branch isolation in packed attention (§21)

Positions alone cannot isolate branches: two tokens with the same local j in different
branches have different absolute positions, but the *prefix* is shared and must be visible
to both while branch 3's suffix must be invisible to branch 4. In the packed layout the
suffix rows of all B branches are physically contiguous per layer
`[layer][branch][position][kv_head][head_dim]` (W1/W3 decision), so a branch-4 suffix row
sits just past branch-3's. The gate is `token_to_branch[t]` + `branch_offsets[]`:

```
prefix keys:  all keys at positions [0, prefix_len)  -> always readable
suffix keys:  only keys with branch_id == token_to_branch[t], i.e. the row range
              [prefix_len + branch_offsets[b], prefix_len + branch_offsets[b] + S_b)
visible = min(pos+1, prefix_len + branch_offsets[b] + S_b)   /* causal within suffix */
```

A branch-3 query therefore reads prefix + its own suffix slice and can never address a
branch-4 row because the upper index is `prefix_len + branch_offsets[3] + S_3`, which is
exactly branch 3's end. Two-segment read = q38 gather pattern (`q38_qsa_cuda.cu` L489-506,
donor doc §1): build key ids `[0..prefix_len) ++ [prefix_len+off_b .. +len_b)`. W3 owns the
kernel body; W2 only supplies `token_to_branch` and `branch_offsets` so the id list is
constructible per query.

## Capacity / max_tokens for B=1..32

`max_tokens` is fixed once in `q3_forward_create` (L664) and drives (a) the workspace
slices L742-757 sized by `T`, and (b) `q3_cuda_q4k_linear_init(device, max_tokens,
max_features, max_k)` (L706-709) → `q3_mmq_arena_bytes(max_tokens, max_features, max_k)`
(`cuda/q3_mmq.cu` L213-253), whose Q8_1 stage is `f(max_features, max_tokens, max_k)`.

Required capacity: the packed forward holds only suffix tokens; the prefix lives in its own
arena (W1) and is never materialized as activation rows. So

```
max_tokens_required = B * S_max          /* NOT B*S_max + prefix_len */
```

`prefix_len` must NOT be folded into `max_tokens`: it would inflate every `[T, ...]`
workspace slice (and the MMQ Q8_1 arena) by prefix_len rows that are never projected.
Note MMVQ only covers `T <= Q3_MMVQ_MAX_BATCH = 8` (`cuda/q3_mmq.h` L77, dispatch
`src/q3_q4_linear.cu` L190-200): B=2 with S≤4 stays MMVQ, but any realistic B≥8 uses the
MMQ tile path, so `q3_mmq_arena_bytes` must be sized for `T = B*S_max` up front.

Concrete: B=32, S_max=32 → `max_tokens = 1024`. With
`workspace ≈ T*(hidden + inter*2 + ...)·4`, T=1024 alone is ~180 MB (hidden 5120,
inter 25600); the suffix KV arena is `64 layers · 1024 · 8 · 128 · 2 · 4` ≈ 512 MiB (W1).
No change to any allocation *code path*, only to the `max_tokens` argument the caller
passes at `q3_forward_create`. If per-branch S_max can be capped at create time, pass
`B*S_max`; otherwise pass `32*S_max`.

## Reuse list

| Function / kernel | file:line | Verdict |
|---|---|---|
| `rms_norm_batched_kernel` + `q3_cuda_rms_norm` | L117 / L404 | VERBATIM (rows=T) |
| `silu_mul_kernel` + wrapper | L154 / L419 | VERBATIM |
| `residual_add_kernel` + wrapper | L167 / L433 | VERBATIM |
| `embedding_kernel` + wrapper | L295 / L502 | VERBATIM (token_ids[T] packed) |
| `rope_kernel` | L181-206 | RENAME + EDIT: scalar → `token_positions[t]` (L200) |
| `attention_kernel` | L218-300 | RENAME + EDIT: add `token_positions`,`token_to_branch`; replace `visible` L245 |
| `q3_cuda_q4k_linear` | `src/q3_q4_linear.cu` L141 | VERBATIM (token_count=T) |
| `q3_mmq_arena_bytes` | `cuda/q3_mmq.cu` L213 | VERBATIM (caller passes new max_tokens) |
| `run_layer` | L815 | NEEDS-CHANGE: thread `token_positions`+`token_to_branch` into rope/attn (L916/L922); attention now branch-aware |
| KV append D2D copies | L899-905 | NEEDS-CHANGE (W1): per branch into `[layer][branch][pos]` slots, advance B lengths |
| `q3_cuda_attention` wrapper | L467-499 | EDIT signature (+2 ptr args); per-layer slice math stays |
| `q3_forward_create` slices | L742-757 | EDIT: add side-array scratch; keep layout otherwise |
| `q3_forward_run_range` | L1000-1106 | RENAME → packed entry; loop stays `for layer (L1058)`; `kv->length += token_count` L1064 replaced by per-branch advance |
| `q3_forward_final_norm` | L1073 | VERBATIM (but must select each branch's LAST token row, not row T-1) |

## Files to touch

- `src/q3_forward.cu` — rope/attn kernels + wrappers, `run_layer`, new `q3_forward_run_packed`.
- `src/q3_forward.h` — new `q3_decision_item/batch` (§15), packed run ABI, side-array scratch.
- `cuda/q3_mmq.h` / `q3_mmq.cu` — none (VERBATIM), caller size only.
- `src/q3_q4_linear.cu` — none (dispatch already by token_count).
- New: `docs/research/m3/04_final_kv_layout.md` (W1/W3, suffix row addressing).
