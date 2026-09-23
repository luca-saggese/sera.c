# M3 final KV layout (W1 — KV layout only)

All line numbers are `src/q3_forward.cu` unless noted; ABI in `src/q3_forward.h`.

## Layout
Keep M2 FP32, slab shape `[layer][capacity][kv_head][head_dim]`, two slabs (K and V).
The kernel (`attention_kernel` L221) walks positions with `k_rows + j*n_kv_heads*head_dim`
(L250,266) and `q3_cuda_attention` re-derives `kv->k + layer*per_layer` (L474-477); a rank-2
row layout keeps both untouched. FP16 would shrink bytes but break Test-4 mechanics (L32
comment) — not worth it.

- row/token/layer = `kv_heads*head_dim = 8*128 = 1024` floats.
- bytes/token/layer = `1024*4 = 4096` B per slab; **8192** B K+V.
- bytes/token/model = `8192*64 = 524288` B ≈ **512 KiB/token**.
- prefix bytes = `P * num_layers * capacity_stride` where
  `per_layer(layer) = layer * capacity * kv_heads * head_dim` and `row = kv_heads*head_dim`.
- Single helper `q3_kv_bytes_for_tokens(layers,kv_heads,head_dim,tokens)` (M3 §7) feeds all
  accounting; no inline `*4` scattered.

## Prefix object (immutable after seal)
```c
typedef struct q3_prefix_kv {
    uint32_t num_layers, num_kv_heads, head_dim;
    uint32_t token_count;      /* P, positions 0..P-1 */
    uint32_t capacity;         /* token slots, >= token_count */
    float *k, *v;              /* [num_layers][capacity][kv_heads][head_dim] */
    size_t  bytes;             /* 2 * capacity * per-token bytes */
    uint64_t generation;       /* bumped on seal/reset */
    bool sealed;
    const void *owner_runtime; /* identity check, §53 */
} q3_prefix_kv;
```
Pointer form `prefix_k(l) = k + l*capacity*row`. **Two cudaMalloc calls total** (one K, one V),
identical to `q3_kv_cache_init` L541-542 — no per-layer allocation, satisfies §24.

## Prefill direct-write (key decision)
`run_layer`'s append (L899-905) already writes roped K/V into
`kv->k + layer*per_layer + kv->length*row` via D2D memcpy, where `per_layer = kv->capacity*row`.
**No adapter needed if prefill uses `q3_prefix_kv*` masquerading as the M2 backing store.**
Concretely: make prefill run the ordinary M2 path (`q3_forward_run_range`, L1033) against a
`q3_kv_cache` whose `k/v/capacity/length` fields *are* the prefix slabs and the running position
count. So the prefix owns the slabs and a thin `q3_kv_cache` header aliases them
(`kv.k = prefix->k; kv.capacity = prefix->capacity; kv.length` advances normally). K/V land
**directly in prefix storage**; then `sealed = true` and the header is discarded.
Implementation: add `q3_prefix_kv_create(rt, P, …)` and have prefill call
`q3_forward_run` with an aliasing cache. Zero temp KV, zero extra memcpy (M3 §9).

## Branch object (suffix arena)
One suffix arena allocated once at max capacity, pre-sliced; branches only get descriptors.
```c
typedef struct q3_kv_branch {
    const q3_prefix_kv *prefix;
    uint32_t suffix_tokens, suffix_capacity;
    uint32_t branch_id;
    int32_t  suffix_slot;      /* slot in arena; -1 when inactive */
    float *suffix_k, *suffix_v;/* base of this slot */
} q3_kv_branch;

typedef struct q3_suffix_arena {
    uint32_t num_layers, num_kv_heads, head_dim;
    uint32_t max_branches, max_suffix;
    float *k, *v;              /* [max_branches][num_layers][max_suffix][kv_heads][head_dim] */
    size_t  bytes; uint32_t used_slots;
} q3_suffix_arena;
```
Offset formula (per branch slot `b`, layer `l`):
`suffix_off = ((b*num_layers + l)*max_suffix) * row`, `row = num_kv_heads*head_dim`;
`suffix_k(l) = arena.k + suffix_off`. Position `p` row at `suffix_off + p*row`.
Arena size = `max_branches * num_layers * max_suffix * row * 4` bytes per slab. The arena
lives in the batch/session (§52), allocated once (2 cudaMalloc).

## Lifecycle
- `q3_branch_create(prefix, arena)`: assert `prefix->sealed`, `arena->used_slots < max_branches`;
  descriptor init + slice assignment only (`suffix_k/v`, `suffix_tokens=0`). **~0 GPU work.
  No cudaMalloc, no copy.** Bumps `branch_create_ms` (§25).
- `q3_branch_reset(b)`: `suffix_tokens = 0`; capacity/pointers unchanged, **no free/realloc**
  (§51). Equivalent to `q3_kv_cache_reset` L585 (sets `length=0`) but per-branch.
- `q3_branch_destroy(b)`: clears descriptor and returns `b->suffix_slot` to the arena free
  list; arena itself freed only at `q3_suffix_arena_destroy` (§48-52).
- `q3_prefix_kv_destroy`: frees K/V slabs (2 cudaFree) + host struct, mirrors
  `q3_kv_cache_destroy` L589.

## prefix_copy_bytes accounting
Counter lives on the batch/session object alongside the arena. Only the append-into-suffix
path (mirror of L899-905, writing branch-local rows) may touch it, and that writes **suffix**,
not prefix — so nothing increments it in the normal path. Any accidental prefix→suffix
memcpy would call an explicit `q3_account_prefix_copy(bytes)`; grep-proof:
`prefix_copy_bytes` appears exactly once in a write site. Verifiable via Test 4 (§33): run
B ∈ {1,2,4,8,16,32}, assert the counter is 0 and the prefix bytes fingerprint (§49) is
unchanged after each batch.

## Failure semantics (§53)
Return `false` + `set_error`, never a silent full-forward fallback:
too many branches (`arena->used_slots == max_branches`), suffix overflow
(`suffix_tokens + S > suffix_capacity`), unsealed prefix (`!prefix->sealed` in decide),
different runtime/model (`prefix->owner_runtime != rt`), candidate_count > capacity
(`rt->candidate_capacity`, L632), invalid position (`position_base < prefix->token_count`),
unsupported qtype.

## Files to touch
- `src/q3_forward.h` — add `q3_prefix_kv`, `q3_kv_branch`, `q3_suffix_arena`, `q3_kv_view`,
  `q3_kv_bytes_for_tokens`, API decls.
- `src/q3_forward.cu` — arena/prefix alloc (mirror L520-547), aliasing-cache prefill hook,
  branch create/reset/destroy, counter.
- `src/q3_forward_cli.cu` — M3 diagnostic mode driving prefix→branch→decide, JSON incl.
  `prefix_copy_bytes` (L169-197 pattern).
- `docs/research/m3/04_final_kv_layout.md` — this file (only extra design doc, §22).
