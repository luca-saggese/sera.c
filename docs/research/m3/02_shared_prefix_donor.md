# Shared-prefix donor patterns

Research for M3: ONE state (immutable shared prefix KV) → MANY decisions (B branch
suffixes), batched. Question: what can be copied from the two donors, vs what must
stay custom/simple.

## Donor inventory (what exists where)

- q38.c: full attention is `q38_qsa_cuda.cu` only (QSA: index/key gather → rope →
  softmax → weighted sum). `q38_forward_cuda.cu` is GR/GDN, not attention. KV cache
  is `q38_kvstore.h/.c` (a *disk* key-value session store, not a tensor KV cache).
- llama.cpp ggml-cuda: FLASH attention `fattn.cu`/`fattn-common.cuh`/`fattn-*.cuh`;
  KV storage/eviction is CPU-side `src/llama-kv-cache*.h/.cpp` + `llama-batch.h`.
- NOTE: **`llama_kv_cache_view*` does not exist in this tree** (search hits only
  `test-backend-ops.cpp`/docs CSV, cosmetic). No segment/view struct to copy.

## KV view patterns

- q38:cuda/q38_qsa_cuda.cu L489-506 `gather_kernel`: decoupled layout — a flat
  `device_k`/`device_v` [kv_count][kv_heads][head_dim] plus an explicit
  `device_ids` index array. Row index = `selected` (0..kv_count); a separate
  compaction step finally produces a contiguous selected-K/V buffer (L466-484,
  L533-538). **This is the directly copyable pattern**: two-segment reads = build
  `(prefix_idx, suffix_idx)` per branch, gather once.
- llama.cpp: K/V are full contiguous projections inside fattn kernels
  (`fattn-common.cuh` L26-27 index into `K12/V12` by one integer KV column). Sparse
  masking (`FATTN_*`) produces `KV_max`/`indices`/`counts` per query
  (`fattn.cu` L13-89, `fattn-common.cuh` L666-721 `flash_attn_mask_to_KV_max` +
  `fflash_attn_compact_mask`). Good reference for per-row variable-length limits,
  but assumes one contiguous KV tensor — not two segments.

## Two-segment attention patterns

- Neither donor has a native prefix+suffix two-segment reader.
- q38 gather (above) is segment-agnostic: it reads *any* subset of rows by id; two
  segments = two row ranges concatenated in the ids array. **Best fit to copy.**
- llama.cpp fattn is single-segment KV + per-query sparse mask. It could mask out
  the gap but wastes the full `ne[1]` (incl. other branches' suffixes) → the
  opposite of our "read only prefix+own suffix" goal.

## Batch descriptor patterns

- Copyable struct shape: `llama-ubatch` / `llama_batch_allocr`
  (`src/llama-batch.h` L21-52): `n_tokens`, `n_seqs` (B), `n_seqs_unq`,
  per-token `pos[n_tokens]`, `n_seq_id[n_tokens]`, `seq_id[n_tokens][...]`,
  `seq_id_unq[n_seqs_unq]`. The **shape** (arrays for pos/seq_id, per-seq position
  min/max) is the useful part.
- The **allocator machinery** (`ubatch_reserve`/`ubatch_add`, pos_set_t seq_pos
  bookkeeping, L115-149) should NOT be copied — it exists to fold hundreds of
  variable, overlapping llama.cpp sequences into micro-batches.

## Variable-length causal patterns

- q38 attention_kernel: single scalar fixed head loop; row-limited by iterating
  `selected_count` only (q38_qsa_cuda.cu L526-547). Per-row limit would be a per-row
  `kv_limit[r]` instead of a scalar.
- llama.cpp sparse-mask path is the rich reference: `flash_attn_mask_to_KV_max`
  (fattn-common.cuh L666) finds the last non-masked KV per query row, and
  compacted `counts[]`/`indices[]` encode per-row KV length; softmax denominator
  and output respect `n_kv_max` per row (fattn.cu L88-89). Copy *this algorithm* —
  not its -inf mask allocation (a `-INF` mask tensor sized ne[1]×ncols1).

## What can be copied (file:line references)

1. q38 two-phase gather structure — `q38_qsa_cuda.cu` L466-484 (`project_device`),
   L489-506 (`gather_kernel`), L525-557 (`gather_attention`+`attention_kernel`).
   Rewrite row-id source to be per-branch `(prefix_ids || suffix_ids)`.
2. llama.cpp per-row KV max/compact — `fattn-common.cuh` L666-724
   (`flash_attn_mask_to_KV_max`, `flash_attn_ext_compact_mask`), `fattn.cu` L13-99
   (indices+counts, `count=min(row_count,n_kv_max)`). Adapt to two-segment row
   bounds without a full -inf mask tensor.
3. llama.cpp batch-descriptor *shape* — `llama-batch.h` L21-52: `seq_id[]`,
   `seq_id_unq[]`, `pos[]`, per-seq `seq_pos_min/max` (L95-97). Port as a small
   `q3_decision_batch`.
4. q38 softmax+weighted-sum kernel — `q38_qsa_cuda.cu` L519-547 (reuse as the final
   per-branch attention body).

## What should remain custom/simple

- **No prefix-replication / no eviction / no paged allocator**: llama.cpp
  `llama-kv-cache-dsa/iswa/dsv4/*` and `find_slot`/ring-buffer eviction are overkill.
  Our prefix KV is immutable once created; we never evict, never page, never grow.
- **No ubatch folding**: static batch (B fixed per call, all suffixes same shape),
  so per-token `n_seq_id`/multi-seq union logic in `llama_batch_allocr` is dead
  weight. One flat pos array + per-branch suffix offset suffices.
- **No `llama_kv_cache_view` layer**: it doesn't even exist here; and a shared-prefix
  view = just an offset/length pair (prefix_len + branch suffix_len) over one
  physical K/V buffer — a plain struct, no view indirection.
- **Keep q38 FP32 flat-row layout** (M2 doc 01_m2_kv_inventory.md L7): a custom
  `q3_segmented_kv { float *k,*v; uint32 prefix_len; uint32 branch_offsets[B]; }`
  beats llama's fp16/KFE/rope-per-call pipeline for a static unfrozen prefix.

## Recommendation for M3

1. Physically store prefix KV **once**; per branch store only its suffix rows. A
   per-branch row-id list is `[0..prefix_len) ++ [prefix_len+off_b .. off_b+len_b)`.
2. Port q38's gather+attention two-pass shape (copy-list #1, #4); it already hits
   non-contiguous rows via an id array → two segments fall out for free.
3. Take llama.cpp's per-row `counts[]`/`indices[]` sparse-limit idea (copy-list #2)
   as the per-branch causal limit (prefix full-length, suffix capped at own pos)
   instead of its `-inf` mask tensor.
4. Batch with a minimal struct shaped after `llama-ubatch` (copy-list #3), static:
   `B`, `pos[]`, `suffix_off[]`, `suffix_len[]`. No allocator, no eviction.
5. Net: **copy ~q38 qsa layering + llama.cpp compact-mask algorithm + llama-batch
   struct shape**; **keep custom**: segmented KV storage, immutable-prefix invalidation
   (rebuild prefix only when STATE changes), static batch descriptor.
