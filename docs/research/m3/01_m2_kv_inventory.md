# M2 KV inventory

Scope: M2 KV cache B1 (`src/q3_forward.h` public ABI §17-18, `src/q3_forward.cu` impl,
`src/q3_forward_cli.cu` driver, `src/q3_model.h` config). All line numbers = q3_forward.cu unless noted.

## Layout
- Dtype: **FP32** (`float *`), chosen deliberately over FP16/BF16 so the prefix+suffix
  equivalence test is purely mechanical (q3_forward.h comment, KV-cache block; M2.md §17
  says "preferably FP16/BF16" but the impl uses FP32).
- Per-layer layout `[capacity][kv_heads][head_dim]`; across layers contiguous:
  `k[layer][position][kv_head][head_dim]` (struct comment L380-386; per_layer stride L540/484).
- Two separate device buffers: `kv->k` and `kv->v` (one each for all layers — no interleave).
- Bytes/token/layer: `2 * kv_heads * head_dim * 4 = 2*8*128*4 = 8192` B.
- Bytes/token/model (Qwen3, 64 layers): `8192 * 64 = 524288` B ≈ **512 KiB/token**.
  Runtime workspace stores no split of this; the KV cache is separate from the runtime arena.
- Stride math (kernel, L250/266): `k_rows + j * n_kv_heads * head_dim` — one row per
  (position, single kv_head), so each head's dims are `head_dim` apart within a position block.

## Allocation
- API: `q3_kv_cache_init(num_layers, num_kv_heads, head_dim, capacity, error, err_len)` (L520).
- Host-side: `calloc` for the `q3_kv_cache` struct (L530).
- Device-side: **two `cudaMalloc`s** — one for `kv->k`, one for `kv->v`, each
  `capacity * num_kv_heads * head_dim * num_layers * 4` bytes (L541-542).
- One arena per buffer across all layers (single contiguous K slab + single contiguous V slab),
  **not** per-layer buffers and **not** one combined K/V arena.
- Growth: `q3_kv_cache_reserve(capacity, …)` (L551) — if new cap > current, cudaMalloc new K+V,
  copy only the used prefix (`length * kv_heads * head_dim * 4` rows/layer, D2D, L568-575),
  free old, swap. Called once in the CLI.
- CLI size: `q3_kv_cache_init(..., token_count, ...)` — capacity == number of forward tokens,
  single B1 pass (q3_forward_cli.cu L249). Per-layer diag scan creates its own cache per layer.
- Runtime workspace (`struct q3_forward_runtime`) is a separate single arena (incl. rt->k/rt->v
  pre-cache row buffers), see `q3_forward_create` L660+.

## Position indexing
- `position_base` = absolute position of the first token of this batch (ABI doc, q3_forward.h).
- Cache length tracked as a single scalar `kv->length` (L385) — positions are 0..length-1.
- Append: `run_layer` copies this batch's roped K/V into the cache at
  `kv->k[layer*per_layer + kv->length*row]` then `kv->length += token_count`
  happens in the caller (`run_range` L1048, `forward_layer` L1103). Copy is a D2D
  `cudaMemcpyAsync` of `token_count*row*4` bytes per layer (L899-905).
- Reset: `q3_kv_cache_reset` sets `length = 0` (L585); destroy frees both cuda buffers + host struct.
- RoPE position = `position_base + token` per output row (rope_kernel L200-201).

## Attention access path
- Public primitive: `bool q3_cuda_attention(const float *q, const float *k, const float *v,
  uint32_t tokens, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
  uint32_t position_base, q3_kv_cache *kv, uint32_t layer, float scale, float *out,
  void *stream, char *error, size_t error_len)` (q3_forward.h; impl L460).
- Inside (L474-491): if `kv != NULL` it **re-derives the contiguous per-layer slice**:
  `k_src = kv->k + layer*per_layer`, `v_src = kv->v + layer*per_layer`, and
  `kv_length = kv->length + tokens`. The `k`/`v` args are used only in the prefill/
  uncached path (`kv == NULL` → k_src/v_src = args, kv_length = tokens).
- Kernel launch: `dim3 grid(tokens, n_heads)`, `<<<grid, 32>>>` (L491-493). One block per
  (token, head), one warp (32 lanes).
- **Contiguous per-layer buffer assumed**: k/v are single device pointers; position j is
  `base + j*kv_heads*head_dim`. No separate pointer/offset list.
- **batch=1 is NOT a hard requirement of the kernel**: `tokens` is a grid dim and the kernel
  loops over all cached positions (it is fully batched-capable). But `kv->length` is one
  scalar, so a batch shares one base offset — no per-batch position segmentation.

## batch=1 assumptions
- `q3_forward_run` / `q3_forward_run_range` (L993-1106): process `token_count` tokens as one
  contiguous batch from `position_base`; single append of `kv->length += token_count`.
- `run_layer` (L815): all Q/K/V projections, norm, RoPE operate on the whole `token_count`
  batch in one go; single append copy. Supports batch>1, but only as contiguous suffix.
- `q3_kv_cache` has **no batching concept**: scalar `length`, scalar append offset.
- Real B1 usage in CLI: token_count read from one file, `position_base=0`.
- `q3_forward_layer` (L1084) = single-layer variant of the same contiguous-batch path.

## Copy/allocation points
- **Append copies (hot path, 1/layer when kv set)**: D2D `cudaMemcpyAsync` of K (L899) and V
  (L902) into the cache. Two per layer = 2*64 = 128 memcpys per forward.
- **Cache growth D2D copies**: `q3_kv_cache_reserve` copies used prefix per layer (L569-575).
- **Zero cudaMalloc in the hot path**: runtime workspace is allocated once in
  `q3_forward_create` (L695+); per forward it does not malloc (stats track cuda_allocations==0).
- Allocation points (not hot): `q3_kv_cache_init` (2 cudaMallocs), `q3_kv_cache_reserve`
  (2 cudaMallocs), `q3_forward_create` (1 workspace cudaMalloc), CLI hidden_out/logits
  (2 cudaMallocs). All freed by `_destroy`.

## Attention kernel indexing
- `attention_kernel` (L221): block `(x=token, y=head)`. `group = n_heads/n_kv_heads`
  (Qwen3: 64/8=8), `kv_head = head/group` (L229).
- `visible = position_base + token + 1`; if `visible > kv_length` return (causal gate, L245).
- Loop `j in 0..visible-1`: `k_row = k_rows + j*n_kv_heads*head_dim` (L250, 266) and
  `v_row = v_rows + j*n_kv_heads*head_dim` (L267). So query at position i reads keys 0..i
  **contiguously** from the per-layer slab.
- Two-pass max-softmax per (token,head): pass1 max_score over visible; pass2 den + weighted V
  sum; warp-shuffle dot reduction over head_dim (32 lanes × 4 dims). Out written
  `out[token*n_heads + head]` (L292-296).
- Causal mask is NOT a computed mask — it is the `visible` bound (positions beyond the
  current absolute position are never read).

## Struct fields
- `struct q3_kv_cache` (L380-387): `num_layers, num_kv_heads, head_dim, capacity,
  length` (uint32), `float *k, *v` (`[num_layers][capacity][num_kv_heads][head_dim]`).
- `struct q3_forward_runtime` (L607-640, KV-relevant): `float *k` `[max_tokens, kv_heads*head_dim]`,
  `float *v` (pre-cache roped rows, L618-619), `float *q`, `float *attn`, workspace arena
  `float *workspace` + `workspace_bytes`. Config from `rt->config` (num_layers/heads/kv_heads/head_dim).

## M3 implications (what must change for segmented prefix+suffix)
- Kernel takes a **single contiguous per-layer slice** (`k_src = kv->k + layer*per_layer`) and a
  scalar `kv_length`. Segmented prefix+suffix cannot be expressed today: no start/stop position
  list, no way to read prefix positions and suffix positions as two discontiguous views in one pass.
- `kv->length` is one scalar — a shared prefix shared by B suffixes must either be (a) copied
  B times (M2 §18 explicitly excludes physical fork), or (b) the kernel/attention signature must
  grow a segmented view (e.g. prefix_len + per-branch suffix offsets, or an index/offset array
  replacing the `j` linear loop), and `kv_length` must become per-branch or summed-with-offset.
- Position math `visible = position_base + token + 1` collapses a batch into one contiguous
  window; per-suffix absolute positions need per-branch position_base (query grid is currently
  (token, head) with a single shared position_base).
- Append copy (L899-905) writes the whole batch at offset `kv->length*row`; batched B suffix
  appends must write into B distinct suffix slots and advance B separate lengths.
- FP32 → a segmented M3 prefix KV would likely keep FP32 or add FP16 halves; dtype is a
  storage decision independent of the view refactor.

