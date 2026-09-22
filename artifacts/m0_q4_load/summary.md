# M0 Q4 resident load — GB10 baseline

## Model

| field | value |
|---|---|
| model | Qwen3-32B-Q4_K_M.gguf |
| model bytes | 19762149024 |
| git commit | bfd512a |
| GGUF version | 3 |
| architecture | qwen3 |
| tensors | 707 |
| qtypes | f32 (257), q4_k (385), q6_k (65) |

## GPU

| field | value |
|---|---|
| device | NVIDIA GB10 |
| SM | sm_121 |
| driver | 13.0 |
| runtime | 13.0 |

## Timings (ms)

| stage | cold | warm median |
|---|---|---|
| total load | 5451.694 | 4774.269 |
| gguf_open | 6.228 | 2.678 |
| plan | 1.185 | 1.157 |
| cuda_alloc | 2181.848 | 2239.364 |
| source_copy | 1971.946 | 2029.486 |
| h2d_enqueue | 1.307 | 1.272 |
| d2d_enqueue | 5.327 | 5.148 |
| final_wait | 16.481 | 16.486 |

## Memory

| field | cold |
|---|---|
| resident bytes | 19756174336 |
| planned bytes | 19756174336 |
| staging bytes (per slot) | 638131200 |
| h2d bytes | 19756174336 |
| cuda allocated bytes | 22308699136 |
| RSS peak bytes | 21147238400 |
| CUDA free before | 63917240320 |
| CUDA free after | 41139363840 |

## Loader shape

| counter | value |
|---|---|
| planned spans | 88 |
| transfer calls (H2D) | 88 |
| device copies (D2D) | 707 |
| CUDA allocations | 711 |
| final syncs | 1 |
| device syncs | 0 |
| coverage ok | true |

## Notes

- Invariant preserved: mmap source -> 2x bounded pinned staging
  (638131200 bytes per slot, derived from the largest plan span)
  -> cudaMemcpyAsync -> final resident allocations.
- No per-chunk cudaMalloc/cudaFree, no per-chunk sync, no whole-file
  cudaHostRegister, no host model mirror, no dequant mirror.
- Exactly one final cudaStreamSynchronize; stage reuse is event-guarded.
- `mincore` pages stay flat (4824744 before and after),
  so no full-file host residency is created.
- The GGUF mapping is not host-registered; RSS peak stays at
  ~21.1 GB, dominated by the CUDA allocation
  over unified memory.
- Cold run: `drop_caches` was not permitted in this environment, so "cold"
  is the first run of the session (counters reset across runs). Warm runs
  reuse the page cache.
- The source_copy stage dominates warm variation; cuda_alloc is the largest
  fixed cost. No tuning was attempted, per M0 §8.

## Files

- `cold.json`, `warm_1.json`, `warm_2.json`, `warm_3.json`
