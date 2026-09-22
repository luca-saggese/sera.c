# M0 → M1 notes

Short observations only, not implemented in M0.

- `q3_loader_load()` already stops after the final sync with stable resident
  pointers (707 tensors, 19,756,174,336 bytes), so M1 can build on
  `q3_loader_tensor()` without re-porting the load path.
- `q3_resident_tensor` carries `qtype/rows/cols/gguf_offset`, enough to route
  tensors to per-qtype GEMM kernels.
- `cuda_alloc_ms` (~2.2 s) and `source_copy_ms` (~2.0 s) dominate the load;
  the H2D enqueue itself is ~1.3 ms. If M1 cares about load time, those two
  stages are the first bottlenecks to separate, not the transfer path.
- The staged path copies whole coalesced spans through pinned staging even when
  a span is only partly resident, so `staged_bytes` slightly exceeds
  `resident_bytes` already at M0 scale (equal here; will diverge once a planner
  excludes tensors).
- `attn_v` and `ffn_down` are mixed q4_k/q6_k, so an M1 Q4 GEMM cannot assume a
  uniform quant type per layer.
