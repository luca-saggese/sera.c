# Research B — donor fast-loader path reconstruction (q38.c)

Donor: `/home/lvx/sera.c/_reference/q38.c` @ branch `qwen38-spark-proto`. Read-only.
All line refs are donor lines. Factual reconstruction only — no alternative design.

## 0. Entry points

| symbol | file:line |
|---|---|
| `q38_runtime_init` (startup orchestration) | `q38_session.c:178` |
| residency enable + lm_head prep | `q38_session.c:239-258` |
| `q38_forward_cuda_context_create` | `cuda/q38_forward_cuda.cu:1178-1194` |
| `q38_forward_cuda_enable_all_non_ple_residency` (THE loader) | `cuda/q38_forward_cuda.cu:1317-1644` |
| `q38_forward_cuda_enable_nvfp4_residency` (alt path) | `cuda/q38_forward_cuda.cu:1645-1726` |
| `q38_nvfp4_cuda_residency_load` (alt staged loader) | `cuda/q38_nvfp4_residency.cu:68-189` |
| `q38_forward_cuda_context_destroy` | `cuda/q38_forward_cuda.cu:1727-1811` |

Note: donor `--load-only` (`q38.c:269-285`) binds an **NVFP4 pack**, it does not load a GGUF. There is
no donor GGUF→CUDA CLI verb; the GGUF loader is only reached through `q38_runtime_init`.

## 1. Data path (real symbols)

```
q38_gguf_open()                       open(O_RDONLY)+mmap(PROT_READ,MAP_PRIVATE)  q38_gguf.c
   model->map (const uint8_t*), model->size, model->tensors[].abs_offset
        |
        v
q38_residency_plan_build(model, residency_plan_is_ple, NULL,
                         64*1024, 256*1024*1024, &plan)                fwd_cuda.cu:1332-1334
   plan.entries[]  {tensor_index,file_offset,bytes}   q38_residency_plan.h:17-22
   plan.spans[]    {file_offset,bytes,first_entry,entry_count}  .h:24-29
        |
        +--> per-entry destination:  cudaMalloc(&entries[at].device, tensor->bytes)   :1413
        |       entries[]  = persistent_tensor{host,device,bytes}    :72-76
        |       exec->ptr = entries[at].device                       :1440
        v
   stage[2] = cudaMallocHost(largest_span)   :1449      transfer[2] = cudaMalloc(largest_span) :1451
   reuse_events[2] = cudaEventCreateWithFlags(...,cudaEventDisableTiming)        :1477
        |
   for s in plan.spans:  slot = s % 2
        if (s >= 2) cudaEventSynchronize(reuse_events[slot])                     :1525
        memcpy(stage[slot], model->map + span->file_offset, span->bytes)         :1539
        cudaMemcpyAsync(transfer[slot], stage[slot], span->bytes, H2D, stream)   :1549
        for each entry j in span: cudaMemcpyAsync(entries[e].device,
              (char*)transfer[slot] + (entry.file_offset - span->file_offset),
              entry.bytes, D2D, stream)                                          :1576
        cudaEventRecord(reuse_events[slot], stream)                              :1587
        |
        v
   ONE final: cudaStreamSynchronize(context->stream)                             :1604-1606
        |
        v
   exec_tensors[i].ptr  (resident device pointer, indexed by GGUF tensor index)
```

## 2. Staging buffers: count and size

- **Main path: count = 2.** Hardcoded loop `for (size_t slot = 0; slot < 2 && largest_span; ++slot)`
  (`cuda/q38_forward_cuda.cu:1448`). Arrays are fixed-size: `void *residency_stage_buffers[2];`
  `void *residency_transfer_buffers[2]; cudaEvent_t residency_reuse_events[2];` (`:256-259`).
- **Size = `largest_span`**, i.e. the largest `plan.spans[i].bytes` (`:1444-1447`), NOT a fixed MiB
  constant. `largest_span ≤ max_span_bytes = 256u*1024u*1024u` (256 MiB) because that is the planner
  cap passed at `:1334`. So per-slot stage is `min(largest_span, 256 MiB)`, and both slots are equal
  (`residency_stage_bytes = residency_transfer_bytes = largest_span`, `:1517-1518`).
- Memory type: stage = pinned host (`cudaMallocHost` `:1449`); transfer = device (`cudaMalloc` `:1451`).
  Aliases `context->residency_stage` / `residency_transfer` are set to **slot 0** (`:1505-1508`).
- **NVFP4 alternative path: count = 2, size = fixed `Q38_NVFP4_STAGE_BYTES = 128u*1024u*1024u`**
  = 128 MiB each (`cuda/q38_nvfp4_residency.cu:9`, alloc `:91`), pinned host only (staging is copied
  directly into `device_regions[region]`, no separate device bounce buffer).
- A third, unrelated pool exists (`q38_ple_stage_pool`, `cuda/q38_ple_stage.cu:21-47`, `cudaHostAlloc`
  at `:33`) with caller-supplied `count`/`bytes_per_buffer`; **no in-tree caller** for
  `q38_ple_stage_pool_init` outside its own header — dead for the load path.

## 3. Double buffering / stage reuse

- Slot selection is purely positional: `const size_t slot = s % 2;` (`:1523`).
- Reuse guard: only from the 3rd span onward (`if (s >= 2)`) the host waits for that slot's event
  (`cudaEventSynchronize(residency_reuse_events[slot])`, `:1524-1530`). Spans 0 and 1 never wait.
- Event is recorded after the span's D2D copies are enqueued (`:1587-1592`), so the event covers both
  the H2D and every D2D of that span.
- The 2-stage pipeline overlaps only *host memcpy(s+1)* with *GPU H2D+D2D(s)*: the host `memcpy`
  (`:1539`) and the `cudaMemcpyAsync` (`:1549`) alternate, so with 2 slots the CPU copy of span s+1
  proceeds while the GPU drains span s. There is no thread pool, no reader thread, no `pread`.
- Source copy is a plain `memcpy` from the mmap (`:1539`), i.e. file-backed page faults happen inline
  on the loader thread.

## 4. CUDA stream

- Sole stream creation for the loader: `cudaStreamCreate(&context->stream)` at
  `cuda/q38_forward_cuda.cu:1188` inside `q38_forward_cuda_context_create`. **Default flags** (no
  `cudaStreamNonBlocking`, no `cudaStreamCreateWithFlags`, no `cudaStreamCreateWithPriority`).
- Second stream, NVFP4 path only: `cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking)`
  (`cuda/q38_nvfp4_residency.cu:84-85`), destroyed at `:165`/`:181`.
- `cudaSetDevice` is never called anywhere in the donor; device 0 is implicit
  (`cuda/q38_cuda.cu:16-29`, `q38_cuda_probe` reads `cudaGetDevice` but does not set).
- Destroy: `cudaStreamDestroy(context->stream)` (`cuda/q38_forward_cuda.cu:1808`).

## 5. CUDA events

| purpose | creation | record | wait |
|---|---|---|---|
| stage reuse guard ×2 | `cudaEventCreateWithFlags(..., cudaEventDisableTiming)` `:1477-1478` | `cudaEventRecord(reuse_events[slot], stream)` `:1587` | `cudaEventSynchronize(reuse_events[slot])` `:1525` |
| telemetry H2D/kernel | `cudaEventCreate` ×4 in `telemetry_events_create` `:758-778` | `telemetry_event_record` `:789-799` | elapsed only (`event_elapsed` `:749-752`) — not in load path |
| expert/MoE staging reuse | `:908-911` | `:791` | `:52`/`:53` |
| argmax | `:4000-4001` | `:4002`,`:4006` | `cudaEventSynchronize(stop)` `:4008` |
| NVFP4 stage reuse ×2 | `cuda/q38_nvfp4_residency.cu:96-97` | `:148` | `:128` |

Events in the load path: exactly the 2 `residency_reuse_events`. Destroyed at
`cuda/q38_forward_cuda.cu:1794-1795` and on alloc-failure cleanup `:1454-1455`, `:1480-1481`.

## 6. cudaMalloc / cudaMallocAsync

`cudaMallocAsync` is **never used** anywhere in the donor. `cudaMalloc` sites in `q38_forward_cuda.cu`
(9 call sites) and their role:

| line | role | in load path? |
|---|---|---|
| `:1171` (`ensure_buffer`) | generic grow-only scratch buffer | used by backends only |
| `:1413` | **per-resident-tensor destination** | YES — one `cudaMalloc` per plan entry |
| `:1451` | 2 × device transfer buffer | YES |
| `:1706-1713` | 4 × NVFP4/MoE scratch | NVFP4 path |
| `:1821` | directional steering | no |
| `:1906` | LM-head fallback weights | YES if lm_head ∉ persistent set |
| `:2947-3000` `ensure_gr_buffers` | GR workspace (via `ensure_buffer`) | no |

`cudaMallocHost` sites: `cuda/q38_forward_cuda.cu:1449` (2 × stage, load path),
`cuda/q38_nvfp4_residency.cu:91` (2 × 128 MiB stage).
`cudaHostAlloc`: only `cuda/q38_ple_stage.cu:33` (unused pool). **`cudaHostRegister` and
`cudaHostGetDevicePointer` appear nowhere in the donor — the whole mmap is never registered.**

## 7. cudaMemcpyAsync

49 `cudaMemcpyAsync` call sites in `cuda/q38_forward_cuda.cu`; the load-path ones are:

- `:1549` — `transfer[slot] ← stage[slot]`, `cudaMemcpyHostToDevice`, `span->bytes` (1 per span).
- `:1576` — `entries[e].device ← (char*)transfer[slot] + relative`, `cudaMemcpyDeviceToDevice`,
  `planned->bytes` (1 per plan entry).
- `:1909` — lm_head H2D, only when `persistent_find()` missed.
There is also a device→host lm_head/gdn/qsa family (`:1236-1241`, `:1266-1271`, `:1298-1309`) — those are
**state upload/download**, not model loading.

**No synchronous `cudaMemcpy` exists in the donor at all** (zero occurrences); every transfer is
`cudaMemcpyAsync` on a stream. `cudaMemcpy2DAsync`/`cudaMallocManaged`/`cudaMemAdvise` are unused.

## 8. Synchronization count in the load path

`cudaDeviceSynchronize` — **not present anywhere in the donor** (0 occurrences).
`cudaStreamSynchronize` — 19 occurrences in `q38_forward_cuda.cu`, but in
`q38_forward_cuda_enable_all_non_ple_residency` there is exactly **one**:

- `cuda/q38_forward_cuda.cu:1604-1606` — `Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_RESIDENCY_INIT,
  cudaStreamSynchronize(context->stream))`. **This is the single final sync** of the loader;
  it drains all spans' H2D+D2D before returning. Counter `context->residency_final_syncs++` (`:1622`).

Plus **`span_count - 2` host-blocking `cudaEventSynchronize` calls** on the reuse events (only for
`s >= 2`, `:1525`). These are stage-reuse waits, not stream syncs. So in `--load-only`-equivalent
telemetry: `residency_final_syncs == 1`, and event waits = `max(0, planned_spans - 2)`.
LM-head fallback (`:1912-1913`) and steering (`:1829`) each add their own `cudaStreamSynchronize`
(`Q38_CUDA_SYNC_LM_HEAD_RESIDENCY_INIT`, `Q38_CUDA_SYNC_STEERING_INIT`), so a
`--load-only` that keeps lm_head special handling would report 2.
NVFP4 path: 1 final `cudaStreamSynchronize(transfer_stream)` (`q38_nvfp4_residency.cu:159`),
plus `slot_in_flight` `cudaEventSynchronize` waits (`:128`), plus a second sync only on the error path
(`:175`).

## 9. Timing counters, page faults, mincore

`#if Q38_DIAGNOSTICS` gates all of it (`cuda/q38_forward_cuda.cu:636-665`, `:708-757`).

Context fields (`cuda/q38_forward_cuda.cu:252-282`; mirrored in `q38_forward_cuda.h:116-139`):

| field | set at |
|---|---|
| `residency_plan_ms` | `:1337` (`residency_now_ms()` around `q38_residency_plan_build`) |
| `residency_device_alloc_ms` | `:1394` / `:1514-1515` |
| `residency_source_copy_ms` + `residency_span_timings[s].source_copy_ms` | `:1542-1546` |
| `residency_h2d_enqueue_ms` + `...h2d_enqueue_ms` | `:1557-1560` |
| `residency_d2d_enqueue_ms` | `:1594-1595` (`d2d_started` `:1564-1569`) |
| `residency_final_wait_ms` | `:1598-1613` |
| `residency_planned_bytes / planned_spans / staged_bytes / h2d_bytes / transfer_calls / device_copies / allocations / allocated_bytes / final_syncs` | `:1338`, `:1357`, `:1548`, `:1562-1563`, `:1436-1437`, `:1510-1511`, `:1585`, `:1622` |
| `residency_span_timings` array alloc | `:1340-1346` (freed `:1799`) |

Clock: `residency_now_ms()` = `CLOCK_MONOTONIC_RAW` (`:637-642`); `host_now_ms()` = `CLOCK_MONOTONIC`
(`:709-713`).

Page faults: `getrusage(RUSAGE_SELF, &usage_before)` at `:1325-1326`, `usage_after` at `:1614-1615`;
`ru_minflt`/`ru_majflt` stored into `residency_minor_faults_before/after`,
`residency_major_faults_before/after` (`:1616-1619`). **Fault counters are sampled full-load only**
(no per-span sampling).

mincore: `residency_mincore_pages()` (`:644-665`) — page-aligns `model->map`, `calloc`s a `vec`,
`mincore(base, pages*page, vec)`, counts `vec[i] & 1`. Called before (`:1327`) and after (`:1620`);
results in `residency_mincore_pages_before/after`. `q38_memory.c` does not use mincore.

Emission of all of the above to stderr as one-line JSON: `q38_session.c:263-327`
(keys `residency_*`, `mincore_pages_*`, `minor_faults_*`, `major_faults_*`, `span_timings[]`).
`q38_forward_cuda_get_residency_stats` copies context→public struct: `:1953-2092`.

## 10. Resident pointer table and resolution

Two parallel tables, both owned by the context:

1. `persistent_tensor *persistent; size_t persistent_count; size_t persistent_bytes;`
   (`:236-238`; struct at `:72-76`). One entry per **plan entry**, in plan order; holds
   `host` (mmap pointer), `device`, `bytes`. Built at `:1438-1441`. Lookup:
   `persistent_find(context, host)` — **linear scan** (`:683-691`), used only by lm_head prep
   (`:1887`); a duplicate `host` anywhere in the set is a fatal error (`:1403-1412`).
2. `q38_exec_tensor *exec_tensors; size_t exec_tensor_count; const q38_gguf *exec_model;`
   (`:283-285`; struct at `:83-94`: `host, ptr, bytes, rows, cols, qtype, tensor_id, storage,
   gguf_offset, name`). Indexed **by GGUF tensor index**, allocated `:1367-1373`, sized
   `model->n_tensors` (`:1375`), populated for every tensor at `:1376-1390`
   (`storage = Q38_STORAGE_RESIDENT` or `Q38_STORAGE_FILE_BACKED_PLE`).

Resolution: `exec_tensor_for(context, model, tensor)` (`:692-701`) does pointer arithmetic
`&exec_tensors[tensor - model->tensors]` (O(1), no name lookup, guarded by `model == exec_model`);
`exec_tensor_is_resident(exec, tensor)` (`:702-708`) checks `storage == Q38_STORAGE_RESIDENT &&
ptr && bytes == tensor->bytes`. Call sites: `:875-880`, `:1058-1061`, `:2214-2216`, `:2355-2371`,
`:2503-2505`, `:2652-2653`, `:2814-2815`, `:3035-3044`, `:3096-3100`, `:3292-3322`.
Coefficient use: `(const unsigned char *)exec->ptr + row * row_bytes` (`:2254`).
Strictness: `Q38_EXEC_STRICT` env (`:1187`) makes a non-resident non-PLE tensor a hard error
(`:2217-2220`, `:2361-2364`, `:1639-1641`).
Fingerprints for cross-run stability: `persistent_pointer_fingerprint` (FNV-1a over device pointers),
`cuda_context_identity`, `cuda_stream_identity`, `workspace_pointer_fingerprint`
(`:1968-2004`).

## 11. How the plan drives the load loop

- **Level 1 — tensor-level pre-pass** (`:1396-1443`): iterate `plan.entries` in **plan order**
  (sorted by file offset, `q38_residency_plan.c:18-26`, `:119-120`) and `cudaMalloc` one destination
  per entry. `exec->ptr` is written here.
- **Level 2 — span-level transfer loop** (`:1521-1597`): iterate `plan.spans`; per span one host
  `memcpy` of the whole contiguous span (including the ≤64 KiB non-tensor gaps allowed by
  `max_gap_bytes = 64*1024`) then one H2D, then one D2D per entry using span-relative offsets
  (`entry.file_offset - span.file_offset`, `:1574-1575`).
- Span adjacency rules: `q38_residency_plan.c:128-156` — a new entry joins the current span only if
  ordered, `gap <= max_gap_bytes`, span bytes `<= max_span_bytes`, and no excluded PLE range lies in
  the gap (`span_has_ple_between`, `:38-58`); `:1333-1334` passes 64 KiB / 256 MiB.
- Spans are the unit of coalescing, entries are the unit of destination copy. Destination is chosen
  **once**, before any transfer — no per-chunk alloc/free.
- Progress callback fires **once**, after the load, with `loaded_bytes` and `cudaMemGetInfo`
  (`:1624-1629`).

## 12. Destination allocation strategy

- **Per-tensor `cudaMalloc`, not one big arena** (`:1413`). Alignment is whatever the CUDA allocator
  returns; **no explicit alignment is applied to destinations**. Consequence: tensor access uses
  `exec->ptr` directly with `row * row_bytes` offsets (`:2254`).
- Failure mid-loop is non-fatal-by-design: on `cudaMalloc` failure the already-built set is kept
  (`context->persistent = entries; persistent_count = at; all_non_ple_resident = false`, `:1415-1435`)
  and the reason is recorded in `context->persistent_failure[256]`.
- Accounting only: `residency_allocations++` / `residency_allocated_bytes += tensor->bytes`
  (`:1436-1437`) and `+4`/`+= largest_span*4` for the staging quartets (`:1510-1511`).
- `cudaMemGetInfo` is deliberately *not* a rejection gate on unified memory (`:1358-1363` comment);
  it is only used for the progress observer (`:1625`).
- Coverage assertion: `persistent_coverage_ok = (at == expected_tensors) && (total == expected_bytes)
  && (persistent_ple_entries == 0)` (`:1635-1638`).
- The **256-byte `Q38_RESIDENT_ALIGNMENT` arena** in `q38_residency.h:14-53` / `q38_residency.c:15-39`
  and `q38_model_residency` (used from `q38_weights.c:685-721`, `q38_weights.h:91`) is a *host-side
  byte-count accounting* path only; its `base` is never allocated for CUDA and it is not the loader's
  allocator.

## 13. Cleanup path

`q38_forward_cuda_context_destroy` (`:1727-1811`), order:
1. `cudaFree` of every scratch pointer `device_weights … device_gr_updated` (`:1730-1792`).
2. Per slot 0..1: `cudaEventDestroy` (if `residency_reuse_events_ready`), `cudaFree(transfer)`,
   `cudaFreeHost(stage)` (`:1793-1798`).
3. `free(residency_span_timings)` `:1799`; `free(host_qsa_output)` `:1800`.
4. `if (!lm_head_uses_persistent) cudaFree(lm_head_device_weights)` `:1801-1802` — avoids
   double-free when lm_head points into the persistent set (or into an NVFP4 region).
5. `for (i < persistent_count) cudaFree(persistent[i].device)` `:1803-1804`; `free(persistent)` `:1805`.
6. `free(exec_tensors)` `:1806`.
7. `q38_nvfp4_cuda_residency_destroy` `:1807`.
8. `cudaStreamDestroy(stream)` `:1808`; `free(context)` `:1809`.
Partial-load failure paths re-enter residency bookkeeping before returning (`:1407-1434`,
`:1453-1475`, `:1479-1501`) so destroy stays correct. Runtime-level teardown:
`q38_runtime_destroy` `q38_session.c:350-358`; device teardown `q38_cuda_cleanup` →
`cudaDeviceReset()` `cuda/q38_cuda.cu:31-35`.

## 14. Qwen3.8-specific regions to DELETE when porting

`cuda/q38_forward_cuda.cu` (4030 lines) — function-granular delete list:

| lines | symbol | why |
|---|---|---|
| `13-15`, `26`, `28-29` | `cooperative_groups.h`, `cuda_fp16.h`, NVTX decls | not needed by loader |
| `321-343` | `q38_directional_steering_kernel` | steering |
| `344-352` | `gr_bf16_value`, `gr_silu` | GR |
| `353-414` | `gr_fused_normalize_down_kernel` | GR |
| `415-448` | `gr_fused_lowrank_up_kernel` | GR |
| `449-460` | `gr_fused_branch_read_kernel` | GR |
| `461-492` | `gr_normalize_kernel` | GR |
| `493-505` | `gr_fused_writeback_kernel` | GR |
| `506-513` | `moe_shared_silu_mul_kernel` | MoE |
| `514-522` | `moe_shared_add_kernel` | MoE |
| `528-548` | `ensure_qsa_chain_capacity` | QSA |
| `549-554` | `is_lm_head_tensor` | LM-head special handling |
| `576-586` | `subsystem_for_stage` (can keep a stub) | forward telemetry |
| `587-602` | `is_gr_projection_stage`, `is_gdn_projection_stage` | GR/GDN |
| `612-621`, `800-853` | `tensor_id_for`, `emit_telemetry` | forward-only telemetry |
| `667-682` | `residency_group` (used only in the failure message `:1427`) | keep only if the message is kept |
| `854-1038` | `q38_forward_cuda_expert_backend` | experts/MoE |
| `1039-1163` | `q38_forward_cuda_moe_layer_q2_backend` | MoE/routing |
| `1164-1177` | `ensure_buffer` | forward scratch allocator |
| `1196-1213` | `q38_forward_cuda_reset_gdn_state` | GDN |
| `1214-1246` | `q38_forward_cuda_load_gdn_state` | GDN state |
| `1247-1278` | `q38_forward_cuda_sync_gdn_state` | GDN state |
| `1279-1316` | `q38_forward_cuda_load_qsa_state` | QSA state |
| `1645-1726` | `q38_forward_cuda_enable_nvfp4_residency` | NVFP4/MoE path (delete unless Q4 pack needs it) |
| `1812-1871` | `load_directional_steering`, `set_directional_steering_scales`, `apply_directional_steering_device`, `apply_directional_steering` | steering |
| `1872-1922` | `q38_forward_cuda_prepare_lm_head` | LM-head special handling |
| `1923-1952` | `q38_forward_cuda_prepare_nvfp4_lm_head` | LM-head + NVFP4 |
| `2093-2138` | `get_sync_stats`, `reset_sync_stats`, `sync_reason_name` | forward sync profiling (optional) |
| `2139-2161` | `set_qsa_candidate`, `record_route`, `get_expert_layer_calls` | QSA/MoE/routing |
| `2199-2331` | `q38_forward_cuda_matvec_backend` | forward execution |
| `2332-2482` | `q38_forward_cuda_matrix_backend` | forward execution (holds `use_resident_lm_head`) |
| `2483-2623` | `q38_forward_cuda_matrix_batch_backend` | forward execution |
| `2624-2788` | `q38_forward_cuda_gdn_layer_backend` | GDN |
| `2789-2925` | `q38_forward_cuda_gdn_layer_device` | GDN |
| `2926-3170` | `gr_cooperative_grid`, `ensure_gr_buffers`, `gr_read_device_impl`, `gr_write_device_impl`, `ensure_qsa_chain_workspace`, `qsa_chain_tensor`, `qsa_chain_device_impl`, `*_gr_read_device`, `*_gr_write_device`, `*_qsa_chain_device` | GR + QSA |
| `3171-3454` | `q38_forward_cuda_decoder_layer_chain_backend` | forward execution |
| `3455-3620` | `q38_forward_cuda_gr_read_backend`, `gr_write_backend` | GR |
| `3621-3982` | `ensure_qsa_chain_workspace`, `qsa_chain_tensor`, `prepare_qsa_chain_state`, `qsa_chain_device_impl`, `qsa_chain_backend`, `qsa_qkv_backend` | QSA |
| `3983-4030` | `q38_forward_cuda_greedy_argmax` | argmax/decode |
| `96-320` | `struct q38_forward_cuda_context` | **reduce, not delete**: keep `stream`, `persistent*`, `exec_model/exec_tensors/exec_tensor_count`, `residency_*`, `all_non_ple_resident`, `persistent_*`, `exec_strict`, `*_observer`, `nvfp4_*` only if NVFP4 kept; drop all `device_moe_*`, `device_qsa_*`, `device_gdn_*`, `device_gr_*`, `device_nvfp4_*`, `qsa_chain_*`, `expert_*`, `q2_*`, steering and sync-stats fields |

Crop with fixed line numbers first, then re-derive boundaries — deletions above are structurally
independent ranges.

PLE-specific code:
- `is_ple_embedding_table` (`:622-630`) and `residency_plan_is_ple` (`:631-636`) — matching
  `.ple.ple_embedding.ngram_embedding.shard_`; **replace with an always-false predicate** so the
  planner excludes nothing (`q38_residency_plan_build` requires a non-NULL `is_ple`).
- `Q38_STORAGE_FILE_BACKED_PLE` branch (`:78-81` enum, `:1388-1389`), `ple_file_backed_accesses` /
  `ple_file_bytes` (`:290-291`), PLE clauses in `emit_telemetry` (`:810-814`, `:829-831`),
  `persistent_ple_entries` (`:1638`).
- Delete `q38_ple_stage.cu/.h`, `q38_ple_cuda.cu`, `q38_ple_prefetch.c*`, `q38_ple_cache.*`,
  `q38_ple_ref.*`, `q38_ple.c/.h` from the port entirely.
- `cuda/q38_nvfp4_residency.cu` encodes Qwen3.8 expert regions; drop unless the real Q4 file is a
  native NVFP4 pack.

Also drop `q38_gdn.h`, `q38_gr_ref.h`, `q38_moe_cuda.h`, `q38_qsa_cuda.h`, `q38_topk_cuda.h`,
`q38_qsa_candidate.h`, `q38_directional_steering.h` includes (`:4-9`) and the enum
`q38_forward_cuda_sync_reason` members other than `Q38_CUDA_SYNC_RESIDENCY_INIT`
(`q38_forward_cuda.h:21-38`).

## 15. Facts the port must not silently change

- 2 stage slots, `slot = s % 2`, wait only from span 2 — this is what keeps 1 final sync honest.
- Stage size is span-derived, not 128 MiB, in the GGUF path; 128 MiB is the NVFP4 path constant.
- Destinations are per-tensor `cudaMalloc` with no alignment; mmap is never host-registered.
- Zero synchronous `cudaMemcpy`, zero `cudaDeviceSynchronize`, zero `cudaMallocAsync`.
- LM-head special handling adds a second `cudaStreamSynchronize`; dropping it is required for
  `final_syncs == 1`.