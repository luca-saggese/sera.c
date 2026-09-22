# Research A — donor map (q38.c @ qwen38-spark-proto)

Donor root: `_reference/q38.c/`. Read-only. All paths below are relative to that root.
Format: `symbol -> file:LINE -> purpose`. Line numbers are from the donor as read.

Shared types live in `q38.h` (mode enum, options, `q38_platform_info`, `q38_memory_snapshot`),
so any port touches it:
- `q38_mode` (Q38_MODE_PLATFORM/INSPECT/LIST_TENSORS/MEMORY_PLAN/LOAD_ONLY/PREFLIGHT/GENERATE) -> q38.h:23
- `q38_options` (inspect/list_tensors/memory_plan/load_only/platform/json/verbose flags) -> q38.h:35
- `q38_platform_info` (cc_major/minor, cuda free/total, mem total/available, names/versions) -> q38.h:65
- `q38_memory_snapshot` (phase, rss, mem_available, cuda free/total, model file/mapped, cuda_allocated, peak_internal) -> q38.h:80

## GGUF parser

- `q38_gguf` struct (fd, map, size, version, n_kv, n_tensors, alignment, tensor_data_pos, max_tensor_bytes, kv[], tensors[], native_nvfp4) -> q38_gguf.h:50
- `q38_str {ptr,len}` -> q38_gguf.h:25 ; `q38_kv {key,type,value_pos}` -> q38_gguf.h:32
- `q38_tensor {name,ndim,dim[8],type,data,rel_offset,abs_offset,elements,bytes}` -> q38_gguf.h:38
- `Q38_GGUF_MAGIC 0x46554747` -> q38_gguf.h:22 ; `Q38_MAX_DIMS 8` -> q38_gguf.h:23
- `q38_gguf_open(path,err,errlen)` -> q38_gguf.c:346 -> calloc state, open O_RDONLY, fstat, reject <32B
- mmap step (PROT_READ, MAP_PRIVATE, fd, 0) -> q38_gguf.c:376 -> requires
- header parse: magic, version(u32), n_tensors(u64), n_kv(u64) -> q38_gguf.c:396 -> **v3 only** (rejects others at :410)
- `parse_metadata` -> q38_gguf.c:256 -> calloc kv[], records `value_pos`, honours `general.alignment` (UINT32) at :275, default alignment **32** at :267, then `skip_value`
- `parse_tensors` -> q38_gguf.c:288 -> calloc tensors[], name, ndim (1..8), dim[], type, rel_offset; bytes via type table; `tensor_data_pos = align_up(c->pos, m->alignment)` -> q38_gguf.c:326; abs_offset = tensor_data_pos + rel_offset -> :334; bounds-check vs file size; tracks `max_tensor_bytes` -> :341
- `q38_gguf_close` -> q38_gguf.c:433 -> free kv, free tensors, munmap(map,size), close(fd), free state
- `q38_gguf_find_kv/get_string/get_u32/get_u64/get_bool` -> q38_gguf.c:442/449/456/463/477
- `q38_gguf_tensor_data` -> q38_gguf.c:168 -> returns `m->map + abs_offset` (no host registration), NULL if OOB
- `q38_cursor` (base,size,pos,error[256]) + `cursor_at/read/skip/u32/u64/string` -> q38_gguf.c:23-73
- `skip_value` (recursive metadata skip, depth-limited) -> q38_gguf.c:198 ; `scalar_value_size` -> q38_gguf.c:176
- `align_up(value,alignment)` -> q38_gguf.c:75

## GGUF type table

- `gguf_type_info {name, block_elems, block_bytes}` -> q38_gguf.c:113 (struct)
- `gguf_types[]` type id -> block geometry -> bytes -> q38_gguf.c:115..145. Present ids:
  `0 f32(1,4) 1 f16(1,2) 2 q4_0(32,18) 3 q4_1(32,20) 6 q5_0(32,22) 7 q5_1(32,24)
   8 q8_0(32,34) 9 q8_1(32,40) 10 q2_k(256,84) 11 q3_k(256,110) 12 q4_k(256,144)
   13 q5_k(256,176) 14 q6_k(256,210) 15 q8_k(256,292) 16 iq2_xxs 17 iq2_xs 18 iq3_xxs
   19 iq1_s 20 iq4_nl 21 iq3_s 22 iq2_s 23 iq4_xs 24 i8(1,1) 25 i16(1,2) 26 i32(1,4)
   27 i64(1,8) 28 f64(1,8) 29 iq1_m 30 bf16(1,2) 39 mxfp4(32,17)`
  **Note: ids 4,5 (q4_2/q4_3), 31-38, 40+ absent**; `tensor_type()` returns NULL at :148.
- `q38_gguf_type_name(type)` -> q38_gguf.c:154 -> returns `"unknown"` when unmapped
- `q38_gguf_type_nbytes(type,elements,*bytes)` -> q38_gguf.c:159 -> ceil-div blocks * block_bytes, overflow-safe; **returns false for unknown type**
- unknown-type handling in parse: sets `t->bytes = 0`, non-fatal -> q38_gguf.c:322

## mmap open/close semantics

- single private read-only mapping, never host-registered, never shared with device -> q38_gguf.c:375
- payload stays in the mmap; `q38_str`/`q38_kv.key` point into it -> q38_gguf.h:17
- close unmaps then closes fd then frees -> q38_gguf.c:433
- default alignment 32 unless `general.alignment` present -> q38_gguf.c:267, :275

## GB10 / SM121 (cc 12.1) platform guard

- `q38_platform_validate(info,reason,len)` -> q38_platform.c:51 -> refusal if `cuda_device_count != 1` (:54) and if `cc != 12.1` (:60); no fallback
- `q38_platform_probe(out,reason,len)` -> q38_platform.c:69 -> /proc meminfo -> `q38_cuda_probe` -> validate
- `q38_platform_host_memory(total,available)` -> q38_platform.c:17 -> `MemTotal`/`MemAvailable` from /proc/meminfo
- `q38_cuda_probe(out)` -> cuda/q38_cuda.cu:37 -> cudaGetDeviceCount, cudaGetDevice, cudaGetDeviceProperties (cc_major/minor, name), cudaDriverGetVersion/cudaRuntimeGetVersion, cudaMemGetInfo -> cuda_free/total
- `q38_cuda_init/cleanup` -> cuda/q38_cuda.cu:16/31 ; `q38_cuda_get_shared_memory_info` -> :93
- build flag `-gencode arch=compute_121a,code=sm_121a` -> Makefile:24
- guard test -> tests/test_platform.c:26 (`test_guard_logic`), main :55

## Memory telemetry (spec §8)

- `q38_memory_tracker {internal_allocated_bytes, peak_internal_bytes}` -> q38_memory.h:20
- `q38_memory_tracker_init` -> q38_memory.c:43 ; `q38_memory_track_alloc` -> :48 (monotonic peak) ; `q38_memory_track_free` -> :55
- `q38_memory_capture(t,phase,model_file,model_mapped,cuda_allocated,*out)` -> q38_memory.c:63 -> RSS + MemAvailable live, peak from tracker
- `q38_memory_snapshot_json(s,buf,len)` -> q38_memory.c:84 -> deterministic single-line JSON keys (phase,rss_bytes,mem_available_bytes,cuda_free_bytes,cuda_total_bytes,model_file_bytes,model_mapped_bytes,cuda_allocated_bytes,peak_internal_bytes)
- RSS read `proc_rss_bytes` (/proc/self/statm) -> q38_memory.c:11 ; MemAvailable `proc_mem_available_bytes` -> q38_memory.c:26
- duplicate RSS/host readers also in `q38_platform_rss_bytes` -> q38_platform.c:37 (used by --load-only before/after)

## Residency accounting

- `q38_resident_arena {base,capacity,used,tensor_count}` -> q38_residency.h:16
- `q38_model_residency {core, expert_banks[], expert_bank_count, non_ple_bytes, ple_bytes, aligned_bytes}` -> q38_residency.h:23
- `Q38_RESIDENT_ALIGNMENT 256` -> q38_residency.h:14 ; `q38_align_up_u64` (inline) -> q38_residency.h:32
- `q38_resident_arena_init/reserve/reset` -> q38_residency.c:11/15/41 (reserve aligns up to 256, bumps capacity high-water, counts tensors)
- `q38_model_residency_init(expert_bank_count)` -> q38_residency.c:47 ; `destroy` -> :70 ; `reset` -> :79
- `q38_model_residency_account_tensor(residency,tensor,ple,expert_bank,err)` -> q38_residency.c:84 -> PLE bytes counted separately and never entered into an arena (:92); expert_bank selects a bank arena, `UINT32_MAX` = core (:101); accumulates `non_ple_bytes`
- **Qwen3.8-specific hooks here**: `ple_bytes`, `expert_banks`/`expert_bank_count`. Dense port: init with 0 banks, pass `ple=false`, `expert_bank=UINT32_MAX`.
- test -> tests/test_residency.c:6 (asserts 256-alignment: 257->512, 513->768, aligned_bytes 1280)

## Residency planner

- `q38_residency_plan_is_ple_fn` callback -> q38_residency_plan.h:14 -> the only model-family hook; dense port returns false
- `q38_residency_plan_entry {tensor_index,file_offset,bytes,ownership_class}` -> q38_residency_plan.h:17
- `q38_residency_plan_span {file_offset,bytes,first_entry,entry_count}` -> q38_residency_plan.h:24
- `q38_residency_plan {entries,entry_count,spans,span_count,resident_bytes,excluded_ple_bytes,excluded_ple_tensors}` -> q38_residency_plan.h:31
- `Q38_RESIDENCY_OWNERSHIP_RESIDENT = 1` -> q38_residency_plan.c:8 (**file-static enum, not in header**)
- `q38_residency_plan_init/destroy` -> q38_residency_plan.c:60/64
- `q38_residency_plan_build(model,is_ple,user,max_gap_bytes,max_span_bytes,plan,err,len)` -> q38_residency_plan.c:71 -> skip zero-byte tensors, bounds-check, exclude PLE, qsort by file_offset, then merge spans: `compare_entries` :18, `ranges_overlap` :28, `span_has_ple_between` :38; span merge requires ordered+`gap<=max_gap_bytes`+`span bytes<=max_span_bytes`+no PLE inside (:118-150, `fits` :148)
- caller params used in production: max_gap 64 KiB, max_span 256 MiB -> cuda/q38_forward_cuda.cu:1332
- test -> tests/test_residency_plan.c:13 (expects entries 3, resident 250, excluded_ple 90/1, spans 2)

## CUDA residency loader (allocations, staging, events, sync, timing)

- `q38_forward_cuda_context_create` -> cuda/q38_forward_cuda.cu:1179 -> calloc ctx + `cudaStreamCreate(&context->stream)` (:1188)
- `q38_forward_cuda_context_destroy` -> cuda/q38_forward_cuda.cu:1728
- `q38_forward_cuda_enable_all_non_ple_residency(ctx,model,err,len)` -> cuda/q38_forward_cuda.cu:1317 -> the fast loader; plan build (:1332), per-tensor `cudaMalloc` device destinations, double-buffered pinned staging
- `residency_plan_is_ple` (Qwen-specific PLE name filter) -> cuda/q38_forward_cuda.cu:631
- `cudaMallocHost(&residency_stage_buffers[slot], largest_span)` -> cuda/q38_forward_cuda.cu:1449
- `cudaMalloc(&residency_transfer_buffers[slot], largest_span)` -> cuda/q38_forward_cuda.cu:1451
- span loop (source memcpy -> `cudaMemcpyAsync` H2D -> per-entry `cudaMemcpyAsync` D2D -> `cudaEventRecord`) -> cuda/q38_forward_cuda.cu:1531..1610
- final `cudaStreamSynchronize` + `residency_final_syncs++` -> cuda/q38_forward_cuda.cu:1622 ; `cudaMemGetInfo` progress snapshot -> :1626
- `q38_forward_cuda_enable_nvfp4_residency` -> cuda/q38_forward_cuda.cu:1645 -> NVFP4 pack path (regions, activation scratch cudaMalloc)
- `q38_forward_cuda_prepare_lm_head` -> cuda/q38_forward_cuda.cu:1872 ; `..._prepare_nvfp4_lm_head` -> :1923 (Qwen LM-head special case)
- `q38_forward_cuda_get_residency_stats` -> cuda/q38_forward_cuda.cu:1953 -> copies all residency counters/timings into `q38_forward_cuda_residency_stats`
- `q38_forward_cuda_get_sync_stats/reset_sync_stats/sync_reason_name` -> cuda/q38_forward_cuda.cu:2093/2105/2117 ; `set_stage_context` -> :2191
- timing helpers: `event_elapsed` -> cuda/q38_forward_cuda.cu:749 ; `residency_now_ms` -> :637 ; `residency_mincore_pages` (mincore residency) -> :644
- page-fault counters via `getrusage(RUSAGE_SELF)` (`ru_minflt`/`ru_majflt`) captured around load -> cuda/q38_forward_cuda.cu:1325, :1615-1619
- load timings exposed: `residency_plan_ms`, `residency_device_alloc_ms`, `residency_source_copy_ms`, `residency_h2d_enqueue_ms`, `residency_d2d_enqueue_ms`, `residency_final_wait_ms` -> q38_forward_cuda.h:124-129 ; per-span `q38_residency_span_timing` -> q38_forward_cuda.h:90
- byte/call counters: `residency_transfer_calls`, `residency_device_copies`, `residency_final_syncs`, `residency_planned_spans`, `residency_planned_bytes`, `residency_staged_bytes`, `residency_h2d_bytes`, `residency_allocations`, `residency_allocated_bytes`, `resident_hits/misses` -> q38_forward_cuda.h:117-131
- context holds `void *residency_stage_buffers[2]`, `void *residency_transfer_buffers[2]`, `cudaEvent_t residency_reuse_events[2]` -> cuda/q38_forward_cuda.cu:256-259
- telemetry events (upload/kernel start-stop) + observer -> cuda/q38_forward_cuda.cu:758/779/789, `q38_forward_cuda_telemetry` -> q38_forward_cuda.h:49
- `Q38_CUDA_DIAG_ONLY(...)`/`Q38_DIAG_ONLY` gating -> q38_diagnostics.h:5 ; enabled by `-DQ38_DIAGNOSTICS=1` -> Makefile:9
- allocation observer hook `q38_forward_cuda_set_allocation_observer` -> q38_forward_cuda.h:19,178

## Pinned staging buffers (count and size)

Two independent staging implementations exist:
- coalesced all-non-PLE loader: **2 pinned host buffers + 2 device transfer buffers, sized `largest_span`** (derived from max span bytes) -> cuda/q38_forward_cuda.cu:1444-1451 ; reuse gated by `residency_reuse_events[slot]` -> :1529, recorded :1588
- NVFP4 loader: **2 pinned host buffers, fixed `Q38_NVFP4_STAGE_BYTES = 128 MiB` each** -> cuda/q38_nvfp4_residency.cu:9, :91 ; non-blocking `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` -> :79 ; reuse events with `cudaEventDisableTiming` -> :96 ; slot pick `(offset/STAGE_BYTES)&1` -> :126 ; event sync before reuse -> :128 ; `cudaEventRecord` after transfer -> :148

## CLI entry points

Usage text (flags) -> q38.c:41-46
- `print_platform_human` -> q38.c:71 ; `print_platform_json` -> q38.c:92
- `cmd_platform` (--platform / --platform-json) -> q38.c:106 -> probe, refuse non-SM121
- `cmd_inspect` (--inspect) -> q38.c:131 -> `model_summary` :124 (tensor bytes + logical params), name/arch, per-type histogram; `--json` form prints version/metadata_keys/tensors/file_bytes/tensor_bytes/logical_parameters
- `cmd_list_tensors` (--list-tensors) -> q38.c:186 -> per-tensor name/type/elements/bytes; `--json` emits array
- `cmd_memory_plan` (--memory-plan) -> q38.c:220 -> dry run: gguf open, tracker init, platform probe for cuda free/total, `q38_memory_capture(...,"gguf_mapped",...)`, no cudaMalloc; human + `q38_memory_snapshot_json` forms
- `cmd_load_only` (--load-only) -> q38.c:269 -> NVFP4 pack open + `q38_forward_cuda_context_create` + `q38_nvfp4_cuda_residency_load` + rss before/after; **depends on q38_nvfp4_* (Qwen3.8)**; human+JSON with format/rss/… at :341
- `cmd_preflight` -> q38.c:413 (not in M0 scope) ; `cmd_generate` -> q38.c:1283 (out of scope)
- `main` arg dispatch -> q38.c:1691 ; `--json` -> :1791 ; `--verbose` -> :1793 ; `--platform/--inspect/…` -> :1699+
- **`q38_cli.c` is unrelated** (HTTP client for the server, `main` -> q38_cli.c:337). Do not port.

## Relevant existing tests

- `tests/test_gguf.c` (178 lines) -> builds a valid GGUF, `write_file` :110, `build_valid` :75, main :118 -> asserts v3, n_kv, n_tensors, get_string/get_u32, tensor elements/bytes, type name, plus truncated-file and bad-magic failures. Writes to `/tmp/q38_test_gguf.bin`.
- `tests/test_residency_plan.c` (56) -> main :13, `is_ple` :6
- `tests/test_residency.c` (31) -> main :6
- `tests/test_memory.c` -> q38_memory.o linked (Makefile:277) ; `tests/test_m2_memory.c` (61) -> main :25, rss_kb :19, fd_count :10 -> RSS/fd leak check around GGUF open
- `tests/test_platform.c` (82) -> main :55, `test_guard_logic` :26
- `tests/bench_residency_startup.cu` (287) -> `run_per_tensor` :88, `run_coalesced` :121, `main` :191 -> per-tensor vs coalesced load benchmark; allocates pinned host + device transfer of `largest_span` :133-136; verifies with hashes :62
- build wiring: `TEST_BINS` -> Makefile:63 ; test_platform :269 ; test_gguf :274 ; test_memory :277 ; test_residency :289 ; test_residency_plan :293 ; bench_residency_startup :130 ; target `bench-startup-residency` :134
- prod objs: `PRODUCTION_C_OBJS` (:35) includes q38_gguf/q38_memory/q38_platform/q38_residency/q38_residency_plan ; `PRODUCTION_CUDA_OBJS` (:42) includes q38_cuda/q38_forward_cuda/q38_nvfp4_residency

## Model-family-agnostic vs Qwen3.8-specific

Agnostic (port near-verbatim):
- `q38_gguf.c` / `q38_gguf.h` — explicitly "isolated from any model-family binding" (q38_gguf.h:13). Type table + mmap + descriptors only.
- `q38_memory.c` / `q38_memory.h` — pure telemetry.
- `q38_platform.c` / `q38_platform.h`, `q38_cuda.h`, `cuda/q38_cuda.cu` — device/OS guard only.
- `q38_residency_plan.c` / `q38_residency_plan.h` — generic; model coupling confined to the `is_ple` callback.
- `q38.h` — but carries `q38_options` fields for Qwen-only modes (preflight, steering, disable_ple).

Qwen3.8-specific (strip or leave out of M0):
- `q38_residency.c` / `q38_residency.h` — PLE accounting (`ple_bytes`) and MoE `expert_banks`. Dense: 0 banks, `ple=false`.
- `cuda/q38_forward_cuda.cu` (:1317-1725 section is reusable; rest is not) — QSA (`prepare_qsa_chain_state` :3727, `q38_qsa_cuda.h`), GDN (`q38_gdn.h`), GR (`q38_gr_ref.h`), MoE/expert routing (`q38_moe_cuda.h`, expert_calls_by_layer), directional steering (`q38_directional_steering.h`), LM-head special handling (`prepare_lm_head` :1872), GPU argmax (`gpu_argmax_kernel_ms` q38_forward_cuda.h:147), decode (`resident_lookup_in_decode`, `gguf_name_lookup_in_decode` :141), forward.
- `q38_forward_cuda.h` — mixed: keep the residency stats/observer/telemetry typedefs; the rest pulls `q38_forward.h`, `q38_qsa_candidate.h`, `q38_directional_steering.h`, `q38_nvfp4_pack.h`.
- `cuda/q38_nvfp4_residency.cu` + `q38_nvfp4_residency.h` — NVFP4 pack specific (only reachable via --load-only).
- `q38.c` — mostly reusable for --platform/--inspect/--list-tensors/--memory-plan ; `cmd_load_only` and `cmd_preflight`/`cmd_generate` are Qwen3.8.
- Not in M0 scope: q38_ple*, q38_qsa*, q38_gdn*, q38_gr*, q38_moe*, q38_weights.c/.h, q38_decode.*, q38_model_config.*, q38_session.*, q38_tokenizer.*, q38_server*, q38_cli.c, q38_replay.c, q38_profile*.

## Dependencies (who needs whom)

- `q38_gguf.c` -> `q38_gguf.h` (+ fcntl/inttypes/stdio/stdlib/string/sys/mman/sys/stat/unistd). No q38.h.
- `q38_memory.c` -> `q38_memory.h` -> `q38.h`
- `q38_platform.c` -> `q38_platform.h` -> `q38.h` ; + `q38_cuda.h`
- `cuda/q38_cuda.cu` -> `q38_cuda.h` -> `q38.h` ; + cuda_runtime.h
- `q38_residency.c` -> `q38_residency.h` -> `q38_gguf.h`
- `q38_residency_plan.c` -> `q38_residency_plan.h` -> `q38_gguf.h`
- `q38.c` -> `q38.h` + `q38_cuda.h` + `q38_gguf.h` + `q38_memory.h` + `q38_platform.h` + `q38_nvfp4_residency.h` + (Qwen) `q38_decode.h` `q38_diagnostics.h` `q38_directional_steering.h` `q38_forward_cuda.h` `q38_session.h` `q38_tokenizer.h` `q38_weights.h` `q38_nvfp4_pack.h`
- `q38_forward_cuda.h` -> `q38_diagnostics.h` + `q38_forward.h` + `q38_qsa_candidate.h` + `q38_directional_steering.h` + `q38_nvfp4_pack.h`
- `cuda/q38_forward_cuda.cu` -> `q38_forward_cuda.h` + `q38_nvfp4_residency.h` + `q38_cuda_primitives.h` + `q38_gdn.h` + `q38_moe_cuda.h` + `q38_qsa_cuda.h` + `q38_topk_cuda.h` + `q38_gr_ref.h` + `q38_diagnostics.h` + `q38_model_config.h` + `q38_residency_plan.h` (+ cooperative_groups/cuda_fp16/cuda_runtime/math/stdio/stdlib/string/time/sys/resource/sys/mman/unistd)
- `q38_nvfp4_residency.h` -> `q38_nvfp4_pack.h`
- includes `cuda/q38_nvfp4_residency.cu` -> `q38_nvfp4_residency.h` + cuda_runtime

Exact include graph (files that matter for M0):

```
q38.h                      (no local includes; stdbool/stddef/stdint/stdio)
q38_gguf.h                 (no local includes)
q38_memory.h      -> q38.h
q38_platform.h    -> q38.h
q38_cuda.h        -> q38.h
q38_residency.h   -> q38_gguf.h
q38_residency_plan.h -> q38_gguf.h

q38_gguf.c        -> q38_gguf.h
q38_memory.c      -> q38_memory.h
q38_platform.c    -> q38_platform.h, q38_cuda.h
cuda/q38_cuda.cu  -> q38_cuda.h
q38_residency.c   -> q38_residency.h
q38_residency_plan.c -> q38_residency_plan.h

q38_forward_cuda.h -> q38_diagnostics.h, q38_forward.h, q38_qsa_candidate.h,
                      q38_directional_steering.h, q38_nvfp4_pack.h
cuda/q38_forward_cuda.cu -> q38_forward_cuda.h, q38_nvfp4_residency.h,
                      q38_cuda_primitives.h, q38_gdn.h, q38_moe_cuda.h,
                      q38_qsa_cuda.h, q38_topk_cuda.h, q38_gr_ref.h,
                      q38_diagnostics.h, q38_model_config.h,
                      q38_residency_plan.h
q38_nvfp4_residency.h -> q38_nvfp4_pack.h
```

Minimal M0 closure (dense, no Qwen3.8): `q38.h`, `q38_gguf.{c,h}`,
`q38_memory.{c,h}`, `q38_platform.{c,h}`, `q38_cuda.{h}`, `cuda/q38_cuda.cu`,
`q38_residency_plan.{c,h}`. `q38_forward_cuda.{cu,h}` must be **carved**, not copied:
only the residency loader/telemetry section (cu :1317-1725, :1953-2200) and the
residency stats/observer typedefs (h :82-172) are reusable.