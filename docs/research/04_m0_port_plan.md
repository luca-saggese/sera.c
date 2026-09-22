# Research D — M0 minimal port graph (`q38.c` -> `sera.c`)

Donor: `/home/lvx/sera.c/_reference/q38.c` (branch `qwen38-spark-proto`), read-only.
Target tree: `src/`, `cuda/`, `tests/`, `Makefile`, binary `q3`.
Scope: only `--platform`, `--inspect`, `--list-tensors`, `--memory-plan`, `--load-only`, plus `--json` / `--verbose`.
Mechanical rename rule after every copy: `sed -i 's/q38_/q3_/g; s/Q38_/Q3_/g'` (covers symbols, macros, include guards, log prefixes).

## 1. File mapping table

| DONOR | TARGET | ACTION | Notes |
|---|---|---|---|
| `q38_gguf.c` | `src/q3_gguf.c` | copy+rename as-is | 485 L; libc only; no donor include but its own header. Zero cuts. |
| `q38_gguf.h` | `src/q3_gguf.h` | copy+rename as-is | 2617 B; type table lives here (f32/f16/bf16/q4_k/q6_k/iq*/mxfp4). |
| `q38.h` | `src/q3.h` | copy+rename+reduce | Keep: `q3_mode` (5 modes only), `q3_platform_info`, `q3_memory_snapshot`, reduced `q3_options`. Remove: generate/tokenizer/prompt/prefill/steering/preflight fields, `Q38_MODE_PREFLIGHT`, `Q38_MODE_GENERATE`. |
| `q38_memory.c` | `src/q3_memory.c` | copy+rename as-is | 104 L; RSS + MemAvailable + internal/peak accounting + snapshot JSON. |
| `q38_memory.h` | `src/q3_memory.h` | copy+rename as-is | Includes `q38.h` -> becomes `q3.h`. |
| `q38_platform.c` | `src/q3_platform.c` | copy+rename as-is | 89 L; `/proc` host memory + RSS + SM121-only guard (`cc_major==12 && cc_minor==1`). Includes `q38_cuda.h`. |
| `q38_platform.h` | `src/q3_platform.h` | copy+rename as-is | Includes `q3.h`. |
| `q38_cuda.h` | `src/q3_cuda.h` | copy+rename+reduce | Keep `q3_cuda_probe`, `q3_cuda_init/cleanup`. Delete `q38_cuda_shared_memory_info` + getter (compute-only, unused in M0). |
| `cuda/q38_cuda.cu` | `cuda/q3_cuda.cu` | copy+rename+reduce | 106 L. Keep probe/init/cleanup; delete `q38_cuda_get_shared_memory_info`. |
| `q38_residency.c` | `src/q3_residency.c` | copy+rename+reduce | 118 L. Keep arenas + accounting. `q38_model_residency_init(..., expert_bank_count, ...)` -> reduce to `expert_bank_count = 0` (drop `expert_banks` array and `Q38_MODEL_EXPERTS` use); `ple_bytes` counter kept but never incremented. |
| `q38_residency.h` | `src/q3_residency.h` | copy+rename+reduce | Drop `q38_resident_arena *expert_banks`, `expert_bank_count`; keep `q38_align_up_u64` inline, `Q3_RESIDENT_ALIGNMENT` 256. |
| `q38_residency_plan.c` | `src/q3_residency_plan.c` | copy+rename as-is | 158 L; planner is model-family agnostic (offset sort, span grouping, gap/span caps, overlap + overflow checks). Port **unchanged**. |
| `q38_residency_plan.h` | `src/q3_residency_plan.h` | copy+rename as-is | Entry/span/plan structs + `is_ple` callback typedef (keep the callback, see below). |
| `cuda/q38_forward_cuda.h` | `src/q3_model_loader_cuda.h` | copy+rename+reduce | Keep: ctx fwd decl, `Q3_CUDA_SYNC_REASON` (reduced to residency reasons), `q3_cuda_sync_stats`, `q3_residency_span_timing`, `q3_forward_cuda_residency_stats` (reduced), `*_create/destroy`, `*_stream`, allocation+progress observers, `enable_all_non_ple_residency`, `get_residency_stats`, `get_sync_stats`, `reset_sync_stats`, `sync_reason_name`. Delete: all forward/matvec/matrix/expert/gdn/gr/qsa/argmax/steering/nvfp4/lm-head decls and the `q38_forward.h`, `q38_qsa_candidate.h`, `q38_directional_steering.h`, `q38_nvfp4_pack.h` includes. |
| `cuda/q38_forward_cuda.cu` | `cuda/q3_model_loader_cuda.cu` | copy+rename+reduce | 4030 L -> target ~700-900 L. Kept region = the residency loader (`enable_all_non_ple_residency` body, L1317-1645) + its helpers; delete list in §4. |
| `q38.c` | `src/q3_main.c` | copy+rename+reduce | 1856 L -> ~500 L. Keep `usage`, `print_platform_human/json`, `cmd_platform`, `model_summary`, `cmd_inspect`, `cmd_list_tensors`, `cmd_memory_plan`, arg parse + dispatch, `monotonic_ms`. Rewrite `cmd_load_only` (donor body is NVFP4-pack, §5). Delete generate/preflight/everything else. |
| `q38_diagnostics.h` | `src/q3_diagnostics.h` | copy+rename as-is | 423 B macro header (`Q3_DIAG_ENABLED`, `Q3_DIAG_ONLY`, `Q3_DIAG_EXPR`) used by the reduced CUDA loader. |
| *new* | `Makefile` | new | Rules derived from donor, see §6. |
| *new* | `artifacts/m0_q4_load/` | new (empty dir) | Filled in M0 step 12. |

## 2. Explicitly NOT ported

| DONOR | ACTION | Why |
|---|---|---|
| `q38_json.c/.h` | do NOT port | Zero consumers on the M0 path: `q38.c --json` uses `printf`, and `q38_json.h` is included only by `q38_cli.c`, `q38_server*.c`, `q38_prompt.c`. Revisit in M2. |
| `q38_cli.c` | do NOT port | Misnomer: it is an HTTP/SSE **client to the q38 server** (`connect_server`, `build_chat_body`, `send_chat`), not the mode dispatcher. Nothing to do with the M0 modes. |
| `q38_weights.c/.h` | do NOT port | Qwen3.8 layer/expert/PLE/GDN/QSA weight binder. M0 §11 forbids the Qwen3 binder. |
| `q38_model_config.c/.h` | do NOT port | Only `Q38_MODEL_LAYERS`/`Q38_MODEL_EXPERTS` are used by the CUDA loader, exclusively for arrays of deleted subsystems. Edit the include out; if a constant is still referenced after reduction, add a local `#define Q3_MAX_LAYERS 48` in the `.cu`. |
| `q38_profile.c/.h` | do NOT port | NVTX subsystem profiler for forward/decode. The loader only calls `q38_profile_nvtx_push/pop` (2 extern decls) and the observer typedef; delete both. |
| `q38_quant.c/.h` | do NOT port | Dequantization for compute; M0 only needs **physical tensor bytes**, which the `gguf_types[]` table in `q3_gguf.c` already provides. |
| `q38_cuda_timing.h` + `cuda/q38_cuda_timing.cu` | do NOT port | Used only by test/bench targets; loader timing is the residency fields. |
| `q38_cuda_primitives.h` + `.cu` | do NOT port | RMSNorm/SiLU/matvec/dequant kernels = compute, out of M0 scope. |
| `q38_forward.c/.h`, `q38_forward_probe.c` | do NOT port | Forward execution. `q38_forward.h` is a hub header (qsa, steering, ple_prefetch, moe_ref, session_types, state, weights) — it must be **edited out** of `src/q3_model_loader_cuda.h`, otherwise it drags ~10 files in. |
| `q38_nvfp4_pack.*`, `q38_nvfp4_residency.h/.cu`, `q38_nvfp4_runtime.*` | do NOT port | Native-NVFP4 pack path (`--load-only` in the donor). M0 loads **standard Q4 GGUF**. |
| `q38_ple*.c/.h`, `cuda/q38_ple*.cu`, `q38_gdn*`, `q38_gr*`, `q38_qsa*`, `q38_moe*`, `q38_state.*`, `q38_session*.*`, `q38_decode.*`, `q38_tokenizer.*`, `q38_replay.*`, `q38_server*.*`, `q38_kvstore.*`, `q38_oracle.*`, `q38_golden.*`, `q38_prompt.*`, `q38_directional_steering.*`, `cuda/q38_gdn.cu`, `q38_moe_cuda.cu`, `q38_qsa_cuda.cu`, `q38_qsa_candidate.cu`, `q38_ple_cuda.cu`, `q38_ple_stage.cu`, `q38_gr.cu`, `q38_topk_cuda.cu`, `q38_profile_cuda.cu` | do NOT port | Qwen3.8 model-family subsystems. Explicitly out of M0 scope (M0 §18). |
| `third_party/gguf-tools/`, `dir-steering/`, `models/`, `artifacts/`, donor `docs/`, donor `tests/` (except the two in §7) | do NOT port | Reference/quant tooling and fixtures; M0 uses only `cc`/`nvcc` and one real GGUF. |

## 3. Include graph of the ported set (what to keep / edit)

| File | Donor include | Target action |
|---|---|---|
| `src/q3_gguf.c` | `q38_gguf.h` + libc (`fcntl/inttypes/stdio/stdlib/string/sys/mman/sys/stat/unistd`) | keep all, rename only |
| `src/q3_memory.c/.h` | `q38_memory.h` -> `q38.h` | keep |
| `src/q3_platform.c` | `q38_platform.h`, `q38_cuda.h` | keep both |
| `src/q3_platform.h` | `q38.h` | keep |
| `src/q3_residency.h` | `q38_gguf.h` | keep |
| `src/q3_residency_plan.h` | `q38_gguf.h` | keep |
| `src/q3_model_loader_cuda.h` | `q38_diagnostics.h` | keep |
| `src/q3_model_loader_cuda.h` | `q38_forward.h` | **delete** (hub header, §2) |
| `src/q3_model_loader_cuda.h` | `q38_qsa_candidate.h`, `q38_directional_steering.h`, `q38_nvfp4_pack.h` | **delete** |
| `cuda/q3_model_loader_cuda.cu` | `q38_forward_cuda.h`, `q38_residency_plan.h`, `q38_diagnostics.h` | keep (renamed) |
| `cuda/q3_model_loader_cuda.cu` | `q38_nvfp4_residency.h`, `q38_cuda_primitives.h`, `q38_gdn.h`, `q38_moe_cuda.h`, `q38_qsa_cuda.h`, `q38_topk_cuda.h`, `q38_gr_ref.h`, `q38_model_config.h` | **delete all 8** (only needed by deleted regions) |
| `cuda/q3_model_loader_cuda.cu` | `cooperative_groups.h`, `cuda_fp16.h` | delete (no kernels left) |
| `cuda/q3_model_loader_cuda.cu` | `cuda_runtime.h`, `math.h`, `stdio.h`, `stdlib.h`, `string.h`, `time.h`, `sys/resource.h`, `sys/mman.h`, `unistd.h` | keep (`sys/resource.h`, `sys/mman.h`, `unistd.h` only for `getrusage`/`mincore` telemetry) |
| `src/q3_main.c` | `q3.h`, `q3_cuda.h`, `q3_gguf.h`, `q3_memory.h`, `q3_platform.h`, `q3_model_loader_cuda.h`, `q3_residency*.h` | keep (after reduction) |
| `src/q3_main.c` | `q38_decode.h`, `q38_diagnostics.h`, `q38_directional_steering.h`, `q38_session.h`, `q38_tokenizer.h`, `q38_weights.h`, `q38_nvfp4_pack.h`, `q38_nvfp4_residency.h`, `math.h` | **delete** |

## 4. DELETE LIST inside `cuda/q3_model_loader_cuda.cu`

| Region (donor symbol / line) | Action |
|---|---|
| `q38_directional_steering_kernel` + `q38_forward_cuda_load/set/apply_directional_steering` (L~320, 1814-1840) | delete |
| MoE: `q38_forward_cuda_expert_backend`, `*_moe_layer_q2_backend`, `*_record_route`, `*_get_expert_layer_calls`, all `q2_gate_up_*`/`q2_down_*` fields | delete |
| GDN: `*_gdn_layer_*`, `*_reset_gdn_state`, `*_load_gdn_state`, `*_sync_gdn_state`, gdn workspace/buffers | delete |
| QSA: `*_qsa_qkv_backend`, `*_qsa_chain_*`, `*_qsa_chain_device`, `*_set_qsa_candidate`, `ensure_qsa_chain_workspace`, `qsa_chain_state[]` | delete |
| GR: `*_gr_read/write_*`, `ensure_gr_buffers`, `is_gr_projection_stage` | delete |
| PLE: all `ple_*` fields, `q38_forward_cuda_set_stage_context`, `Q38_CUDA_SYNC_PLE_STAGE_WAIT` | delete |
| LM head: `*_prepare_lm_head`, `*_prepare_nvfp4_lm_head`, `lm_head_*` fields | delete |
| NVFP4 residency: `*_enable_nvfp4_residency`, `nvfp4_residency`, `nvfp4_model`, `nvfp4_enabled` | delete |
| Compute entrypoints: `*_matvec_backend`, `*_matrix_backend`, `*_matrix_batch_backend`, `*_greedy_argmax`, `*_decoder_layer_chain_backend`, `ensure_buffer`, all other `__global__` kernels | delete |
| Telemetry: `emit_telemetry`, `telemetry_events_*`, `q38_forward_cuda_telemetry` struct + observer, `subsystem_for_stage`, `q38_profile_nvtx_push/pop` externs, `host_now_ms` where unused | delete |
| `is_ple_embedding_table` / `residency_plan_is_ple` | reduce to `static bool residency_plan_is_ple(const q3_tensor *t, void *u){ (void)t;(void)u; return false; }` (M0 §3: no PLE, planner exclusion never fires). Delete `Q3_STORAGE_FILE_BACKED_PLE` and set every `exec->storage = Q3_STORAGE_RESIDENT`. |
| `Q3_CUDA_SYNC_*` enum | reduce to `Q3_CUDA_SYNC_RESIDENCY_INIT` (+ `_COUNT`); drop the other 14 reasons |
| `tensor_shape` | keep verbatim (still used to fill `rows/cols` of `q3_exec_tensor`) |
| KEEP verbatim | `fail`, `copy_tensor_name`, `residency_now_ms`, `residency_mincore_pages`, `record_cuda_sync`, `Q3_CUDA_SYNC_CALL`, `Q3_CUDA_DIAG_*`, `exec_tensor_is_resident`, `persistent_find`, `persistent_tensor`/`q3_exec_tensor` structs, `context_create/destroy`, `*_stream`, `enable_all_non_ple_residency` body, `get_residency_stats`, `get/reset_sync_stats`, `sync_reason_name` |

## 5. New / rewritten glue (content boundary only)

| File | Minimal content boundary |
|---|---|
| `src/q3_main.c` (reduced from `q38.c`) | `usage()` (5 modes, `--json`, `--verbose`), 5 `cmd_*`, arg parse + `switch`. `cmd_load_only` **rewritten** to the GGUF path: `q3_gguf_open` -> `q3_platform_probe(before)` -> `q3_forward_cuda_context_create` -> `q3_forward_cuda_enable_all_non_ple_residency(ctx, model)` -> `q3_forward_cuda_get_residency_stats` -> `q3_platform_probe(after)` + `q3_platform_rss_bytes` -> print telemetry JSON (§13 of M0) -> `context_destroy` -> `q3_gguf_close`. No NVFP4 pack, no `--source-root`, no inference. |
| `Makefile` | Donor variables verbatim (§6) with `q38_`->`q3_`, `-I. -Isrc`, object lists = 6 C objects + 2 CUDA objects, `q3` link rule, 2 test rules. |

## 6. Exact build flags implied by the donor Makefile

| Purpose | Command |
|---|---|
| C compile | `cc -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. -Isrc -pthread -c -o $@ $<` |
| CUDA compile | `$(CUDA_HOME)/bin/nvcc -O3 -g -lineinfo --use_fast_math -I. -Isrc -gencode arch=compute_121a,code=sm_121a -c -o $@ $<` |
| CUDA diag variant | add `-DQ3_DIAGNOSTICS=1` (enables residency telemetry) and `-DQ3_ENABLE_NVTX=1` only if NVTX is re-added |
| Link | `$(CUDA_HOME)/bin/nvcc -O3 -g -lineinfo --use_fast_math -I. -Isrc -gencode arch=compute_121a,code=sm_121a -o q3 $(OBJS) -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -pthread` |
| GGUF test | `cc -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. -Isrc -pthread -o tests/test_q3_gguf tests/test_q3_gguf.c src/q3_gguf.o` |
| Planner test | same, with `src/q3_residency_plan.o` |

`CUDA_HOME` detection block: copy donor lines 16-22 verbatim. Object lists: `C_OBJS := src/q3_gguf.o src/q3_memory.o src/q3_platform.o src/q3_residency.o src/q3_residency_plan.o src/q3_main.o`, `CUDA_OBJS := cuda/q3_cuda.o cuda/q3_model_loader_cuda.o`. Note `-DQ3_DIAGNOSTICS=1` is **required** for `--load-only` timing telemetry (all `residency_*_ms` fields are behind it).

## 7. Tests to copy+rename+reduce

| DONOR | TARGET | CUTS |
|---|---|---|
| `tests/test_gguf.c` | `tests/test_q3_gguf.c` | Only 2 edits: (1) fixture path `"/tmp/q38_test_gguf.bin"` -> `tests/fixtures/q3_test_gguf.bin` written in-tree (/tmp is forbidden); (2) mechanical `q38_`->`q3_`. **Keep**: in-memory GGUF v3 builder, magic/version/n_kv/n_tensors, `q3_gguf_get_string`, `q3_gguf_get_u32`, tensor `elements`/`bytes`/type name, truncated-file and bad-magic failure. **Add**: one explicit `abs_offset + bytes <= size` assertion (M0 §4). **Cut**: nothing else. |
| `tests/test_residency_plan.c` | `tests/test_q3_residency_plan.c` | Mechanical rename only. The PLE-named fixture + `is_ple` callback exercise the *planner's exclusion path*, which is still ported; M0's "never exclude" applies only to the production callback in the loader. **Keep** ordering/spans/totals/exclusion asserts; **add** an explicit overlap/overflow fixture (gap > `max_gap_bytes` splits a span; `UINT64_MAX` bytes must fail). **Cut**: nothing. |

## 8. Ordered port sequence (compile checkpoint after each step)

| # | Step | Checkpoint |
|---|---|---|
| 1 | `mkdir -p src cuda tests/fixtures artifacts/m0_q4_load`; `cp q38_gguf.c/.h` -> `src/q3_gguf.*`; apply sed rename | `cc -c src/q3_gguf.c -Isrc` clean |
| 2 | `cp q38.h` -> `src/q3.h`; reduce to 5 modes / 2 structs / trimmed options | `cc -fsyntax-only src/q3.h` (or via step 3 include) |
| 3 | `cp tests/test_gguf.c` -> `tests/test_q3_gguf.c`; rename + fixture-path fix; add in-tree `tests/fixtures/q3_test_gguf.bin` generation | **T1 green**: `tests/test_q3_gguf` passes |
| 4 | `cp q38_memory.c/.h`, `q38_platform.c/.h`, `q38_cuda.h` -> `src/q3_*`; write `Makefile` with §6 flags | `cc -c src/q3_memory.c src/q3_platform.c` clean |
| 5 | `cp cuda/q38_cuda.cu` -> `cuda/q3_cuda.cu`; reduce to probe/init/cleanup; minimal `src/q3_main.c` with `--platform` only | `nvcc -c cuda/q3_cuda.cu` clean; `./q3 --platform` + `--json` works |
| 6 | Extend `src/q3_main.c` with `--inspect`, `--list-tensors`, `--verbose` (copy donor `cmd_*` + `model_summary`) | `./q3 --inspect <gguf>` reports version/arch/kv/tensors/bytes-by-type with no payload read |
| 7 | `cp q38_residency.c/.h`, `q38_residency_plan.c/.h` -> `src/q3_*`; reduce residency expert banks | `cc -c src/q3_residency.c src/q3_residency_plan.c` clean |
| 8 | Add `--memory-plan` to `src/q3_main.c` (donor body, `cuda_allocated_bytes = planned resident bytes`) | `./q3 --memory-plan <gguf> --json` prints plan |
| 9 | `cp tests/test_residency_plan.c` -> `tests/test_q3_residency_plan.c`; rename + add overlap/overflow fixtures | **T2 green** |
| 10 | `cp cuda/q38_forward_cuda.h` -> `src/q3_model_loader_cuda.h`; delete §3 lines + forward/steering/qsa/nvfp4 decls; reduce stats struct | `cc -fsyntax-only` on a TU including it, no missing symbols |
| 11 | `cp cuda/q38_forward_cuda.cu` -> `cuda/q3_model_loader_cuda.cu`; apply §4 delete list iteratively + `enable_all_non_ple_residency` stays | `nvcc -DQ3_DIAGNOSTICS=1 -c cuda/q3_model_loader_cuda.cu` clean |
| 12 | Rewrite `cmd_load_only` (§5); link `q3`; run on GB10; write `artifacts/m0_q4_load/*` | **T3 green**: `./q3 --load-only <q4.gguf> --json` exit 0, coverage complete, `final_syncs == 1` |

## 9. Loader facts to preserve (measured donor baseline)

| Fact | Donor value | M0 note |
|---|---|---|
| Stage count | 2 (double buffer, `residency_stage_buffers[2]`) | keep |
| Stage sizing | `largest_span` (dynamic), **not** fixed 128 MiB; planner caps `max_span_bytes` at 256 MiB, `max_gap_bytes` 64 KiB | document; M0 §8 says keep donor sizing for the first baseline |
| Stage type | `cudaMallocHost` pinned host stage + paired device `cudaMalloc` transfer buffer, per slot | 4 extra allocations total |
| Transfer shape | `memcpy` mmap->pinned stage, `cudaMemcpyAsync` H2D per span, then `cudaMemcpyAsync` D2D per tensor | `residency_transfer_calls` = spans, `residency_device_copies` = tensors |
| Destination | one `cudaMalloc` per resident tensor (donor design), freed in `context_destroy` | per-tensor, never per-chunk |
| Sync | `cudaEventRecord` per span for stage reuse slot + **1** `cudaStreamSynchronize` at the end, counted in `residency_final_syncs` | must read 1 |
| Telemetry source | `getrusage` minflt/majflt, `mincore` pages before/after, `CLOCK_MONOTONIC_RAW` span timings | keep, `-DQ3_DIAGNOSTICS=1` |
| Resident pointer table | `q3_exec_tensor[]` (bytes/rows/cols/qtype/tensor_id/gguf_offset/name/ptr/storage) | satisfies M0 §11 |