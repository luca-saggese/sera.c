# M1 — primary donor: `luca-saggese/q38.c @ main` (`ds4_mmq_q4_K_dense`)

**Rule applied:** this donor was named primary by the M1 scope correction. Research scope is
**only** the dependency closure of `ds4_mmq_q4_K_dense`. Research stops here.

**Provenance (verified on this machine):**

```
$ git -C _reference/q38-main rev-parse HEAD
c1d4597a80e300b803dc642519718f2c999589da   (branch main, 2026-08-23 15:26:19 +0200)
```

- Donor path: `/home/lvx/sera.c/_reference/q38-main` (read-only — never modified).
- The M0 donor `_reference/q38.c` (branch `qwen38-spark-proto`) was **not** touched and is **not**
  the M1 donor. Its Q4_K inventory is kept for reference in `01b_q38_proto_q4_inventory.md`; it is
  inferior (no Q8_1, no vecdotq, naive scalar MoE kernels only) and is **not** used by M1.
- `_reference/llama.cpp` (`709fe755dfa810d77e2ac386292b29648b536864`) is **provenance only**
  (see `02_q4_cuda_donor.md`). It is **not** compiled or linked.
- License: MIT. `cuda/mmq/VENDOR.md` (donor) pins the vendored upstream to llama.cpp commit
  `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`; keep that attribution.

---

## 1. Entry point

Declared `cuda/mmq/ds4_mmq.h:149-156`, defined `cuda/mmq/ds4_mmq.cu:825-829`:

```c
int ds4_mmq_q4_K_dense(const void *W_q4_K, const float *X_f32, float *out_f32,
                       int M, int N, int K, cudaStream_t stream);
```

`M/N/K` are already the logical geometry M1 wants. It is an `extern "C"` host wrapper around
`ds4_mmq_dense_impl<GGML_TYPE_Q4_K>(tag, W, X, out, M, N, K, stream)` (`ds4_mmq.cu:466-608`).

Semantics (from the impl header comment, `ds4_mmq.cu:411-418`): `out[col, row] = sum_k W[row,k]*X[k,col]`,
`X` is **K-innermost row-major**, `out` is **column-major** (`out[col*M + row]`). M1 §9 wants
`weight=[N,K]`, `input=[M,K]`, `output=[M,N]`; therefore at the binder boundary
`weight_tensor.rows = M1_N`, `weight_tensor.cols = M1_K`, and the MMQ `M` argument is the token
count while `N` is the weight row count. The kernel's own `M` = `nrows_dst`, `N` = `ncols_dst` = `ne11`.

**GEMM vs GEMV, this is the whole dispatch story.** `ds4_mmq_q4_K_dense` is the **MMQ**
(128×128 tile) path only. There is **no dense Q4_K MMVQ entry in the donor**:

| dense vec entry | status |
|---|---|
| `ds4_mmq_q8_0_dense_vec` (`ds4_mmq.h:839`, impl `ds4_mmq.cu:2885`, instantiated `:4495`) | exists, **Q8_0 only** |
| `ds4_mmq_q4_K_dense_vec` | **does not exist** |
| MoE Q4_K vec entries (`ds4_mmq_q4_K_moe_vec` `:463`, `_moe_pair_vec` `:783`, `_moe_gate_up_mid_vec` `:721`, `_moe_pair_raw_vec` `:819`, `_moe_down_sum6_vec` `:658`) | exist, all route through `mul_mat_vec_q_switch_type` + `ids` |

So M=1 in M1 is served by the **same MMQ kernel** in the donor (MMQ handles `nrows_dst`=1 fine;
`use_stream_k` is enabled for NVIDIA cc ≥ VOLTA). Adding a true dense Q4_K MMVQ entry is an
**optional M1 optimization**, not a donor port: it is `ds4_mmq_dense_vec_impl` (`:2885-3007`) with
`Q8_0` swapped for `Q4_K` plus a one-line `extern "C"` wrapper and one template instantiation. That
function's only Q8_0-specific parts are the type template parameter and `ggml_blck_size(type)`;
`quantize_row_q8_1_cuda` and `mul_mat_vec_q_switch_type` are already type-generic. **Cheapest correct
plan: port `ds4_mmq_dense_vec_impl<T>` with both `Q8_0` and `Q4_K` instantiations; the `Q4_K`
instantiation is M1's M≤8 path at essentially zero porting cost.**

---

## 2. Required files — exact minimal closure

Build flags (from donor `Makefile:326-351`): `-O3 --use_fast_math -std=c++17 -Icuda/mmq`
plus `-gencode arch=compute_121a,code=sm_121a`, and `-DDS4_CUDA_HAVE_MXF4=1` (MXFP4-only; M1 can
drop it if MXFP4 branches are trimmed).

| # | donor file | LOC | port as | keep / trim |
|---|---|---|---|---|
| 1 | `cuda/mmq/ds4_mmq.cu` | 4512 | `cuda/q3_mmq.cu` | **trim hard** — keep only `ds4_mmq_dense_impl<T>` + a `dense_vec_impl<T>`; delete everything §8 lists |
| 2 | `cuda/mmq/ds4_mmq.h` | 857 | `cuda/q3_mmq.h` | keep ~40 lines: the Q4_K entries + `ds4_mmq_init` + pool-stream setter |
| 3 | `cuda/mmq/mmq.cuh` | 4415 | `cuda/vendor/mmq.cuh` | **keep verbatim** (vendored llama.cpp) |
| 4 | `cuda/mmq/vecdotq.cuh` | 1317 | `cuda/vendor/vecdotq.cuh` | keep verbatim (DT_Q4_K / vmmq / mmq impls all needed) |
| 5 | `cuda/mmq/mma.cuh` | 1456 | `cuda/vendor/mma.cuh` | keep verbatim (included unconditionally by `mmq.cuh`) |
| 6 | `cuda/mmq/common.cuh` | 1489 | `cuda/vendor/common.cuh` | keep verbatim |
| 7 | `cuda/mmq/quantize.cu` | 443 | `cuda/vendor/quantize.cu` | keep; drop the `quantize_mmq_fp4_cuda` entry if MXFP4 is trimmed (it is 28 lines) |
| 8 | `cuda/mmq/quantize.cuh` | 41 | `cuda/vendor/quantize.cuh` | keep verbatim |
| 9 | `cuda/mmq/mmvq.cu` | 1183 | `cuda/vendor/mmvq.cu` | keep; remove `#include "unary.cuh"` and the GLU-fusion epilogue (3 call sites) |
| 10 | `cuda/mmq/mmvq.cuh` | 34 | `cuda/vendor/mmvq.cuh` | keep verbatim (also holds `MMVQ_MAX_BATCH_SIZE 8`) |
| 11 | `cuda/mmq/ggml-common.h` | 1900 | `cuda/vendor/ggml-common.h` | keep (block structs, `QK_K`, `QK8_1`, static_asserts) |
| 12 | `cuda/mmq/vendors/cuda.h` | 28 | `cuda/vendor/vendors/cuda.h` | keep verbatim |
| 13 | `cuda/mmq/ds4_ggml_stubs.h` | 303 | `cuda/vendor/q3_ggml_stubs.h` | keep — this **is** the ggml-runtime-free shim |
| 14 | `cuda/mmq/ds4_ggml_stubs.cu` | 160 | `cuda/vendor/q3_ggml_stubs.cu` | keep; rename `ds4_naive_pool` → `q3_naive_pool` |

**Total ≈ 15.6k LOC, of which ≈ 13.2k is verbatim vendored llama.cpp** (`common.cuh`, `mma.cuh`,
`vecdotq.cuh`, `mmq.cuh`, `quantize.*`, `mmvq.*`, `ggml-common.h`, `vendors/cuda.h`) and only
**≈ 2.4k LOC is q38-authored adapter** (`ds4_mmq.*`, `ds4_ggml_stubs.*`) — and M1 needs roughly
**300 lines of that adapter** (one dense MMQ template + one dense vec template + wrappers).

### Verified: no ggml runtime needed, and it compiles here

`ds4_ggml_stubs.h` is 5-line-redirect-proof (`ggml.h`, `ggml-impl.h`, `ggml-cuda.h` in
`cuda/mmq/` are redirects into it). Verified on this GB10:

```
$ nvcc -O3 --use_fast_math -gencode arch=compute_121a,code=sm_121a \
       -DDS4_CUDA_HAVE_MXF4=1 -std=c++17 -Icuda/mmq \
       -c cuda/mmq/ds4_ggml_stubs.cu -o /tmp/.../ds4_ggml_stubs.o     # 4.1 s, clean
$ nvcc ... -c cuda/mmq/ds4_mmq.cu    -o /tmp/.../ds4_mmq.o           # 3 m 39 s, clean
```

No `ggml_backend_cuda_context` from ggml is linked: the stub provides a concrete
`ggml_backend_cuda_context` class and `ds4_naive_pool : ggml_cuda_pool`. **No part of the ggml
graph / scheduler / allocator / model loader / CPU backend is required.** This satisfies the
"STOP if the closure needs a substantial part of the ggml runtime" criterion — it does not.

---

## 3. Required symbols (the real closure list)

Host-side (`ds4_mmq.cu`):

- `ds4_mmq_dense_impl<GGML_TYPE_Q4_K>` (`:466-608`) — the pipeline to port.
- `get_ctx_for_device(int)` (`:443-451`) — per-device singleton `ggml_backend_cuda_context`.
- `ds4_pool_set_stream(cudaStream_t)` — routes pool `cudaMallocAsync`/`cudaFreeAsync` onto the caller stream.
- `ds4_mmq_k_tile_supported<T>` (`:453-462`) — for Q4_K it is `return true` (only MXFP4/NVFP4 checks).
- `ds4_mmq_sanitize_f32` (`:427-440`) — NaN/Inf scrub kernel on the output.
- `ybuf_memset` (`:213-231`) / `ybuf_memset_mode` / `out_memset_enabled` (`:187`) — env-gated determinism knobs (`DS4_MMQ_YBUF_MEMSET`, `DS4_MMQ_OUT_MEMSET`). **Keep the code; the env-var tuning hooks are removable** (hardcode the safe default = memset).
- `mul_mat_q_case<GGML_TYPE_Q4_K>` — explicit instantiation at `:4509-4510`; declared in `mmq.cuh:4295`.
- `ggml_cuda_pool_alloc<char>` — from the stub.

Device-side, from `mmq.cuh` (all reachable from `mul_mat_q_case<T>`, keep verbatim):

- `mmq_args` struct; `launch_mul_mat_q<T,mmq_x>` (`:4180`); `mmq_get_nbytes_shared` (`:4171`);
  `mul_mat_q<T,mmq_x,need_check>` (`:3758`); `mul_mat_q_stream_k_fixup` (`:4015`);
  `mul_mat_q_process_tile` (`:3642`); `mmq_write_back_{dp4a,mma}`.
- Host geometry helpers: `get_mmq_x_max_host`, `get_mmq_y_host`, `get_mmq_y_device` (`:177`),
  `get_mmq_x_max_device` (`:140`), `get_iter_k` (`:168`), `mmq_get_dp4a_tile_x_sizes` (`:213`),
  `mmq_get_mma_tile_x_k` (`:258`), `mmq_get_nwarps_host`/`_device` (`:328`),
  `mmq_get_granularity_device`, `mmq_get_q8_1_ds_layout`, `ggml_cuda_highest_compiled_arch`.
- Types: `block_q8_1_mmq` (`mmq.cuh:30`), `block_fp4_mmq` (`:54`), `tile_x_sizes` (`:105`), `QK_FP4_MMQ` (`:17`), `MMQ_ITER_K 256` (`:13`), `MMQ_DP4A_MAX_BATCH_SIZE 64` (`:12`).
- `unpack_scales_q45_K` (`mmq.cuh:2201`); tile loaders `load_tiles_*` in `mmq.cuh`.

From `vecdotq.cuh`: `get_int_b2` (`:18`), `get_int_b4` (`:27`), `vec_dot_q4_K_q8_1` (`:918`),
`vec_dot_q4_K_q8_1_impl_vmmq` (`:508`), `vec_dot_q4_K_q8_1_impl_mmq` (`:533`).

From `quantize.cu`: `quantize_mmq_q8_1_cuda` (`:385`) for MMQ, `quantize_row_q8_1_cuda` (`:369`)
for MMVQ. `CUDA_QUANTIZE_BLOCK_SIZE 256`, `CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128` (`quantize.cuh`).

From `mmvq.cu`: `mul_mat_vec_q_switch_type` (`:901`; q38 promoted it from `static` — keep),
`get_mmvq_mmid_max_batch` (`:241`), `ggml_cuda_mm_fusion_args_device` (stub).

**`mmid.cu`/`mmid.cuh` are NOT needed for dense Q4_K.** `mmid.cuh` is a 9-line declaration of
`ggml_cuda_launch_mm_ids_helper` used only by the `ids`-carrying MoE paths; the only references in
`ds4_mmq.cu` are inside `_moe_*_impl` bodies (`:949`, `:1350`). `nm -u ds4_mmq.o` shows
`U ggml_cuda_launch_mm_ids_helper` as an unresolved symbol that is **never reached** from
`ds4_mmq_dense_impl` — when trimming to dense-only, delete the MoE bodies and this symbol vanishes
along with `ds4_mmid_large_enabled`. **Do not port `mmid.cu` (339 LOC).**

**`ds4_mmq_d2r.cu` (3370 LOC) is not needed.** 62 references in `ds4_mmq.cu`, all in
`ds4_mmq_*_d2r*` entries (§8 removal list). Drop the `.cuh` include too.

---

## 4. Q8_1 activation quantization path

Both regimes quantize the activation; neither supports a raw FP32 activation, so M1 §10's
`FP32 → Q8_1 → GEMM → FP32` pipeline is exactly what the donor does.

**MMQ (`M` large, `ds4_mmq_dense_impl`) — interleaved `block_q8_1_mmq`:**

```c
const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);   // MATRIX_ROW_PADDING = 512 (common.cuh:151)
const size_t y_block_size = sizeof(block_q8_1_mmq);            // 4*QK8_1 bytes
const size_t y_values_per_block = 4 * QK8_1;                   // 128
nbytes_src1_q8_1 = ne13*ne12*ne11*ne10_padded * y_block_size / y_values_per_block
                 + get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);   // tail-tile over-read guard
ybuf_memset(src1_q8_1.get(), nbytes_src1_q8_1, stream);   // REQUIRED: deterministic tail
quantize_mmq_q8_1_cuda(X, nullptr, src1_q8_1, type, /*ne00=*/K, s11=K, s12=0, s13=0,
                       ne0=ne10_padded, ne1=ne11, ne2=ne12, ne3=ne13, stream);
```

The `ybuf_memset` is not cosmetic — it is the documented S1.1a correctness fix
(`ds4_mmq.cu:534-547`): `quantize_mmq_q8_1_cuda` writes only `ne11` valid columns while the kernel
unconditionally loads the full column tile; pool reuse makes the unwritten tail non-deterministic.
**M1 must keep this memset.** Also required in the port: `static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0)` from `quantize.cuh`.

**MMVQ (`M ≤ 8`) — canonical `block_q8_1` layout**, via `quantize_row_q8_1_cuda` (`ds4_mmq.cu:2918-2930`):

```
ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING)
nbytes_q8_1 = N * ne10_padded * sizeof(block_q8_1) / QK8_1
quantize_row_q8_1_cuda(X, nullptr, buf, type, K, s11=K, s12=K*N, s13=K*N,
                       ne0=ne10_padded, ne1=N, ne2=1, ne3=1, stream)
```

Note `s12 = K*N` (F32 element stride), **not** the byte stride used by the MMQ variant. The two
outputs are different memory layouts; do not reuse one buffer for both.

---

## 5. MMVQ path used for small N (and small M)

Upstream policy literal (donor comment `ds4_mmq.cu:2047-2058`): *"mmvq is upstream's matrix-vector
matmul family, optimised for the `n_tokens <= MMVQ_MAX_BATCH_SIZE=8` regime"*.

- `MMVQ_MAX_BATCH_SIZE = 8` (`mmvq.cuh:4`).
- Per-arch cap `col_cap = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc))`
  (`mmvq.cu:241-250`); on TURING_PLUS (`ct >= 750`) the Q4_K value comes from
  `get_mmvq_mmid_max_batch_turing_plus` (`mmvq.cu:133-144`). Columns beyond `col_cap` are split into
  `ceil(n_tokens/col_cap)` launches rather than rejected (FD Inc2a, `ds4_mmq.cu:2221-2260`).
- Grid: `(ceil(nrows_x/c_rows), ncols_dst)`; `kby = kbx * (qk / QK8_1)`; `kqs = vdr*(tid % (qi/vdr))`
  with `get_vdr_mmvq` per type (`VDR_Q4_K_Q8_1_MMVQ`).
- Dense Q8_0 usage contract (`ds4_mmq.cu:2969-3007`) is the template for the Q4_K dense variant:
  `ncols_dst = N`, `nchannels_y = nchannels_dst = 1`, `stride_col_y = ne10_padded/QK8_1`,
  `stride_channel_y = N * (ne10_padded/QK8_1)`, `stride_col_dst = M`, then
  `cudaMemsetAsync(out,0,M*N*sizeof(float))` before the launch and `ds4_mmq_sanitize_f32` after.
- **DGX-Spark-specific branch exists** in `mmvq.cu`: `#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
  → mmvq_should_prefetch()` + `mmvq_prefetch_l2()`, `pf_dist = 2`. `common.cuh:56` defines
  `GGML_CUDA_CC_DGX_SPARK 1210`, and GB10 is SM 12.1 → **this branch is live for us**. It is a
  weight-load prefetch, i.e. it targets exactly M1's memory-bound M=1 case. Keep it; it may be why
  a dense Q4_K MMVQ entry is worth the ~50 lines.

## 6. MMQ path used for larger N/M

`ds4_mmq_dense_impl` (`ds4_mmq.cu:466-608`), step by step:

1. Validate: non-null `W/X/out`; `M,N,K > 0`; **`K % 256 == 0`** (K-quant super-block). M1 §13: fail
   loudly here, never fall back.
2. `ds4_mmq_k_tile_supported<Q4_K>(tag,K,cc)` → always true for Q4_K.
3. `get_ctx_for_device(dev)` → `ctx`; `ds4_pool_set_stream(stream)`.
4. `ne10_padded = GGML_PAD(K, 512)`; allocate `src1_q8_1` from `ctx->pool()`.
5. `ybuf_memset` the whole Y buffer (see §4).
6. `quantize_mmq_q8_1_cuda(...)` → `cudaGetLastError()` → return `-2` on failure.
7. Geometry: `blck = ggml_blck_size(Q4_K) = 256`; `s01 = K/blck` (**weight blocks per row**);
   `s1 = M`; `s12 = ne11*ne10_padded*y_block_size/(y_values_per_block*sizeof(int))`;
   `s13 = ne12*s12`.
8. `use_stream_k = (CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= CC_VOLTA) || CC_IS_CDNA(cc)`
   → **true on GB10**.
9. `out_memset_enabled()` → `cudaMemsetAsync(out,0,M*N*sizeof(float),stream)`.
10. Build `mmq_args` (see below) and call `mul_mat_q_case<type>(*ctx,args,stream)`.
11. `cudaGetLastError()` → `-3` on failure; then `ds4_mmq_sanitize_f32(out, M*N, stream)`.

`mmq_args` field order (`mmq.cuh`, trimmed of the two DS4-only trailing fields):

```
x, type_x, y, ids_dst=nullptr, expert_bounds=nullptr, dst,
ncols_x=K, nrows_x=M, ncols_dst=N, stride_row_x=K/256, ncols_y=N, nrows_dst=M,
nchannels_x=1, nchannels_y=1, stride_channel_x=0, stride_channel_y=s12, stride_channel_dst=0,
nsamples_x=1, nsamples_y=1, stride_sample_x=0, stride_sample_y=s13, stride_sample_dst=0,
use_stream_k, ncols_max=N
```

**DS4-specific removal from `mmq_args`:** q38-main appended `const char * x_soa; int64_t soa_blocks;`
for the DS4 P4 Inc3 aligned-SoA artifact (Q2_K / IQ2_XXS only). For Q4_K dense both are unused →
**delete them from our copy of the struct** (and from `mmq.cuh`'s uses, which are behind
`if constexpr (type == Q2_K || type == IQ2_XXS)`).

## 7. Workspace / scratch requirements

| buffer | size | lifetime in donor | M1 requirement (§14: allocate once, reuse) |
|---|---|---|---|
| `src1_q8_1` (MMQ Y) | `N*GGML_PAD(K,512)*sizeof(block_q8_1_mmq)/(4*QK8_1) + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq)` | `ggml_cuda_pool_alloc<char>` **per call** | promote to a **q3 persistent workspace** grown on demand |
| `src1_q8_1` (MMVQ Y) | `N*GGML_PAD(K,512)*sizeof(block_q8_1)/QK8_1` | per call (`:2922`) | same persistent workspace (MMQ layout is the larger of the two) |
| stream-K fixup scratch | `mmq_get_nbytes_shared()`-derived, inside `mul_mat_q_case` → pool | per call | inside donor; hidden pool alloc. **Verify with `cuda_allocations_per_linear == 0` (§14)** — if the donor pool allocates per call, pre-reserve via the pool and/or pass our workspace |
| `out` | caller-owned | — | q3-owned |
| optional persistent Q8_1 scratch | `g_q81_scratch_*` (`ds4_mmq.cu:116-129`, `ds4_mmq_set_aligned_q81_scratch`) | DS4 CUDA-graph workaround | **keep the mechanism** — it is exactly the "allocate once, reuse" shape M1 wants; drop the graph rationale |

DS4 pool: `ds4_naive_pool` in `ds4_ggml_stubs.*` (concrete `ggml_cuda_pool` using
`cudaMallocAsync`/`cudaFreeAsync` on the thread-local stream set by `ds4_pool_set_stream`).
For M1 replace its allocator with a **q3 bump/persistent arena** so hot calls allocate nothing.

## 8. CUDA architecture dispatch

The donor **removed the upstream per-arch config-table dispatch**: there is no
`ggml_cuda_mmq_get_config`, no `mmq-config-*.cuh`, no `mul_mat_q_switch_config`. Instead
`mmq.cuh` derives everything from `__CUDA_ARCH__` at compile time plus these host helpers:

- `get_mmq_x_max_host(cc)` / `get_mmq_x_max_device()`, `get_mmq_y_host(cc)` / `get_mmq_y_device()`
- `mmq_get_nwarps_host(cc, mmq_x, mmq_y)` (`mmq.cuh:328`) / `_device`
- `mmq_get_dp4a_tile_x_sizes(type, mmq_x)` (`:213`) / `mmq_get_mma_tile_x_k(type, mmq_y)` (`:258`)
- `mmq_get_granularity_device(cc)`; `ggml_cuda_highest_compiled_arch(cc)`
- `blackwell_mma_available(cc)` (relevant only to MXFP4/NVFP4 → trimmed)

**Consequence for M1: only ONE architecture is compiled** — `-gencode arch=compute_121a,code=sm_121a`.
GB10 is cc 12.1 = `GGML_CUDA_CC_DGX_SPARK 1210` (`common.cuh:56`) and is ≥ VOLTA ≥ TURING, so
`use_stream_k = true`, the DGX-Spark MMVQ prefetch branch is active, and Q4_K MMA uses the
Turing+ path. **No multi-arch fatbin, no runtime cc table needed** — this directly satisfies the
review's "no generic multi-architecture dispatch non necessary to GB10".

## 9. DS4-specific dependencies to remove

From `ds4_mmq.cu` (4512 LOC → target ≈ 250-350 LOC for dense Q4_K only):

| removal | LOC region |
|---|---|
| MoE: `ds4_mmq_*_moe_*` incl. `_id`, pair, gate/up/mid, down_sum6, raw-variant bodies | `:850-1180`, `:1181-1785`, `:3592+` |
| `mmid.cu`/`mmid.cuh` usage, `ds4_mmid_large_enabled`, `ggml_cuda_launch_mm_ids_helper` | `:26`, `:949`, `:1350` |
| D2R ("aligned artifact") path — `ds4_mmq_d2r.cu`/`.cuh` + all `*_d2r*` entries | `:27` + all d2r call sites |
| Q2_K / IQ2_XXS / MXFP4 / NVFP4 dense entries and instantiations | `:800-840`, `:4503-4513` (keep only Q4_K, optionally Q8_0) |
| DS4 P4 Inc3 aligned-SoA (`x_soa`, `soa_blocks`) | `mmq.cuh` Q2_K/IQ2_XXS paths + `mmq_args` fields |
| `ds4_mmq_fused_down`, `ds4_swiglu_weighted_f32`, fused expert gate/up (Marco Palaferri / xangel82 port) | header attribution block `:4-8` |
| DS4 env tuning hooks (`DS4_MMID_LARGE`, `DS4_MMQ_YBUF_MEMSET`, `DS4_MMQ_OUT_MEMSET`, `g_q81_scratch_enabled` env parse) | `:116-231`, `:313-316` |
| CUDA-graph / D2R graph experiments | `ds4_mmq_set_aligned_q81_scratch` rationale, stream-compat comments |
| NVTX instrumentation not needed for M1 | grep `nvtx` in `cuda/mmq` |
| `unary.cuh` (114 LOC, included solely for GLU fusion epilogue) | `mmvq.cu:3`, `:578-584` |
| `quantize_mmq_fp4_cuda` (MXFP4-only) | `quantize.cu:415` |

Do **not** remove: the `ybuf_memset` determinism fix, `ds4_mmq_sanitize_f32`, `use_stream_k`,
`out_memset`, the DGX-Spark prefetch branch, or the `MATRIX_ROW_PADDING` static_asserts.

## 10. Conclusion — what q38-main provides vs. what M1 must add

**q38-main provides (copy/rename/trim only):**
`ds4_mmq_q4_K_dense` = the complete Q4_K MMQ pipeline (Q8_1 activation quant → `mmq_args` →
`mul_mat_q_case<Q4_K>` → sanitize), all Q4_K block semantics and vecdots, the arch-agnostic
`mmq.cuh` with no config table, the fully ggml-runtime-free stubs, and a re-runnable parity test
(`cuda/mmq/test/test_mmq_parity.cu`, 1245 LOC). Verified compiling clean for `sm_121a` with CUDA 13.0.

**M1 must add (small, wiring-level):**
1. A dense Q4_K MMVQ entry (M ≤ 8) — clone `ds4_mmq_dense_vec_impl<T>` with `T = GGML_TYPE_Q4_K`
   (~50 LOC + 1 wrapper + 1 instantiation). Optional; the MMQ path alone is numerically correct.
2. Persistent workspace instead of per-call pool allocs, so hot-path `cuda_allocations == 0`.
3. `q3_cuda_q4k_linear()` ABI: bind `q3_exec_tensor` → `(W, M, N, K, stream)`, convert GGUF physical
   dims → logical N,K once, own the stream/workspace/lifecycle.
4. Stats plumbing (`q3_q4_linear_stats`) around the donor's two timed stages (activation quant,
   kernel) — the primitive itself must not synchronize (§15); the harness times.
5. CPU Q4_K oracle — reuse the M0/proto donor inventory (`01b_*.md` §4) `dequant_q4` +
   `scale_min_q4` + `q38_half_to_float`, or the identical `ds4q_get_scale_min_k4` in the vendored
   `ggml-common.h`/quants path. The vendored `block_q4_K` layout is **144 B,
   `{ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128];}`** — byte-identical to the
   GGUF physical layout, so the oracle reads the resident bytes directly.

**Nothing is reimplemented.** M1 is a port/trimming/wiring exercise. Research stops here.