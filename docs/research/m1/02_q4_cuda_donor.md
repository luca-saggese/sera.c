# M1 / Worker B — llama.cpp Q4_K CUDA donor dependency-closure map

**Donor (read-only):** `/home/lvx/sera.c/_reference/llama.cpp`
**Commit (verified):** `git -C /home/lvx/sera.c/_reference/llama.cpp rev-parse HEAD` → `709fe755dfa810d77e2ac386292b29648b536864`
**Target:** sera.c M1 production-grade Q4_K CUDA linear (Q4_K weights × activations → fp32) on NVIDIA GB10, SM 12.1 (`compute_121a, sm_121a`), aarch64, CUDA 13.0.
**All line numbers below are 1-based and refer to that donor commit.**

Everything is read-only research. No donor file is modified.

---

## 0. TL;DR routing for GB10 (cc == 1210 == `GGML_CUDA_CC_DGX_SPARK`)

- M == 1 (and until M <= 8): **MMVQ** — `mul_mat_vec_q<GGML_TYPE_Q4_K, ...>`, dp4a vecdot.
- M >  8: **MMQ** — `mul_mat_q_case<GGML_TYPE_Q4_K>` → `launch_mul_mat_q` → `mul_mat_q` (Turing+ MMA data layout path on GB10).
- Both regimes **require Q8_1 activation quantization** (no fp32-activation Q4_K variant exists).

Routing literals in `ggml/src/ggml-cuda/ggml-cuda.cu`:
- `:1823` `static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, ...)`
- `:1868` `if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) { ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst); return; }`
- `:1872` `if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts=*/0)) { ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst); return; }`
- `:1875` (else) `ggml_cuda_mul_mat_cublas(...)`.

---

## 1. Type layouts / constants — `ggml/src/ggml-common.h` + `ggml/include/ggml.h`

| file | symbol | line | value / why needed |
|---|---|---|---|
| `ggml/src/ggml-common.h` | `#define QK_K` | 89 | `256`. Super-block size; every Q4_K loop bound. |
| `ggml/src/ggml-common.h` | `#define K_SCALE_SIZE` | 90 | `12`. Bytes of packed 6-bit scales+mins in `block_q4_K`. |
| `ggml/src/ggml-common.h` | `#define QR4_K` | 134 | `2`. Quantization "range" used by vecdot loops (`QR4_K*VDR/...`). |
| `ggml/src/ggml-common.h` | `#define QI4_K (QK_K/(4*QR4_K))` | 133 | `32`. int4s per block-row in MMQ tile math (TXS_Q4_K). |
| `ggml/src/ggml-common.h` | `#define QK8_1` | 258 | `32`. Elements per `block_q8_1`. |
| `ggml/src/ggml-common.h` | `#define QR8_1` | 125 | `1`. |
| `ggml/src/ggml-common.h` | `#define QI8_1 (QK8_1/(4*QR8_1))` | 124 | `8`. |
| `ggml/src/ggml-common.h` | `block_q4_K` | 325–337 | see layout below. |
| `ggml/src/ggml-common.h` | `static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, ...)` | 338 | **= 144 bytes**. |
| `ggml/src/ggml-common.h` | `block_q8_1` | 259–268 | see layout below. |
| `ggml/src/ggml-common.h` | `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, ...)` | 269 | **= 36 bytes**. |
| `ggml/include/ggml.h` | `GGML_TYPE_Q8_0 = 8` | 398 | id (context; Q8_1 sits above). |
| `ggml/include/ggml.h` | `GGML_TYPE_Q8_1 = 9` | 399 | activation quant id. |
| `ggml/include/ggml.h` | `GGML_TYPE_Q4_K = 12` | 402 | weight id — the switch discriminator. |
| `ggml/include/ggml.h` | `GGML_TYPE_COUNT = 43` | 433 | upper bound (config sentinel uses `GGML_TYPE_COUNT`). |

### Exact byte layouts

`block_q4_K` (144 B):
```
offset 0   : union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; }  // 4 B
offset 4   : uint8_t scales[K_SCALE_SIZE];   // 12 B, 6-bit packed scales+mins
offset 16  : uint8_t qs[QK_K/2];             // 128 B, 4-bit quants
total      : 144 B
```
`block_q8_1` (36 B):
```
offset 0   : union { struct { ggml_half d; ggml_half s; }; ggml_half2 ds; }  // 4 B  (s = d*sum(qs))
offset 4   : int8_t qs[QK8_1];               // 32 B
total      : 36 B
```

### Header prerequisites (IMPORTANT)
`ggml-common.h` gates its aggregate/extension macros on `GGML_COMMON_DECL_CUDA` + `GGML_COMMON_IMPL_CUDA` (they must be `#define`d **before** the include; done in `common.cuh:16–17`). For a trimmed port, either keep those two defines or strip `GGML_EXTENSION`/`GGML_COMMON_AGGR_*` down to plain C (`union { struct {...}; ggml_half2 dm; };`). `ggml_half`/`ggml_half2` come from CUDA `__half`.

### CPU oracle
`ggml/src/ggml-quants.c:1529` `dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k)` (decl `ggml/src/ggml-quants.h:60`) — independent reference for the `get_scale_min_k4` unpacking; use to validate the device kernel.

---

## 2. Device dequantize — `ggml/src/ggml-cuda/dequantize.cuh`

| file | symbol | line | why needed |
|---|---|---|---|
| `dequantize.cuh` | `get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m)` | 177 | unpacks packed 6-bit scale/min pair; shared by vecdot + dequant. |
| `dequantize.cuh` | `template<typename dst_t> dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid)` | 187 | **32-thread** block dequant; uses `__low2half/__high2half(x[ib].dm)` and `ggml_cuda_cast<dst_t>`. `ggml_cuda_cast` lives in `convert.cuh`. |
| `dequantize.cuh` includes | `common.cuh`, `convert.cuh` | 1–3 | `convert.cuh` pulls `common.cuh`. |
| `convert.cu` | `static __global__ void dequantize_block_q4_K(...)` | 156 | grid-stride wrapper calling `dequantize_q4_K(..., threadIdx.x)` (call at line 159). |
| `convert.cu` | `static void dequantize_row_q4_K_cuda(...)` | 300 | launcher: `nb = k/QK_K`; `dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y)` at 302. |

Port note: `ggml_cuda_cast` (in `convert.cuh`) is trivial (`static_cast` + fp16 special-case). If fp32-only, `dequantize_q4_K<float>` needs no cast helper.

---

## 3. Vecdot machinery — `ggml/src/ggml-cuda/vecdotq.cuh`

| file | symbol | line | why needed |
|---|---|---|---|
| `vecdotq.cuh` | `#define VDR_Q4_K_Q8_1_MMVQ` | 504 | `2` — vec-dot ratio (int-words consumed per dot). |
| `vecdotq.cuh` | `#define VDR_Q4_K_Q8_1_MMQ` | 505 | `8` — MMQ tile variant. |
| `vecdotq.cuh` | `vec_dot_q4_K_q8_1_impl_vmmq(const int *v, const int *u, const uint8_t *sc, const uint8_t *m, const half2 &dm4, const float *d8)` | 508 | MMVQ kernel body for one QK_K vs a 32-elem q8_1; loops `QR4_K`; **uses `ggml_cuda_dp4a`**; returns `dm4f.x*sumf_d - dm4f.y*sumf_m`. |
| `vecdotq.cuh` | `vec_dot_q4_K_q8_1_impl_mmq(const int*, const int*, const uint8_t *sc, const uint8_t *m, const half2 &dm4, const half2 *ds8)` | 533 | MMQ dp4a per-tile body; loop bound `QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1` (=2). |
| `vecdotq.cuh` | `vec_dot_q4_K_q8_1(const void *vbq, const block_q8_1 *bq8_1, const int & kbx, const int & iqs)` | 918 | **entry point** for the MMVQ `vec_dot_q_cuda_t` table; branchless scale decode (936–964) then `impl_vmmq` (call at 965). |

**Arithmetic variant:** Q4_K is **dp4a-only** (integer `__dp4a`). There is **no** `_fp16` / `_half2` *arithmetic* variant of `vec_dot_q4_K_q8_1`. `half2` appears only as the packed `dm`/`ds` container (`bq4_K->dm`, `bq8_1->ds`). `ggml_cuda_dp4a` (`common.cuh:720`) lowers to `__dp4a` whenever `__CUDA_ARCH__ >= GGML_CUDA_CC_DP4A (610)` — always true on SM 12.1.

MMQ uses the **MMA data layout** path on GB10 (see §5): the per-tile dot is `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma` in `mmq-vec-dot.cuh:313` (ldmatrix + `mma()`), **not** the dp4a `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a` (`mmq-vec-dot.cuh:905`). The dp4a variant is still needed for the `fallback=false/true` non-MMA configs and as the correctness reference.

---

## 4. MMVQ path — `ggml/src/ggml-cuda/mmvq.cu` (+ `mmvq.cuh`)

| file | symbol | line | why needed |
|---|---|---|---|
| `mmvq.cuh` | `#define MMVQ_MAX_BATCH_SIZE` | 3 | `8` — generic MMVQ M-threshold. |
| `mmvq.cu` | `get_vec_dot_q_cuda(ggml_type)` | 40 | dispatch table; `case GGML_TYPE_Q4_K: return vec_dot_q4_K_q8_1;` at **53**. |
| `mmvq.cu` | `get_vdr_mmvq(ggml_type)` | 69 | `case GGML_TYPE_Q4_K: return VDR_Q4_K_Q8_1_MMVQ;` at **82**. |
| `mmvq.cu` | `enum mmvq_parameter_table_id { GENERIC=0, TURING, GCN, RDNA2, RDNA3_0, RDNA4, GB10 }` | 97–103 | **`MMVQ_PARAMETERS_GB10`** is the GB10 tuning id. |
| `mmvq.cu` | device `get_device_table_id()` → `#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK ... return MMVQ_PARAMETERS_GB10;` | ~112–118 | **selects GB10 table only when compiled for arch 1210.** |
| `mmvq.cu` | host `get_device_table_id(cc)` → `return MMVQ_PARAMETERS_GB10;` | ~141 | host-side mirror. |
| `mmvq.cu` | `get_mmvq_mmid_max_batch_*(type)` tables incl. GB10 entries | 147+ | tuning of the `halve_iters`/small_k heuristics. Q4_K rows at 167/205/233/251/274. |
| `mmvq.cu` | `calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters)` | 452 | GB10 branch at 555: for `ncols_dst==1 && halve_iters`, Q4_K (565) returns `2*generic`. |
| `mmvq.cu` | `calc_rows_per_block(ncols_dst, table_id, small_k, nwarps)` | 579 | `ncols_dst==1 → small_k?nwarps:1`; `2..8 → 2`. |
| `mmvq.cu` | `__global__ void mul_mat_vec_q<ggml_type type,int ncols_dst,bool has_fusion,bool small_k=false,bool halve_iters=false>(...)` | 599–601 | **the kernel**; `__launch_bounds__(calc_nwarps(...)*warp_size, 1)` at 600. |
| `mmvq.cu` | `calc_launch_params(...)` | 999 | `nblocks=(nrows_x+rpb-1)/rpb; dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens); dim3 block_dims(warp_size, nwarps, 1);` (1002–1003). |
| `mmvq.cu` | `mul_mat_vec_q_switch_ncols_dst<type>` | 1075 | splits on ncols_dst; launch sites 1190–1257. |
| `mmvq.cu` | type switch, `case GGML_TYPE_Q4_K: mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>` | **1343–1344** | Q4_K instantiation entry. |
| `mmvq.cu` | `void ggml_cuda_mul_mat_vec_q(...)` | 1421 | **host entry** (workspace alloc + quant + launch); workspace alloc around 1504; `quantize_row_q8_1_cuda(...)` at **1506**. |
| `mmvq.cu` | `ggml_cuda_op_mul_mat_vec_q(...)` | 1538 | non-fused wrapper (not needed by a thin adapter). |

### MMVQ launch config (GB10)
- `warp_size = 32`; `nwarps = calc_nwarps(GGML_TYPE_Q4_K, ncols_dst, MMVQ_PARAMETERS_GB10, small_k, halve_iters)`.
- block = `dim3(warp_size, nwarps, 1)`; grid = `dim3(ceil(nrows_x/rpb), nchannels_dst=1, nsamples=1)`.
- Kernel args: `vx_ptr` (Q4_K weights), `vy_ptr` (Q8_1 activations), `ids_ptr` (nullptr here), `fusion` struct, `dst_ptr` (fp32), `ncols_x (=K)`, `nchannels_y` (uint3), strides `s01/s11/s1`, `ids_stride`.

---

## 5. MMQ path — `ggml/src/ggml-cuda/mmq.cuh` / `mmq.cu`

### Tile constants (`mmq.cuh`)
| symbol | line | value |
|---|---|---|
| `MMQ_DP4A_MAX_BATCH_SIZE` | 8 | `64` |
| `MMQ_ITER_K` | 9 | `256` (K-strip per iteration; `K_vram`) |
| `MMQ_NWARPS` | 11 | `8` |
| `MMQ_TILE_NE_K` | 116 | `32` |
| `MMQ_TILE_Y_K` | 119 | `MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1` = `36` |
| `struct block_q8_1_mmq` | 24 | `QK8_1_MMQ = 4*QK8_1 = 128`; layout union `d4[4]/ds4[4]/d2s6[8]` + `int8_t qs[128]`; static asserts 67–69. |
| `enum mmq_q8_1_ds_layout {D4, DS4, D2S6}` | 39 | **Q4_K → `MMQ_Q8_1_DS_LAYOUT_DS4`** (`mmq_get_q8_1_ds_layout`, 72; Q4_K/Q5_K at 87). |
| `enum ggml_cuda_mmq_sram_layout` | 104 | Q8_1 layout stride = `2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4` (via `ggml_cuda_mmq_get_sram_stride`, 123). |
| `struct ggml_cuda_mmq_config{type,nthreads,occupancy,I,J,sram_layout,K_vram,stream_k,fallback}` | 165 | per-(type,J,fallback) tile config. |
| `MMQ_DP4A_TXS_Q4_K` | 394 | `tile_x_sizes{I*MMQ_TILE_NE_K + I, I*MMQ_TILE_NE_K/QI4_K, I*MMQ_TILE_NE_K/8 + I/8}`; selected at 411. |

### Config selection
- `mmq.cuh:230` host `ggml_cuda_mmq_get_config(type,J,fallback,cc)` → `ggml_cuda_mmq_get_config_blackwell` (250) for 12xx; that file (`mmq-config-blackwell.cuh`) has **only MXFP4/NVFP4** entries and **falls back to ampere** for Q4_K.
- `mmq-config-ampere.cuh:157–172` = the Q4_K configs GB10 actually uses: `nthreads=256, occupancy=1, I=128, J∈{8,16,...,128}, sram_layout=Q8_1, K_vram=MMQ_ITER_K(256), stream_k=true, fallback=false|true`.
- `mmq.cuh:189` `use_mma_data_layout(cc)` / `:196` constexpr device version — Tony+/Blackwell → **true**, so GB10 uses the MMA data layout.
- `mmq.cuh:261` device `ggml_cuda_mmq_get_config(...)` (constexpr mirror used at compile time inside the kernel).
- `ggml_cuda_mmq_get_util_funcs<type,J,fallback>()` (546): non-MMA layout Q4_K (≈603) → `{VDR_Q4_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q4_K, ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a, ggml_cuda_mmq_write_back_dp4a}`; MMA layout Q4_K (≈767) → `ggml_cuda_mmq_load_tiles_q4_K + vec_dot_q8_1_q8_1_mma + write_back_mma`.

### Load / dot helpers
| file | symbol | line | why |
|---|---|---|---|
| `mmq-load-tiles.cuh` | `unpack_scales_q45_K` | 701 | scale/min decode inside tile load. |
| `mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_q4_K<type,J,fallback>` | 711 | MMA branch + dp4a branch. |
| `mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a<type,J,fallback>` | 905 | dp4a tile dot (fallback configs). |
| `mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type,J,fallback>` | 313 | **MMA tile dot used on GB10**. |

### Kernel / launch / case
| file | symbol | line | why |
|---|---|---|---|
| `mmq.cuh` | `static __global__ void mul_mat_q<type,J,fallback>(...)` | 955 | the MMQ kernel. |
| `mmq.cuh` | `mul_mat_q_stream_k_fixup<type,J,fallback>` | 1242 | stream-K reduction (needs `args.dst`, tmp from pool). |
| `mmq.cuh` | `struct mmq_args{...}` | 1378 | **see §9 fields**. |
| `mmq.cuh` | `static void launch_mul_mat_q<type,J,fallback>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream)` | 1396 | reads `ggml_cuda_info().devices[id].{cc,nsm,warp_size}` (↔ `cudaGetDeviceProperties`), computes `block_dims(warp_size,nwarps,1)` and `block_nums_xy_tiling(nty,ntx,ntzw)`, `CUDA_SET_SHARED_MEMORY_LIMIT`. |
| `mmq.cuh` | `void mul_mat_q_switch_J<type,fallback>(...)` | 1478 | J = 8..128 step 8 → `launch_mul_mat_q<type, J, fallback>` (launch sites 1506–1551). |
| `mmq.cuh` | `void mul_mat_q_case<type>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t)` | 1561 | `fallback = (args.nrows_x % 128 != 0)`; picks `..._switch_J<type,false>` or `<type,true>`. |
| `mmq.cuh` | `#define DECL_MMQ_CASE(type)` | 1571 | template inst macro. |
| `mmq.cuh` | `extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);` | 1584 | Q4_K instantiation decl. |
| `mmq.cu` | `case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);` | 38–39 | dispatch into case. |
| `mmq.cu` | `void ggml_cuda_mul_mat_q(...)` | 85 | **host entry**: `fallback = ne01 % 128 != 0` (129); workspace sizing 136–138; `quantize_mmq_q8_1_cuda(...)` at 156. |
| `mmq.cu` | `bool ggml_cuda_should_use_mmq(type, cc, ne11, n_experts)` | **266** | predicate — see below. |
| `template-instances/mmq-instance-q4_k.cu` | `#include "../mmq.cuh"` + `DECL_MMQ_CASE(GGML_TYPE_Q4_K);` | — | explicit instantiation TU. |

### `ggml_cuda_should_use_mmq` predicate (mmq.cu:266)
- `:278–300` type switch — Q4_K listed as MMQ-supported (`case GGML_TYPE_Q4_K:` at 284).
- `:~310` requires `smpbo >= 48*1024` (per-block shared mem) else → `false` (BLAS).
- `:321` `if (turing_mma_available(cc)) return true;` — **always true on GB10**, so any M that reaches here and fits shared mem uses MMQ.
- `:329` `#ifdef GGML_CUDA_FORCE_MMQ return true;`
- `:334` NVIDIA non-MMA fallback: `return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;` (`fp16_mma_hardware_available` at `common.cuh:326`).

### Which M uses which path (shipped dispatcher)
- `ggml_cuda_should_use_mmvq` (`mmvq.cu:318`):
  - cc == `GGML_CUDA_CC_BLACKWELL` (1200): `case GGML_TYPE_Q4_K: return ne11 <= 5;` (338).
  - cc == `GGML_CUDA_CC_DGX_SPARK` (1210, **GB10**): only Q2_K special (`<=6`); Q4_K falls to `default: return ne11 <= MMVQ_MAX_BATCH_SIZE;` (**8**) (349–355).
  - => **GB10 Q4_K: MMVQ for M <= 8, MMQ for M > 8** (MMQ reached via `mmq.cu:321`). (For a 1200-class Blackwell: MMVQ M<=5.)

---

## 6. Activation quantization — `ggml/src/ggml-cuda/quantize.cu` / `quantize.cuh`

| file | symbol | line | why needed |
|---|---|---|---|
| `quantize.cuh` | `#define CUDA_QUANTIZE_BLOCK_SIZE` / `..._MMQ` | 8–9 | `256` / `128`. |
| `quantize.cuh` | `static_assert(MATRIX_ROW_PADDING % CUDA_QUANTIZE_BLOCK_SIZE == 0, ...)` etc. | 11–12 | padding invariants (`MATRIX_ROW_PADDING = 512`, `common.cuh:186`). |
| `quantize.cu` | `static __global__ void quantize_q8_1(const float *x, void *vy, ne00, s01,s02,s03, ne0, ne1, uint3 ne2)` | 54 (launch_bounds 53) | **MMVQ activation quant**: each warp → one `block_q8_1`; `warp_reduce_max/sum<QK8_1>`; writes `y[ib].qs` and `y[ib].ds=make_half2(d,sum)`. |
| `quantize.cu` | `void quantize_row_q8_1_cuda(...)` | 558 | launcher: asserts `ne0 % QK8_1 == 0`; `block_num_x=(ne0+255)/256; dim3 num_blocks(block_num_x, ne1, ne2*ne3); dim3 block_size(256,1,1);` (567–569). **No padding past ne0.** |
| `quantize.cu` | `static __global__ void quantize_mmq_q8_1<ds_layout,scatter>(...)` | 458 | **MMQ activation quant**: 128-value blocks (`QK8_1_MMQ`), 128 threads, reads `float4`, requires `ne00%4==0` and `ne0%128==0`; DS4 layout for Q4_K. |
| `quantize.cu` | `void quantize_mmq_q8_1_cuda(...)` | 575 | launcher: `block_num_y=(ne0+4*128-1)/(4*128); dim3 num_blocks(ne1, block_num_y, ne2*ne3); dim3 block_size(128,1,1);` switch → Q4_K `DS4` (583–585). |

### Workspace sizing
- MMVQ (mmvq.cu ≈1504): `ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);` where `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)`.
- MMQ (mmq.cu ≈136): `nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * (y_block_size/y_values_per_block) + ggml_cuda_mmq_get_J_max(type,fallback,cc,ne11) * sizeof(block_q8_1_mmq);` where `y_block_size=sizeof(block_q8_1_mmq)` and `y_values_per_block=QK8_1_MMQ`. (There is **no** exported `ggml_cuda_quantize_row_q8_1_size` helper; replicate the expression inline.)

**Answer:** Q8_1 quantization is **MANDATORY** in both regimes — every Q4_K vecdot consumes `block_q8_1`/`block_q8_1_mmq`. No pure-fp32-activation Q4_K path exists. Padding note: MMVQ quant allows only `ne0%QK8_1==0`; MMQ quant needs `ne00%4==0` and `ne0%128==0`; both rely on `GGML_PAD(ne10, 512)`.

---

## 7. Transitive `#include` closure (project headers only)

```
mmvq.cu
 ├─ mmvq.cuh                 (MMVQ_MAX_BATCH_SIZE)
 ├─ quantize.cuh ── common.cuh, mmq.cuh
 ├─ unary.cuh                (fused-activation; only needed for has_fusion=true)
 ├─ vecdotq.cuh ── common.cuh
 └─ <cstdint> <type_traits>

mmq.cu
 ├─ common.cuh
 ├─ mmq.cuh ── common.cuh, mmq-config-*.cuh, mmq-load-tiles.cuh, mmq-vec-dot.cuh
 │      ├─ mmq-load-tiles.cuh ── vecdotq.cuh, mmq.cuh
 │      └─ mmq-vec-dot.cuh   ── vecdotq.cuh, mma.cuh, mmq.cuh
 ├─ quantize.cuh
 ├─ mmid.cuh                 (#pragma once + 1 decl, no includes)
 └─ <cstdint>

vecdotq.cuh ── common.cuh
dequantize.cuh ── common.cuh, convert.cuh ── common.cuh
mma.cuh ── common.cuh
# NOTE: cp-async.cuh is NOT in the Q4_K dependency closure (only fattn-mma-f16.cuh uses it).

common.cuh  (1717 lines) ── ggml.h, ggml-impl.h, ggml-cuda.h, ggml-common.h,
                            vendors/cuda.h (or hip/musa), + STL
```

### Heavyweight (graph/scheduler/allocator/tensor/pool) — MUST be replaced by a thin shim
- `ggml/include/ggml.h`, `ggml/src/ggml-impl.h`, `ggml/include/ggml-cuda.h`, `ggml/include/ggml-backend.h`.
- `common.cuh` itself: defines `ggml_cuda_pool`/`ggml_cuda_pool_alloc` (1207/1215), `ggml_backend_cuda_context` (1455), `ggml_cuda_info()` device table, `ggml_cuda_kernel_launch_params`/`ggml_cuda_kernel_launch` (1594/1699), `ggml_cuda_pdl_*`, `CUDA_SET_SHARED_MEMORY_LIMIT`. Keep only: `WARP_SIZE 32`, `GGML_CUDA_CC_*`, `ggml_cuda_dp4a` (720), `ggml_cuda_type_traits<>` (982+), `MATRIX_ROW_PADDING` (186), MMA availability macros, `ggml_cuda_get_physical_warp_size()`.
- `ggml/src/ggml-cuda/ggml-cuda.cu` (5788 L) — the whole graph/scheduler/allocator. **Not copied**; replaced by the thin adapter.

### Lightweight / copyable
`ggml-common.h` (struct section), `dequantize.cuh`, `vecdotq.cuh`, `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mma.cuh`, `mmid.cuh`, `mmvq.cu`, `mmq.cu`, `quantize.cu`.

---

## 8. Architecture dispatch for GB10 (SM 12.1)

CC constants (`common.cuh`): `GGML_CUDA_CC_PASCAL 600` (50), `GGML_CUDA_CC_DP4A 610` (51), `GGML_CUDA_CC_TURING 750` (53), `GGML_CUDA_CC_BLACKWELL 1200` (60), **`GGML_CUDA_CC_DGX_SPARK 1210` (61)** (GB10), `GGML_CUDA_CC_RUBIN 1300`.

Guards that actually select the Q4_K path on GB10:
- `ggml_cuda_dp4a` (common.cuh:720): `#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)` → `__dp4a` (749/720). Always taken on 12.1. `GGML_USE_HIP` selects the `__builtin_amdgcn_sdot4` branch — **irrelevant on NVIDIA; do not define.**
- `TURING_MMA_AVAILABLE` (common.cuh `:288` `__CUDA_ARCH__ >= GGML_CUDA_CC_TURING`) is defined for 12.1 and is the **actual selector** of `use_mma_data_layout()` (`mmq.cuh:189`/`:196`) → MMA data layout + `vec_dot_q8_1_q8_1_mma`. `AMPERE_MMA_AVAILABLE` (`:292`, `>= 800`) and `BLACKWELL_MMA_AVAILABLE` (`:296`, `>= 1200 && < RUBIN`) are also defined for 12.1 and gate Blackwell-specific MMA ops inside `mma.cuh`/`mmq.cuh`.
- MMVQ table: `#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK (1210) return MMVQ_PARAMETERS_GB10;` (mmvq.cu ~118) — requires compiling the TU for arch `1210`.
- Host `ggml_cuda_should_use_mmq` `:321 turing_mma_available(cc)` true; but GB10 device cc is **1210**, which is numerically `> GGML_CUDA_CC_BLACKWELL (1200)`, so the `cc == GGML_CUDA_CC_BLACKWELL` MMVQ branch is **not** taken; the `DGX_SPARK` branch is.
- `GGML_CUDA_FORCE_MMQ` → forces `should_use_mmq` = true (mmq.cu:329). `GGML_CUDA_FORCE_CUBLAS` → forces `should_use_mmq` = false (mmq.cu:267). **`GGML_CUDA_F16` is NOT present anywhere in `ggml/src/ggml-cuda/`** (searched) — no effect on the Q4_K path; do not define.

**Macros a trimmed port must define for GB10:**
- Build for `compute_121a, sm_121a` (CMake `ggml/src/ggml-cuda/CMakeLists.txt:48–55` appends `120a-real`/`121a-real`; equivalently `-gencode arch=compute_121a,code=sm_121a`). This makes `__CUDA_ARCH__ == 1210` inside device code.
- Do **not** define HIP/MUSA macros. `GGML_CUDA_FORCE_MMQ`/`GGML_CUDA_FORCE_CUBLAS` optional (routing override only).
- Keep `GGML_CUDA_CC_*` numeric constants verbatim (host side) so `ggml_cuda_should_use_mmvq/mmq` reproduce the shipped thresholds.

---

## 9. Minimal launch/setup + thin adapter (no graph/scheduler/tensor/allocator)

### `mmq_args` fields to fill (mmq.cuh:1378)
`x` (const char* weights), `type_x = GGML_TYPE_Q4_K`, `y` (const int* → Q8_1_mmq workspace), `ids_dst = nullptr`, `expert_bounds = nullptr`, `dst` (fp32 out), `y_scale = nullptr`, `ncols_x = K`, `nrows_x = N (weight rows)`, `ncols_dst = M`, `stride_row_x = K` (elements of Q4_K type), `ncols_y = M`, `nrows_dst = N`, channel/sample fields = 1/1 with identity strides, `ncols_max = M`, `ncols_opt = M`.

### MMVQ adapter sketch
1. `K_padded = GGML_PAD(K, 512)`; alloc `M*N * sizeof(block_q8_1)/QK8_1` bytes Q8_1 workspace (+ tail pad).
2. Launch `quantize_q8_1` grid `(ceil(K/256), M, 1)`, block `256` (or call `quantize_row_q8_1_cuda(x, nullptr, ws, GGML_TYPE_Q4_K, K,K,K,K, K_padded, M,1,1, stream)`).
3. `nwarps = calc_nwarps(GGML_TYPE_Q4_K, M, MMVQ_PARAMETERS_GB10, small_k, halve_iters)`; `rpb = calc_rows_per_block(M, GB10, small_k, nwarps)`; `grid=dim3(ceil(N/rpb),1,1)`, `block=dim3(32,nwarps,1)`.
4. `mul_mat_vec_q<GGML_TYPE_Q4_K, M, /*has_fusion=*/false><<<grid,block,0,stream>>>(vx, ws, nullptr, fusion{}, dst, K, nchannels_y{}, strides..., ids_stride=0);`

### MMQ adapter sketch
1. `fallback = (N % 128 != 0)`.
2. Alloc `M*K_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + J_max*sizeof(block_q8_1_mmq)` bytes.
3. Launch `quantize_mmq_q8_1<DS4,false>` grid `(M, ceil(K_padded/(4*128)), 1)`, block `128`.
4. Fill `mmq_args`; call `mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream)`.
   - `ctx` is only used for `ctx.pool()` (stream-K fixup temp) and device info. **Thin shim options:** (a) supply a trivial `ggml_backend_cuda_context` exposing `pool()` returning a `cudaMallocAsync`-backed allocator + a pin of device props; or (b) copy `launch_mul_mat_q` and replace `ggml_cuda_info()` reads with a cached `cudaGetDeviceProperties` (cc/nsm/warp_size) and allocate the fixup buffer directly.
5. Device props needed: `cc, nsm, warp_size, smpbo` (smpbo must be >= 48 KiB or MMQ is rejected — true on GB10).

Both adapters need only: `stream`, `nrows_x (=N)`, `nrows_y (=M)`, `ncols_x (=K)`, row-major strided inputs, and the `GGML_TYPE_Q4_K` switch. No ggml graph, scheduler, tensor or backend allocator.

---

## Explicit answers

- **Q8_1 mandatory?** YES for both MMVQ and MMQ. No fp32-activation Q4_K variant exists (all `vec_dot_q4_K_*` take `block_q8_1`/`block_q8_1_mmq`).
- **`_dp4a` vs `_fp16`/`_half2` variants?** Q4_K has **only** an integer `dp4a` vecdot (`vec_dot_q4_K_q8_1_impl_vmmq`, `ggml_cuda_dp4a`); there is no fp16/half2 *arithmetic* variant. `half2` is used solely as the packed `dm`/`ds` scale container. On GB10, MMVQ uses dp4a; MMQ uses the **MMA data layout** (`vec_dot_q8_1_q8_1_mma`, ldmatrix + `mma()`) which is expected faster for M>8, while dp4a remains for fallback configs.
- **MMVQ vs MMQ by M (shipped dispatcher, literal):**
  - GB10 (`cc==1210`): MMVQ when `ne11 <= MMVQ_MAX_BATCH_SIZE` i.e. **M <= 8** — `mmvq.cu:349–355` (default arm of the `GGML_CUDA_CC_DGX_SPARK` switch; only Q2_K has `<=6`). MMQ above that via `mmq.cu:321 (turing_mma_available → true)`.
  - 1200-class Blackwell: Q4_K MMVQ when `ne11 <= 5` — `mmvq.cu:338`.
  - Route order: `ggml-cuda.cu:1868` (MMVQ) → `:1872` (MMQ) → `:1875` (cuBLAS).

---

## License / attribution (point 10)

`/home/lvx/sera.c/_reference/llama.cpp/LICENSE`: **MIT License**, `Copyright (c) 2023-2026 The ggml authors` (lines 1 and 3).

Attribution string for `THIRD_PARTY_NOTES.md`:
> llama.cpp / ggml — MIT License — Copyright (c) 2023-2026 The ggml authors — donor commit `709fe755dfa810d77e2ac386292b29648b536864`.

---

## Exact files to copy into the q3 tree

| donor (under `ggml/src/ggml-cuda/` unless noted) | suggested target | treatment |
|---|---|---|
| `ggml-common.h` (struct/const section) | `cuda/q3_common.h` | **Trim** to QK_K/K_SCALE_SIZE/QR4_K/QI4_K/QK8_1/QI8_1 + `block_q4_K`,`block_q8_1`; keep `GGML_COMMON_DECL_CUDA`/`IMPL_CUDA` or de-macro the union. |
| `dequantize.cuh` | `cuda/q3_dequantize.cuh` | **Trim** to `get_scale_min_k4` + `dequantize_q4_K`; inline a trivial `ggml_cuda_cast`. |
| `convert.cuh` | `cuda/q3_convert.cuh` | **Trim** to the cast helper only (or drop and inline). |
| `convert.cu` (`dequantize_block_q4_K`, `dequantize_row_q4_K_cuda`) | `cuda/q3_dequantize.cu` | **Trim** to the q4_K block kernel + launcher. |
| `vecdotq.cuh` | `cuda/q3_vecdotq.cuh` | **Trim** to Q4_K (VDR defs 504–505, impl 508/533, entry 918) + Q8_1 helpers they need. |
| `mmvq.cuh` | `cuda/q3_mmvq.cuh` | **Copy nearly verbatim** (one macro). |
| `mmvq.cu` | `cuda/q3_mmvq.cu` | **Trim** to Q4_K only; keep GB10 table + `calc_nwarps/calc_rows_per_block/calc_launch_params`; strip ggml tensor/op plumbing. |
| `mmq.cuh` | `cuda/q3_mmq.cuh` | **Trim** to Q4_K + Q8_1 MMA layout; keep tile consts, config struct, util-funcs, `mmq_args`, `launch_mul_mat_q`, `mul_mat_q_switch_J`, `mul_mat_q_case`, `DECL_MMQ_CASE`. |
| `mmq.cu` | `cuda/q3_mmq.cu` | **Trim + thin adapter** (host entry `ggml_cuda_mul_mat_q`, `should_use_mmq`). |
| `mmq-load-tiles.cuh` | `cuda/q3_mmq_load_tiles.cuh` | **Trim** to `unpack_scales_q45_K` + `ggml_cuda_mmq_load_tiles_q4_K`. |
| `mmq-vec-dot.cuh` | `cuda/q3_mmq_vec_dot.cuh` | **Trim** to `..._vec_dot_q8_1_q8_1_mma` + `..._vec_dot_q4_K_q8_1_dp4a`. |
| `mmq-config-blackwell.cuh` | `cuda/q3_mmq_config_blackwell.cuh` | **Trim** (only the ampere fallback matters for Q4_K). |
| `mmq-config-ampere.cuh` | `cuda/q3_mmq_config_ampere.cuh` | **Trim** to Q4_K entries (157–172). |
| `quantize.cu` | `cuda/q3_quantize.cu` | **Trim** to `quantize_q8_1`, `quantize_mmq_q8_1`, the two launchers (DS4 only). |
| `quantize.cuh` | `cuda/q3_quantize.cuh` | **Copy nearly verbatim** (constants + decls). |
| `mma.cuh` | `cuda/q3_mma.cuh` | **Copy nearly verbatim** (needs `common.cuh` shim). |
| `mmid.cuh` | `cuda/q3_mmid.cuh` | **Copy verbatim** (trivial). |
| `template-instances/mmq-instance-q4_k.cu` | `cuda/q3_mmq_instance_q4_k.cu` | **Copy verbatim**. |
| `common.cuh` | `cuda/q3_common.cuh` | **MUST REWRITE as thin shim** (drop pool/context/info/launch/pdl; keep WARP_SIZE, CC macros, dp4a, type_traits, MMA macros, MATRIX_ROW_PADDING). |
| `ggml-cuda.h`, `ggml-cuda.cu`, `ggml.h`, `ggml-impl.h`, `ggml-backend.h` | `src/q3_adapter.{h,cu}` | **NOT copied** — replaced by a thin C/C++ adapter exposing `q3_linear_q4_k(...)` (see §9). |

**No file under `_reference/llama.cpp` is modified.**
