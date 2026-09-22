# M1 Worker A — Q4_K donor inventory (`_reference/q38.c`, branch `qwen38-spark-proto`)

Legend: `keep` = copy as-is; `edit` = copy then rename/adapt; `delete` = do not carry into `sera.c`.

## 1. Q4_K block struct, sizes, QK_K

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/q38_quant.h` | `Q38_QUANT_Q4_K = 12` (L14) | none | edit (rename to sera type id) |
| `_reference/q38.c/q38_quant.h` | `#define Q38_QUANT_QK_K 256` (L19) | none | keep (rename macro) |
| `_reference/q38.c/q38_quant.h` | `#define Q38_QUANT_Q4_K_BLOCK_BYTES 144` (L21) | none | keep (rename) |
| `_reference/q38.c/q38_quant.h` | `typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } q38_q4_k_block;` (L30-35) | `<stdint.h>` | edit (rename struct only) |
| `_reference/q38.c/q38_quant.h` | `typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } q38_q2_k_block;` (L23-28) | `<stdint.h>` | keep (adjacent, Q2_K) |
| `_reference/q38.c/third_party/gguf-tools/quants.h` | `DS4Q_TYPE_Q4_K = 12` (L29) | none | keep (GGML id match) |
| `_reference/q38.c/third_party/gguf-tools/quants.h` | `DS4Q_TYPE_Q2_K = 10` (L27) | none | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `#define QK_K 256` (L34) | none | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `ds4q_traits` table entry `[DS4Q_TYPE_Q4_K] = { "q4_K", QK_K, 144, true, false }` (L50) | `ds4q_traits` (L25-32) | edit (rename if kept) |
| `_reference/q38.c/q38_gguf.c` | `gguf_types[12] = {"q4_k", 256, 144}` (L126) | `gguf_type_info` (L109-113) | keep |

**Exact Q4_K field layout / size**: offset 0 `uint16_t d` (fp16 scale, 2B); offset 2 `uint16_t dmin` (fp16 min, 2B); offset 4 `uint8_t scales[12]` (12B); offset 16 `uint8_t qs[128]` (128B); total **144 bytes** = `2 + 2 + 12 + 128`. `sizeof(q38_q4_k_block) == 144` (no padding: all members `uint16_t`/`uint8_t[]` arrays; struct alignment 2). `qs[128]` packs 256 4-bit quants (QK_K=256). There is **NO `static_assert`/`_Static_assert` anywhere in the donor** (verified: grep for `static_assert|_Static_assert` over all `.c/.h/.cu` returned nothing). `QK_K = 256`.

Nearest relatives present in the donor: `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K/Q3_K/Q5_K/Q6_K/Q8_K` are all **declared** in `ds4q_type_traits` (`third_party/gguf-tools/quants.c` L42-74) and `q38_gguf.c` gguf_types (L118-129), but **only Q8_0, Q8_K, Q2_K, Q4_K, IQ2_XXS have implementations** (`ds4q_quantize_chunk` dispatch, quants.c L1109-1137). `Q4_0/Q4_1/Q5_K/Q6_K/Q8_1` have trait entries (block sizes 18/20/176/210/36) but **no quant/dequant kernels and no block structs**. `q38_gguf.c` L123 declares q8_1 with block_bytes **40** while quants.c L47 uses **36** (inconsistent; do not trust the gguf.c value).

## 2. fp16 <-> fp32 helpers

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/q38_quant.c` | `float q38_half_to_float(uint16_t bits)` (L11-35) — bit-twiddling fp16->fp32, no table | `<math.h>`,`<string.h>` (memcpy) | keep (host dequant oracle) |
| `_reference/q38.c/q38_quant.h` | `float q38_half_to_float(uint16_t);` (L37) | stdint | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `float ds4q_f16_to_f32(uint16_t)` (L1139-1153) — fp16->fp32 via fp32 bit manipulation | `ds4q_f32_from_bits` (L76), `ds4q_f32_to_bits` (L84) | keep (alternative oracle) |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static uint16_t ds4q_f32_to_f16(float)` (L92-109) | `ds4q_f32_to_bits/from_bits` | keep (quantizer side) |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `void ds4q_f32_to_f16_row(const float*, uint16_t*, int64_t)` (L1159-1161) | `ds4q_f32_to_f16` | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `float ds4q_bf16_to_f32(uint16_t)` (L1155), `ds4q_f32_to_bf16_row` (L1163) | — | keep (adjacent) |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `__device__ static float half_to_float_device(uint16_t bits)` (L5-27) — portable CUDA fp16->fp32 bit twiddle, ends `__uint_as_float` | none | keep (device decode) |
| `_reference/q38.c/cuda/q38_ple_cuda.cu` | `__device__ static float half_to_float_device(uint16_t bits)` (L9-31) — **duplicate** of above | none | delete (dedupe) |
| `_reference/q38.c/cuda/q38_gdn.cu` | `__device__ static float gdn_half_to_float(uint16_t bits)` (L8-30) — **duplicate** | none | delete (dedupe) |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | uses CUDA intrinsic `__half2float` (L537-538, L570-573, L713-714) via `<cuda_fp16.h>` (L11) | `cuda_fp16.h` | keep (device fp16 path) |

**No `ggml_fp16_to_fp32` table-lookup implementation exists in the donor.** No `__half2`, no `half2`, no `__float2half`, no `__hadd/__hmul` anywhere (`grep half2|__float2half|__hadd|__hmul` returns nothing). `__half2float` is only used in `cuda/q38_moe_cuda.cu` (and `#include <cuda_fp16.h>` in `cuda/q38_forward_cuda.cu` L14 but not used for quant). The donor's portable fp16->fp32 (bit-twiddle, no hardware intrinsic, no lookup table) is repeated 4x.

## 3. Q4_K scale/min unpacking

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/q38_quant.c` | `static void scale_min_q4(unsigned index, const uint8_t *scales, uint8_t *scale, uint8_t *min)` (L62-73) | none | keep (host oracle) |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)` (L333-341) | none | keep (quantizer, same math) |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `__device__ static void q4_scale_min(unsigned index, const uint8_t *scales, uint8_t *scale, uint8_t *minimum)` (L57-68) | none | keep |
| `_reference/q38.c/cuda/q38_ple_cuda.cu` | `__device__ static void q4_scale_min(unsigned index, const uint8_t*)` (L48-59) — **duplicate** | none | delete (dedupe) |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `__device__ static void q4_scale_min(const q38_q4_k_block *block, unsigned index, uint8_t *scale, uint8_t *minimum)` (L685-697) — takes block ptr instead of `scales` | none | edit (dedupe to primitives version) |

**Exact unpacking math** (`scale_min_q4`, `j = 0..7`): if `j < 4`: `scale = scales[j] & 63`, `min = scales[j+4] & 63`. Else: `scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)`, `min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)`. 12-byte `scales[]` holds 8 6-bit scales + 8 6-bit mins.

## 4. CPU Q4_K dequantization

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/q38_quant.c` | `static void dequant_q4(const q38_q4_k_block *block, float *out)` (L75-94) | `q38_half_to_float`, `scale_min_q4`, `Q38_QUANT_QK_K` | keep |
| `_reference/q38.c/q38_quant.c` | `bool q38_quant_dequantize_row(uint32_t type, const void *blocks, size_t block_count, float *out, size_t out_elements, char *error, size_t error_len)` (L96-119); Q4_K branch L112-116 | dispatches `dequant_q4` per block | keep (rename) |
| `_reference/q38.c/q38_quant.c` | `static void dequant_q2(const q38_q2_k_block*, float*)` (L37-60) | Q2_K adjacent | keep |

**Behavior**: dequantizes **out-of-place** into caller-provided `float* out` (`out` must hold `block_count * QK_K` floats; validated `out_elements == block_count*QK_K`, L103). Never in-place. Loop template: for `j = 0; j < QK_K; j += 64`, unpack scale_index `2*group` (low nibbles: `out[l] = d1*(q[l]&0xF) - m1`) and `2*group+1` (high nibbles: `out[l] = d2*(q[l]>>4) - m2`), advancing `q += 32` per 64 elements. Formula `value = d*scale*quant - dmin*min`. `static void set_error` (L7-9) is the error helper.

## 5. CUDA Q4_K block decode / dequantization (device)

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `__device__ static float q4_value(const q38_q4_k_block *block, unsigned element)` (L70-79) — elementwise decode | `half_to_float_device` (L5), `q4_scale_min` (L57) | keep |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `__global__ static void dequant_kernel(uint32_t type, const void *blocks, size_t block_count, float *out)` (L81-92) — 1 thread/element, `type==Q2_K` else Q4_K | `q2_value`/`q4_value` | keep |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `extern "C" bool q38_cuda_dequantize_row(uint32_t type, const void *blocks, size_t block_count, float *out, cudaStream_t stream, char *error, size_t error_len)` (L215-236); launch `dequant_kernel<<<(elements+255)/256, 256, 0, stream>>>` (L227-228) | `q38_cuda_primitives.h` | keep (rename) |
| `_reference/q38.c/cuda/q38_ple_cuda.cu` | `__device__ static float q4_value(const q38_q4_k_block *block, unsigned element)` (L61-71) — **duplicate** | local `half_to_float_device` (L9), `q4_scale_min` (L48) | delete (dedupe) |
| `_reference/q38.c/cuda/q38_ple_cuda.cu` | `__global__ static void lookup_rows_kernel(uint32_t qtype, const void *table, uint64_t table_rows, uint32_t row_width, const uint32_t *ids, size_t id_count, float *out)` (L73-103) — gathers rows, decodes Q4_K elementwise | q2_value/q4_value | keep (optional row-gather path) |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `__device__ static float q4_value(const q38_q4_k_block *blocks, size_t row, size_t column, size_t blocks_per_row)` (L699-717) — row/column decode, uses `__half2float` | `q4_scale_min` (L685), `<cuda_fp16.h>` | keep (best generic decoder) |

**Element index math for `q4_value` (row/col form, moe_cuda L699-717)**: `element = column % 256`; `block = blocks + row*blocks_per_row + column/256`; `group = element/64`; `within = element%64`; `scale_index = group*2 + (within>=32)`; `qindex = within%32`; `packed = block->qs[group*32 + qindex]`; `quant = (within>=32) ? packed>>4 : packed&0xF`; returns `d*scale*quant - m*minimum`.

## 6. Existing CUDA matvec/matmul kernels consuming Q4_K

| file | symbol (line) | geometry | vectorized? |
|---|---|---|---|
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `__global__ static void q4_gate_up_kernel(const q38_q4_k_block *weights, const float *hidden, float *mid)` (L719-730) | `<<<3, 256>>>`; 1 thread = 1 intermediate row `i`; scalar loop `for d in 0..Q38_MOE_HIDDEN` calling `q4_value(weights,i,d,10)`; gate & up accumulate in-register; SiLU(gate)*up | **NAIVE scalar** (fused decoder+dot; one thread per output row) |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `__global__ static void q4_down_kernel(const q38_q4_k_block *weights, const float *mid, float *output)` (L732-740) | `<<<10, 256>>>`; 1 thread = 1 hidden output `d`; scalar loop over 640 intermediates | **NAIVE scalar** |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `extern "C" bool q38_moe_cuda_expert_q4_workspace(...)` (L742-757) | launches gate_up `<<<3,256>>>` (L749) then down `<<<10,256>>>` (L751) | naive |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `extern "C" bool q38_moe_cuda_q4_gate_up(...)` (L759-770), `q38_moe_cuda_q4_down(...)` (L772-783), `q38_moe_cuda_expert_q4(...)` (L785-800, mallocs mid, syncs stream) | as above | naive |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `q2_matvec_kernel` (L126-160) + `q38_cuda_q2_matvec` (L276-295) | `<<<rows, 256>>>`, 8 warps/block, each warp owns whole Q2_K blocks, `__shfl_down_sync` reduction | **Q2_K only**; scalar decode, no dp4a/half2 |
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `matrix_batch_generic_kernel` (L384-398) + `q38_cuda_matrix_batch_generic` (L400-428) | `<<<(total+255)/256, 256>>>`, 1 thread per (token,row), scalar loop over all cols calling `matrix_batch_weight_value` (L360-382) | **NAIVE**; dispatched types 0,8,10,30 — **type 12 (Q4_K) rejected at L406** |
| `_reference/q38.c/q38_gdn.h` | `q38_cuda_gdn_project(uint32_t weight_type,...)` (L43-46, impl `cuda/q38_gdn.cu` L531) | `<<<rows*tokens,256>>>`; handles F32/Q8_0 and *delegates* Q2_K to `q38_cuda_q2_matvec` per token (L503-514), BF16 to bf16 matvec; **no Q4_K branch** | naive |
| `_reference/q38.c/cuda/q38_forward_cuda.cu` | `q38_forward_cuda_matvec_backend` (L2199-2330) | type gate L2229 only accepts 0,8,10,30; **Q4_K (12) rejected** ("unsupported CUDA forward matvec type"); no Q4_K call anywhere | naive |
| `_reference/q38.c/cuda/q38_forward_cuda.cu` | `q38_forward_cuda_matrix_batch_backend` (L2483-...) | calls `q38_cuda_matrix_batch_generic` only (L2570); type gate excludes Q4_K | naive |

**No CUDA Q4_K matvec exists in `q38_cuda_primitives`** (there is `q38_cuda_q2_matvec` but no `q38_cuda_q4_matvec`). The only Q4_K compute kernels are the two naive MoE kernels in `cuda/q38_moe_cuda.cu`, and they are non-vectorized: **no `__dp4a`, no `half2`, no `__half2`, no integer SIMD, no `vecdotq`** (verified by grep: `dp4a` and `vec_dot|vecdotq|ggml_vec` return zero matches in the whole donor). Decode is per-element float with full fp16->fp32 conversion per element.

## 7. Q8_1 activation quantization

**None.** `DS4Q_TYPE_Q8_1 = 9` is declared (`third_party/gguf-tools/quants.h` L26) and has a traits row `{ "q8_1", 32, 36, false, false }` (`quants.c` L47), and `q38_gguf.c` L123 lists `{"q8_1",32,40}`. There is **no Q8_1 block struct, no `ds4q_quantize_q8_1`, no CUDA Q8_1 kernel, no activation-quantization path**. `ds4q_quantize_chunk` dispatch (quants.c L1109-1137) rejects Q8_1 via `assert("unsupported DS4 quantization target")` (L1135). Q8_K (id 15) has a writer (`ds4q_write_q8_k_block` L371, `ds4q_quantize_q8_k` L410) but is also not wired into CUDA. Q8_0 (id 8) is the only Q8 family member with both a quantizer (L343-369) and a CUDA device decode (`cuda/q38_gdn.cu` `gdn_q8_value` L36-39, `cuda/q38_cuda_primitives.cu` L336-380).

## 8. CUDA utility dependencies

| file | symbol (line) | purpose | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/cuda/q38_cuda_primitives.cu` | `static void set_error(char*, size_t, const char*)` (L211-213) | error string helper | edit (dedupe/rename) |
| `_reference/q38.c/cuda/q38_moe_cuda.cu` | `static bool fail(char*, size_t, const char*)` (L14-17) | same role, returns false | edit (canonicalize with set_error) |
| `_reference/q38.c/cuda/q38_ple_cuda.cu` | `static bool fail(...)` (L105-108) + `static void cuda_fatal(cudaError_t, const char*)` (L110-116, `_exit(134)`) | error + abort | edit |
| launch error-check idiom | `cudaError_t status = cudaGetLastError(); if (status != cudaSuccess) ...` e.g. `q38_cuda_primitives.cu` L229-234, L249-254, L288-293, L327-333, L419-426 | no macro; repeated | edit (consider a `Q38_CUDA_CHECK_LAUNCH` macro) |
| `_reference/q38.c/q38_cuda_timing.h` | `q38_cuda_timing` struct (L13-21), `q38_cuda_timing_init/destroy/begin/end/record_launch/record_allocation` (L23-29) | event timing + counters | keep |
| `_reference/q38.c/cuda/q38_cuda_timing.cu` | implementations (L5-54) | cudaEvent timing | keep |
| `_reference/q38.c/cuda/q38_forward_cuda.cu` | `#define Q38_CUDA_DIAG_ONLY(statement)` (L64/L66), `Q38_CUDA_DIAG_COLLECT(context)` (L69), `#define Q38_CUDA_SYNC_CALL(context, reason, call)` (L735/L745), sync-reason enum (`q38_forward_cuda.h` L22-37) | diagnostics/timing instrumentation | keep (optional) |
| `_reference/q38.c/q38_cuda.h` | `q38_cuda_probe` (L24), `q38_cuda_shared_memory_info` (L26-33), `q38_cuda_get_shared_memory_info` (L35), `q38_cuda_init/cleanup` (L38-39) | device lifecycle / smem sizing | keep |
| `_reference/q38.c/cuda/q38_cuda.cu` | implementations (L16-106) | lifecycle | keep |

No launch-config/smem helper macros exist; shared memory is declared inline per kernel

## 9. Tests covering Q4_K in the donor

| file | symbol (line) | what | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/tests/test_m2_quant.c` | `check_q4()` (L39-64) + `main` (L66-69) | CPU Q4_K `q38_quant_dequantize_row` boundary/oracle test (d=1.0, scales=1, qs=0x21) | keep (port to sera test) |
| `_reference/q38.c/tests/test_m2_cuda.cu` | `check_type(uint32_t,size_t)` (L24-70), `main` (L72-84) | CUDA vs CPU `q38_cuda_dequantize_row` parity for Q2_K/Q4_K | keep |
| `_reference/q38.c/tests/test_m8_q4_moe_cuda.cu` | `fill_block` (L11), `decode_matrix` (L20), `decode_down` (L36), `main` (L54-118) | Q4_K naive MoE CUDA vs `q38_moe_expert_ref` parity, rel err <= 2e-5 | keep |
| `_reference/q38.c/tests/test_quant_blocks.c` | `check_type(ds4q_type,int)` (L7-37), `main` (L39-48) | ds4q writer no-op safety for Q4_K/Q2_K/Q8_0/IQ2_XXS | keep |
| `_reference/q38.c/tests/test_m2_matvec.cu` | Q2_K matvec test (L30-...) | Q2_K only, **not Q4_K** (template for a future Q4_K matvec test) | edit (extend to Q4_K) |
| `_reference/q38.c/tests/test_m4_ple_cuda.cu` | `init_blocks<q38_q4_k_block>` (L31-41), `check_type<uint32_t>` (L43-120+) | Q4_K row decode via `q38_ple_cuda_lookup_rows` vs `q38_ple_decode_row_ref` | keep |
| `_reference/q38.c/tests/test_m4_ple_cuda_batch.cu` | `run_type` (calls L105-106) | dedup CUDA row lookup for Q2_K/Q4_K | keep |
| `_reference/q38.c/tests/test_m4_ple_row.c` | Q4_K scalar row decode (L38-85) | `q38_ple_decode_row_ref` vs `q38_quant_dequantize_row` | keep |
| `_reference/q38.c/tests/test_m6_gguf_dequant.c`/`.py` | — | gguf dequant coverage (check if Q4_K included) | edit |

No test exercises a Q4_K **matvec**; only decode + naive MoE.

## 10. CPU-side quantization reference implementations (oracle)

| file | symbol (line) | dependencies | keep/edit/delete |
|---|---|---|---|
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static void ds4q_write_q4_k_block_ref(const float *x, uint8_t *y)` (L428-491) | `ds4q_make_qkx2_quants` (L119), `ds4q_nearest_int` (L111), `ds4q_f32_to_f16` (L92), `ds4q_get_scale_min_k4` (L333), `ds4q_f16_to_f32` (L1139) | keep (unweighted oracle) |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static void ds4q_write_q4_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights)` (L493-563) | `ds4q_make_qp_quants` (L267), `ds4q_make_qkx2_quants`, `ds4q_get_scale_min_k4` | keep (imatrix oracle) |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static size_t ds4q_quantize_q4_k(const float *src, void *dst, int64_t start, int64_t nrows, int64_t ncols, const float *imatrix)` (L565-585) | the two writers above; dispatches on imatrix | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `ds4q_quantize_chunk` (L1109-1137), `ds4q_quantize_init` (L1103), `ds4q_row_size` (L1091), `ds4q_block_size` (L1086), `ds4q_can_quantize` (L1081), `ds4q_requires_imatrix` (L1098) | traits table (L39-74) | keep |
| `_reference/q38.c/third_party/gguf-tools/quants.c` | `static float ds4q_make_qkx2_quants(...)` (L119-190), `ds4q_make_qkx3_quants` (L192-265), `ds4q_make_qp_quants` (L267-330), `ds4q_nearest_int` (L111-117) | math | keep |
| `_reference/q38.c/tools/q38_quantize.c` | CLI streaming Safetensors->GGUF block quantizer, `parse_type` maps "Q4_K" (L43) | `quants.h` API | keep (oracle harness) |
| `_reference/q38.c/tools/q38_quant_block_test.py`, `q38_quant_audit.py`, `q38_quant_sensitivity.py` | python oracle/audit | — | keep (reference) |
| `_reference/q38.c/docs/qwen_quant_matched_reference.md` | semantics doc | — | keep (read for conventions) |

Also: `q38_oracle.c/.h` — `q38_oracle_compare`/`q38_oracle_metrics` numeric comparison used by tests (referenced in `tests/test_m2_quant.c` L34-36, `tests/test_m2_cuda.cu` L61-64).

---

## What can be copied directly

- `q38_quant.h` L19-21, L30-35 — QK_K, Q4_K block bytes, `q38_q4_k_block` struct; byte-exact GGUF Q4_K layout, only rename needed.
- `q38_quant.h` L13-14 — Q2_K/Q4_K type ids (match GGML ids 10/12).
- `q38_quant.c` L11-35 `q38_half_to_float` — portable host fp16->fp32, no table, used by every host decode path.
- `q38_quant.c` L62-73 `scale_min_q4` — exact 6-bit scale/min unpacking (matches `ds4q_get_scale_min_k4`).
- `q38_quant.c` L75-94 `dequant_q4` and L96-119 `q38_quant_dequantize_row` — complete CPU Q4_K row dequant oracle (out-of-place, validated).
- `third_party/gguf-tools/quants.c` L1139-1172 `ds4q_f16_to_f32`/`ds4q_f32_to_f16_row` — independent fp16 conversion oracle for cross-checking.
- `third_party/gguf-tools/quants.c` L428-585 — CPU Q4_K quantizer (`ds4q_write_q4_k_block_ref` + `_weighted` + `ds4q_quantize_q4_k`); the only way to generate ground-truth Q4_K blocks.
- `third_party/gguf-tools/quants.c` L39-74 traits table + L1081-1137 accessors — block sizes for Q4_K and relatives (Q4_0/Q4_1 = 18/20, Q5_K/Q6_K = 176/210, Q8_1 = 36).
- `cuda/q38_cuda_primitives.cu` L5-27 `half_to_float_device` — device fp16->fp32 bit twiddle (no intrinsic/table).
- `cuda/q38_cuda_primitives.cu` L57-79 `q4_scale_min` + `q4_value(block,element)` — device elementwise Q4_K decode; direct basis for a device dot.
- `cuda/q38_cuda_primitives.cu` L81-92 + L215-236 — `dequant_kernel` and `q38_cuda_dequantize_row` host wrapper (drop the Q2 branch or keep both).
- `cuda/q38_moe_cuda.cu` L699-717 `q4_value(blocks,row,column,blocks_per_row)` — generic row/column decoder, the cleanest signature to port.
- `cuda/q38_moe_cuda.cu` L685-697 `q4_scale_min(block,...)` — usable if a block-pointer signature is preferred.
- `cuda/q38_moe_cuda.cu` L719-740 `q4_gate_up_kernel` / `q4_down_kernel` — working (naive) Q4_K kernels with a correctness test, useful as the initial numerical baseline.
- `cuda/q38_gdn.cu` L41-79 `gdn_dense_project_kernel` + `q38_cuda_gdn_project` (L531) and L543-548 `q38_cuda_bf16_matvec_device` — ready-made warp-reduction projection skeleton to clone for Q4_K.
- `cuda/q38_cuda_primitives.cu` L126-160 `q2_matvec_kernel` + L276-295 `q38_cuda_q2_matvec` — direct structural template for `q38_cuda_q4_matvec` (8-warp, warp-owned-block, shfl reduce).
- `q38_cuda_timing.h` + `cuda/q38_cuda_timing.cu` — event timing/counters (L5-54).
- `q38_cuda.h` + `cuda/q38_cuda.cu` L16-106 — device probe/smem info lifecycle.
- Tests: `tests/test_m2_quant.c` L39-64 (CPU Q4_K boundary), `tests/test_m2_cuda.cu` L24-84 (CPU/CUDA decode parity), `tests/test_m8_q4_moe_cuda.cu` (end-to-end Q4_K kernel parity), `tests/test_quant_blocks.c`.
- `q38_oracle.c/.h` `q38_oracle_compare`/`q38_oracle_metrics` — numeric comparison used by all quant tests.
- `q38_gguf.c` L115-129 gguf_types table and `q38_gguf_type_nbytes` — type id/size metadata (fix the q8_1 = 40 vs 36 inconsistency).
- `q38_moe_ref.h` L12-15 constants and `q38_moe_cuda.h` L45-61 Q4_K API declarations — shape/contract reference for the naive path.

## What q38 does NOT provide

- **No truly vectorized production Q4_K batched GEMM.** The donor has **no `q38_cuda_q4_matvec`, no `q38_cuda_q4_matrix_batch`, no Q4_K GEMM at all**. The only Q4_K compute kernels are `q4_gate_up_kernel`/`q4_down_kernel` in `cuda/q38_moe_cuda.cu` (L719-740): naive scalar loops, one thread per output element, hardcoded to MoE dimensions (10 blocks/row = 2560 cols, 640 intermediates), full fp16->fp32 decode per MAC. They are decoder-level kernels, not production batched GEMM.
- **No `__dp4a` / integer-SIMD / `half2` quantized dot product anywhere** (grep for `dp4a` and `vec_dot|vecdotq|ggml_vec` returns zero matches).
- **No Q8_1 activation quantization**: only the type enum + a traits row; no struct, no kernel, no quantization function, no activation-quantize path. Q8_K writer exists but is not wired to CUDA.
- **No Q4_0 / Q4_1 / Q5_K / Q6_K code** other than trait-table entries (block sizes only); no struct, no dequant, no kernel.
- **No `static_assert`/`_Static_assert`** validating `sizeof(q38_q4_k_block) == 144` or `sizeof(q38_q2_k_block) == 84`.
- **No fp16 lookup table / hardware-intrinsic conversion**: `half_to_float_device` is a portable bit-twiddle duplicated in 3 files; `__half2`/`half2`/`__float2half`/`__hadd`/`__hmul` are entirely absent.
- **No shared-memory staging, no tiling, no `cp.async`, no launch-config/smem macro helpers**; the bf16 matvec `__shared__`/warp-shuffle idiom is the most sophisticated reduction available.
- **No generic (non-MoE, arbitrary rows/cols) Q4_K matvec**: `q38_cuda_matrix_batch_generic` explicitly rejects type 12 (L406), and `q38_forward_cuda_matvec_backend` rejects Q4_K (L2229-2231). `q38_cuda_gdn_project` has Q4_K absent too (only F32/Q8_0/Q2_K/BF16).
- **No Q4_K test for a matvec** (only decode-parity and the naive MoE end-to-end test).
- **No Q4_K scale/min extraction in a vectorized/`uint32`-packed form** (scalar byte extraction only).
