/* SPDX-License-Identifier: MIT
 *
 * q3_mmq.h - host-side C ABI for the q3 Q4_K quantized matmul kernels.
 *
 * Ported from the q38-main donor (COPY -> RENAME -> EDIT):
 *   _reference/q38-main/cuda/mmq/ds4_mmq.h
 *
 * Trimmed for M1: dense only, Q4_K weights only, Q8_1 activation
 * quantization only, GB10 / SM121 only. No MoE, no expert routing, no MMID,
 * no D2R, no MXFP4/NVFP4, no Q2_K/IQ2_XXS/Q8_0 weight paths, no environment
 * tuning hooks, no CUDA-graph scaffolding.
 *
 * The matmul is:
 *
 *     out[col, row] = sum_k W[row, k] * X[k, col]
 *
 * with W in the Q4_K block layout, X FP32 (K innermost, row major) and the
 * result FP32 in the donor's column-major destination layout
 * (out[col * M + row]). The M1 binder owns the single physical->logical
 * dimension conversion; the kernel receives resolved geometry.
 */
#ifndef Q3_MMQ_H
#define Q3_MMQ_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One-time init. Sets the current CUDA device and triggers lazy population
 * of the device-info singleton. Safe to call repeatedly. Returns 0. */
int q3_mmq_init(int device);

/* Optional enqueue-side timestamps. Events are recorded on the caller's
 * stream and are not synchronized here; read them with
 * q3_mmq_read_timings() after the caller has synchronized. Passing NULL
 * disables timing. Setup is one-time, so it never allocates per call. */
typedef struct q3_mmq_timings q3_mmq_timings;

q3_mmq_timings *q3_mmq_timings_create(cudaStream_t stream);
void            q3_mmq_timings_destroy(q3_mmq_timings *timings);

/* Fills activation-quantization and kernel milliseconds. Must be called
 * after the stream the timings were recorded on has been synchronized. */
void q3_mmq_read_timings(q3_mmq_timings *timings,
                         double *quant_ms, double *kernel_ms);

/* Dense Q4_K x FP32 -> FP32 through the MMQ (tile) path. All M.
 *   W     : resident Q4_K weight, N rows of K elements (K % 256 == 0)
 *   X_f32 : FP32 activation, K innermost, row major, M rows
 *   out   : FP32 output, column major, out[col * M + row]
 * Returns 0 on success, non-zero on failure (loudly, never silently). */
int q3_mmq_q4_K_dense(const void *W, const float *X_f32, float *out_f32,
                      int M, int N, int K, cudaStream_t stream,
                      q3_mmq_timings *timings);

/* Dense Q4_K x FP32 -> FP32 through the MMVQ (matrix-vector) path for the
 * small-M regime (M <= Q3_MMVQ_MAX_BATCH). Same memory contract as above. */
int q3_mmq_q4_K_dense_vec(const void *W, const float *X_f32, float *out_f32,
                          int M, int N, int K, cudaStream_t stream,
                          q3_mmq_timings *timings);

/* Dense Q6_K variants of the two entry points above. Identical memory
 * contract; the only difference is the resident weight block layout.
 * Needed for output.weight and the Q6_K attn_v / ffn_down layers. */
int q3_mmq_q6_K_dense(const void *W, const float *X_f32, float *out_f32,
                      int M, int N, int K, cudaStream_t stream,
                      q3_mmq_timings *timings);
int q3_mmq_q6_K_dense_vec(const void *W, const float *X_f32, float *out_f32,
                          int M, int N, int K, cudaStream_t stream,
                          q3_mmq_timings *timings);

/* Bound of the MMVQ regime (donor MMVQ_MAX_BATCH_SIZE). */
#define Q3_MMVQ_MAX_BATCH 8

/* Route the internal pool's cudaMallocAsync/cudaFreeAsync onto the caller's
 * stream. Pass NULL to reset. */
void q3_mmq_pool_set_stream(cudaStream_t stream);

/* Persistent scratch arena, owned by the caller. Every transient buffer the
 * dense path needs (Q8_1 activation staging, stream-K fixup scratch) is
 * sliced out of it, which is what keeps the hot path at zero
 * cudaMalloc/cudaFree. Attach once after q3_mmq_init(). */
int    q3_mmq_set_arena(void *ptr, size_t bytes);
size_t q3_mmq_arena_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* Q3_MMQ_H */