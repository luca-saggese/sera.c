/* SPDX-License-Identifier: MIT
 *
 * q3_q4_linear.h - M1 resident Q4_K quantized linear primitive.
 *
 *     resident Q4_K weight [N, K]
 *          x
 *     FP32 device activation [M, K]
 *          |
 *     Q8_1 activation quantization
 *          |
 *     MMVQ (small M) / MMQ (larger M)
 *          |
 *     FP32 device output [M, N]
 *
 * Weights stay Q4_K resident throughout: no persistent dequantized mirror,
 * no H2D/D2H inside the primitive, no host synchronization inside the
 * primitive. GEMM geometry is resolved once by the binder and validated.
 */
#ifndef Q3_Q4_LINEAR_H
#define Q3_Q4_LINEAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double total_ms;
    double activation_quant_ms;
    double kernel_ms;

    uint64_t weight_bytes;
    uint64_t activation_read_bytes;
    uint64_t activation_write_bytes;

    uint64_t kernel_launches;
    uint64_t cuda_allocations;
    uint64_t host_syncs;
} q3_q4_linear_stats;

/* Resolved, validated GEMM geometry. weight = [N, K], input = [M, K],
 * output = [M, N]; out[col*M + row] is the physical layout the kernel
 * writes (column major over the batch). */
typedef struct {
    int32_t M;
    int32_t N;
    int32_t K;
    uint32_t qtype;   /* GGUF/GGML quant type of the weight */
    uint64_t weight_bytes;
} q3_q4_linear_geometry;

/* Convert the GGUF physical dimensions of a Q4_K tensor into the logical
 * (N, K) the kernel needs, and validate them. One conversion per tensor,
 * done by the binder - never inside the kernel.
 *
 * GGUF stores dim[0] = K (contiguous row length) and dim[1] = N (row count).
 * The M0 resident descriptor exposes those as `.rows = dim[0]` and
 * `.cols = dim[1]`, i.e. rows is really K. This function is the single place
 * that inverts that, so the rest of M1 never touches GGUF dimension order. */
bool q3_q4_linear_bind_geometry(uint32_t qtype,
                                uint32_t gguf_rows_dim0,
                                uint32_t gguf_cols_dim1,
                                q3_q4_linear_geometry *out,
                                char *error, size_t error_len);

/* One-time setup: selects the device, initializes the kernel dispatch, and
 * attaches the persistent scratch arena that keeps the hot path at zero
 * cudaMalloc/cudaFree. Call after the CUDA device is chosen and before any
 * q3_cuda_q4k_linear() call. */
bool q3_cuda_q4k_linear_init(int device, char *error, size_t error_len);

/* Bytes of persistent scratch the linear primitive needs for the given
 * worst-case geometry. Reserve this once per process. */
size_t q3_cuda_q4k_linear_workspace_bytes(void);

/* Attach the persistent scratch arena (device memory, `bytes` long). */
bool q3_cuda_q4k_linear_set_workspace(void *device_ptr, size_t bytes,
                                      char *error, size_t error_len);

/* Which internal path the primitive will pick for this batch size. */
const char *q3_cuda_q4k_linear_path_for_batch(size_t token_count);

/* Resident Q4_K weight x FP32 device activation -> FP32 device output.
 *
 * `weight` must be the resident device pointer of a Q4_K_K tensor with the
 * geometry returned by q3_q4_linear_bind_geometry(). Input and output are
 * device-side; this function never copies to or from the host and never
 * synchronizes. `stream` orders the work; the caller owns synchronization.
 *
 * Returns false with `error` populated on any failure. There is no silent
 * slow fallback: an unsupported geometry fails loudly. */
bool q3_cuda_q4k_linear(const q3_q4_linear_geometry *geometry,
                        const void *weight,
                        const float *device_input,
                        size_t token_count,
                        float *device_output,
                        void *stream,
                        q3_q4_linear_stats *stats,
                        char *error, size_t error_len);

/* M1 diagnostic: --bench-q4-linear. Defined in q3_bench_q4_linear.cu so the
 * CLI (plain C) can call it without seeing CUDA types. */
int q3_cmd_bench_q4_linear(const q3_options *opt);

/* Fill activation_quant_ms / kernel_ms / total_ms from the stream events
 * recorded by the last q3_cuda_q4k_linear() call. Call only after the
 * caller has synchronized that stream (the benchmark harness does). The
 * primitive itself never synchronizes. */
void q3_cuda_q4k_linear_read_stats(q3_q4_linear_stats *stats, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* Q3_Q4_LINEAR_H */