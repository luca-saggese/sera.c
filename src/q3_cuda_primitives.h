/*
 * CUDA quantized primitives for the q3 runtime.
 *
 * Ported from the q38 donor (COPY -> RENAME -> EDIT):
 * src/q38_cuda_primitives.h and cuda/q38_cuda_primitives.cu.
 * Trimmed to the Q4_K decode primitive that M1 needs as a device-side oracle
 * for the resident Q4_K linear kernel. Everything unrelated to Q4_K
 * (RMSNorm, SiLU, Q2_K, BF16, generic batch kernels) was removed.
 */
#ifndef Q3_CUDA_PRIMITIVES_H
#define Q3_CUDA_PRIMITIVES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "q3_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

bool q3_cuda_dequantize_row(uint32_t type, const void *blocks,
                             size_t block_count, float *out,
                             cudaStream_t stream, char *error,
                             size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
