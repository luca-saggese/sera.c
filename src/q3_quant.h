/*
 * Quantized block semantics for the q3 runtime.
 *
 * Ported from the q38 donor (COPY -> RENAME -> EDIT): src/q38_quant.h/.c.
 * Trimmed to the Q4_K layout that M1 requires; the CPU dequantization path is
 * retained as the local correctness oracle for the resident CUDA kernel.
 */
#ifndef Q3_QUANT_H
#define Q3_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    Q3_QUANT_Q4_K = 12,
    Q3_QUANT_Q6_K = 14,
};

#define Q3_QUANT_QK_K 256
#define Q3_QUANT_Q4_K_BLOCK_BYTES 144
#define Q3_QUANT_Q6_K_BLOCK_BYTES 210

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} q3_q4_k_block;

float q3_half_to_float(uint16_t bits);

bool q3_quant_dequantize_row(uint32_t type, const void *blocks,
                             size_t block_count, float *out,
                             size_t out_elements, char *error,
                             size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
