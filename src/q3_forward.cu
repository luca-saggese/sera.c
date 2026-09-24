/* SPDX-License-Identifier: MIT
 *
 * q3_forward.cu - M2 Qwen3-32B Q4 one-pass forward (CUDA implementation).
 *
 *     token IDs -> embedding -> 64 x (attention + MLP) -> final norm
 *               -> candidate-only LM rows -> logits
 *
 * Primitive kernels are adapted from the q38 donor (COPY -> RENAME -> EDIT):
 *   - rms_norm_kernel / silu_kernel : _reference/q38.c/cuda/q38_cuda_primitives.cu
 *   - rope_kernel                   : _reference/q38.c/cuda/q38_qsa_cuda.cu:328
 *     (NeoX half-split layout kept; theta fixed to 1e6 for Qwen3; the DS4
 *      sections plumbing is dropped)
 *   - attention_kernel structure    : _reference/q38.c/cuda/q38_qsa_cuda.cu:506
 *     (direct kv_head = q_head / group mapping kept; causal mask added;
 *      the QSA gather pre-step is removed)
 *
 * The hot forward path performs zero cudaMalloc/cudaFree and zero host
 * synchronizations (M2 §21, §33). All scratch lives in a grow-once workspace
 * owned by the runtime.
 */
#include "q3_forward.h"

#include "q3_q4_linear.h"
#include "q3_quant.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Device helpers
 * ========================================================================= */

__device__ static float q3_half_to_float_dev(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x3ffu;
    uint32_t value;
    if (!exponent) {
        if (!fraction) value = sign;
        else {
            exponent = 1;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x3ffu;
            value = sign | ((exponent + 112u) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return __uint_as_float(value);
}

__device__ static void q4_scale_min_dev(unsigned index, const uint8_t *scales,
                                        uint8_t *scale, uint8_t *minimum) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *minimum = scales[index + 4] & 63u;
    } else {
        *scale = (scales[index + 4] & 0xfu) |
                 ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) |
                   ((scales[index] >> 6) << 4);
    }
}

__device__ static float q4_value_dev(const q3_q4_k_block *block,
                                     unsigned element) {
    const unsigned group = element / 32;
    const unsigned l = element % 32;
    uint8_t scale, minimum;
    q4_scale_min_dev(group, block->scales, &scale, &minimum);
    const uint8_t packed = block->qs[(group / 2) * 32 + l];
    const unsigned quant = group % 2 ? packed >> 4 : packed & 0xfu;
    return q3_half_to_float_dev(block->d) * scale * (float)quant -
           q3_half_to_float_dev(block->dmin) * minimum;
}

__device__ static float q6_value_dev(const uint8_t *block_bytes,
                                     unsigned element) {
    /* block_q6_K: ql[128] low nibbles, qh[64] high 2 bits, scales[16] int8,
     * d half. Layout matches ggml dequantize_row_q6_K and src/q3_quant.c. */
    const uint8_t *ql = block_bytes;
    const uint8_t *qh = block_bytes + 128;
    const int8_t *sc = (const int8_t *)(block_bytes + 128 + 64);
    const uint16_t d_bits = *(const uint16_t *)(block_bytes + 128 + 64 + 16);
    const float d = q3_half_to_float_dev(d_bits);
    const unsigned j = element / 128;
    const unsigned l = element % 128;
    const unsigned is = l / 16;
    int q;
    if (l < 32) {
        q = (int)((ql[l] & 0xfu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
    } else if (l < 64) {
        q = (int)((ql[l] & 0xfu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
    } else if (l < 96) {
        q = (int)((ql[l - 64] >> 4) | (((qh[l - 64] >> 4) & 3u) << 4)) - 32;
    } else {
        q = (int)((ql[l - 64] >> 4) | (((qh[l - 64] >> 6) & 3u) << 4)) - 32;
    }
    const int scale_index = (j * 8) + (is * 2) + (l % 32 >= 16 ? 1 : 0);
    return d * (float)sc[scale_index] * (float)q;
}

/* =========================================================================
 * Batched RMSNorm (M2 §10). One CTA per row; 8 warps reduce the row.
 * Adapted from the q38 donor rms_norm_kernel (single vector) to batched rows.
 * ========================================================================= */

__global__ static void rms_norm_batched_kernel(const float *input,
                                               const float *weight,
                                               float *output, uint32_t rows,
                                               uint32_t width, float epsilon) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *in = input + (size_t)row * width;
    float *out = output + (size_t)row * width;

    constexpr unsigned warp_count = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    __shared__ float warp_sums[warp_count];

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        sum += (double)in[i] * (double)in[i];
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(warp_sums[0] / (float)width + epsilon);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        out[i] = in[i] * inv_rms * weight[i];
}

/* =========================================================================
 * SiLU * up (M2 §20). Fused elementwise: out = silu(gate) * up.
 * ========================================================================= */

__global__ static void silu_mul_kernel(const float *gate, const float *up,
                                       float *out, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float g = gate[index];
        out[index] = (g / (1.0f + expf(-g))) * up[index];
    }
}

/* =========================================================================
 * Residual add (M2 §19). accumulator += addend.
 * ========================================================================= */

__global__ static void residual_add_kernel(float *accumulator,
                                           const float *addend,
                                           size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) accumulator[index] += addend[index];
}

/* =========================================================================
 * RoPE (M2 §13). NeoX half-split over all head_dim dims, theta = 1e6.
 * Adapted from the q38 donor rope_kernel: layout kept, theta fixed, the
 * sections plumbing dropped, and the per-pair powf loop replaced by a
 * precomputed inv_freq table (host-side, once).
 * ========================================================================= */

__global__ static void rope_kernel(float *q, float *k, uint32_t tokens,
                                   uint32_t n_heads, uint32_t n_kv_heads,
                                   uint32_t head_dim, uint32_t position_base,
                                   const float *inv_freq) {
    const uint32_t pairs = head_dim / 2;
    const size_t total = (size_t)tokens * (n_heads + n_kv_heads) * pairs;
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    const size_t p = index % pairs;
    const size_t head = (index / pairs) % (n_heads + n_kv_heads);
    const size_t token = index / (pairs * (n_heads + n_kv_heads));

    const bool is_q = head < n_heads;
    const uint32_t h = is_q ? (uint32_t)head : (uint32_t)(head - n_heads);
    const uint32_t heads = is_q ? n_heads : n_kv_heads;
    const uint32_t position = position_base + (uint32_t)token;

    const float angle = (float)position * inv_freq[p];
    const float c = cosf(angle), s = sinf(angle);

    float *tensor = is_q ? q : k;
    const size_t base = ((size_t)token * heads + h) * head_dim;
    const size_t a = base + p;
    const size_t b = base + head_dim / 2 + p;
    const float x = tensor[a], y = tensor[b];
    tensor[a] = x * c - y * s;
    tensor[b] = x * s + y * c;
}

/* =========================================================================
 * Causal GQA attention (M2 §14-§16). Two-pass max-softmax, one CTA per
 * (token, q_head). Direct kv_head = q_head / group mapping, no KV
 * repetition. Reads K/V from the KV cache (FP32). Written fresh, following
 * the q38 donor attention_kernel structure.
 * ========================================================================= */

__global__ static void attention_kernel(const float *q, const float *k,
                                        const float *v, float *out,
                                        uint32_t tokens, uint32_t n_heads,
                                        uint32_t n_kv_heads, uint32_t head_dim,
                                        uint32_t position_base,
                                        uint32_t kv_length, float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= tokens || head >= n_heads) return;

    const uint32_t group = n_heads / n_kv_heads;
    const uint32_t kv_head = head / group;

    const float *q_row = q + ((size_t)token * n_heads + head) * head_dim;
    const float *k_rows = k + (size_t)kv_head * head_dim;
    const float *v_rows = v + (size_t)kv_head * head_dim;

    const uint32_t lane = threadIdx.x;
    /* One warp per (token, head). Each lane owns head_dim/32 = 4 dims, so
     * the warp shuffle reduces the dot over all head_dim elements. The
     * butterfly reduction leaves the full sum only in lane 0, so it is
     * broadcast back to every lane before max_score/den/acc are computed:
     * all lanes must agree on the same dot. */
    const uint32_t dims_per_lane = head_dim / 32;
    /* Causal: the current token attends to all cached positions up to and
     * including its own absolute position. */
    const uint32_t visible = position_base + token + 1;
    if (visible > kv_length) return;

    /* Pass 1: max score over visible keys. */
    float max_score = -INFINITY;
    for (uint32_t j = 0; j < visible; j++) {
        const float *k_row = k_rows + (size_t)j * n_kv_heads * head_dim;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        max_score = fmaxf(max_score, dot * scale);
    }

    /* Pass 2: softmax denominator + weighted sum of V. */
    float den = 0.0f;
    float acc[4];
    for (uint32_t i = 0; i < dims_per_lane; i++) acc[i] = 0.0f;
    for (uint32_t j = 0; j < visible; j++) {
        const float *k_row = k_rows + (size_t)j * n_kv_heads * head_dim;
        const float *v_row = v_rows + (size_t)j * n_kv_heads * head_dim;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        const float p = __expf(dot * scale - max_score);
        den += p;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            acc[i] += p * v_row[d];
        }
    }

    float *out_row = out + ((size_t)token * n_heads + head) * head_dim;
    for (uint32_t i = 0; i < dims_per_lane; i++) {
        const uint32_t d = lane + i * 32;
        out_row[d] = acc[i] / den;
    }
}

/* =========================================================================
 * Embedding gather (M2 §9). Device-side gather from the resident Q4_K
 * embedding tensor: hidden[t, :] = dequant(token_embd[token_id, :]).
 * ========================================================================= */

__global__ static void embedding_kernel(float *hidden, const uint32_t *token_ids,
                                        const void *weight, uint32_t qtype,
                                        uint32_t hidden_size, uint32_t vocab_size) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = (size_t)gridDim.x * blockDim.x;
    if (index >= elements) return;
    const uint32_t token = (uint32_t)(index / hidden_size);
    const uint32_t feature = (uint32_t)(index % hidden_size);
    if (token >= gridDim.x) return;
    const uint32_t id = token_ids[token];
    if (id >= vocab_size) return;
    const uint32_t blocks_per_row = hidden_size / Q3_QUANT_QK_K;
    const uint32_t block = feature / Q3_QUANT_QK_K;
    const uint32_t element = feature % Q3_QUANT_QK_K;
    const size_t block_index = (size_t)id * blocks_per_row + block;
    float value;
    if (qtype == Q3_QUANT_Q4_K) {
        value = q4_value_dev((const q3_q4_k_block *)weight + block_index, element);
    } else if (qtype == Q3_QUANT_Q6_K) {
        value = q6_value_dev((const uint8_t *)weight + block_index * Q3_QUANT_Q6_K_BLOCK_BYTES,
                             element);
    } else {
        value = 0.0f;
    }
    hidden[index] = value;
}

/* =========================================================================
 * Candidate LM rows (M2 §26-§27). Dot of the final hidden with each
 * candidate row of output.weight (Q6_K or Q4_K). One CTA per candidate.
 * ========================================================================= */

__global__ static void candidate_logits_kernel(const float *hidden,
                                               const void *weight,
                                               uint32_t qtype,
                                               uint32_t hidden_size,
                                               const uint32_t *candidate_ids,
                                               float *logits) {
    const uint32_t c = blockIdx.x;
    const uint32_t id = candidate_ids[c];
    const uint32_t blocks_per_row = hidden_size / Q3_QUANT_QK_K;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    __shared__ float warp_sums[8];
    float sum = 0.0f;
    for (uint32_t block = 0; block < blocks_per_row; block++) {
        const size_t block_index = (size_t)id * blocks_per_row + block;
        for (uint32_t e = threadIdx.x; e < Q3_QUANT_QK_K; e += blockDim.x) {
            const uint32_t feature = block * Q3_QUANT_QK_K + e;
            float w;
            if (qtype == Q3_QUANT_Q4_K) {
                w = q4_value_dev((const q3_q4_k_block *)weight + block_index, e);
            } else {
                w = q6_value_dev((const uint8_t *)weight +
                                     block_index * Q3_QUANT_Q6_K_BLOCK_BYTES, e);
            }
            sum += w * hidden[feature];
        }
    }
    /* Intra-warp reduction, then a cross-warp reduction through shared
     * memory. The block is wider than one warp, so a single shfl_down chain
     * would drop every warp except warp 0. */
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        const uint32_t warps = (blockDim.x + 31u) >> 5u;
        float block_sum = lane < warps ? warp_sums[lane] : 0.0f;
        for (unsigned offset = 16; offset; offset >>= 1)
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        if (lane == 0) logits[c] = block_sum;
    }
}

/* =========================================================================
 * M3 kernels: per-token RoPE positions, two-segment branch-isolated
 * attention, per-branch KV scatter, last-row gather, batched LM head.
 * ========================================================================= */

/* rope_kernel with per-token absolute positions instead of a scalar base
 * (M3 §13). A packed batch mixes branches whose suffix starts at different
 * absolute positions, so the position cannot be `base + token`. */
__global__ static void rope_kernel_positions(float *q, float *k, uint32_t tokens,
                                             uint32_t n_heads,
                                             uint32_t n_kv_heads,
                                             uint32_t head_dim,
                                             const uint32_t *token_positions,
                                             const float *inv_freq) {
    const uint32_t pairs = head_dim / 2;
    const size_t total = (size_t)tokens * (n_heads + n_kv_heads) * pairs;
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    const size_t p = index % pairs;
    const size_t head = (index / pairs) % (n_heads + n_kv_heads);
    const size_t token = index / (pairs * (n_heads + n_kv_heads));

    const bool is_q = head < n_heads;
    const uint32_t h = is_q ? (uint32_t)head : (uint32_t)(head - n_heads);
    const uint32_t heads = is_q ? n_heads : n_kv_heads;
    const uint32_t position = token_positions[token];

    const float angle = (float)position * inv_freq[p];
    const float c = cosf(angle), s = sinf(angle);

    float *tensor = is_q ? q : k;
    const size_t base = ((size_t)token * heads + h) * head_dim;
    const size_t a = base + p;
    const size_t b = base + head_dim / 2 + p;
    const float x = tensor[a], y = tensor[b];
    tensor[a] = x * c - y * s;
    tensor[b] = x * s + y * c;
}

/* Causal GQA attention over the logical view [ shared prefix | own suffix ].
 * The two segments are walked in sequence, so the key indices covered are
 * 0..prefix_len-1 followed by prefix_len..prefix_len+lp, i.e. exactly
 * 0..position with no mask tensor and no materialised concatenation
 * (M3 §12-§14). Branch isolation is the branch term in the suffix base
 * address: no other branch's rows are reachable (§21). */
__global__ static void attention_segmented_kernel(
    const float *q, float *out, uint32_t tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale, const float *prefix_k,
    const float *prefix_v, uint32_t prefix_len, uint32_t prefix_capacity,
    const float *suffix_k, const float *suffix_v, uint32_t suffix_capacity,
    uint32_t suffix_num_layers, uint32_t suffix_layer,
    const uint32_t *token_branch, const uint32_t *token_local_pos) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= tokens || head >= n_heads) return;

    const uint32_t group = n_heads / n_kv_heads;
    const uint32_t kv_head = head / group;
    const uint32_t row = n_kv_heads * head_dim;

    const float *q_row = q + ((size_t)token * n_heads + head) * head_dim;
    const uint32_t lane = threadIdx.x;
    const uint32_t dims_per_lane = head_dim / 32;

    const uint32_t b = token_branch[token];
    const uint32_t lp = token_local_pos[token];
    const size_t branch_base =
        ((size_t)b * suffix_num_layers + suffix_layer) * suffix_capacity * row;
    /* The prefix slabs are [num_layers][prefix_capacity][kv_heads][head_dim]:
     * the layer offset applies to both segments. */
    const size_t prefix_layer_base =
        (size_t)suffix_layer * prefix_capacity * row;
    const float *pk =
        prefix_k + prefix_layer_base + (size_t)kv_head * head_dim;
    const float *pv =
        prefix_v + prefix_layer_base + (size_t)kv_head * head_dim;
    const float *sk = suffix_k + branch_base + (size_t)kv_head * head_dim;
    const float *sv = suffix_v + branch_base + (size_t)kv_head * head_dim;

    /* Pass 1: max score over the two visible segments. */
    float max_score = -INFINITY;
    for (uint32_t j = 0; j < prefix_len; j++) {
        const float *k_row = pk + (size_t)j * row;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        max_score = fmaxf(max_score, dot * scale);
    }
    for (uint32_t j = 0; j <= lp; j++) {
        const float *k_row = sk + (size_t)j * row;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        max_score = fmaxf(max_score, dot * scale);
    }

    /* Pass 2: softmax denominator + weighted sum of V. */
    float den = 0.0f;
    float acc[4];
    for (uint32_t i = 0; i < dims_per_lane; i++) acc[i] = 0.0f;
    for (uint32_t j = 0; j < prefix_len; j++) {
        const float *k_row = pk + (size_t)j * row;
        const float *v_row = pv + (size_t)j * row;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        const float p = __expf(dot * scale - max_score);
        den += p;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            acc[i] += p * v_row[d];
        }
    }
    for (uint32_t j = 0; j <= lp; j++) {
        const float *k_row = sk + (size_t)j * row;
        const float *v_row = sv + (size_t)j * row;
        float dot = 0.0f;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            dot += q_row[d] * k_row[d];
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        const float p = __expf(dot * scale - max_score);
        den += p;
        for (uint32_t i = 0; i < dims_per_lane; i++) {
            const uint32_t d = lane + i * 32;
            acc[i] += p * v_row[d];
        }
    }

    float *out_row = out + ((size_t)token * n_heads + head) * head_dim;
    for (uint32_t i = 0; i < dims_per_lane; i++) {
        const uint32_t d = lane + i * 32;
        out_row[d] = acc[i] / den;
    }
}

/* Scatter the packed batch's roped K/V rows into each token's private
 * branch slab (M3 §22). One launch covers both K and V. */
__global__ static void kv_scatter_kernel(float *suffix_k, float *suffix_v,
                                         const float *k_src,
                                         const float *v_src,
                                         const uint32_t *token_branch,
                                         const uint32_t *token_local_pos,
                                         uint32_t tokens, uint32_t row,
                                         uint32_t suffix_capacity,
                                         uint32_t num_layers,
                                         uint32_t layer) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)tokens * row;
    if (index >= total) return;
    const uint32_t token = (uint32_t)(index / row);
    const uint32_t e = (uint32_t)(index % row);
    const uint32_t b = token_branch[token];
    const uint32_t lp = token_local_pos[token];
    const size_t dst =
        ((size_t)b * num_layers + layer) * suffix_capacity * row +
        (size_t)lp * row + e;
    suffix_k[dst] = k_src[index];
    suffix_v[dst] = v_src[index];
}

/* Gather each branch's last suffix row out of the final-norm buffer. */
__global__ static void gather_last_rows_kernel(const float *src,
                                               const uint32_t *branch_offsets,
                                               uint32_t branch_count,
                                               uint32_t hidden, float *out) {
    const uint32_t b = blockIdx.x;
    if (b >= branch_count) return;
    const uint32_t last = branch_offsets[b + 1] - 1;
    const float *row = src + (size_t)last * hidden;
    float *dst = out + (size_t)b * hidden;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
        dst[i] = row[i];
}

/* Batched selected-row LM head: one CTA per (branch, candidate). */
__global__ static void candidate_logits_batched_kernel(
    const float *hidden_per_branch, const void *weight, uint32_t qtype,
    uint32_t hidden_size, const uint32_t *candidate_ids, uint32_t candidate_count,
    float *logits) {
    const uint32_t index = blockIdx.x;
    const uint32_t b = index / candidate_count;
    const uint32_t id = candidate_ids[index];
    const float *hidden = hidden_per_branch + (size_t)b * hidden_size;
    const uint32_t blocks_per_row = hidden_size / Q3_QUANT_QK_K;
    const uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t block = 0; block < blocks_per_row; block++) {
        const size_t block_index = (size_t)id * blocks_per_row + block;
        for (uint32_t e = lane; e < Q3_QUANT_QK_K; e += blockDim.x) {
            const uint32_t feature = block * Q3_QUANT_QK_K + e;
            float w;
            if (qtype == Q3_QUANT_Q4_K) {
                w = q4_value_dev((const q3_q4_k_block *)weight + block_index, e);
            } else {
                w = q6_value_dev((const uint8_t *)weight +
                                     block_index * Q3_QUANT_Q6_K_BLOCK_BYTES, e);
            }
            sum += w * hidden[feature];
        }
    }
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) logits[index] = sum;
}

/* =========================================================================
 * Host-side helpers
 * ========================================================================= */

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) {
        error[0] = '\0';
        snprintf(error, error_len, "%s", message);
    }
}

static bool fail_cuda_code(char *error, size_t error_len, const char *what,
                           cudaError_t code) {
    if (error && error_len) {
        error[0] = '\0';
        snprintf(error, error_len, "%s: %s", what, cudaGetErrorString(code));
    }
    return false;
}

/* Check the last CUDA call and report it. Must be used immediately after the
 * call: cudaGetLastError() clears the error, so capturing the code here (and
 * not re-reading it in the reporter) is what keeps the message meaningful. */
#define Q3_CUDA_CHECK(what) do { \
    cudaError_t q3_cuda_code_ = cudaGetLastError(); \
    if (q3_cuda_code_ != cudaSuccess) \
        return fail_cuda_code(error, error_len, (what), q3_cuda_code_); \
} while (0)

/* =========================================================================
 * KV cache (M2 §17-§18). FP32, layout [layer][position][kv_head][head_dim].
 * ========================================================================= */

struct q3_kv_cache {
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t capacity;
    uint32_t length;
    float *k; /* [num_layers][capacity][num_kv_heads][head_dim] */
    float *v;
    /* False for the aliasing header used by the M3 prefix prefill: the slabs
     * belong to the prefix and must not be freed here (M3 §9). */
    bool owns_storage;
};

/* =========================================================================
 * Primitive surface (Test 1 — primitive semantics)
 * ========================================================================= */

extern "C" bool q3_cuda_rms_norm(const float *input, const float *weight,
                                 uint32_t rows, uint32_t width, float epsilon,
                                 float *output, void *stream, char *error,
                                 size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!input || !weight || !output || rows == 0 || width == 0) {
        set_error(error, error_len, "q3_cuda_rms_norm: invalid argument");
        return false;
    }
    rms_norm_batched_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        input, weight, output, rows, width, epsilon);
    Q3_CUDA_CHECK("q3_cuda_rms_norm launch failed");
    return true;
}

extern "C" bool q3_cuda_silu_mul(const float *gate, const float *up, float *out,
                                 size_t elements, void *stream, char *error,
                                 size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!gate || !up || !out || elements == 0) {
        set_error(error, error_len, "q3_cuda_silu_mul: invalid argument");
        return false;
    }
    const unsigned grid = (unsigned)((elements + 255) / 256);
    silu_mul_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(gate, up, out, elements);
    Q3_CUDA_CHECK("q3_cuda_silu_mul launch failed");
    return true;
}

extern "C" bool q3_cuda_residual_add(float *accumulator, const float *addend,
                                     size_t elements, void *stream, char *error,
                                     size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!accumulator || !addend || elements == 0) {
        set_error(error, error_len, "q3_cuda_residual_add: invalid argument");
        return false;
    }
    const unsigned grid = (unsigned)((elements + 255) / 256);
    residual_add_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        accumulator, addend, elements);
    Q3_CUDA_CHECK("q3_cuda_residual_add launch failed");
    return true;
}

extern "C" bool q3_cuda_rope(float *q, float *k, uint32_t tokens,
                             uint32_t n_heads, uint32_t n_kv_heads,
                             uint32_t head_dim, uint32_t position_base,
                             const float *inv_freq, void *stream, char *error,
                             size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q || !k || !inv_freq || tokens == 0 || head_dim == 0 ||
        head_dim % 2 != 0) {
        set_error(error, error_len, "q3_cuda_rope: invalid argument");
        return false;
    }
    const size_t total = (size_t)tokens * (n_heads + n_kv_heads) * (head_dim / 2);
    const unsigned grid = (unsigned)((total + 255) / 256);
    rope_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        q, k, tokens, n_heads, n_kv_heads, head_dim, position_base, inv_freq);
    Q3_CUDA_CHECK("q3_cuda_rope launch failed");
    return true;
}

extern "C" bool q3_cuda_attention(const float *q, const float *k,
                                  const float *v, uint32_t tokens,
                                  uint32_t n_heads, uint32_t n_kv_heads,
                                  uint32_t head_dim, uint32_t position_base,
                                  q3_kv_cache *kv, uint32_t layer, float scale,
                                  float *out, void *stream, char *error,
                                  size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q || !out || tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0) {
        set_error(error, error_len, "q3_cuda_attention: invalid argument");
        return false;
    }
    /* Source of K/V: the layer's slice of the KV cache when one is provided
     * (cached-suffix path), otherwise the caller's workspace rows (prefill).
     * The runtime appends this batch's K/V rows before calling, so the
     * effective visible length is the cached prefix plus this batch. */
    const float *k_src = k;
    const float *v_src = v;
    uint32_t kv_length = tokens;
    if (kv) {
        const size_t per_layer =
            (size_t)kv->capacity * n_kv_heads * head_dim;
        k_src = kv->k + (size_t)layer * per_layer;
        v_src = kv->v + (size_t)layer * per_layer;
        kv_length = kv->length + tokens;
    }
    dim3 grid(tokens, n_heads);
    attention_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        q, k_src, v_src, out, tokens, n_heads, n_kv_heads, head_dim,
        position_base, kv_length, scale);
    Q3_CUDA_CHECK("q3_cuda_attention launch failed");
    return true;
}

extern "C" bool q3_cuda_embedding(float *hidden, const uint32_t *token_ids,
                                  uint32_t token_count, const void *weight,
                                  uint32_t qtype, uint32_t hidden_size,
                                  uint32_t vocab_size, void *stream, char *error,
                                  size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !token_ids || !weight || token_count == 0 ||
        hidden_size == 0 || hidden_size % Q3_QUANT_QK_K != 0) {
        set_error(error, error_len, "q3_cuda_embedding: invalid argument");
        return false;
    }
    const size_t elements = (size_t)token_count * hidden_size;
    const unsigned grid = (unsigned)((elements + 255) / 256);
    embedding_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        hidden, token_ids, weight, qtype, hidden_size, vocab_size);
    Q3_CUDA_CHECK("q3_cuda_embedding launch failed");
    return true;
}

/* --- M3 primitive surface ------------------------------------------ */

extern "C" bool q3_cuda_rope_positions(float *q, float *k, uint32_t tokens,
                                       uint32_t n_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim,
                                       const uint32_t *token_positions,
                                       const float *inv_freq, void *stream,
                                       char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q || !k || !inv_freq || !token_positions || tokens == 0 ||
        head_dim == 0 || head_dim % 2 != 0) {
        set_error(error, error_len, "q3_cuda_rope_positions: invalid argument");
        return false;
    }
    const size_t total = (size_t)tokens * (n_heads + n_kv_heads) * (head_dim / 2);
    const unsigned grid = (unsigned)((total + 255) / 256);
    rope_kernel_positions<<<grid, 256, 0, (cudaStream_t)stream>>>(
        q, k, tokens, n_heads, n_kv_heads, head_dim, token_positions, inv_freq);
    Q3_CUDA_CHECK("q3_cuda_rope_positions launch failed");
    return true;
}

extern "C" bool q3_cuda_attention_segmented(
    const float *q, const float *prefix_k, const float *prefix_v,
    uint32_t prefix_len, uint32_t prefix_capacity, const float *suffix_k,
    const float *suffix_v, uint32_t suffix_capacity, uint32_t suffix_num_layers,
    uint32_t suffix_layer, const uint32_t *token_branch,
    const uint32_t *token_local_pos, uint32_t tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float scale, float *out,
    void *stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q || !out || !prefix_k || !prefix_v || !suffix_k || !suffix_v ||
        !token_branch || !token_local_pos || tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || head_dim % 32 != 0 ||
        prefix_len > prefix_capacity || suffix_num_layers == 0) {
        set_error(error, error_len,
                  "q3_cuda_attention_segmented: invalid argument");
        return false;
    }
    dim3 grid(tokens, n_heads);
    attention_segmented_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        q, out, tokens, n_heads, n_kv_heads, head_dim, scale, prefix_k, prefix_v,
        prefix_len, prefix_capacity, suffix_k, suffix_v, suffix_capacity,
        suffix_num_layers, suffix_layer, token_branch, token_local_pos);
    Q3_CUDA_CHECK("q3_cuda_attention_segmented launch failed");
    return true;
}

extern "C" bool q3_candidate_logits_batched(
    q3_forward_runtime *runtime, const float *device_hidden_per_branch,
    uint32_t branch_count, const uint32_t *candidate_ids_host,
    uint32_t candidate_count, float *device_logits, char *error,
    size_t error_len);

/* =========================================================================
 * KV cache API (M2 §17-§18). FP32, layout [layer][position][kv_head][head_dim].
 * ========================================================================= */

extern "C" q3_kv_cache *q3_kv_cache_init(uint32_t num_layers,
                                         uint32_t num_kv_heads,
                                         uint32_t head_dim, uint32_t capacity,
                                         char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (num_layers == 0 || num_kv_heads == 0 || head_dim == 0 ||
        capacity == 0) {
        set_error(error, error_len, "q3_kv_cache_init: invalid argument");
        return NULL;
    }
    q3_kv_cache *kv = (q3_kv_cache *)calloc(1, sizeof(q3_kv_cache));
    if (!kv) {
        set_error(error, error_len, "q3_kv_cache_init: allocation failed");
        return NULL;
    }
    kv->num_layers = num_layers;
    kv->num_kv_heads = num_kv_heads;
    kv->head_dim = head_dim;
    kv->capacity = capacity;
    kv->length = 0;
    kv->owns_storage = true;
    const size_t per_layer = (size_t)capacity * num_kv_heads * head_dim;
    if (cudaMalloc(&kv->k, per_layer * num_layers * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&kv->v, per_layer * num_layers * sizeof(float)) != cudaSuccess) {
        set_error(error, error_len, "q3_kv_cache_init: cudaMalloc failed");
        if (kv->k) cudaFree(kv->k);
        free(kv);
        return NULL;
    }
    return kv;
}

extern "C" bool q3_kv_cache_reserve(q3_kv_cache *kv, uint32_t capacity,
                                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!kv) {
        set_error(error, error_len, "q3_kv_cache_reserve: null cache");
        return false;
    }
    if (capacity <= kv->capacity) return true;
    /* Grow: allocate new, copy the used prefix, free old. */
    const size_t per_layer = (size_t)capacity * kv->num_kv_heads * kv->head_dim;
    float *new_k = NULL, *new_v = NULL;
    if (cudaMalloc(&new_k, per_layer * kv->num_layers * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&new_v, per_layer * kv->num_layers * sizeof(float)) != cudaSuccess) {
        if (new_k) cudaFree(new_k);
        set_error(error, error_len, "q3_kv_cache_reserve: cudaMalloc failed");
        return false;
    }
    const size_t used = (size_t)kv->length * kv->num_kv_heads * kv->head_dim;
    for (uint32_t l = 0; l < kv->num_layers; l++) {
        cudaMemcpyAsync(new_k + (size_t)l * per_layer,
                        kv->k + (size_t)l * kv->capacity * kv->num_kv_heads * kv->head_dim,
                        used * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(new_v + (size_t)l * per_layer,
                        kv->v + (size_t)l * kv->capacity * kv->num_kv_heads * kv->head_dim,
                        used * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaFree(kv->k);
    cudaFree(kv->v);
    kv->k = new_k;
    kv->v = new_v;
    kv->capacity = capacity;
    return true;
}

extern "C" void q3_kv_cache_reset(q3_kv_cache *kv) {
    if (kv) kv->length = 0;
}

extern "C" void q3_kv_cache_destroy(q3_kv_cache *kv) {
    if (!kv) return;
    if (kv->owns_storage) {
        if (kv->k) cudaFree(kv->k);
        if (kv->v) cudaFree(kv->v);
    }
    free(kv);
}

/* Borrowing header: the M3 prefix prefill points the ordinary append path at
 * the prefix slabs so roped K/V land there directly, with no temporary KV and
 * no copy (M3 §9). */
extern "C" q3_kv_cache *q3_kv_cache_alias(float *k, float *v,
                                          uint32_t num_layers,
                                          uint32_t num_kv_heads,
                                          uint32_t head_dim, uint32_t capacity,
                                          uint32_t length, char *error,
                                          size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!k || !v || num_layers == 0 || num_kv_heads == 0 || head_dim == 0 ||
        capacity == 0) {
        set_error(error, error_len, "q3_kv_cache_alias: invalid argument");
        return NULL;
    }
    q3_kv_cache *kv = (q3_kv_cache *)calloc(1, sizeof(q3_kv_cache));
    if (!kv) {
        set_error(error, error_len, "q3_kv_cache_alias: allocation failed");
        return NULL;
    }
    kv->num_layers = num_layers;
    kv->num_kv_heads = num_kv_heads;
    kv->head_dim = head_dim;
    kv->capacity = capacity;
    kv->length = length;
    kv->k = k;
    kv->v = v;
    kv->owns_storage = false;
    return kv;
}

extern "C" uint32_t q3_kv_cache_length(const q3_kv_cache *kv) {
    return kv ? kv->length : 0;
}

/* =========================================================================
 * Forward runtime
 * ========================================================================= */

struct q3_forward_runtime {
    q3_weights weights;
    q3_model_config config;
    uint32_t max_tokens;
    int device;
    cudaStream_t stream;

    /* Workspace (one grow-once allocation, sliced). */
    float *workspace;
    size_t workspace_bytes;

    float *hidden;   /* [max_tokens, hidden] */
    float *normed;   /* [max_tokens, hidden] */
    float *q;        /* [max_tokens, n_heads*head_dim] */
    float *k;        /* [max_tokens, n_kv_heads*head_dim] */
    float *v;        /* [max_tokens, n_kv_heads*head_dim] */
    float *attn;     /* [max_tokens, n_heads*head_dim] */
    float *oproj;    /* [max_tokens, hidden] */
    float *gate;     /* [max_tokens, intermediate] */
    float *up;       /* [max_tokens, intermediate] */
    float *down;     /* [max_tokens, hidden] */
    float *final;    /* [max_tokens, hidden] */
    float *logits;   /* [candidate_count] */
    float *inv_freq; /* [head_dim/2] */
    uint32_t *candidate_ids_dev; /* [candidate_count] */
    uint32_t candidate_capacity;

    /* MMQ arena (sliced from the same allocation). */
    void *mmq_arena;
    size_t mmq_arena_bytes;

    /* Timing events. */
    cudaEvent_t phase_begin[Q3_FWD_PHASE_COUNT];
    cudaEvent_t phase_end[Q3_FWD_PHASE_COUNT];

    /* Last-run bookkeeping for the buffer accessor and stats. */
    q3_forward_stats *last_stats;
    uint32_t last_token_count;
    uint32_t last_first_layer;
    uint32_t last_last_layer;
    bool last_ran;

    /* Which phases the most recent run actually recorded. run_range() skips
     * the embedding phase when it does not start at layer 0 and never records
     * final-norm, and q3_candidate_logits() is a separate call, so reading an
     * unrecorded event would raise a sticky CUDA error that poisons every
     * later call. */
    bool last_embedding_ran;
    bool last_final_norm_ran;
    bool last_candidate_ran;
};

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

extern "C" q3_forward_runtime *q3_forward_create(const q3_weights *weights,
                                                 const q3_model_config *config,
                                                 uint32_t max_tokens, int device,
                                                 char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weights || !config || max_tokens == 0) {
        set_error(error, error_len, "q3_forward_create: invalid argument");
        return NULL;
    }
    q3_forward_runtime *rt = (q3_forward_runtime *)calloc(1, sizeof(*rt));
    if (!rt) {
        set_error(error, error_len, "q3_forward_create: allocation failed");
        return NULL;
    }
    rt->weights = *weights;
    rt->config = *config;
    rt->max_tokens = max_tokens;
    rt->device = device;

    if (cudaSetDevice(device) != cudaSuccess ||
        cudaStreamCreate(&rt->stream) != cudaSuccess) {
        set_error(error, error_len, "q3_forward_create: CUDA init failed");
        free(rt);
        return NULL;
    }

    const uint32_t H = config->num_attention_heads;
    const uint32_t Hkv = config->num_kv_heads;
    const uint32_t D = config->head_dim;
    const uint32_t hidden = config->hidden_size;
    const uint32_t inter = config->intermediate_size;
    const size_t T = max_tokens;

    /* MMQ arena size (needed before the workspace layout). Size it for the
     * worst case the runtime can actually reach: the largest feature count
     * (the LM head's row count) and the largest contraction length (the
     * FFN's intermediate_size), at this runtime's token capacity.
     *
     * The feature count must come from the bound output tensor, not from
     * config->vocab_size: the Qwen3 GGUF carries no vocab key, so that
     * field is legitimately 0 and would collapse the arena back to its
     * fallback floor. */
    {
        const int64_t max_features =
            (int64_t) (weights->output.n ? weights->output.n
                                         : config->vocab_size);
        const int64_t max_k = (int64_t) config->intermediate_size;
        if (!q3_cuda_q4k_linear_init(device, (size_t) max_tokens, max_features,
                                     max_k, error, error_len)) {
            free(rt);
            return NULL;
        }
    }
    rt->mmq_arena_bytes = q3_cuda_q4k_linear_workspace_bytes();
    rt->candidate_capacity = Q3_CANDIDATE_CAPACITY;

    /* Workspace layout: compute the total size first, then slice. */
    const size_t align = 256;
    size_t total = 0;
    total += align_up(T * hidden * sizeof(float), align);      /* hidden   */
    total += align_up(T * hidden * sizeof(float), align);      /* normed   */
    total += align_up(T * H * D * sizeof(float), align);       /* q        */
    total += align_up(T * Hkv * D * sizeof(float), align);     /* k        */
    total += align_up(T * Hkv * D * sizeof(float), align);     /* v        */
    total += align_up(T * H * D * sizeof(float), align);       /* attn     */
    total += align_up(T * hidden * sizeof(float), align);      /* oproj    */
    total += align_up(T * inter * sizeof(float), align);       /* gate     */
    total += align_up(T * inter * sizeof(float), align);       /* up       */
    total += align_up(T * hidden * sizeof(float), align);      /* down     */
    total += align_up(T * hidden * sizeof(float), align);      /* final    */
    total += align_up(Q3_CANDIDATE_CAPACITY * sizeof(float), align);     /* logits   */
    total += align_up((D / 2) * sizeof(float), align);         /* inv_freq */
    total += align_up(Q3_CANDIDATE_CAPACITY * sizeof(uint32_t), align);  /* cand ids */
    total += align_up(rt->mmq_arena_bytes, align);             /* mmq arena */

    rt->workspace_bytes = total;
    if (cudaMalloc(&rt->workspace, rt->workspace_bytes) != cudaSuccess) {
        set_error(error, error_len, "q3_forward_create: workspace cudaMalloc failed");
        cudaStreamDestroy(rt->stream);
        free(rt);
        return NULL;
    }

    {
        char *p = (char *)rt->workspace;
        rt->hidden = (float *)p; p += align_up(T * hidden * sizeof(float), align);
        rt->normed = (float *)p; p += align_up(T * hidden * sizeof(float), align);
        rt->q = (float *)p; p += align_up(T * H * D * sizeof(float), align);
        rt->k = (float *)p; p += align_up(T * Hkv * D * sizeof(float), align);
        rt->v = (float *)p; p += align_up(T * Hkv * D * sizeof(float), align);
        rt->attn = (float *)p; p += align_up(T * H * D * sizeof(float), align);
        rt->oproj = (float *)p; p += align_up(T * hidden * sizeof(float), align);
        rt->gate = (float *)p; p += align_up(T * inter * sizeof(float), align);
        rt->up = (float *)p; p += align_up(T * inter * sizeof(float), align);
        rt->down = (float *)p; p += align_up(T * hidden * sizeof(float), align);
        rt->final = (float *)p; p += align_up(T * hidden * sizeof(float), align);
        rt->logits = (float *)p; p += align_up(Q3_CANDIDATE_CAPACITY * sizeof(float), align);
        rt->inv_freq = (float *)p; p += align_up((D / 2) * sizeof(float), align);
        rt->candidate_ids_dev = (uint32_t *)p; p += align_up(Q3_CANDIDATE_CAPACITY * sizeof(uint32_t), align);
        rt->mmq_arena = (void *)p; p += align_up(rt->mmq_arena_bytes, align);
    }

    if (q3_cuda_q4k_linear_set_workspace(rt->mmq_arena, rt->mmq_arena_bytes,
                                         error, error_len) != true) {
        cudaFree(rt->workspace);
        cudaStreamDestroy(rt->stream);
        free(rt);
        return NULL;
    }

    /* Precompute inv_freq on the host, upload once. */
    {
        float *host_freq = (float *)malloc((D / 2) * sizeof(float));
        if (!host_freq) {
            cudaFree(rt->workspace);
            cudaStreamDestroy(rt->stream);
            free(rt);
            set_error(error, error_len, "q3_forward_create: allocation failed");
            return NULL;
        }
        const double theta = config->rope_theta > 0.0 ? (double)config->rope_theta : 1e6;
        for (uint32_t i = 0; i < D / 2; i++)
            host_freq[i] = (float)(1.0 / pow(theta, (double)(2 * i) / (double)D));
        cudaMemcpy(rt->inv_freq, host_freq, (D / 2) * sizeof(float),
                   cudaMemcpyHostToDevice);
        free(host_freq);
    }

    for (int i = 0; i < Q3_FWD_PHASE_COUNT; i++) {
        cudaEventCreate(&rt->phase_begin[i]);
        cudaEventCreate(&rt->phase_end[i]);
    }

    return rt;
}

extern "C" void q3_forward_destroy(q3_forward_runtime *runtime) {
    if (!runtime) return;
    for (int i = 0; i < Q3_FWD_PHASE_COUNT; i++) {
        if (runtime->phase_begin[i]) cudaEventDestroy(runtime->phase_begin[i]);
        if (runtime->phase_end[i]) cudaEventDestroy(runtime->phase_end[i]);
    }
    if (runtime->workspace) cudaFree(runtime->workspace);
    if (runtime->stream) cudaStreamDestroy(runtime->stream);
    free(runtime);
}

/* =========================================================================
 * One decoder layer (M2 §22).
 * ========================================================================= */

/* =========================================================================
 * Public runtime API
 * ========================================================================= */

/* Phase-level NaN diagnostics (Q3_DIAG_PHASES=1). Debug-only: syncs the
 * stream and inspects a producer's output immediately. Removed once the
 * first broken boundary is found. */
static void diag_check(cudaStream_t stream, const char *name,
                       const float *dev, size_t elements) {
    if (!getenv("Q3_DIAG_PHASES")) return;
    cudaStreamSynchronize(stream);
    float *host = (float *)malloc(elements * sizeof(float));
    if (!host) return;
    cudaMemcpy(host, dev, elements * sizeof(float), cudaMemcpyDeviceToHost);
    size_t nan = 0, inf = 0;
    float max_abs = 0.0f;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < elements; i++) {
        const float v = host[i];
        if (isnan(v)) { nan++; if (first_bad == (size_t)-1) first_bad = i; }
        else if (isinf(v)) { inf++; if (first_bad == (size_t)-1) first_bad = i; }
        else { const float a = fabsf(v); if (a > max_abs) max_abs = a; }
    }
    fprintf(stderr, "diag: %-28s nan=%zu inf=%zu max_abs=%.4e first_bad=%zu\n",
            name, nan, inf, max_abs, first_bad);
    free(host);
}

static bool run_layer(q3_forward_runtime *rt, uint32_t layer, q3_kv_cache *kv,
                      uint32_t position_base, uint32_t token_count,
                      float *hidden_inout, char *error, size_t error_len) {
    const q3_layer_weights *w = &rt->weights.layer[layer];
    const q3_model_config *cfg = &rt->config;
    const uint32_t H = cfg->num_attention_heads;
    const uint32_t Hkv = cfg->num_kv_heads;
    const uint32_t D = cfg->head_dim;
    const uint32_t hidden = cfg->hidden_size;
    const uint32_t inter = cfg->intermediate_size;
    cudaStream_t stream = rt->stream;
    const float eps = cfg->rms_norm_eps;
    const float scale = 1.0f / sqrtf((float)D);

    q3_q4_linear_geometry geo;
    q3_q4_linear_stats lstats;

    /* 1. input norm. */
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_NORM], stream);
    if (!q3_cuda_rms_norm(hidden_inout, (const float *)w->input_norm.device,
                          token_count, hidden, eps, rt->normed, stream,
                          error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_NORM], stream);
    diag_check(stream, "input_norm", rt->normed, (size_t)token_count * hidden);

    /* 2. Q/K/V projections. */
    if (!q3_q4_linear_bind_geometry(w->q_proj.qtype, w->q_proj.k, w->q_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_Q], stream);
    if (!q3_cuda_q4k_linear(&geo, w->q_proj.device, rt->normed, token_count,
                            rt->q, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_Q], stream);
    diag_check(stream, "q_proj", rt->q, (size_t)token_count * H * D);

    if (!q3_q4_linear_bind_geometry(w->k_proj.qtype, w->k_proj.k, w->k_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_K], stream);
    if (!q3_cuda_q4k_linear(&geo, w->k_proj.device, rt->normed, token_count,
                            rt->k, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_K], stream);
    diag_check(stream, "k_proj", rt->k, (size_t)token_count * Hkv * D);

    if (!q3_q4_linear_bind_geometry(w->v_proj.qtype, w->v_proj.k, w->v_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_V], stream);
    if (!q3_cuda_q4k_linear(&geo, w->v_proj.device, rt->normed, token_count,
                            rt->v, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_V], stream);
    diag_check(stream, "v_proj", rt->v, (size_t)token_count * Hkv * D);

    /* 3. Q/K per-head norm (over head_dim). */
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_QK_NORM], stream);
    if (!q3_cuda_rms_norm(rt->q, (const float *)w->q_norm.device,
                          token_count * H, D, eps, rt->q, stream,
                          error, error_len))
        return false;
    if (!q3_cuda_rms_norm(rt->k, (const float *)w->k_norm.device,
                          token_count * Hkv, D, eps, rt->k, stream,
                          error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_QK_NORM], stream);
    diag_check(stream, "q_norm", rt->q, (size_t)token_count * H * D);
    diag_check(stream, "k_norm", rt->k, (size_t)token_count * Hkv * D);

    /* 4. RoPE (in place on q and k). */
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ROPE], stream);
    if (!q3_cuda_rope(rt->q, rt->k, token_count, H, Hkv, D, position_base,
                      rt->inv_freq, stream, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ROPE], stream);
    diag_check(stream, "rope_q", rt->q, (size_t)token_count * H * D);
    diag_check(stream, "rope_k", rt->k, (size_t)token_count * Hkv * D);

    /* 5. Append K/V to the cache (D2D row copy), then attention. */
    if (kv) {
        const size_t row = (size_t)kv->num_kv_heads * D;
        const size_t per_layer = (size_t)kv->capacity * row;
        const size_t dst = (size_t)kv->length * row;
        cudaMemcpyAsync(kv->k + (size_t)layer * per_layer + dst, rt->k,
                        (size_t)token_count * row * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(kv->v + (size_t)layer * per_layer + dst, rt->v,
                        (size_t)token_count * row * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ATTN], stream);
    if (!q3_cuda_attention(rt->q, rt->k, rt->v, token_count, H, Hkv, D,
                           position_base, kv, layer, scale, rt->attn, stream,
                           error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ATTN], stream);
    diag_check(stream, "attention", rt->attn, (size_t)token_count * H * D);

    /* 6. O projection + attention residual. */
    if (!q3_q4_linear_bind_geometry(w->o_proj.qtype, w->o_proj.k, w->o_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_OPROJ], stream);
    if (!q3_cuda_q4k_linear(&geo, w->o_proj.device, rt->attn, token_count,
                            rt->oproj, stream, &lstats, error, error_len))
        return false;
    if (!q3_cuda_residual_add(hidden_inout, rt->oproj,
                              (size_t)token_count * hidden, stream,
                              error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_OPROJ], stream);
    diag_check(stream, "o_proj", rt->oproj, (size_t)token_count * hidden);
    diag_check(stream, "attn_residual", hidden_inout, (size_t)token_count * hidden);

    /* 7. MLP: norm -> gate/up -> silu*up -> down -> residual. */
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_MLP_NORM], stream);
    if (!q3_cuda_rms_norm(hidden_inout, (const float *)w->post_attn_norm.device,
                          token_count, hidden, eps, rt->normed, stream,
                          error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_MLP_NORM], stream);
    diag_check(stream, "mlp_norm", rt->normed, (size_t)token_count * hidden);

    if (!q3_q4_linear_bind_geometry(w->gate_proj.qtype, w->gate_proj.k,
                                    w->gate_proj.n, &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_GATE], stream);
    if (!q3_cuda_q4k_linear(&geo, w->gate_proj.device, rt->normed, token_count,
                            rt->gate, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_GATE], stream);
    diag_check(stream, "gate_proj", rt->gate, (size_t)token_count * inter);

    if (!q3_q4_linear_bind_geometry(w->up_proj.qtype, w->up_proj.k,
                                    w->up_proj.n, &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_UP], stream);
    if (!q3_cuda_q4k_linear(&geo, w->up_proj.device, rt->normed, token_count,
                            rt->up, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_UP], stream);
    diag_check(stream, "up_proj", rt->up, (size_t)token_count * inter);

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ACT], stream);
    if (!q3_cuda_silu_mul(rt->gate, rt->up, rt->gate,
                          (size_t)token_count * inter, stream,
                          error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ACT], stream);
    diag_check(stream, "silu_mul", rt->gate, (size_t)token_count * inter);

    if (!q3_q4_linear_bind_geometry(w->down_proj.qtype, w->down_proj.k,
                                    w->down_proj.n, &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_DOWN], stream);
    if (!q3_cuda_q4k_linear(&geo, w->down_proj.device, rt->gate, token_count,
                            rt->down, stream, &lstats, error, error_len))
        return false;
    if (!q3_cuda_residual_add(hidden_inout, rt->down,
                              (size_t)token_count * hidden, stream,
                              error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_DOWN], stream);
    diag_check(stream, "down_proj", rt->down, (size_t)token_count * hidden);
    diag_check(stream, "mlp_residual", hidden_inout, (size_t)token_count * hidden);

    return true;
}

/* =========================================================================
 * Public runtime API
 * ========================================================================= */

extern "C" bool q3_forward_run(q3_forward_runtime *runtime,
                               const uint32_t *token_ids_host,
                               uint32_t token_count, q3_kv_cache *kv,
                               uint32_t position_base, float *device_hidden_out,
                               q3_forward_stats *stats, char *error,
                               size_t error_len) {
    return q3_forward_run_range(runtime, token_ids_host, token_count, kv,
                                position_base, 0, runtime->config.num_layers,
                                stats, error, error_len) &&
           q3_forward_final_norm(runtime, token_count, device_hidden_out,
                                 stats, error, error_len);
}

extern "C" bool q3_forward_run_range(q3_forward_runtime *runtime,
                                     const uint32_t *token_ids_host,
                                     uint32_t token_count, q3_kv_cache *kv,
                                     uint32_t position_base,
                                     uint32_t first_layer, uint32_t last_layer,
                                     q3_forward_stats *stats, char *error,
                                     size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !token_ids_host || token_count == 0 ||
        token_count > runtime->max_tokens || first_layer >= last_layer ||
        last_layer > runtime->config.num_layers) {
        set_error(error, error_len, "q3_forward_run_range: invalid argument");
        return false;
    }
    if (stats) memset(stats, 0, sizeof(*stats));
    runtime->last_stats = stats;
    runtime->last_token_count = token_count;
    runtime->last_first_layer = first_layer;
    runtime->last_last_layer = last_layer;
    runtime->last_ran = true;
    runtime->last_embedding_ran = (first_layer == 0);
    runtime->last_final_norm_ran = false;
    runtime->last_candidate_ran = false;

    cudaStream_t stream = runtime->stream;

    /* Embedding gather (only when starting at layer 0). */
    if (first_layer == 0) {
        cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_EMBED], stream);
        if (!q3_cuda_embedding(runtime->hidden, token_ids_host, token_count,
                               runtime->weights.embedding.device,
                               runtime->weights.embedding.qtype,
                               runtime->config.hidden_size,
                               runtime->weights.embedding.n, stream,
                               error, error_len))
            return false;
        cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_EMBED], stream);
        diag_check(stream, "embedding", runtime->hidden,
                   (size_t)token_count * runtime->config.hidden_size);
    }

    for (uint32_t l = first_layer; l < last_layer; l++) {
        if (!run_layer(runtime, l, kv, position_base, token_count,
                       runtime->hidden, error, error_len))
            return false;
    }

    if (kv) kv->length += token_count;
    if (stats) stats->token_count = token_count;
    if (stats) stats->layers_executed = last_layer - first_layer;
    return true;
}

/* =========================================================================
 * M3 packed forward (M3 §18-§22). One pass over the ragged suffix batch:
 * every token attends to the shared prefix plus its own branch's suffix rows.
 * Structurally the same layer body as run_layer, with three changes: RoPE
 * takes per-token positions, attention is the two-segment branch-isolated
 * kernel, and the KV append scatters into per-branch slabs.
 * ========================================================================= */

static bool run_layer_packed(q3_forward_runtime *rt, uint32_t layer,
                             const q3_pack_view *view, uint32_t token_count,
                             float *hidden_inout, char *error,
                             size_t error_len) {
    const q3_layer_weights *w = &rt->weights.layer[layer];
    const q3_model_config *cfg = &rt->config;
    const uint32_t H = cfg->num_attention_heads;
    const uint32_t Hkv = cfg->num_kv_heads;
    const uint32_t D = cfg->head_dim;
    const uint32_t hidden = cfg->hidden_size;
    const uint32_t inter = cfg->intermediate_size;
    cudaStream_t stream = rt->stream;
    const float eps = cfg->rms_norm_eps;
    const float scale = 1.0f / sqrtf((float)D);

    q3_q4_linear_geometry geo;
    q3_q4_linear_stats lstats;

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_NORM], stream);
    if (!q3_cuda_rms_norm(hidden_inout, (const float *)w->input_norm.device,
                          token_count, hidden, eps, rt->normed, stream, error,
                          error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_NORM], stream);

    if (!q3_q4_linear_bind_geometry(w->q_proj.qtype, w->q_proj.k, w->q_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_Q], stream);
    if (!q3_cuda_q4k_linear(&geo, w->q_proj.device, rt->normed, token_count,
                            rt->q, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_Q], stream);

    if (!q3_q4_linear_bind_geometry(w->k_proj.qtype, w->k_proj.k, w->k_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_K], stream);
    if (!q3_cuda_q4k_linear(&geo, w->k_proj.device, rt->normed, token_count,
                            rt->k, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_K], stream);

    if (!q3_q4_linear_bind_geometry(w->v_proj.qtype, w->v_proj.k, w->v_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_V], stream);
    if (!q3_cuda_q4k_linear(&geo, w->v_proj.device, rt->normed, token_count,
                            rt->v, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_V], stream);

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_QK_NORM], stream);
    if (!q3_cuda_rms_norm(rt->q, (const float *)w->q_norm.device,
                          token_count * H, D, eps, rt->q, stream, error, error_len))
        return false;
    if (!q3_cuda_rms_norm(rt->k, (const float *)w->k_norm.device,
                          token_count * Hkv, D, eps, rt->k, stream, error,
                          error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_QK_NORM], stream);

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ROPE], stream);
    if (!q3_cuda_rope_positions(rt->q, rt->k, token_count, H, Hkv, D,
                                view->token_positions, rt->inv_freq, stream,
                                error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ROPE], stream);

    /* Scatter this batch's K/V into each token's own branch slab. */
    {
        const uint32_t row = Hkv * D;
        const size_t total = (size_t)token_count * row;
        const unsigned grid = (unsigned)((total + 255) / 256);
        kv_scatter_kernel<<<grid, 256, 0, stream>>>(
            view->suffix_k, view->suffix_v, rt->k, rt->v, view->token_branch,
            view->token_local_pos, token_count, row, view->suffix_capacity,
            view->suffix_num_layers, layer);
        Q3_CUDA_CHECK("kv_scatter_kernel launch failed");
    }

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ATTN], stream);
    if (!q3_cuda_attention_segmented(
            rt->q, view->prefix_k, view->prefix_v, view->prefix_len,
            view->prefix_capacity, view->suffix_k, view->suffix_v,
            view->suffix_capacity, view->suffix_num_layers, layer,
            view->token_branch, view->token_local_pos, token_count, H, Hkv, D,
            scale, rt->attn, stream, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ATTN], stream);

    if (!q3_q4_linear_bind_geometry(w->o_proj.qtype, w->o_proj.k, w->o_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_OPROJ], stream);
    if (!q3_cuda_q4k_linear(&geo, w->o_proj.device, rt->attn, token_count,
                            rt->oproj, stream, &lstats, error, error_len))
        return false;
    if (!q3_cuda_residual_add(hidden_inout, rt->oproj,
                              (size_t)token_count * hidden, stream, error,
                              error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_OPROJ], stream);

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_MLP_NORM], stream);
    if (!q3_cuda_rms_norm(hidden_inout, (const float *)w->post_attn_norm.device,
                          token_count, hidden, eps, rt->normed, stream, error,
                          error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_MLP_NORM], stream);

    if (!q3_q4_linear_bind_geometry(w->gate_proj.qtype, w->gate_proj.k,
                                    w->gate_proj.n, &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_GATE], stream);
    if (!q3_cuda_q4k_linear(&geo, w->gate_proj.device, rt->normed, token_count,
                            rt->gate, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_GATE], stream);

    if (!q3_q4_linear_bind_geometry(w->up_proj.qtype, w->up_proj.k, w->up_proj.n,
                                    &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_UP], stream);
    if (!q3_cuda_q4k_linear(&geo, w->up_proj.device, rt->normed, token_count,
                            rt->up, stream, &lstats, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_UP], stream);

    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_ACT], stream);
    if (!q3_cuda_silu_mul(rt->gate, rt->up, rt->gate,
                          (size_t)token_count * inter, stream, error, error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_ACT], stream);

    if (!q3_q4_linear_bind_geometry(w->down_proj.qtype, w->down_proj.k,
                                    w->down_proj.n, &geo, error, error_len))
        return false;
    cudaEventRecord(rt->phase_begin[Q3_FWD_PHASE_DOWN], stream);
    if (!q3_cuda_q4k_linear(&geo, w->down_proj.device, rt->gate, token_count,
                            rt->down, stream, &lstats, error, error_len))
        return false;
    if (!q3_cuda_residual_add(hidden_inout, rt->down,
                              (size_t)token_count * hidden, stream, error,
                              error_len))
        return false;
    cudaEventRecord(rt->phase_end[Q3_FWD_PHASE_DOWN], stream);

    return true;
}

extern "C" bool q3_forward_run_packed(q3_forward_runtime *runtime,
                                      const uint32_t *token_ids_host,
                                      uint32_t token_count,
                                      const q3_pack_view *view,
                                      float *device_hidden_per_branch,
                                      q3_forward_stats *stats, char *error,
                                      size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !token_ids_host || !view || !device_hidden_per_branch ||
        token_count == 0 || token_count > runtime->max_tokens ||
        view->branch_count == 0 || view->branch_count > Q3_PACK_MAX_BRANCHES ||
        !view->token_positions || !view->token_branch ||
        !view->token_local_pos || !view->branch_offsets || !view->prefix_k ||
        !view->prefix_v || !view->suffix_k || !view->suffix_v) {
        set_error(error, error_len, "q3_forward_run_packed: invalid argument");
        return false;
    }
    if (stats) memset(stats, 0, sizeof(*stats));
    runtime->last_stats = stats;
    runtime->last_token_count = token_count;
    runtime->last_first_layer = 0;
    runtime->last_last_layer = runtime->config.num_layers;
    runtime->last_ran = true;
    runtime->last_embedding_ran = true;
    runtime->last_final_norm_ran = false;
    runtime->last_candidate_ran = false;

    cudaStream_t stream = runtime->stream;

    cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_EMBED], stream);
    if (!q3_cuda_embedding(runtime->hidden, token_ids_host, token_count,
                           runtime->weights.embedding.device,
                           runtime->weights.embedding.qtype,
                           runtime->config.hidden_size,
                           runtime->weights.embedding.n, stream, error, error_len))
        return false;
    cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_EMBED], stream);

    for (uint32_t l = 0; l < runtime->config.num_layers; l++) {
        if (!run_layer_packed(runtime, l, view, token_count, runtime->hidden,
                              error, error_len))
            return false;
    }

    cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_FINAL_NORM], stream);
    if (!q3_cuda_rms_norm(runtime->hidden,
                          (const float *)runtime->weights.final_norm.device,
                          token_count, runtime->config.hidden_size,
                          runtime->config.rms_norm_eps, runtime->final, stream,
                          error, error_len))
        return false;
    gather_last_rows_kernel<<<view->branch_count, 128, 0, stream>>>(
        runtime->final, view->branch_offsets, view->branch_count,
        runtime->config.hidden_size, device_hidden_per_branch);
    Q3_CUDA_CHECK("gather_last_rows_kernel launch failed");
    cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_FINAL_NORM], stream);
    runtime->last_final_norm_ran = true;

    if (stats) {
        stats->token_count = token_count;
        stats->layers_executed = runtime->config.num_layers;
    }
    return true;
}

extern "C" bool q3_forward_final_norm(q3_forward_runtime *runtime,
                                      uint32_t token_count,
                                      float *device_hidden_out,
                                      q3_forward_stats *stats, char *error,
                                      size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !device_hidden_out || token_count == 0 ||
        token_count > runtime->max_tokens) {
        set_error(error, error_len, "q3_forward_final_norm: invalid argument");
        return false;
    }
    cudaStream_t stream = runtime->stream;
    cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_FINAL_NORM], stream);
    if (!q3_cuda_rms_norm(runtime->hidden,
                          (const float *)runtime->weights.final_norm.device,
                          token_count, runtime->config.hidden_size,
                          runtime->config.rms_norm_eps, runtime->final, stream,
                          error, error_len))
        return false;
    /* Copy the last token's row out. */
    const size_t last_row = (size_t)(token_count - 1) * runtime->config.hidden_size;
    cudaMemcpyAsync(device_hidden_out, runtime->final + last_row,
                    runtime->config.hidden_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_FINAL_NORM], stream);
    runtime->last_final_norm_ran = true;
    if (stats) stats->token_count = token_count;
    return true;
}

extern "C" bool q3_forward_layer(q3_forward_runtime *runtime, uint32_t layer,
                                 q3_kv_cache *kv, uint32_t position_base,
                                 uint32_t token_count, float *hidden_inout,
                                 q3_forward_stats *stats, char *error,
                                 size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !hidden_inout || layer >= runtime->config.num_layers ||
        token_count == 0 || token_count > runtime->max_tokens) {
        set_error(error, error_len, "q3_forward_layer: invalid argument");
        return false;
    }
    runtime->last_stats = stats;
    runtime->last_token_count = token_count;
    runtime->last_first_layer = layer;
    runtime->last_last_layer = layer + 1;
    runtime->last_ran = true;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!run_layer(runtime, layer, kv, position_base, token_count,
                   hidden_inout, error, error_len))
        return false;
    if (kv) kv->length += token_count;
    if (stats) stats->token_count = token_count;
    if (stats) stats->layers_executed = 1;
    return true;
}

extern "C" const float *q3_forward_buffer(const q3_forward_runtime *runtime,
                                          int which, uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!runtime || !runtime->last_ran) return NULL;
    const uint32_t T = runtime->last_token_count;
    const uint32_t hidden = runtime->config.hidden_size;
    const uint32_t inter = runtime->config.intermediate_size;
    const uint32_t H = runtime->config.num_attention_heads;
    const uint32_t Hkv = runtime->config.num_kv_heads;
    const uint32_t D = runtime->config.head_dim;
    const float *ptr = NULL;
    uint64_t bytes = 0;
    switch (which) {
    case Q3_FB_HIDDEN:  ptr = runtime->hidden; bytes = (uint64_t)T * hidden * 4; break;
    case Q3_FB_NORMED:  ptr = runtime->normed; bytes = (uint64_t)T * hidden * 4; break;
    case Q3_FB_Q:       ptr = runtime->q;      bytes = (uint64_t)T * H * D * 4; break;
    case Q3_FB_Q2:      ptr = runtime->q;      bytes = (uint64_t)T * H * D * 4; break;
    case Q3_FB_K:       ptr = runtime->k;      bytes = (uint64_t)T * Hkv * D * 4; break;
    case Q3_FB_K2:      ptr = runtime->k;      bytes = (uint64_t)T * Hkv * D * 4; break;
    case Q3_FB_V:       ptr = runtime->v;      bytes = (uint64_t)T * Hkv * D * 4; break;
    case Q3_FB_ATTN:    ptr = runtime->attn;   bytes = (uint64_t)T * H * D * 4; break;
    case Q3_FB_OPROJ:   ptr = runtime->oproj;  bytes = (uint64_t)T * hidden * 4; break;
    case Q3_FB_GATE:    ptr = runtime->gate;   bytes = (uint64_t)T * inter * 4; break;
    case Q3_FB_UP:      ptr = runtime->up;     bytes = (uint64_t)T * inter * 4; break;
    case Q3_FB_DOWN:    ptr = runtime->down;   bytes = (uint64_t)T * hidden * 4; break;
    case Q3_FB_FINAL:   ptr = runtime->final;  bytes = (uint64_t)T * hidden * 4; break;
    case Q3_FB_LOGITS:  ptr = runtime->logits; bytes = 64 * 4; break;
    default: return NULL;
    }
    if (bytes_out) *bytes_out = bytes;
    return ptr;
}

extern "C" bool q3_forward_synchronize(q3_forward_runtime *runtime,
                                       char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime) {
        set_error(error, error_len, "q3_forward_synchronize: null runtime");
        return false;
    }
    {
        const cudaError_t code = cudaStreamSynchronize(runtime->stream);
        if (code != cudaSuccess)
            return fail_cuda_code(error, error_len,
                                  "q3_forward_synchronize failed", code);
    }

    /* Read the phase events into the last stats. Only phases the last run
     * actually recorded are read: an unrecorded event would raise a sticky
     * CUDA error. */
    q3_forward_stats *s = runtime->last_stats;
    if (s) {
        float ms = 0.0f;
        if (runtime->last_embedding_ran) {
            cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_EMBED],
                                 runtime->phase_end[Q3_FWD_PHASE_EMBED]);
            s->embedding_ms = ms;
        }
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_NORM],
                             runtime->phase_end[Q3_FWD_PHASE_NORM]);
        s->norm_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_Q],
                             runtime->phase_end[Q3_FWD_PHASE_Q]);
        s->q_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_K],
                             runtime->phase_end[Q3_FWD_PHASE_K]);
        s->k_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_V],
                             runtime->phase_end[Q3_FWD_PHASE_V]);
        s->v_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_QK_NORM],
                             runtime->phase_end[Q3_FWD_PHASE_QK_NORM]);
        s->qk_norm_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_ROPE],
                             runtime->phase_end[Q3_FWD_PHASE_ROPE]);
        s->rope_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_ATTN],
                             runtime->phase_end[Q3_FWD_PHASE_ATTN]);
        s->attention_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_OPROJ],
                             runtime->phase_end[Q3_FWD_PHASE_OPROJ]);
        s->o_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_MLP_NORM],
                             runtime->phase_end[Q3_FWD_PHASE_MLP_NORM]);
        s->mlp_norm_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_GATE],
                             runtime->phase_end[Q3_FWD_PHASE_GATE]);
        s->gate_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_UP],
                             runtime->phase_end[Q3_FWD_PHASE_UP]);
        s->up_proj_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_ACT],
                             runtime->phase_end[Q3_FWD_PHASE_ACT]);
        s->activation_ms = ms;
        cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_DOWN],
                             runtime->phase_end[Q3_FWD_PHASE_DOWN]);
        s->down_proj_ms = ms;
        if (runtime->last_final_norm_ran) {
            cudaEventElapsedTime(&ms,
                                 runtime->phase_begin[Q3_FWD_PHASE_FINAL_NORM],
                                 runtime->phase_end[Q3_FWD_PHASE_FINAL_NORM]);
            s->final_norm_ms = ms;
        }
        if (runtime->last_candidate_ran) {
            cudaEventElapsedTime(&ms, runtime->phase_begin[Q3_FWD_PHASE_CAND],
                                 runtime->phase_end[Q3_FWD_PHASE_CAND]);
            s->candidate_head_ms = ms;
        }
        s->total_forward_ms = s->embedding_ms + s->norm_ms + s->q_proj_ms +
            s->k_proj_ms + s->v_proj_ms + s->qk_norm_ms + s->rope_ms +
            s->attention_ms + s->o_proj_ms + s->mlp_norm_ms + s->gate_proj_ms +
            s->up_proj_ms + s->activation_ms + s->down_proj_ms +
            s->final_norm_ms + s->candidate_head_ms;
    }
    return true;
}

extern "C" bool q3_candidate_logits(q3_forward_runtime *runtime,
                                    const float *device_hidden_last,
                                    const uint32_t *candidate_ids_host,
                                    uint32_t candidate_count,
                                    float *device_logits, char *error,
                                    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !device_hidden_last || !candidate_ids_host ||
        candidate_count == 0 || candidate_count > runtime->candidate_capacity) {
        set_error(error, error_len, "q3_candidate_logits: invalid argument");
        return false;
    }
    cudaStream_t stream = runtime->stream;
    cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_CAND], stream);
    cudaMemcpyAsync(runtime->candidate_ids_dev, candidate_ids_host,
                    candidate_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                    stream);
    const q3_exec_tensor *out = &runtime->weights.output;
    candidate_logits_kernel<<<candidate_count, 128, 0, stream>>>(
        device_hidden_last, out->device, out->qtype, runtime->config.hidden_size,
        runtime->candidate_ids_dev, device_logits);
    Q3_CUDA_CHECK("q3_candidate_logits launch failed");
    cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_CAND], stream);
    runtime->last_candidate_ran = true;
    return true;
}

extern "C" void q3_forward_geometry(const q3_forward_runtime *runtime,
                                    uint32_t *num_layers, uint32_t *hidden_size,
                                    uint32_t *num_heads, uint32_t *num_kv_heads,
                                    uint32_t *head_dim, uint32_t *max_tokens) {
    if (!runtime) return;
    if (num_layers) *num_layers = runtime->config.num_layers;
    if (hidden_size) *hidden_size = runtime->config.hidden_size;
    if (num_heads) *num_heads = runtime->config.num_attention_heads;
    if (num_kv_heads) *num_kv_heads = runtime->config.num_kv_heads;
    if (head_dim) *head_dim = runtime->config.head_dim;
    if (max_tokens) *max_tokens = runtime->max_tokens;
}

extern "C" bool q3_candidate_logits_batched(
    q3_forward_runtime *runtime, const float *device_hidden_per_branch,
    uint32_t branch_count, const uint32_t *candidate_ids_host,
    uint32_t candidate_count, float *device_logits, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !device_hidden_per_branch || !candidate_ids_host ||
        !device_logits || branch_count == 0 || candidate_count == 0 ||
        branch_count * candidate_count > runtime->candidate_capacity) {
        set_error(error, error_len,
                  "q3_candidate_logits_batched: invalid argument");
        return false;
    }
    cudaStream_t stream = runtime->stream;
    cudaEventRecord(runtime->phase_begin[Q3_FWD_PHASE_CAND], stream);
    cudaMemcpyAsync(runtime->candidate_ids_dev, candidate_ids_host,
                    (size_t)branch_count * candidate_count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
    const q3_exec_tensor *out = &runtime->weights.output;
    candidate_logits_batched_kernel<<<branch_count * candidate_count, 128, 0,
                                      stream>>>(
        device_hidden_per_branch, out->device, out->qtype,
        runtime->config.hidden_size, runtime->candidate_ids_dev, candidate_count,
        device_logits);
    Q3_CUDA_CHECK("q3_candidate_logits_batched launch failed");
    cudaEventRecord(runtime->phase_end[Q3_FWD_PHASE_CAND], stream);
    runtime->last_candidate_ran = true;
    return true;
}
