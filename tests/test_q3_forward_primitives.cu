/* SPDX-License-Identifier: MIT
 *
 * test_q3_forward_primitives.cu - M2 Test 1: primitive semantics.
 *
 * Validates the four M2 activation primitives against a CPU oracle on
 * synthetic data. No model is loaded: this test is fully self-contained
 * and runs without the Qwen3 GGUF.
 *
 *   1. batched RMSNorm
 *   2. RoPE (NeoX half-split, theta = 1e6)
 *   3. SiLU * up
 *   4. residual add
 *
 * Each primitive is launched on a private CUDA stream and synchronized
 * before the host-side comparison (test-only sync; the hot path never
 * synchronizes).
 */

#include "q3_forward.h"
#include "q3_cuda.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

static uint32_t rng_state = 0x9e3779b9u;
static float rng_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)0x00ffffffu) - 0.5f;
}

/* ------------------------------------------------------------------ */
/* CPU oracles                                                         */
/* ------------------------------------------------------------------ */

static void cpu_rms_norm(const float *input, const float *weight, uint32_t rows,
                         uint32_t width, float epsilon, float *output) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *in = input + (size_t)r * width;
        float *out = output + (size_t)r * width;
        double sum = 0.0;
        for (uint32_t i = 0; i < width; i++) sum += (double)in[i] * (double)in[i];
        const float inv_rms = 1.0f / sqrtf((float)(sum / (double)width) + epsilon);
        for (uint32_t i = 0; i < width; i++) out[i] = in[i] * inv_rms * weight[i];
    }
}

static void cpu_rope(float *q, float *k, uint32_t tokens, uint32_t n_heads,
                     uint32_t n_kv_heads, uint32_t head_dim,
                     uint32_t position_base, const float *inv_freq) {
    const uint32_t pairs = head_dim / 2;
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t position = position_base + t;
        for (uint32_t h = 0; h < n_heads; h++) {
            float *row = q + ((size_t)t * n_heads + h) * head_dim;
            for (uint32_t p = 0; p < pairs; p++) {
                const float angle = (float)position * inv_freq[p];
                const float c = cosf(angle), s = sinf(angle);
                const float x = row[p], y = row[head_dim / 2 + p];
                row[p] = x * c - y * s;
                row[head_dim / 2 + p] = x * s + y * c;
            }
        }
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            float *row = k + ((size_t)t * n_kv_heads + h) * head_dim;
            for (uint32_t p = 0; p < pairs; p++) {
                const float angle = (float)position * inv_freq[p];
                const float c = cosf(angle), s = sinf(angle);
                const float x = row[p], y = row[head_dim / 2 + p];
                row[p] = x * c - y * s;
                row[head_dim / 2 + p] = x * s + y * c;
            }
        }
    }
}

static void cpu_silu_mul(const float *gate, const float *up, size_t elements,
                         float *out) {
    for (size_t i = 0; i < elements; i++) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* ------------------------------------------------------------------ */
/* Test 1a - batched RMSNorm                                            */
/* ------------------------------------------------------------------ */

static void test_rms_norm(void) {
    fprintf(stderr, "-- test 1a: batched RMSNorm\n");
    float max_abs = 0.0f;
    cudaStream_t stream;
    char err[256] = {0};

    const uint32_t rows = 7, width = 5120;
    const float epsilon = 1e-6f;

    float *input = (float *)malloc((size_t)rows * width * sizeof(float));
    float *weight = (float *)malloc((size_t)width * sizeof(float));
    float *ref = (float *)malloc((size_t)rows * width * sizeof(float));
    float *dev = (float *)malloc((size_t)rows * width * sizeof(float));
    float *d_input = nullptr, *d_weight = nullptr, *d_out = nullptr;
    if (!input || !weight || !ref || !dev) {
        fprintf(stderr, "FAIL: host allocation\n");
        failures++;
        goto done;
    }
    for (size_t i = 0; i < (size_t)rows * width; i++) input[i] = rng_float();
    for (size_t i = 0; i < width; i++) weight[i] = 0.5f + rng_float();

    cpu_rms_norm(input, weight, rows, width, epsilon, ref);

    cudaMalloc(&d_input, (size_t)rows * width * sizeof(float));
    cudaMalloc(&d_weight, (size_t)width * sizeof(float));
    cudaMalloc(&d_out, (size_t)rows * width * sizeof(float));
    cudaMemcpy(d_input, input, (size_t)rows * width * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, (size_t)width * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    if (!q3_cuda_rms_norm(d_input, d_weight, rows, width, epsilon, d_out,
                          stream, err, sizeof(err))) {
        fprintf(stderr, "FAIL: rms_norm: %s\n", err);
        failures++;
        cudaStreamDestroy(stream);
        goto done;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(dev, d_out, (size_t)rows * width * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < (size_t)rows * width; i++) {
        const float e = fabsf(dev[i] - ref[i]);
        if (e > max_abs) max_abs = e;
    }
    fprintf(stderr, "   rows=%u width=%u max_abs=%.3g\n", rows, width, max_abs);
    CHECK(max_abs < 1e-4f, "RMSNorm matches CPU oracle (max_abs < 1e-4)");

done:
    if (d_out) cudaFree(d_out);
    if (d_weight) cudaFree(d_weight);
    if (d_input) cudaFree(d_input);
    free(dev);
    free(ref);
    free(weight);
    free(input);
}

/* ------------------------------------------------------------------ */
/* Test 1b - RoPE                                                       */
/* ------------------------------------------------------------------ */

static void test_rope(void) {
    fprintf(stderr, "-- test 1b: RoPE\n");
    float q_max = 0.0f, k_max = 0.0f;
    cudaStream_t stream;
    char err[256] = {0};

    const uint32_t tokens = 5, n_heads = 64, n_kv_heads = 8, head_dim = 128;
    const uint32_t position_base = 10;
    const uint32_t pairs = head_dim / 2;
    const double theta = 1e6;

    float *inv_freq = (float *)malloc(pairs * sizeof(float));
    for (uint32_t i = 0; i < pairs; i++)
        inv_freq[i] = (float)(1.0 / pow(theta, (double)(2 * i) / (double)head_dim));

    const size_t q_elems = (size_t)tokens * n_heads * head_dim;
    const size_t k_elems = (size_t)tokens * n_kv_heads * head_dim;
    float *q = (float *)malloc(q_elems * sizeof(float));
    float *k = (float *)malloc(k_elems * sizeof(float));
    float *q_ref = (float *)malloc(q_elems * sizeof(float));
    float *k_ref = (float *)malloc(k_elems * sizeof(float));
    float *q_dev = (float *)malloc(q_elems * sizeof(float));
    float *k_dev = (float *)malloc(k_elems * sizeof(float));
    float *d_q = nullptr, *d_k = nullptr, *d_freq = nullptr;
    if (!inv_freq || !q || !k || !q_ref || !k_ref || !q_dev || !k_dev) {
        fprintf(stderr, "FAIL: host allocation\n");
        failures++;
        goto done;
    }
    for (size_t i = 0; i < q_elems; i++) q[i] = rng_float();
    for (size_t i = 0; i < k_elems; i++) k[i] = rng_float();
    memcpy(q_ref, q, q_elems * sizeof(float));
    memcpy(k_ref, k, k_elems * sizeof(float));

    cpu_rope(q_ref, k_ref, tokens, n_heads, n_kv_heads, head_dim,
             position_base, inv_freq);

    cudaMalloc(&d_q, q_elems * sizeof(float));
    cudaMalloc(&d_k, k_elems * sizeof(float));
    cudaMalloc(&d_freq, pairs * sizeof(float));
    cudaMemcpy(d_q, q, q_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, k_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, inv_freq, pairs * sizeof(float), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    if (!q3_cuda_rope(d_q, d_k, tokens, n_heads, n_kv_heads, head_dim,
                      position_base, d_freq, stream, err, sizeof(err))) {
        fprintf(stderr, "FAIL: rope: %s\n", err);
        failures++;
        cudaStreamDestroy(stream);
        goto done;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(q_dev, d_q, q_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_dev, d_k, k_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < q_elems; i++) {
        const float e = fabsf(q_dev[i] - q_ref[i]);
        if (e > q_max) q_max = e;
    }
    for (size_t i = 0; i < k_elems; i++) {
        const float e = fabsf(k_dev[i] - k_ref[i]);
        if (e > k_max) k_max = e;
    }
    fprintf(stderr, "   tokens=%u q_max=%.3g k_max=%.3g\n", tokens, q_max, k_max);
    CHECK(q_max < 1e-4f, "RoPE Q matches CPU oracle (max_abs < 1e-4)");
    CHECK(k_max < 1e-4f, "RoPE K matches CPU oracle (max_abs < 1e-4)");

done:
    if (d_freq) cudaFree(d_freq);
    if (d_k) cudaFree(d_k);
    if (d_q) cudaFree(d_q);
    free(k_dev);
    free(q_dev);
    free(k_ref);
    free(q_ref);
    free(k);
    free(q);
    free(inv_freq);
}

/* ------------------------------------------------------------------ */
/* Test 1c - SiLU * up                                                  */
/* ------------------------------------------------------------------ */

static void test_silu_mul(void) {
    fprintf(stderr, "-- test 1c: SiLU * up\n");
    float max_abs = 0.0f;
    cudaStream_t stream;
    char err[256] = {0};

    const size_t elements = 4096 * 3 + 17;
    float *gate = (float *)malloc(elements * sizeof(float));
    float *up = (float *)malloc(elements * sizeof(float));
    float *ref = (float *)malloc(elements * sizeof(float));
    float *dev = (float *)malloc(elements * sizeof(float));
    float *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    if (!gate || !up || !ref || !dev) {
        fprintf(stderr, "FAIL: host allocation\n");
        failures++;
        goto done;
    }
    for (size_t i = 0; i < elements; i++) {
        gate[i] = rng_float() * 4.0f;
        up[i] = rng_float();
    }
    cpu_silu_mul(gate, up, elements, ref);

    cudaMalloc(&d_gate, elements * sizeof(float));
    cudaMalloc(&d_up, elements * sizeof(float));
    cudaMalloc(&d_out, elements * sizeof(float));
    cudaMemcpy(d_gate, gate, elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up, elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    if (!q3_cuda_silu_mul(d_gate, d_up, d_out, elements, stream,
                          err, sizeof(err))) {
        fprintf(stderr, "FAIL: silu_mul: %s\n", err);
        failures++;
        cudaStreamDestroy(stream);
        goto done;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(dev, d_out, elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < elements; i++) {
        const float e = fabsf(dev[i] - ref[i]);
        if (e > max_abs) max_abs = e;
    }
    fprintf(stderr, "   elements=%zu max_abs=%.3g\n", elements, max_abs);
    CHECK(max_abs < 1e-5f, "SiLU*up matches CPU oracle (max_abs < 1e-5)");

done:
    if (d_out) cudaFree(d_out);
    if (d_up) cudaFree(d_up);
    if (d_gate) cudaFree(d_gate);
    free(dev);
    free(ref);
    free(up);
    free(gate);
}

/* ------------------------------------------------------------------ */
/* Test 1d - residual add                                               */
/* ------------------------------------------------------------------ */

static void test_residual_add(void) {
    fprintf(stderr, "-- test 1d: residual add\n");
    float max_abs = 0.0f;
    cudaStream_t stream;
    char err[256] = {0};

    const size_t elements = 5120 * 3 + 5;
    float *acc = (float *)malloc(elements * sizeof(float));
    float *add = (float *)malloc(elements * sizeof(float));
    float *ref = (float *)malloc(elements * sizeof(float));
    float *dev = (float *)malloc(elements * sizeof(float));
    float *d_acc = nullptr, *d_add = nullptr;
    if (!acc || !add || !ref || !dev) {
        fprintf(stderr, "FAIL: host allocation\n");
        failures++;
        goto done;
    }
    for (size_t i = 0; i < elements; i++) {
        acc[i] = rng_float();
        add[i] = rng_float();
        ref[i] = acc[i] + add[i];
    }

    cudaMalloc(&d_acc, elements * sizeof(float));
    cudaMalloc(&d_add, elements * sizeof(float));
    cudaMemcpy(d_acc, acc, elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_add, add, elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    if (!q3_cuda_residual_add(d_acc, d_add, elements, stream,
                              err, sizeof(err))) {
        fprintf(stderr, "FAIL: residual_add: %s\n", err);
        failures++;
        cudaStreamDestroy(stream);
        goto done;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(dev, d_acc, elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < elements; i++) {
        const float e = fabsf(dev[i] - ref[i]);
        if (e > max_abs) max_abs = e;
    }
    fprintf(stderr, "   elements=%zu max_abs=%.3g\n", elements, max_abs);
    CHECK(max_abs == 0.0f, "residual add is exact (max_abs == 0)");

done:
    if (d_add) cudaFree(d_add);
    if (d_acc) cudaFree(d_acc);
    free(dev);
    free(ref);
    free(add);
    free(acc);
}

int main(void) {
    fprintf(stderr, "M2 test 1: primitive semantics\n");

    if (q3_cuda_init() != 0) {
        fprintf(stderr, "FAIL: CUDA initialization failed\n");
        return 1;
    }

    test_rms_norm();
    test_rope();
    test_silu_mul();
    test_residual_add();

    if (failures) {
        fprintf(stderr, "M2 test 1: %d FAILURE(S)\n", failures);
        return 1;
    }
    fprintf(stderr, "M2 test 1: all primitive checks passed\n");
    return 0;
}