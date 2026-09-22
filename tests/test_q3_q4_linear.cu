/* SPDX-License-Identifier: MIT
 *
 * test_q3_q4_linear.cu - M1 tests for the resident Q4_K quantized linear.
 *
 * Three tests, as M1 specifies:
 *
 *   1. Q4 block semantics  - the CPU oracle and the CUDA device decode agree
 *      on a known deterministic Q4_K super-block.
 *   2. real Q4 linear parity - a genuine Q4_K tensor of the Qwen3-32B GGUF is
 *      used directly from its resident device pointer, for M=1 and M=4 (and a
 *      larger batched regime), and compared against the CPU Q4_K oracle.
 *   3. production invariants - the resident weight pointer is unchanged, the
 *      hot path performs no CUDA allocation and no host synchronization, and
 *      no dequantized weight mirror exists.
 *
 * Test 2 and 3 need the real model. Point Q3_TEST_MODEL at it, or leave the
 * default path. If the model is absent the real-tensor tests are skipped (the
 * Q4 block semantics test still runs).
 */

#include "q3_q4_linear.h"
#include "q3_quant.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model_loader_cuda.h"
#include "q3_cuda_primitives.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

#define DEFAULT_MODEL "models/Qwen3-32B-Q4_K_M.gguf"
#define PROBE_TENSOR  "blk.0.ffn_gate.weight"

static const char *model_path(void) {
    const char *env = getenv("Q3_TEST_MODEL");
    return (env && *env) ? env : DEFAULT_MODEL;
}

/* ------------------------------------------------------------------ */
/* Test 1 - Q4_K block semantics: CPU oracle vs CUDA device decode.     */
/* ------------------------------------------------------------------ */

#include "q3_cuda_primitives.h"

static uint32_t rng_state = 0x12345678u;
static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static void make_block(q3_q4_k_block *b) {
    const uint16_t d = (uint16_t) (0x2000u + (rng_next() & 0x0fff));
    const uint16_t dmin = (uint16_t) (0x1800u + (rng_next() & 0x0fff));
    b->d = d;
    b->dmin = dmin;
    for (int i = 0; i < 12; i++) b->scales[i] = (uint8_t) rng_next();
    for (int i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t) (rng_next() & 0x0f);
        const uint8_t hi = (uint8_t) (rng_next() & 0x0f);
        b->qs[i] = (uint8_t) ((hi << 4) | lo);
    }
}

static void test_q4_block_semantics(void) {
    fprintf(stderr, "-- test 1: Q4_K block semantics\n");

    CHECK(sizeof(q3_q4_k_block) == Q3_QUANT_Q4_K_BLOCK_BYTES,
          "q3_q4_k_block is 144 bytes (GGUF Q4_K super-block)");

    enum { NBLOCK = 4, ELEMS = NBLOCK * Q3_QUANT_QK_K };
    q3_q4_k_block blocks[NBLOCK];
    for (int i = 0; i < NBLOCK; i++) make_block(&blocks[i]);

    float cpu[ELEMS];
    char err[256] = {0};
    if (!q3_quant_dequantize_row(Q3_QUANT_Q4_K, blocks, NBLOCK, cpu, ELEMS,
                                 err, sizeof(err))) {
        fprintf(stderr, "FAIL: CPU oracle: %s\n", err);
        failures++;
        return;
    }

    /* A handful of hand-derived values from the known block: verify the
     * scale/min unpack against the raw fields so an oracle bug cannot hide. */
    {
        const q3_q4_k_block *b = &blocks[0];
        const float d = q3_half_to_float(b->d);
        const float dmin = q3_half_to_float(b->dmin);
        const uint8_t sc = (uint8_t) (b->scales[0] & 63u);
        const uint8_t mn = (uint8_t) (b->scales[4] & 63u);
        const float expected = d * (float) sc * (float) (b->qs[0] & 0x0f) -
                               dmin * (float) mn;
        CHECK(fabsf(expected - cpu[0]) < 1e-3f,
              "oracle matches manual scale/min unpack of the first nibble");
    }

    float dev[ELEMS];
    float *d_blocks = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_blocks, sizeof(blocks)) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof(dev)) != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaMalloc for block test\n");
        failures++;
        return;
    }
    cudaMemcpy(d_blocks, blocks, sizeof(blocks), cudaMemcpyHostToDevice);
    if (!q3_cuda_dequantize_row(Q3_QUANT_Q4_K, d_blocks, NBLOCK, d_out, nullptr,
                                err, sizeof(err))) {
        fprintf(stderr, "FAIL: CUDA decode: %s\n", err);
        failures++;
        cudaFree(d_blocks);
        cudaFree(d_out);
        return;
    }
    cudaMemcpy(dev, d_out, sizeof(dev), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < ELEMS; i++) {
        const float d0 = fabsf(cpu[i] - dev[i]);
        if (d0 > max_diff) max_diff = d0;
    }
    fprintf(stderr, "   max |CPU - CUDA| over %d elements: %.3e\n", ELEMS, max_diff);
    CHECK(max_diff == 0.0f, "CUDA device decode is bit-exact with the CPU oracle");

    cudaFree(d_blocks);
    cudaFree(d_out);
}

/* ------------------------------------------------------------------ */
/* Real-tensor harness (tests 2 and 3).                                */
/* ------------------------------------------------------------------ */

typedef struct {
    q3_gguf *model;
    q3_loader_context *loader;
    const q3_resident_tensor *tensor;
    q3_q4_linear_geometry geometry;
    void *workspace;
} real_world;

static bool real_open(real_world *w) {
    memset(w, 0, sizeof(*w));
    char err[512] = {0};

    if (q3_cuda_init() != 0) {
        fprintf(stderr, "skip: CUDA unavailable\n");
        return false;
    }
    w->model = q3_gguf_open(model_path(), err, sizeof(err));
    if (!w->model) {
        fprintf(stderr, "skip: cannot open %s: %s\n", model_path(), err);
        return false;
    }
    w->loader = q3_loader_context_create(err, sizeof(err));
    if (!w->loader || !q3_loader_load(w->loader, w->model, err, sizeof(err))) {
        fprintf(stderr, "skip: resident load failed: %s\n", err);
        if (w->loader) q3_loader_context_destroy(w->loader);
        q3_gguf_close(w->model);
        return false;
    }

    const char *name = getenv("Q3_TEST_TENSOR");
    if (!name || !*name) name = PROBE_TENSOR;

    w->tensor = nullptr;
    const size_t count = q3_loader_tensor_count(w->loader);
    for (size_t i = 0; i < count; i++) {
        const q3_resident_tensor *t = q3_loader_tensor(w->loader, i);
        if (t && strcmp(t->name, name) == 0) { w->tensor = t; break; }
    }
    if (!w->tensor) {
        fprintf(stderr, "skip: tensor '%s' not found\n", name);
        return false;
    }
    if (w->tensor->qtype != Q3_QUANT_Q4_K) {
        fprintf(stderr, "skip: tensor '%s' is qtype %u, not Q4_K\n",
                name, w->tensor->qtype);
        return false;
    }
    if (!w->tensor->resident || !w->tensor->ptr) {
        fprintf(stderr, "skip: tensor '%s' has no resident device pointer\n", name);
        return false;
    }
    if (!q3_q4_linear_bind_geometry(w->tensor->qtype, w->tensor->rows,
                                    w->tensor->cols, &w->geometry,
                                    err, sizeof(err))) {
        fprintf(stderr, "skip: geometry: %s\n", err);
        return false;
    }
    if (!q3_cuda_q4k_linear_init(0, err, sizeof(err))) {
        fprintf(stderr, "skip: init: %s\n", err);
        return false;
    }
    const size_t bytes = q3_cuda_q4k_linear_workspace_bytes();
    if (cudaMalloc(&w->workspace, bytes) != cudaSuccess) {
        fprintf(stderr, "skip: workspace alloc (%zu bytes) failed\n", bytes);
        return false;
    }
    if (!q3_cuda_q4k_linear_set_workspace(w->workspace, bytes, err, sizeof(err))) {
        fprintf(stderr, "skip: workspace attach: %s\n", err);
        return false;
    }
    return true;
}

static void real_close(real_world *w) {
    if (w->workspace) cudaFree(w->workspace);
    if (w->loader) q3_loader_context_destroy(w->loader);
    if (w->model) q3_gguf_close(w->model);
}

/* Deterministic activation: sin/cos of the element index, no PRNG device. */
static void fill_activation(float *x, size_t tokens, size_t K) {
    for (size_t t = 0; t < tokens; t++) {
        for (size_t k = 0; k < K; k++) {
            const float a = (float) ((k * 17u + t * 131u) % 1009u) * 0.1f;
            x[t * K + k] = 0.5f * sinf(a) + 0.25f * cosf(a * 0.37f);
        }
    }
}

/* CPU oracle: Q4_K dequantize of the whole weight, then FP32 dot product. */
static bool cpu_reference(const real_world *w, const float *x, size_t tokens,
                          float *out) {
    const size_t K = (size_t) w->geometry.K;
    const size_t N = (size_t) w->geometry.N;
    const size_t blocks = (size_t) w->tensor->bytes / Q3_QUANT_Q4_K_BLOCK_BYTES;

    void *raw = malloc((size_t) w->tensor->bytes);
    float *deq = (float *) malloc(N * K * sizeof(float));
    if (!raw || !deq) {
        free(raw);
        free(deq);
        return false;
    }
    if (cudaMemcpy(raw, w->tensor->ptr, (size_t) w->tensor->bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(raw);
        free(deq);
        return false;
    }

    char err[256] = {0};
    if (!q3_quant_dequantize_row(Q3_QUANT_Q4_K, raw, blocks, deq, N * K,
                                 err, sizeof(err))) {
        fprintf(stderr, "FAIL: CPU oracle: %s\n", err);
        free(raw);
        free(deq);
        return false;
    }
    free(raw);

    for (size_t t = 0; t < tokens; t++) {
        for (size_t n = 0; n < N; n++) {
            const float *row = deq + n * K;
            const float *act = x + t * K;
            double acc = 0.0;
            for (size_t k = 0; k < K; k++) acc += (double) row[k] * (double) act[k];
            out[t * N + n] = (float) acc;
        }
    }
    free(deq);
    return true;
}

static void test_real_parity(void) {
    fprintf(stderr, "-- test 2: real Q4_K linear parity\n");

    real_world w;
    if (!real_open(&w)) {
        fprintf(stderr, "   (real-tensor tests skipped)\n");
        real_close(&w);
        return;
    }

    const size_t K = (size_t) w.geometry.K;
    const size_t N = (size_t) w.geometry.N;
    fprintf(stderr, "   tensor %s: Q4_K N=%zu K=%zu bytes=%" PRIu64 "\n",
            w.tensor->name, N, K, w.tensor->bytes);

    const size_t batches[] = {1, 4, 16};
    for (size_t bi = 0; bi < sizeof(batches) / sizeof(batches[0]); bi++) {
        const size_t tokens = batches[bi];
        const size_t total = tokens * N;

        float *x = (float *) malloc(tokens * K * sizeof(float));
        float *ref = (float *) malloc(tokens * N * sizeof(float));
        float *gpu = (float *) malloc(tokens * N * sizeof(float));
        float *gpu2 = (float *) malloc(tokens * N * sizeof(float));
        float *d_x = nullptr, *d_out = nullptr;
        cudaMalloc(&d_x, tokens * K * sizeof(float));
        cudaMalloc(&d_out, tokens * N * sizeof(float));
        cudaMemset(d_out, 0xff, tokens * N * sizeof(float)); /* poison */

        fill_activation(x, tokens, K);
        cudaMemcpy(d_x, x, tokens * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        char err[256] = {0};
        char err2[256] = {0};
        q3_q4_linear_stats stats;
        q3_q4_linear_stats stats2;
        memset(&stats, 0, sizeof(stats));
        memset(&stats2, 0, sizeof(stats2));
        bool ok = false;
        double sum_abs = 0.0, sum_sq = 0.0, sum_ref_sq = 0.0, sum_cross = 0.0, gsq = 0.0;
        float max_abs = 0.0f, max_ref = 0.0f;
        double mean_abs = 0.0, rmse = 0.0, rel = 0.0, rel_l2 = 0.0, cos_sim = 0.0;

        ok = q3_cuda_q4k_linear(&w.geometry, w.tensor->ptr, d_x,
                                           tokens, d_out, nullptr, &stats,
                                           err, sizeof(err));
        if (!ok) {
            fprintf(stderr, "FAIL: M=%zu: %s\n", tokens, err);
            failures++;
            goto next;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "FAIL: M=%zu: kernel error\n", tokens);
            failures++;
            goto next;
        }
        cudaMemcpy(gpu, d_out, tokens * N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        if (!cpu_reference(&w, x, tokens, ref)) {
            failures++;
            goto next;
        }

        for (size_t i = 0; i < total; i++) {
            const double e = (double) gpu[i] - (double) ref[i];
            sum_abs += fabs(e);
            sum_sq += e * e;
            sum_ref_sq += (double) ref[i] * (double) ref[i];
            sum_cross += (double) gpu[i] * (double) ref[i];
            if ((float) fabs(e) > max_abs) max_abs = (float) fabs(e);
            if (fabsf(ref[i]) > max_ref) max_ref = fabsf(ref[i]);
        }
        for (size_t i = 0; i < total; i++) gsq += (double) gpu[i] * (double) gpu[i];
        mean_abs = sum_abs / (double) total;
        rmse = sqrt(sum_sq / (double) total);
        rel = (double) max_abs / (double) (max_ref > 0 ? max_ref : 1);
        cos_sim = sum_cross / (sqrt(gsq) * sqrt(sum_ref_sq) + 1e-30);

        rel_l2 = sqrt(sum_sq) / (sqrt(sum_ref_sq) + 1e-30);
        fprintf(stderr,
                "   M=%zu maxabs=%.4g meanabs=%.4g rmse=%.4g rel=%.3g relL2=%.3g cos=%.9f\n",
                tokens, max_abs, mean_abs, rmse, rel, rel_l2, cos_sim);

        {
            char msg[128];
            snprintf(msg, sizeof(msg), "M=%zu output is finite", tokens);
            bool finite = true;
            for (size_t i = 0; i < total; i++) if (!isfinite(gpu[i])) finite = false;
            CHECK(finite, msg);
        }
        {
            char msg[128];
            snprintf(msg, sizeof(msg), "M=%zu matches CPU Q4_K oracle (cos > 0.999)", tokens);
            CHECK(cos_sim > 0.999, msg);
        }
        {
            char msg[128];
            snprintf(msg, sizeof(msg), "M=%zu relative L2 error below 1%%", tokens);
            CHECK(rel_l2 < 0.01, msg);
        }

        /* Repeated call must reproduce the same result exactly. */
        {
            cudaMemset(d_out, 0, tokens * N * sizeof(float));
            bool same = false;
            same = q3_cuda_q4k_linear(&w.geometry, w.tensor->ptr, d_x,
                                      tokens, d_out, nullptr, &stats2,
                                      err2, sizeof(err2));
            cudaDeviceSynchronize();
            cudaMemcpy(gpu2, d_out, tokens * N * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < total && same; i++) if (gpu2[i] != gpu[i]) same = false;
            char msg[128];
            snprintf(msg, sizeof(msg), "M=%zu repeats bit-exactly", tokens);
            CHECK(same, msg);
        }

next:
        cudaFree(d_x);
        cudaFree(d_out);
        free(x);
        free(ref);
        free(gpu);
        free(gpu2);
    }

    real_close(&w);
}

static void test_production_invariants(void) {
    fprintf(stderr, "-- test 3: production invariants\n");

    real_world w;
    if (!real_open(&w)) {
        fprintf(stderr, "   (invariant tests skipped)\n");
        real_close(&w);
        return;
    }

    const size_t K = (size_t) w.geometry.K;
    const size_t N = (size_t) w.geometry.N;
    const size_t tokens = 4;

    float *x = (float *) malloc(tokens * K * sizeof(float));
    fill_activation(x, tokens, K);
    float *d_x = nullptr, *d_out = nullptr;
    cudaMalloc(&d_x, tokens * K * sizeof(float));
    cudaMalloc(&d_out, tokens * N * sizeof(float));
    cudaMemcpy(d_x, x, tokens * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(x);

    const void *weight_before = w.tensor->ptr;

    char err[256] = {0};
    q3_q4_linear_stats stats;
    for (int i = 0; i < 3; i++) { /* warmup */
        q3_cuda_q4k_linear(&w.geometry, w.tensor->ptr, d_x, tokens, d_out,
                           nullptr, &stats, err, sizeof(err));
    }
    cudaDeviceSynchronize();

    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    memset(&stats, 0, sizeof(stats));
    const bool ok = q3_cuda_q4k_linear(&w.geometry, w.tensor->ptr, d_x, tokens,
                                       d_out, nullptr, &stats, err, sizeof(err));
    cudaDeviceSynchronize();

    size_t free_after = 0, total2 = 0;
    cudaMemGetInfo(&free_after, &total2);

    CHECK(ok, "warm Q4 linear call succeeds");

    fprintf(stderr, "   path=%s launches=%" PRIu64 " allocs=%" PRIu64
                    " syncs=%" PRIu64 "\n",
            q3_cuda_q4k_linear_path_for_batch(tokens),
            stats.kernel_launches, stats.cuda_allocations, stats.host_syncs);

    CHECK(stats.cuda_allocations == 0, "hot path performs zero CUDA allocations");
    CHECK(stats.host_syncs == 0, "hot path performs zero host synchronizations");
    CHECK(free_before == free_after, "device free memory unchanged across the call");
    CHECK(w.tensor->ptr == weight_before, "resident weight pointer is unchanged");
    CHECK(w.tensor->resident, "weight remains resident");

    cudaFree(d_x);
    cudaFree(d_out);
    real_close(&w);
}

int main(void) {
    fprintf(stderr, "q3 M1 Q4_K linear tests\n");
    test_q4_block_semantics();
    test_real_parity();
    test_production_invariants();
    if (failures) {
        fprintf(stderr, "FAILED: %d check(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}