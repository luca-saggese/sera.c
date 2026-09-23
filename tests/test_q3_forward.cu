/* SPDX-License-Identifier: MIT
 *
 * test_q3_forward.cu - M2 forward tests (docs/M2.md §30).
 *
 * Three tests, all using the M2 runtime itself as the reference (no Python
 * oracle, no full model load beyond the resident weights):
 *
 *   2. layer 0 parity - every boundary buffer of layer 0 is finite, and a
 *      repeated run reproduces the same result bit-exactly (determinism).
 *   3. full 64-layer parity - the complete one-pass forward is finite, the
 *      final-norm hidden and candidate logits are finite, the argmax is
 *      stable, and the warm path performs zero CUDA allocations and zero
 *      host synchronizations.
 *   4. KV equivalence (B1) - the full prompt forward and the prefix prefill +
 *      suffix-through-KV-cache forward produce the same argmax and logits
 *      within tolerance.
 *
 * The real model is required. Point Q3_TEST_MODEL at it, or leave the default
 * path. If the model is absent the tests are skipped.
 */

#include "q3_forward.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model.h"
#include "q3_binder.h"
#include "q3_model_loader_cuda.h"

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

static const char *model_path(void) {
    const char *env = getenv("Q3_TEST_MODEL");
    return (env && *env) ? env : DEFAULT_MODEL;
}

/* ------------------------------------------------------------------ */
/* Harness: open GGUF, resolve config, bind weights, create runtime.   */
/* ------------------------------------------------------------------ */

typedef struct {
    q3_gguf *model;
    q3_loader_context *loader;
    q3_model_config config;
    q3_weights weights;
    q3_forward_runtime *runtime;
    uint32_t hidden;
    uint32_t inter;
    uint32_t layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
} fwd_world;

static bool fwd_open(fwd_world *w) {
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
    if (!q3_model_config_from_gguf(w->model, &w->config, err, sizeof(err))) {
        fprintf(stderr, "skip: config: %s\n", err);
        return false;
    }
    if (!q3_model_config_validate(w->model, &w->config, err, sizeof(err))) {
        fprintf(stderr, "skip: config validate: %s\n", err);
        return false;
    }
    w->loader = q3_loader_context_create(err, sizeof(err));
    if (!w->loader || !q3_loader_load(w->loader, w->model, err, sizeof(err))) {
        fprintf(stderr, "skip: resident load failed: %s\n", err);
        if (w->loader) q3_loader_context_destroy(w->loader);
        q3_gguf_close(w->model);
        return false;
    }
    if (!q3_weights_bind(w->model, w->loader, &w->config, &w->weights,
                         err, sizeof(err))) {
        fprintf(stderr, "skip: bind: %s\n", err);
        return false;
    }
    w->runtime = q3_forward_create(&w->weights, &w->config, 64, 0,
                                   err, sizeof(err));
    if (!w->runtime) {
        fprintf(stderr, "skip: forward create: %s\n", err);
        return false;
    }
    w->hidden = w->config.hidden_size;
    w->inter = w->config.intermediate_size;
    w->layers = w->config.num_layers;
    w->n_heads = w->config.num_attention_heads;
    w->n_kv_heads = w->config.num_kv_heads;
    w->head_dim = w->config.head_dim;
    w->vocab = w->config.vocab_size;
    return true;
}

static void fwd_close(fwd_world *w) {
    if (w->runtime) q3_forward_destroy(w->runtime);
    q3_weights_free(&w->weights);
    if (w->loader) q3_loader_context_destroy(w->loader);
    if (w->model) q3_gguf_close(w->model);
}

/* Deterministic token sequence (no file dependency). */
static void fill_tokens(uint32_t *t, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) t[i] = (uint32_t)((i * 2654435761u) % 1000u);
}

/* Copy a device buffer to host and report NaN/Inf/max_abs/first_bad. */
static bool buffer_stats(const float *dev, size_t elements, float *max_abs,
                         size_t *first_bad) {
    float *h = (float *) malloc(elements * sizeof(float));
    if (!h) return false;
    if (cudaMemcpy(h, dev, elements * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(h);
        return false;
    }
    *max_abs = 0.0f;
    *first_bad = elements;
    for (size_t i = 0; i < elements; i++) {
        const float v = h[i];
        if (!isfinite(v)) {
            if (*first_bad == elements) *first_bad = i;
        } else if (fabsf(v) > *max_abs) {
            *max_abs = fabsf(v);
        }
    }
    free(h);
    return true;
}

/* ------------------------------------------------------------------ */
/* Test 2 - layer 0 parity: every boundary finite + deterministic.     */
/* ------------------------------------------------------------------ */

static void test_layer0_parity(void) {
    fprintf(stderr, "-- test 2: layer 0 boundary parity\n");

    fwd_world w;
    if (!fwd_open(&w)) {
        fprintf(stderr, "   (forward tests skipped)\n");
        fwd_close(&w);
        return;
    }

    const uint32_t T = 4;
    uint32_t tokens[T];
    fill_tokens(tokens, T);

    char err[512] = {0};
    q3_forward_stats stats;
    memset(&stats, 0, sizeof(stats));

    /* Run layer 0 twice; the second run must reproduce the first exactly. */
    float *hidden0 = (float *) malloc((size_t)T * w.hidden * sizeof(float));
    float *hidden1 = (float *) malloc((size_t)T * w.hidden * sizeof(float));
    if (!hidden0 || !hidden1) {
        fprintf(stderr, "FAIL: host alloc\n");
        failures++;
        goto out;
    }

    if (!q3_forward_run_range(w.runtime, tokens, T, NULL, 0, 0, 1,
                              &stats, err, sizeof(err))) {
        fprintf(stderr, "FAIL: run_range layer 0: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_forward_synchronize(w.runtime, err, sizeof(err))) {
        fprintf(stderr, "FAIL: synchronize: %s\n", err);
        failures++;
        goto out;
    }
    cudaMemcpy(hidden0, q3_forward_buffer(w.runtime, Q3_FB_HIDDEN, NULL),
               (size_t)T * w.hidden * sizeof(float), cudaMemcpyDeviceToHost);

    if (!q3_forward_run_range(w.runtime, tokens, T, NULL, 0, 0, 1,
                              &stats, err, sizeof(err))) {
        fprintf(stderr, "FAIL: run_range layer 0 (2nd): %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_forward_synchronize(w.runtime, err, sizeof(err))) {
        fprintf(stderr, "FAIL: synchronize (2nd): %s\n", err);
        failures++;
        goto out;
    }
    cudaMemcpy(hidden1, q3_forward_buffer(w.runtime, Q3_FB_HIDDEN, NULL),
               (size_t)T * w.hidden * sizeof(float), cudaMemcpyDeviceToHost);

    {
        bool same = true;
        for (size_t i = 0; i < (size_t)T * w.hidden; i++)
            if (hidden0[i] != hidden1[i]) { same = false; break; }
        CHECK(same, "layer 0 repeats bit-exactly");
    }

    /* Every boundary buffer of the most recent run must be finite. */
    {
        const int which[] = {
            Q3_FB_HIDDEN, Q3_FB_NORMED, Q3_FB_Q, Q3_FB_Q2, Q3_FB_K,
            Q3_FB_K2, Q3_FB_V, Q3_FB_ATTN, Q3_FB_OPROJ, Q3_FB_GATE,
            Q3_FB_UP, Q3_FB_DOWN
        };
        const char *names[] = {
            "hidden", "normed", "q", "q2", "k", "k2", "v", "attn",
            "oproj", "gate", "up", "down"
        };
        for (int i = 0; i < 12; i++) {
            uint64_t bytes = 0;
            const float *dev = q3_forward_buffer(w.runtime, which[i], &bytes);
            if (!dev || bytes == 0) {
                fprintf(stderr, "FAIL: buffer %s unavailable\n", names[i]);
                failures++;
                continue;
            }
            float max_abs = 0.0f;
            size_t first_bad = 0;
            if (!buffer_stats(dev, bytes / sizeof(float), &max_abs, &first_bad)) {
                fprintf(stderr, "FAIL: buffer %s copy\n", names[i]);
                failures++;
                continue;
            }
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "layer0 %s finite (max_abs=%.4g%s)", names[i], max_abs,
                     first_bad < bytes / sizeof(float) ? ", BAD" : "");
            CHECK(first_bad == bytes / sizeof(float), msg);
        }
    }

out:
    free(hidden0);
    free(hidden1);
    fwd_close(&w);
}

/* ------------------------------------------------------------------ */
/* Test 3 - full 64-layer parity: finite, deterministic, warm path.    */
/* ------------------------------------------------------------------ */

static void test_full_forward(void) {
    fprintf(stderr, "-- test 3: full forward parity\n");

    fwd_world w;
    if (!fwd_open(&w)) {
        fprintf(stderr, "   (forward tests skipped)\n");
        fwd_close(&w);
        return;
    }

    const uint32_t T = 8;
    uint32_t tokens[T];
    fill_tokens(tokens, T);

    const uint32_t candidates[] = {16, 17, 18, 19, 33, 969};
    const uint32_t n_cand = (uint32_t)(sizeof(candidates) / sizeof(candidates[0]));

    char err[512] = {0};
    q3_forward_stats stats;
    memset(&stats, 0, sizeof(stats));

    float *d_hidden = NULL;
    float *d_logits = NULL;
    if (cudaMalloc(&d_hidden, w.hidden * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits, n_cand * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaMalloc\n");
        failures++;
        goto out;
    }

    if (!q3_forward_run(w.runtime, tokens, T, NULL, 0, d_hidden, &stats,
                        err, sizeof(err))) {
        fprintf(stderr, "FAIL: forward run: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_candidate_logits(w.runtime, d_hidden, candidates, n_cand,
                             d_logits, err, sizeof(err))) {
        fprintf(stderr, "FAIL: candidate logits: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_forward_synchronize(w.runtime, err, sizeof(err))) {
        fprintf(stderr, "FAIL: synchronize: %s\n", err);
        failures++;
        goto out;
    }

    {
        float max_abs = 0.0f;
        size_t first_bad = 0;
        if (!buffer_stats(d_hidden, w.hidden, &max_abs, &first_bad)) {
            fprintf(stderr, "FAIL: hidden copy\n");
            failures++;
        } else {
            CHECK(first_bad == w.hidden, "final-norm hidden is finite");
        }
    }
    {
        float logits[8];
        cudaMemcpy(logits, d_logits, n_cand * sizeof(float),
                   cudaMemcpyDeviceToHost);
        bool finite = true;
        for (uint32_t i = 0; i < n_cand; i++) if (!isfinite(logits[i])) finite = false;
        CHECK(finite, "candidate logits are finite");
        uint32_t argmax = 0;
        for (uint32_t i = 1; i < n_cand; i++) if (logits[i] > logits[argmax]) argmax = i;
        fprintf(stderr, "   argmax=%u logits=[", argmax);
        for (uint32_t i = 0; i < n_cand; i++)
            fprintf(stderr, "%s%.4f", i ? " " : "", logits[i]);
        fprintf(stderr, "]\n");
    }

    /* Determinism: a second run must reproduce the same logits exactly. */
    {
        float *d_hidden2 = NULL, *d_logits2 = NULL;
        cudaMalloc(&d_hidden2, w.hidden * sizeof(float));
        cudaMalloc(&d_logits2, n_cand * sizeof(float));
        q3_forward_stats stats2;
        memset(&stats2, 0, sizeof(stats2));
        if (!q3_forward_run(w.runtime, tokens, T, NULL, 0, d_hidden2,
                            &stats2, err, sizeof(err)) ||
            !q3_candidate_logits(w.runtime, d_hidden2, candidates, n_cand,
                                 d_logits2, err, sizeof(err))) {
            fprintf(stderr, "FAIL: forward run (2nd): %s\n", err);
            failures++;
        } else {
            q3_forward_synchronize(w.runtime, err, sizeof(err));
            float l1[8], l2[8];
            cudaMemcpy(l1, d_logits, n_cand * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(l2, d_logits2, n_cand * sizeof(float), cudaMemcpyDeviceToHost);
            bool same = true;
            for (uint32_t i = 0; i < n_cand; i++) if (l1[i] != l2[i]) { same = false; break; }
            CHECK(same, "full forward repeats bit-exactly");
        }
        if (d_hidden2) cudaFree(d_hidden2);
        if (d_logits2) cudaFree(d_logits2);
    }

    /* Warm-path invariants: the measured run must not allocate or sync. */
    fprintf(stderr, "   layers=%" PRIu64 " allocs=%" PRIu64 " syncs=%" PRIu64
                    " total=%.3fms\n",
            stats.layers_executed, stats.cuda_allocations, stats.host_syncs,
            stats.total_forward_ms);
    CHECK(stats.layers_executed == w.layers, "all layers executed");
    CHECK(stats.cuda_allocations == 0, "warm forward performs zero CUDA allocations");
    CHECK(stats.host_syncs == 0, "warm forward performs zero host synchronizations");

out:
    if (d_hidden) cudaFree(d_hidden);
    if (d_logits) cudaFree(d_logits);
    fwd_close(&w);
}

/* ------------------------------------------------------------------ */
/* Test 4 - KV equivalence (B1): full prompt vs prefix + KV suffix.    */
/* ------------------------------------------------------------------ */

/* Isolated KV attention check: layer 0 only, comparing the raw attention
 * output (before o_proj) of a one-shot T=8 pass against a P+suffix pass that
 * reads the segmented KV cache. This is the M2 KV gate: it isolates the
 * segmented-attention mechanics from the 64-layer forward and from the
 * candidate LM head. */
static void test_kv_attention_isolated(void) {
    fprintf(stderr, "-- test 4a: KV attention isolated (layer 0, attn buffer)\n");

    fwd_world w;
    if (!fwd_open(&w)) {
        fprintf(stderr, "   (forward tests skipped)\n");
        fwd_close(&w);
        return;
    }

    const uint32_t T = 8;
    const uint32_t P = 4;
    const uint32_t attn_elems = w.n_heads * w.head_dim;
    uint32_t tokens[T];
    fill_tokens(tokens, T);

    char err[512] = {0};
    q3_forward_stats stats;
    memset(&stats, 0, sizeof(stats));

    float *ref = (float *)malloc((size_t)attn_elems * sizeof(float));
    float *got = (float *)malloc((size_t)attn_elems * sizeof(float));
    q3_kv_cache *kv = NULL;
    if (!ref || !got) {
        fprintf(stderr, "FAIL: malloc\n");
        failures++;
        goto out;
    }

    /* Reference: layer 0, T=8, no cache. */
    if (!q3_forward_run_range(w.runtime, tokens, T, NULL, 0, 0, 1, &stats,
                              err, sizeof(err))) {
        fprintf(stderr, "FAIL: layer 0 full: %s\n", err);
        failures++;
        goto out;
    }
    q3_forward_synchronize(w.runtime, err, sizeof(err));
    {
        uint64_t bytes = 0;
        const float *d = q3_forward_buffer(w.runtime, Q3_FB_ATTN, &bytes);
        if (!d || bytes < (uint64_t)attn_elems * sizeof(float)) {
            fprintf(stderr, "FAIL: attn buffer\n");
            failures++;
            goto out;
        }
        /* Last token row of the T=8 attention output. */
        cudaMemcpy(ref, d + (size_t)(T - 1) * attn_elems,
                   (size_t)attn_elems * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /* Segmented: prefix P into the cache, then suffix reading the cache. */
    kv = q3_kv_cache_init(w.layers, w.n_kv_heads, w.head_dim, T, err, sizeof(err));
    if (!kv) {
        fprintf(stderr, "FAIL: kv init: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_forward_run_range(w.runtime, tokens, P, kv, 0, 0, 1, &stats,
                              err, sizeof(err))) {
        fprintf(stderr, "FAIL: layer 0 prefix: %s\n", err);
        failures++;
        goto out;
    }
    q3_forward_synchronize(w.runtime, err, sizeof(err));
    if (!q3_forward_run_range(w.runtime, tokens + P, T - P, kv, P, 0, 1, &stats,
                              err, sizeof(err))) {
        fprintf(stderr, "FAIL: layer 0 suffix: %s\n", err);
        failures++;
        goto out;
    }
    q3_forward_synchronize(w.runtime, err, sizeof(err));
    {
        uint64_t bytes = 0;
        const float *d = q3_forward_buffer(w.runtime, Q3_FB_ATTN, &bytes);
        if (!d || bytes < (uint64_t)attn_elems * sizeof(float)) {
            fprintf(stderr, "FAIL: attn buffer\n");
            failures++;
            goto out;
        }
        /* Last token row of the T-P suffix attention output. */
        cudaMemcpy(got, d + (size_t)(T - P - 1) * attn_elems,
                   (size_t)attn_elems * sizeof(float), cudaMemcpyDeviceToHost);
    }

    {
        float max_diff = 0.0f, max_abs = 0.0f;
        for (uint32_t i = 0; i < attn_elems; i++) {
            const float d = fabsf(ref[i] - got[i]);
            if (d > max_diff) max_diff = d;
            if (fabsf(ref[i]) > max_abs) max_abs = fabsf(ref[i]);
        }
        fprintf(stderr, "   attn(last token) max|diff|=%.4g (|ref|max=%.4g)\n",
                max_diff, max_abs);
        CHECK(max_diff < 1e-5f,
              "segmented KV attention matches one-shot attention");
    }

out:
    if (kv) q3_kv_cache_destroy(kv);
    free(ref);
    free(got);
    fwd_close(&w);
}

static void test_kv_equivalence(void) {
    fprintf(stderr, "-- test 4: KV prefix+suffix equivalence\n");

    fwd_world w;
    if (!fwd_open(&w)) {
        fprintf(stderr, "   (forward tests skipped)\n");
        fwd_close(&w);
        return;
    }

    const uint32_t T = 8;
    const uint32_t P = 4; /* prefix length */
    uint32_t tokens[T];
    fill_tokens(tokens, T);

    const uint32_t candidates[] = {16, 17, 18, 19, 33, 969};
    const uint32_t n_cand = (uint32_t)(sizeof(candidates) / sizeof(candidates[0]));

    char err[512] = {0};
    q3_forward_stats stats;
    memset(&stats, 0, sizeof(stats));

    float *d_hidden_full = NULL, *d_hidden_suf = NULL;
    float *d_logits_full = NULL, *d_logits_suf = NULL;
    q3_kv_cache *kv = NULL;
    if (cudaMalloc(&d_hidden_full, w.hidden * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_hidden_suf, w.hidden * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits_full, n_cand * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits_suf, n_cand * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaMalloc\n");
        failures++;
        goto out;
    }
    kv = q3_kv_cache_init(w.layers, w.n_kv_heads, w.head_dim, T,
                          err, sizeof(err));
    if (!kv) {
        fprintf(stderr, "FAIL: kv cache init: %s\n", err);
        failures++;
        goto out;
    }

    /* Full prompt in one pass. */
    if (!q3_forward_run(w.runtime, tokens, T, NULL, 0, d_hidden_full,
                        &stats, err, sizeof(err)) ||
        !q3_candidate_logits(w.runtime, d_hidden_full, candidates, n_cand,
                             d_logits_full, err, sizeof(err))) {
        fprintf(stderr, "FAIL: full forward: %s\n", err);
        failures++;
        goto out;
    }
    q3_forward_synchronize(w.runtime, err, sizeof(err));

    /* Prefix prefill into the cache, then the suffix through the cache.
     * q3_forward_run() always applies the final norm, so the prefix pass needs
     * a scratch output buffer; it is overwritten by the suffix pass. */
    if (!q3_forward_run(w.runtime, tokens, P, kv, 0, d_hidden_suf, &stats,
                        err, sizeof(err))) {
        fprintf(stderr, "FAIL: prefix prefill: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_forward_run(w.runtime, tokens + P, T - P, kv, P, d_hidden_suf,
                        &stats, err, sizeof(err))) {
        fprintf(stderr, "FAIL: suffix forward: %s\n", err);
        failures++;
        goto out;
    }
    if (!q3_candidate_logits(w.runtime, d_hidden_suf, candidates, n_cand,
                             d_logits_suf, err, sizeof(err))) {
        fprintf(stderr, "FAIL: suffix candidate logits: %s\n", err);
        failures++;
        goto out;
    }
    q3_forward_synchronize(w.runtime, err, sizeof(err));

    {
        float lf[8], ls[8];
        cudaMemcpy(lf, d_logits_full, n_cand * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ls, d_logits_suf, n_cand * sizeof(float), cudaMemcpyDeviceToHost);
        uint32_t af = 0, as = 0;
        for (uint32_t i = 1; i < n_cand; i++) {
            if (lf[i] > lf[af]) af = i;
            if (ls[i] > ls[as]) as = i;
        }
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            const float d = fabsf(lf[i] - ls[i]);
            if (d > max_diff) max_diff = d;
        }
        fprintf(stderr, "   full argmax=%u suffix argmax=%u max|diff|=%.4g\n",
                af, as, max_diff);
        CHECK(af == as, "KV suffix argmax matches full-prompt argmax");
        /* The KV mechanics themselves are validated to ~1e-7 by test 4a. What
         * remains here is FP accumulation-order drift: MMVQ picks a different
         * internal configuration for M=4 than for M=8 (calc_nwarps /
         * calc_rows_per_block), so per-row products differ at ~1e-6 and that
         * is amplified across 64 layers. It is not a KV bug, so the gate is
         * "same discrete candidate + logits close", not bitwise equality. */
        CHECK(max_diff < 0.25f,
              "KV suffix logits close to full prompt (M-shape drift bound)");
        CHECK(q3_kv_cache_length(kv) == T, "KV cache length is the full sequence");
    }

out:
    if (kv) q3_kv_cache_destroy(kv);
    if (d_hidden_full) cudaFree(d_hidden_full);
    if (d_hidden_suf) cudaFree(d_hidden_suf);
    if (d_logits_full) cudaFree(d_logits_full);
    if (d_logits_suf) cudaFree(d_logits_suf);
    fwd_close(&w);
}

int main(void) {
    fprintf(stderr, "q3 M2 forward tests\n");
    test_layer0_parity();
    test_full_forward();
    test_kv_attention_isolated();
    test_kv_equivalence();
    if (failures) {
        fprintf(stderr, "FAILED: %d check(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}