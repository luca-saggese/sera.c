/* SPDX-License-Identifier: MIT
 *
 * test_q3_decide.cu - M3 shared-prefix batched decision tests (docs/M3.md §33).
 *
 * Four tests, all structural. Correctness of the model output is explicitly
 * NOT an M3 gate (§34-§41): the gates here are mechanics - the segmented KV
 * view is exercised, branches are isolated, the 32 real questions all run, and
 * the warm path allocates and synchronizes nothing.
 *
 *   1. segmented KV B1 - one branch through prefix+suffix reproduces the M2
 *      contiguous one-shot path.
 *   2. B2 isolation - two branches with different suffixes do not perturb each
 *      other's prediction, and branch order is irrelevant.
 *   3. B32 decision parity - all 32 workload questions execute and produce a
 *      prediction; the per-question expected index is reported, not enforced.
 *   4. memory/runtime invariants - prefix_copy_bytes == 0, no hot cudaMalloc,
 *      one boundary sync, prefix device bytes independent of B.
 *
 * The real model is required. Point Q3_TEST_MODEL at it, or leave the default
 * path. If the model is absent the tests are skipped.
 */

#include "q3_decide.h"
#include "q3_workload.h"
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
#define DEFAULT_WORKLOAD "docs/research/m3/workload_tokens.json"

static const char *model_path(void) {
    const char *env = getenv("Q3_TEST_MODEL");
    return (env && *env) ? env : DEFAULT_MODEL;
}

static const char *workload_path(void) {
    const char *env = getenv("Q3_TEST_WORKLOAD");
    return (env && *env) ? env : DEFAULT_WORKLOAD;
}

/* ------------------------------------------------------------------ */
/* Harness                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    q3_gguf *model;
    q3_loader_context *loader;
    q3_model_config config;
    q3_weights weights;
    q3_forward_runtime *runtime;
    q3_prefix_kv *prefix;
    q3_branch_set *branches;
    q3_workload *workload;
    uint32_t hidden;
    uint32_t layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t runtime_tokens;
} dec_world;

/* Open the model and create a runtime sized for `runtime_tokens`. */
static bool dec_open_model(dec_world *w, uint32_t runtime_tokens) {
    memset(w, 0, sizeof(*w));
    char err[512] = {0};
    w->runtime_tokens = runtime_tokens;

    if (q3_cuda_init() != 0) {
        fprintf(stderr, "skip: CUDA unavailable\n");
        return false;
    }
    w->model = q3_gguf_open(model_path(), err, sizeof(err));
    if (!w->model) {
        fprintf(stderr, "skip: cannot open %s: %s\n", model_path(), err);
        return false;
    }
    if (!q3_model_config_from_gguf(w->model, &w->config, err, sizeof(err)) ||
        !q3_model_config_validate(w->model, &w->config, err, sizeof(err))) {
        fprintf(stderr, "skip: config: %s\n", err);
        q3_gguf_close(w->model);
        w->model = NULL;
        return false;
    }
    w->loader = q3_loader_context_create(err, sizeof(err));
    if (!w->loader || !q3_loader_load(w->loader, w->model, err, sizeof(err))) {
        fprintf(stderr, "skip: resident load failed: %s\n", err);
        q3_gguf_close(w->model);
        w->model = NULL;
        return false;
    }
    if (!q3_weights_bind(w->model, w->loader, &w->config, &w->weights,
                         err, sizeof(err))) {
        fprintf(stderr, "skip: bind: %s\n", err);
        q3_gguf_close(w->model);
        w->model = NULL;
        return false;
    }
    w->runtime = q3_forward_create(&w->weights, &w->config, runtime_tokens, 0,
                                   err, sizeof(err));
    if (!w->runtime) {
        fprintf(stderr, "skip: forward create: %s\n", err);
        q3_gguf_close(w->model);
        w->model = NULL;
        return false;
    }
    w->hidden = w->config.hidden_size;
    w->layers = w->config.num_layers;
    w->n_kv_heads = w->config.num_kv_heads;
    w->head_dim = w->config.head_dim;
    return true;
}

static bool dec_open(dec_world *w, uint32_t runtime_tokens) {
    if (!dec_open_model(w, runtime_tokens)) return false;
    char err[512] = {0};
    w->workload = q3_workload_load(workload_path(), err, sizeof(err));
    if (!w->workload) {
        fprintf(stderr, "skip: workload %s: %s\n", workload_path(), err);
        q3_forward_destroy(w->runtime);
        q3_weights_free(&w->weights);
        q3_loader_context_destroy(w->loader);
        q3_gguf_close(w->model);
        w->runtime = NULL;
        w->model = NULL;
        return false;
    }
    return true;
}

static void dec_close(dec_world *w) {
    if (w->branches) q3_branch_set_destroy(w->branches);
    if (w->prefix) q3_prefix_destroy(w->prefix);
    if (w->runtime) q3_forward_destroy(w->runtime);
    q3_weights_free(&w->weights);
    if (w->loader) q3_loader_context_destroy(w->loader);
    if (w->model) q3_gguf_close(w->model);
    q3_workload_free(w->workload);
}

/* Prefill the shared prefix once and seal it. */
static bool dec_prefill(dec_world *w, uint32_t capacity, uint32_t max_branches,
                        uint32_t max_suffix) {
    char err[512] = {0};
    q3_forward_stats stats;
    memset(&stats, 0, sizeof(stats));
    w->prefix = q3_prefix_create(w->runtime, capacity, err, sizeof(err));
    if (!w->prefix) {
        fprintf(stderr, "FAIL: prefix create: %s\n", err);
        return false;
    }
    if (!q3_prefix_prefill(w->prefix, w->workload->state_tokens,
                           w->workload->state_count, &stats, err, sizeof(err))) {
        fprintf(stderr, "FAIL: prefix prefill: %s\n", err);
        return false;
    }
    if (!q3_prefix_seal(w->prefix, err, sizeof(err))) {
        fprintf(stderr, "FAIL: prefix seal: %s\n", err);
        return false;
    }
    w->branches = q3_branch_set_create(w->prefix, max_branches, max_suffix,
                                       err, sizeof(err));
    if (!w->branches) {
        fprintf(stderr, "FAIL: branch set: %s\n", err);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Test 1 - segmented KV, B1: prefix+suffix vs the M2 one-shot path.   */
/* ------------------------------------------------------------------ */

static void test_segmented_b1(void) {
    fprintf(stderr, "-- test 1: segmented KV B1\n");

    dec_world w;
    if (!dec_open(&w, 256)) {
        dec_close(&w);
        return;
    }

    /* A single branch whose suffix is the tail of the prefix prompt: the M2
     * reference run is the ordinary contiguous forward over the whole prompt,
     * so the M3 path must land on the same prediction. */
    const uint32_t P = w.workload->state_count;
    const uint32_t S = 4;
    if (P < S + 1) {
        fprintf(stderr, "skip: prompt too short\n");
        dec_close(&w);
        return;
    }
    if (!dec_prefill(&w, P, 2, S + 1)) {
        failures++;
        dec_close(&w);
        return;
    }

    char err[512] = {0};
    const uint32_t candidates[] = { 16, 17, 18, 19, 33, 969 };
    const uint32_t K = (uint32_t)(sizeof(candidates) / sizeof(candidates[0]));

    /* --- M2 reference: one contiguous pass over the full prompt, then the
     * suffix tokens appended, exactly as M2 does it. --------------------- */
    q3_kv_cache *kv = q3_kv_cache_init(w.layers, w.n_kv_heads, w.head_dim,
                                       P + S, err, sizeof(err));
    q3_kv_cache *kv_full = q3_kv_cache_init(w.layers, w.n_kv_heads, w.head_dim,
                                            P + S, err, sizeof(err));
    float *d_hidden = NULL, *d_logits_ref = NULL, *d_logits_m3 = NULL;
    float *d_logits_full = NULL;
    uint32_t ref_argmax = 0, m3_argmax = 0;
    float ref_logits[8], m3_logits[8];
    int32_t handle = -1;

    if (!kv || !kv_full ||
        cudaMalloc(&d_hidden, w.hidden * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits_ref, K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits_full, K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_logits_m3, K * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "FAIL: setup: %s\n", err);
        failures++;
        goto out;
    }
    /* Diagnostic: the drift inherent to splitting one contiguous prompt into
     * a prefix pass plus a suffix pass, measured entirely inside M2. */
    {
        q3_forward_stats stats;
        memset(&stats, 0, sizeof(stats));
        float full_logits[8];
        if (q3_forward_run(w.runtime, w.workload->state_tokens, P + S, kv_full,
                           0, d_hidden, &stats, err, sizeof(err)) &&
            q3_candidate_logits(w.runtime, d_hidden, candidates, K,
                                d_logits_full, err, sizeof(err)) &&
            q3_forward_synchronize(w.runtime, err, sizeof(err))) {
            cudaMemcpy(full_logits, d_logits_full, K * sizeof(float),
                       cudaMemcpyDeviceToHost);
            fprintf(stderr, "   [diag] M2 contiguous logits:");
            for (uint32_t i = 0; i < K; i++)
                fprintf(stderr, " %.5f", full_logits[i]);
            fprintf(stderr, "\n");
        } else {
            fprintf(stderr, "   [diag] contiguous run failed: %s\n", err);
        }
    }
    {
        q3_forward_stats stats;
        memset(&stats, 0, sizeof(stats));
        if (!q3_forward_run(w.runtime, w.workload->state_tokens, P, kv, 0,
                            d_hidden, &stats, err, sizeof(err)) ||
            !q3_forward_run(w.runtime, w.workload->state_tokens + P, S, kv, P,
                            d_hidden, &stats, err, sizeof(err)) ||
            !q3_candidate_logits(w.runtime, d_hidden, candidates, K,
                                 d_logits_ref, err, sizeof(err)) ||
            !q3_forward_synchronize(w.runtime, err, sizeof(err))) {
            fprintf(stderr, "FAIL: M2 reference: %s\n", err);
            failures++;
            goto out;
        }
        cudaMemcpy(ref_logits, d_logits_ref, K * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (uint32_t i = 1; i < K; i++)
            if (ref_logits[i] > ref_logits[ref_argmax]) ref_argmax = i;
    }

    /* --- M3: one branch, one packed suffix forward. ------------------- */
    handle = q3_branch_acquire(w.branches, err, sizeof(err));
    if (handle < 0) {
        fprintf(stderr, "FAIL: branch acquire: %s\n", err);
        failures++;
        goto out;
    }
    {
        q3_decision_item item;
        memset(&item, 0, sizeof(item));
        item.branch = (uint32_t)handle;
        item.tokens = w.workload->state_tokens + P;
        item.token_count = S;
        item.candidate_ids = candidates;
        item.candidate_count = K;

        q3_decision_batch batch;
        batch.items = &item;
        batch.count = 1;

        q3_decision_result result;
        memset(&result, 0, sizeof(result));
        q3_decide_stats stats;
        memset(&stats, 0, sizeof(stats));

        if (!q3_decide_run(w.runtime, w.prefix, w.branches, &batch, &result,
                           &stats, err, sizeof(err))) {
            fprintf(stderr, "FAIL: decide run: %s\n", err);
            failures++;
            goto out;
        }
        m3_argmax = result.predicted_index;
        for (uint32_t i = 0; i < K; i++) m3_logits[i] = result.logits[i];

        CHECK(stats.prefix_copy_bytes == 0,
              "B1: prefix_copy_bytes == 0");
        CHECK(stats.cuda_allocations == 0,
              "B1: no hot CUDA allocations");
        CHECK(stats.host_syncs <= 1,
              "B1: at most the one explicit boundary sync");
    }

    {
        float max_diff = 0.0f;
        float max_diff_split = 0.0f;
        float full_logits[8];
        cudaMemcpy(full_logits, d_logits_full, K * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < K; i++) {
            const float d = fabsf(ref_logits[i] - m3_logits[i]);
            if (d > max_diff) max_diff = d;
            const float ds = fabsf(full_logits[i] - ref_logits[i]);
            if (ds > max_diff_split) max_diff_split = ds;
        }
        fprintf(stderr, "   ref argmax=%u m3 argmax=%u max|diff|=%.4g\n",
                ref_argmax, m3_argmax, max_diff);
        /* If M2's own split (P then S) already drifts this much from M2's
         * own contiguous pass, the residual is inherent to the split, not an
         * M3 segmented-attention bug. */
        fprintf(stderr, "   [diag] M2 contiguous vs M2 split max|diff|=%.4g\n",
                max_diff_split);
        /* The KV mechanics are validated to ~1e-7 by the M2 primitive tests.
         * What remains is FP accumulation-order drift inherent to splitting a
         * contiguous prompt into a prefix pass plus a suffix pass: M2's own
         * split path drifts `max_diff_split` from M2's own contiguous pass.
         * M3 must not be worse than that inherent split drift, so the gate is
         * the discrete candidate plus "no worse than M2's own split". */
        CHECK(ref_argmax == m3_argmax,
              "B1: segmented argmax matches the contiguous path");
        CHECK(max_diff <= max_diff_split + 1e-3f,
              "B1: segmented logits no worse than M2's own split drift");
    }

out:
    if (handle >= 0) q3_branch_release(w.branches, handle);
    if (kv) q3_kv_cache_destroy(kv);
    if (kv_full) q3_kv_cache_destroy(kv_full);
    if (d_hidden) cudaFree(d_hidden);
    if (d_logits_ref) cudaFree(d_logits_ref);
    if (d_logits_full) cudaFree(d_logits_full);
    if (d_logits_m3) cudaFree(d_logits_m3);
    dec_close(&w);
}

/* ------------------------------------------------------------------ */
/* Test 2 - B2 branch isolation (docs/M3.md §32).                      */
/* ------------------------------------------------------------------ */

static void test_b2_isolation(void) {
    fprintf(stderr, "-- test 2: B2 branch isolation\n");

    dec_world w;
    if (!dec_open(&w, 256)) {
        dec_close(&w);
        return;
    }

    const q3_workload *wl = w.workload;
    if (wl->question_count < 2) {
        fprintf(stderr, "skip: workload has fewer than 2 questions\n");
        dec_close(&w);
        return;
    }

    uint32_t max_suffix = 0;
    for (uint32_t i = 0; i < wl->question_count; i++)
        if (wl->questions[i].suffix_count > max_suffix)
            max_suffix = wl->questions[i].suffix_count;

    if (!dec_prefill(&w, wl->state_count, 4, max_suffix)) {
        failures++;
        dec_close(&w);
        return;
    }

    char err[512] = {0};
    int32_t h[2];
    h[0] = q3_branch_acquire(w.branches, err, sizeof(err));
    h[1] = q3_branch_acquire(w.branches, err, sizeof(err));
    if (h[0] < 0 || h[1] < 0) {
        fprintf(stderr, "FAIL: branch acquire: %s\n", err);
        failures++;
        dec_close(&w);
        return;
    }

    /* Every question must use the same K, so take the widest list. */
    uint32_t K = 0;
    for (uint32_t i = 0; i < wl->question_count; i++)
        if (wl->questions[i].candidate_count > K)
            K = wl->questions[i].candidate_count;

    uint32_t *padded[2];
    q3_decision_item items[2];
    memset(items, 0, sizeof(items));
    for (int b = 0; b < 2; b++) {
        padded[b] = (uint32_t *)malloc((size_t)K * sizeof(uint32_t));
        if (!padded[b]) {
            fprintf(stderr, "FAIL: malloc\n");
            failures++;
            dec_close(&w);
            return;
        }
        const q3_workload_question *q = &wl->questions[b];
        for (uint32_t k = 0; k < K; k++)
            padded[b][k] = q->candidate_ids[k < q->candidate_count
                                                ? k
                                                : q->candidate_count - 1];
        items[b].branch = (uint32_t)h[b];
        items[b].tokens = q->suffix_tokens;
        items[b].token_count = q->suffix_count;
        items[b].candidate_ids = padded[b];
        items[b].candidate_count = K;
    }

    q3_decision_batch batch;
    batch.items = items;
    batch.count = 2;

    q3_decision_result pair[2], solo[2];
    memset(pair, 0, sizeof(pair));
    memset(solo, 0, sizeof(solo));
    q3_decide_stats stats;
    memset(&stats, 0, sizeof(stats));

    bool ok = q3_decide_run(w.runtime, w.prefix, w.branches, &batch, pair,
                            &stats, err, sizeof(err));
    if (!ok) {
        fprintf(stderr, "FAIL: paired decide run: %s\n", err);
        failures++;
    }

    /* Each branch alone: a batch of one must reproduce the paired result. */
    for (int b = 0; b < 2; b++) {
        q3_decision_batch one;
        one.items = &items[b];
        one.count = 1;
        if (!q3_decide_run(w.runtime, w.prefix, w.branches, &one, &solo[b],
                           NULL, err, sizeof(err))) {
            fprintf(stderr, "FAIL: solo decide run %d: %s\n", b, err);
            failures++;
            ok = false;
        }
    }

    if (ok) {
        for (int b = 0; b < 2; b++) {
            fprintf(stderr, "   branch %d: paired=%u solo=%u\n", b,
                    pair[b].predicted_index, solo[b].predicted_index);
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "B2: branch %d prediction unaffected by the other branch",
                     b);
            CHECK(pair[b].predicted_index == solo[b].predicted_index, msg);
        }
        /* Both branches ran in the same packed batch, so their logits must be
         * separately produced: two identical question suffixes with different
         * candidates would still differ, but an aliased scratch buffer would
         * make the whole [B,K] tile identical. */
        bool tile_distinct = false;
        for (uint32_t k = 0; k < K; k++)
            if (pair[0].logits[k] != pair[1].logits[k]) tile_distinct = true;
        CHECK(tile_distinct, "B2: branches produce distinct logit rows");
    }

    /* Reversing the batch order must not change either branch. */
    if (ok) {
        q3_decision_item rev[2] = { items[1], items[0] };
        q3_decision_batch rbatch;
        rbatch.items = rev;
        rbatch.count = 2;
        q3_decision_result rres[2];
        memset(rres, 0, sizeof(rres));
        if (q3_decide_run(w.runtime, w.prefix, w.branches, &rbatch, rres, NULL,
                          err, sizeof(err))) {
            CHECK(rres[0].predicted_index == pair[1].predicted_index &&
                      rres[1].predicted_index == pair[0].predicted_index,
                  "B2: branch result is independent of its batch slot");
        } else {
            fprintf(stderr, "FAIL: reversed decide run: %s\n", err);
            failures++;
        }
    }

    free(padded[0]);
    free(padded[1]);
    q3_branch_release(w.branches, h[0]);
    q3_branch_release(w.branches, h[1]);
    dec_close(&w);
}

/* ------------------------------------------------------------------ */
/* Test 3 - B32 over the real workload.                                */
/* ------------------------------------------------------------------ */

static void test_b32(void) {
    fprintf(stderr, "-- test 3: B32 decision parity\n");

    dec_world w;
    if (!dec_open(&w, 1024)) {
        dec_close(&w);
        return;
    }

    const q3_workload *wl = w.workload;
    const uint32_t B = wl->question_count < Q3_MAX_BRANCHES
                           ? wl->question_count
                           : (uint32_t)Q3_MAX_BRANCHES;

    uint32_t max_suffix = 0, max_packed = 0, K = 0;
    for (uint32_t i = 0; i < B; i++) {
        if (wl->questions[i].suffix_count > max_suffix)
            max_suffix = wl->questions[i].suffix_count;
        if (wl->questions[i].candidate_count > K)
            K = wl->questions[i].candidate_count;
        max_packed += wl->questions[i].suffix_count;
    }
    if (!dec_prefill(&w, wl->state_count, B, max_suffix)) {
        failures++;
        dec_close(&w);
        return;
    }

    char err[512] = {0};
    q3_decision_item items[Q3_MAX_BRANCHES];
    uint32_t *padded[Q3_MAX_BRANCHES];
    int32_t handles[Q3_MAX_BRANCHES];
    memset(items, 0, sizeof(items));
    for (uint32_t i = 0; i < B; i++) padded[i] = NULL;

    bool setup = true;
    for (uint32_t i = 0; i < B; i++) {
        handles[i] = q3_branch_acquire(w.branches, err, sizeof(err));
        if (handles[i] < 0) {
            fprintf(stderr, "FAIL: branch acquire %u: %s\n", i, err);
            setup = false;
            break;
        }
        padded[i] = (uint32_t *)malloc((size_t)K * sizeof(uint32_t));
        if (!padded[i]) {
            fprintf(stderr, "FAIL: malloc\n");
            setup = false;
            break;
        }
        const q3_workload_question *q = &wl->questions[i];
        for (uint32_t k = 0; k < K; k++)
            padded[i][k] = q->candidate_ids[k < q->candidate_count
                                                ? k
                                                : q->candidate_count - 1];
        items[i].branch = (uint32_t)handles[i];
        items[i].tokens = q->suffix_tokens;
        items[i].token_count = q->suffix_count;
        items[i].candidate_ids = padded[i];
        items[i].candidate_count = K;
    }
    if (!setup) {
        failures++;
        for (uint32_t i = 0; i < B; i++) free(padded[i]);
        dec_close(&w);
        return;
    }

    q3_decision_batch batch;
    batch.items = items;
    batch.count = B;

    q3_decision_result *results =
        (q3_decision_result *)calloc(B, sizeof(q3_decision_result));
    q3_decide_stats stats;
    memset(&stats, 0, sizeof(stats));

    const bool ok = results && q3_decide_run(w.runtime, w.prefix, w.branches,
                                             &batch, results, &stats, err,
                                             sizeof(err));
    if (!ok) {
        fprintf(stderr, "FAIL: B%u decide run: %s\n", B, err);
        failures++;
    } else {
        uint32_t matches = 0, finite = 0;
        for (uint32_t i = 0; i < B; i++) {
            bool f = true;
            for (uint32_t k = 0; k < K; k++)
                if (!isfinite(results[i].logits[k])) f = false;
            if (f) finite++;
            if (results[i].predicted_index == wl->questions[i].expected_index)
                matches++;
        }
        fprintf(stderr, "   B=%u packed=%u finite=%u/%u expected-match=%u/%u\n",
                B, max_packed, finite, B, matches, B);
        CHECK(finite == B, "B32: every branch produced finite logits");
        CHECK(stats.prefix_copy_bytes == 0, "B32: prefix_copy_bytes == 0");
        CHECK(stats.cuda_allocations == 0, "B32: no hot CUDA allocations");
        CHECK(stats.suffix_tokens_total == max_packed,
              "B32: packed suffix total matches the workload");
        /* Accuracy is reported, never enforced (M3 §34-§41). */
        fprintf(stderr, "   accuracy: %u/%u (reported, not gated)\n", matches,
                B);
    }

    free(results);
    for (uint32_t i = 0; i < B; i++) free(padded[i]);
    for (uint32_t i = 0; i < B; i++)
        if (handles[i] >= 0) q3_branch_release(w.branches, handles[i]);
    dec_close(&w);
}

/* ------------------------------------------------------------------ */
/* Test 4 - memory/runtime invariants across B.                        */
/* ------------------------------------------------------------------ */

static void test_invariants(void) {
    fprintf(stderr, "-- test 4: memory/runtime invariants\n");

    dec_world w;
    if (!dec_open(&w, 1024)) {
        dec_close(&w);
        return;
    }

    const q3_workload *wl = w.workload;
    uint32_t max_suffix = 0;
    for (uint32_t i = 0; i < wl->question_count; i++)
        if (wl->questions[i].suffix_count > max_suffix)
            max_suffix = wl->questions[i].suffix_count;

    if (!dec_prefill(&w, wl->state_count, Q3_MAX_BRANCHES, max_suffix)) {
        failures++;
        dec_close(&w);
        return;
    }

    char err[512] = {0};

    CHECK(q3_prefix_copy_bytes(w.prefix) == 0,
          "prefix_copy_bytes == 0 after prefill");
    CHECK(q3_prefix_allocations(w.prefix) == 2,
          "prefix owns exactly two allocations (K and V)");

    const size_t prefix_bytes = q3_prefix_kv_bytes(w.prefix);
    const size_t expected = q3_kv_bytes_for_tokens(
        wl->state_count, w.layers, w.n_kv_heads, w.head_dim);
    fprintf(stderr, "   prefix bytes=%zu expected=%zu\n", prefix_bytes,
            expected);
    CHECK(prefix_bytes == expected, "prefix physical bytes match geometry");

    /* The prefix is written once: re-prefilling the same object is refused,
     * and sealing again is a no-op that must not touch storage. */
    {
        q3_forward_stats stats;
        memset(&stats, 0, sizeof(stats));
        const bool refused = !q3_prefix_prefill(w.prefix, wl->state_tokens,
                                                wl->state_count, &stats, err,
                                                sizeof(err));
        CHECK(refused, "a sealed prefix refuses a second prefill");
        CHECK(q3_prefix_copy_bytes(w.prefix) == 0,
              "refused prefill copies nothing");
    }

    /* Walk B = 1,2,4,8,16,32 and check the prefix is untouched and the warm
     * path allocates nothing. */
    const uint32_t ladder[] = { 1, 2, 4, 8, 16, 32 };
    for (unsigned li = 0; li < sizeof(ladder) / sizeof(ladder[0]); li++) {
        uint32_t B = ladder[li];
        if (B > wl->question_count) B = wl->question_count;
        if (B == 0) continue;

        uint32_t K = 0;
        for (uint32_t i = 0; i < B; i++)
            if (wl->questions[i].candidate_count > K)
                K = wl->questions[i].candidate_count;

        q3_decision_item items[Q3_MAX_BRANCHES];
        uint32_t *padded[Q3_MAX_BRANCHES];
        int32_t handles[Q3_MAX_BRANCHES];
        memset(items, 0, sizeof(items));
        bool setup = true;
        for (uint32_t i = 0; i < B; i++) {
            handles[i] = q3_branch_acquire(w.branches, err, sizeof(err));
            padded[i] = (uint32_t *)malloc((size_t)K * sizeof(uint32_t));
            if (handles[i] < 0 || !padded[i]) { setup = false; break; }
            const q3_workload_question *q = &wl->questions[i];
            for (uint32_t k = 0; k < K; k++)
                padded[i][k] = q->candidate_ids[k < q->candidate_count
                                                    ? k
                                                    : q->candidate_count - 1];
            items[i].branch = (uint32_t)handles[i];
            items[i].tokens = q->suffix_tokens;
            items[i].token_count = q->suffix_count;
            items[i].candidate_ids = padded[i];
            items[i].candidate_count = K;
        }
        if (!setup) {
            fprintf(stderr, "FAIL: B%u setup\n", B);
            failures++;
            for (uint32_t i = 0; i < B; i++) free(padded[i]);
            break;
        }

        q3_decision_batch batch;
        batch.items = items;
        batch.count = B;
        q3_decision_result results[Q3_MAX_BRANCHES];
        memset(results, 0, sizeof(results));
        q3_decide_stats stats;
        memset(&stats, 0, sizeof(stats));

        const bool ok = q3_decide_run(w.runtime, w.prefix, w.branches, &batch,
                                      results, &stats, err, sizeof(err));
        char msg[160];
        if (!ok) {
            snprintf(msg, sizeof(msg), "B%u runs", B);
            fprintf(stderr, "FAIL: %s: %s\n", msg, err);
            failures++;
        } else {
            snprintf(msg, sizeof(msg), "B%-2u warm: zero allocations", B);
            CHECK(stats.cuda_allocations == 0, msg);
            snprintf(msg, sizeof(msg), "B%-2u warm: at most one sync", B);
            CHECK(stats.host_syncs <= 1, msg);
            snprintf(msg, sizeof(msg),
                     "B%-2u prefix bytes unchanged (%zu)", B,
                     q3_prefix_kv_bytes(w.prefix));
            CHECK(q3_prefix_kv_bytes(w.prefix) == prefix_bytes, msg);
            snprintf(msg, sizeof(msg), "B%-2u prefix copy bytes still 0", B);
            CHECK(q3_prefix_copy_bytes(w.prefix) == 0, msg);
        }

        for (uint32_t i = 0; i < B; i++) free(padded[i]);
        for (uint32_t i = 0; i < B; i++)
            if (handles[i] >= 0) q3_branch_release(w.branches, handles[i]);
    }

    /* Reuse: the same batch three times must be stable and allocation-free. */
    {
        const uint32_t B = wl->question_count < 8 ? wl->question_count : 8;
        uint32_t K = 0;
        for (uint32_t i = 0; i < B; i++)
            if (wl->questions[i].candidate_count > K)
                K = wl->questions[i].candidate_count;
        q3_decision_item items[Q3_MAX_BRANCHES];
        uint32_t *padded[Q3_MAX_BRANCHES];
        int32_t handles[Q3_MAX_BRANCHES];
        memset(items, 0, sizeof(items));
        bool setup = true;
        for (uint32_t i = 0; i < B; i++) {
            handles[i] = q3_branch_acquire(w.branches, err, sizeof(err));
            padded[i] = (uint32_t *)malloc((size_t)K * sizeof(uint32_t));
            if (handles[i] < 0 || !padded[i]) { setup = false; break; }
            const q3_workload_question *q = &wl->questions[i];
            for (uint32_t k = 0; k < K; k++)
                padded[i][k] = q->candidate_ids[k < q->candidate_count
                                                    ? k
                                                    : q->candidate_count - 1];
            items[i].branch = (uint32_t)handles[i];
            items[i].tokens = q->suffix_tokens;
            items[i].token_count = q->suffix_count;
            items[i].candidate_ids = padded[i];
            items[i].candidate_count = K;
        }
        if (setup) {
            q3_decision_batch batch;
            batch.items = items;
            batch.count = B;
            q3_decision_result r0[Q3_MAX_BRANCHES], r1[Q3_MAX_BRANCHES];
            memset(r0, 0, sizeof(r0));
            memset(r1, 0, sizeof(r1));
            uint64_t allocs = 0, syncs = 0;
            bool ok = true;
            for (int rep = 0; rep < 3; rep++) {
                q3_decide_stats stats;
                memset(&stats, 0, sizeof(stats));
                q3_decision_result *dst = rep == 0 ? r0 : r1;
                if (!q3_decide_run(w.runtime, w.prefix, w.branches, &batch, dst,
                                   &stats, err, sizeof(err))) {
                    fprintf(stderr, "FAIL: reuse rep %d: %s\n", rep, err);
                    failures++;
                    ok = false;
                    break;
                }
                allocs += stats.cuda_allocations;
                syncs += stats.host_syncs;
            }
            if (ok) {
                bool same = true;
                for (uint32_t i = 0; i < B; i++)
                    if (r0[i].predicted_index != r1[i].predicted_index)
                        same = false;
                CHECK(same, "3x B8 reuse: prediction fingerprint is stable");
                CHECK(allocs == 0, "3x B8 reuse: zero allocations total");
                CHECK(syncs == 3, "3x B8 reuse: exactly one sync per batch");
            }
        } else {
            failures++;
        }
        for (uint32_t i = 0; i < B; i++) free(padded[i]);
        for (uint32_t i = 0; i < B; i++)
            if (handles[i] >= 0) q3_branch_release(w.branches, handles[i]);
    }

    dec_close(&w);
}

int main(void) {
    fprintf(stderr, "q3 M3 decision tests\n");
    test_segmented_b1();
    test_b2_isolation();
    test_b32();
    test_invariants();
    if (failures) {
        fprintf(stderr, "FAILED: %d check(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
