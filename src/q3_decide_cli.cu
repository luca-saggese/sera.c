/* SPDX-License-Identifier: MIT
 *
 * q3_decide_cli.cu - M3 --bench-decisions diagnostic (docs/M3.md §54-§55).
 *
 *     ./q3 --bench-decisions MODEL.gguf --workload workload.json \
 *          --batch 1,2,4,8,16,32 --json
 *
 * One prefix prefill is shared by every batch size in the ladder; only the
 * suffix batch is rebuilt and re-run per size. The measured warm path must
 * show zero CUDA allocations, zero frees and zero host syncs beyond the single
 * explicit decision boundary (§42-§44).
 *
 * Correctness is deliberately NOT gated here (M3 §34-§41): accuracy is
 * reported, never enforced.
 */
#include "q3_decide_cli.h"

#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model.h"
#include "q3_binder.h"
#include "q3_model_loader_cuda.h"
#include "q3_decide.h"
#include "q3_workload.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* A batch of the ladder: how many questions, and the packed suffix total. */
typedef struct {
    uint32_t batch;
    uint32_t question_count;
    uint32_t suffix_tokens;
    uint32_t max_suffix;
    uint32_t candidates;
} decide_batch_plan;

static decide_batch_plan plan_for(const q3_workload *w, uint32_t batch) {
    decide_batch_plan p;
    memset(&p, 0, sizeof(p));
    p.batch = batch;
    if (batch > w->question_count) batch = w->question_count;
    p.question_count = batch;
    for (uint32_t i = 0; i < batch; i++) {
        p.suffix_tokens += w->questions[i].suffix_count;
        if (w->questions[i].suffix_count > p.max_suffix)
            p.max_suffix = w->questions[i].suffix_count;
        if (w->questions[i].candidate_count > p.candidates)
            p.candidates = w->questions[i].candidate_count;
    }
    return p;
}

int q3_cmd_bench_decisions(const q3_options *opt) {
    char err[512] = {0};

    if (!opt->workload_path) {
        fprintf(stderr, "q3: --bench-decisions requires --workload <file>\n");
        return 2;
    }
    if (q3_cuda_init() != 0) {
        fprintf(stderr, "q3: CUDA initialization failed\n");
        return 1;
    }

    q3_workload *w = q3_workload_load(opt->workload_path, err, sizeof(err));
    if (!w) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) { fprintf(stderr, "q3: %s\n", err); q3_workload_free(w); return 1; }

    q3_model_config cfg;
    if (!q3_model_config_from_gguf(m, &cfg, err, sizeof(err)) ||
        !q3_model_config_validate(m, &cfg, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    q3_loader_context *ctx = q3_loader_context_create(err, sizeof(err));
    if (!ctx || !q3_loader_load(ctx, m, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        if (ctx) q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    q3_weights weights;
    if (!q3_weights_bind(m, ctx, &cfg, &weights, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    /* Batch ladder. The runtime and the prefix must cover the largest packed
     * suffix total in the ladder, plus the prefix itself for the prefill. */
    int batches[16];
    uint32_t batch_count = 0;
    if (opt->batch_count > 0) {
        for (int i = 0; i < opt->batch_count && batch_count < 16; i++)
            batches[batch_count++] = opt->batches[i];
    } else {
        static const int def[] = { 1, 2, 4, 8, 16, 32 };
        for (unsigned i = 0; i < sizeof(def) / sizeof(def[0]); i++)
            batches[batch_count++] = def[i];
    }

    uint32_t max_packed = 0, max_suffix = w->max_suffix_count;
    for (uint32_t i = 0; i < batch_count; i++) {
        const decide_batch_plan p = plan_for(w, (uint32_t)batches[i]);
        if (p.suffix_tokens > max_packed) max_packed = p.suffix_tokens;
    }
    if (max_packed == 0) max_packed = 1;

    /* The runtime must never be created with a capacity below the workload it
     * is about to run: max(prefix prefill, largest packed suffix batch). The
     * prefix prefill runs through the same runtime, so its token count has to
     * fit too. */
    uint32_t runtime_tokens = w->state_count;
    if (max_packed > runtime_tokens) runtime_tokens = max_packed;
    if (runtime_tokens == 0) runtime_tokens = 1;
    if (opt->verbose)
        fprintf(stderr,
                "q3: prefix=%u max_packed=%u runtime_tokens=%u\n",
                w->state_count, max_packed, runtime_tokens);

    q3_forward_runtime *rt = q3_forward_create(&weights, &cfg, runtime_tokens,
                                               0, err, sizeof(err));
    if (!rt) {
        fprintf(stderr, "q3: %s\n", err);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    q3_prefix_kv *prefix = q3_prefix_create(rt, w->state_count, err,
                                            sizeof(err));
    if (!prefix) {
        fprintf(stderr, "q3: %s\n", err);
        q3_forward_destroy(rt);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    q3_branch_set *set = q3_branch_set_create(prefix, Q3_MAX_BRANCHES,
                                              max_suffix, err, sizeof(err));
    if (!set) {
        fprintf(stderr, "q3: %s\n", err);
        q3_prefix_destroy(prefix);
        q3_forward_destroy(rt);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    /* ---- Prefix prefill: once, shared by every batch size (M3 §8-§11). */
    q3_forward_stats pstats;
    const double prefill_begin = now_ms();
    if (!q3_prefix_prefill(prefix, w->state_tokens, w->state_count, &pstats,
                           err, sizeof(err)) ||
        !q3_prefix_seal(prefix, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_branch_set_destroy(set);
        q3_prefix_destroy(prefix);
        q3_forward_destroy(rt);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }
    const double prefix_prefill_ms = now_ms() - prefill_begin;

    /* Branch handles: one per question, acquired once and reused across every
     * batch size, so branch creation is never in the measured path. */
    int32_t *handles = (int32_t *)malloc((size_t)w->question_count * sizeof(int32_t));
    if (!handles) {
        fprintf(stderr, "q3: out of memory\n");
        q3_branch_set_destroy(set);
        q3_prefix_destroy(prefix);
        q3_forward_destroy(rt);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }
    const double branch_begin = now_ms();
    for (uint32_t i = 0; i < w->question_count; i++) {
        handles[i] = q3_branch_acquire(set, err, sizeof(err));
        if (handles[i] < 0) {
            fprintf(stderr, "q3: %s\n", err);
            free(handles);
            q3_branch_set_destroy(set);
            q3_prefix_destroy(prefix);
            q3_forward_destroy(rt);
            q3_weights_free(&weights);
            q3_loader_context_destroy(ctx);
            q3_gguf_close(m);
            q3_workload_free(w);
            return 1;
        }
    }
    const double branch_create_ms = now_ms() - branch_begin;

    q3_decision_item *items =
        (q3_decision_item *)calloc(w->question_count, sizeof(*items));
    q3_decision_result *results =
        (q3_decision_result *)calloc(w->question_count, sizeof(*results));
    if (!items || !results) {
        fprintf(stderr, "q3: out of memory\n");
        free(items); free(results); free(handles);
        q3_branch_set_destroy(set);
        q3_prefix_destroy(prefix);
        q3_forward_destroy(rt);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        q3_workload_free(w);
        return 1;
    }

    /* The branch-isolation canary (M3 §32): two questions with different
     * suffixes sharing the same prefix must not perturb each other. */
    uint32_t canary_ok = 0;

    if (!opt->json) {
        printf("prefix tokens:     %u\n", w->state_count);
        printf("questions:         %u\n", w->question_count);
        printf("prefix prefill:    %.2f ms\n", prefix_prefill_ms);
        printf("branch create:     %.2f ms (%u branches)\n", branch_create_ms,
               w->question_count);
        printf("prefix kv bytes:   %zu\n", q3_prefix_kv_bytes(prefix));
        printf("suffix kv bytes:   %zu\n", q3_branch_set_kv_bytes(set));
        printf("prefix copy bytes: %" PRIu64 "\n",
               q3_prefix_copy_bytes(prefix));
    } else {
        printf("{\n");
        printf("  \"prefix_tokens\": %u,\n", w->state_count);
        printf("  \"question_count\": %u,\n", w->question_count);
        printf("  \"prefix_prefill_ms\": %.4f,\n", prefix_prefill_ms);
        printf("  \"prefix_kv_bytes\": %zu,\n", q3_prefix_kv_bytes(prefix));
        printf("  \"suffix_kv_bytes\": %zu,\n", q3_branch_set_kv_bytes(set));
        printf("  \"prefix_allocations\": %" PRIu64 ",\n",
               q3_prefix_allocations(prefix));
        printf("  \"prefix_copy_bytes\": %" PRIu64 ",\n",
               q3_prefix_copy_bytes(prefix));
        printf("  \"branch_create_ms\": %.4f,\n", branch_create_ms);
        printf("  \"results\": [\n");
    }

    uint32_t total_matches = 0, total_questions = 0;
    for (uint32_t bi = 0; bi < batch_count; bi++) {
        const decide_batch_plan plan = plan_for(w, (uint32_t)batches[bi]);
        if (plan.question_count == 0) continue;

        /* Candidates are uniform per batch (§15): the batched LM head writes a
         * dense [B, K] tile, so every item in one batch uses the same K. Take
         * the maximum any item in this batch declares. */
        const uint32_t K = plan.candidates;
        for (uint32_t i = 0; i < plan.question_count; i++) {
            const q3_workload_question *q = &w->questions[i];
            items[i].branch = handles[i];
            items[i].tokens = q->suffix_tokens;
            items[i].token_count = q->suffix_count;
            items[i].candidate_ids = q->candidate_ids;
            /* Pad a shorter candidate list by repeating its last id: the extra
             * rows are scored but never reported for that question. */
            items[i].candidate_count = K;
        }
        /* Uniform-K buffers, reused across the ladder. */
        uint32_t *padded_ids = (uint32_t *)malloc((size_t)plan.question_count * K *
                                                  sizeof(uint32_t));
        if (!padded_ids) {
            fprintf(stderr, "q3: out of memory\n");
            break;
        }
        for (uint32_t i = 0; i < plan.question_count; i++) {
            const q3_workload_question *q = &w->questions[i];
            for (uint32_t k = 0; k < K; k++)
                padded_ids[(size_t)i * K + k] =
                    q->candidate_ids[k < q->candidate_count ? k
                                                            : q->candidate_count - 1];
            items[i].candidate_ids = padded_ids + (size_t)i * K;
        }

        q3_decision_batch batch;
        batch.items = items;
        batch.count = plan.question_count;

        q3_decide_stats stats;
        memset(&stats, 0, sizeof(stats));
        const double run_begin = now_ms();
        const bool ok = q3_decide_run(rt, prefix, set, &batch, results, &stats,
                                      err, sizeof(err));
        const double run_ms = now_ms() - run_begin;

        uint32_t matches = 0;
        if (ok) {
            for (uint32_t i = 0; i < plan.question_count; i++) {
                if (results[i].predicted_index == w->questions[i].expected_index)
                    matches++;
                if (opt->verbose) {
                    const q3_workload_question *q = &w->questions[i];
                    fprintf(stderr, "  %-6s pred=%u exp=%u logits=", q->id,
                            results[i].predicted_index, q->expected_index);
                    for (uint32_t k = 0; k < results[i].candidate_count; k++)
                        fprintf(stderr, "%s%.4f", k ? "," : "",
                                results[i].logits[k]);
                    fprintf(stderr, "\n");
                }
            }
            total_matches += matches;
            total_questions += plan.question_count;
        } else {
            fprintf(stderr, "q3: batch %u: %s\n", plan.batch, err);
        }

        /* Canary: with B>=2, re-running must not change the first branch's
         * prediction (branch isolation, §32). */
        if (ok && plan.question_count >= 2) {
            q3_decision_result again[Q3_MAX_BRANCHES];
            if (q3_decide_run(rt, prefix, set, &batch, again, NULL, err,
                              sizeof(err)) &&
                again[0].predicted_index == results[0].predicted_index)
                canary_ok = 1;
        }

        if (opt->json) {
            printf("    {\n");
            printf("      \"batch\": %u,\n", plan.batch);
            printf("      \"suffix_tokens_total\": %u,\n", plan.suffix_tokens);
            printf("      \"prefix_prefill_ms\": %.4f,\n", prefix_prefill_ms);
            printf("      \"branch_create_ms\": %.4f,\n", branch_create_ms);
            printf("      \"suffix_forward_ms\": %.4f,\n", stats.suffix_forward_ms);
            printf("      \"candidate_head_ms\": %.4f,\n", stats.candidate_head_ms);
            printf("      \"warm_total_ms\": %.4f,\n", run_ms);
            printf("      \"warm_ms_per_question\": %.4f,\n",
                   plan.question_count ? run_ms / plan.question_count : 0.0);
            printf("      \"warm_questions_per_s\": %.4f,\n",
                   run_ms > 0.0 ? 1000.0 * plan.question_count / run_ms : 0.0);
            printf("      \"prefix_kv_bytes\": %zu,\n", stats.prefix_kv_bytes_physical);
            printf("      \"expected_prefix_kv_bytes\": %zu,\n",
                   stats.expected_prefix_kv_bytes);
            printf("      \"suffix_kv_bytes\": %zu,\n", stats.suffix_kv_bytes);
            printf("      \"prefix_copy_bytes\": %" PRIu64 ",\n",
                   stats.prefix_copy_bytes);
            printf("      \"cuda_allocations\": %" PRIu64 ",\n",
                   stats.cuda_allocations);
            printf("      \"host_syncs\": %" PRIu64 ",\n", stats.host_syncs);
            printf("      \"accuracy\": %u,\n", matches);
            printf("      \"questions\": %u\n", plan.question_count);
            printf("    }%s\n", bi + 1 < batch_count ? "," : "");
        } else {
            printf("batch %-3u  suffix %-4u  warm %8.2f ms  %7.2f ms/q  "
                   "acc %u/%u  alloc %" PRIu64 "  sync %" PRIu64 "\n",
                   plan.batch, plan.suffix_tokens, run_ms,
                   plan.question_count ? run_ms / plan.question_count : 0.0,
                   matches, plan.question_count, stats.cuda_allocations,
                   stats.host_syncs);
        }
        free(padded_ids);
    }

    if (opt->json) {
        printf("  ],\n");
        printf("  \"canary_branch_isolation\": %u,\n", canary_ok);
        printf("  \"accuracy_total\": %u,\n", total_matches);
        printf("  \"questions_total\": %u\n", total_questions);
        printf("}\n");
    } else {
        printf("canary isolation:  %s\n", canary_ok ? "pass" : "not run");
        printf("accuracy:          %u/%u\n", total_matches, total_questions);
    }

    free(items);
    free(results);
    free(handles);
    q3_branch_set_destroy(set);
    q3_prefix_destroy(prefix);
    q3_forward_destroy(rt);
    q3_weights_free(&weights);
    q3_loader_context_destroy(ctx);
    q3_gguf_close(m);
    q3_workload_free(w);
    return 0;
}
