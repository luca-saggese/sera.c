/* SPDX-License-Identifier: MIT
 *
 * q3_forward_cli.cu - M2 --forward diagnostic.
 *
 * Loads the real model, binds the stable weight descriptors, creates the
 * forward runtime, runs one one-pass forward over pre-tokenized token IDs,
 * computes candidate-only LM logits, and reports the aggregated timing tree
 * plus KV/cache stats. This is the M2 §1 acceptance surface:
 *
 *     ./q3 --model MODEL.gguf --tokens tokens.bin \
 *          --candidate-token-ids 16,17,18,19 --forward --json
 *
 * The hot path performs zero cudaMalloc/cudaFree and zero host syncs; the
 * only synchronization is the explicit measurement boundary at the end.
 */
#include "q3.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model.h"
#include "q3_binder.h"
#include "q3_model_loader_cuda.h"
#include "q3_forward.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * M2 --forward
 * ========================================================================= */

static bool read_tokens_file(const char *path, uint32_t **out_ids,
                             uint32_t *out_count, char *err, size_t err_len) {
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        snprintf(err, err_len, "cannot open tokens file '%s'", path);
        return false;
    }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    if (size <= 0 || size % 4 != 0) {
        snprintf(err, err_len, "tokens file must be a multiple of 4 bytes");
        fclose(fh);
        return false;
    }
    uint32_t *ids = (uint32_t *)malloc((size_t)size);
    if (!ids) {
        snprintf(err, err_len, "out of memory reading tokens file");
        fclose(fh);
        return false;
    }
    if (fread(ids, 1, (size_t)size, fh) != (size_t)size) {
        snprintf(err, err_len, "short read on tokens file");
        free(ids);
        fclose(fh);
        return false;
    }
    fclose(fh);
    *out_ids = ids;
    *out_count = (uint32_t)(size / 4);
    return true;
}

extern "C" int q3_cmd_forward(const q3_options *opt) {
    char err[256] = {0};

    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) { fprintf(stderr, "q3: %s\n", err); return 1; }

    q3_model_config cfg;
    if (!q3_model_config_from_gguf(m, &cfg, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_gguf_close(m);
        return 1;
    }
    if (!q3_model_config_validate(m, &cfg, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_gguf_close(m);
        return 1;
    }

    if (q3_cuda_init() != 0) {
        fprintf(stderr, "q3: CUDA initialization failed\n");
        q3_gguf_close(m);
        return 1;
    }

    q3_loader_context *ctx = q3_loader_context_create(err, sizeof(err));
    if (!ctx) { fprintf(stderr, "q3: %s\n", err); q3_gguf_close(m); return 1; }

    if (!q3_loader_load(ctx, m, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    q3_weights weights;
    if (!q3_weights_bind(m, ctx, &cfg, &weights, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Token IDs: --tokens file, or a single-token default for smoke. */
    uint32_t *token_ids = NULL;
    uint32_t token_count = 0;
    if (opt->tokens_path) {
        if (!read_tokens_file(opt->tokens_path, &token_ids, &token_count,
                              err, sizeof(err))) {
            fprintf(stderr, "q3: %s\n", err);
            q3_weights_free(&weights);
            q3_loader_context_destroy(ctx);
            q3_gguf_close(m);
            return 1;
        }
    } else {
        static const uint32_t fallback[] = { 785, 8251, 7578 }; /* "The cat sat" */
        token_ids = (uint32_t *)malloc(sizeof(fallback));
        memcpy(token_ids, fallback, sizeof(fallback));
        token_count = 3;
    }

    /* Candidate token IDs. */
    uint32_t candidates[64];
    uint32_t candidate_count = 0;
    if (opt->candidate_ids) {
        const char *spec = opt->candidate_ids;
        while (*spec && candidate_count < 64) {
            char *end = NULL;
            const long v = strtol(spec, &end, 10);
            if (end == spec || v < 0) break;
            candidates[candidate_count++] = (uint32_t)v;
            spec = (*end == ',') ? end + 1 : end;
        }
        if (candidate_count == 0) {
            fprintf(stderr, "q3: --candidate-token-ids expects a list\n");
            free(token_ids);
            q3_weights_free(&weights);
            q3_loader_context_destroy(ctx);
            q3_gguf_close(m);
            return 2;
        }
    } else {
        static const uint32_t fallback_cand[] = { 16, 17, 18, 19 };
        memcpy(candidates, fallback_cand, sizeof(fallback_cand));
        candidate_count = 4;
    }

    /* Forward runtime: workspace sized for the token count. */
    q3_forward_runtime *rt = q3_forward_create(&weights, &cfg, token_count, 0,
                                               err, sizeof(err));
    if (!rt) {
        fprintf(stderr, "q3: %s\n", err);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* KV cache: capacity = token_count (B1, single forward). */
    q3_kv_cache *kv = q3_kv_cache_init(cfg.num_layers, cfg.num_kv_heads,
                                       cfg.head_dim, token_count,
                                       err, sizeof(err));
    if (!kv) {
        fprintf(stderr, "q3: %s\n", err);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Device buffers for the final hidden and the candidate logits. */
    float *hidden_out = NULL, *logits = NULL;
    if (cudaMalloc(&hidden_out, (size_t)cfg.hidden_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&logits, (size_t)candidate_count * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "q3: cudaMalloc for forward output failed\n");
        q3_kv_cache_destroy(kv);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    q3_forward_stats stats;
    if (!q3_forward_run(rt, token_ids, token_count, kv, 0, hidden_out,
                        &stats, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        cudaFree(logits);
        cudaFree(hidden_out);
        q3_kv_cache_destroy(kv);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    if (!q3_candidate_logits(rt, hidden_out, candidates, candidate_count,
                             logits, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        cudaFree(logits);
        cudaFree(hidden_out);
        q3_kv_cache_destroy(kv);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Diagnostic: check the residual stream for NaN/Inf after the run. */
    if (getenv("Q3_DIAG_HIDDEN")) {
        uint64_t hb = 0;
        const float *h = q3_forward_buffer(rt, Q3_FB_HIDDEN, &hb);
        if (h) {
            float *host_h = (float *)malloc(hb);
            if (host_h) {
                cudaMemcpy(host_h, h, hb, cudaMemcpyDeviceToHost);
                size_t nan_count = 0, inf_count = 0;
                float minv = 1e30f, maxv = -1e30f;
                for (size_t i = 0; i < hb / 4; i++) {
                    const float v = host_h[i];
                    if (isnan(v)) nan_count++;
                    else if (isinf(v)) inf_count++;
                    else { if (v < minv) minv = v; if (v > maxv) maxv = v; }
                }
                fprintf(stderr,
                        "diag: hidden[%zu] nan=%zu inf=%zu min=%.4f max=%.4f\n",
                        hb / 4, nan_count, inf_count, minv, maxv);
                free(host_h);
            }
        }
    }

    /* Diagnostic: per-layer NaN scan to find the first broken boundary.
 * Each layer runs on a FRESH runtime + KV cache (the main run already
 * consumed the shared ones), so the scan is not polluted by stale state. */
    if (getenv("Q3_DIAG_LAYERS")) {
        for (uint32_t l = 0; l < cfg.num_layers; l++) {
            q3_forward_runtime *lrt = q3_forward_create(&weights, &cfg,
                                                        token_count, 0,
                                                        err, sizeof(err));
            if (!lrt) { fprintf(stderr, "diag: create failed: %s\n", err); break; }
            q3_kv_cache *lkv = q3_kv_cache_init(cfg.num_layers, cfg.num_kv_heads,
                                                cfg.head_dim, token_count,
                                                err, sizeof(err));
            if (!lkv) { q3_forward_destroy(lrt); break; }
            q3_forward_stats lstats;
            if (!q3_forward_run_range(lrt, token_ids, token_count, lkv, 0,
                                      0, l + 1, &lstats, err, sizeof(err))) {
                fprintf(stderr, "diag: layer %u failed: %s\n", l, err);
                q3_kv_cache_destroy(lkv);
                q3_forward_destroy(lrt);
                break;
            }
            uint64_t hb = 0;
            const float *h = q3_forward_buffer(lrt, Q3_FB_HIDDEN, &hb);
            float *host_h = (float *)malloc(hb);
            cudaMemcpy(host_h, h, hb, cudaMemcpyDeviceToHost);
            size_t nan_count = 0;
            for (size_t i = 0; i < hb / 4; i++)
                if (isnan(host_h[i])) nan_count++;
            fprintf(stderr, "diag: layer %u nan=%zu\n", l, nan_count);
            free(host_h);
            if (nan_count) {
                /* Inspect intermediate buffers of the broken layer. */
                const int bufs[] = { Q3_FB_NORMED, Q3_FB_Q, Q3_FB_K, Q3_FB_V,
                                     Q3_FB_ATTN, Q3_FB_OPROJ, Q3_FB_GATE,
                                     Q3_FB_UP, Q3_FB_DOWN, Q3_FB_FINAL };
                const char *names[] = { "normed", "q", "k", "v", "attn",
                                        "oproj", "gate", "up", "down", "final" };
                for (int bi = 0; bi < 10; bi++) {
                    uint64_t bb = 0;
                    const float *b = q3_forward_buffer(lrt, bufs[bi], &bb);
                    if (!b) continue;
                    float *host_b = (float *)malloc(bb);
                    cudaMemcpy(host_b, b, bb, cudaMemcpyDeviceToHost);
                    size_t bn = 0, bi2 = 0;
                    float bmin = 1e30f, bmax = -1e30f;
                    for (size_t i = 0; i < bb / 4; i++) {
                        const float v = host_b[i];
                        if (isnan(v)) bn++;
                        else if (isinf(v)) bi2++;
                        else { if (v < bmin) bmin = v; if (v > bmax) bmax = v; }
                    }
                    fprintf(stderr, "diag:   %-7s nan=%zu inf=%zu min=%.4f max=%.4f\n",
                            names[bi], bn, bi2, bmin, bmax);
                    free(host_b);
                }
                q3_kv_cache_destroy(lkv);
                q3_forward_destroy(lrt);
                break;
            }
            q3_kv_cache_destroy(lkv);
            q3_forward_destroy(lrt);
        }
    }

    /* Explicit measurement boundary: the only sync in the whole path. */
    if (!q3_forward_synchronize(rt, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        cudaFree(logits);
        cudaFree(hidden_out);
        q3_kv_cache_destroy(kv);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Read the candidate logits back. */
    float *host_logits = (float *)malloc((size_t)candidate_count * sizeof(float));
    if (!host_logits) {
        fprintf(stderr, "q3: out of memory\n");
        cudaFree(logits);
        cudaFree(hidden_out);
        q3_kv_cache_destroy(kv);
        q3_forward_destroy(rt);
        free(token_ids);
        q3_weights_free(&weights);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }
    cudaMemcpy(host_logits, logits, (size_t)candidate_count * sizeof(float),
               cudaMemcpyDeviceToHost);

    /* Argmax over the candidates. */
    int argmax_idx = 0;
    for (uint32_t i = 1; i < candidate_count; i++)
        if (host_logits[i] > host_logits[argmax_idx]) argmax_idx = (int)i;

    if (opt->json) {
        printf("{\n");
        printf("  \"tokens\": [");
        for (uint32_t i = 0; i < token_count; i++)
            printf("%s%" PRIu32, i ? "," : "", token_ids[i]);
        printf("],\n");
        printf("  \"candidate_token_ids\": [");
        for (uint32_t i = 0; i < candidate_count; i++)
            printf("%s%" PRIu32, i ? "," : "", candidates[i]);
        printf("],\n");
        printf("  \"candidate_logits\": [");
        for (uint32_t i = 0; i < candidate_count; i++)
            printf("%s%.6f", i ? "," : "", host_logits[i]);
        printf("],\n");
        printf("  \"candidate_argmax\": %" PRIu32 ",\n", candidates[argmax_idx]);
        printf("  \"kv_length\": %" PRIu32 ",\n", q3_kv_cache_length(kv));
        printf("  \"timing_ms\": {\n");
        printf("    \"embedding\": %.4f,\n", stats.embedding_ms);
        printf("    \"input_norm\": %.4f,\n", stats.norm_ms);
        printf("    \"q_proj\": %.4f,\n", stats.q_proj_ms);
        printf("    \"k_proj\": %.4f,\n", stats.k_proj_ms);
        printf("    \"v_proj\": %.4f,\n", stats.v_proj_ms);
        printf("    \"qk_norm\": %.4f,\n", stats.qk_norm_ms);
        printf("    \"rope\": %.4f,\n", stats.rope_ms);
        printf("    \"attention\": %.4f,\n", stats.attention_ms);
        printf("    \"o_proj\": %.4f,\n", stats.o_proj_ms);
        printf("    \"mlp_norm\": %.4f,\n", stats.mlp_norm_ms);
        printf("    \"gate_proj\": %.4f,\n", stats.gate_proj_ms);
        printf("    \"up_proj\": %.4f,\n", stats.up_proj_ms);
        printf("    \"activation\": %.4f,\n", stats.activation_ms);
        printf("    \"down_proj\": %.4f,\n", stats.down_proj_ms);
        printf("    \"final_norm\": %.4f,\n", stats.final_norm_ms);
        printf("    \"candidate_head\": %.4f,\n", stats.candidate_head_ms);
        printf("    \"total_forward\": %.4f\n", stats.total_forward_ms);
        printf("  },\n");
        printf("  \"counters\": {\n");
        printf("    \"token_count\": %" PRIu64 ",\n", stats.token_count);
        printf("    \"layers_executed\": %" PRIu64 ",\n", stats.layers_executed);
        printf("    \"cuda_allocations\": %" PRIu64 ",\n", stats.cuda_allocations);
        printf("    \"cuda_frees\": %" PRIu64 ",\n", stats.cuda_frees);
        printf("    \"host_syncs\": %" PRIu64 "\n", stats.host_syncs);
        printf("  }\n");
        printf("}\n");
    } else {
        printf("tokens:          ");
        for (uint32_t i = 0; i < token_count; i++)
            printf("%s%" PRIu32, i ? " " : "", token_ids[i]);
        printf("\n");
        printf("candidate ids:   ");
        for (uint32_t i = 0; i < candidate_count; i++)
            printf("%s%" PRIu32, i ? " " : "", candidates[i]);
        printf("\n");
        printf("candidate logits:");
        for (uint32_t i = 0; i < candidate_count; i++)
            printf(" %s%.4f", i ? "" : "", host_logits[i]);
        printf("\n");
        printf("candidate argmax: %" PRIu32 "\n", candidates[argmax_idx]);
        printf("kv length:        %" PRIu32 "\n", q3_kv_cache_length(kv));
        printf("total forward:    %.4f ms\n", stats.total_forward_ms);
        printf("cuda allocations: %" PRIu64 "\n", stats.cuda_allocations);
        printf("host syncs:       %" PRIu64 "\n", stats.host_syncs);
    }

    free(host_logits);
    cudaFree(logits);
    cudaFree(hidden_out);
    q3_kv_cache_destroy(kv);
    q3_forward_destroy(rt);
    free(token_ids);
    q3_weights_free(&weights);
    q3_loader_context_destroy(ctx);
    q3_gguf_close(m);
    return 0;
}