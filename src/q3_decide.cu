/* SPDX-License-Identifier: MIT
 *
 * q3_decide.cu - M3 shared-prefix decision batching.
 *
 *     prefill once  ->  immutable shared prefix KV
 *     B private suffix branches  ->  one packed batched forward
 *     ->  candidate logits [B, K]
 *
 * The prefix is written once by an ordinary M2 forward against an aliasing
 * cache header and then sealed: it is never copied, never re-materialised as
 * activation rows, and never freed until the prefix object dies (§8-§11).
 * Branches are descriptors into one suffix arena allocated once (§10, §23):
 * acquire/reset/release touch no GPU memory and allocate nothing.
 */

#include "q3_decide.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Shared immutable prefix KV (M3 §8-§11)
 * ========================================================================= */

struct q3_prefix_kv {
    q3_forward_runtime *runtime; /* borrowed, prefill only */
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_size;
    uint32_t max_tokens; /* runtime token capacity, cached */
    uint32_t capacity;
    uint32_t token_count;
    float *k; /* [num_layers][capacity][num_kv_heads][head_dim] */
    float *v;
    bool sealed;

    uint64_t allocations;
    uint64_t copy_calls;
    uint64_t copy_bytes;
};

static void decide_set_error(char *error, size_t error_len, const char *msg) {
    if (error && error_len) {
        error[0] = '\0';
        snprintf(error, error_len, "%s", msg);
    }
}

size_t q3_kv_bytes_for_tokens(uint32_t tokens, uint32_t num_layers,
                              uint32_t num_kv_heads, uint32_t head_dim) {
    const size_t row = (size_t)num_kv_heads * head_dim;
    /* Two slabs (K and V), each [layers][tokens][kv_heads][head_dim]. */
    return 2 * (size_t)num_layers * tokens * row * sizeof(float);
}

extern "C" q3_prefix_kv *q3_prefix_create(q3_forward_runtime *runtime,
                                          uint32_t capacity_tokens, char *error,
                                          size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || capacity_tokens == 0) {
        decide_set_error(error, error_len, "q3_prefix_create: invalid argument");
        return NULL;
    }
    uint32_t layers = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0;
    uint32_t max_tokens = 0;
    q3_forward_geometry(runtime, &layers, &hidden, &heads, &kv_heads, &head_dim,
                        &max_tokens);
    if (layers == 0 || kv_heads == 0 || head_dim == 0) {
        decide_set_error(error, error_len, "q3_prefix_create: bad geometry");
        return NULL;
    }
    q3_prefix_kv *prefix = (q3_prefix_kv *)calloc(1, sizeof(*prefix));
    if (!prefix) {
        decide_set_error(error, error_len, "q3_prefix_create: allocation failed");
        return NULL;
    }
    prefix->runtime = runtime;
    prefix->num_layers = layers;
    prefix->num_kv_heads = kv_heads;
    prefix->head_dim = head_dim;
    prefix->hidden_size = hidden;
    prefix->max_tokens = max_tokens;
    prefix->capacity = capacity_tokens;

    /* Exactly two device allocations for the whole immutable prefix. */
    const size_t slab = (size_t)layers * capacity_tokens * kv_heads * head_dim;
    if (cudaMalloc(&prefix->k, slab * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&prefix->v, slab * sizeof(float)) != cudaSuccess) {
        if (prefix->k) cudaFree(prefix->k);
        free(prefix);
        decide_set_error(error, error_len, "q3_prefix_create: cudaMalloc failed");
        return NULL;
    }
    prefix->allocations = 2;
    return prefix;
}

extern "C" bool q3_prefix_prefill(q3_prefix_kv *prefix,
                                  const uint32_t *token_ids_host,
                                  uint32_t token_count, q3_forward_stats *stats,
                                  char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!prefix || !token_ids_host || token_count == 0) {
        decide_set_error(error, error_len, "q3_prefix_prefill: invalid argument");
        return false;
    }
    if (prefix->sealed) {
        decide_set_error(error, error_len,
                         "q3_prefix_prefill: prefix already sealed");
        return false;
    }
    if (token_count > prefix->capacity) {
        decide_set_error(error, error_len,
                         "q3_prefix_prefill: token_count exceeds capacity");
        return false;
    }
    if (token_count > prefix->max_tokens) {
        decide_set_error(error, error_len,
                         "q3_prefix_prefill: token_count exceeds runtime capacity");
        return false;
    }

    /* Direct-write prefill (M3 §9): the ordinary M2 layer path appends roped
     * K/V through this borrowing header, so the rows land in the prefix slabs
     * with no temporary KV, no adapter and no copy. */
    q3_kv_cache *alias = q3_kv_cache_alias(
        prefix->k, prefix->v, prefix->num_layers, prefix->num_kv_heads,
        prefix->head_dim, prefix->capacity, 0, error, error_len);
    if (!alias) return false;

    const bool ok = q3_forward_run_range(
        prefix->runtime, token_ids_host, token_count, alias, 0, 0,
        prefix->num_layers, stats, error, error_len);
    q3_kv_cache_destroy(alias);
    if (!ok) return false;

    prefix->token_count = token_count;
    return true;
}

extern "C" bool q3_prefix_seal(q3_prefix_kv *prefix, char *error,
                               size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!prefix) {
        decide_set_error(error, error_len, "q3_prefix_seal: null prefix");
        return false;
    }
    if (prefix->token_count == 0) {
        decide_set_error(error, error_len, "q3_prefix_seal: prefix is empty");
        return false;
    }
    prefix->sealed = true;
    return true;
}

extern "C" void q3_prefix_destroy(q3_prefix_kv *prefix) {
    if (!prefix) return;
    if (prefix->k) cudaFree(prefix->k);
    if (prefix->v) cudaFree(prefix->v);
    free(prefix);
}

extern "C" uint32_t q3_prefix_token_count(const q3_prefix_kv *prefix) {
    return prefix ? prefix->token_count : 0;
}

extern "C" size_t q3_prefix_kv_bytes(const q3_prefix_kv *prefix) {
    if (!prefix) return 0;
    return q3_kv_bytes_for_tokens(prefix->capacity, prefix->num_layers,
                                  prefix->num_kv_heads, prefix->head_dim);
}

extern "C" uint64_t q3_prefix_copy_bytes(const q3_prefix_kv *prefix) {
    return prefix ? prefix->copy_bytes : 0;
}

extern "C" uint64_t q3_prefix_allocations(const q3_prefix_kv *prefix) {
    return prefix ? prefix->allocations : 0;
}

/* =========================================================================
 * Branch set (M3 §10). One suffix arena plus one decision scratch, both
 * allocated once; the branches themselves are descriptors only.
 * ========================================================================= */

struct q3_branch_set {
    q3_prefix_kv *prefix;
    uint32_t max_branches;
    uint32_t max_suffix;
    uint32_t num_layers;
    uint32_t row; /* num_kv_heads * head_dim */
    uint32_t hidden_size;
    uint32_t max_candidates;
    uint32_t live;
    bool *in_use;

    float *suffix_k; /* [max_branches][num_layers][max_suffix][row] */
    float *suffix_v;
    /* Decision scratch: [max_branches][hidden] then
     * [max_branches][max_candidates]. */
    float *hidden_per_branch;
    float *logits_dev;
    uint64_t allocations;
};

extern "C" q3_branch_set *q3_branch_set_create(q3_prefix_kv *prefix,
                                               uint32_t max_branches,
                                               uint32_t max_suffix_tokens,
                                               char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!prefix || max_branches == 0 || max_suffix_tokens == 0 ||
        max_branches > Q3_MAX_BRANCHES) {
        decide_set_error(error, error_len,
                         "q3_branch_set_create: invalid argument");
        return NULL;
    }
    q3_branch_set *set = (q3_branch_set *)calloc(1, sizeof(*set));
    if (!set) {
        decide_set_error(error, error_len,
                         "q3_branch_set_create: allocation failed");
        return NULL;
    }
    set->prefix = prefix;
    set->max_branches = max_branches;
    set->max_suffix = max_suffix_tokens;
    set->num_layers = prefix->num_layers;
    set->row = prefix->num_kv_heads * prefix->head_dim;
    set->hidden_size = prefix->hidden_size;
    set->max_candidates = Q3_MAX_CANDIDATES;
    set->in_use = (bool *)calloc(max_branches, sizeof(bool));
    if (!set->in_use) {
        free(set);
        decide_set_error(error, error_len,
                         "q3_branch_set_create: allocation failed");
        return NULL;
    }

    const size_t per_branch =
        (size_t)set->num_layers * max_suffix_tokens * set->row;
    const size_t slab = per_branch * max_branches;
    if (cudaMalloc(&set->suffix_k, slab * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&set->suffix_v, slab * sizeof(float)) != cudaSuccess) {
        if (set->suffix_k) cudaFree(set->suffix_k);
        free(set->in_use);
        free(set);
        decide_set_error(error, error_len,
                         "q3_branch_set_create: cudaMalloc failed");
        return NULL;
    }
    set->allocations = 2;

    /* Decision scratch (one allocation for both halves). */
    const size_t hidden_elems = (size_t)max_branches * set->hidden_size;
    const size_t logits_elems = (size_t)max_branches * set->max_candidates;
    const size_t scratch_bytes = (hidden_elems + logits_elems) * sizeof(float);
    if (cudaMalloc(&set->hidden_per_branch, scratch_bytes) != cudaSuccess) {
        cudaFree(set->suffix_k);
        cudaFree(set->suffix_v);
        free(set->in_use);
        free(set);
        decide_set_error(error, error_len,
                         "q3_branch_set_create: cudaMalloc failed");
        return NULL;
    }
    set->logits_dev = set->hidden_per_branch + hidden_elems;
    set->allocations++;
    return set;
}

extern "C" int32_t q3_branch_acquire(q3_branch_set *set, char *error,
                                     size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!set) {
        decide_set_error(error, error_len, "q3_branch_acquire: null set");
        return -1;
    }
    for (uint32_t b = 0; b < set->max_branches; b++) {
        if (!set->in_use[b]) {
            set->in_use[b] = true;
            set->live++;
            return (int32_t)b;
        }
    }
    decide_set_error(error, error_len,
                     "q3_branch_acquire: branch limit exceeded");
    return -1;
}

extern "C" void q3_branch_reset(q3_branch_set *set, int32_t branch) {
    /* Descriptor-only: a branch's suffix length lives in the caller's decision
     * item, so nothing on the device has to be cleared (M3 §10). */
    (void)set;
    (void)branch;
}

extern "C" void q3_branch_release(q3_branch_set *set, int32_t branch) {
    if (!set || branch < 0 || (uint32_t)branch >= set->max_branches) return;
    if (set->in_use[branch]) {
        set->in_use[branch] = false;
        set->live--;
    }
}

extern "C" void q3_branch_set_destroy(q3_branch_set *set) {
    if (!set) return;
    if (set->suffix_k) cudaFree(set->suffix_k);
    if (set->suffix_v) cudaFree(set->suffix_v);
    if (set->hidden_per_branch) cudaFree(set->hidden_per_branch);
    free(set->in_use);
    free(set);
}

extern "C" size_t q3_branch_set_kv_bytes(const q3_branch_set *set) {
    if (!set) return 0;
    return q3_kv_bytes_for_tokens(set->max_suffix * set->max_branches,
                                  set->num_layers, set->prefix->num_kv_heads,
                                  set->prefix->head_dim);
}

extern "C" uint32_t q3_branch_set_max_branches(const q3_branch_set *set) {
    return set ? set->max_branches : 0;
}

extern "C" uint32_t q3_branch_set_max_suffix(const q3_branch_set *set) {
    return set ? set->max_suffix : 0;
}

extern "C" float *q3_branch_set_suffix_k(const q3_branch_set *set) {
    return set ? set->suffix_k : NULL;
}

extern "C" float *q3_branch_set_suffix_v(const q3_branch_set *set) {
    return set ? set->suffix_v : NULL;
}

extern "C" uint32_t q3_branch_set_num_layers(const q3_branch_set *set) {
    return set ? set->num_layers : 0;
}

extern "C" uint32_t q3_branch_set_row(const q3_branch_set *set) {
    return set ? set->row : 0;
}

/* =========================================================================
 * The one entry point (M3 §17-§27)
 * ========================================================================= */

extern "C" void q3_decide_result_softmax(q3_decision_result *result) {
    if (!result || result->candidate_count == 0) return;
    const uint32_t n = result->candidate_count;
    float max = result->logits[0];
    for (uint32_t i = 1; i < n; i++)
        if (result->logits[i] > max) max = result->logits[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float e = expf(result->logits[i] - max);
        result->probabilities[i] = e;
        sum += e;
    }
    if (sum > 0.0f)
        for (uint32_t i = 0; i < n; i++) result->probabilities[i] /= sum;
}

extern "C" bool q3_decide_run(q3_forward_runtime *runtime, q3_prefix_kv *prefix,
                              q3_branch_set *set,
                              const q3_decision_batch *batch,
                              q3_decision_result *results, q3_decide_stats *stats,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !prefix || !set || !batch || !batch->items || !results) {
        decide_set_error(error, error_len, "q3_decide_run: invalid argument");
        return false;
    }
    const uint32_t B = batch->count;
    if (B == 0 || B > Q3_MAX_BRANCHES || B > set->max_branches) {
        decide_set_error(error, error_len, "q3_decide_run: bad branch count");
        return false;
    }
    if (!prefix->sealed) {
        decide_set_error(error, error_len, "q3_decide_run: prefix not sealed");
        return false;
    }
    if (prefix->runtime != runtime) {
        decide_set_error(error, error_len,
                         "q3_decide_run: prefix belongs to another runtime");
        return false;
    }

    /* One uniform candidate count for the whole batch (§15, §26): the batched
     * LM head writes a dense [B, K] tile. */
    const uint32_t K = batch->items[0].candidate_count;
    if (K == 0 || K > Q3_MAX_CANDIDATES) {
        decide_set_error(error, error_len, "q3_decide_run: bad candidate count");
        return false;
    }

    uint32_t packed = 0;
    for (uint32_t b = 0; b < B; b++) {
        const q3_decision_item *item = &batch->items[b];
        if (!item->tokens || item->token_count == 0 ||
            item->token_count > set->max_suffix || !item->candidate_ids ||
            item->candidate_count != K) {
            decide_set_error(error, error_len,
                             "q3_decide_run: invalid decision item");
            return false;
        }
        packed += item->token_count;
    }
    if (packed > prefix->max_tokens) {
        decide_set_error(error, error_len,
                         "q3_decide_run: packed batch exceeds runtime capacity");
        return false;
    }

    /* --- Host-side packing (M3 §16, §18). Tight: no padding rows, so there is
     * no "fake" token to exclude anywhere downstream. ------------------- */
    uint32_t *token_ids = (uint32_t *)malloc((size_t)packed * sizeof(uint32_t));
    uint32_t *token_positions =
        (uint32_t *)malloc((size_t)packed * sizeof(uint32_t));
    uint32_t *token_branch =
        (uint32_t *)malloc((size_t)packed * sizeof(uint32_t));
    uint32_t *token_local_pos =
        (uint32_t *)malloc((size_t)packed * sizeof(uint32_t));
    uint32_t *branch_offsets =
        (uint32_t *)malloc((size_t)(B + 1) * sizeof(uint32_t));
    uint32_t *candidate_ids =
        (uint32_t *)malloc((size_t)B * K * sizeof(uint32_t));
    float *host_logits = (float *)malloc((size_t)B * K * sizeof(float));
    if (!token_ids || !token_positions || !token_branch || !token_local_pos ||
        !branch_offsets || !candidate_ids || !host_logits) {
        decide_set_error(error, error_len, "q3_decide_run: allocation failed");
        goto fail;
    }

    {
        uint32_t cursor = 0;
        for (uint32_t b = 0; b < B; b++) {
            const q3_decision_item *item = &batch->items[b];
            if (item->branch >= set->max_branches) {
                decide_set_error(error, error_len,
                                 "q3_decide_run: invalid branch handle");
                goto fail;
            }
            branch_offsets[b] = cursor;
            for (uint32_t j = 0; j < item->token_count; j++) {
                const uint32_t t = cursor + j;
                token_ids[t] = item->tokens[j];
                token_branch[t] = (uint32_t)item->branch;
                token_local_pos[t] = j;
                /* Absolute position: the suffix continues right after the
                 * shared prefix, identically for every branch (M3 §13). */
                token_positions[t] = prefix->token_count + j;
            }
            for (uint32_t k = 0; k < K; k++)
                candidate_ids[b * K + k] = item->candidate_ids[k];
            cursor += item->token_count;
        }
        branch_offsets[B] = cursor;
    }

    q3_pack_view view;
    memset(&view, 0, sizeof(view));
    view.token_positions = token_positions;
    view.token_branch = token_branch;
    view.token_local_pos = token_local_pos;
    view.branch_offsets = branch_offsets;
    view.branch_count = B;
    view.prefix_k = prefix->k;
    view.prefix_v = prefix->v;
    view.prefix_len = prefix->token_count;
    view.prefix_capacity = prefix->capacity;
    view.suffix_k = set->suffix_k;
    view.suffix_v = set->suffix_v;
    view.suffix_capacity = set->max_suffix;
    view.suffix_num_layers = set->num_layers;

    q3_forward_stats fstats;
    memset(&fstats, 0, sizeof(fstats));
    /* The whole suffix forward is one packed pass; the device scratch was
     * allocated once when the branch set was created (§23: no hot cudaMalloc,
     * no per-branch allocation). */
    if (!q3_forward_run_packed(runtime, token_ids, packed, &view,
                               set->hidden_per_branch, &fstats, error,
                               error_len))
        goto fail;

    if (!q3_candidate_logits_batched(runtime, set->hidden_per_branch, B,
                                     candidate_ids, K, set->logits_dev, error,
                                     error_len))
        goto fail;

    /* The one explicit measurement-boundary sync (§43, §44). */
    if (!q3_forward_synchronize(runtime, error, error_len)) goto fail;

    if (cudaMemcpy(host_logits, set->logits_dev, (size_t)B * K * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        decide_set_error(error, error_len, "q3_decide_run: logits copy failed");
        goto fail;
    }

    for (uint32_t b = 0; b < B; b++) {
        q3_decision_result *r = &results[b];
        r->candidate_count = K;
        for (uint32_t k = 0; k < K; k++) r->logits[k] = host_logits[b * K + k];
        q3_decide_result_softmax(r);
        uint32_t best = 0;
        for (uint32_t k = 1; k < K; k++)
            if (r->logits[k] > r->logits[best]) best = k;
        r->predicted_index = best;
    }

    if (stats) {
        memset(stats, 0, sizeof(*stats));
        /* total_forward_ms is recomputed by the boundary sync and therefore
         * already contains the candidate-head phase; subtract it so the two
         * reported numbers stay disjoint. */
        stats->suffix_forward_ms =
            fstats.total_forward_ms - fstats.candidate_head_ms;
        stats->candidate_head_ms = fstats.candidate_head_ms;
        stats->prefix_kv_bytes_physical = q3_prefix_kv_bytes(prefix);
        stats->expected_prefix_kv_bytes = q3_kv_bytes_for_tokens(
            prefix->token_count, prefix->num_layers, prefix->num_kv_heads,
            prefix->head_dim);
        stats->suffix_kv_bytes = q3_branch_set_kv_bytes(set);
        stats->prefix_allocations = prefix->allocations;
        stats->prefix_bytes = q3_prefix_kv_bytes(prefix);
        stats->prefix_copy_calls = prefix->copy_calls;
        stats->prefix_copy_bytes = prefix->copy_bytes;
        stats->suffix_allocations = set->allocations;
        stats->suffix_bytes = q3_branch_set_kv_bytes(set);
        stats->cuda_allocations = 0; /* nothing allocated inside the batch */
        stats->cuda_frees = 0;
        stats->host_syncs = 1; /* the single explicit boundary sync */
        stats->batch = B;
        stats->suffix_tokens_total = packed;
    }

    free(token_ids); free(token_positions); free(token_branch);
    free(token_local_pos); free(branch_offsets); free(candidate_ids);
    free(host_logits);
    return true;

fail:
    free(token_ids); free(token_positions); free(token_branch);
    free(token_local_pos); free(branch_offsets); free(candidate_ids);
    free(host_logits);
    return false;
}
