/* q3_decide.h — M3 shared-prefix decision batching.
 *
 * The M3 shape: prefill a shared immutable prefix KV once, run B private
 * suffix branches in one packed batched forward, and read candidate logits
 * [B, K]. The prefix is never copied (§11); each branch owns a private suffix
 * slab inside one arena allocated once (§10, §23).
 *
 * This header is the public ABI. `q3_forward.h` carries the lower-level
 * packed-forward surface this builds on.
 */
#ifndef Q3_DECIDE_H
#define Q3_DECIDE_H

#include <stddef.h>
#include <stdint.h>

#include "q3_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q3_MAX_BRANCHES Q3_PACK_MAX_BRANCHES
#define Q3_MAX_CANDIDATES Q3_PACK_MAX_CANDIDATES
#define Q3_DEFAULT_SUFFIX_CAP 256

/* One decision question: a private suffix plus the candidate rows to score. */
typedef struct {
    uint32_t branch;             /* handle from q3_branch_acquire            */
    const uint32_t *tokens;      /* suffix token IDs, host                   */
    uint32_t token_count;
    const uint32_t *candidate_ids; /* [candidate_count], host                */
    uint32_t candidate_count;
} q3_decision_item;

typedef struct {
    const q3_decision_item *items;
    uint32_t count;              /* B: 1..Q3_MAX_BRANCHES                    */
} q3_decision_batch;

/* Result per branch. Only the candidates are scored — never the full
 * 151936-wide vocabulary (§26). */
typedef struct {
    uint32_t predicted_index;    /* argmax over this branch's candidates     */
    uint32_t candidate_count;
    float logits[Q3_MAX_CANDIDATES];
    float probabilities[Q3_MAX_CANDIDATES];
} q3_decision_result;

typedef struct q3_decide_stats {
    double prefix_prefill_ms;
    double branch_create_ms;
    double suffix_forward_ms;
    double candidate_head_ms;
    double cold_total_ms;
    double warm_total_ms;

    double suffix_questions_per_s;
    double warm_ms_per_question;

    size_t prefix_kv_bytes_physical;
    size_t expected_prefix_kv_bytes;
    size_t suffix_kv_bytes;
    size_t workspace_bytes;
    size_t peak_cuda_bytes;

    uint64_t prefix_allocations;
    uint64_t prefix_bytes;
    uint64_t prefix_copy_calls;
    uint64_t prefix_copy_bytes;
    uint64_t suffix_allocations;
    uint64_t suffix_bytes;

    uint64_t cuda_allocations;   /* must be 0 during a warm batch */
    uint64_t cuda_frees;
    uint64_t host_syncs;         /* must be 0 during a warm batch */

    uint64_t kernel_launches_per_batch;
    uint64_t kernel_launches_per_question;

    uint32_t batch;
    uint32_t suffix_tokens_total;

    uint32_t accuracy_matches;
    uint32_t accuracy_total;
} q3_decide_stats;

/* Physical KV bytes for `tokens` positions across the model geometry. */
size_t q3_kv_bytes_for_tokens(uint32_t tokens, uint32_t num_layers,
                              uint32_t num_kv_heads, uint32_t head_dim);

/* --- Prefix lifecycle (§8-§11) -------------------------------------- */

/* Shared immutable prefix KV: written once by an ordinary M2 forward, then
 * sealed. Two cudaMalloc total (K and V). */
typedef struct q3_prefix_kv q3_prefix_kv;
typedef struct q3_branch_set q3_branch_set;

q3_prefix_kv *q3_prefix_create(q3_forward_runtime *runtime,
                               uint32_t capacity_tokens, char *error,
                               size_t error_len);

/* Prefill: one ordinary M2 forward whose roped K/V land straight in the
 * prefix slabs through an aliasing cache header. No temporary KV, no copy. */
bool q3_prefix_prefill(q3_prefix_kv *prefix, const uint32_t *token_ids_host,
                       uint32_t token_count, q3_forward_stats *stats,
                       char *error, size_t error_len);

/* Seal: the prefix becomes immutable and is shared by every branch. */
bool q3_prefix_seal(q3_prefix_kv *prefix, char *error, size_t error_len);
void q3_prefix_destroy(q3_prefix_kv *prefix);
uint32_t q3_prefix_token_count(const q3_prefix_kv *prefix);
size_t q3_prefix_kv_bytes(const q3_prefix_kv *prefix);
/* Bytes copied out of the immutable prefix. Must stay 0 (§11). */
uint64_t q3_prefix_copy_bytes(const q3_prefix_kv *prefix);
uint64_t q3_prefix_allocations(const q3_prefix_kv *prefix);

/* --- Branch set (§10) ---------------------------------------------- */

/* One suffix arena sized `max_branches * num_layers * max_suffix * row`, two
 * cudaMalloc. Acquire/reset/release are descriptor-only: no GPU work, no
 * malloc. */
q3_branch_set *q3_branch_set_create(q3_prefix_kv *prefix,
                                    uint32_t max_branches,
                                    uint32_t max_suffix_tokens, char *error,
                                    size_t error_len);
/* Returns a branch slot, or -1 on failure. Descriptor only: no GPU work. */
int32_t q3_branch_acquire(q3_branch_set *set, char *error, size_t error_len);
void q3_branch_reset(q3_branch_set *set, int32_t branch);
void q3_branch_release(q3_branch_set *set, int32_t branch);
void q3_branch_set_destroy(q3_branch_set *set);
size_t q3_branch_set_kv_bytes(const q3_branch_set *set);
uint32_t q3_branch_set_max_branches(const q3_branch_set *set);
uint32_t q3_branch_set_max_suffix(const q3_branch_set *set);
/* Suffix-arena K/V bases, for building a q3_pack_view. */
float *q3_branch_set_suffix_k(const q3_branch_set *set);
float *q3_branch_set_suffix_v(const q3_branch_set *set);
uint32_t q3_branch_set_num_layers(const q3_branch_set *set);
uint32_t q3_branch_set_row(const q3_branch_set *set);

/* --- The one entry point (§17-§21, §27) ----------------------------- */

/* Run one decision batch: pack every item's suffix into a single ragged
 * forward, attend over [shared prefix | own suffix], score the candidates and
 * fill `results[0..batch->count-1]`. Requires the runtime's token capacity to
 * be at least the packed total. */
bool q3_decide_run(q3_forward_runtime *runtime, q3_prefix_kv *prefix,
                   q3_branch_set *set, const q3_decision_batch *batch,
                   q3_decision_result *results, q3_decide_stats *stats,
                   char *error, size_t error_len);

/* Softmax over one result's candidate logits (small, host-side). */
void q3_decide_result_softmax(q3_decision_result *result);

#ifdef __cplusplus
}
#endif

#endif /* Q3_DECIDE_H */
