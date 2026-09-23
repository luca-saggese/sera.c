/* SPDX-License-Identifier: MIT
 *
 * q3_forward.h - M2 Qwen3-32B Q4 one-pass forward (public C ABI).
 *
 *     token IDs -> embedding -> 64 x (attention + MLP) -> final norm
 *               -> candidate-only LM rows -> logits
 *
 * All activation tensors are device-side. The runtime owns one grow-once
 * workspace; after warmup the hot path performs zero cudaMalloc/cudaFree and
 * zero unexpected host synchronizations (M2 §21). Weight descriptors come from
 * the stable binder (q3_binder.h) and are never re-resolved here.
 */
#ifndef Q3_FORWARD_H
#define Q3_FORWARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_binder.h"
#include "q3_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward phase index (M2 §32 timing tree). */
enum q3_forward_phase {
    Q3_FWD_PHASE_EMBED = 0,
    Q3_FWD_PHASE_NORM,
    Q3_FWD_PHASE_Q,
    Q3_FWD_PHASE_K,
    Q3_FWD_PHASE_V,
    Q3_FWD_PHASE_QK_NORM,
    Q3_FWD_PHASE_ROPE,
    Q3_FWD_PHASE_ATTN,
    Q3_FWD_PHASE_OPROJ,
    Q3_FWD_PHASE_MLP_NORM,
    Q3_FWD_PHASE_GATE,
    Q3_FWD_PHASE_UP,
    Q3_FWD_PHASE_ACT,
    Q3_FWD_PHASE_DOWN,
    Q3_FWD_PHASE_FINAL_NORM,
    Q3_FWD_PHASE_CAND,
    Q3_FWD_PHASE_COUNT
};

/* Aggregated timing tree (M2 §32). Layer times are summed over all layers;
 * --trace-layer can re-enable per-layer detail. */
typedef struct {
    double embedding_ms;
    double norm_ms;
    double q_proj_ms;
    double k_proj_ms;
    double v_proj_ms;
    double qk_norm_ms;
    double rope_ms;
    double attention_ms;
    double o_proj_ms;
    double mlp_norm_ms;
    double gate_proj_ms;
    double up_proj_ms;
    double activation_ms;
    double down_proj_ms;
    double final_norm_ms;
    double candidate_head_ms;
    double total_forward_ms;

    uint64_t token_count;
    uint64_t layers_executed;

    uint64_t cuda_allocations;    /* must be 0 during warm forward */
    uint64_t cuda_frees;
    uint64_t host_syncs;          /* must be 0 during warm forward */
} q3_forward_stats;

/* --- KV cache B1 (M2 §17-§18) -------------------------------------- *
 * One contiguous K and V store per layer, layout [capacity][kv_heads][head_dim]
 * in FP32. FP32 is chosen over FP16/BF16 deliberately: it makes the
 * prefix+suffix equivalence check (Test 4) purely about mechanics, and the
 * Q4 weight error dominates activation rounding by orders of magnitude. */
typedef struct q3_kv_cache q3_kv_cache;

q3_kv_cache *q3_kv_cache_init(uint32_t num_layers, uint32_t num_kv_heads,
                              uint32_t head_dim, uint32_t capacity,
                              char *error, size_t error_len);
bool q3_kv_cache_reserve(q3_kv_cache *kv, uint32_t capacity,
                         char *error, size_t error_len);
void q3_kv_cache_reset(q3_kv_cache *kv);
void q3_kv_cache_destroy(q3_kv_cache *kv);
uint32_t q3_kv_cache_length(const q3_kv_cache *kv);

/* --- Forward runtime ----------------------------------------------- */
typedef struct q3_forward_runtime q3_forward_runtime;

q3_forward_runtime *q3_forward_create(const q3_weights *weights,
                                      const q3_model_config *config,
                                      uint32_t max_tokens, int device,
                                      char *error, size_t error_len);
void q3_forward_destroy(q3_forward_runtime *runtime);

/* Run the one-pass forward for `token_count` pre-tokenized IDs (host uint32).
 * `kv` may be NULL for an uncached forward, or a cache whose current length is
 * the number of already-cached positions. `position_base` is the absolute
 * position of the first input token. On success `device_hidden_out` holds the
 * final-norm hidden of the last token as [hidden_size] (last-position-row). */
bool q3_forward_run(q3_forward_runtime *runtime,
                    const uint32_t *token_ids_host, uint32_t token_count,
                    q3_kv_cache *kv, uint32_t position_base,
                    float *device_hidden_out, q3_forward_stats *stats,
                    char *error, size_t error_len);

/* Run only layers [first, last) after the embedding gather. Does not run the
 * final norm. Used by the layer-parity tests to isolate the first broken
 * boundary (M2 §23-§24). */
bool q3_forward_run_range(q3_forward_runtime *runtime,
                          const uint32_t *token_ids_host, uint32_t token_count,
                          q3_kv_cache *kv, uint32_t position_base,
                          uint32_t first_layer, uint32_t last_layer,
                          q3_forward_stats *stats, char *error, size_t error_len);

/* Apply the final RMSNorm to the current hidden stream and write the last
 * token's row into `device_hidden_out` [hidden_size]. */
bool q3_forward_final_norm(q3_forward_runtime *runtime, uint32_t token_count,
                           float *device_hidden_out, q3_forward_stats *stats,
                           char *error, size_t error_len);

/* One decoder layer boundary (M2 §22). `hidden_inout` is [token_count, hidden]
 * device memory, updated in place from the residual stream. */
bool q3_forward_layer(q3_forward_runtime *runtime, uint32_t layer,
                      q3_kv_cache *kv, uint32_t position_base,
                      uint32_t token_count, float *hidden_inout,
                      q3_forward_stats *stats, char *error, size_t error_len);

/* Debug accessor: expose one named internal workspace buffer of the most
 * recent run so the parity tests can compare a boundary against the oracle
 * without the runtime writing files. Returns the device pointer and its byte
 * length, or NULL for an unknown buffer. */
enum q3_forward_buffer {
    Q3_FB_HIDDEN = 0,   /* residual stream [T, hidden]                  */
    Q3_FB_NORMED,       /* input/post-attn norm output [T, hidden]       */
    Q3_FB_Q,            /* q_proj output [T, n_heads*head_dim]           */
    Q3_FB_Q2,           /* q after q_norm + RoPE                         */
    Q3_FB_K,            /* k_proj output                                 */
    Q3_FB_K2,           /* k after k_norm + RoPE                         */
    Q3_FB_V,            /* v_proj output                                 */
    Q3_FB_ATTN,         /* attention output [T, n_heads*head_dim]        */
    Q3_FB_OPROJ,        /* o_proj output [T, hidden]                     */
    Q3_FB_GATE,         /* gate_proj output [T, intermediate]            */
    Q3_FB_UP,           /* up_proj output [T, intermediate]              */
    Q3_FB_DOWN,         /* down_proj output [T, hidden]                  */
    Q3_FB_FINAL,        /* final-norm output [T, hidden]                 */
    Q3_FB_LOGITS,       /* candidate logits [candidate_count]            */
    Q3_FB_COUNT
};

const float *q3_forward_buffer(const q3_forward_runtime *runtime, int which,
                              uint64_t *bytes_out);

/* Round-trip the internal CUDA stream so the caller can read results. This is
 * the explicit measurement-boundary sync; it is never used inside the hot
 * forward path. */
bool q3_forward_synchronize(q3_forward_runtime *runtime, char *error,
                            size_t error_len);

/* Selected-row LM head: dot of the final hidden [hidden] with each candidate
 * row of output.weight. Never touches the full vocabulary. */
bool q3_candidate_logits(q3_forward_runtime *runtime,
                         const float *device_hidden_last,
                         const uint32_t *candidate_ids_host,
                         uint32_t candidate_count, float *device_logits,
                         char *error, size_t error_len);

/* --- Primitive surface (Test 1 — primitive semantics) -------------- *
 * Every function is device-side, takes a void* CUDA stream, and never
 * synchronizes. Exposed so tests can validate them in isolation. */
bool q3_cuda_rms_norm(const float *input, const float *weight, uint32_t rows,
                      uint32_t width, float epsilon, float *output,
                      void *stream, char *error, size_t error_len);
bool q3_cuda_silu_mul(const float *gate, const float *up, float *out,
                      size_t elements, void *stream, char *error, size_t error_len);
bool q3_cuda_residual_add(float *accumulator, const float *addend,
                          size_t elements, void *stream,
                          char *error, size_t error_len);
bool q3_cuda_rope(float *q, float *k, uint32_t tokens, uint32_t n_heads,
                  uint32_t n_kv_heads, uint32_t head_dim, uint32_t position_base,
                  const float *inv_freq, void *stream, char *error, size_t error_len);
bool q3_cuda_attention(const float *q, const float *k, const float *v,
                       uint32_t tokens, uint32_t n_heads, uint32_t n_kv_heads,
                       uint32_t head_dim, uint32_t position_base,
                       q3_kv_cache *kv, uint32_t layer, float scale, float *out,
                       void *stream, char *error, size_t error_len);
bool q3_cuda_embedding(float *hidden, const uint32_t *token_ids,
                       uint32_t token_count, const void *weight, uint32_t qtype,
                       uint32_t hidden_size, uint32_t vocab_size, void *stream,
                       char *error, size_t error_len);

/* --- M3 packed multi-branch surface --------------------------------- *
 * M2's B1 path is untouched: every entry point below is additive. The
 * logical KV a suffix token attends to is
 *     [ shared prefix 0..P-1 | own branch suffix 0..lp ]
 * addressed as two segments with no materialised concatenation and no copy
 * of the prefix (M3 §11, §12, §14). */

#define Q3_PACK_MAX_BRANCHES 32
#define Q3_PACK_MAX_CANDIDATES 8
/* Device staging for the batched candidate head: one row per (branch,
 * candidate) pair (M3 §26). */
#define Q3_CANDIDATE_CAPACITY (Q3_PACK_MAX_BRANCHES * Q3_PACK_MAX_CANDIDATES)

/* Everything the packed layer runner needs to address the two KV segments.
 * Device pointers; the integer arrays are read by the kernels. */
typedef struct {
    const uint32_t *token_positions; /* [T] absolute position of each token   */
    const uint32_t *token_branch;    /* [T] owning branch index               */
    const uint32_t *token_local_pos; /* [T] position inside the branch suffix */
    const uint32_t *branch_offsets;  /* [B+1]; branch_offsets[B] == T         */
    uint32_t branch_count;           /* B                                     */

    const float *prefix_k;           /* [L][prefix_capacity][kv_heads][D]     */
    const float *prefix_v;
    uint32_t prefix_len;             /* P: positions 0..P-1                   */
    uint32_t prefix_capacity;

    float *suffix_k;                 /* [B][L][suffix_capacity][kv_heads][D]  */
    float *suffix_v;
    uint32_t suffix_capacity;        /* max suffix tokens per branch          */
    uint32_t suffix_num_layers;      /* L: per-branch layer stride factor     */
} q3_pack_view;

/* One packed forward: embedding, L layers with two-segment branch-isolated
 * attention, final norm, then each branch's last row gathered into
 * `device_hidden_per_branch` [branch_count][hidden]. Never copies the prefix. */
bool q3_forward_run_packed(q3_forward_runtime *runtime,
                           const uint32_t *token_ids_host, uint32_t token_count,
                           const q3_pack_view *view,
                           float *device_hidden_per_branch,
                           q3_forward_stats *stats, char *error, size_t error_len);

/* Selected-row LM head for a whole decision batch in one launch.
 * `candidate_ids_host` is [branch_count][candidate_count] row-major;
 * `device_logits` is [branch_count][candidate_count]. */
bool q3_candidate_logits_batched(q3_forward_runtime *runtime,
                                 const float *device_hidden_per_branch,
                                 uint32_t branch_count,
                                 const uint32_t *candidate_ids_host,
                                 uint32_t candidate_count, float *device_logits,
                                 char *error, size_t error_len);

/* RoPE with per-token absolute positions (M3 §13). q3_cuda_rope is the
 * identity-mapping special case. */
bool q3_cuda_rope_positions(float *q, float *k, uint32_t tokens,
                            uint32_t n_heads, uint32_t n_kv_heads,
                            uint32_t head_dim, const uint32_t *token_positions,
                            const float *inv_freq, void *stream, char *error,
                            size_t error_len);

/* Branch-isolated causal GQA attention over [prefix | own suffix].
 * `suffix_k`/`suffix_v` are the suffix-arena bases; branch b, layer l,
 * suffix row p sits at
 *     suffix_k + (b*suffix_num_layers + l)*suffix_capacity*row + p*row. */
bool q3_cuda_attention_segmented(const float *q, const float *prefix_k,
                                 const float *prefix_v, uint32_t prefix_len,
                                 uint32_t prefix_capacity, const float *suffix_k,
                                 const float *suffix_v, uint32_t suffix_capacity,
                                 uint32_t suffix_num_layers,
                                 uint32_t suffix_layer,
                                 const uint32_t *token_branch,
                                 const uint32_t *token_local_pos, uint32_t tokens,
                                 uint32_t n_heads, uint32_t n_kv_heads,
                                 uint32_t head_dim, float scale, float *out,
                                 void *stream, char *error, size_t error_len);

/* A q3_kv_cache that borrows slabs it does not own. The prefix prefill uses
 * this so roped K/V land straight in the immutable prefix storage (M3 §9):
 * the M2 append path needs no adapter, and the header is discarded once the
 * prefix is sealed. q3_kv_cache_destroy never frees borrowed slabs. */
q3_kv_cache *q3_kv_cache_alias(float *k, float *v, uint32_t num_layers,
                               uint32_t num_kv_heads, uint32_t head_dim,
                               uint32_t capacity, uint32_t length,
                               char *error, size_t error_len);

/* Geometry of a runtime, so callers outside this translation unit can size
 * their own arenas without reaching into the opaque struct. */
void q3_forward_geometry(const q3_forward_runtime *runtime,
                         uint32_t *num_layers, uint32_t *hidden_size,
                         uint32_t *num_heads, uint32_t *num_kv_heads,
                         uint32_t *head_dim, uint32_t *max_tokens);

/* --- M3 shared-prefix KV and private suffix branches ----------------- *
 * Declared in q3_decide.h: the prefix is written once by an ordinary M2
 * forward and then sealed; each branch owns a private suffix slab inside one
 * arena allocated once (§8-§11). */

#ifdef __cplusplus
}
#endif

#endif /* Q3_FORWARD_H */
