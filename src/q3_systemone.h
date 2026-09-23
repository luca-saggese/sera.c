/* SPDX-License-Identifier: MIT
 *
 * q3_systemone.h - the single runtime boundary behind the System One server
 * (docs/M4.md §5, §6-§13).
 *
 * The server never sees layers, KV, Q4, workspaces or attention. It hands the
 * parsed request over and receives typed answers. One request is one prefix
 * prefill and one packed suffix job on the single GPU worker (§14).
 */
#ifndef Q3_SYSTEMONE_H
#define Q3_SYSTEMONE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hd_json.h"
#include "q3_decide.h"

#ifdef __cplusplus
extern "C" {
#endif

/* At most Q3_MAX_BRANCHES questions per request: the packed suffix batch is
 * one branch per question (M4 §17). */
#define Q3_INFER_MAX_Q Q3_MAX_BRANCHES

/* Bounded resident capacities. The runtime is created once, at load, with a
 * capacity that covers the longest prefix prefill and the largest packed
 * suffix batch; a request beyond these limits is refused, never resized. */
#define Q3_SYSTEMONE_PREFIX_CAPACITY 2048u
#define Q3_SYSTEMONE_MAX_SUFFIX Q3_DEFAULT_SUFFIX_CAP
#define Q3_SYSTEMONE_MAX_PACKED \
    ((uint32_t)Q3_INFER_MAX_Q * (uint32_t)Q3_SYSTEMONE_MAX_SUFFIX)

/* Laya status codes, kept verbatim so the ported server code is unchanged. */
typedef enum q3_status {
    Q3_OK = 0,
    Q3_ERR_IO = 1,
    Q3_ERR_PARSE = 2,
    Q3_ERR_PROFILE = 3,
    Q3_ERR_MANIFEST = 4,
    Q3_ERR_MISMATCH = 5,
    Q3_ERR_MISSING = 6,
    Q3_ERR_OOM = 7,
    Q3_ERR_RUNTIME = 8,
    Q3_ERR_UNSUPPORTED = 9,
} q3_status;

/* The three System One question types (M4 §6-§11). */
typedef enum q3_qtype {
    Q3_QTYPE_CHOICE = 0,
    Q3_QTYPE_SCORE = 1,
    Q3_QTYPE_NOUL = 2,
} q3_qtype;

/* One typed answer. `option_keys` is the caller's own key order for `choice`
 * (so the reported `choice` is the original key) and "0".."N-1" otherwise. */
typedef struct q3_systemone_result {
    q3_qtype qtype;
    int n_options;
    char **option_keys;   /* n_options, owned */
    char **option_labels; /* n_options, owned */
    int choice;           /* argmax index, or -1 */
    double score;         /* expected value over the levels (score type) */
    double noul;          /* P(true) (noul type) */
    double confidence;    /* 1 - H/log K, concentration only, not calibrated */
    float *probs;         /* n_options, owned */
    float *raw_logits;    /* n_options, owned */
} q3_systemone_result;

/* The resident handle. The server only ever stores it by value and passes it
 * back, so it is declared here with opaque members: no CUDA, GGUF, binder or
 * tokenizer header leaks into the HTTP server translation unit. */
typedef struct q3_model {
    char *path;
    int device;
    bool ready;
    void *gguf;      /* q3_gguf *            */
    void *loader;    /* q3_loader_context *  */
    void *weights;   /* q3_weights *         */
    void *config;    /* q3_model_config *    */
    void *runtime;   /* q3_forward_runtime * */
    void *tokenizer; /* q3_tokenizer *       */
    void *prefix;    /* q3_prefix_kv *       */
    void *branch_set;/* q3_branch_set *      */
    char error[512];
} q3_model;

/* Load the resident model once, at startup. Never called per request. */
q3_status q3_model_load(const char *model_path, int device_id, q3_model *out);
void q3_model_free(q3_model *model);
const char *q3_model_last_error(void);
const char *q3_last_error(void);

/* The single runtime entry point (M4 §5). `request` is the normalized
 * {"state":...,"questions":{...}} object. On success `*results` is a
 * caller-owned array of `*count` answers. */
q3_status q3_systemone_run(q3_model *model, const hd_json *request,
                           q3_systemone_result **results, int *count,
                           long *input_tokens);
void q3_systemone_results_free(q3_systemone_result *results, int count);

/* The one central prompt builder (M4 §12): structured question in, bare
 * suffix text plus the option key/label/score texts out. */
bool q3_build_decision_suffix(q3_qtype qtype, const char *instructions,
                              const hd_json *criteria, char **suffix_out,
                              int *n_options_out, char ***keys_out,
                              char ***labels_out, char ***texts_out,
                              char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif /* Q3_SYSTEMONE_H */