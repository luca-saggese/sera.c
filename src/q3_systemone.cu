/* SPDX-License-Identifier: MIT
 *
 * q3_systemone.cu - the single runtime boundary behind the System One server
 * (docs/M4.md §5, §12-§17).
 *
 * One HTTP request becomes exactly one prefix prefill plus one packed suffix
 * job on the single GPU worker:
 *
 *     state text -> tokens -> one shared prefix prefill (sealed, immutable)
 *     question i -> one suffix branch i
 *     all branches -> ONE packed forward -> candidate logits [B, K]
 *     logits -> typed answers (choice / score / noul)
 *
 * Correctness of the logits themselves is deferred (M3 §71); this file only
 * builds the boundary. Nothing here is an optimization.
 */
extern "C" {
#include "hd_json.h"
}

#include "q3_systemone.h"

extern "C" {
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model.h"
#include "q3_binder.h"
#include "q3_model_loader_cuda.h"
#include "q3_forward.h"
#include "q3_tokenizer.h"
}

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

static char g_last_error[512] = {0};

static void so_set_error(char *error, size_t error_len, const char *fmt, ...) {
    if (!error || error_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_len, fmt, ap);
    va_end(ap);
    snprintf(g_last_error, sizeof(g_last_error), "%s", error);
}

const char *q3_last_error(void) { return g_last_error; }

const char *q3_model_last_error(void) { return g_last_error; }

/* ------------------------------------------------------------------ */
/* The one central prompt builder (M4 §12)                             */
/* ------------------------------------------------------------------ */

static char *so_strdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

/* The suffix is the bare question text (M3 §56): the candidates are scored as
 * the NEXT token, so they must not appear in the suffix. */
static char *so_mask_to_space(const char *s) {
    if (!s) return so_strdup("");
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    for (size_t i = 0; i < n; i++) p[i] = (s[i] == '\0') ? ' ' : s[i];
    p[n] = '\0';
    /* replace every "[MASK]" with a single space, in place */
    char *w = p;
    const char *r = p;
    while (*r) {
        if (r[0] == '[' && strncmp(r, "[MASK]", 6) == 0) { *w++ = ' '; r += 6; }
        else *w++ = *r++;
    }
    *w = '\0';
    return p;
}

static void so_free_strv(char **v, int n) {
    if (!v) return;
    for (int i = 0; i < n; i++) free(v[i]);
    free(v);
}

extern "C" bool q3_build_decision_suffix(q3_qtype qtype,
                                         const char *instructions,
                                         const hd_json *criteria,
                                         char **suffix_out, int *n_options_out,
                                         char ***keys_out, char ***labels_out,
                                         char ***texts_out, char *error,
                                         size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!suffix_out || !n_options_out || !keys_out || !labels_out || !texts_out) {
        so_set_error(error, error_len, "q3_build_decision_suffix: bad argument");
        return false;
    }
    *suffix_out = NULL;
    *n_options_out = 0;
    *keys_out = NULL;
    *labels_out = NULL;
    *texts_out = NULL;

    char *suffix = so_mask_to_space(instructions ? instructions : "");
    if (!suffix) {
        so_set_error(error, error_len, "out of memory");
        return false;
    }

    int n = 0;
    if (qtype == Q3_QTYPE_CHOICE) {
        if (!criteria || criteria->type != HD_JSON_OBJECT) {
            so_set_error(error, error_len,
                         "choice criteria must be an object");
            free(suffix);
            return false;
        }
        n = (int)criteria->u.object.count;
        if (n < 2 || n > Q3_MAX_CANDIDATES) {
            so_set_error(error, error_len,
                         "choice criteria must have 2..%d entries",
                         (int)Q3_MAX_CANDIDATES);
            free(suffix);
            return false;
        }
    } else if (qtype == Q3_QTYPE_SCORE) {
        if (!criteria || criteria->type != HD_JSON_ARRAY) {
            so_set_error(error, error_len, "score criteria must be an array");
            free(suffix);
            return false;
        }
        n = (int)criteria->u.array.count;
        if (n < 2 || n > Q3_MAX_CANDIDATES) {
            so_set_error(error, error_len,
                         "score criteria must have 2..%d entries",
                         (int)Q3_MAX_CANDIDATES);
            free(suffix);
            return false;
        }
    } else {
        n = 2; /* false, true */
    }

    char **keys = (char **)calloc((size_t)n, sizeof(char *));
    char **labels = (char **)calloc((size_t)n, sizeof(char *));
    char **texts = (char **)calloc((size_t)n, sizeof(char *));
    if (!keys || !labels || !texts) {
        so_free_strv(keys, n);
        so_free_strv(labels, n);
        so_free_strv(texts, n);
        free(suffix);
        so_set_error(error, error_len, "out of memory");
        return false;
    }

    bool ok = true;
    if (qtype == Q3_QTYPE_CHOICE) {
        for (int i = 0; i < n && ok; i++) {
            const char *k = criteria->u.object.keys[i];
            const hd_json *v = criteria->u.object.values[i];
            const char *t = hd_json_string(v);
            if (!t || t[0] == '\0') t = k; /* a bare key scores the key text */
            keys[i] = so_strdup(k);
            labels[i] = so_strdup(t);
            texts[i] = so_strdup(t);
            ok = keys[i] && labels[i] && texts[i];
        }
    } else if (qtype == Q3_QTYPE_SCORE) {
        char buf[32];
        for (int i = 0; i < n && ok; i++) {
            const char *t = hd_json_string(criteria->u.array.items[i]);
            if (!t) t = "";
            snprintf(buf, sizeof(buf), "%d", i);
            keys[i] = so_strdup(buf);
            labels[i] = so_strdup(t);
            texts[i] = so_strdup(t);
            ok = keys[i] && labels[i] && texts[i];
        }
    } else {
        const char *f = "no, the statement does not hold";
        const char *t = "yes, the statement holds";
        if (criteria && criteria->type == HD_JSON_OBJECT) {
            const hd_json *fv = hd_json_get(criteria, "false");
            const hd_json *tv = hd_json_get(criteria, "true");
            const char *fs = hd_json_string(fv);
            const char *ts = hd_json_string(tv);
            /* A criteria value overrides the default label; an empty string
             * would leave the option with no token to score. */
            if (fs && fs[0]) f = fs;
            if (ts && ts[0]) t = ts;
        }
        keys[0] = so_strdup("false"); labels[0] = so_strdup(f); texts[0] = so_strdup(f);
        keys[1] = so_strdup("true");  labels[1] = so_strdup(t); texts[1] = so_strdup(t);
        ok = keys[0] && labels[0] && texts[0] && keys[1] && labels[1] && texts[1];
    }

    if (!ok) {
        so_free_strv(keys, n);
        so_free_strv(labels, n);
        so_free_strv(texts, n);
        free(suffix);
        so_set_error(error, error_len, "out of memory");
        return false;
    }

    /* v4 numeric protocol: numbered options plus an explicit ANSWER: cue, so
     * the scored candidate is the option number and not the option text. */
    {
        size_t cap = strlen(suffix) + 64;
        for (int i = 0; i < n; i++) cap += strlen(labels[i]) + 24;
        char *full = (char *)malloc(cap);
        if (!full) {
            so_free_strv(keys, n);
            so_free_strv(labels, n);
            so_free_strv(texts, n);
            free(suffix);
            so_set_error(error, error_len, "out of memory");
            return false;
        }
        size_t used = 0;
        used += (size_t)snprintf(full + used, cap - used, "QUESTION:\n%s\n\nOPTIONS:\n",
                                 suffix);
        for (int i = 0; i < n; i++)
            used += (size_t)snprintf(full + used, cap - used, "%d. %s\n",
                                     i + 1, labels[i]);
        snprintf(full + used, cap - used, "\nANSWER: ");
        free(suffix);
        suffix = full;
    }

    *suffix_out = suffix;
    *n_options_out = n;
    *keys_out = keys;
    *labels_out = labels;
    *texts_out = texts;
    return true;
}

/* ------------------------------------------------------------------ */
/* Model lifecycle                                                     */
/* ------------------------------------------------------------------ */

q3_status q3_model_load(const char *model_path, int device_id, q3_model *out) {
    char err[512] = {0};
    if (!model_path || !out) return Q3_ERR_IO;
    memset(out, 0, sizeof(*out));
    out->device = device_id;
    out->path = so_strdup(model_path);
    if (!out->path) return Q3_ERR_OOM;

    if (q3_cuda_init() != 0) {
        so_set_error(out->error, sizeof(out->error), "CUDA initialization failed");
        return Q3_ERR_RUNTIME;
    }

    q3_gguf *m = q3_gguf_open(model_path, err, sizeof(err));
    if (!m) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        return Q3_ERR_IO;
    }
    out->gguf = m;

    q3_model_config *cfg = (q3_model_config *)calloc(1, sizeof(*cfg));
    if (!cfg) { so_set_error(out->error, sizeof(out->error), "out of memory"); return Q3_ERR_OOM; }
    out->config = cfg;
    if (!q3_model_config_from_gguf(m, cfg, err, sizeof(err)) ||
        !q3_model_config_validate(m, cfg, err, sizeof(err))) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        return Q3_ERR_MISMATCH;
    }

    q3_loader_context *ctx = q3_loader_context_create(err, sizeof(err));
    if (!ctx || !q3_loader_load(ctx, m, err, sizeof(err))) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        if (ctx) q3_loader_context_destroy(ctx);
        return Q3_ERR_IO;
    }
    out->loader = ctx;

    q3_weights *weights = (q3_weights *)calloc(1, sizeof(*weights));
    if (!weights) { so_set_error(out->error, sizeof(out->error), "out of memory"); return Q3_ERR_OOM; }
    out->weights = weights;
    if (!q3_weights_bind(m, ctx, cfg, weights, err, sizeof(err))) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        return Q3_ERR_MISMATCH;
    }

    /* The runtime covers both the prefix prefill and the largest packed suffix
     * batch; it is created once, at load, and never resized (§14). */
    uint32_t max_tokens = Q3_SYSTEMONE_PREFIX_CAPACITY;
    if (Q3_SYSTEMONE_MAX_PACKED > max_tokens) max_tokens = Q3_SYSTEMONE_MAX_PACKED;

    q3_forward_runtime *rt =
        q3_forward_create(weights, cfg, max_tokens, device_id, err, sizeof(err));
    if (!rt) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        return Q3_ERR_RUNTIME;
    }
    out->runtime = rt;

    q3_tokenizer *tok = (q3_tokenizer *)calloc(1, sizeof(*tok));
    if (!tok) { so_set_error(out->error, sizeof(out->error), "out of memory"); return Q3_ERR_OOM; }
    out->tokenizer = tok;
    if (!q3_tokenizer_init_gguf(tok, m, err, sizeof(err))) {
        so_set_error(out->error, sizeof(out->error), "%s", err);
        return Q3_ERR_MANIFEST;
    }

    out->ready = true;
    return Q3_OK;
}

void q3_model_free(q3_model *model) {
    if (!model) return;
    if (model->tokenizer) {
        q3_tokenizer_destroy((q3_tokenizer *)model->tokenizer);
        free(model->tokenizer);
    }
    if (model->runtime) q3_forward_destroy((q3_forward_runtime *)model->runtime);
    if (model->weights) {
        q3_weights_free((q3_weights *)model->weights);
        free(model->weights);
    }
    if (model->loader) q3_loader_context_destroy((q3_loader_context *)model->loader);
    if (model->config) free(model->config);
    if (model->gguf) q3_gguf_close((q3_gguf *)model->gguf);
    free(model->path);
    memset(model, 0, sizeof(*model));
}

void q3_systemone_results_free(q3_systemone_result *results, int count) {
    if (!results) return;
    for (int i = 0; i < count; i++) {
        so_free_strv(results[i].option_keys, results[i].n_options);
        so_free_strv(results[i].option_labels, results[i].n_options);
        free(results[i].probs);
        free(results[i].raw_logits);
    }
    free(results);
}

/* ------------------------------------------------------------------ */
/* The runtime entry point (M4 §5, §14)                                */
/* ------------------------------------------------------------------ */

static bool so_json_dump(const hd_json *v, char *buf, size_t cap, size_t *used) {
    switch (v->type) {
    case HD_JSON_NULL:
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "null")) <= cap;
    case HD_JSON_BOOL:
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "%s",
                                          v->u.boolean ? "true" : "false")) <= cap;
    case HD_JSON_INT:
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "%lld",
                                          (long long)v->u.integer)) <= cap;
    case HD_JSON_DOUBLE:
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "%.17g",
                                          v->u.number)) <= cap;
    case HD_JSON_STRING: {
        const char *s = v->u.string ? v->u.string : "";
        if (*used + 2 >= cap) return false;
        buf[(*used)++] = '"';
        for (const char *p = s; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c == '"' || c == '\\') {
                if (*used + 2 >= cap) return false;
                buf[(*used)++] = '\\';
                buf[(*used)++] = (char)c;
            } else if (c < 0x20) {
                if (*used + 6 >= cap) return false;
                *used += (size_t)snprintf(buf + *used, cap - *used, "\\u%04x", c);
            } else {
                if (*used + 1 >= cap) return false;
                buf[(*used)++] = (char)c;
            }
        }
        if (*used + 1 >= cap) return false;
        buf[(*used)++] = '"';
        return true;
    }
    case HD_JSON_ARRAY:
        if ((*used += (size_t)snprintf(buf + *used, cap - *used, "[")) > cap) return false;
        for (size_t i = 0; i < v->u.array.count; i++) {
            if (i && (*used += (size_t)snprintf(buf + *used, cap - *used, ",")) > cap) return false;
            if (!so_json_dump(v->u.array.items[i], buf, cap, used)) return false;
        }
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "]")) <= cap;
    case HD_JSON_OBJECT:
        if ((*used += (size_t)snprintf(buf + *used, cap - *used, "{")) > cap) return false;
        for (size_t i = 0; i < v->u.object.count; i++) {
            if (i && (*used += (size_t)snprintf(buf + *used, cap - *used, ",")) > cap) return false;
            if ((*used += (size_t)snprintf(buf + *used, cap - *used, "\"%s\":",
                                           v->u.object.keys[i])) > cap) return false;
            if (!so_json_dump(v->u.object.values[i], buf, cap, used)) return false;
        }
        return (*used += (size_t)snprintf(buf + *used, cap - *used, "}")) <= cap;
    }
    return false;
}

static bool so_encode(q3_tokenizer *tok, const char *text, uint32_t **ids,
                      uint32_t *count, char *err, size_t err_len) {
    q3_token_batch b;
    memset(&b, 0, sizeof(b));
    if (!q3_tokenizer_encode(tok, text, false, &b, err, err_len)) return false;
    *ids = b.tokens;
    *count = b.token_count;
    return true;
}

static bool so_softmax(const float *logits, int n, float *probs) {
    if (n <= 0) return false;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { probs[i] = (float)exp((double)(logits[i] - mx)); sum += probs[i]; }
    if (!(sum > 0.0) || !isfinite(sum)) return false;
    for (int i = 0; i < n; i++) probs[i] = (float)(probs[i] / sum);
    return true;
}

q3_status q3_systemone_run(q3_model *model, const hd_json *request,
                           q3_systemone_result **results, int *count,
                           long *input_tokens) {
    char err[512] = {0};
    if (results) *results = NULL;
    if (count) *count = 0;
    if (input_tokens) *input_tokens = 0;
    if (!model || !model->ready || !request || !results || !count) {
        so_set_error(err, sizeof(err), "runtime is not ready");
        return Q3_ERR_RUNTIME;
    }

    q3_forward_runtime *rt = (q3_forward_runtime *)model->runtime;
    q3_tokenizer *tok = (q3_tokenizer *)model->tokenizer;

    const hd_json *state = hd_json_get(request, "state");
    const hd_json *questions = hd_json_get(request, "questions");
    if (!state || state->type == HD_JSON_NULL) {
        so_set_error(err, sizeof(err), "request has no state");
        return Q3_ERR_PARSE;
    }
    if (!questions || questions->type != HD_JSON_OBJECT ||
        questions->u.object.count == 0) {
        so_set_error(err, sizeof(err), "request has no questions");
        return Q3_ERR_PARSE;
    }
    const int B = (int)questions->u.object.count;
    if (B > Q3_INFER_MAX_Q) {
        so_set_error(err, sizeof(err), "at most %d questions per request",
                     (int)Q3_INFER_MAX_Q);
        return Q3_ERR_UNSUPPORTED;
    }

    /* ---- state text: a string passes through, a container becomes JSON ---- */
    char *state_text = NULL;
    if (state->type == HD_JSON_STRING) {
        state_text = so_strdup(state->u.string ? state->u.string : "");
    } else {
        size_t cap = 1 << 16;
        state_text = (char *)malloc(cap);
        size_t used = 0;
        if (!state_text || !so_json_dump(state, state_text, cap, &used)) {
            free(state_text);
            so_set_error(err, sizeof(err), "state is too large");
            return Q3_ERR_PARSE;
        }
        state_text[used] = '\0';
    }
    if (!state_text) { so_set_error(err, sizeof(err), "out of memory"); return Q3_ERR_OOM; }

    /* ---- one prefix per request (§14) ---- */
    uint32_t *state_ids = NULL;
    uint32_t state_count = 0;
    if (!so_encode(tok, state_text, &state_ids, &state_count, err, sizeof(err))) {
        free(state_text);
        so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
        return Q3_ERR_RUNTIME;
    }
    free(state_text);
    if (state_count == 0 || state_count > Q3_SYSTEMONE_PREFIX_CAPACITY) {
        free(state_ids);
        so_set_error(err, sizeof(err), "state has %u tokens; limit is %u",
                     state_count, Q3_SYSTEMONE_PREFIX_CAPACITY);
        return Q3_ERR_UNSUPPORTED;
    }

    q3_prefix_kv *prefix =
        q3_prefix_create(rt, state_count, err, sizeof(err));
    if (!prefix) {
        free(state_ids);
        so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
        return Q3_ERR_RUNTIME;
    }
    q3_branch_set *set =
        q3_branch_set_create(prefix, Q3_INFER_MAX_Q, Q3_SYSTEMONE_MAX_SUFFIX,
                             err, sizeof(err));
    if (!set) {
        q3_prefix_destroy(prefix);
        free(state_ids);
        so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
        return Q3_ERR_RUNTIME;
    }

    if (!q3_prefix_prefill(prefix, state_ids, state_count, NULL, err, sizeof(err)) ||
        !q3_prefix_seal(prefix, err, sizeof(err))) {
        q3_branch_set_destroy(set);
        q3_prefix_destroy(prefix);
        free(state_ids);
        so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
        return Q3_ERR_RUNTIME;
    }
    free(state_ids);

    q3_systemone_result *res =
        (q3_systemone_result *)calloc((size_t)B, sizeof(*res));
    q3_decision_item *items = (q3_decision_item *)calloc((size_t)B, sizeof(*items));
    q3_decision_result *dres = (q3_decision_result *)calloc((size_t)B, sizeof(*dres));
    int32_t *handles = (int32_t *)calloc((size_t)B, sizeof(int32_t));
    uint32_t *padded = NULL;
    uint32_t *suffix_ids[Q3_INFER_MAX_Q];
    uint32_t *cand_ids[Q3_INFER_MAX_Q];
    uint32_t suffix_counts[Q3_INFER_MAX_Q];
    uint32_t cand_counts[Q3_INFER_MAX_Q];
    memset(suffix_ids, 0, sizeof(suffix_ids));
    memset(cand_ids, 0, sizeof(cand_ids));
    memset(suffix_counts, 0, sizeof(suffix_counts));
    memset(cand_counts, 0, sizeof(cand_counts));

    q3_status st = Q3_OK;
    long total_tokens = (long)state_count;
    uint32_t K = 0;

    if (!res || !items || !dres || !handles) {
        so_set_error(err, sizeof(err), "out of memory");
        st = Q3_ERR_OOM;
        goto cleanup;
    }

    /* ---- build every branch suffix, then collate (M4 §12-§13) ---- */
    for (int i = 0; i < B && st == Q3_OK; i++) {
        const char *key = questions->u.object.keys[i];
        const hd_json *qdef = questions->u.object.values[i];
        if (!qdef || qdef->type != HD_JSON_OBJECT) {
            so_set_error(err, sizeof(err), "question %s is not an object", key);
            st = Q3_ERR_PARSE;
            break;
        }
        const char *type = hd_json_string(hd_json_get(qdef, "type"));
        q3_qtype qt;
        if (type && strcmp(type, "choice") == 0) qt = Q3_QTYPE_CHOICE;
        else if (type && strcmp(type, "score") == 0) qt = Q3_QTYPE_SCORE;
        else if (type && strcmp(type, "noul") == 0) qt = Q3_QTYPE_NOUL;
        else {
            so_set_error(err, sizeof(err), "question %s has unknown type", key);
            st = Q3_ERR_UNSUPPORTED;
            break;
        }
        const char *ins = hd_json_string(hd_json_get(qdef, "instructions"));

        char *suffix = NULL;
        int n_opts = 0;
        char **keys = NULL, **labels = NULL, **texts = NULL;
        if (!q3_build_decision_suffix(qt, ins, hd_json_get(qdef, "criteria"),
                                      &suffix, &n_opts, &keys, &labels, &texts,
                                      err, sizeof(err))) {
            so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
            st = Q3_ERR_PARSE;
            break;
        }

        uint32_t *sids = NULL, *cids = NULL;
        uint32_t scount = 0;
        if (!so_encode(tok, suffix, &sids, &scount, err, sizeof(err))) {
            so_free_strv(keys, n_opts); so_free_strv(labels, n_opts);
            so_free_strv(texts, n_opts); free(suffix);
            so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
            st = Q3_ERR_RUNTIME;
            break;
        }
        free(suffix);
        if (scount == 0 || scount > Q3_SYSTEMONE_MAX_SUFFIX) {
            free(sids);
            so_free_strv(keys, n_opts); so_free_strv(labels, n_opts);
            so_free_strv(texts, n_opts);
            so_set_error(err, sizeof(err),
                         "question %s suffix has %u tokens; limit is %u", key,
                         scount, (uint32_t)Q3_SYSTEMONE_MAX_SUFFIX);
            st = Q3_ERR_UNSUPPORTED;
            break;
        }

        cids = (uint32_t *)calloc((size_t)n_opts, sizeof(uint32_t));
        if (!cids) {
            free(sids);
            so_free_strv(keys, n_opts); so_free_strv(labels, n_opts);
            so_free_strv(texts, n_opts);
            so_set_error(err, sizeof(err), "out of memory");
            st = Q3_ERR_OOM;
            break;
        }
        for (int k = 0; k < n_opts; k++) {
            char num[16];
            snprintf(num, sizeof(num), "%d", k + 1);
            uint32_t *oid = NULL;
            uint32_t oc = 0;
            if (!so_encode(tok, num, &oid, &oc, err, sizeof(err)) || oc != 1) {
                free(oid);
                free(cids); free(sids);
                so_free_strv(keys, n_opts); so_free_strv(labels, n_opts);
                so_free_strv(texts, n_opts);
                so_set_error(err, sizeof(err),
                             "question %s option %d number did not tokenize",
                             key, k);
                st = Q3_ERR_RUNTIME;
                break;
            }
            cids[k] = oid[0]; /* v4: the candidate is the option number */
            free(oid);
        }
        if (st != Q3_OK) break;

        res[i].qtype = qt;
        res[i].n_options = n_opts;
        res[i].option_keys = keys;
        res[i].option_labels = labels;
        res[i].choice = -1;
        so_free_strv(texts, n_opts);

        suffix_ids[i] = sids;
        suffix_counts[i] = scount;
        cand_ids[i] = cids;
        cand_counts[i] = (uint32_t)n_opts;
        total_tokens += (long)scount;
        if ((uint32_t)n_opts > K) K = (uint32_t)n_opts;
    }

    if (st != Q3_OK) goto cleanup;

    if ((uint32_t)total_tokens > Q3_SYSTEMONE_MAX_PACKED) {
        so_set_error(err, sizeof(err), "packed suffix is %ld tokens; limit is %u",
                     total_tokens, (uint32_t)Q3_SYSTEMONE_MAX_PACKED);
        st = Q3_ERR_UNSUPPORTED;
        goto cleanup;
    }

    /* The batched LM head writes a dense [B, K] tile, so every item in one
     * batch uses the same K; shorter lists repeat their last id (M3 §15). */
    padded = (uint32_t *)malloc((size_t)B * K * sizeof(uint32_t));
    if (!padded) {
        so_set_error(err, sizeof(err), "out of memory");
        st = Q3_ERR_OOM;
        goto cleanup;
    }

    for (int i = 0; i < B; i++) {
        handles[i] = q3_branch_acquire(set, err, sizeof(err));
        if (handles[i] < 0) {
            so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
            st = Q3_ERR_RUNTIME;
            goto cleanup;
        }
        for (uint32_t k = 0; k < K; k++)
            padded[(size_t)i * K + k] =
                cand_ids[i][k < cand_counts[i] ? k : cand_counts[i] - 1];
        items[i].branch = (uint32_t)handles[i];
        items[i].tokens = suffix_ids[i];
        items[i].token_count = suffix_counts[i];
        items[i].candidate_ids = padded + (size_t)i * K;
        items[i].candidate_count = K;
    }

    q3_decision_batch batch;
    batch.items = items;
    batch.count = (uint32_t)B;
    if (!q3_decide_run(rt, prefix, set, &batch, dres, NULL, err, sizeof(err))) {
        so_set_error(g_last_error, sizeof(g_last_error), "%s", err);
        st = Q3_ERR_RUNTIME;
        goto cleanup;
    }

    /* ---- typed answers (§8-§11) ---- */
    for (int i = 0; i < B; i++) {
        const int n = res[i].n_options;
        res[i].probs = (float *)calloc((size_t)n, sizeof(float));
        res[i].raw_logits = (float *)calloc((size_t)n, sizeof(float));
        if (!res[i].probs || !res[i].raw_logits) {
            so_set_error(err, sizeof(err), "out of memory");
            st = Q3_ERR_OOM;
            goto cleanup;
        }
        for (int k = 0; k < n; k++) res[i].raw_logits[k] = dres[i].logits[k];
        if (!so_softmax(res[i].raw_logits, n, res[i].probs)) {
            so_set_error(err, sizeof(err), "question %d produced no distribution", i);
            st = Q3_ERR_RUNTIME;
            goto cleanup;
        }

        int best = 0;
        for (int k = 1; k < n; k++)
            if (res[i].raw_logits[k] > res[i].raw_logits[best]) best = k;
        res[i].choice = best;

        if (res[i].qtype == Q3_QTYPE_SCORE) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += (double)k * res[i].probs[k];
            res[i].score = s;
        } else if (res[i].qtype == Q3_QTYPE_NOUL) {
            res[i].noul = res[i].probs[1];
        }

        /* confidence = 1 - H/log K: concentration only, NOT calibrated. */
        double h = 0.0;
        for (int k = 0; k < n; k++) {
            double p = res[i].probs[k];
            if (p > 0.0) h -= p * log(p);
        }
        res[i].confidence = (n > 1) ? 1.0 - h / log((double)n) : 1.0;
    }

cleanup:
    if (padded) free(padded);
    for (int i = 0; i < B; i++) {
        free(suffix_ids[i]);
        free(cand_ids[i]);
    }
    free(handles);
    free(items);
    free(dres);
    if (set) q3_branch_set_destroy(set);
    if (prefix) q3_prefix_destroy(prefix);

    if (st != Q3_OK) {
        if (res) q3_systemone_results_free(res, B);
        return st;
    }
    *results = res;
    *count = B;
    if (input_tokens) *input_tokens = total_tokens;
    return Q3_OK;
}
