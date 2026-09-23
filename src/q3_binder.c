/* SPDX-License-Identifier: MIT
 *
 * q3_binder.c - M2 stable weight binder.
 *
 * Resolves every Qwen3 tensor name in the real inventory exactly once. The
 * resulting descriptors are stable pointers plus explicit geometry and qtype;
 * nothing downstream re-searches the GGUF or the residency table (M2 §7).
 */
#include "q3_binder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GGML tensor type ids as stored in GGUF (subset this runtime dispatches). */
enum {
    Q3_GGML_F32  = 0,
    Q3_GGML_F16  = 1,
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

bool q3_binder_qtype_is_matmul(uint32_t qtype) {
    return qtype == 12 /* Q4_K */ || qtype == 14 /* Q6_K */;
}

bool q3_binder_qtype_is_f32(uint32_t qtype) {
    return qtype == Q3_GGML_F32;
}

/* Locate a GGUF tensor by name and return its index, or SIZE_MAX. */
static size_t find_index(const q3_gguf *m, const char *name) {
    const size_t n = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == n &&
            memcmp(m->tensors[i].name.ptr, name, n) == 0) {
            return (size_t) i;
        }
    }
    return SIZE_MAX;
}

/* Bind a 2-D weight tensor: K = dim[0], N = dim[1], qtype must be a matmul
 * type this runtime supports. */
static bool bind_matrix(const q3_gguf *m, const q3_loader_context *ctx,
                        const char *name, q3_exec_tensor *out,
                        char *error, size_t error_len) {
    memset(out, 0, sizeof(*out));
    const size_t index = find_index(m, name);
    if (index == SIZE_MAX) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' missing", name);
        return fail(error, error_len, msg);
    }
    const q3_tensor *t = &m->tensors[index];
    const q3_resident_tensor *res = q3_loader_tensor(ctx, index);
    if (!res || !res->resident || !res->ptr) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' not resident", name);
        return fail(error, error_len, msg);
    }
    if (t->ndim != 2) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' is not 2-D", name);
        return fail(error, error_len, msg);
    }
    if (!q3_binder_qtype_is_matmul(t->type)) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "q3_binder: tensor '%s' qtype %u unsupported (Q4_K/Q6_K only)",
                 name, t->type);
        return fail(error, error_len, msg);
    }
    if (t->dim[0] == 0 || t->dim[1] == 0 ||
        t->dim[0] > UINT32_MAX || t->dim[1] > UINT32_MAX) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' degenerate", name);
        return fail(error, error_len, msg);
    }
    out->index = (uint32_t) index;
    out->qtype = t->type;
    out->name = t->name_buf;
    out->device = res->ptr;
    out->k = (uint32_t) t->dim[0];
    out->n = (uint32_t) t->dim[1];
    out->elements = (uint32_t) t->elements;
    out->is_1d = false;
    out->resident = true;
    return true;
}

/* Bind a 1-D f32 norm tensor. */
static bool bind_vector(const q3_gguf *m, const q3_loader_context *ctx,
                        const char *name, uint32_t expected, q3_exec_tensor *out,
                        char *error, size_t error_len) {
    memset(out, 0, sizeof(*out));
    const size_t index = find_index(m, name);
    if (index == SIZE_MAX) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' missing", name);
        return fail(error, error_len, msg);
    }
    const q3_tensor *t = &m->tensors[index];
    const q3_resident_tensor *res = q3_loader_tensor(ctx, index);
    if (!res || !res->resident || !res->ptr) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' not resident", name);
        return fail(error, error_len, msg);
    }
    if (!q3_binder_qtype_is_f32(t->type)) {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "q3_binder: tensor '%s' qtype %u is not F32", name, t->type);
        return fail(error, error_len, msg);
    }
    if (t->ndim != 1 || t->dim[0] != expected) {
        char msg[160];
        snprintf(msg, sizeof(msg), "q3_binder: tensor '%s' wrong length", name);
        return fail(error, error_len, msg);
    }
    out->index = (uint32_t) index;
    out->qtype = t->type;
    out->name = t->name_buf;
    out->device = res->ptr;
    out->k = 0;
    out->n = 0;
    out->elements = (uint32_t) t->dim[0];
    out->is_1d = true;
    out->resident = true;
    return true;
}

bool q3_weights_bind(const q3_gguf *m, const q3_loader_context *ctx,
                     const q3_model_config *cfg, q3_weights *out,
                     char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!m || !ctx || !cfg || !out)
        return fail(error, error_len, "q3_binder: null argument");
    memset(out, 0, sizeof(*out));

    out->layer = (q3_layer_weights *) calloc(cfg->num_layers,
                                             sizeof(q3_layer_weights));
    if (!out->layer)
        return fail(error, error_len, "q3_binder: layer array allocation failed");

    if (!bind_matrix(m, ctx, "token_embd.weight", &out->embedding,
                     error, error_len))
        goto fail;
    if (!bind_vector(m, ctx, "output_norm.weight", cfg->hidden_size,
                     &out->final_norm, error, error_len))
        goto fail;
    if (!bind_matrix(m, ctx, "output.weight", &out->output, error, error_len))
        goto fail;

    for (uint32_t l = 0; l < cfg->num_layers; l++) {
        q3_layer_weights *w = &out->layer[l];
        char name[64];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        if (!bind_vector(m, ctx, name, cfg->hidden_size, &w->input_norm,
                         error, error_len))
            goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
        if (!bind_matrix(m, ctx, name, &w->q_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", l);
        if (!bind_vector(m, ctx, name, cfg->head_dim, &w->q_norm,
                         error, error_len))
            goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
        if (!bind_matrix(m, ctx, name, &w->k_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", l);
        if (!bind_vector(m, ctx, name, cfg->head_dim, &w->k_norm,
                         error, error_len))
            goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        if (!bind_matrix(m, ctx, name, &w->v_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        if (!bind_matrix(m, ctx, name, &w->o_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        if (!bind_vector(m, ctx, name, cfg->hidden_size, &w->post_attn_norm,
                         error, error_len))
            goto fail;

        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
        if (!bind_matrix(m, ctx, name, &w->gate_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        if (!bind_matrix(m, ctx, name, &w->up_proj, error, error_len)) goto fail;

        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        if (!bind_matrix(m, ctx, name, &w->down_proj, error, error_len)) goto fail;
    }

    /* Geometry cross-checks against the resolved config. */
    {
        const uint32_t q_width = cfg->num_attention_heads * cfg->head_dim;
        const uint32_t kv_width = cfg->num_kv_heads * cfg->head_dim;
        for (uint32_t l = 0; l < cfg->num_layers; l++) {
            const q3_layer_weights *w = &out->layer[l];
            if (w->q_proj.k != cfg->hidden_size || w->q_proj.n != q_width ||
                w->k_proj.k != cfg->hidden_size || w->k_proj.n != kv_width ||
                w->v_proj.k != cfg->hidden_size || w->v_proj.n != kv_width ||
                w->o_proj.k != q_width || w->o_proj.n != cfg->hidden_size ||
                w->gate_proj.k != cfg->hidden_size ||
                w->gate_proj.n != cfg->intermediate_size ||
                w->up_proj.n != cfg->intermediate_size ||
                w->down_proj.k != cfg->intermediate_size ||
                w->down_proj.n != cfg->hidden_size) {
                char msg[96];
                snprintf(msg, sizeof(msg),
                         "q3_binder: layer %u geometry mismatch", l);
                return fail(error, error_len, msg);
            }
        }
        if (out->embedding.k != cfg->hidden_size ||
            out->output.k != cfg->hidden_size) {
            return fail(error, error_len, "q3_binder: embedding/output geometry mismatch");
        }
    }
    return true;

fail:
    q3_weights_free(out);
    return false;
}

void q3_weights_free(q3_weights *weights) {
    if (!weights) return;
    free(weights->layer);
    weights->layer = NULL;
}
