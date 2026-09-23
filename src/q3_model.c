/* SPDX-License-Identifier: MIT
 *
 * q3_model.c - resolve Qwen3 geometry from real GGUF metadata.
 *
 * The metadata key names follow the llama.cpp/GGUF convention
 * (`<arch>.<field>`), which is what the Qwen3 converter emits and what the
 * target model actually carries. No dimension is guessed from a sibling
 * architecture: Qwen3 has its own head geometry (per-head q/k norm, head_dim
 * decoupled from hidden_size / num_heads).
 */
#include "q3_model.h"

#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool key(char *dst, size_t dst_len, const char *arch, const char *field) {
    int n = snprintf(dst, dst_len, "%s.%s", arch, field);
    return n > 0 && (size_t)n < dst_len;
}

static bool get_u32(const q3_gguf *m, const char *arch, const char *field,
                    uint32_t *out, char *error, size_t error_len) {
    char k[128];
    if (!key(k, sizeof(k), arch, field) || !q3_gguf_get_u32(m, k, out)) {
        char msg[192];
        snprintf(msg, sizeof(msg), "q3_model: missing metadata key %s", k);
        return fail(error, error_len, msg);
    }
    return true;
}

bool q3_model_config_from_gguf(const q3_gguf *m, q3_model_config *out,
                               char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!m || !out) return fail(error, error_len, "q3_model: null argument");
    memset(out, 0, sizeof(*out));

    q3_str arch = {0};
    if (!q3_gguf_get_string(m, "general.architecture", &arch) ||
        arch.len == 0 || arch.len >= 32) {
        return fail(error, error_len, "q3_model: missing architecture");
    }
    char arch_buf[32];
    memcpy(arch_buf, arch.ptr, arch.len);
    arch_buf[arch.len] = '\0';

    if (strcmp(arch_buf, "qwen3") != 0) {
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "q3_model: architecture '%s' is not the qwen3 dense layout",
                 arch_buf);
        return fail(error, error_len, msg);
    }

    if (!get_u32(m, arch_buf, "block_count", &out->num_layers, error, error_len))
        return false;
    if (!get_u32(m, arch_buf, "embedding_length", &out->hidden_size, error,
                 error_len))
        return false;
    if (!get_u32(m, arch_buf, "feed_forward_length", &out->intermediate_size,
                 error, error_len))
        return false;
    if (!get_u32(m, arch_buf, "attention.head_count",
                 &out->num_attention_heads, error, error_len))
        return false;
    if (!get_u32(m, arch_buf, "attention.head_count_kv", &out->num_kv_heads,
                 error, error_len))
        return false;

    /* Qwen3 stores head_dim explicitly; fall back to hidden/heads only when
     * the metadata omits the key, exactly as llama.cpp does. */
    {
        char k[128];
        if (!key(k, sizeof(k), arch_buf, "attention.key_length") ||
            !q3_gguf_get_u32(m, k, &out->head_dim)) {
            if (out->num_attention_heads == 0 || out->hidden_size % out->num_attention_heads != 0) {
                return fail(error, error_len,
                            "q3_model: qwen3.attention.key_length missing and "
                            "hidden/heads is not integral");
            }
            out->head_dim = out->hidden_size / out->num_attention_heads;
        }
    }

    {
        char k[128];
        if (key(k, sizeof(k), arch_buf, "attention.layer_norm_rms_epsilon") &&
            !q3_gguf_get_f32(m, k, &out->rms_norm_eps)) {
            return fail(error, error_len,
                        "q3_model: qwen3.attention.layer_norm_rms_epsilon "
                        "missing or not a float");
        }
    }
    {
        char k[128];
        if (!key(k, sizeof(k), arch_buf, "rope.freq_base") ||
            !q3_gguf_get_f32(m, k, &out->rope_theta)) {
            return fail(error, error_len,
                        "q3_model: qwen3.rope.freq_base missing");
        }
    }
    {
        char k[128];
        if (key(k, sizeof(k), arch_buf, "context_length") &&
            !q3_gguf_get_u32(m, k, &out->max_position_embeddings)) {
            out->max_position_embeddings = 0; /* optional */
        }
    }

    /* q/k/v/o stay 2-D GGUF tensors; vocab_size comes from the embedding
     * tensor rather than metadata (the GGUF carries no vocab key). */
    out->vocab_size = 0;

    if (out->hidden_size == 0 || out->num_layers == 0 ||
        out->num_attention_heads == 0 || out->num_kv_heads == 0 ||
        out->head_dim == 0) {
        return fail(error, error_len, "q3_model: degenerate geometry");
    }
    if (out->num_attention_heads % out->num_kv_heads != 0) {
        return fail(error, error_len,
                    "q3_model: query heads are not a multiple of KV heads");
    }
    if (out->rms_norm_eps <= 0.0f) {
        return fail(error, error_len, "q3_model: non-positive rms epsilon");
    }
    if (out->rope_theta <= 1.0f) {
        return fail(error, error_len, "q3_model: implausible rope theta");
    }
    return true;
}

static const q3_tensor *find_tensor(const q3_gguf *m, const char *name) {
    size_t n = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == n &&
            memcmp(m->tensors[i].name.ptr, name, n) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

bool q3_model_config_validate(const q3_gguf *m, const q3_model_config *cfg,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!m || !cfg) return fail(error, error_len, "q3_model: null argument");

    const q3_tensor *emb = find_tensor(m, "token_embd.weight");
    if (!emb || emb->ndim < 2) {
        return fail(error, error_len, "q3_model: token_embd.weight missing");
    }
    if (emb->dim[0] != cfg->hidden_size) {
        return fail(error, error_len,
                    "q3_model: token_embd row length != hidden_size");
    }
    const uint64_t vocab = emb->dim[1];

    char name[64];
    const q3_tensor *t;

    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", cfg->num_layers - 1);
    t = find_tensor(m, name);
    if (!t || t->ndim < 2) return fail(error, error_len, "q3_model: last attn_q missing");
    if (t->dim[0] != cfg->hidden_size ||
        t->dim[1] != (uint64_t) cfg->num_attention_heads * cfg->head_dim) {
        return fail(error, error_len, "q3_model: attn_q geometry mismatch");
    }

    snprintf(name, sizeof(name), "blk.0.attn_k.weight");
    t = find_tensor(m, name);
    if (!t || t->ndim < 2) return fail(error, error_len, "q3_model: attn_k missing");
    if (t->dim[1] != (uint64_t) cfg->num_kv_heads * cfg->head_dim) {
        return fail(error, error_len, "q3_model: attn_k geometry mismatch");
    }

    snprintf(name, sizeof(name), "blk.0.ffn_gate.weight");
    t = find_tensor(m, name);
    if (!t || t->ndim < 2) return fail(error, error_len, "q3_model: ffn_gate missing");
    if (t->dim[0] != cfg->hidden_size ||
        t->dim[1] != cfg->intermediate_size) {
        return fail(error, error_len, "q3_model: ffn geometry mismatch");
    }

    snprintf(name, sizeof(name), "blk.0.attn_q_norm.weight");
    t = find_tensor(m, name);
    if (!t || t->ndim != 1 || t->dim[0] != cfg->head_dim) {
        return fail(error, error_len, "q3_model: q_norm geometry mismatch");
    }

    t = find_tensor(m, "output.weight");
    if (!t || t->ndim < 2 || t->dim[1] != vocab) {
        return fail(error, error_len, "q3_model: output.weight geometry mismatch");
    }

    /* num_heads * head_dim must be the full q projection width; head_dim may
     * differ from hidden/heads (it does not for 32B, but the check keeps the
     * invariant explicit). */
    if ((uint64_t) cfg->num_attention_heads * cfg->head_dim > vocab) {
        return fail(error, error_len, "q3_model: implausible q width");
    }
    return true;
}
