/* SPDX-License-Identifier: MIT
 *
 * q3_binder.h - M2 stable weight binder.
 *
 * Turns GGUF tensor names into stable exec descriptors once, at startup, so
 * the forward hot path never does a name lookup, a tensor search or a
 * residency lookup. Every descriptor points straight at the resident device
 * allocation and carries the real qtype for explicit dispatch (M2 §7-§8).
 */
#ifndef Q3_BINDER_H
#define Q3_BINDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_gguf.h"
#include "q3_model.h"
#include "q3_model_loader_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One fully resolved executable tensor. `device` is the resident device
 * pointer; `k`/`n` are the logical GEMM geometry already disambiguated from
 * the GGUF dim order (GGUF dim[0] is K, dim[1] is N). */
typedef struct {
    uint32_t index;        /* GGUF tensor index (== resident slot) */
    uint32_t qtype;        /* real GGML type; fail loudly if unsupported */
    const char *name;      /* borrowed from the GGUF descriptor table */
    const void *device;    /* resident device pointer, never host */
    uint32_t k;            /* contiguous row length */
    uint32_t n;            /* row count (0 for 1-D norms) */
    uint32_t elements;     /* 1-D element count (norms) */
    bool is_1d;
    bool resident;
} q3_exec_tensor;

typedef struct {
    q3_exec_tensor input_norm;

    q3_exec_tensor q_proj;
    q3_exec_tensor q_norm;

    q3_exec_tensor k_proj;
    q3_exec_tensor k_norm;

    q3_exec_tensor v_proj;
    q3_exec_tensor o_proj;

    q3_exec_tensor post_attn_norm;

    q3_exec_tensor gate_proj;
    q3_exec_tensor up_proj;
    q3_exec_tensor down_proj;
} q3_layer_weights;

typedef struct {
    q3_exec_tensor embedding;

    q3_layer_weights *layer;   /* num_layers entries */

    q3_exec_tensor final_norm;
    q3_exec_tensor output;
} q3_weights;

/* Bind every tensor the forward needs from the real inventory. `m` and `ctx`
 * must outlive `out`. On success `out->layer` is heap-allocated and must be
 * released with q3_weights_free(). Fails loudly on a missing tensor or a qtype
 * the forward cannot dispatch. */
bool q3_weights_bind(const q3_gguf *m, const q3_loader_context *ctx,
                     const q3_model_config *cfg, q3_weights *out,
                     char *error, size_t error_len);

void q3_weights_free(q3_weights *weights);

/* qtype support check used by the binder and by the forward dispatch. */
bool q3_binder_qtype_is_matmul(uint32_t qtype);
bool q3_binder_qtype_is_f32(uint32_t qtype);

#ifdef __cplusplus
}
#endif

#endif /* Q3_BINDER_H */
