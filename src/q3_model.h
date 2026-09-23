/* SPDX-License-Identifier: MIT
 *
 * q3_model.h - Qwen3 model-family geometry, resolved once from the real GGUF
 * metadata.
 *
 * M2 needs a single place that turns GGUF metadata into typed dimensions so
 * the binder, the workspace planner and the forward never hardcode sizes.
 * The GGUF file is the source of truth; anything the metadata omits has an
 * explicit fallback that is validated against the tensor inventory, and an
 * incompatible geometry fails loudly instead of being silently patched.
 */
#ifndef Q3_MODEL_H
#define Q3_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t hidden_size;
    uint32_t intermediate_size;

    uint32_t num_layers;

    uint32_t num_attention_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    uint32_t vocab_size;

    float rms_norm_eps;
    float rope_theta;

    uint32_t max_position_embeddings;
} q3_model_config;

/* Resolve the config from GGUF metadata. Returns false with `error` set when
 * the model is not a dense Qwen3 layout this runtime can bind (wrong
 * architecture, missing required keys, inconsistent head geometry). */
bool q3_model_config_from_gguf(const q3_gguf *m, q3_model_config *out,
                               char *error, size_t error_len);

/* Cross-check the resolved config against the real tensor inventory so a
 * mis-resolved head_dim or intermediate size is caught before any CUDA work
 * happens. */
bool q3_model_config_validate(const q3_gguf *m, const q3_model_config *cfg,
                              char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif /* Q3_MODEL_H */
