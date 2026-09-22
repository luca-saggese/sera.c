#ifndef Q3_RESIDENCY_H
#define Q3_RESIDENCY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q3_RESIDENT_ALIGNMENT 256u

typedef struct {
    void *base;
    uint64_t capacity;
    uint64_t used;
    uint32_t tensor_count;
} q3_resident_arena;

typedef struct {
    q3_resident_arena core;
    q3_resident_arena *expert_banks;
    uint32_t expert_bank_count;
    uint64_t non_ple_bytes;
    uint64_t ple_bytes;
    uint64_t aligned_bytes;
} q3_model_residency;

static inline uint64_t q3_align_up_u64(uint64_t value, uint64_t alignment) {
    if (!alignment || value > UINT64_MAX - (alignment - 1u)) return UINT64_MAX;
    return (value + alignment - 1u) / alignment * alignment;
}

void q3_resident_arena_init(q3_resident_arena *arena);
bool q3_resident_arena_reserve(q3_resident_arena *arena, uint64_t bytes,
                                char *error, size_t error_len);
void q3_resident_arena_reset(q3_resident_arena *arena);

bool q3_model_residency_init(q3_model_residency *residency,
                              uint32_t expert_bank_count,
                              char *error, size_t error_len);
void q3_model_residency_destroy(q3_model_residency *residency);
void q3_model_residency_reset(q3_model_residency *residency);

/* Account a tensor without copying or registering its backing GGUF mapping.
 * PLE tensors are deliberately counted separately and never enter an arena. */
bool q3_model_residency_account_tensor(q3_model_residency *residency,
                                        const q3_tensor *tensor,
                                        bool ple, uint32_t expert_bank,
                                        char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
