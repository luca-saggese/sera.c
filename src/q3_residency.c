#include "q3_residency.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

void q3_resident_arena_init(q3_resident_arena *arena) {
    if (arena) memset(arena, 0, sizeof(*arena));
}

bool q3_resident_arena_reserve(q3_resident_arena *arena, uint64_t bytes,
                                char *error, size_t error_len) {
    if (!arena) {
        set_error(error, error_len, "residency arena is null");
        return false;
    }
    if (bytes > UINT64_MAX - arena->used) {
        set_error(error, error_len, "residency arena size overflow");
        return false;
    }
    uint64_t end = q3_align_up_u64(arena->used + bytes,
                                    Q3_RESIDENT_ALIGNMENT);
    if (end == UINT64_MAX) {
        set_error(error, error_len, "residency arena alignment overflow");
        return false;
    }
    arena->used = end;
    if (arena->used > arena->capacity) arena->capacity = arena->used;
    if (arena->tensor_count == UINT32_MAX) {
        set_error(error, error_len, "residency tensor count overflow");
        return false;
    }
    arena->tensor_count++;
    return true;
}

void q3_resident_arena_reset(q3_resident_arena *arena) {
    if (!arena) return;
    free(arena->base);
    q3_resident_arena_init(arena);
}

bool q3_model_residency_init(q3_model_residency *residency,
                              uint32_t expert_bank_count,
                              char *error, size_t error_len) {
    if (!residency) {
        set_error(error, error_len, "residency is null");
        return false;
    }
    memset(residency, 0, sizeof(*residency));
    q3_resident_arena_init(&residency->core);
    if (expert_bank_count) {
        residency->expert_banks = calloc(expert_bank_count,
                                         sizeof(*residency->expert_banks));
        if (!residency->expert_banks) {
            set_error(error, error_len, "expert residency table allocation failed");
            return false;
        }
        residency->expert_bank_count = expert_bank_count;
        for (uint32_t i = 0; i < expert_bank_count; ++i)
            q3_resident_arena_init(&residency->expert_banks[i]);
    }
    return true;
}

void q3_model_residency_destroy(q3_model_residency *residency) {
    if (!residency) return;
    q3_resident_arena_reset(&residency->core);
    for (uint32_t i = 0; i < residency->expert_bank_count; ++i)
        q3_resident_arena_reset(&residency->expert_banks[i]);
    free(residency->expert_banks);
    memset(residency, 0, sizeof(*residency));
}

void q3_model_residency_reset(q3_model_residency *residency) {
    if (!residency) return;
    q3_model_residency_destroy(residency);
}

bool q3_model_residency_account_tensor(q3_model_residency *residency,
                                        const q3_tensor *tensor,
                                        bool ple, uint32_t expert_bank,
                                        char *error, size_t error_len) {
    if (!residency || !tensor) {
        set_error(error, error_len, "invalid residency tensor");
        return false;
    }
    if (ple) {
        if (tensor->bytes > UINT64_MAX - residency->ple_bytes) {
            set_error(error, error_len, "PLE residency byte count overflow");
            return false;
        }
        residency->ple_bytes += tensor->bytes;
        return true;
    }
    q3_resident_arena *arena = &residency->core;
    if (expert_bank != UINT32_MAX) {
        if (expert_bank >= residency->expert_bank_count) {
            set_error(error, error_len, "expert residency bank is out of range");
            return false;
        }
        arena = &residency->expert_banks[expert_bank];
    }
    if (!q3_resident_arena_reserve(arena, tensor->bytes, error, error_len))
        return false;
    if (tensor->bytes > UINT64_MAX - residency->non_ple_bytes) {
        set_error(error, error_len, "resident byte count overflow");
        return false;
    }
    residency->non_ple_bytes += tensor->bytes;
    residency->aligned_bytes += q3_align_up_u64(
        tensor->bytes, Q3_RESIDENT_ALIGNMENT);
    return true;
}
