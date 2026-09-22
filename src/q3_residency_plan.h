#ifndef Q3_RESIDENCY_PLAN_H
#define Q3_RESIDENCY_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*q3_residency_plan_is_ple_fn)(const q3_tensor *tensor,
                                              void *user);

typedef struct {
    uint32_t tensor_index;
    uint64_t file_offset;
    uint64_t bytes;
    uint32_t ownership_class;
} q3_residency_plan_entry;

typedef struct {
    uint64_t file_offset;
    uint64_t bytes;
    size_t first_entry;
    size_t entry_count;
} q3_residency_plan_span;

typedef struct {
    q3_residency_plan_entry *entries;
    size_t entry_count;
    q3_residency_plan_span *spans;
    size_t span_count;
    uint64_t resident_bytes;
    uint64_t excluded_ple_bytes;
    uint64_t excluded_ple_tensors;
} q3_residency_plan;

void q3_residency_plan_init(q3_residency_plan *plan);
void q3_residency_plan_destroy(q3_residency_plan *plan);

/*
 * Build a deterministic file-offset ordered plan. Spans may include small
 * non-tensor gaps, but never cross an excluded PLE range or a large gap.
 */
bool q3_residency_plan_build(
    const q3_gguf *model,
    q3_residency_plan_is_ple_fn is_ple,
    void *is_ple_user,
    uint64_t max_gap_bytes,
    uint64_t max_span_bytes,
    q3_residency_plan *plan,
    char *error,
    size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
