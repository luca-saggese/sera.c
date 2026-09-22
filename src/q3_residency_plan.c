#include "q3_residency_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    Q3_RESIDENCY_OWNERSHIP_RESIDENT = 1,
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) {
        snprintf(error, error_len, "%s", message);
    }
    return false;
}

static int compare_entries(const void *left, const void *right) {
    const q3_residency_plan_entry *a = left;
    const q3_residency_plan_entry *b = right;
    if (a->file_offset < b->file_offset) return -1;
    if (a->file_offset > b->file_offset) return 1;
    if (a->tensor_index < b->tensor_index) return -1;
    if (a->tensor_index > b->tensor_index) return 1;
    return 0;
}

static bool ranges_overlap(uint64_t a_offset, uint64_t a_bytes,
                           uint64_t b_offset, uint64_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    if (a_offset > UINT64_MAX - a_bytes ||
        b_offset > UINT64_MAX - b_bytes)
        return true;
    return a_offset < b_offset + b_bytes &&
           b_offset < a_offset + a_bytes;
}

static bool span_has_ple_between(
    const q3_gguf *model,
    q3_residency_plan_is_ple_fn is_ple,
    void *is_ple_user,
    uint64_t span_offset,
    uint64_t span_bytes,
    uint64_t next_offset) {
    const uint64_t span_end = span_offset + span_bytes;
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const q3_tensor *candidate = &model->tensors[i];
        if (!candidate->bytes || !is_ple(candidate, is_ple_user))
            continue;
        if (ranges_overlap(span_offset, span_bytes, candidate->abs_offset,
                           candidate->bytes))
            return true;
        if (candidate->abs_offset >= span_end &&
            candidate->abs_offset <= next_offset)
            return true;
    }
    return false;
}

void q3_residency_plan_init(q3_residency_plan *plan) {
    if (plan) memset(plan, 0, sizeof(*plan));
}

void q3_residency_plan_destroy(q3_residency_plan *plan) {
    if (!plan) return;
    free(plan->entries);
    free(plan->spans);
    q3_residency_plan_init(plan);
}

bool q3_residency_plan_build(
    const q3_gguf *model,
    q3_residency_plan_is_ple_fn is_ple,
    void *is_ple_user,
    uint64_t max_gap_bytes,
    uint64_t max_span_bytes,
    q3_residency_plan *plan,
    char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!model || !plan || !is_ple || !max_span_bytes)
        return fail(error, error_len, "invalid residency planner arguments");

    q3_residency_plan_destroy(plan);
    if (model->n_tensors > SIZE_MAX / sizeof(*plan->entries))
        return fail(error, error_len, "residency planner tensor overflow");
    plan->entries = calloc((size_t)model->n_tensors ?
                               (size_t)model->n_tensors : 1,
                           sizeof(*plan->entries));
    if (!plan->entries)
        return fail(error, error_len, "residency planner allocation failed");

    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const q3_tensor *tensor = &model->tensors[i];
        if (!tensor->bytes) continue;
        if (tensor->abs_offset > model->size ||
            tensor->bytes > model->size - tensor->abs_offset) {
            q3_residency_plan_destroy(plan);
            return fail(error, error_len, "residency tensor outside model");
        }
        if (is_ple(tensor, is_ple_user)) {
            plan->excluded_ple_bytes += tensor->bytes;
            plan->excluded_ple_tensors++;
            continue;
        }
        q3_residency_plan_entry *entry =
            &plan->entries[plan->entry_count++];
        entry->tensor_index = (uint32_t)i;
        entry->file_offset = tensor->abs_offset;
        entry->bytes = tensor->bytes;
        entry->ownership_class = Q3_RESIDENCY_OWNERSHIP_RESIDENT;
        if (plan->resident_bytes > UINT64_MAX - tensor->bytes) {
            q3_residency_plan_destroy(plan);
            return fail(error, error_len, "residency byte count overflow");
        }
        plan->resident_bytes += tensor->bytes;
    }

    qsort(plan->entries, plan->entry_count, sizeof(*plan->entries),
          compare_entries);
    if (plan->entry_count == 0) return true;
    plan->spans = calloc(plan->entry_count, sizeof(*plan->spans));
    if (!plan->spans) {
        q3_residency_plan_destroy(plan);
        return fail(error, error_len, "residency span allocation failed");
    }

    for (size_t i = 0; i < plan->entry_count; ++i) {
        const q3_residency_plan_entry *entry = &plan->entries[i];
        if (!plan->span_count) {
            plan->spans[0] = (q3_residency_plan_span){
                entry->file_offset, entry->bytes, i, 1};
            plan->span_count = 1;
            continue;
        }
        q3_residency_plan_span *span = &plan->spans[plan->span_count - 1];
        const uint64_t span_end = span->file_offset + span->bytes;
        const uint64_t entry_end = entry->file_offset + entry->bytes;
        const bool ordered = entry->file_offset >= span_end;
        const uint64_t gap = ordered ? entry->file_offset - span_end : 0;
        const bool has_ple_between = span_has_ple_between(
            model, is_ple, is_ple_user, span->file_offset, span->bytes,
            entry->file_offset);
        const bool fits = ordered && gap <= max_gap_bytes &&
                          entry_end >= span->file_offset &&
                          entry_end - span->file_offset <= max_span_bytes &&
                          !has_ple_between;
        if (fits) {
            span->bytes = entry_end - span->file_offset;
            span->entry_count++;
        } else {
            plan->spans[plan->span_count++] =
                (q3_residency_plan_span){
                    entry->file_offset, entry->bytes, i, 1};
        }
    }
    return true;
}
