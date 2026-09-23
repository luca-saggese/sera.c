/* SPDX-License-Identifier: MIT
 *
 * q3_workload.h - reader for the M3 semantic decision workload (docs/M3.md §56).
 *
 * The workload is the frozen artifact produced by the tokenizer step: one
 * shared state prompt plus N questions, each carrying its own suffix token IDs
 * and the candidate token IDs to score. Only those fields are read; the reader
 * is deliberately a tiny hand-rolled scanner, not a JSON library.
 */
#ifndef Q3_WORKLOAD_H
#define Q3_WORKLOAD_H

#include <stddef.h>
#include <stdint.h>

#include "q3_decide.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q3_WORKLOAD_ID_MAX 16
#define Q3_WORKLOAD_CATEGORY_MAX 32

typedef struct {
    char id[Q3_WORKLOAD_ID_MAX];
    char category[Q3_WORKLOAD_CATEGORY_MAX];
    uint32_t *suffix_tokens;
    uint32_t suffix_count;
    uint32_t candidate_ids[Q3_MAX_CANDIDATES];
    uint32_t candidate_count;
    uint32_t expected_index;
} q3_workload_question;

typedef struct {
    uint32_t *state_tokens;
    uint32_t state_count;
    q3_workload_question *questions;
    uint32_t question_count;
    uint32_t max_suffix_count;
} q3_workload;

/* Parse `path`. Returns NULL and fills `error` on any malformed input. */
q3_workload *q3_workload_load(const char *path, char *error, size_t error_len);
void q3_workload_free(q3_workload *w);

#ifdef __cplusplus
}
#endif

#endif /* Q3_WORKLOAD_H */
