#ifndef Q3_MODEL_LOADER_CUDA_H
#define Q3_MODEL_LOADER_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q3_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q3_model_loader_cuda — reduced resident CUDA loader (M0).
 *
 * Ported from the q38 donor fast-residency path with every non-loading
 * subsystem removed (QSA, GDN, GR, PLE, MoE, decode, forward, LM-head).
 *
 * Invariant (M0 §7):
 *     mmap source -> 2x bounded pinned staging -> cudaMemcpyAsync
 *                 -> final resident CUDA allocations
 *
 * No per-chunk cudaMalloc/cudaFree, no per-chunk sync, no whole-file
 * cudaHostRegister, no full host model mirror, no dequant mirror.
 * ========================================================================= */

typedef struct q3_loader_context q3_loader_context;

/* Telemetry captured by one load. Field names track the required M0 §9 keys;
 * the JSON object itself is emitted by the CLI. */
typedef struct {
    double total_ms;          /* whole load wall clock          */
    double plan_ms;           /* residency planning             */
    double cuda_alloc_ms;     /* all cudaMalloc/cudaMallocHost  */
    double source_copy_ms;    /* mmap -> pinned staging memcpy  */
    double h2d_enqueue_ms;    /* cudaMemcpyAsync H2D enqueue    */
    double d2d_enqueue_ms;    /* cudaMemcpyAsync D2D enqueue    */
    double final_wait_ms;     /* the single final stream sync   */

    uint64_t planned_bytes;   /* bytes selected by the plan     */
    uint64_t planned_spans;   /* number of coalesced spans      */
    uint64_t resident_tensors;
    uint64_t resident_bytes;  /* bytes actually in final allocs */
    uint64_t staging_bytes;   /* pinned bytes per stage slot    */
    uint64_t h2d_bytes;       /* bytes enqueued H2D             */
    uint64_t staged_bytes;    /* bytes copied through staging   */
    uint64_t transfer_calls;  /* H2D cudaMemcpyAsync calls      */
    uint64_t device_copies;   /* D2D cudaMemcpyAsync calls      */
    uint64_t cuda_allocations;
    uint64_t cuda_allocated_bytes;
    uint64_t final_syncs;     /* must be exactly 1              */
    uint64_t device_syncs;    /* cudaDeviceSynchronize count    */

    uint64_t minor_faults_before;
    uint64_t minor_faults_after;
    uint64_t major_faults_before;
    uint64_t major_faults_after;
    uint64_t mincore_pages_before;
    uint64_t mincore_pages_after;

    bool coverage_ok;         /* every planned tensor became resident */
} q3_load_stats;

q3_loader_context *q3_loader_context_create(char *error, size_t error_len);
void q3_loader_context_destroy(q3_loader_context *context);

/* Load every planned tensor into stable resident device allocations.
 * `model` must stay open (and mapped) for the lifetime of `context`. */
bool q3_loader_load(q3_loader_context *context, const q3_gguf *model,
                    char *error, size_t error_len);

void q3_loader_get_stats(const q3_loader_context *context,
                         q3_load_stats *stats);

/* Resident descriptor table accessor, indexed by GGUF tensor index. */
typedef struct {
    const void *host;
    const void *ptr;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t qtype;
    uint32_t tensor_id;
    uint64_t gguf_offset;
    const char *name;
    bool resident;
} q3_resident_tensor;

size_t q3_loader_tensor_count(const q3_loader_context *context);
const q3_resident_tensor *q3_loader_tensor(const q3_loader_context *context,
                                           size_t index);

#ifdef __cplusplus
}
#endif

#endif /* Q3_MODEL_LOADER_CUDA_H */