/* q3_model_loader_cuda.cu — reduced resident CUDA loader (M0).
 *
 * Ported from the q38 donor fast-residency path
 * (cuda/q38_forward_cuda.cu, q38_forward_cuda_enable_all_non_ple_residency)
 * with every non-loading subsystem deleted: QSA, GDN, GR, PLE, MoE, experts,
 * routing, steering, decode, forward, argmax, LM-head execution.
 *
 * The load shape is preserved exactly:
 *   - residency plan grouped into coalesced spans
 *   - two pinned staging slots + two paired device transfer buffers
 *   - per-span: memcpy(mmap -> pinned), cudaMemcpyAsync H2D, then one
 *     cudaMemcpyAsync D2D per resident tensor, then cudaEventRecord
 *   - stage reuse waits on the paired event (only from the third span on)
 *   - exactly one final cudaStreamSynchronize
 */

#include "q3_model_loader_cuda.h"
#include "q3_residency_plan.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>

#define Q3_LOAD_MAX_GAP_BYTES  (64ull * 1024ull)
#define Q3_LOAD_MAX_SPAN_BYTES (256ull * 1024ull * 1024ull)
#define Q3_STAGE_SLOTS         2u

struct q3_resident_slot {
    const void *host;
    void *device;
    size_t bytes;
};

struct q3_loader_context {
    cudaStream_t stream;

    q3_resident_slot *entries;
    size_t entry_count;
    size_t entry_capacity;

    q3_resident_tensor *tensors;
    size_t tensor_count;
    const q3_gguf *model;

    void *stage_buffers[Q3_STAGE_SLOTS];
    void *transfer_buffers[Q3_STAGE_SLOTS];
    cudaEvent_t reuse_events[Q3_STAGE_SLOTS];
    bool reuse_events_ready[Q3_STAGE_SLOTS];
    size_t staging_bytes;

    bool loaded;
    q3_load_stats stats;
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint64_t mincore_pages(const q3_gguf *model) {
    if (!model || !model->map || !model->size) return 0;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t address = (uintptr_t)model->map;
    const uintptr_t base = address & ~(uintptr_t)(page - 1);
    const size_t offset = (size_t)(address - base);
    const size_t pages = (offset + model->size + page - 1) / page;
    unsigned char *vec = (unsigned char *)calloc(pages, 1);
    if (!vec) return 0;
    uint64_t resident = 0;
    if (mincore((void *)base, pages * page, vec) == 0) {
        for (size_t i = 0; i < pages; ++i) resident += (vec[i] & 1u) != 0;
    }
    free(vec);
    return resident;
}

static bool tensor_shape(const q3_tensor *tensor, size_t *rows, size_t *cols) {
    if (!tensor || !rows || !cols || !tensor->ndim || tensor->ndim > 3)
        return false;
    size_t r = 1;
    for (uint32_t i = 0; i + 1 < tensor->ndim; ++i) {
        if (!tensor->dim[i] || r > SIZE_MAX / (size_t)tensor->dim[i])
            return false;
        r *= (size_t)tensor->dim[i];
    }
    if (!tensor->dim[tensor->ndim - 1] ||
        tensor->dim[tensor->ndim - 1] > SIZE_MAX)
        return false;
    *rows = r;
    *cols = (size_t)tensor->dim[tensor->ndim - 1];
    return true;
}

static void release_staging(q3_loader_context *context) {
    for (size_t slot = 0; slot < Q3_STAGE_SLOTS; ++slot) {
        if (context->reuse_events_ready[slot])
            cudaEventDestroy(context->reuse_events[slot]);
        context->reuse_events_ready[slot] = false;
        cudaFree(context->transfer_buffers[slot]);
        cudaFreeHost(context->stage_buffers[slot]);
        context->transfer_buffers[slot] = NULL;
        context->stage_buffers[slot] = NULL;
    }
}

static void release_resident(q3_loader_context *context) {
    for (size_t i = 0; i < context->entry_count; ++i)
        cudaFree(context->entries[i].device);
    context->entry_count = 0;
}

q3_loader_context *q3_loader_context_create(char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q3_loader_context *context =
        (q3_loader_context *)calloc(1, sizeof(*context));
    if (!context) {
        fail(error, error_len, "loader context allocation failed");
        return NULL;
    }
    if (cudaStreamCreate(&context->stream) != cudaSuccess) {
        free(context);
        fail(error, error_len, "loader stream creation failed");
        return NULL;
    }
    return context;
}

void q3_loader_context_destroy(q3_loader_context *context) {
    if (!context) return;
    release_resident(context);
    release_staging(context);
    free(context->entries);
    free(context->tensors);
    if (context->stream) cudaStreamDestroy(context->stream);
    free(context);
}

static bool ensure_capacity(q3_loader_context *context, size_t needed,
                            char *error, size_t error_len) {
    if (needed <= context->entry_capacity) return true;
    size_t capacity = context->entry_capacity ? context->entry_capacity : 1024u;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = needed;
            break;
        }
        capacity *= 2u;
    }
    q3_resident_slot *entries =
        (q3_resident_slot *)realloc(context->entries,
                                    capacity * sizeof(*entries));
    if (!entries)
        return fail(error, error_len, "resident slot table allocation failed");
    context->entries = entries;
    context->entry_capacity = capacity;
    return true;
}

bool q3_loader_load(q3_loader_context *context, const q3_gguf *model,
                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model || !model->tensors || !model->n_tensors)
        return fail(error, error_len, "invalid loader arguments");
    if (context->loaded) return true;

    const double total_started = now_ms();
    struct rusage usage_before = {};
    getrusage(RUSAGE_SELF, &usage_before);
    const uint64_t mincore_before = mincore_pages(model);

    /* --- plan ---------------------------------------------------------
     * Dense Qwen3: every tensor with a known physical size is resident, so
     * the PLE predicate always answers false. */
    q3_residency_plan plan;
    q3_residency_plan_init(&plan);
    const double plan_started = now_ms();
    if (!q3_residency_plan_build(
            model,
            [](const q3_tensor *, void *) -> bool { return false; },
            NULL, Q3_LOAD_MAX_GAP_BYTES, Q3_LOAD_MAX_SPAN_BYTES,
            &plan, error, error_len))
        return false;
    context->stats.plan_ms = now_ms() - plan_started;
    context->stats.planned_bytes = plan.resident_bytes;
    context->stats.planned_spans = plan.span_count;

    if (plan.resident_bytes > SIZE_MAX) {
        q3_residency_plan_destroy(&plan);
        return fail(error, error_len, "residency size overflow");
    }

    if (!ensure_capacity(context, plan.entry_count, error, error_len)) {
        q3_residency_plan_destroy(&plan);
        return false;
    }
    free(context->tensors);
    context->tensors = (q3_resident_tensor *)calloc(
        model->n_tensors, sizeof(*context->tensors));
    if (!context->tensors) {
        q3_residency_plan_destroy(&plan);
        return fail(error, error_len, "descriptor table allocation failed");
    }
    context->tensor_count = (size_t)model->n_tensors;
    context->model = model;
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const q3_tensor *tensor = &model->tensors[i];
        q3_resident_tensor *entry = &context->tensors[i];
        size_t rows = 0, cols = 0;
        (void)tensor_shape(tensor, &rows, &cols);
        entry->host = q3_gguf_tensor_data(model, tensor);
        entry->bytes = tensor->bytes;
        entry->rows = (uint32_t)rows;
        entry->cols = (uint32_t)cols;
        entry->qtype = tensor->type;
        entry->tensor_id = (uint32_t)i;
        entry->gguf_offset = tensor->abs_offset;
        entry->name = tensor->name.ptr;
    }

    /* --- final resident allocations (one per planned tensor) ---------- */
    const double alloc_started = now_ms();
    size_t total = 0;
    uint64_t loaded_bytes = 0;
    for (size_t p = 0; p < plan.entry_count; ++p) {
        const q3_residency_plan_entry *planned = &plan.entries[p];
        const uint32_t tensor_index = planned->tensor_index;
        const q3_tensor *tensor = &model->tensors[tensor_index];
        void *device = NULL;
        if (cudaMalloc(&device, (size_t)tensor->bytes) != cudaSuccess) {
            context->entry_count = total;
            q3_residency_plan_destroy(&plan);
            if (error && error_len)
                snprintf(error, error_len,
                         "resident allocation failed for %.*s: %s",
                         (int)tensor->name.len, tensor->name.ptr,
                         cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        context->entries[total].host = q3_gguf_tensor_data(model, tensor);
        context->entries[total].device = device;
        context->entries[total].bytes = (size_t)tensor->bytes;
        context->tensors[tensor_index].ptr = device;
        context->tensors[tensor_index].resident = true;
        ++total;
        loaded_bytes += tensor->bytes;
    }
    context->entry_count = total;
    context->stats.cuda_allocations += plan.entry_count;
    context->stats.cuda_allocated_bytes += plan.resident_bytes;

    /* --- bounded pinned staging: 2 slots of largest_span ------------- */
    size_t largest_span = 0;
    for (size_t i = 0; i < plan.span_count; ++i)
        if (plan.spans[i].bytes > largest_span)
            largest_span = (size_t)plan.spans[i].bytes;

    for (size_t slot = 0; slot < Q3_STAGE_SLOTS && largest_span; ++slot) {
        if (cudaMallocHost(&context->stage_buffers[slot], largest_span) !=
                cudaSuccess ||
            cudaMalloc(&context->transfer_buffers[slot], largest_span) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&context->reuse_events[slot],
                                     cudaEventDisableTiming) != cudaSuccess) {
            release_staging(context);
            q3_residency_plan_destroy(&plan);
            return fail(error, error_len, "staging allocation failed");
        }
        context->reuse_events_ready[slot] = true;
    }
    if (largest_span) {
        context->stats.cuda_allocations += 2u * Q3_STAGE_SLOTS;
        context->stats.cuda_allocated_bytes += largest_span * 2u * Q3_STAGE_SLOTS;
    }
    context->staging_bytes = largest_span;
    context->stats.staging_bytes = largest_span;
    context->stats.cuda_alloc_ms = now_ms() - alloc_started;

    /* --- stream the payload through staging -------------------------- */
    for (size_t s = 0; s < plan.span_count; ++s) {
        const q3_residency_plan_span *span = &plan.spans[s];
        const size_t slot = s % Q3_STAGE_SLOTS;
        if (s >= Q3_STAGE_SLOTS &&
            cudaEventSynchronize(context->reuse_events[slot]) != cudaSuccess) {
            q3_residency_plan_destroy(&plan);
            return fail(error, error_len, "staging reuse wait failed");
        }

        void *stage = context->stage_buffers[slot];
        void *transfer = context->transfer_buffers[slot];

        const double source_started = now_ms();
        memcpy(stage, model->map + span->file_offset, (size_t)span->bytes);
        context->stats.source_copy_ms += now_ms() - source_started;
        context->stats.staged_bytes += span->bytes;

        const double h2d_started = now_ms();
        if (cudaMemcpyAsync(transfer, stage, (size_t)span->bytes,
                            cudaMemcpyHostToDevice, context->stream) !=
            cudaSuccess) {
            q3_residency_plan_destroy(&plan);
            return fail(error, error_len, "H2D upload failed");
        }
        context->stats.h2d_enqueue_ms += now_ms() - h2d_started;
        context->stats.h2d_bytes += span->bytes;
        context->stats.transfer_calls++;

        const double d2d_started = now_ms();
        for (size_t j = 0; j < span->entry_count; ++j) {
            const size_t entry_index = span->first_entry + j;
            const q3_residency_plan_entry *planned =
                &plan.entries[entry_index];
            const uint64_t relative =
                planned->file_offset - span->file_offset;
            if (cudaMemcpyAsync(context->entries[entry_index].device,
                                (const char *)transfer + relative,
                                (size_t)planned->bytes,
                                cudaMemcpyDeviceToDevice,
                                context->stream) != cudaSuccess) {
                q3_residency_plan_destroy(&plan);
                return fail(error, error_len, "resident tensor copy failed");
            }
            context->stats.device_copies++;
        }
        if (cudaEventRecord(context->reuse_events[slot], context->stream) !=
            cudaSuccess) {
            q3_residency_plan_destroy(&plan);
            return fail(error, error_len, "staging event record failed");
        }
        context->stats.d2d_enqueue_ms += now_ms() - d2d_started;
    }

    /* --- one normal final synchronization ---------------------------- */
    const double final_started = now_ms();
    if (cudaStreamSynchronize(context->stream) != cudaSuccess) {
        q3_residency_plan_destroy(&plan);
        return fail(error, error_len, "final stream synchronization failed");
    }
    context->stats.final_wait_ms = now_ms() - final_started;
    context->stats.final_syncs++;

    struct rusage usage_after = {};
    getrusage(RUSAGE_SELF, &usage_after);
    context->stats.minor_faults_before = usage_before.ru_minflt;
    context->stats.minor_faults_after = usage_after.ru_minflt;
    context->stats.major_faults_before = usage_before.ru_majflt;
    context->stats.major_faults_after = usage_after.ru_majflt;
    context->stats.mincore_pages_before = mincore_before;
    context->stats.mincore_pages_after = mincore_pages(model);

    context->stats.resident_tensors = total;
    context->stats.resident_bytes = loaded_bytes;
    context->stats.coverage_ok =
        total == plan.entry_count && loaded_bytes == plan.resident_bytes;
    context->stats.total_ms = now_ms() - total_started;
    context->loaded = true;

    q3_residency_plan_destroy(&plan);
    if (!context->stats.coverage_ok)
        return fail(error, error_len, "incomplete resident coverage");
    return true;
}

void q3_loader_get_stats(const q3_loader_context *context, q3_load_stats *stats) {
    if (!stats) return;
    if (!context) {
        memset(stats, 0, sizeof(*stats));
        return;
    }
    *stats = context->stats;
}

size_t q3_loader_tensor_count(const q3_loader_context *context) {
    return context ? context->tensor_count : 0;
}

const q3_resident_tensor *q3_loader_tensor(const q3_loader_context *context,
                                           size_t index) {
    if (!context || index >= context->tensor_count) return NULL;
    return &context->tensors[index];
}