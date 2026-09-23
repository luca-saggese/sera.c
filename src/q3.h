#ifndef Q3_H
#define Q3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q3 — Qwen3-32B Q4 GGUF model loader (M0).
 *
 * The CLI exposes platform probe, GGUF inspection, residency planning, and a
 * fast resident CUDA load. The only supported target is GB10 / SM 12.1 CUDA on
 * Linux aarch64. Anything else is refused explicitly, never silently degraded.
 * ========================================================================= */

/* --- Command mode ------------------------------------------------ */
typedef enum {
    Q3_MODE_NONE = 0,
    Q3_MODE_PLATFORM,     /* --platform           */
    Q3_MODE_INSPECT,      /* --inspect model.gguf */
    Q3_MODE_LIST_TENSORS, /* --list-tensors model.gguf */
    Q3_MODE_MEMORY_PLAN,  /* --memory-plan model.gguf */
    Q3_MODE_LOAD_ONLY,    /* --load-only model.gguf */
    Q3_MODE_BENCH_Q4_LINEAR, /* --bench-q4-linear model.gguf */
    Q3_MODE_FORWARD,      /* --forward model.gguf */
    Q3_MODE_BENCH_DECISIONS, /* --bench-decisions model.gguf */
} q3_mode;

/* --- Narrowed engine options -------------------------------------- */
typedef struct {
    const char *model_path;
    bool inspect;         /* --inspect */
    bool list_tensors;    /* --list-tensors */
    bool memory_plan;     /* --memory-plan */
    bool load_only;       /* --load-only */
    bool platform;        /* --platform */
    bool json;            /* --json */
    bool verbose;         /* --verbose */
    const char *tensor_name; /* --tensor <name> */
    int batches[16];      /* --batch 1,4,16,32 */
    int batch_count;
    const char *tokens_path;   /* --tokens <file> (M2 forward) */
    const char *candidate_ids; /* --candidate-token-ids a,b,c (M2 forward) */
    const char *workload_path; /* --workload <file> (M3 decisions) */
} q3_options;

/* --- Platform probe ----------------------------------------------- */
typedef struct {
    int cuda_device_count;
    int cuda_device;
    int cc_major;
    int cc_minor;
    uint64_t cuda_total_bytes;
    uint64_t cuda_free_bytes;
    uint64_t mem_total_bytes;
    uint64_t mem_available_bytes;
    char device_name[128];
    char driver_version[32];
    char runtime_version[32];
} q3_platform_info;

/* --- Memory telemetry snapshot ------------------------------------ */
typedef struct {
    const char *phase;             /* e.g. "gguf_mapped" */
    uint64_t rss_bytes;
    uint64_t mem_available_bytes;
    uint64_t cuda_free_bytes;
    uint64_t cuda_total_bytes;
    uint64_t model_file_bytes;
    uint64_t model_mapped_bytes;
    uint64_t cuda_allocated_bytes;
    uint64_t peak_internal_bytes;
} q3_memory_snapshot;

#ifdef __cplusplus
}
#endif

#endif /* Q3_H */
