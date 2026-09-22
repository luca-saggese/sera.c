/* q3_memory.c — memory telemetry (spec §8). */

#include "q3_memory.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Current process RSS via /proc/self/statm. Returns 0 on success. */
static int proc_rss_bytes(uint64_t *out) {
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp) return -1;
    unsigned long pages = 0;
    unsigned long ignored_size = 0;
    int rc = fscanf(fp, "%lu %lu", &ignored_size, &pages);
    fclose(fp);
    if (rc != 2) return -1;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    *out = (uint64_t)pages * (uint64_t)page_size;
    return 0;
}

/* MemAvailable from /proc/meminfo. Returns 0 on success. */
static int proc_mem_available_bytes(uint64_t *out) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return -1;
    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long kb = 0;
        if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
            *out = (uint64_t)kb * 1024ull;
            found = 1;
            break;
        }
    }
    fclose(fp);
    return found ? 0 : -1;
}

void q3_memory_tracker_init(q3_memory_tracker *t) {
    t->internal_allocated_bytes = 0;
    t->peak_internal_bytes = 0;
}

void q3_memory_track_alloc(q3_memory_tracker *t, uint64_t bytes) {
    t->internal_allocated_bytes += bytes;
    if (t->internal_allocated_bytes > t->peak_internal_bytes) {
        t->peak_internal_bytes = t->internal_allocated_bytes;
    }
}

void q3_memory_track_free(q3_memory_tracker *t, uint64_t bytes) {
    if (bytes > t->internal_allocated_bytes) {
        t->internal_allocated_bytes = 0;
    } else {
        t->internal_allocated_bytes -= bytes;
    }
}

void q3_memory_capture(q3_memory_tracker *t,
                        const char *phase,
                        uint64_t model_file_bytes,
                        uint64_t model_mapped_bytes,
                        uint64_t cuda_allocated_bytes,
                        q3_memory_snapshot *out) {
    memset(out, 0, sizeof(*out));
    out->phase = phase;
    out->model_file_bytes = model_file_bytes;
    out->model_mapped_bytes = model_mapped_bytes;
    out->cuda_allocated_bytes = cuda_allocated_bytes;

    uint64_t rss = 0;
    if (proc_rss_bytes(&rss) == 0) out->rss_bytes = rss;

    uint64_t avail = 0;
    if (proc_mem_available_bytes(&avail) == 0) out->mem_available_bytes = avail;

    out->peak_internal_bytes = t ? t->peak_internal_bytes : 0;
}

int q3_memory_snapshot_json(const q3_memory_snapshot *s,
                             char *buf, size_t buf_len) {
    return snprintf(buf, buf_len,
        "{\"phase\":\"%s\",\"rss_bytes\":%" PRIu64
        ",\"mem_available_bytes\":%" PRIu64
        ",\"cuda_free_bytes\":%" PRIu64
        ",\"cuda_total_bytes\":%" PRIu64
        ",\"model_file_bytes\":%" PRIu64
        ",\"model_mapped_bytes\":%" PRIu64
        ",\"cuda_allocated_bytes\":%" PRIu64
        ",\"peak_internal_bytes\":%" PRIu64 "}",
        s->phase ? s->phase : "",
        s->rss_bytes,
        s->mem_available_bytes,
        s->cuda_free_bytes,
        s->cuda_total_bytes,
        s->model_file_bytes,
        s->model_mapped_bytes,
        s->cuda_allocated_bytes,
        s->peak_internal_bytes);
}
