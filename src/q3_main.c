/* q3_main.c — q3 inspection CLI: platform probe, GGUF inventory, residency
 * plan and fast resident CUDA load.
 *
 * Ported from the q38 donor (COPY -> RENAME -> EDIT). The surface is
 * deliberately narrow: no decode, no generation, no server, no tokenizer.
 * M1 adds one diagnostic mode, --bench-q4-linear, which exercises the
 * resident Q4_K quantized linear primitive on a real model tensor.
 */

#include "q3.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_memory.h"
#include "q3_platform.h"
#include "q3_residency_plan.h"
#include "q3_model_loader_cuda.h"
#include "q3_q4_linear.h"
#include "q3_forward_cli.h"
#include "q3_decide_cli.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "usage: q3 <mode> [options]\n"
        "\n"
        "modes:\n"
        "  --platform                 Probe the platform (CUDA + host memory)\n"
        "  --inspect <model.gguf>     Print GGUF metadata and tensor summary\n"
        "  --list-tensors <model.gguf> List individual tensors with offsets\n"
        "  --memory-plan <model.gguf> Dry-run residency plan (no allocation)\n"
        "  --load-only <model.gguf>   Fast resident CUDA load; no inference\n"
        "  --bench-q4-linear <model.gguf>\n"
        "                             Benchmark the resident Q4_K linear primitive\n"
        "  --forward <model.gguf>     One-pass forward over pre-tokenized IDs\n"
        "  --bench-decisions <model.gguf>\n"
        "                             M3 shared-prefix batched decision benchmark\n"
        "\n"
        "options:\n"
        "  --tensor <name>            Tensor to benchmark (default: largest Q4_K)\n"
        "  --batch <a,b,...>          Batch sizes (default: 1,4,16,32)\n"
        "  --tokens <file>            Token IDs (uint32 LE) for --forward\n"
        "  --candidate-token-ids <a,b,c>\n"
        "                             Candidate LM rows for --forward\n"
        "  --workload <file>          Decision workload JSON for --bench-decisions\n"
        "  --json                     Machine-readable output\n"
        "  --verbose                  Extra diagnostics\n");
}

static void print_bytes_plain(uint64_t b) { printf("%" PRIu64, b); }

static void print_platform_human(const q3_platform_info *p) {
    printf("cuda devices:       %d\n", p->cuda_device_count);
    printf("cuda device:        %d\n", p->cuda_device);
    printf("device name:        %s\n", p->device_name);
    printf("compute capability: sm_%d%d\n", p->cc_major, p->cc_minor);
    printf("driver version:     %s\n", p->driver_version[0] ? p->driver_version : "n/a");
    printf("runtime version:    %s\n", p->runtime_version[0] ? p->runtime_version : "n/a");
    printf("cuda total:         ");
    print_bytes_plain(p->cuda_total_bytes);
    printf(" bytes\n");
    printf("cuda free:          ");
    print_bytes_plain(p->cuda_free_bytes);
    printf(" bytes\n");
    printf("host mem total:     ");
    print_bytes_plain(p->mem_total_bytes);
    printf(" bytes\n");
    printf("host mem available: ");
    print_bytes_plain(p->mem_available_bytes);
    printf(" bytes\n");
}

static void print_platform_json(const q3_platform_info *p) {
    printf("{\"cuda_device_count\":%d,\"cuda_device\":%d,"
           "\"cc_major\":%d,\"cc_minor\":%d,"
           "\"cuda_total_bytes\":%" PRIu64 ",\"cuda_free_bytes\":%" PRIu64
           ",\"mem_total_bytes\":%" PRIu64 ",\"mem_available_bytes\":%" PRIu64
           ",\"device_name\":\"%s\",\"driver_version\":\"%s\","
           "\"runtime_version\":\"%s\"}\n",
           p->cuda_device_count, p->cuda_device,
           p->cc_major, p->cc_minor,
           p->cuda_total_bytes, p->cuda_free_bytes,
           p->mem_total_bytes, p->mem_available_bytes,
           p->device_name, p->driver_version, p->runtime_version);
}

static int cmd_platform(const q3_options *opt) {
    q3_platform_info p;
    char reason[256];
    if (q3_platform_probe(&p, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q3: unsupported platform: %s\n", reason);
        return 1;
    }
    if (opt->json) {
        print_platform_json(&p);
    } else {
        print_platform_human(&p);
    }
    return 0;
}

static void model_summary(const q3_gguf *m, uint64_t *tensor_bytes,
                          uint64_t *params) {
    *tensor_bytes = 0;
    *params = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        *tensor_bytes += m->tensors[i].bytes;
        *params += m->tensors[i].elements;
    }
}

static const q3_tensor *largest_tensor(const q3_gguf *m) {
    const q3_tensor *best = NULL;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (!best || m->tensors[i].bytes > best->bytes) best = &m->tensors[i];
    }
    return best;
}

static int cmd_inspect(const q3_options *opt) {
    char err[256];
    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    q3_str name = {0}, arch = {0};
    q3_gguf_get_string(m, "general.name", &name);
    q3_gguf_get_string(m, "general.architecture", &arch);

    uint64_t tensor_bytes = 0, params = 0;
    model_summary(m, &tensor_bytes, &params);
    const q3_tensor *largest = largest_tensor(m);

    if (opt->json) {
        printf("{\"name\":\"%.*s\",\"architecture\":\"%.*s\","
               "\"version\":%u,\"alignment\":%" PRIu64
               ",\"tensor_data_pos\":%" PRIu64
               ",\"metadata_keys\":%" PRIu64
               ",\"tensors\":%" PRIu64
               ",\"file_bytes\":%" PRIu64
               ",\"tensor_bytes\":%" PRIu64
               ",\"logical_parameters\":%" PRIu64
               ",\"largest_tensor\":\"%.*s\""
               ",\"largest_tensor_bytes\":%" PRIu64 "}\n",
               (int)name.len, name.ptr ? name.ptr : "",
               (int)arch.len, arch.ptr ? arch.ptr : "",
               m->version, m->alignment, m->tensor_data_pos, m->n_kv,
               m->n_tensors, m->size, tensor_bytes, params,
               (int)(largest ? largest->name.len : 0),
               largest ? largest->name.ptr : "",
               largest ? largest->bytes : 0);
    } else {
        printf("model:     %.*s\n", (int)name.len, name.ptr ? name.ptr : "");
        printf("arch:      %.*s\n", (int)arch.len, arch.ptr ? arch.ptr : "");
        printf("gguf:      v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
               m->version, m->n_kv, m->n_tensors);
        printf("alignment: %" PRIu64 "\n", m->alignment);
        printf("tensor data starts at: %" PRIu64 "\n", m->tensor_data_pos);
        printf("file size: %" PRIu64 " bytes\n", m->size);
        printf("tensor bytes: %" PRIu64 "\n", tensor_bytes);
        printf("logical parameters: %" PRIu64 "\n", params);
        if (largest) {
            printf("largest tensor: %.*s (%" PRIu64 " bytes)\n",
                   (int)largest->name.len, largest->name.ptr, largest->bytes);
        }

        printf("tensor types:\n");
        for (uint32_t type = 0; type < 64; type++) {
            uint64_t count = 0, bytes = 0;
            for (uint64_t i = 0; i < m->n_tensors; i++) {
                if (m->tensors[i].type == type) {
                    count++;
                    bytes += m->tensors[i].bytes;
                }
            }
            if (count != 0) {
                printf("  %-8s %5" PRIu64 " tensors, %" PRIu64 " bytes\n",
                       q3_gguf_type_name(type), count, bytes);
            }
        }
    }

    q3_gguf_close(m);
    return 0;
}

static int cmd_list_tensors(const q3_options *opt) {
    char err[256];
    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    if (opt->json) {
        printf("{\"tensors\":[");
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q3_tensor *t = &m->tensors[i];
            printf("%s{\"name\":\"%.*s\",\"type\":\"%s\",\"ndim\":%u,"
                   "\"shape\":[",
                   i ? "," : "", (int)t->name.len, t->name.ptr,
                   q3_gguf_type_name(t->type), t->ndim);
            for (uint32_t d = 0; d < t->ndim; d++) {
                printf("%s%" PRIu64, d ? "," : "", t->dim[d]);
            }
            printf("],\"elements\":%" PRIu64 ",\"bytes\":%" PRIu64
                   ",\"rel_offset\":%" PRIu64 ",\"abs_offset\":%" PRIu64 "}",
                   t->elements, t->bytes, t->rel_offset, t->abs_offset);
        }
        printf("]}\n");
    } else {
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q3_tensor *t = &m->tensors[i];
            char shape[128];
            size_t pos = 0;
            shape[0] = '\0';
            for (uint32_t d = 0; d < t->ndim; d++) {
                int n = snprintf(shape + pos, sizeof(shape) - pos, "%s%" PRIu64,
                                 d ? "x" : "", t->dim[d]);
                if (n < 0 || (size_t)n >= sizeof(shape) - pos) break;
                pos += (size_t)n;
            }
            if (opt->verbose) {
                printf("%-44.*s %-6s %-14s off=%" PRIu64 " abs=%" PRIu64
                       " bytes=%" PRIu64 "\n",
                       (int)t->name.len, t->name.ptr,
                       q3_gguf_type_name(t->type), shape,
                       t->rel_offset, t->abs_offset, t->bytes);
            } else {
                printf("%-44.*s %-6s %-14s %" PRIu64 " elems %" PRIu64 " bytes\n",
                       (int)t->name.len, t->name.ptr,
                       q3_gguf_type_name(t->type), shape,
                       t->elements, t->bytes);
            }
        }
    }

    q3_gguf_close(m);
    return 0;
}

/* M0 dense Qwen3 residency plan: every tensor with a known physical size is
 * resident. There are no PLE banks, no experts and no special banks. */
static bool plan_exclude_none(const q3_tensor *tensor, void *user) {
    (void)tensor;
    (void)user;
    return false;
}

#define Q3_PLAN_MAX_GAP_BYTES   (64ull * 1024ull)
#define Q3_PLAN_MAX_SPAN_BYTES  (256ull * 1024ull * 1024ull)

static bool plan_validate(const q3_residency_plan *plan, char *err,
                          size_t err_len) {
    uint64_t total = 0;
    for (size_t i = 0; i < plan->entry_count; i++) {
        const q3_residency_plan_entry *e = &plan->entries[i];
        if (i && e->file_offset < plan->entries[i - 1].file_offset) {
            snprintf(err, err_len, "entries not in ascending offset order");
            return false;
        }
        if (e->bytes > UINT64_MAX - e->file_offset) {
            snprintf(err, err_len, "entry offset overflow");
            return false;
        }
        if (total > UINT64_MAX - e->bytes) {
            snprintf(err, err_len, "entry byte total overflow");
            return false;
        }
        total += e->bytes;
    }
    if (total != plan->resident_bytes) {
        snprintf(err, err_len, "resident byte total mismatch");
        return false;
    }
    for (size_t s = 0; s < plan->span_count; s++) {
        const q3_residency_plan_span *sp = &plan->spans[s];
        if (sp->bytes > UINT64_MAX - sp->file_offset) {
            snprintf(err, err_len, "span offset overflow");
            return false;
        }
        for (size_t k = 0; k < sp->entry_count; k++) {
            const q3_residency_plan_entry *e = &plan->entries[sp->first_entry + k];
            if (e->file_offset < sp->file_offset ||
                e->file_offset + e->bytes > sp->file_offset + sp->bytes) {
                snprintf(err, err_len, "entry %zu outside its span", sp->first_entry + k);
                return false;
            }
        }
    }
    for (size_t s = 1; s < plan->span_count; s++) {
        const q3_residency_plan_span *prev = &plan->spans[s - 1];
        if (plan->spans[s].file_offset < prev->file_offset + prev->bytes) {
            snprintf(err, err_len, "span %zu overlaps previous span", s);
            return false;
        }
    }
    return true;
}

static int cmd_memory_plan(const q3_options *opt) {
    char err[256];
    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    q3_residency_plan plan;
    q3_residency_plan_init(&plan);
    if (!q3_residency_plan_build(m, plan_exclude_none, NULL,
                                 Q3_PLAN_MAX_GAP_BYTES, Q3_PLAN_MAX_SPAN_BYTES,
                                 &plan, err, sizeof(err))) {
        fprintf(stderr, "q3: residency plan: %s\n", err);
        q3_gguf_close(m);
        return 1;
    }

    char perr[256] = {0};
    if (!plan_validate(&plan, perr, sizeof(perr))) {
        fprintf(stderr, "q3: residency plan invalid: %s\n", perr);
        q3_residency_plan_destroy(&plan);
        q3_gguf_close(m);
        return 1;
    }

    uint64_t span_bytes = 0, largest_span = 0;
    for (size_t s = 0; s < plan.span_count; s++) {
        span_bytes += plan.spans[s].bytes;
        if (plan.spans[s].bytes > largest_span) largest_span = plan.spans[s].bytes;
    }

    q3_memory_tracker tracker;
    q3_memory_tracker_init(&tracker);

    q3_platform_info p;
    char reason[256];
    uint64_t cuda_total = 0, cuda_free = 0;
    if (q3_platform_probe(&p, reason, sizeof(reason)) == 0) {
        cuda_total = p.cuda_total_bytes;
        cuda_free = p.cuda_free_bytes;
    }

    q3_memory_snapshot snap;
    q3_memory_capture(&tracker, "gguf_mapped", m->size, m->size, 0, &snap);
    snap.cuda_total_bytes = cuda_total;
    snap.cuda_free_bytes = cuda_free;

    if (opt->json) {
        char buf[2048];
        q3_memory_snapshot_json(&snap, buf, sizeof(buf));
        /* splice the plan fields into the snapshot object */
        size_t len = strlen(buf);
        if (len && buf[len - 1] == '}') buf[--len] = '\0';
        printf("%s,\"planned_tensors\":%zu,\"planned_spans\":%zu,"
               "\"resident_bytes\":%" PRIu64 ",\"excluded_ple_bytes\":%" PRIu64
               ",\"plan_span_bytes\":%" PRIu64 ",\"plan_largest_span_bytes\":%" PRIu64
               ",\"max_gap_bytes\":%llu,\"max_span_bytes\":%llu}\n",
               buf, plan.entry_count, plan.span_count, plan.resident_bytes,
               plan.excluded_ple_bytes, span_bytes, largest_span,
               (unsigned long long)Q3_PLAN_MAX_GAP_BYTES,
               (unsigned long long)Q3_PLAN_MAX_SPAN_BYTES);
    } else {
        printf("model file:        %" PRIu64 " bytes\n", snap.model_file_bytes);
        printf("model mapped:      %" PRIu64 " bytes\n", snap.model_mapped_bytes);
        printf("rss:               %" PRIu64 " bytes\n", snap.rss_bytes);
        printf("mem available:     %" PRIu64 " bytes\n", snap.mem_available_bytes);
        printf("cuda free:         %" PRIu64 " bytes\n", snap.cuda_free_bytes);
        printf("cuda total:        %" PRIu64 " bytes\n", snap.cuda_total_bytes);
        printf("planned tensors:   %zu\n", plan.entry_count);
        printf("planned spans:     %zu\n", plan.span_count);
        printf("resident bytes:    %" PRIu64 "\n", plan.resident_bytes);
        printf("excluded ple bytes:%" PRIu64 "\n", plan.excluded_ple_bytes);
        printf("span bytes:        %" PRIu64 "\n", span_bytes);
        printf("largest span:      %" PRIu64 " bytes\n", largest_span);
        printf("peak internal:     %" PRIu64 " bytes\n", snap.peak_internal_bytes);
    }

    q3_residency_plan_destroy(&plan);
    q3_gguf_close(m);
    return 0;
}

static uint64_t read_rss_bytes(void) {
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp) return 0;
    unsigned long total = 0, resident = 0;
    if (fscanf(fp, "%lu %lu", &total, &resident) != 2) resident = 0;
    fclose(fp);
    return (uint64_t)resident * (uint64_t)sysconf(_SC_PAGESIZE);
}

static uint64_t read_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return (uint64_t)usage.ru_maxrss * 1024u;
}

static uint64_t read_mem_available_bytes(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %" SCNu64 " kB", &kb) == 1) break;
    }
    fclose(fp);
    return kb * 1024u;
}

static void emit_load_json(const q3_load_stats *s, double gguf_open_ms,
                           uint64_t model_bytes, uint64_t rss,
                           uint64_t peak_rss, uint64_t mem_avail,
                           uint64_t cuda_free_before, uint64_t cuda_free_after) {
    printf("{");
    printf("\"gguf_open_ms\":%.3f,", gguf_open_ms);
    printf("\"plan_ms\":%.3f,", s->plan_ms);
    printf("\"cuda_alloc_ms\":%.3f,", s->cuda_alloc_ms);
    printf("\"source_copy_ms\":%.3f,", s->source_copy_ms);
    printf("\"h2d_enqueue_ms\":%.3f,", s->h2d_enqueue_ms);
    printf("\"d2d_enqueue_ms\":%.3f,", s->d2d_enqueue_ms);
    printf("\"final_wait_ms\":%.3f,", s->final_wait_ms);
    printf("\"total_load_ms\":%.3f,", s->total_ms);
    printf("\"model_file_bytes\":%" PRIu64 ",", model_bytes);
    printf("\"planned_bytes\":%" PRIu64 ",", s->planned_bytes);
    printf("\"planned_spans\":%" PRIu64 ",", s->planned_spans);
    printf("\"resident_tensors\":%" PRIu64 ",", s->resident_tensors);
    printf("\"resident_bytes\":%" PRIu64 ",", s->resident_bytes);
    printf("\"staging_bytes\":%" PRIu64 ",", s->staging_bytes);
    printf("\"staged_bytes\":%" PRIu64 ",", s->staged_bytes);
    printf("\"h2d_bytes\":%" PRIu64 ",", s->h2d_bytes);
    printf("\"transfer_calls\":%" PRIu64 ",", s->transfer_calls);
    printf("\"device_copies\":%" PRIu64 ",", s->device_copies);
    printf("\"cuda_allocations\":%" PRIu64 ",", s->cuda_allocations);
    printf("\"cuda_allocated_bytes\":%" PRIu64 ",", s->cuda_allocated_bytes);
    printf("\"final_syncs\":%" PRIu64 ",", s->final_syncs);
    printf("\"device_syncs\":%" PRIu64 ",", s->device_syncs);
    printf("\"minor_faults_before\":%" PRIu64 ",", s->minor_faults_before);
    printf("\"minor_faults_after\":%" PRIu64 ",", s->minor_faults_after);
    printf("\"major_faults_before\":%" PRIu64 ",", s->major_faults_before);
    printf("\"major_faults_after\":%" PRIu64 ",", s->major_faults_after);
    printf("\"mincore_pages_before\":%" PRIu64 ",", s->mincore_pages_before);
    printf("\"mincore_pages_after\":%" PRIu64 ",", s->mincore_pages_after);
    printf("\"rss_bytes\":%" PRIu64 ",", rss);
    printf("\"peak_rss_bytes\":%" PRIu64 ",", peak_rss);
    printf("\"mem_available_bytes\":%" PRIu64 ",", mem_avail);
    printf("\"cuda_free_before\":%" PRIu64 ",", cuda_free_before);
    printf("\"cuda_free_after\":%" PRIu64 ",", cuda_free_after);
    printf("\"coverage_ok\":%s", s->coverage_ok ? "true" : "false");
    printf("}\n");
}

static double mono_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}


static int cmd_load_only(const q3_options *opt) {
    char err[256];
    const double open_started = mono_ms();
    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    const double gguf_open_ms = mono_ms() - open_started;
    if (!m) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    q3_platform_info before = {0}, after = {0};
    char reason[256] = {0};
    if (q3_platform_probe(&before, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q3: %s\n", reason);
        q3_gguf_close(m);
        return 1;
    }
    if (q3_cuda_init() != 0) {
        fprintf(stderr, "q3: CUDA initialization failed\n");
        q3_gguf_close(m);
        return 1;
    }

    q3_loader_context *context =
        q3_loader_context_create(err, sizeof(err));
    if (!context) {
        fprintf(stderr, "q3: %s\n", err);
        q3_gguf_close(m);
        return 1;
    }

    if (!q3_loader_load(context, m, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(context);
        q3_gguf_close(m);
        return 1;
    }

    (void)q3_platform_probe(&after, reason, sizeof(reason));

    q3_load_stats stats;
    q3_loader_get_stats(context, &stats);
    const uint64_t rss = read_rss_bytes();
    const uint64_t peak_rss = read_peak_rss_bytes();
    const uint64_t mem_avail = read_mem_available_bytes();

    if (opt->json) {
        emit_load_json(&stats, gguf_open_ms, m->size, rss, peak_rss,
                       mem_avail, before.cuda_free_bytes,
                       after.cuda_free_bytes);
    } else {
        printf("model file:        %" PRIu64 " bytes\n", m->size);
        printf("planned bytes:     %" PRIu64 "\n", stats.planned_bytes);
        printf("planned spans:     %" PRIu64 "\n", stats.planned_spans);
        printf("resident tensors:  %" PRIu64 "\n", stats.resident_tensors);
        printf("resident bytes:    %" PRIu64 "\n", stats.resident_bytes);
        printf("staging bytes:     %" PRIu64 "\n", stats.staging_bytes);
        printf("h2d bytes:         %" PRIu64 "\n", stats.h2d_bytes);
        printf("transfer calls:    %" PRIu64 "\n", stats.transfer_calls);
        printf("device copies:     %" PRIu64 "\n", stats.device_copies);
        printf("cuda allocations:  %" PRIu64 "\n", stats.cuda_allocations);
        printf("final syncs:       %" PRIu64 "\n", stats.final_syncs);
        printf("device syncs:      %" PRIu64 "\n", stats.device_syncs);
        printf("plan:              %.3f ms\n", stats.plan_ms);
        printf("cuda alloc:        %.3f ms\n", stats.cuda_alloc_ms);
        printf("source copy:       %.3f ms\n", stats.source_copy_ms);
        printf("h2d enqueue:       %.3f ms\n", stats.h2d_enqueue_ms);
        printf("d2d enqueue:       %.3f ms\n", stats.d2d_enqueue_ms);
        printf("final wait:        %.3f ms\n", stats.final_wait_ms);
        printf("total load:        %.3f ms\n", stats.total_ms);
        printf("rss:               %" PRIu64 " bytes\n", rss);
        printf("peak rss:          %" PRIu64 " bytes\n", peak_rss);
        printf("mem available:     %" PRIu64 " bytes\n", mem_avail);
        printf("cuda free before:  %" PRIu64 "\n", before.cuda_free_bytes);
        printf("cuda free after:   %" PRIu64 "\n", after.cuda_free_bytes);
    }

    q3_loader_context_destroy(context);
    q3_gguf_close(m);
    return stats.coverage_ok ? 0 : 1;
}

int main(int argc, char **argv) {
    q3_options opt = {0};
    q3_mode mode = Q3_MODE_NONE;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--platform") == 0) {
            mode = Q3_MODE_PLATFORM;
        } else if (strcmp(a, "--inspect") == 0) {
            mode = Q3_MODE_INSPECT;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--list-tensors") == 0) {
            mode = Q3_MODE_LIST_TENSORS;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--memory-plan") == 0) {
            mode = Q3_MODE_MEMORY_PLAN;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--load-only") == 0) {
            mode = Q3_MODE_LOAD_ONLY;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--bench-q4-linear") == 0) {
            mode = Q3_MODE_BENCH_Q4_LINEAR;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--forward") == 0) {
            mode = Q3_MODE_FORWARD;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--bench-decisions") == 0) {
            mode = Q3_MODE_BENCH_DECISIONS;
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.model_path = argv[++i];
        } else if (strcmp(a, "--workload") == 0) {
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.workload_path = argv[++i];
        } else if (strcmp(a, "--tokens") == 0) {
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.tokens_path = argv[++i];
        } else if (strcmp(a, "--candidate-token-ids") == 0) {
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.candidate_ids = argv[++i];
        } else if (strcmp(a, "--tensor") == 0) {
            if (i + 1 >= argc) { usage(stderr); return 2; }
            opt.tensor_name = argv[++i];
        } else if (strcmp(a, "--batch") == 0) {
            if (i + 1 >= argc) { usage(stderr); return 2; }
            const char *spec = argv[++i];
            opt.batch_count = 0;
            while (*spec && opt.batch_count < 16) {
                char *end = NULL;
                const long v = strtol(spec, &end, 10);
                if (end == spec || v <= 0) break;
                opt.batches[opt.batch_count++] = (int) v;
                spec = (*end == ',') ? end + 1 : end;
            }
            if (opt.batch_count == 0) {
                fprintf(stderr, "q3: --batch expects a positive list\n");
                return 2;
            }
        } else if (strcmp(a, "--json") == 0) {
            opt.json = true;
        } else if (strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "q3: unknown argument '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (mode == Q3_MODE_NONE) {
        usage(stderr);
        return 2;
    }

    int rc;
    switch (mode) {
    case Q3_MODE_PLATFORM:     rc = cmd_platform(&opt); break;
    case Q3_MODE_INSPECT:      rc = cmd_inspect(&opt); break;
    case Q3_MODE_LIST_TENSORS: rc = cmd_list_tensors(&opt); break;
    case Q3_MODE_MEMORY_PLAN:  rc = cmd_memory_plan(&opt); break;
    case Q3_MODE_LOAD_ONLY:    rc = cmd_load_only(&opt); break;
    case Q3_MODE_BENCH_Q4_LINEAR: rc = q3_cmd_bench_q4_linear(&opt); break;
    case Q3_MODE_FORWARD:       rc = q3_cmd_forward(&opt); break;
    case Q3_MODE_BENCH_DECISIONS: rc = q3_cmd_bench_decisions(&opt); break;
    default:                   rc = 2; break;
    }

    q3_cuda_cleanup();
    return rc;
}
