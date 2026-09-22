/* q3_main.c — q3 inspection CLI: platform probe, GGUF inventory, residency
 * plan and fast resident CUDA load.
 *
 * Ported from the q38 donor (COPY -> RENAME -> EDIT). The M0 surface is
 * deliberately narrow: no decode, no generation, no server, no tokenizer.
 */

#include "q3.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_memory.h"
#include "q3_platform.h"
#include "q3_residency_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        "\n"
        "options:\n"
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

static int cmd_memory_plan(const q3_options *opt) {
    char err[256];
    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
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
        char buf[1024];
        q3_memory_snapshot_json(&snap, buf, sizeof(buf));
        printf("%s\n", buf);
    } else {
        printf("model file:        %" PRIu64 " bytes\n", snap.model_file_bytes);
        printf("model mapped:      %" PRIu64 " bytes\n", snap.model_mapped_bytes);
        printf("rss:               %" PRIu64 " bytes\n", snap.rss_bytes);
        printf("mem available:     %" PRIu64 " bytes\n", snap.mem_available_bytes);
        printf("cuda free:         %" PRIu64 " bytes\n", snap.cuda_free_bytes);
        printf("cuda total:        %" PRIu64 " bytes\n", snap.cuda_total_bytes);
        printf("peak internal:     %" PRIu64 " bytes\n", snap.peak_internal_bytes);
    }

    q3_gguf_close(m);
    return 0;
}

static int cmd_load_only(const q3_options *opt) {
    (void)opt;
    fprintf(stderr, "q3: --load-only is not implemented yet\n");
    return 1;
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
    default:                   rc = 2; break;
    }

    q3_cuda_cleanup();
    return rc;
}
