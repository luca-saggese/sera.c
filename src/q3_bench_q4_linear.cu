/* SPDX-License-Identifier: MIT
 *
 * q3_bench_q4_linear.cu - M1 --bench-q4-linear diagnostic.
 *
 * Benchmarks the resident Q4_K quantized linear primitive on a real
 * projection matrix of the loaded model, at a few batch sizes. This is a
 * diagnostic surface, not a benchmark framework: it loads the model,
 * attaches the persistent kernel workspace, runs one warmup plus three
 * measured iterations per batch, and reports the median.
 */
#include "q3.h"
#include "q3_cuda.h"
#include "q3_gguf.h"
#include "q3_model_loader_cuda.h"
#include "q3_q4_linear.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * M1 --bench-q4-linear
 *
 * Loads the real model, picks a real Q4_K projection matrix, attaches the
 * persistent kernel workspace, and measures the resident quantized linear at
 * a few batch sizes. It answers one question: how fast is Q4_K weight x FP32
 * activation on this GB10, while the weights stay Q4_K resident.
 * ========================================================================= */

static bool tensor_is_q4_k(const q3_resident_tensor *t) {
    return t && t->qtype == 12 /* GGML_TYPE_Q4_K */;
}

static const q3_resident_tensor *find_tensor(const q3_loader_context *ctx,
                                             const char *name,
                                             char *err, size_t err_len) {
    const size_t n = q3_loader_tensor_count(ctx);
    if (name) {
        for (size_t i = 0; i < n; i++) {
            const q3_resident_tensor *t = q3_loader_tensor(ctx, i);
            if (t->name && strcmp(t->name, name) == 0) return t;
        }
        snprintf(err, err_len, "tensor '%s' not found", name);
        return nullptr;
    }
    /* No name given: use the largest resident Q4_K matrix. */
    const q3_resident_tensor *best = nullptr;
    for (size_t i = 0; i < n; i++) {
        const q3_resident_tensor *t = q3_loader_tensor(ctx, i);
        if (!tensor_is_q4_k(t) || !t->resident) continue;
        if (!best || t->bytes > best->bytes) best = t;
    }
    if (!best) snprintf(err, err_len, "no resident Q4_K tensor found");
    return best;
}

extern "C" int q3_cmd_bench_q4_linear(const q3_options *opt) {
    char err[256] = {0};

    q3_gguf *m = q3_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) { fprintf(stderr, "q3: %s\n", err); return 1; }

    if (q3_cuda_init() != 0) {
        fprintf(stderr, "q3: CUDA initialization failed\n");
        q3_gguf_close(m);
        return 1;
    }

    q3_loader_context *ctx = q3_loader_context_create(err, sizeof(err));
    if (!ctx) { fprintf(stderr, "q3: %s\n", err); q3_gguf_close(m); return 1; }

    if (!q3_loader_load(ctx, m, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    const q3_resident_tensor *tensor =
        find_tensor(ctx, opt->tensor_name, err, sizeof(err));
    if (!tensor) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }
    if (!tensor->resident || !tensor->ptr) {
        fprintf(stderr, "q3: tensor '%s' is not resident\n", tensor->name);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    q3_q4_linear_geometry geo;
    if (!q3_q4_linear_bind_geometry(tensor->qtype, tensor->rows, tensor->cols,
                                    &geo, err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Batches, default 1,4,16,32. */
    int batches[16];
    int batch_count = 0;
    if (opt->batch_count > 0) {
        for (int i = 0; i < opt->batch_count && i < 16; i++)
            batches[batch_count++] = opt->batches[i];
    } else {
        const int defaults[] = { 1, 4, 16, 32 };
        for (int i = 0; i < 4; i++) batches[batch_count++] = defaults[i];
    }

    int max_m = 0;
    for (int i = 0; i < batch_count; i++)
        if (batches[i] > max_m) max_m = batches[i];

    if (!q3_cuda_q4k_linear_init(0, (size_t) max_m, geo.N, geo.K,
                                 err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    /* Persistent kernel scratch, allocated once for the process. */
    const size_t workspace_bytes = q3_cuda_q4k_linear_workspace_bytes();
    void *workspace = nullptr;
    if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
        fprintf(stderr, "q3: workspace cudaMalloc(%zu) failed\n", workspace_bytes);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }
    if (!q3_cuda_q4k_linear_set_workspace(workspace, workspace_bytes,
                                          err, sizeof(err))) {
        fprintf(stderr, "q3: %s\n", err);
        cudaFree(workspace);
        q3_loader_context_destroy(ctx);
        q3_gguf_close(m);
        return 1;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    /* Device activation (deterministic sin/cos) and output, sized for the
     * largest batch. Input is [M, K], output is [M, N]. */
    const size_t k = (size_t) geo.K;
    const size_t n = (size_t) geo.N;
    float *input = nullptr, *output = nullptr;
    cudaMalloc(&input, (size_t) max_m * k * sizeof(float));
    cudaMalloc(&output, (size_t) max_m * n * sizeof(float));

    float *host_in = (float *) malloc((size_t) max_m * k * sizeof(float));
    for (size_t i = 0; i < (size_t) max_m * k; i++) {
        const double x = (double) (i % 97) * 0.0173 + (double) (i / 97) * 0.00071;
        host_in[i] = (float) (0.41 * sin(x) + 0.23 * cos(3.1 * x));
    }
    cudaMemcpy(input, host_in, (size_t) max_m * k * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);

    struct bench_row {
        int batch;
        const char *path;
        double quant_ms, kernel_ms, total_ms;
        double tokens_per_s, weight_gbps;
        uint64_t launches, allocs, syncs, workspace;
    };
    struct bench_row rows[16];
    memset(rows, 0, sizeof(rows));

    const uint64_t weight_bytes = geo.weight_bytes;
    q3_q4_linear_stats stats;

    for (int bi = 0; bi < batch_count; bi++) {
        const int batch = batches[bi];
        const char *path = q3_cuda_q4k_linear_path_for_batch((size_t) batch);

        /* 1 warmup + 3 measured runs; the median is reported. */
        double q[3] = {0}, k[3] = {0}, tot[3] = {0};
        for (int r = 0; r < 4; r++) {
            stats = (q3_q4_linear_stats){0};
            if (!q3_cuda_q4k_linear(&geo, tensor->ptr, input, (size_t) batch,
                                    output, stream, &stats, err, sizeof(err))) {
                fprintf(stderr, "q3: %s\n", err);
                break;
            }
            cudaStreamSynchronize(stream);
            q3_cuda_q4k_linear_read_stats(&stats, stream);
            if (r > 0) {
                q[r - 1] = stats.activation_quant_ms;
                k[r - 1] = stats.kernel_ms;
                tot[r - 1] = stats.total_ms;
                rows[bi].launches = stats.kernel_launches;
                rows[bi].allocs = stats.cuda_allocations;
                rows[bi].syncs = stats.host_syncs;
                rows[bi].workspace = workspace_bytes;
            }
        }
        for (int a = 0; a < 3; a++) {
            for (int b = a + 1; b < 3; b++) {
                if (tot[b] < tot[a]) {
                    double t;
                    t = tot[a]; tot[a] = tot[b]; tot[b] = t;
                    t = q[a]; q[a] = q[b]; q[b] = t;
                    t = k[a]; k[a] = k[b]; k[b] = t;
                }
            }
        }
        rows[bi].batch = batch;
        rows[bi].path = path;
        rows[bi].quant_ms = q[1];
        rows[bi].kernel_ms = k[1];
        rows[bi].total_ms = tot[1];
        rows[bi].tokens_per_s = tot[1] > 0.0
            ? (double) batch / (tot[1] / 1000.0) : 0.0;
        rows[bi].weight_gbps = tot[1] > 0.0
            ? (double) weight_bytes / (tot[1] / 1000.0) / 1e9
            : 0.0;
    }

    if (opt->json) {
        printf("{\n");
        printf("  \"tensor\": \"%s\",\n", tensor->name);
        printf("  \"qtype\": \"Q4_K\",\n");
        printf("  \"rows\": %u,\n", tensor->rows);
        printf("  \"cols\": %u,\n", tensor->cols);
        printf("  \"N\": %d,\n", geo.N);
        printf("  \"K\": %d,\n", geo.K);
        printf("  \"weight_bytes\": %" PRIu64 ",\n", weight_bytes);
        printf("  \"workspace_bytes\": %zu,\n", workspace_bytes);
        printf("  \"results\": [\n");
        for (int i = 0; i < batch_count; i++) {
            printf("    {\n");
            printf("      \"batch\": %d,\n", rows[i].batch);
            printf("      \"path\": \"%s\",\n", rows[i].path);
            printf("      \"activation_quant_ms\": %.4f,\n", rows[i].quant_ms);
            printf("      \"kernel_ms\": %.4f,\n", rows[i].kernel_ms);
            printf("      \"total_ms\": %.4f,\n", rows[i].total_ms);
            printf("      \"tokens_per_s\": %.2f,\n", rows[i].tokens_per_s);
            printf("      \"effective_weight_gbps\": %.2f,\n", rows[i].weight_gbps);
            printf("      \"kernel_launches\": %" PRIu64 ",\n", rows[i].launches);
            printf("      \"cuda_allocations\": %" PRIu64 ",\n", rows[i].allocs);
            printf("      \"host_syncs\": %" PRIu64 ",\n", rows[i].syncs);
            printf("      \"workspace_bytes\": %" PRIu64 "\n", rows[i].workspace);
            printf("    }%s\n", i + 1 < batch_count ? "," : "");
        }
        printf("  ]\n}\n");
    } else {
        printf("tensor:        %s\n", tensor->name);
        printf("qtype:         Q4_K\n");
        printf("N (features):  %d\n", geo.N);
        printf("K (inputs):    %d\n", geo.K);
        printf("weight bytes:  %" PRIu64 "\n", weight_bytes);
        printf("workspace:     %zu bytes\n", workspace_bytes);
        printf("\n%4s %6s %12s %12s %12s %10s %10s %6s %6s\n",
               "M", "path", "quant ms", "kernel ms", "total ms",
               "tok/s", "weight GB/s", "alloc", "sync");
        for (int i = 0; i < batch_count; i++) {
            printf("%4d %6s %12.4f %12.4f %12.4f %10.2f %10.2f %6" PRIu64
                   " %6" PRIu64 "\n",
                   rows[i].batch, rows[i].path, rows[i].quant_ms,
                   rows[i].kernel_ms, rows[i].total_ms, rows[i].tokens_per_s,
                   rows[i].weight_gbps, rows[i].allocs, rows[i].syncs);
        }
    }

    free(host_in);
    cudaFree(input);
    cudaFree(output);
    cudaStreamDestroy(stream);
    cudaFree(workspace);
    q3_loader_context_destroy(ctx);
    q3_gguf_close(m);
    return 0;
}

