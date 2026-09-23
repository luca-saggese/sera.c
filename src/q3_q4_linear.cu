/* SPDX-License-Identifier: MIT
 *
 * q3_q4_linear.cu - M1 resident Q4_K quantized linear primitive.
 *
 * Thin binder + launcher over the ported q38-main Q4_K dense kernels
 * (cuda/q3_mmq.cu). Responsibilities, and nothing more:
 *
 *   - resolve GGUF physical dims once into logical (N, K);
 *   - pick the internal path for the batch size;
 *   - enqueue activation quantization + kernel on the caller's stream;
 *   - report telemetry.
 *
 * It never copies across the host boundary, never synchronizes, and never
 * allocates inside the hot call: all scratch comes from a persistent arena
 * attached by q3_cuda_q4k_linear_set_workspace().
 */
#include "q3_q4_linear.h"

#include "q3_mmq.h"
#include "q3_quant.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

namespace {

void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) {
        error[0] = '\0';
        snprintf(error, error_len, "%s", message);
    }
}

size_t g_workspace_bytes = 0;
bool   g_initialized     = false;
int    g_device          = -1;

/* Persistent timing events. Created once; recording them costs nothing and
 * they are only read by the caller after it synchronizes. */
q3_mmq_timings *g_timings = nullptr;
cudaEvent_t     g_total_begin = nullptr;
cudaEvent_t     g_total_end   = nullptr;

} // anonymous namespace

extern "C" bool q3_q4_linear_bind_geometry(uint32_t qtype,
                                           uint32_t gguf_rows_dim0,
                                           uint32_t gguf_cols_dim1,
                                           q3_q4_linear_geometry *out,
                                           char *error, size_t error_len) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));

    if (qtype != Q3_QUANT_Q4_K && qtype != Q3_QUANT_Q6_K) {
        set_error(error, error_len,
                  "q3_q4_linear: tensor qtype is not Q4_K/Q6_K");
        return false;
    }
    /* GGUF dim[0] is the contiguous row length = K; dim[1] is the row count
     * = N. The M0 descriptor exposes them as rows/cols respectively, so the
     * caller passes rows as dim0 and cols as dim1. */
    const uint64_t K = gguf_rows_dim0;
    const uint64_t N = gguf_cols_dim1;
    if (K == 0 || N == 0) {
        set_error(error, error_len, "q3_q4_linear: degenerate geometry");
        return false;
    }
    if (K % Q3_QUANT_QK_K != 0) {
        char buffer[160];
        snprintf(buffer, sizeof(buffer),
                 "q3_q4_linear: K=%llu must be a multiple of %d",
                 (unsigned long long) K, Q3_QUANT_QK_K);
        set_error(error, error_len, buffer);
        return false;
    }
    if (K > INT32_MAX || N > INT32_MAX) {
        set_error(error, error_len, "q3_q4_linear: geometry exceeds int32");
        return false;
    }

    out->K = (int32_t) K;
    out->N = (int32_t) N;
    out->qtype = qtype;
    out->M = 0; /* resolved per call from the activation batch */
    out->weight_bytes = N * (K / Q3_QUANT_QK_K) *
                        (qtype == Q3_QUANT_Q6_K ? Q3_QUANT_Q6_K_BLOCK_BYTES
                                                : Q3_QUANT_Q4_K_BLOCK_BYTES);
    return true;
}

extern "C" bool q3_cuda_q4k_linear_init(int device, size_t max_tokens,
                                        int64_t max_features, int64_t max_k,
                                        char *error, size_t error_len) {
    if (g_initialized && g_device == device) return true;
    if (device < 0) {
        set_error(error, error_len, "q3_q4_linear: invalid device index");
        return false;
    }
    if (q3_mmq_init(device) != 0) {
        set_error(error, error_len, "q3_q4_linear: q3_mmq_init failed");
        return false;
    }
    g_workspace_bytes = q3_mmq_arena_bytes(max_tokens, max_features, max_k);
    if (!g_timings) {
        g_timings = q3_mmq_timings_create(nullptr);
        cudaEventCreate(&g_total_begin);
        cudaEventCreate(&g_total_end);
    }
    g_device = device;
    g_initialized = true;
    return true;
}

extern "C" size_t q3_cuda_q4k_linear_workspace_bytes(void) {
    return g_workspace_bytes;
}

extern "C" bool q3_cuda_q4k_linear_set_workspace(void *device_ptr, size_t bytes,
                                                 char *error, size_t error_len) {
    if (!device_ptr || bytes < g_workspace_bytes) {
        char buffer[192];
        snprintf(buffer, sizeof(buffer),
                 "q3_q4_linear: workspace too small (%zu < %zu)",
                 bytes, g_workspace_bytes);
        set_error(error, error_len, buffer);
        return false;
    }
    if (q3_mmq_set_arena(device_ptr, bytes) != 0) {
        set_error(error, error_len, "q3_q4_linear: arena attach failed");
        return false;
    }
    return true;
}

extern "C" const char *q3_cuda_q4k_linear_path_for_batch(size_t token_count) {
    return token_count <= Q3_MMVQ_MAX_BATCH ? "MMVQ" : "MMQ";
}

extern "C" bool q3_cuda_q4k_linear(const q3_q4_linear_geometry *geometry,
                                   const void *weight,
                                   const float *device_input,
                                   size_t token_count,
                                   float *device_output,
                                   void *stream,
                                   q3_q4_linear_stats *stats,
                                   char *error, size_t error_len) {
    if (!geometry || !weight || !device_input || !device_output) {
        set_error(error, error_len, "q3_q4_linear: null argument");
        return false;
    }
    if (!g_initialized) {
        set_error(error, error_len, "q3_q4_linear: not initialized");
        return false;
    }
    if (token_count == 0 || token_count > INT32_MAX) {
        set_error(error, error_len, "q3_q4_linear: bad token count");
        return false;
    }
    if (geometry->N <= 0 || geometry->K <= 0) {
        set_error(error, error_len, "q3_q4_linear: unresolved geometry");
        return false;
    }

    q3_q4_linear_stats local;
    memset(&local, 0, sizeof(local));
    q3_q4_linear_stats *s = stats ? stats : &local;
    memset(s, 0, sizeof(*s));
    s->weight_bytes = geometry->weight_bytes;

    const int K = geometry->K;
    cudaStream_t cuda_stream = (cudaStream_t) stream;

    /* The donor kernel ABI takes the *weight* row count as its first operand
     * and the batch as its second (it is the reverse of the M1 naming). The
     * physical destination it writes is out[token * features + feature],
     * which is exactly the M1 logical [tokens, features] row-major layout, so
     * no transpose is needed - only this argument swap. */
    const int features = geometry->N;
    const int tokens   = (int) token_count;

    s->kernel_launches = 1;
    s->activation_read_bytes = token_count * (size_t) K * sizeof(float);
    s->activation_write_bytes = token_count * (size_t) features * sizeof(float);

    cudaEventRecord(g_total_begin, cuda_stream);

    int rc;
    if (tokens <= Q3_MMVQ_MAX_BATCH) {
        rc = (geometry->qtype == Q3_QUANT_Q6_K)
            ? q3_mmq_q6_K_dense_vec(weight, device_input, device_output,
                                    features, tokens, K, cuda_stream, g_timings)
            : q3_mmq_q4_K_dense_vec(weight, device_input, device_output,
                                    features, tokens, K, cuda_stream, g_timings);
    } else {
        rc = (geometry->qtype == Q3_QUANT_Q6_K)
            ? q3_mmq_q6_K_dense(weight, device_input, device_output,
                                features, tokens, K, cuda_stream, g_timings)
            : q3_mmq_q4_K_dense(weight, device_input, device_output,
                                features, tokens, K, cuda_stream, g_timings);
    }

    cudaEventRecord(g_total_end, cuda_stream);

    if (rc != 0) {
        char buffer[160];
        snprintf(buffer, sizeof(buffer),
                 "q3_q4_linear: %s path failed (rc=%d) for M=%d N=%d K=%d",
                 q3_cuda_q4k_linear_path_for_batch(token_count), rc, tokens,
                 features, K);
        set_error(error, error_len, buffer);
        return false;
    }

    /* Event reads below are host calls that wait for the recorded events, so
     * the primitive reports timings only when a caller asks for them (i.e.
     * from the benchmark harness). The compute path itself is unmodified. */
    return true;
}

/* Resolve the event-based timings once the caller synchronized the stream. */
extern "C" void q3_cuda_q4k_linear_read_stats(q3_q4_linear_stats *stats,
                                              void *stream) {
    if (!stats || !g_initialized) return;
    (void) stream;
    double quant_ms = 0.0, kernel_ms = 0.0;
    if (g_timings) q3_mmq_read_timings(g_timings, &quant_ms, &kernel_ms);
    stats->activation_quant_ms = quant_ms;
    stats->kernel_ms = kernel_ms;
    float total_ms = 0.0f;
    if (g_total_begin) cudaEventElapsedTime(&total_ms, g_total_begin, g_total_end);
    stats->total_ms = total_ms;
}