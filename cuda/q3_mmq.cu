/* SPDX-License-Identifier: MIT
 *
 * q3_mmq.cu - host wrappers for the q3 Q4_K dense quantized matmul.
 *
 * Ported from the q38-main donor (COPY -> RENAME -> EDIT):
 *   _reference/q38-main/cuda/mmq/ds4_mmq.cu
 *     q38-main HEAD c1d4597a80e300b803dc642519718f2c999589da (MIT)
 *
 * The native device kernels in cuda/mmq/ are llama.cpp ggml-cuda files
 * copied verbatim by q38 (MIT, (c) 2023-2026 The ggml authors); see
 * THIRD_PARTY_NOTES.md for provenance. No ggml runtime, graph, scheduler or
 * allocator participates: the vendored headers resolve ggml.h / ggml-impl.h /
 * ggml-cuda.h to cuda/mmq/q3_ggml_stubs.h.
 *
 * M1 scope: Q4_K weights x FP32 activations -> FP32, dense only, Q8_1
 * activation quantization, GB10 / SM121. Everything DS4-specific was removed
 * (MoE, expert routing, MMID, D2R, MXFP4/NVFP4, Q2_K/IQ2_XXS/Q8_0 weights,
 * environment tuning knobs, CUDA graphs, fused epilogues).
 */
#include "q3_mmq.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmvq.cuh"

#include <cstdio>
#include <cstdlib>

/* mmq.cuh only declares the mul_mat_q_case<type> specializations; the donor
 * defines them one per type in ds4_mmq.cu. M2 instantiates Q4_K (embedding and
 * most projections) and Q6_K (output.weight, and attn_v / ffn_down on 32 of
 * the 64 layers). Nothing else is instantiated: no Q2_K/IQ2/Q8_0 paths. */
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);

/* ------------------------------------------------------------------ *
 * Optional enqueue-side timing. Two event pairs bracket the activation
 * quantization and the matmul launch. Creation is one-time; reading is a
 * host call and only happens after the caller synchronized, never inside
 * the primitive's normal path.
 * ------------------------------------------------------------------ */
struct q3_mmq_timings {
    cudaEvent_t quant_begin;
    cudaEvent_t quant_end;
    cudaEvent_t kernel_begin;
    cudaEvent_t kernel_end;
};

extern "C" q3_mmq_timings *q3_mmq_timings_create(cudaStream_t stream) {
    (void) stream;
    q3_mmq_timings *t = new q3_mmq_timings();
    cudaEventCreate(&t->quant_begin);
    cudaEventCreate(&t->quant_end);
    cudaEventCreate(&t->kernel_begin);
    cudaEventCreate(&t->kernel_end);
    return t;
}

extern "C" void q3_mmq_timings_destroy(q3_mmq_timings *t) {
    if (!t) return;
    cudaEventDestroy(t->quant_begin);
    cudaEventDestroy(t->quant_end);
    cudaEventDestroy(t->kernel_begin);
    cudaEventDestroy(t->kernel_end);
    delete t;
}

extern "C" void q3_mmq_read_timings(q3_mmq_timings *t,
                                    double *quant_ms, double *kernel_ms) {
    if (!t) return;
    float ms = 0.0f;
    if (quant_ms) {
        cudaEventElapsedTime(&ms, t->quant_begin, t->quant_end);
        *quant_ms = ms;
    }
    if (kernel_ms) {
        cudaEventElapsedTime(&ms, t->kernel_begin, t->kernel_end);
        *kernel_ms = ms;
    }
}

namespace {

/* ------------------------------------------------------------------ *
 * Peak shared memory the mul_mat_q tile kernel will request on this
 * device for the donor's own mmq_x / nwarps selection. Needed so the
 * caller can raise our private pool high-water mark above it and keep
 * the hot path allocation-free.
 * ------------------------------------------------------------------ */
size_t q3_mmq_tile_shared_bytes(int cc, int warp_size) {
    const int mmq_x = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);
    const int nwarps = mmq_get_nwarps_host(cc, warp_size);
    const size_t q4 = mmq_get_nbytes_shared<GGML_TYPE_Q4_K>(mmq_x, mmq_y, cc, warp_size, nwarps);
    const size_t q6 = mmq_get_nbytes_shared<GGML_TYPE_Q6_K>(mmq_x, mmq_y, cc, warp_size, nwarps);
    return q4 > q6 ? q4 : q6;
}

/* ------------------------------------------------------------------ *
 * Persistent scratch arena, owned by the caller (the q3 runtime).
 *
 * Everything transient that the Q4_K dense path needs - the Q8_1 activation
 * buffer and the stream-K fixup scratch allocated inside mul_mat_q_case -
 * is sliced out of this one arena, so the hot path performs no
 * cudaMalloc/cudaFree once warm.
 * ------------------------------------------------------------------ */
unsigned char * g_arena_base     = nullptr;
size_t          g_arena_bytes    = 0;

/* Single bump pointer, shared with the pool the vendored kernel reaches
 * through ctx.pool() (cuda/mmq/q3_ggml_stubs.cu keeps the same cursor).
 * Sharing it is essential: the stream-K fixup scratch is allocated from the
 * pool *after* the Q8_1 activation buffer was already sliced out here, so a
 * second independent cursor would hand back the activation buffer again and
 * have the fixup pass zero the matmul result. */
void * arena_alloc(size_t bytes) {
    const size_t aligned = (bytes + 255u) & ~(size_t)255u;
    if (!g_arena_base || q3_arena_cursor + aligned > g_arena_bytes) {
        fprintf(stderr, "q3_mmq: arena too small (%zu + %zu > %zu)\n",
                q3_arena_cursor, aligned, g_arena_bytes);
        std::abort();
    }
    void * ptr = g_arena_base + q3_arena_cursor;
    q3_arena_cursor += aligned;
    return ptr;
}

/* Reset the shared cursor before every launcher invocation. */
void arena_begin(void) {
    if (g_arena_base) q3_arena_set(g_arena_base, g_arena_bytes);
}

/* Deterministic zeroing of the Q8_1 staging tail.
 *
 * quantize_mmq_q8_1_cuda writes only the valid columns, but the MMQ kernel
 * unconditionally loads the full column tile (including a tail over-read of
 * get_mmq_x_max_host() blocks). With a reused buffer that tail would be
 * stale, so it is zeroed every call: a zero q8_1 block contributes 0 to the
 * dot product, which makes the masked-out tail deterministic.
 *
 * Ported from the donor's S1.1a fix; its DS4_MMQ_YBUF_MEMSET environment
 * override was dropped, the safe behaviour is now unconditional.
 * ------------------------------------------------------------------ */
void ybuf_memset(void *ptr, size_t bytes, cudaStream_t stream) {
    if (!ptr || !bytes) return;
    cudaMemsetAsync(ptr, 0, bytes, stream);
}

} // anonymous namespace

/* ------------------------------------------------------------------ *
 * Device singleton context.
 *
 * The donor kept a ggml_backend_cuda_context per device purely so
 * mul_mat_q_case() could reach a pool and the device-info singleton. The
 * object shape is kept because the vendored signature expects it; its pool
 * is backed by our arena (cuda/mmq/q3_ggml_stubs.cu).
 * ------------------------------------------------------------------ */
static ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

extern "C" int q3_mmq_init(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return -1;
    ggml_cuda_set_device(device);
    (void) ggml_cuda_info(); /* populate the lazy device-info singleton */
    return get_ctx_for_device(device) ? 0 : -1;
}

extern "C" void q3_mmq_pool_set_stream(cudaStream_t stream) {
    (void) stream; /* the aria is stream-agnostic; kept for ABI completeness */
}

/* Bytes needed by the Q8_1 activation stage for a given geometry.
 * mmq_path != 0 selects the MMQ (interleaved) layout, else the canonical
 * block_q8_1 layout used by MMVQ. */
extern "C" size_t q3_mmq_q8_1_workspace_bytes(int M, int N, int K, int mmq_path) {
    (void) M;
    if (K <= 0 || N <= 0) return 0;
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);

    if (mmq_path) {
        return (size_t) N * (size_t) ne10_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1) +
               (size_t) get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    }
    return (size_t) N * (size_t) ne10_padded * sizeof(block_q8_1) / QK8_1;
}

/* Attach the caller-owned scratch arena. One-time. */
extern "C" int q3_mmq_set_arena(void *ptr, size_t bytes) {
    g_arena_base  = (unsigned char *) ptr;
    g_arena_bytes = bytes;
    q3_arena_set(g_arena_base, g_arena_bytes);
    return 0;
}

/* Caller-side helper: report the arena size the dense path needs so the
 * runtime can reserve it together with the activation workspace. */
extern "C" size_t q3_mmq_arena_bytes(void) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    size_t bytes = q3_mmq_tile_shared_bytes(cc,
        ggml_cuda_info().devices[dev].warp_size);
    /* stream-K fixup scratch: at most nsm waves of mmq_x*mmq_y floats. */
    const size_t nsm   = (size_t) ggml_cuda_info().devices[dev].nsm;
    const size_t mmq_x = (size_t) get_mmq_x_max_host(cc);
    const size_t mmq_y = (size_t) get_mmq_y_host(cc);
    bytes += (nsm + 4) * mmq_x * mmq_y * sizeof(float);
    /* Q8_1 activation stage, worst case (MMQ interleaved layout, K padded). */
    bytes += (size_t) 1 << 21;
    return bytes + 8192;
}

/* ------------------------------------------------------------------ *
 * Dense Q4_K x FP32 -> FP32, MMQ tile path. All M.
 *
 * Mirrors the donor's ds4_mmq_dense_impl<GGML_TYPE_Q4_K> with the DS4-only
 * branches removed: no fp4 path, no env knobs, no facebook expert
 * handling.
 * ------------------------------------------------------------------ */
extern "C" int q3_mmq_q4_K_dense(const void *W, const float *X_f32,
                                 float *out_f32, int M, int N, int K,
                                 cudaStream_t stream,
                                 q3_mmq_timings *timings);

template <ggml_type type>
static int q3_mmq_dense_impl(const void *W, const float *X_f32,
                             float *out_f32, int M, int N, int K,
                             cudaStream_t stream, q3_mmq_timings *timings) {
    const char * tag = (type == GGML_TYPE_Q6_K) ? "q3_mmq_q6_K_dense"
                                                 : "q3_mmq_q4_K_dense";
    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    /* K-quant super-block constraint. Fail loudly: M2 has no slow fallback. */
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: no CUDA context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    const size_t  nbytes_src1_q8_1 =
        (size_t) N * (size_t) ne10_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1) +
        (size_t) get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    if (!g_arena_base) {
        fprintf(stderr, "%s: no persistent arena attached "
                        "(need >= %zu bytes)\n", tag, nbytes_src1_q8_1);
        return -1;
    }
    arena_begin();
    void * ybuf = arena_alloc(nbytes_src1_q8_1);

    if (timings) cudaEventRecord(timings->quant_begin, stream);

    /* Deterministic tail (donor S1.1a fix). */
    ybuf_memset(ybuf, nbytes_src1_q8_1, stream);

    /* The activation is FP32 and is always quantized to Q8_1, but the Q8_1
     * *layout* is keyed on the weight type, not on the destination shape:
     * mmq_get_q8_1_ds_layout() maps Q4_K -> DS4 (half2 scale+sum per 32
     * values) and Q6_K -> D4 (float scale per 32 values). Both layouts live
     * in the same block_q8_1_mmq union and have identical size/stride, so
     * only the type argument selects the correct interpretation. Passing
     * Q4_K for a Q6_K weight would write ds4[] where the kernel reads d4[],
     * producing garbage scales. */
    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, ybuf,
        type, /*ne00=*/K, /*s11=*/(int64_t) K,
        /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    if (timings) cudaEventRecord(timings->quant_end, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck = ggml_blck_size(type);       /* 256 for both K-quants */
    const int64_t s01  = (int64_t) K / blck;         /* weight blocks per row */
    const int64_t s1   = (int64_t) M;
    const int64_t s12  = (int64_t) N * ne10_padded * sizeof(block_q8_1_mmq) /
                         (4 * QK8_1 * sizeof(int));
    const int64_t s13  = s12;

    /* GB10 is >= VOLTA, so the donor enabled stream-K unconditionally. */
    const bool use_stream_k =
        GGML_CUDA_CC_IS_NVIDIA(cc) &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;

    cudaMemsetAsync(out_f32, 0, (size_t) M * (size_t) N * sizeof(float), stream);

    mmq_args args;
    args.x = (const char *) W;
    args.type_x = type;
    args.y = (const int *) ybuf;
    args.ids_dst = nullptr;
    args.expert_bounds = nullptr;
    args.dst = out_f32;
    args.ncols_x = K;
    args.nrows_x = M;
    args.ncols_dst = N;
    args.stride_row_x = s01;
    args.ncols_y = N;
    args.nrows_dst = s1;
    args.nchannels_x = 1;
    args.nchannels_y = 1;
    args.stride_channel_x = 0;
    args.stride_channel_y = s12;
    args.stride_channel_dst = 0;
    args.nsamples_x = 1;
    args.nsamples_y = 1;
    args.stride_sample_x = 0;
    args.stride_sample_y = s13;
    args.stride_sample_dst = 0;
    args.use_stream_k = use_stream_k;
    args.ncols_max = N;

    /* mul_mat_q_case slices its stream-K fixup scratch out of the same
     * arena through the pool returned by ctx.pool(). */
    if (timings) cudaEventRecord(timings->kernel_begin, stream);
    mul_mat_q_case<type>(*ctx, args, stream);
    if (timings) cudaEventRecord(timings->kernel_end, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int q3_mmq_q4_K_dense(const void *W, const float *X_f32,
                                 float *out_f32, int M, int N, int K,
                                 cudaStream_t stream,
                                 q3_mmq_timings *timings) {
    return q3_mmq_dense_impl<GGML_TYPE_Q4_K>(W, X_f32, out_f32, M, N, K, stream, timings);
}

extern "C" int q3_mmq_q6_K_dense(const void *W, const float *X_f32,
                                 float *out_f32, int M, int N, int K,
                                 cudaStream_t stream,
                                 q3_mmq_timings *timings) {
    return q3_mmq_dense_impl<GGML_TYPE_Q6_K>(W, X_f32, out_f32, M, N, K, stream, timings);
}

/* ------------------------------------------------------------------ *
 * Dense Q4_K x FP32 -> FP32, MMVQ matrix-vector path for small M.
 *
 * Ported from the donor's ds4_mmq_dense_vec_impl<T> (which was
 * instantiated for Q8_0 only). Q4_K uses the canonical block_q8_1
 * activation layout, not the MMQ-interleaved one.
 * ------------------------------------------------------------------ */
extern "C" int q3_mmq_q4_K_dense_vec(const void *W, const float *X_f32,
                                     float *out_f32, int M, int N, int K,
                                     cudaStream_t stream,
                                     q3_mmq_timings *timings);

template <ggml_type type>
static int q3_mmq_dense_vec_impl(const void *W, const float *X_f32,
                                 float *out_f32, int M, int N, int K,
                                 cudaStream_t stream, q3_mmq_timings *timings) {
    const char * tag = (type == GGML_TYPE_Q6_K) ? "q3_mmq_q6_K_dense_vec"
                                                 : "q3_mmq_q4_K_dense_vec";
    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    /* The MMVQ cap bounds the *batch* (ncols_dst), which is the donor's
     * second operand N - not the weight row count M. */
    if (N > Q3_MMVQ_MAX_BATCH) {
        fprintf(stderr, "%s: batch N=%d exceeds MMVQ cap %d\n",
                tag, N, Q3_MMVQ_MAX_BATCH);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: no CUDA context for device %d\n", tag, dev);
        return -1;
    }

    /* Canonical Q8_1 layout, indexed [K innermost, N]. */
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t) N * (size_t) ne10_padded *
                                sizeof(block_q8_1) / QK8_1;

    if (!g_arena_base) {
        fprintf(stderr, "%s: no persistent arena attached "
                        "(need >= %zu bytes)\n", tag, nbytes_q8_1);
        return -1;
    }
    arena_begin();
    void * ybuf = arena_alloc(nbytes_q8_1);

    if (timings) cudaEventRecord(timings->quant_begin, stream);

    /* Activation quantization is always Q4_K-keyed (layout only). */
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, ybuf,
        GGML_TYPE_Q4_K, /*ne00=*/K,
        /*s11=*/(int64_t) K, /*s12=*/(int64_t) K * N, /*s13=*/(int64_t) K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    if (timings) cudaEventRecord(timings->quant_end, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: activation quantization failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck    = ggml_blck_size(type);
    const int64_t s01_row = (int64_t) K / blck;
    const int64_t s11_y   = ne10_padded / QK8_1;
    const int64_t s12_y   = (int64_t) N * s11_y;
    const int64_t s1_dst  = (int64_t) M;

    ggml_cuda_mm_fusion_args_device fusion = {};

    cudaMemsetAsync(out_f32, 0, (size_t) M * (size_t) N * sizeof(float), stream);

    if (timings) cudaEventRecord(timings->kernel_begin, stream);
    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/ybuf,
        /*ids=*/nullptr, fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int) s01_row,
        /*stride_col_y=*/(int) s11_y,
        /*stride_col_dst=*/(int) s1_dst,
        /*nchannels_x=*/1, /*nchannels_y=*/1, /*nchannels_dst=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/(int) s12_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);
    if (timings) cudaEventRecord(timings->kernel_end, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mmvq launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int q3_mmq_q4_K_dense_vec(const void *W, const float *X_f32,
                                     float *out_f32, int M, int N, int K,
                                     cudaStream_t stream,
                                     q3_mmq_timings *timings) {
    return q3_mmq_dense_vec_impl<GGML_TYPE_Q4_K>(W, X_f32, out_f32, M, N, K, stream, timings);
}

extern "C" int q3_mmq_q6_K_dense_vec(const void *W, const float *X_f32,
                                     float *out_f32, int M, int N, int K,
                                     cudaStream_t stream,
                                     q3_mmq_timings *timings) {
    return q3_mmq_dense_vec_impl<GGML_TYPE_Q6_K>(W, X_f32, out_f32, M, N, K, stream, timings);
}