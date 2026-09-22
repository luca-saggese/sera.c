/* q3_cuda.cu — narrow CUDA surface for M0.
 *
 * Only device interrogation and lifecycle. No tensor allocators, no
 * host-registration, no MoE assumptions.
 */

#include "q3_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

static int g_cuda_initialized = 0;

int q3_cuda_init(void) {
    if (g_cuda_initialized) return 0;
    /* cudaGetDeviceCount implicitly initializes the runtime; an explicit
     * setDevice is not needed for the single-device probe. */
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "q3: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    g_cuda_initialized = 1;
    return 0;
}

void q3_cuda_cleanup(void) {
    if (!g_cuda_initialized) return;
    cudaDeviceReset();
    g_cuda_initialized = 0;
}

int q3_cuda_probe(q3_platform_info *out) {
    if (q3_cuda_init() != 0) return -1;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "q3: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    out->cuda_device_count = count;
    if (count < 1) {
        fprintf(stderr, "q3: no CUDA devices found\n");
        return -1;
    }

    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        out->cuda_device = dev;
    } else {
        out->cuda_device = 0;
    }

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    err = cudaGetDeviceProperties(&prop, out->cuda_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "q3: cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    out->cc_major = prop.major;
    out->cc_minor = prop.minor;
    snprintf(out->device_name, sizeof(out->device_name), "%s", prop.name);

    int driver_ver = 0, runtime_ver = 0;
    if (cudaDriverGetVersion(&driver_ver) == cudaSuccess) {
        snprintf(out->driver_version, sizeof(out->driver_version),
                 "%d.%d", driver_ver / 1000, (driver_ver % 1000) / 10);
    }
    if (cudaRuntimeGetVersion(&runtime_ver) == cudaSuccess) {
        snprintf(out->runtime_version, sizeof(out->runtime_version),
                 "%d.%d", runtime_ver / 1000, (runtime_ver % 1000) / 10);
    }

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        out->cuda_free_bytes = (uint64_t)free_b;
        out->cuda_total_bytes = (uint64_t)total_b;
    }

    return 0;
}

int q3_cuda_get_shared_memory_info(q3_cuda_shared_memory_info *out) {
    if (!out || q3_cuda_init() != 0) return -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return -1;
    out->max_shared_memory_per_block = prop.sharedMemPerBlock;
    out->max_shared_memory_per_block_optin = prop.sharedMemPerBlockOptin;
    out->warp_size = prop.warpSize;
    out->multiprocessor_count = prop.multiProcessorCount;
    out->cc_major = prop.major;
    out->cc_minor = prop.minor;
    return 0;
}
