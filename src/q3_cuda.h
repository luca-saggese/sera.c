#ifndef Q3_CUDA_H
#define Q3_CUDA_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "q3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q3_cuda — narrow CUDA primitive surface for M0.
 *
 * Only platform interrogation and device lifecycle. No tensor allocators, no
 * attention/cache semantics, no MoE or host-registration primitives: those
 * belong to M1+ and are introduced behind an explicit, measured API.
 * ========================================================================= */

/* Interrogate the CUDA device. Returns 0 on success, non-zero on failure.
 * On failure the platform is considered unsupported and must be refused. */
int q3_cuda_probe(q3_platform_info *out);

typedef struct {
    int max_shared_memory_per_block;
    int max_shared_memory_per_block_optin;
    int warp_size;
    int multiprocessor_count;
    int cc_major;
    int cc_minor;
} q3_cuda_shared_memory_info;

int q3_cuda_get_shared_memory_info(q3_cuda_shared_memory_info *out);

/* Minimal device lifecycle. Returns 0 on success. */
int q3_cuda_init(void);
void q3_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* Q3_CUDA_H */
