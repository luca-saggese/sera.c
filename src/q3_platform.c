/* q3_platform.c — host platform guard.
 *
 * M0 target is DGX Spark (GB10 / SM 12.1). Host-side memory comes from
 * /proc; the CUDA side is supplied by q3_cuda.cu. The platform is refused
 * explicitly when it does not match.
 */

#include "q3_platform.h"

#include "q3_cuda.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int q3_platform_host_memory(uint64_t *total_bytes, uint64_t *available_bytes) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return -1;
    char line[256];
    int found_total = 0, found_avail = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long kb = 0;
        if (!found_total && sscanf(line, "MemTotal: %lu kB", &kb) == 1) {
            *total_bytes = (uint64_t)kb * 1024ull;
            found_total = 1;
        } else if (!found_avail && sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
            *available_bytes = (uint64_t)kb * 1024ull;
            found_avail = 1;
        }
        if (found_total && found_avail) break;
    }
    fclose(fp);
    return (found_total && found_avail) ? 0 : -1;
}

int q3_platform_rss_bytes(uint64_t *rss_bytes) {
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp) return -1;
    unsigned long pages = 0;
    unsigned long ignored_size = 0;
    int rc = fscanf(fp, "%lu %lu", &ignored_size, &pages);
    fclose(fp);
    if (rc != 2) return -1;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    *rss_bytes = (uint64_t)pages * (uint64_t)page_size;
    return 0;
}

int q3_platform_validate(const q3_platform_info *info,
                          char *reason, size_t reason_len) {
    if (reason && reason_len > 0) reason[0] = '\0';
    if (info->cuda_device_count != 1) {
        snprintf(reason, reason_len,
                 "expected exactly 1 CUDA device, found %d",
                 info->cuda_device_count);
        return -1;
    }
    if (info->cc_major != 12 || info->cc_minor != 1) {
        snprintf(reason, reason_len,
                 "unsupported compute capability sm_%d%d (require sm_121)",
                 info->cc_major, info->cc_minor);
        return -1;
    }
    return 0;
}

int q3_platform_probe(q3_platform_info *out, char *reason, size_t reason_len) {
    if (reason && reason_len > 0) reason[0] = '\0';

    memset(out, 0, sizeof(*out));
    out->cuda_device = -1;

    if (q3_platform_host_memory(&out->mem_total_bytes,
                                 &out->mem_available_bytes) != 0) {
        snprintf(reason, reason_len,
                 "host memory unavailable (not Linux /proc?)");
        return -1;
    }

    if (q3_cuda_probe(out) != 0) {
        snprintf(reason, reason_len, "CUDA probe failed");
        return -1;
    }

    /* GB10 / SM 12.1 guard. No silent degradation. */
    return q3_platform_validate(out, reason, reason_len);
}
