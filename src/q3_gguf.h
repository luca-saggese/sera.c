#ifndef Q3_GGUF_H
#define Q3_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q3_gguf — GGUF v3 parser core, extracted from ds4.c and isolated from any
 * model-family binding.
 *
 * The loader maps the file once (read-only, private) and leaves tensor bytes
 * in place. M0 never host-registers or allocates the payload: it only records
 * metadata and tensor descriptors. Values stay in the mmap; string keys point
 * into it.
 * ========================================================================= */

#define Q3_GGUF_MAGIC 0x46554747u /* "GGUF", little endian */
#define Q3_MAX_DIMS   8

typedef struct {
    const char *ptr;
    uint64_t len;
} q3_str;

/* One metadata key/value: key points into the mmap, value_pos is the offset
 * where the value begins (already decoded on demand). */
typedef struct {
    q3_str key;
    uint32_t type;
    uint64_t value_pos;
} q3_kv;

typedef struct {
    q3_str name;
    uint32_t ndim;
    uint64_t dim[Q3_MAX_DIMS];
    uint32_t type;
    const void *data;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} q3_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    q3_kv *kv;
    q3_tensor *tensors;
    bool native_nvfp4;
} q3_gguf;

/* Open + map + parse. On failure returns NULL and writes a message into
 * err_buf (if non-NULL, cap err_len). Never allocates the payload. */
q3_gguf *q3_gguf_open(const char *path, char *err_buf, size_t err_len);

void q3_gguf_close(q3_gguf *m);

/* Metadata accessors. Return false when the key is absent or wrong-typed. */
q3_kv *q3_gguf_find_kv(const q3_gguf *m, const char *key);
bool q3_gguf_get_string(const q3_gguf *m, const char *key, q3_str *out);
bool q3_gguf_get_u32(const q3_gguf *m, const char *key, uint32_t *out);
bool q3_gguf_get_u64(const q3_gguf *m, const char *key, uint64_t *out);
bool q3_gguf_get_bool(const q3_gguf *m, const char *key, bool *out);

/* Tensor-type helpers. */
const char *q3_gguf_type_name(uint32_t type);
bool q3_gguf_type_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);

/* Return a read-only pointer into the mmap-backed tensor payload. */
const void *q3_gguf_tensor_data(const q3_gguf *m, const q3_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif /* Q3_GGUF_H */
