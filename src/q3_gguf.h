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
#define Q3_MAX_NAME   128

typedef struct {
    const char *ptr;
    uint64_t len;
} q3_str;

/* GGUF metadata value types (container format, not model-family). */
enum {
    Q3_GGUF_VALUE_UINT8   = 0,
    Q3_GGUF_VALUE_INT8    = 1,
    Q3_GGUF_VALUE_UINT16  = 2,
    Q3_GGUF_VALUE_INT16   = 3,
    Q3_GGUF_VALUE_UINT32  = 4,
    Q3_GGUF_VALUE_INT32   = 5,
    Q3_GGUF_VALUE_FLOAT32 = 6,
    Q3_GGUF_VALUE_BOOL    = 7,
    Q3_GGUF_VALUE_STRING  = 8,
    Q3_GGUF_VALUE_ARRAY   = 9,
    Q3_GGUF_VALUE_UINT64  = 10,
    Q3_GGUF_VALUE_INT64   = 11,
    Q3_GGUF_VALUE_FLOAT64 = 12,
};

/* One metadata key/value: key points into the mmap, value_pos is the offset
 * where the value begins (already decoded on demand). */
typedef struct {
    q3_str key;
    uint32_t type;
    uint64_t value_pos;
} q3_kv;

typedef struct {
    q3_str name;
    /* NUL-terminated copy of name. The mmap'd bytes are not terminated, so
     * callers that need a C string (resident descriptor table, list-tensors)
     * use this instead of name.ptr. */
    char name_buf[Q3_MAX_NAME];
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

/* Float accessor: accepts FLOAT32 and FLOAT64 GGUF values and yields float.
 * Model configs store rope theta / rms eps as FLOAT32 in every GGUF seen so
 * far, but the parser must not assume that. */
bool q3_gguf_get_f32(const q3_gguf *m, const char *key, float *out);

/* Raw metadata value access: exposes the mmap'd value bytes and the GGUF
 * value type so model-family code (e.g. the config binder) can decode any
 * scalar without the container growing family-specific accessors. */
bool q3_gguf_get_value(const q3_gguf *m, const char *key, const void **ptr,
                       uint64_t *size, uint32_t *type);

/* Array metadata access. Decodes the GGUF array header (item type + count)
 * and returns a cursor over the items. `*item_type` is a Q3_GGUF_VALUE_* code
 * and `*count` the element count. The returned cursor is opaque; advance it
 * with q3_gguf_array_next_*(). Returns false when the key is absent, is not an
 * array, or the header is malformed. */
typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    uint32_t item_type;
    uint64_t count;
    uint64_t index;
} q3_gguf_array;

bool q3_gguf_get_array(const q3_gguf *m, const char *key, q3_gguf_array *out);

/* Read the next array element. `q3_gguf_array_next_string` yields a q3_str
 * pointing into the mmap; `q3_gguf_array_next_u32`/`_i32` yield the scalar.
 * Return false at end of array or on a type mismatch. */
bool q3_gguf_array_next_string(q3_gguf_array *a, q3_str *out);
bool q3_gguf_array_next_u32(q3_gguf_array *a, uint32_t *out);
bool q3_gguf_array_next_i32(q3_gguf_array *a, int32_t *out);
/* GGUF exporters write the same logical "small integer" array as either
 * INT32 or UINT32; token_type is UINT32 in the Qwen3 export. Accept both. */
bool q3_gguf_array_next_int(q3_gguf_array *a, int32_t *out);

/* Tensor-type helpers. */
const char *q3_gguf_type_name(uint32_t type);
bool q3_gguf_type_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);

/* Return a read-only pointer into the mmap-backed tensor payload. */
const void *q3_gguf_tensor_data(const q3_gguf *m, const q3_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif /* Q3_GGUF_H */
