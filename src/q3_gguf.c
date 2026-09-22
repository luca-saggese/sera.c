/* q3_gguf.c — GGUF v3 parser core for M0.
 *
 * Extracted from ds4.c (model_open / parse_metadata / parse_tensors / cursor
 * helpers) and isolated from all DeepSeek/GLM model-family bindings. No graph,
 * no tokenizer policy, no tensor placement: just the container format.
 */

#include "q3_gguf.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Cursor                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} q3_cursor;

static void cursor_error(q3_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(q3_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(q3_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(q3_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(q3_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(q3_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(q3_cursor *c, q3_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

static q3_cursor cursor_at(const q3_gguf *m, uint64_t pos) {
    q3_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.error[0] = '\0';
    return c;
}

/* ------------------------------------------------------------------ */
/* Type tables                                                         */
/* ------------------------------------------------------------------ */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = (uint32_t)(sizeof(gguf_types) / sizeof(gguf_types[0]));
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

const char *q3_gguf_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

bool q3_gguf_type_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

const void *q3_gguf_tensor_data(const q3_gguf *m, const q3_tensor *tensor) {
    if (tensor && tensor->data) return tensor->data;
    if (!m || !tensor || tensor->abs_offset > m->size ||
        tensor->bytes > m->size - tensor->abs_offset)
        return NULL;
    return m->map + tensor->abs_offset;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(q3_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        q3_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

/* ------------------------------------------------------------------ */
/* String helpers                                                      */
/* ------------------------------------------------------------------ */

static bool q3_streq(q3_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

/* ------------------------------------------------------------------ */
/* Parser                                                              */
/* ------------------------------------------------------------------ */

static void set_error(char *err_buf, size_t err_len, const char *msg) {
    if (err_buf && err_len > 0) {
        snprintf(err_buf, err_len, "%s", msg);
    }
}

static void parse_metadata(q3_gguf *m, q3_cursor *c) {
    if (m->n_kv > c->size - c->pos) {
        cursor_error(c, "GGUF metadata count exceeds file size");
        return;
    }
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) {
        cursor_error(c, "out of memory allocating metadata table");
        return;
    }

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        q3_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) return;
        if (!cursor_u32(c, &kv->type)) return;
        kv->value_pos = c->pos;

        if (q3_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32) {
            q3_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) return;
    }
}

static void parse_tensors(q3_gguf *m, q3_cursor *c) {
    if (m->n_tensors > c->size - c->pos) {
        cursor_error(c, "GGUF tensor count exceeds file size");
        return;
    }
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) {
        cursor_error(c, "out of memory allocating tensor table");
        return;
    }

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q3_tensor *t = &m->tensors[i];
        if (!cursor_string(c, &t->name)) return;
        {
            const uint64_t n = t->name.len < Q3_MAX_NAME - 1
                                   ? t->name.len : Q3_MAX_NAME - 1;
            memcpy(t->name_buf, t->name.ptr, (size_t) n);
            t->name_buf[n] = '\0';
        }
        if (!cursor_u32(c, &t->ndim)) return;
        if (t->ndim == 0 || t->ndim > Q3_MAX_DIMS) {
            cursor_error(c, "tensor has an unsupported number of dimensions");
            return;
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) return;
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                cursor_error(c, "tensor element count overflow");
                return;
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) return;
        if (!cursor_u64(c, &t->rel_offset)) return;

        if (!q3_gguf_type_nbytes(t->type, t->elements, &t->bytes)) {
            t->bytes = 0; /* unsupported type: flagged, not fatal */
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q3_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            cursor_error(c, "tensor offset overflow");
            return;
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            cursor_error(c, "tensor points outside GGUF file");
            return;
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

q3_gguf *q3_gguf_open(const char *path, char *err_buf, size_t err_len) {
    if (err_buf && err_len > 0) err_buf[0] = '\0';

    q3_gguf *m = calloc(1, sizeof(*m));
    if (!m) {
        set_error(err_buf, err_len, "out of memory");
        return NULL;
    }
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        set_error(err_buf, err_len, "cannot open model file");
        free(m);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        set_error(err_buf, err_len, "cannot stat model file");
        close(fd);
        free(m);
        return NULL;
    }
    if (st.st_size < 32) {
        set_error(err_buf, err_len, "model file is too small to be GGUF");
        close(fd);
        free(m);
        return NULL;
    }

    /* Private read-only mapping: M0 never host-registers and never shares the
     * mapping with a device. */
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        set_error(err_buf, err_len, "cannot mmap model file");
        close(fd);
        free(m);
        return NULL;
    }

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    q3_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) {
        set_error(err_buf, err_len, c.error[0] ? c.error : "truncated GGUF file");
        q3_gguf_close(m);
        return NULL;
    }
    if (magic != Q3_GGUF_MAGIC) {
        set_error(err_buf, err_len, "model is not a GGUF file");
        q3_gguf_close(m);
        return NULL;
    }
    if (!cursor_u32(&c, &m->version) ||
        !cursor_u64(&c, &m->n_tensors) ||
        !cursor_u64(&c, &m->n_kv)) {
        set_error(err_buf, err_len, c.error[0] ? c.error : "truncated GGUF file");
        q3_gguf_close(m);
        return NULL;
    }

    if (m->version != 3) {
        set_error(err_buf, err_len, "only GGUF v3 is supported");
        q3_gguf_close(m);
        return NULL;
    }

    parse_metadata(m, &c);
    if (c.error[0]) {
        set_error(err_buf, err_len, c.error);
        q3_gguf_close(m);
        return NULL;
    }
    parse_tensors(m, &c);
    if (c.error[0]) {
        set_error(err_buf, err_len, c.error);
        q3_gguf_close(m);
        return NULL;
    }

    return m;
}

void q3_gguf_close(q3_gguf *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    free(m);
}

q3_kv *q3_gguf_find_kv(const q3_gguf *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (q3_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

bool q3_gguf_get_string(const q3_gguf *m, const char *key, q3_str *out) {
    q3_kv *kv = q3_gguf_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    q3_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

bool q3_gguf_get_u32(const q3_gguf *m, const char *key, uint32_t *out) {
    q3_kv *kv = q3_gguf_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    q3_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

bool q3_gguf_get_u64(const q3_gguf *m, const char *key, uint64_t *out) {
    q3_kv *kv = q3_gguf_find_kv(m, key);
    if (!kv) return false;
    q3_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

bool q3_gguf_get_bool(const q3_gguf *m, const char *key, bool *out) {
    q3_kv *kv = q3_gguf_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    q3_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}
