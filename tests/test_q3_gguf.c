/* test_gguf.c — M0 GGUF parser tests.
 *
 * Builds a minimal valid GGUF v3 file in memory, then verifies parsing,
 * metadata access, tensor directory, and safe failure on truncated/bad
 * input. No CUDA required.
 */

#include "q3_gguf.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

/* Little-endian byte writers. */
static void w_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

static void w_u64(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)((v >> (8 * i)) & 0xff);
}

typedef struct {
    unsigned char *buf;
    size_t len;
    size_t cap;
} builder;

static void b_need(builder *b, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->cap ? b->cap : 256) * 2 + n;
        b->buf = realloc(b->buf, b->cap);
        if (!b->buf) { fprintf(stderr, "oom\n"); exit(1); }
    }
}

static void b_string(builder *b, const char *s) {
    uint64_t n = strlen(s);
    b_need(b, 8 + n);
    w_u64(b->buf + b->len, n);
    memcpy(b->buf + b->len + 8, s, n);
    b->len += 8 + n;
}

static void b_u32(builder *b, uint32_t v) {
    b_need(b, 4);
    w_u32(b->buf + b->len, v);
    b->len += 4;
}

static void b_u64(builder *b, uint64_t v) {
    b_need(b, 8);
    w_u64(b->buf + b->len, v);
    b->len += 8;
}

static void b_bytes(builder *b, const void *p, size_t n) {
    b_need(b, n);
    memcpy(b->buf + b->len, p, n);
    b->len += n;
}

/* Write a minimal GGUF v3 with 2 metadata keys and 1 f32 tensor. */
static void build_valid(builder *b) {
    /* magic */
    b_u32(b, 0x46554747u);
    /* version */
    b_u32(b, 3);
    /* n_tensors */
    b_u64(b, 1);
    /* n_kv */
    b_u64(b, 2);

    /* kv 0: general.name = "test" */
    b_string(b, "general.name");
    b_u32(b, 8); /* STRING */
    b_string(b, "test");

    /* kv 1: general.alignment = 32 */
    b_string(b, "general.alignment");
    b_u32(b, 4); /* UINT32 */
    b_u32(b, 32);

    /* tensor 0: "weight" 1-dim [4], type f32 (0), offset 0 */
    b_string(b, "weight");
    b_u32(b, 1);      /* ndim */
    b_u64(b, 4);      /* dim[0] */
    b_u32(b, 0);      /* f32 */
    b_u64(b, 0);      /* rel_offset */

    /* tensor data (aligned to 32): 4 floats */
    size_t pad = (32 - (b->len % 32)) % 32;
    b_need(b, pad + 16);
    memset(b->buf + b->len, 0, pad);
    b->len += pad;
    b_bytes(b, (const float[4]){1.0f, 2.0f, 3.0f, 4.0f}, 16);
}

static int write_file(const char *path, const unsigned char *buf, size_t len) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t w = fwrite(buf, 1, len, fp);
    fclose(fp);
    return w == len ? 0 : -1;
}

int main(void) {
    const char *path = "/tmp/q3_test_gguf.bin";

    /* Valid file. */
    builder b = {0};
    build_valid(&b);
    if (write_file(path, b.buf, b.len) != 0) {
        fprintf(stderr, "FAIL: cannot write test file\n");
        return 1;
    }

    char err[256];
    q3_gguf *m = q3_gguf_open(path, err, sizeof(err));
    CHECK(m != NULL, "valid GGUF opens");
    if (m) {
        CHECK(m->version == 3, "version == 3");
        CHECK(m->n_kv == 2, "n_kv == 2");
        CHECK(m->n_tensors == 1, "n_tensors == 1");

        q3_str name = {0};
        CHECK(q3_gguf_get_string(m, "general.name", &name), "get name");
        CHECK(name.len == 4 && memcmp(name.ptr, "test", 4) == 0, "name value");

        uint32_t align = 0;
        CHECK(q3_gguf_get_u32(m, "general.alignment", &align), "get alignment");
        CHECK(align == 32, "alignment == 32");

        const q3_tensor *t = &m->tensors[0];
        CHECK(t->elements == 4, "tensor elements == 4");
        CHECK(t->bytes == 16, "tensor bytes == 16");
        CHECK(strcmp(q3_gguf_type_name(t->type), "f32") == 0, "type name f32");
        CHECK(t->ndim == 1 && t->dim[0] == 4, "tensor shape [4]");
        CHECK(t->rel_offset == 0, "tensor rel_offset == 0");
        CHECK(m->tensor_data_pos == m->alignment * ((m->tensor_data_pos +
                                                   m->alignment - 1) /
                                                  m->alignment),
              "tensor data is aligned");
        CHECK(t->abs_offset == m->tensor_data_pos + t->rel_offset,
              "abs_offset == data_pos + rel_offset");
        CHECK(t->abs_offset + t->bytes <= m->size,
              "tensor bytes are within file bounds");
        CHECK(q3_gguf_tensor_data(m, t) == m->map + t->abs_offset,
              "tensor_data points into the mmap at abs_offset");

        uint64_t nbytes = 0;
        CHECK(q3_gguf_type_nbytes(0, 4, &nbytes) && nbytes == 16,
              "type_nbytes(f32, 4) == 16");

        q3_gguf_close(m);
    }

    /* Truncated file: header only. */
    if (write_file(path, b.buf, 8) == 0) {
        q3_gguf *bad = q3_gguf_open(path, err, sizeof(err));
        CHECK(bad == NULL, "truncated GGUF fails safely");
        if (bad) q3_gguf_close(bad);
    }

    /* Bad magic. */
    unsigned char badmagic[4];
    w_u32(badmagic, 0xDEADBEEFu);
    if (write_file(path, badmagic, 4) == 0) {
        q3_gguf *bad = q3_gguf_open(path, err, sizeof(err));
        CHECK(bad == NULL, "bad magic fails safely");
        if (bad) q3_gguf_close(bad);
    }

    free(b.buf);
    remove(path);

    if (failures == 0) {
        printf("test_q3_gguf: all tests passed\n");
        return 0;
    }
    printf("test_q3_gguf: %d failure(s)\n", failures);
    return 1;
}
