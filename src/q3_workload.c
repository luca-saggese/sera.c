/* SPDX-License-Identifier: MIT
 *
 * q3_workload.c - reader for the M3 semantic decision workload (docs/M3.md §56).
 *
 * Deliberately a tiny hand-rolled scanner rather than a JSON library: the file
 * is a frozen artifact with a known shape, and M3 forbids new frameworks.
 *
 *   {
 *     "state_prompt": "...",
 *     "state_tokens": [ ... ],
 *     "questions": [
 *       { "id": "q01", "category": "entity_binding",
 *         "suffix_tokens": [ ... ],
 *         "candidate_token_ids": [ ... ],
 *         "expected_index": 0 }, ...
 *     ]
 *   }
 *
 * `state_prompt` is skipped: only token IDs are needed.
 */

#include "q3_workload.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WL_ERR(...) do { \
    if (error && error_len) { error[0] = '\0'; snprintf(error, error_len, __VA_ARGS__); } \
} while (0)

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

/* Skip a JSON string literal (the opening quote is at *p). */
static bool skip_string(const char **p) {
    if (**p != '"') return false;
    (*p)++;
    while (**p) {
        if (**p == '\\') {
            (*p)++;
            if (!**p) return false;
            (*p)++;
            continue;
        }
        if (**p == '"') {
            (*p)++;
            return true;
        }
        (*p)++;
    }
    return false;
}

/* Read a string literal into `out` (truncating), handling the escapes the
 * workload writer emits. */
static bool read_string(const char **p, char *out, size_t out_len) {
    if (**p != '"') return false;
    (*p)++;
    size_t n = 0;
    while (**p && **p != '"') {
        char c = **p;
        if (c == '\\') {
            (*p)++;
            switch (**p) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case '"': c = '"'; break;
            case '\\': c = '\\'; break;
            case '/': c = '/'; break;
            case 'u': /* Only ASCII escapes are expected here. */
                (*p) += 4;
                c = '?';
                break;
            default: c = **p; break;
            }
        }
        if (n + 1 < out_len) out[n++] = c;
        (*p)++;
    }
    if (**p != '"') return false;
    (*p)++;
    out[n] = '\0';
    return true;
}

static bool read_uint(const char **p, uint32_t *out) {
    skip_ws(p);
    if (**p < '0' || **p > '9') return false;
    unsigned long long v = 0;
    while (**p >= '0' && **p <= '9') {
        v = v * 10ull + (unsigned long long)(**p - '0');
        (*p)++;
    }
    if (v > 0xffffffffull) return false;
    *out = (uint32_t)v;
    return true;
}

/* Consume `expect`, or fail. */
static bool expect(const char **p, char expect) {
    skip_ws(p);
    if (**p != expect) return false;
    (*p)++;
    return true;
}

/* Read a uint32 array body (the opening '[' is at *p). */
static bool read_uint_array(const char **p, uint32_t **out, uint32_t *out_count,
                            char *error, size_t error_len, const char *what) {
    if (!expect(p, '[')) {
        WL_ERR("workload: expected '[' for %s", what);
        return false;
    }
    size_t cap = 16, n = 0;
    uint32_t *ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!ids) {
        WL_ERR("workload: out of memory for %s", what);
        return false;
    }
    skip_ws(p);
    if (**p == ']') {
        (*p)++;
        *out = ids;
        *out_count = 0;
        return true;
    }
    for (;;) {
        if (n == cap) {
            cap *= 2;
            uint32_t *grown = (uint32_t *)realloc(ids, cap * sizeof(uint32_t));
            if (!grown) {
                free(ids);
                WL_ERR("workload: out of memory for %s", what);
                return false;
            }
            ids = grown;
        }
        if (!read_uint(p, &ids[n])) {
            free(ids);
            WL_ERR("workload: malformed integer in %s", what);
            return false;
        }
        n++;
        skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == ']') { (*p)++; break; }
        free(ids);
        WL_ERR("workload: expected ',' or ']' in %s", what);
        return false;
    }
    *out = ids;
    *out_count = (uint32_t)n;
    return true;
}

/* Read a question object (the opening '{' is at *p). */
static bool read_question(const char **p, q3_workload_question *q, char *error,
                          size_t error_len) {
    memset(q, 0, sizeof(*q));
    if (!expect(p, '{')) {
        WL_ERR("workload: expected '{' for a question");
        return false;
    }
    for (;;) {
        skip_ws(p);
        if (**p == '}') {
            (*p)++;
            break;
        }
        char key[64];
        if (!read_string(p, key, sizeof(key))) {
            WL_ERR("workload: malformed question key");
            return false;
        }
        if (!expect(p, ':')) {
            WL_ERR("workload: expected ':' after question key '%s'", key);
            return false;
        }
        skip_ws(p);
        if (strcmp(key, "id") == 0) {
            if (!read_string(p, q->id, sizeof(q->id))) {
                WL_ERR("workload: malformed question id");
                return false;
            }
        } else if (strcmp(key, "category") == 0) {
            if (!read_string(p, q->category, sizeof(q->category))) {
                WL_ERR("workload: malformed question category");
                return false;
            }
        } else if (strcmp(key, "suffix_tokens") == 0) {
            if (!read_uint_array(p, &q->suffix_tokens, &q->suffix_count, error,
                                 error_len, "suffix_tokens"))
                return false;
        } else if (strcmp(key, "candidate_token_ids") == 0) {
            uint32_t *ids = NULL;
            uint32_t n = 0;
            if (!read_uint_array(p, &ids, &n, error, error_len,
                                 "candidate_token_ids"))
                return false;
            if (n == 0 || n > Q3_MAX_CANDIDATES) {
                free(ids);
                WL_ERR("workload: candidate count %u out of range (max %u)", n,
                       (unsigned)Q3_MAX_CANDIDATES);
                return false;
            }
            memcpy(q->candidate_ids, ids, n * sizeof(uint32_t));
            q->candidate_count = n;
            free(ids);
        } else if (strcmp(key, "expected_index") == 0) {
            if (!read_uint(p, &q->expected_index)) {
                WL_ERR("workload: malformed expected_index");
                return false;
            }
        } else {
            /* Unknown keys are skipped so the reader keeps working if the
             * workload writer grows a field. */
            skip_ws(p);
            if (**p == '"') {
                if (!skip_string(p)) {
                    WL_ERR("workload: malformed string for key '%s'", key);
                    return false;
                }
            } else if (**p == '[') {
                uint32_t *scratch = NULL;
                uint32_t scratch_n = 0;
                if (!read_uint_array(p, &scratch, &scratch_n, error, error_len,
                                     key))
                    return false;
                free(scratch);
            } else {
                uint32_t scratch = 0;
                if (!read_uint(p, &scratch)) {
                    WL_ERR("workload: unsupported value for key '%s'", key);
                    return false;
                }
            }
        }
        skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == '}') { (*p)++; break; }
        WL_ERR("workload: expected ',' or '}' in question");
        return false;
    }
    if (q->suffix_count == 0 || !q->suffix_tokens) {
        WL_ERR("workload: question has no suffix tokens");
        return false;
    }
    if (q->candidate_count == 0) {
        WL_ERR("workload: question has no candidate token ids");
        return false;
    }
    if (q->expected_index >= q->candidate_count) {
        WL_ERR("workload: expected_index %u out of range", q->expected_index);
        return false;
    }
    return true;
}

q3_workload *q3_workload_load(const char *path, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!path) {
        WL_ERR("workload: null path");
        return NULL;
    }
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        WL_ERR("workload: cannot open '%s'", path);
        return NULL;
    }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    if (size <= 0) {
        fclose(fh);
        WL_ERR("workload: '%s' is empty", path);
        return NULL;
    }
    char *text = (char *)malloc((size_t)size + 1);
    if (!text) {
        fclose(fh);
        WL_ERR("workload: out of memory");
        return NULL;
    }
    if (fread(text, 1, (size_t)size, fh) != (size_t)size) {
        free(text);
        fclose(fh);
        WL_ERR("workload: short read on '%s'", path);
        return NULL;
    }
    fclose(fh);
    text[size] = '\0';

    q3_workload *w = (q3_workload *)calloc(1, sizeof(*w));
    if (!w) {
        free(text);
        WL_ERR("workload: out of memory");
        return NULL;
    }

    const char *p = text;
    if (!expect(&p, '{')) {
        WL_ERR("workload: expected a top-level object");
        goto fail;
    }
    for (;;) {
        skip_ws(&p);
        if (*p == '}') { p++; break; }
        char key[64];
        if (!read_string(&p, key, sizeof(key))) {
            WL_ERR("workload: malformed top-level key");
            goto fail;
        }
        if (!expect(&p, ':')) {
            WL_ERR("workload: expected ':' after top-level key '%s'", key);
            goto fail;
        }
        skip_ws(&p);
        if (strcmp(key, "state_tokens") == 0) {
            if (!read_uint_array(&p, &w->state_tokens, &w->state_count, error,
                                 error_len, "state_tokens"))
                goto fail;
        } else if (strcmp(key, "questions") == 0) {
            if (!expect(&p, '[')) {
                WL_ERR("workload: expected '[' for questions");
                goto fail;
            }
            size_t cap = 16;
            w->questions = (q3_workload_question *)calloc(cap, sizeof(*w->questions));
            if (!w->questions) {
                WL_ERR("workload: out of memory");
                goto fail;
            }
            skip_ws(&p);
            if (*p == ']') {
                p++;
            } else {
                for (;;) {
                    if (w->question_count == cap) {
                        cap *= 2;
                        q3_workload_question *grown = (q3_workload_question *)
                            realloc(w->questions, cap * sizeof(*w->questions));
                        if (!grown) {
                            WL_ERR("workload: out of memory");
                            goto fail;
                        }
                        w->questions = grown;
                        memset(&w->questions[w->question_count], 0,
                               (cap - w->question_count) * sizeof(*w->questions));
                    }
                    q3_workload_question *q = &w->questions[w->question_count];
                    if (!read_question(&p, q, error, error_len)) goto fail;
                    w->question_count++;
                    if (q->suffix_count > w->max_suffix_count)
                        w->max_suffix_count = q->suffix_count;
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == ']') { p++; break; }
                    WL_ERR("workload: expected ',' or ']' in questions");
                    goto fail;
                }
            }
        } else if (strcmp(key, "state_prompt") == 0) {
            if (!skip_string(&p)) {
                WL_ERR("workload: malformed state_prompt");
                goto fail;
            }
        } else {
            WL_ERR("workload: unsupported top-level key '%s'", key);
            goto fail;
        }
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        WL_ERR("workload: expected ',' or '}' at top level");
        goto fail;
    }

    free(text);
    if (w->state_count == 0) {
        WL_ERR("workload: no state tokens");
        goto fail;
    }
    if (w->question_count == 0) {
        WL_ERR("workload: no questions");
        goto fail;
    }
    if (w->question_count > Q3_MAX_BRANCHES) {
        WL_ERR("workload: %u questions exceed the branch limit %u",
               w->question_count, (unsigned)Q3_MAX_BRANCHES);
        goto fail;
    }
    return w;

fail:
    free(text);
    q3_workload_free(w);
    return NULL;
}

void q3_workload_free(q3_workload *w) {
    if (!w) return;
    free(w->state_tokens);
    if (w->questions) {
        for (uint32_t i = 0; i < w->question_count; i++)
            free(w->questions[i].suffix_tokens);
        free(w->questions);
    }
    free(w);
}
