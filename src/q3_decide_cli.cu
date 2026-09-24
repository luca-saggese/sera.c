/* SPDX-License-Identifier: MIT
 *
 * q3_decide_cli.cu - M3 --bench-decisions correctness entry point.
 *
 *     ./q3 --bench-decisions MODEL.gguf [--workload request.json] [--json]
 *
 * The CLI correctness run does not re-implement the decision protocol. It
 * hands the same System One request fixture the server accepts to the same
 * production runtime entry point, q3_systemone_run(). Tokenization, the
 * QUESTION/OPTIONS/ANSWER suffix, the 1/2/3 candidate digits and the result
 * mapping all live inside that one implementation, so the CLI and the server
 * can only disagree when the runtime itself disagrees. Accuracy is compared
 * against the fixture's own "expected" index.
 */
#include "q3_decide_cli.h"

extern "C" {
#include "hd_json.h"
}

#include "q3_systemone.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The frozen System One request fixture (docs/research/m4/bench_systemone.json). */
#define Q3_BENCH_SYSTEMONE_FIXTURE "docs/research/m4/bench_systemone.json"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static char *read_text_file(const char *path, char *err, size_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, err_len, "cannot open %s", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        snprintf(err, err_len, "cannot seek %s", path);
        return NULL;
    }
    const long n = ftell(f);
    if (n < 0) {
        fclose(f);
        snprintf(err, err_len, "cannot size %s", path);
        return NULL;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    const size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    return buf;
}

int q3_cmd_bench_decisions(const q3_options *opt) {
    char err[512] = {0};
    const char *fixture = opt->workload_path ? opt->workload_path
                                             : Q3_BENCH_SYSTEMONE_FIXTURE;

    char *text = read_text_file(fixture, err, sizeof(err));
    if (!text) {
        fprintf(stderr, "q3: %s\n", err);
        return 1;
    }

    const char *jerr = NULL;
    hd_json *request = hd_json_parse(text, &jerr);
    free(text);
    if (!request) {
        fprintf(stderr, "q3: %s: %s\n", fixture, jerr ? jerr : "invalid JSON");
        return 1;
    }

    const double load_begin = now_ms();
    q3_model model;
    q3_status st = q3_model_load(opt->model_path, 0, &model);
    const double load_ms = now_ms() - load_begin;
    if (st != Q3_OK) {
        fprintf(stderr, "q3: %s\n", q3_model_last_error());
        hd_json_free(request);
        return 1;
    }

    q3_systemone_result *results = NULL;
    int count = 0;
    long input_tokens = 0;
    const double run_begin = now_ms();
    st = q3_systemone_run(&model, request, &results, &count, &input_tokens);
    const double run_ms = now_ms() - run_begin;
    if (st != Q3_OK) {
        fprintf(stderr, "q3: %s\n", q3_model_last_error());
        q3_model_free(&model);
        hd_json_free(request);
        return 1;
    }

    const hd_json *questions = hd_json_get(request, "questions");
    uint32_t matches = 0;
    for (int i = 0; i < count; i++) {
        const char *key = questions->u.object.keys[i];
        const hd_json *qdef = questions->u.object.values[i];
        const int expected = (int)hd_json_int(hd_json_get(qdef, "expected"), -1);
        const int predicted = results[i].choice;
        if (predicted == expected) matches++;
        if (opt->verbose || predicted != expected) {
            const char *label =
                (results[i].option_labels && predicted >= 0 &&
                 predicted < results[i].n_options)
                    ? results[i].option_labels[predicted]
                    : "?";
            fprintf(stderr, "  %-6s pred=%d (%s) exp=%d %s\n", key, predicted,
                    label, expected, predicted == expected ? "" : "MISS");
        }
    }

    if (opt->json) {
        printf("{\n");
        printf("  \"fixture\": \"%s\",\n", fixture);
        printf("  \"questions\": %d,\n", count);
        printf("  \"input_tokens\": %ld,\n", input_tokens);
        printf("  \"model_load_ms\": %.2f,\n", load_ms);
        printf("  \"systemone_ms\": %.2f,\n", run_ms);
        printf("  \"accuracy\": %u\n", matches);
        printf("}\n");
    } else {
        printf("fixture:           %s\n", fixture);
        printf("questions:         %d\n", count);
        printf("input tokens:      %ld\n", input_tokens);
        printf("model load:        %.2f ms\n", load_ms);
        printf("systemone run:     %.2f ms\n", run_ms);
        printf("accuracy:          %u/%d\n", matches, count);
    }

    q3_systemone_results_free(results, count);
    q3_model_free(&model);
    hd_json_free(request);
    return 0;
}
