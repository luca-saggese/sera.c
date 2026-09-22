#include "q3_residency_plan.h"

#include <assert.h>
#include <string.h>

static bool is_ple(const q3_tensor *tensor, void *user) {
    (void)user;
    return tensor->name.ptr &&
           strstr(tensor->name.ptr, ".ple.ple_embedding.ngram_embedding.shard_") !=
               NULL;
}

int main(void) {
    static const char ple_name[] =
        "blk.ple.ple_embedding.ngram_embedding.shard_0";
    static const char normal_name[] = "blk.normal";
    static const char normal_name_2[] = "blk.normal_2";
    static const char normal_name_3[] = "blk.normal_3";
    q3_tensor tensors[] = {
        {.name = {normal_name, sizeof(normal_name) - 1},
         .abs_offset = 100, .bytes = 100},
        {.name = {normal_name_2, sizeof(normal_name_2) - 1},
         .abs_offset = 220, .bytes = 80},
        {.name = {ple_name, sizeof(ple_name) - 1},
         .abs_offset = 310, .bytes = 90},
        {.name = {normal_name_3, sizeof(normal_name_3) - 1},
         .abs_offset = 430, .bytes = 70},
    };
    q3_gguf model = {
        .size = 600,
        .n_tensors = sizeof(tensors) / sizeof(tensors[0]),
        .tensors = tensors,
    };
    q3_residency_plan plan;
    char error[128] = {0};
    q3_residency_plan_init(&plan);

    assert(q3_residency_plan_build(
        &model, is_ple, NULL, 64, 512, &plan, error, sizeof(error)));
    assert(plan.entry_count == 3);
    assert(plan.resident_bytes == 250);
    assert(plan.excluded_ple_bytes == 90);
    assert(plan.excluded_ple_tensors == 1);
    assert(plan.span_count == 2);
    assert(plan.entries[0].tensor_index == 0);
    assert(plan.entries[1].tensor_index == 1);
    assert(plan.entries[2].tensor_index == 3);
    assert(plan.spans[0].file_offset == 100);
    assert(plan.spans[0].bytes == 200);
    assert(plan.spans[0].entry_count == 2);
    assert(plan.spans[1].file_offset == 430);
    assert(plan.spans[1].bytes == 70);

    q3_residency_plan_destroy(&plan);
    return 0;
}
