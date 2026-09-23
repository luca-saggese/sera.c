# 07 — M3 Runtime / API / Telemetry / Tests plan (W4)

Anchors: `src/q3_forward.h:75-190`, `src/q3_forward.cu:607-720`, `cuda/q3_mmq.cu:211-253`,
`src/q3_forward_cli.cu:100-421`, `src/q3.h:34-48`, `src/q3_main.c:585-605`, `Makefile:34-52`.
New code lives in `src/q3_decide.{h,cu}` + `src/q3_decide_cli.cu` (compiled to
`cuda/q3_decide.o`, `cuda/q3_decide_cli.o` by the `cuda/%.o: src/%.cu` rule, Makefile:71).

## Public API (`src/q3_decide.h`)

```c
#define Q3_MAX_BRANCHES       32u
#define Q3_MAX_CANDIDATES      8u
#define Q3_DEFAULT_SUFFIX_CAP 256u

typedef struct q3_prefix_kv  q3_prefix_kv;   /* owns shared KV  */
typedef struct q3_kv_branch  q3_kv_branch;   /* borrows prefix  */
typedef struct q3_branch_set q3_branch_set;  /* owns suffix arena */

/* --- prefix lifecycle (M3 §8-9) --- */
q3_prefix_kv *q3_prefix_create(q3_forward_runtime *rt, uint32_t capacity_tokens,
                               char *error, size_t error_len);
bool  q3_prefix_prefill(q3_prefix_kv *p, const uint32_t *token_ids_host,
                        uint32_t token_count, q3_forward_stats *stats,
                        char *error, size_t error_len);
bool  q3_prefix_seal(q3_prefix_kv *p, char *error, size_t error_len);
void  q3_prefix_destroy(q3_prefix_kv *p);
uint32_t q3_prefix_token_count(const q3_prefix_kv *p);
size_t   q3_prefix_kv_bytes(const q3_prefix_kv *p);

/* --- branch set + branches (M3 §10, §23-25, §51) --- */
q3_branch_set *q3_branch_set_create(const q3_prefix_kv *p, uint32_t max_branches,
                                    uint32_t max_suffix_tokens,
                                    char *error, size_t error_len);
q3_kv_branch *q3_branch_acquire(q3_branch_set *s, uint32_t suffix_capacity,
                                char *error, size_t error_len);
void q3_branch_reset(q3_kv_branch *b);          /* suffix_tokens = 0, no free */
void q3_branch_set_release_all(q3_branch_set *s);
void q3_branch_set_destroy(q3_branch_set *s);

/* --- kernel-facing views (W1/W2/W3) --- */
size_t q3_kv_bytes_for_tokens(uint32_t tokens, uint32_t layers,
                              uint32_t kv_heads, uint32_t head_dim);  /* M3 §7 */

/* --- one entry point (M3 §15, §26) --- */
typedef struct {
    const q3_kv_branch *branch;
    const uint32_t *tokens;         /* host suffix IDs, len token_count   */
    uint32_t        token_count;
    const uint32_t *candidate_ids;  /* host LM rows, 2..8                 */
    uint32_t        candidate_count;
} q3_decision_item;

typedef struct { q3_decision_item *items; uint32_t count; } q3_decision_batch;

typedef struct {
    uint32_t predicted_index;                       /* argmax over candidates */
    float    logits[Q3_MAX_CANDIDATES];
    float    probabilities[Q3_MAX_CANDIDATES];      /* softmax over candidates */
    uint32_t candidate_count;
} q3_decision_result;

bool q3_decide_batch(q3_forward_runtime *rt, const q3_prefix_kv *prefix,
                     const q3_decision_batch *batch, q3_decision_result *results,
                     q3_decide_stats *stats, char *error, size_t error_len);
```

Conventions copied from M2: `extern "C"`, `bool` return, `char *error, size_t error_len` last
args, fail-loud return `false` (never silent fallback, M3 §53). Runtime/weights are borrowed;
prefix borrows `rt` only for `prefill`. `results` is `batch->count` host entries (tiny D2H at
the end only, §43).

## Structs (internal, `src/q3_decide.cu`)

```c
struct q3_prefix_kv {                 /* owns; immutable after seal */
    q3_forward_runtime *runtime;      /* borrowed, prefill only */
    q3_model_config cfg;
    uint32_t capacity_tokens, token_count;
    size_t   bytes;                   /* K+V, constant in B (W1) */
    void    *arena_dev;               /* ONE cudaMalloc: K slabs then V slabs */
    uint64_t generation; bool sealed;
};
struct q3_branch_set {                /* owns the single suffix arena */
    const q3_prefix_kv *prefix;
    uint32_t max_branches, max_suffix_tokens;
    size_t   per_layer_stride, bytes;
    void    *suffix_arena_dev;        /* ONE cudaMalloc, [branch][layer][cap][h][d] */
    q3_kv_branch *slots;              /* max_branches descriptors */
    uint32_t used;
};
struct q3_kv_branch {                 /* descriptor only; cheap create */
    const q3_prefix_kv *prefix;
    q3_branch_set *set;               /* borrowed owner of storage */
    uint32_t suffix_tokens, suffix_capacity, slot;
};
```

Storage is exposed to the packed forward only as arena base pointers + strides, so no
per-layer pointer array is needed and W1/W2/W3 designs are not fixed here.

## Allocation plan

| allocation | bytes (Qwen3-32B, 64L/8KVh/128d) | when | freed by |
|---|---|---|---|
| runtime workspace `rt->workspace` | `max_tokens * (hidden + …)` (M2, `q3_forward.cu:695`) | `q3_forward_create` | `q3_forward_destroy` |
| MMQ arena `rt->mmq_arena` | `q3_mmq_arena_bytes(max_tokens, out_n, inter)` (`q3_mmq.cu:211`) | `q3_forward_create` | `q3_forward_destroy` |
| prefix arena (K+V) | `2*64*cap*8*128*4 = 512 KiB/token` × cap | `q3_prefix_create` | `q3_prefix_destroy` |
| branch-set suffix arena | `2*64*max_branches*max_suffix*8*128*4` | `q3_branch_set_create` | `q3_branch_set_destroy` |
| host descriptors | `~32 * ~1KB`, negligible | create / per call | set destroy / free |
| per-call temporaries | none — sliced from arenas | — | — |

Proofs: prefix/suffix bytes are `f(capacity)`, independent of B; measured prefix bytes must be
identical for B=1..32 (§37/§70). Zero `cudaMalloc`/`cudaFree` after the create calls; warm
`cuda_allocations` must be 0 (§72). Suffix arena allocated exactly once in
`branch_set_create`; `q3_branch_acquire`/`q3_branch_reset` only set `suffix_tokens = 0` and
rewind offsets (§51). `max_tokens = B*S_max`: `q3_forward_create` must be sized for `Σ S_b`
(worst `B*S_max`), which grows the two M2 buffers — same single allocations, larger constants;
the Q8_1 stage in `q3_mmq_arena_bytes` is `tokens*pad(K,512)` so it scales linearly
(`q3_mmq.cu:238-249`).

## Telemetry (`q3_decide_stats`, §34-§45)

```c
typedef struct {
    double prefix_prefill_ms, branch_create_ms, suffix_forward_ms, candidate_head_ms;
    double cold_total_ms, warm_total_ms;
    double suffix_questions_per_s, cold_questions_per_s;
    double warm_ms_per_question, cold_ms_per_question;
    uint64_t prefix_kv_bytes_physical, suffix_kv_bytes, workspace_bytes, peak_cuda_bytes;
    uint64_t prefix_allocations, prefix_bytes;     /* §42 */
    uint64_t suffix_allocations, suffix_bytes;     /* §42 */
    uint64_t prefix_copy_calls, prefix_copy_bytes; /* §11/§42, must be 0 */
    uint64_t expected_prefix_kv_bytes;             /* §41 validation */
    uint64_t cuda_allocations, cuda_frees, host_syncs;
    uint64_t kernel_launches_per_batch, kernel_launches_per_question;
    uint64_t batch, suffix_tokens_total;
    uint32_t accuracy_matches, accuracy_total;     /* §55 */
} q3_decide_stats;
```

`host_syncs`/`cuda_allocations` accumulate the per-forward `q3_forward_stats` returned by the
packed suffix path; the only allowed sync is the explicit boundary after timing (§44).
Counters are plain `uint64_t ++` in host code — no extra kernel.

## CLI + JSON schema

One mode added to `q3_mode` (`src/q3.h:23-31`): `Q3_MODE_BENCH_DECISIONS`; options gain
`const char *workload_path;` (reuse `opt->batches[16]/batch_count` from `--batch`). Parser
(`q3_main.c:585-605` style) accepts:

```bash
./q3 --bench-decisions MODEL.gguf --workload workload.json --batch 1,2,4,8,16,32 --json
```

Workload schema (exactly `docs/research/m3/workload_tokens.json`): `state_prompt` (string,
ignored), `state_tokens` `uint32[129]`, `questions[32]` each
`{id, category, suffix_tokens[], candidate_token_ids[2..8], expected_index}`. A minimal
hand-rolled reader scans for those three arrays (no JSON dep); fail loudly on
missing/`candidate_count>8`/unknown token id. Batch list parsed by the existing `strtol` loop;
each B selects the first B questions (B=32 → all). `--json` emits the §55 object.

## Artifacts (§58-§59)

`artifacts/m3_batch/`: `correctness.json`, `batch_b{1,2,4,8,16,32}.json`, `context_2k_b8.json`,
`context_8k_b8.json`, `summary.md` with exactly: commit, model, prefix tokens, prefix KV
bytes, B1/B2/B4/B8/B16/B32 q/s, B32 suffix latency, B32 warm ms/question, prefix copies, hot
allocations, hot syncs, semantic parity, 2K prefix result, 8K prefix result, largest
bottleneck.

## Test plan (4 tests, §33 + §62 cadence)

| # | name / file | asserts | gate | run when |
|---|---|---|---|---|
| 1 | `test_q3_decide.cu` — `test_b1_segmented_parity` | M2 contiguous B1 forward vs M3 prefix+suffix B1: identical argmax, `max|Δlogit|` within M2 tolerance | §71 (B1) | during segmented KV |
| 2 | `test_q3_decide.cu` — `test_b2_isolation` | 2 branches (ALPHA/OMEGA canary, §32): per-branch parity and cross-branch leak = 0 | §71 | during batching |
| 3 | `test_q3_decide.cu` — `test_b32_parity` | all 32 workload questions: `predicted_index == expected_index` | §71 | during B32 |
| 4 | `test_q3_decide.cu` — `test_memory_runtime_invariants` | after warmup: `prefix_copy_bytes==0`, `prefix_allocations==1`, `prefix_kv_bytes` constant B=1..32, `cuda_allocations==0`, `host_syncs==0`, reuse 3× B=8 fingerprint unchanged | §70, §72, §49-50 | during memory cleanup |

Reused verbatim from M2: `CHECK` macro and harness from `tests/test_q3_forward.cu`, the
argmax/tolerance pattern (`:492-592`), `Q3_TEST_MODEL` env (`:46-48`). Object list mirrors
`TEST_Q4_OBJS` (`Makefile:96`).

## Gate checklist (checkable)

1. `--bench-decisions … --batch 1` exits 0, B1 parity → §71.
2. all `predicted_index==expected_index` → §71, DoD 12.
3. `prefix_copy_bytes == 0 && prefix_copy_calls == 0` in every batch file → §42, DoD 13.
4. `prefix_kv_bytes` equal across B=1..32 → §37/§70, DoD 14.
5. `cuda_allocations == 0 && host_syncs == 0` warm → §72, DoD 15.
6. `kernel_launches_per_batch(B=32) < 32 × per_question(B=1)` → §45/§69.
7. 6 batch files + 2 context files + correctness.json + summary.md with §59 fields → DoD 16-19.
8. Test 4 passes → §48-50, DoD 1-5.
9. Cross-branch leak = 0 (Test 2) → DoD 6-7.
10. Variable suffix lengths validated by Test 3 → DoD 10.
11. Exactly 4 M3 tests → DoD 18.
12. Baseline B=1,8,32 measured → DoD 20.

## Build integration

`Makefile`: add `cuda/q3_decide.o` and `cuda/q3_decide_cli.o` to `CUDA_OBJS`
(Makefile:34-43; rule `cuda/%.o: src/%.cu` at :71 gives `-Icuda/mmq` for MMQ calls). Register
mode in `q3_main.c` usage/parse/dispatch (`:33-49,552-629`) and add
`Q3_MODE_BENCH_DECISIONS` to `q3.h:23-31`; declare `q3_cmd_bench_decisions` in a new
`src/q3_decide_cli.h`. Add the M3 test target + objects to `TESTS`/`test:`
(Makefile:52,90-116).

## Files to touch

- `src/q3_decide.h` (new public ABI), `src/q3_decide.cu` (prefix/branch/decide impl),
  `src/q3_decide_cli.{h,cu}` (CLI + JSON + artifacts), `tests/test_q3_decide.cu` (new),
  `Makefile`, `src/q3.h`, `src/q3_main.c`.
- `src/q3_forward.{h,cu}`: add the segmented attention primitive + packed run entry.
- Read-only reuse: `cuda/q3_mmq.cu`, `src/q3_model.h`, `tests/test_q3_forward.cu`.
