# M4 — Benchmark plan (Worker B)

Scope: `docs/M4.md` §28–§47, §51 (Worker B). Deliverable: the benchmark workload,
the Laya baseline numbers that actually exist, the script plan, the metric
definitions, the artifact schemas and the bottleneck classification.

Donor: `_reference/laya.c` @ `83e7b97fd0fbd53b919d200a11240137a8ffaf04` (read-only).

---

## 0. Sources actually read

| Source | What it gave |
|---|---|
| `docs/M4.md` §28–§47, §51–§54 | binding requirements |
| `_reference/laya.c/fixtures/laya/systemone_smoke.json` | donor System One fixture (1 state, 6 questions) |
| `_reference/laya.c/tests/unit/systemone_smoke.py` | donor wire smoke (no timing, no bench) |
| `_reference/laya.c/tools/oracle_requests.json` | 5 oracle requests, max 3 questions each |
| `_reference/laya.c/dataset/out/questions.json` | 30 question *definitions*, no state |
| `_reference/laya.c/docs/LAYA_CUDA_GB10_IMPLEMENTATION.md:1259` | "Level D — performance benchmark" (spec only) |
| `_reference/laya.c/docs/LAYA_JEV_SERVER_MIGRATION.md:1283` | "Numbers above are placeholders only" |
| `_reference/laya.c/src/laya/laya_infer.h:21` | `LAYA_INFER_MAX_Q 100` |
| `_reference/laya.c/src/server/laya_server.c` | job struct, queue, `/health`, `/v1/models` |
| `docs/research/m3/workload.json` | sera.c text-level workload (32 questions) |
| `docs/research/m3/workload_tokens.json` | sera.c token-level workload (129 + 390 tokens) |
| `artifacts/m3_batch/ladder.json`, `summary.md` | real M3 numbers |
| `artifacts/m0_q4_load/*.json` | real startup/load numbers |
| `src/q3_decide.h`, `src/q3_decide_cli.cu`, `src/q3_decide.cu` | M3 telemetry surface |

---

## 1. Benchmark workload

### 1.1 Donor fixtures found

| Path | Content | Questions available |
|---|---|---|
| `_reference/laya.c/fixtures/laya/systemone_smoke.json` | 1 state (string, 1 sentence, insurance claim CLM-2291), 6 questions: `coverage_choice` (choice, 2 criteria), `handling_team` (choice, 3), `severity` (score, 5), `deductible_bucket` (score, 4), `litigation_risk` (noul), `customer_reply_needed` (noul) | **6** |
| `_reference/laya.c/tools/oracle_requests.json` | 5 requests: `choice-3`, `score-5`, `noul`, `multi-question` (3 questions), `unicode` | **1–3 per request** |
| `_reference/laya.c/dataset/out/questions.json` | `schema_id: claims-auto-it-v2`, 30 question *definitions* (Italian auto-claims), **no state** | 30 definitions, 0 states |

**The donor fixture cannot serve the M4 ladder.** It has 6 questions, so Q=8/16/32
are unreachable, and it is a different domain from the sera.c workload. It is
still the right fixture for the M4 smoke test (M4 §23–§24 explicitly says to copy
`systemone_smoke.py` and reuse the fixture).

### 1.2 sera.c existing workload

`docs/research/m3/workload.json` (text level, 32 questions):

```json
{ "state_prompt": "<515 chars>",
  "questions": [ {"id":"q01","category":"entity_binding",
                  "suffix_prompt":"What is the color of the cat?",
                  "candidates":["gray","black","red"],"expected_index":0}, ... ] }
```

`docs/research/m3/workload_tokens.json` (token level, produced by
`tools/tokenize_workload.py`):

| Field | Value |
|---|---|
| `state_prompt` | same 515-char text |
| `state_tokens` | **129** token IDs |
| `questions` | **32**, each `{id, category, suffix_tokens, candidate_token_ids, expected_index}` |
| suffix tokens | **390 total**; per-question 6–19 (mean 12.2) |
| candidates | **3 per question** (all 32) |
| categories | 12: entity_binding 3, composition 3, negation 3, transitivity 3, temporal 3, multi_hop 3, insurance_routing 3, unknown_entity 3, exception 2, insurance_coverage 2, italian 2, unknown_attribute 2 |

### 1.3 Decision

**Use the sera.c 32-question workload, not the donor fixture, for the ladder.**

Reason: M4 §29 asks for a multi-question workload with the *same state* at
Q=1/2/4/8/16/32. Only the sera.c workload has 32 questions on one state. The
donor fixture is used for the smoke test only.

**What is missing: the workload is token-level, and the M4 HTTP path needs TEXT.**
`workload_tokens.json` carries `suffix_tokens` / `candidate_token_ids`; the server
receives `state` + `questions{type,instructions,criteria}` and must tokenize
itself (M4 §13, §14). So a **text-level System One fixture is required**.

**Create `docs/research/m4/bench_systemone.json`** — a System One request template
derived mechanically from `docs/research/m3/workload.json`:

* `state` = the 515-char `state_prompt` verbatim;
* one question per M3 question, **in file order** (`q01`…`q32`);
* `type` = `"choice"` for all 32 (the M3 workload has no score/noul question);
* `instructions` = the M3 `suffix_prompt` verbatim (no prompt engineering, M4 §12);
* `criteria` = `{candidate_text: candidate_text}` for the 3 M3 candidates, in M3
  order (preserves the §57 candidate-order canary);
* `expected` = the M3 `expected_index` (kept for the accuracy column only).

Schema:

```json
{
  "name": "m4-bench-systemone",
  "model": "q3-local",
  "state": "<515-char state_prompt>",
  "questions": {
    "q01": {"type": "choice",
            "instructions": "What is the color of the cat?",
            "criteria": {"gray": "gray", "black": "black", "red": "red"},
            "expected": 0},
    "...": {}
  }
}
```

> **Divergence to record, not to fix.** The server builds the suffix with
> `q3_build_decision_suffix(...)` (M4 §12) from `instructions` + `criteria`, so
> the tokenized suffix will **not** be byte-identical to the M3
> `suffix_tokens` (which came from `suffix_prompt` alone). M4 measures E2E; the
> M3 ladder stays the token-level reference. The plan must not claim the two are
> the same prompt.

### 1.4 Concrete ladder

Q = number of questions in **one** System One request (M4 §33 — not concurrent
clients). Questions are the **first Q** entries in file order, matching the M3
CLI's `plan_for()` (`src/q3_decide_cli.cu:50-64`, which sums
`w->questions[0..batch-1]`).

| Q | question ids | M3 suffix tokens (reference) |
|---|---|---|
| 1 | q01 | 8 |
| 2 | q01–q02 | 25 |
| 4 | q01–q04 | 53 |
| 8 | q01–q08 | 105 |
| 16 | q01–q16 | 197 |
| 32 | q01–q32 | 390 |

State text (verbatim, 515 chars, identical for every Q):

> The machine is ALPHA. The cat is gray. The horse is black. The car is red. The robot is a machine. The machine BETA is a machine. The cat is grey because the light is dim, but the cat is gray. The code is ALPHA. The robot is the machine. The machine is the robot. Yesterday the code was BETA; today the code is ALPHA. The policy covers repairs of red cars. The policy does not cover machines. The policy covers the cat if it is a pet. Alice is the owner. Alice owns the cat. The owner will be paid. The mark is XQZ.

Question set (32, from `docs/research/m3/workload.json`):

| id | category | instructions (= M3 `suffix_prompt`) | criteria (M3 order) | exp |
|---|---|---|---|---|
| q01 | entity_binding | What is the color of the cat? | gray, black, red | 0 |
| q02 | composition | The robot is a machine, and the machine is ALPHA. What is the robot? | ALPHA, BETA, gray | 0 |
| q03 | negation | The car is not gray. What is the car? | gray, red, black | 1 |
| q04 | transitivity | The robot is a machine, and the machine is ALPHA. What is the robot? | ALPHA, unknown, BETA | 0 |
| q05 | entity_binding | What is the color of the horse? | black, gray, red | 0 |
| q06 | transitivity | Alice owns the cat. The cat is gray. What does Alice own? | a gray cat, a black horse, a red car | 0 |
| q07 | temporal | Today the code is ALPHA. What was the code yesterday? | ALPHA, BETA, red | 1 |
| q08 | multi_hop | Alice owns the cat. The cat is gray. What color is Alice's pet? | gray, black, unknown | 0 |
| q09 | entity_binding | What is the color of the car? | red, gray, black | 0 |
| q10 | exception | The policy does not cover machines. The cat is not a machine. Is the cat covered? | yes, no, unknown | 0 |
| q11 | multi_hop | The owner will be paid. Who is the owner? | Alice, the cat, the machine | 0 |
| q12 | temporal | The machine is ALPHA. Is the machine BETA? | no, yes, unknown | 0 |
| q13 | temporal | In state ALPHA, is the car red? | yes, no, unknown | 0 |
| q14 | negation | The horse is not gray. What is the horse? | black, gray, red | 0 |
| q15 | insurance_coverage | The policy covers repairs of red cars. Is a red car repair covered? | yes, no, unknown | 0 |
| q16 | italian | Di che colore è il gatto? | grigio, nero, rosso | 0 |
| q17 | composition | The machine is ALPHA. The code is ALPHA. What is shared by machine and code? | ALPHA, BETA, gray | 0 |
| q18 | insurance_coverage | The policy does not cover machines. Is a machine covered? | no, yes, unknown | 0 |
| q19 | insurance_routing | A red car needs repair. How is the claim handled? | covered by policy, not covered, unknown | 0 |
| q20 | insurance_routing | A machine needs repair. How is the claim handled? | not covered, covered by policy, unknown | 0 |
| q21 | unknown_entity | What is the color of the zebra? | unknown, gray, black | 0 |
| q22 | unknown_entity | What is XQZ? | unknown, gray, ALPHA | 0 |
| q23 | unknown_entity | What is the color of the elephant? | unknown, red, black | 0 |
| q24 | unknown_attribute | What is the weight of the cat? | unknown, gray, light | 0 |
| q25 | unknown_attribute | What is the height of the horse? | unknown, black, tall | 0 |
| q26 | italian | Di che colore è il cavallo? | nero, grigio, rosso | 0 |
| q27 | transitivity | The robot is the machine, and the machine is the robot. Is the robot the machine? | yes, no, unknown | 0 |
| q28 | negation | The cat is not black. What is the cat? | gray, black, red | 0 |
| q29 | insurance_routing | The cat is a pet. How is the cat claimed? | covered by policy, not covered, unknown | 0 |
| q30 | multi_hop | Alice owns the cat, and the owner will be paid. What will happen to Alice? | paid, unpaid, unknown | 0 |
| q31 | exception | The cat is not gray, but what is its color in the dim light? | grey, gray, black | 0 |
| q32 | composition | The robot is a machine, and the machine is ALPHA. What is the code? | ALPHA, BETA, machine | 0 |

---

## 2. Laya baseline numbers available

**There are no measured Laya benchmark numbers in the donor.** The donor has no
`artifacts/` directory, no benchmark script, and no results JSON:

```
$ ls -d _reference/laya.c/artifacts   ->  does not exist
$ find _reference/laya.c -iname '*bench*' -o -iname '*perf*' -o -iname '*latency*'
  -> (no matches outside .git)
```

What the donor *does* contain is the **specification** of the benchmark, not its
results:

| Path:line | Content | Is it a measurement? |
|---|---|---|
| `docs/LAYA_CUDA_GB10_IMPLEMENTATION.md:1259-1270` | "Level D — performance benchmark": measure startup/residency, single-question latency, multi-question latency, GPU memory | **No** — a list of metrics to measure |
| `docs/LAYA_JEV_SERVER_MIGRATION.md:1283` | "Numbers above are placeholders only." | **No** — explicitly disclaims the numbers |
| `docs/LAYA_JEV_SERVER_MIGRATION.md:1400-1420` | "Do not optimize HTTP before measuring"; expected hot path | **No** — expectation, not measurement |
| `docs/DEFERRED_WORK.md:36` | tokenizer tables 3.6 MB | build artifact size, not a benchmark |
| `models/laya-bf16.manifest.json` | `safetensors_bytes: 842609210` | model size (842.6 MB BF16), not a benchmark metric |

Therefore the Laya column of the §45 comparison table is **`N/A` for every
benchmark row**. Do not estimate. The only real donor number that can be quoted
is the model size (842,609,210 B BF16), which is not one of the §45 rows.

---

## 3. Benchmark script plan

### 3.1 Donor script

There is **no donor benchmark script**. The only donor client is
`_reference/laya.c/tests/unit/systemone_smoke.py` (correctness only: parser,
invalid requests, response shapes, batching, contamination). It has no timing,
no warmup, no ladder, no artifact output. It is the model for the M4 *smoke*
test (M4 §23), not for the benchmark.

### 3.2 Script sera.c needs

**One file: `tools/bench_systemone.py`.** Python standard library only
(`argparse`, `json`, `os`, `statistics`, `subprocess`, `sys`, `threading`,
`time`, `urllib.request`, `urllib.error`). No `requests`, no `numpy`.

CLI:

```
python3 tools/bench_systemone.py \
    --base-url http://127.0.0.1:8000 \
    --fixture docs/research/m4/bench_systemone.json \
    --out artifacts/m4_e2e \
    --requests 10 \
    --ladder 1,2,4,8,16,32 \
    --warmup 2 \
    --concurrency 4
```

Flow:

1. **Wait for `/health`** — poll `GET /health` every 0.25 s until
   `{"status":"ok","ready":true}` or `--health-timeout` (default 300 s, the
   model load is ~5 s per `artifacts/m0_q4_load/warm_1.json` but the first
   `cudaMalloc` of 22 GB can be slower). Record `process_start_to_health_ms`
   from the script's own start (the script starts the server itself with
   `--spawn`, or the operator starts it and passes `--base-url`).
2. **Startup capture** — read the server's startup line / `GET /health` for
   `gguf_open_ms`, `residency_load_ms`, `runtime_workspace_init_ms`,
   `server_ready_ms`, `resident_model_bytes`, `gpu_memory_after_load_bytes`.
   Write `startup.json`.
3. **Cold first request** — one Q=1 request, timed, before warmup. Record
   `first_request_ms` + its breakdown (M4 §40).
4. **Warmup** — 2 requests at Q=1, discarded (M4 §31).
5. **Ladder** — for Q in 1,2,4,8,16,32: send `--requests` (default 10; 5 if
   `--requests 5`) sequential requests, each with the first Q questions of the
   fixture. Record per-request wall latency, HTTP status, and the `stats` block
   if present. Write `q{Q}.json`.
6. **Concurrency smoke** — 4 threads, each one Q=1 request, released by a
   `threading.Barrier`. Record status, latency, `queue_wait_ms`, and the
   discrete `choice` per request. Write `concurrency4.json`. Assert all 200 and
   no crash; do not assert latency.
7. **Summary** — compute the §34–§38 metrics and write `summary.md`.

Rules:

* Sequential requests only in the ladder (M4 §33: Q is questions per request,
  not clients).
* `stats` is only present when the server runs with `--debug-stats` (M4 §19).
  If absent, set every breakdown field to `null` and mark the breakdown section
  `N/A` in `summary.md` — do not fabricate.
* GPU memory: prefer the server-reported `peak_cuda_bytes`; optionally sample
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` via
  `subprocess` before/after each Q. On GB10 unified memory this may be
  unavailable → record `null` / `N/A`.
* Percentiles: nearest-rank on the sorted sample,
  `p50 = sorted[ceil(0.50*n)-1]`, `p95 = sorted[ceil(0.95*n)-1]`. With n=10,
  p95 is the max — state that in `summary.md`.
* Never write outside `--out`.

---

## 4. Metrics to compute (M4 §34–§38)

Per Q, from the `--requests` samples:

| Metric | Definition | Unit |
|---|---|---|
| `p50_ms` | nearest-rank p50 of E2E wall latency | ms |
| `p95_ms` | nearest-rank p95 | ms |
| `mean_ms` | arithmetic mean | ms |
| `min_ms`, `max_ms` | sample extremes | ms |
| `questions_per_s` | `Q / (mean_ms/1000)` | q/s |
| `e2e_ms_per_question` | `mean_ms / Q` | ms/q |
| `breakdown_median_ms` | median of each `stats` field over the samples | ms |
| `server_overhead_ms` | `mean_ms - median(inference_ms)` | ms |
| `server_overhead_percent` | `100 * server_overhead_ms / mean_ms` | % |
| `amortized_prefix_ms_per_question` | `median(prefix_prefill_ms) / Q` | ms/q |
| `gpu_memory.after_load_bytes` | from startup | B |
| `gpu_memory.after_first_request_bytes` | after the cold request | B |
| `gpu_memory.peak_bytes` | max over the Q samples | B |
| `kv.prefix_kv_bytes` | `stats.prefix_kv_bytes` | B |
| `kv.suffix_kv_bytes` | `stats.suffix_kv_bytes` | B |
| `kv.workspace_bytes` | `stats.workspace_bytes` | B |
| `accuracy` | `matches/total` vs fixture `expected` | count |

`inference_ms` = `prefix_prefill_ms + suffix_forward_ms + candidate_head_ms`
(the three disjoint runtime phases; `src/q3_decide.cu:522-524` already subtracts
the head from the forward so they do not overlap).

`server_ms` = `total_ms - inference_ms` (parse + validate + tokenize + queue +
serialize), as defined by the M4 §19 stats block.

---

## 5. Artifacts (M4 §43)

Directory: `artifacts/m4_e2e/`.

### `startup.json`

```json
{
  "commit": "string",
  "model": "models/Qwen3-32B-Q4_K_M.gguf",
  "model_file_bytes": 19762149024,
  "gpu": "NVIDIA GB10 (sm_121a)",
  "server_cmd": ["build/q3-server", "--model", "...", "--port", "8000"],
  "process_start_to_health_ms": 0.0,
  "gguf_open_ms": 0.0,
  "residency_load_ms": 0.0,
  "runtime_workspace_init_ms": 0.0,
  "server_ready_ms": 0.0,
  "resident_model_bytes": 0,
  "gpu_memory_after_load_bytes": 0,
  "first_request_ms": 0.0,
  "first_request_breakdown_ms": {
    "queue_wait_ms": 0.0, "parse_ms": 0.0, "tokenize_ms": 0.0,
    "prefix_prefill_ms": 0.0, "suffix_forward_ms": 0.0,
    "candidate_head_ms": 0.0, "response_build_ms": 0.0,
    "inference_ms": 0.0, "server_ms": 0.0, "total_ms": 0.0
  }
}
```

All times ms, all sizes bytes. `null` when the server does not report the field.

### `q1.json` … `q32.json`

```json
{
  "q": 8,
  "requests": 10,
  "state_chars": 515,
  "state_tokens": 0,
  "suffix_tokens_total": 0,
  "http_status": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
  "latency_ms": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
  "questions_per_s": 0.0,
  "e2e_ms_per_question": 0.0,
  "breakdown_median_ms": {
    "queue_wait_ms": 0.0, "parse_ms": 0.0, "validation_ms": 0.0,
    "tokenize_state_ms": 0.0, "tokenize_questions_ms": 0.0,
    "prefix_prefill_ms": 0.0, "branch_build_ms": 0.0,
    "suffix_forward_ms": 0.0, "candidate_head_ms": 0.0,
    "response_build_ms": 0.0, "inference_ms": 0.0,
    "server_ms": 0.0, "total_ms": 0.0
  },
  "server_overhead_ms": 0.0,
  "server_overhead_percent": 0.0,
  "amortized_prefix_ms_per_question": 0.0,
  "kv": {"prefix_kv_bytes": 0, "suffix_kv_bytes": 0, "workspace_bytes": 0},
  "gpu_memory": {
    "after_load_bytes": 0, "after_first_request_bytes": 0, "peak_bytes": 0
  },
  "accuracy": {"matches": 0, "total": 8},
  "raw": [{"i": 0, "status": 200, "latency_ms": 0.0, "stats": {}}]
}
```

`raw` keeps the unmodified per-request `stats` so the summary can be recomputed
without re-running the server.

### `concurrency4.json`

```json
{
  "clients": 4,
  "questions_per_request": 1,
  "wall_ms": 0.0,
  "all_status_200": true,
  "requests": [
    {"i": 0, "status": 200, "latency_ms": 0.0, "queue_wait_ms": 0.0,
     "choice": "gray", "expected": "gray", "correct": true}
  ],
  "queue_wait_ms": {"min": 0.0, "p50": 0.0, "max": 0.0},
  "distinct_choices": ["gray"],
  "correct": true
}
```

### `summary.md`

See §6.

---

## 6. `summary.md` template (M4 §44)

```text
commit
model
model bytes
GPU

startup_ms
resident_model_bytes
GPU memory after load

Q1:
    p50
    p95
    q/s
    ms/q

Q2:
...

Q32:
...

server overhead:
    Q1
    Q8
    Q32

runtime breakdown:
    tokenize
    prefix
    suffix
    candidate head

GPU peak:
    Q1
    Q8
    Q32

correctness:
    smoke pass/fail

main bottleneck:
    ...
```

Fill-in form (keep it short and numeric):

```markdown
# M4 E2E benchmark

commit: <git rev-parse HEAD>
model: models/Qwen3-32B-Q4_K_M.gguf
model bytes: 19,762,149,024
GPU: NVIDIA GB10 (sm_121a)

startup_ms: <process_start_to_health_ms>
resident_model_bytes: <resident_model_bytes>
GPU memory after load: <gpu_memory_after_load_bytes>

| Q | p50 ms | p95 ms | q/s | ms/q |
|---|--------|--------|-----|------|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 32 | | | | |

server overhead:
  Q1:  <server_overhead_ms> ms (<server_overhead_percent> %)
  Q8:  ...
  Q32: ...

runtime breakdown (median ms):
  tokenize:       <tokenize_state_ms + tokenize_questions_ms>
  prefix:         <prefix_prefill_ms>
  suffix:         <suffix_forward_ms>
  candidate head: <candidate_head_ms>

GPU peak:
  Q1:  <peak_bytes>
  Q8:  <peak_bytes>
  Q32: <peak_bytes>

correctness:
  smoke: PASS|FAIL
  ladder accuracy: <matches>/<total>

main bottleneck:
  1. <bucket> — <ms> ms (<pct> % of E2E)
  2. ...
  3. ...
```

Percentiles are nearest-rank; with 10 samples p95 == max. State the sample count
next to the table.

---

## 7. Laya comparison table (M4 §45)

```text
metric              Laya        Q3 Q4
------------------------------------------------
startup
Q1 latency
Q8 latency
Q32 latency
Q1 questions/s
Q8 questions/s
Q32 questions/s
peak GPU memory
```

Filled:

| metric | Laya | Q3 Q4 |
|---|---|---|
| startup | N/A | `<startup_ms>` |
| Q1 latency | N/A | `<q1.p50_ms>` |
| Q8 latency | N/A | `<q8.p50_ms>` |
| Q32 latency | N/A | `<q32.p50_ms>` |
| Q1 questions/s | N/A | `<q1.questions_per_s>` |
| Q8 questions/s | N/A | `<q8.questions_per_s>` |
| Q32 questions/s | N/A | `<q32.questions_per_s>` |
| peak GPU memory | N/A | `<max peak_bytes>` |

Every Laya cell is `N/A` because the donor contains no measured numbers (§2).
Add one line under the table: *"Laya column: no measured benchmark exists in
`_reference/laya.c` @ 83e7b97; only the Level D metric list
(`docs/LAYA_CUDA_GB10_IMPLEMENTATION.md:1259`) and explicitly-placeholder
numbers (`docs/LAYA_JEV_SERVER_MIGRATION.md:1283`)."*

---

## 8. Bottleneck classification (M4 §46)

Buckets and how each is computed from the telemetry:

| Bucket | Source | Computation |
|---|---|---|
| startup | `startup.json` | `process_start_to_health_ms`; sub-split `gguf_open_ms`, `residency_load_ms`, `runtime_workspace_init_ms`, `server_ready_ms` |
| HTTP/parser | `stats.parse_ms` + `stats.validation_ms` | median over samples |
| tokenization | `stats.tokenize_state_ms` + `stats.tokenize_questions_ms` | median sum |
| prefix prefill | `stats.prefix_prefill_ms` | median |
| suffix forward | `stats.suffix_forward_ms` | median |
| candidate head | `stats.candidate_head_ms` | median |
| response serialization | `stats.response_build_ms` | median |
| queueing | `stats.queue_wait_ms` | median (≈0 sequential; the concurrency smoke is where it shows) |

Then report the **top 3 by measured contribution** to E2E, with ms and % of
`mean_ms`. Do not implement any optimization (M4 §46, §48).

**Prediction to check, not to assume.** From `artifacts/m3_batch/ladder.json` the
prefix prefill is 743.43 ms and is paid **once per HTTP request** (M4 §14: one
request = one prefix). The suffix forward is 5.51 ms at B=1 and 35.64 ms at B=32;
the candidate head is ~0.09 ms. So the expected E2E is ≈ 749 ms at Q=1 and
≈ 779 ms at Q=32, i.e. q/s should scale almost linearly with Q and the prefix
prefill should be the #1 bucket at every Q. If the measured numbers disagree,
the first thing to check is whether the server is re-prefilling per request or
reusing a sealed prefix.

---

## 9. Existing sera.c telemetry → M4 breakdown

`src/q3_decide.h:50-86` (`q3_decide_stats`) already produces:

| M3 field | Line | M4 bucket |
|---|---|---|
| `prefix_prefill_ms` | `q3_decide.h:51` | prefix prefill |
| `branch_create_ms` | `q3_decide.h:52` | branch build (descriptor only) |
| `suffix_forward_ms` | `q3_decide.h:53` | suffix forward |
| `candidate_head_ms` | `q3_decide.h:54` | candidate head |
| `cold_total_ms` | `q3_decide.h:55` | cold first request |
| `warm_total_ms` | `q3_decide.h:56` | inference total |
| `suffix_questions_per_s` | `q3_decide.h:58` | q/s (runtime only) |
| `warm_ms_per_question` | `q3_decide.h:59` | ms/q (runtime only) |
| `prefix_kv_bytes_physical` | `q3_decide.h:61` | KV memory |
| `expected_prefix_kv_bytes` | `q3_decide.h:62` | KV memory (gate) |
| `suffix_kv_bytes` | `q3_decide.h:63` | KV memory |
| `workspace_bytes` | `q3_decide.h:64` | workspace |
| `peak_cuda_bytes` | `q3_decide.h:65` | GPU peak |
| `prefix_allocations` / `prefix_bytes` | `q3_decide.h:67-68` | allocation gate |
| `prefix_copy_calls` / `prefix_copy_bytes` | `q3_decide.h:69-70` | must stay 0 |
| `suffix_allocations` / `suffix_bytes` | `q3_decide.h:71-72` | allocation gate |
| `cuda_allocations` / `cuda_frees` | `q3_decide.h:74-75` | must be 0 warm |
| `host_syncs` | `q3_decide.h:76` | must be 1 warm |
| `kernel_launches_per_batch` / `_per_question` | `q3_decide.h:78-79` | launch overhead |
| `batch` / `suffix_tokens_total` | `q3_decide.h:81-82` | shape |
| `accuracy_matches` / `accuracy_total` | `q3_decide.h:84-85` | correctness column |

Not yet produced by M3 and therefore **new work for the server** (M4 §18):
`parse_ms`, `validation_ms`, `tokenize_state_ms`, `tokenize_questions_ms`,
`response_build_ms`, `queue_wait_ms`, `server_ms`, `total_ms`. The donor already
has `queue_wait_ms` and `total_ms` on the job
(`_reference/laya.c/src/server/laya_server.c:900`, set at `:1112`).

Real M3 numbers to compare against (`artifacts/m3_batch/ladder.json`):

| B | suffix tokens | suffix fwd ms | head ms | warm ms | ms/q | q/s | accuracy |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 5.5089 | 0.1143 | 455.5473 | 455.5473 | 2.1952 | 0/1 |
| 2 | 25 | 4.4706 | 0.0861 | 282.2636 | 141.1318 | 7.0856 | 1/2 |
| 4 | 53 | 6.4485 | 0.0851 | 393.2669 | 98.3167 | 10.1712 | 1/4 |
| 8 | 105 | 10.2038 | 0.0831 | 633.8520 | 79.2315 | 12.6212 | 4/8 |
| 16 | 197 | 18.4907 | 0.0841 | 1168.2208 | 73.0138 | 13.6960 | 8/16 |
| 32 | 390 | 35.6433 | 0.0853 | 2219.2809 | 69.3525 | 14.4191 | 12/32 |

Prefix prefill 743.4293 ms (once), `prefix_kv_bytes` 67,633,152 B,
`suffix_kv_bytes` 318,767,104 B, `prefix_copy_bytes` 0, `cuda_allocations` 0,
`host_syncs` 1 (`artifacts/m3_batch/summary.md:36`).

Startup reference (`artifacts/m0_q4_load/warm_1.json`): `gguf_open_ms` 2.684,
`plan_ms` 1.150, `cuda_alloc_ms` 2211.137, `source_copy_ms` 2039.191,
`total_load_ms` 4774.269, `model_file_bytes` 19,762,149,024,
`resident_bytes` 19,756,174,336, `cuda_allocated_bytes` 22,308,699,136.

> Note: `warm_total_ms` in the M3 ladder (455 ms at B=1) is much larger than
> `suffix_forward_ms` (5.5 ms) because it includes the per-batch host-side
> packing and the single boundary sync. The M4 `inference_ms` must be defined as
> the sum of the three disjoint phases (§4), not as `warm_total_ms`, or the
> server overhead will be understated by ~450 ms.

---

## 10. Risks / open items

| # | Risk | Mitigation |
|---|---|---|
| R1 | The M4 suffix prompt (built from `instructions`+`criteria`) is not the M3 `suffix_prompt`, so M4 token counts and accuracy will differ from the M3 ladder | Record both; M4 measures E2E, M3 stays the token-level reference. Do not claim parity. |
| R2 | `stats` requires `--debug-stats`; without it the whole breakdown is `N/A` | Run the benchmark server with `--debug-stats`; the script must degrade to `null`, never fabricate. |
| R3 | Prefix prefill (743 ms) is paid per request and will dominate every Q | Expected, not a bug. Report `amortized_prefix_ms_per_question`; do not optimize (M4 §48). |
| R4 | `peak_cuda_bytes` is not populated by `q3_decide_run` (`src/q3_decide.cu:536-540` sets only allocations/frees/syncs) | The server must fill it, or the script falls back to `nvidia-smi` / `N/A`. |
| R5 | GB10 unified memory may make `nvidia-smi` memory reporting meaningless | Prefer server-reported bytes; mark `N/A` otherwise. |
| R6 | 10 samples make p95 == max | State the sample count in `summary.md`. |
| R7 | The donor fixture has only 6 questions | Use it for the smoke test only; the ladder uses the 32-question sera.c workload. |