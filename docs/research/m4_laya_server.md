# M4 — Laya System One server port (Worker A)

Scope: `docs/M4.md` §0–§30, §50–§54 (Worker A). Deliverable: the concrete port
plan for putting the M0–M3 Q3 runtime behind the donor's System One HTTP server.

Donor: `_reference/laya.c` @ `83e7b97fd0fbd53b919d200a11240137a8ffaf04` (read-only,
never modified). Clone URL: `https://github.com/luca-saggese/laya.c`.

---

## 0. Sources actually read

| Source | What it gave |
|---|---|
| `docs/M4.md` §0–§30, §50–§54 | binding requirements (copy→rename→remove→wire) |
| `_reference/laya.c/src/server/laya_server.c` (1537 L) | the whole server, read in full |
| `_reference/laya.c/src/io/hd_json.{c,h}` (282+61 L) | the JSON layer the server depends on |
| `_reference/laya.c/src/laya/{laya_infer,decision,laya_sequence,model}.h` | the model seam + calibration internals |
| `_reference/laya.c/include/laya.h` (66 L) | `laya_status` / error ABI |
| `_reference/laya.c/src/tokenizer/laya_tokenizer.{c,h}` (758+44 L) | donor tokenizer (mmBERT — wrong vocab) |
| `_reference/laya.c/tests/unit/systemone_smoke.py` (224 L) | wire smoke to copy |
| `_reference/laya.c/fixtures/laya/systemone_smoke.json` (52 L) | smoke fixture to copy |
| `_reference/laya.c/Makefile` (69 L) | donor build integration |
| `_reference/laya.c/docs/LAYA_CUDA_GB10_IMPLEMENTATION.md:1259` | "Level D" benchmark *methodology* only |
| `_reference/q38.c/q38_tokenizer.c` (609 L) | Qwen3-compatible byte-level BPE tokenizer |
| `src/q3_decide.h` | M3 public ABI the server must drive |
| `src/q3_decide_cli.cu:66–330` | `q3_cmd_bench_decisions`, the end-to-end M3 driver |
| `src/q3_workload.h`, `src/q3_forward.h`, `src/q3.h`, `Makefile` | sera.c integration surface |
| `artifacts/m3_batch/{ladder.json,summary.md}`, `artifacts/m0_q4_load/summary.md` | real sera.c numbers |

---

## 1. Donor commit

| field | value |
|---|---|
| commit | `83e7b97fd0fbd53b919d200a11240137a8ffaf04` |
| date | 2026-09-21 |
| clone URL | `https://github.com/luca-saggese/laya.c` |
| local path | `_reference/laya.c` (read-only) |

---

## 2. Files to copy

| Donor path | Lines | Target in sera.c | Role |
|---|---|---|---|
| `src/server/laya_server.c` | 1537 | `src/server/q3_server.c` | the entire HTTP server (transport + queue + worker + endpoints) |
| `src/io/hd_json.c` | 282 | `src/io/hd_json.c` | JSON parser/emitter the server uses (`hd_json_parse`, `hd_json_get`, …) |
| `src/io/hd_json.h` | 61 | `src/io/hd_json.h` | its public ABI (`hd_json_parse` L50, `hd_json_get` L53, `hd_json_string` L54, `hd_json_array_at` L59) |
| `tests/unit/systemone_smoke.py` | 224 | `tests/systemone_smoke.py` | live wire smoke (M4 §23) |
| `fixtures/laya/systemone_smoke.json` | 52 | `tests/fixtures/systemone_smoke.json` | smoke fixture (1 state, 6 questions) |
| `src/runtime/laya_timing.{c,h}` | 249+74 | **do NOT copy** | only used by donor `src/main.c:152`; the server never calls it |

Not copied (model-specific, replaced by Q3): `src/laya/*`, `src/tokenizer/*`,
`src/cuda/*`, `src/main.c`, `src/io/hd_gguf.c`.

sera.c has **no** `src/server/` and **no** `src/io/` directory today — both must be
created. sera.c has **no JSON library at all**, so `hd_json.{c,h}` is a hard
dependency of the port (the server parses the body, validates it, and re-serializes
a normalized request for the worker: `parse_systemone` L723, `serialize_runtime_request` L869).

---

## 3. Symbols to rename / replace

### 3.1 Mechanical rename (pure text substitution)

| Donor | sera.c | Where |
|---|---|---|
| `laya-server` | `q3-server` | `usage` L1361, `main` L1424/1426/1440/1450/1466/1470/1526 |
| `LAY A_SERVER_*` → `LAYA_SERVER_*` | `Q3_SERVER_*` | L64–66 (`IO_TIMEOUT_SEC`, `SEND_STALL_TIMEOUT_MS`, `MAX_HEADER_BYTES`) |
| `laya_server.c` | `q3_server.c` | build only |
| `server_config` / `g_cfg` | `q3_server_config` / `g_q3_cfg` | L587/600 |
| `server_job` / `g_queue` | `q3_server_job` / `g_q3_queue` | L889/941 |
| `sysone_questions` | `q3_systemone_questions` | L639 |
| `LAY A_SERVER_TEST` | `Q3_SERVER_TEST` | L1378/1536 |
| `laya-local` (default served id) | `q3-local` | smoke test L94 + fixture L3 |

### 3.2 Symbols to REPLACE (model logic — never renamed, deleted)

| Donor symbol | file:line | Replacement |
|---|---|---|
| `laya_model` struct | `src/laya/model.h:112–129` | `q3_weights` + `q3_model_config` + `q3_forward_runtime` |
| `laya_model_load` | `src/laya/model.h:133` | `q3_gguf_open`→`q3_model_config_from_gguf`→`q3_loader_load`→`q3_weights_bind`→`q3_forward_create` (see §5) |
| `laya_model_free` | `src/laya/model.h:139` | `q3_forward_destroy` + loader teardown |
| `laya_model_last_error` | `src/laya/model.h:141` | local `char error[]` from the Q3 calls |
| `laya_model_report` | `src/laya/model.h:137` | drop (or `--debug-stats` summary) |
| `laya_infer_result` | `src/laya/laya_infer.h:23–36` | `q3_decision_result` (`src/q3_decide.h:44–49`) + server-side key/label arrays |
| `laya_infer_run` | `src/laya/laya_infer.h:44` | **`q3_systemone_run(...)`** (new, §5) |
| `laya_infer_results_free` | `src/laya/laya_infer.h:48` | free of the new result array |
| `LAY A_INFER_MAX_Q` | `src/laya/laya_infer.h:21` | `Q3_SYSTEMONE_MAX_Q` (=100) |
| `laya_decision_result` | `src/laya/decision.h:52` | `q3_decision_result` |
| `laya_decision_init` | `src/laya/decision.h:107` | `q3_prefix_create` / `q3_branch_set_create` |
| `laya_decision_forward_batch` | `src/laya/decision.h:124` | `q3_decide_run` |
| `laya_decision_forward` | `src/laya/decision.h:133` | `q3_decide_run` with `count==1` |
| `laya_decision_ws` / `laya_decision` | `src/laya/decision.h:93/96` | `q3_prefix_kv` / `q3_branch_set` |
| `laya_build_sequence` | `src/laya/laya_sequence.h:51` | `q3_build_decision_suffix(...)` (new, M4 §501) |
| `laya_render_options` | `src/laya/laya_sequence.h:44` | `q3_render_options(...)` (new) |
| `laya_question_internal` | `src/laya/laya_sequence.h:58` | `q3_question_internal(...)` (new) |
| `laya_qtype_of` | `src/laya/laya_sequence.h:61` | `q3_qtype_of(...)` |
| `laya_options` | `src/laya/laya_sequence.h:29–35` | `q3_question_options` (new, distinct from `src/q3.h:34` `q3_options`) |
| `LAY A_QTYPE_{CHOICE,SCORE,NOUL}` | `src/laya/laya_sequence.h:22–24` | `Q3_QTYPE_*` |
| `laya_status` / `LAY A_OK` | `include/laya.h:25/26` | `bool` + `char *error` (Q3 convention) |
| `laya_last_error` / `laya_set_error` | `include/laya.h:38/41` | local error buffer |
| `temperature_by_options` | `src/laya/decision.c:361–373` | **delete** — M4 §9 forbids temperature tuning |
| `confidence_from_probs` | `src/laya/decision.c:378` | reimplement as `1 - H/log K` (M4 §9) |
| `softmax_inplace` | `src/laya/decision.c:393` | `q3_decide_result_softmax` (`src/q3_decide.h:153`) |
| `act_probability` / `act_logits` | `src/laya/decision.h:50–51` | **delete** — no `action` field on the wire (smoke L8) |
| `laya_tokenizer_*` | `src/tokenizer/laya_tokenizer.h:30/33/36/38` | **do not reuse** — mmBERT vocab (§9) |

---

## 4. Server architecture to preserve (surgical port)

Every item below must survive the port unchanged except for the `laya_*`→`q3_*`
rename. Line numbers are `src/server/laya_server.c`.

| Mechanism | file:line | Note |
|---|---|---|
| signal state + handler | 61–62, 68 | `g_stop_requested`, `g_listen_fd`, `stop_signal_handler` |
| timeouts / header cap | 64–66 | 10 s IO, 2 s send-stall, 64 KiB headers |
| growable buffer | 83–87, 120–166 | `buf`, `buf_reserve/append/putc/puts/printf/take/free` |
| alloc helpers | 89–118 | `die`, `xmalloc`, `xrealloc`, `xstrdup`, `xstrndup` |
| JSON emit helpers | 175–314 | `json_ws/lit/hex`, `utf8_put`, `json_u16`, `json_string`, `json_number`, `json_escape` |
| `wall_ms` | 316 | monotonic ms clock |
| `send_all` | 322 | full-write with stall timeout |
| CORS | 350 | `append_cors_headers` |
| HTTP status/reason | 357, 375, 398 | `http_reason`, `http_response`, `http_error` |
| request struct + free | 408–424 | `http_request`, `http_request_free` |
| header parsing | 426–481 | `header_end`, `content_length` (returns −1 on malformed), `header_value` |
| request read | 483 | `read_http_request` (drains oversized body before 413) |
| client socket config | 549 | `configure_client_socket` (timeouts, TCP_NODELAY) |
| listen socket | 557 | `listen_on` (SO_REUSEADDR, bind, listen) |
| config + globals | 587–616 | `server_config`, `g_cfg`, `g_model_ready`, `g_requests_served`, `served_model_id`, `model_id_matches` |
| request validation | 639–721 | `sysone_questions`, `reject_question`, `validate_state/choice/score/noul` |
| body parse | 723 | `parse_systemone` (400 malformed / 422 schema / 404 model) |
| JSON re-emit | 821 | `json_emit_value` |
| normalized request | 869 | `serialize_runtime_request` |
| job struct | 889–908 | `server_job` (mutex + cond + `done`/`cancelled`) |
| job lifecycle | 910–939 | `server_job_free/new/fail` |
| bounded queue | 941–987 | `g_queue*`, `enqueue`, `dequeue`, `queue_stop` (mutex + cond) |
| single GPU worker | 990 | `worker_main` — the only thread that touches CUDA |
| float emit | 1008 | `emit_f32` (`%.7g`, NaN/Inf → `null`) |
| response serialization | 1029, 1089 | `emit_answers`, `systemone_body` |
| worker error helper | 1106 | `job_error` |
| client disconnect poll | 1148, 1168 | `client_socket_disconnected`, `wait_for_job_or_disconnect` (100 ms poll, sets `cancelled`) |
| connection thread | 1185 | `client_main` (one thread per connection, detached) |
| GET /v1/models | 1213 | `send_models` |
| GET /health, /healthz | 1228 | `send_health` (unauthenticated) |
| dispatch + response write | 1237 | `dispatch_job` (enqueue stamp → `queue_wait_ms`) |
| POST /v1/systemone | 1272 | `handle_systemone` (415 on wrong Content-Type) |
| Authorization | 1308 | `authorized` (constant-time Bearer compare) |
| routing | 1323 | `handle_request` (OPTIONS 204 → health → auth → models → systemone → 404) |
| CLI + lifecycle | 1359, 1379 | `usage`, `main` (model loaded **before** listen; poll loop 200 ms; graceful shutdown) |
| test guard | 1378, 1536 | `#ifndef LAYA_SERVER_TEST` around `main` |

---

## 5. The seam to replace

The server calls the model in exactly one place. `job_run` (L1110–1144) re-parses
the normalized request on the worker thread and then:

```c
/* src/server/laya_server.c:1127 */
laya_status st = laya_infer_run(&g_model, root, &res, &n, &tokens);
if (st != LAYA_OK) { job_error(j, 500, laya_last_error()); hd_json_free(root); return; }
j->body = systemone_body(res, n, &j->questions, tokens);   /* L1134 */
laya_infer_results_free(res, n);                            /* L1137 */
```

Donor signature (`src/laya/laya_infer.h:44`):

```c
laya_status laya_infer_run(const laya_model *model, const hd_json *root,
                           laya_infer_result **out_res, int *out_n,
                           long *out_input_tokens);
```

**Replacement** — one new function in the runtime, called from the same line:

```c
/* src/q3_systemone.h — new */
bool q3_systemone_run(q3_systemone_ctx *ctx, const hd_json *root,
                      q3_systemone_result **out_res, int *out_n,
                      long *out_input_tokens, char *error, size_t error_len);
```

`q3_systemone_ctx` holds the resident `q3_forward_runtime *`, `q3_weights`,
`q3_model_config`, the reusable `q3_prefix_kv *` and `q3_branch_set *`, and the
tokenizer. `q3_systemone_result` mirrors `laya_infer_result` field-for-field
(`qtype`, `n_options`, `option_keys`, `option_labels`, `choice`, `score`, `noul`,
`confidence`, `probs`) so `emit_answers` (L1029) and `systemone_body` (L1089) need
**no change at all** — only the type name.

Model lifecycle in `main` maps as follows:

| Donor | line | sera.c |
|---|---|---|
| `laya_model_load(model_path, device_id, &g_model)` | 1460 | `q3_cuda_init` → `q3_gguf_open` → `q3_model_config_from_gguf` → `q3_loader_context_create`/`q3_loader_load` → `q3_weights_bind` → `q3_forward_create(&weights,&cfg,fwd_tokens,device,err,len)` → `q3_prefix_create` → `q3_branch_set_create` |
| `laya_model_free(&g_model)` | 1472, 1481, 1532 | `q3_branch_set_destroy` → `q3_prefix_destroy` → `q3_forward_destroy` → loader/gguf teardown |

The exact call order is the one already proven in `src/q3_decide_cli.cu:66–330`
(`q3_cmd_bench_decisions`): `q3_workload_load` L78, `q3_loader_load` L97,
`q3_weights_bind` L106, `q3_forward_create` L150, `q3_prefix_create` L161,
`q3_branch_set_create` L173, `q3_prefix_prefill`+`q3_prefix_seal` L189–191,
`q3_branch_acquire` L220, `q3_decide_run` L324.

Per-request sequence inside `q3_systemone_run` (M4 §14/§15 — one request = one
prefix = one GPU job):

1. tokenize `state` → `state_tokens[]`; `q3_prefix_prefill(prefix, state_tokens, n, &stats, err, len)`; `q3_prefix_seal(prefix, err, len)`.
2. for each question: `q3_build_decision_suffix(...)` → suffix tokens; `q3_branch_acquire(set, err, len)`; fill `q3_decision_item{branch, tokens, token_count, candidate_ids, candidate_count}`.
3. one `q3_decide_run(rt, prefix, set, &batch, results, &stats, err, len)`.
4. `q3_decide_result_softmax(&results[i])`; map `predicted_index`/`probabilities` into `q3_systemone_result`; compute `confidence = 1 - H/log K` (M4 §9), `score = Σ i·p_i` (M4 §10), `noul = p[true]` (M4 §11).
5. `q3_branch_release(set, branch)` for every branch.

Constraints from the M3 ABI: `Q3_MAX_BRANCHES 32` (`src/q3_decide.h:23`),
`Q3_MAX_CANDIDATES 8` (`:24`), `Q3_DEFAULT_SUFFIX_CAP 256` (`:25`). The runtime
token capacity must be `max(prefix prefill, largest packed suffix batch)`; the M3
driver sizes `fwd_tokens = max_packed` (not `max(state_count, max_packed)`) to keep
the MMQ arena small (`src/q3_decide_cli.cu:150`). Candidate lists shorter than K
must be padded by repeating the last id (uniform-K requirement, `q3_decide_cli.cu:220`).

---

## 6. Wire contracts (verbatim from the donor)

Request (`parse_systemone` L723, fixture `fixtures/laya/systemone_smoke.json`):

```json
{ "model": "q3-local",
  "state": "<string | object | array>",
  "questions": { "<qid>": { "type": "choice|score|noul",
                            "instructions": "<string>",
                            "criteria": <object|array|absent> } } }
```

Response envelope (`systemone_body` L1089):

```json
{ "model": "<served_model_id>",
  "answers": { "<qid>": <answer> },
  "usage": { "input_tokens": <int>, "output_tokens": 0 } }
```

Answer shapes (`emit_answers` L1029–1088):

| type | shape | source |
|---|---|---|
| `choice` | `{"type":"choice","choice":<criteria key>,"probabilities":{<key>:<p>},"confidence":<f>}` | L1036–1053 |
| `score` | `{"type":"score","score":<f>,"legend":{"0":<raw criteria value>,…},"probabilities":{"0":<p>,…},"confidence":<f>}` | L1054–1078 |
| `noul` | `{"type":"noul","noul":<P(true)>}` — **no `confidence`** | L1079–1082 |

Details that must be preserved exactly:

- Floats via `emit_f32` (L1008): `%.7g`; NaN/Inf → `null`.
- `choice` keys are the caller's criteria keys, in insertion order; `choice` value is `option_keys[predicted_index]` (L1038–1042).
- `score` `legend` keys are `"0".."N"` and carry the **raw** criteria values verbatim from the request (`crit->u.array.items[k]`, L1067–1071); `probabilities` keys are the same `"0".."N"`.
- `noul` criteria keys must be `"true"`/`"false"` (`validate_noul` L705).
- Limits: choice 2–255 criteria (object, `validate_choice` L665), score 2–10 criteria (array, `validate_score` L686), ≤100 questions (`LAYA_INFER_MAX_Q`, `laya_infer.h:21`).
- `x-typesafe-request-id` request header is echoed on the response (`dispatch_job` L1254–1257).
- Status codes: 400 malformed JSON, 422 schema, 404 unknown model/endpoint, 405 wrong method, 415 wrong Content-Type, 413 body too large, 429 queue full, 401 bad API key.
- `GET /v1/models` (L1213): `{"object":"list","data":[{"id":<served_model_id>,"object":"model","created":0,"owned_by":"local","aliases":[<alias>]}]}`.
- `GET /health` (L1228): `{"status":"ok","ready":<bool>,"endpoints":["/v1/systemone","/v1/models"]}`.

---

## 7. Tests / fixtures to reuse

| Donor path | Target | Notes |
|---|---|---|
| `tests/unit/systemone_smoke.py` (224 L) | `tests/systemone_smoke.py` | edit `FIXTURE` path (L24) and `body["model"] == "laya-local"` → `"q3-local"` (L94) |
| `fixtures/laya/systemone_smoke.json` (52 L) | `tests/fixtures/systemone_smoke.json` | change `"model"` (L3) to `q3-local` |

The smoke covers: `GET /v1/models` (L69), `OPTIONS /v1/systemone` (L77), mixed
POST (L87), batching (L131), contamination — 3 identical questions, drift < 1e-6
(L147), 9 invalid requests (L157), routing 405/404. It asserts
`usage.input_tokens > 0` (L97), `usage.output_tokens == 0` (L99), no
`action`/`act_probability` fields (L8), choice probability keys == criteria keys,
score legend keys `"0".."N"` carrying raw criteria (L114–117), noul has **no**
`confidence` (L125), choice probabilities sum to 1.

The donor fixture has only 6 questions, so it cannot drive the M4 Q=8/16/32 ladder
— the ladder must use `docs/research/m3/workload.json` (32 questions). See
`docs/research/m4_benchmark_plan.md` §1.

---

## 8. Build integration

Donor (`_reference/laya.c/Makefile`):

```make
SERVER_BIN  := build/laya-server
SERVER_SRCS := src/server/laya_server.c $(CORE_SRCS) src/runtime/laya_timing.c
$(SERVER_BIN): $(SERVER_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(CUDA_OBJS) \
	      $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -lpthread
```

- `CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra`; `CPPFLAGS += -Iinclude -Isrc/io -Isrc/laya -Isrc/cuda -Isrc/runtime -Isrc/tokenizer -Isrc/server`.
- `TARGET_ARCH ?= sm_121`; CUDA objects built with `-arch=$(TARGET_ARCH) -O2 -std=c++17`.
- pthread is used by the server (`pthread_create` L1478/1515, mutex/cond L946–947) → `-lpthread` is required.

sera.c `Makefile` additions (current: `BIN := q3`, `CFLAGS ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -Isrc -pthread`, `NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a`, `CUDA_LDLIBS ?= -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -pthread`):

```make
SERVER_BIN  := build/q3-server
SERVER_OBJS := src/server/q3_server.o src/io/hd_json.o src/q3_systemone.o
$(SERVER_BIN): $(SERVER_OBJS) $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -o $@ $(SERVER_OBJS) \
	        $(filter-out src/q3_main.o,$(OBJS)) $(CUDA_LDLIBS) -lpthread
```

Notes: sera.c links with `nvcc` (not `gcc`) because the runtime is C++/CUDA; the
server `.c` files compile with the existing `%.o: %.c` rule (`$(CC) $(CFLAGS) -c`).
`-Isrc/io -Isrc/server` must be added to `CFLAGS`. `build/` does not exist yet and
is gitignored. `src/q3_main.o` must be filtered out (it defines `main`), exactly as
`TEST_Q4_OBJS` already does (`Makefile:97`).

---

## 9. Tokenizer

**The donor's tokenizer is unusable for Qwen3.** `src/tokenizer/laya_tokenizer.c`
(758 L) + `laya_tokenizer_tables.h` (251 283 L) is a byte-level BPE for **mmBERT**
with specials UNK=50280, CLS=50281, SEP=50282, PAD=50283, MASK=50284, EOT=50279
(`laya_tokenizer.h:30/33/36/38`). Wrong vocabulary — do not copy.

**sera.c has no C tokenizer.** M3 tokenization is done offline by
`tools/tokenize_workload.py` (Python `transformers.AutoTokenizer`), which is not
usable inside the server.

**Available Qwen3-compatible option:** `_reference/q38.c/q38_tokenizer.c` (609 L) is
a real byte-level BPE with the GPT-2 byte↔unicode map, longest-match added-token
handling (`encode_special`), NFC normalization (`normalize_nfc` L307) and the
Qwen2 pre-tokenizer rules (`encode_text` L186). Interface:

```c
bool q38_tokenizer_init(q38_tokenizer *t, const char *dir, void *unused,
                        char *err, size_t err_len);          /* reads <dir>/tokenizer.json + <dir>/merges.txt */
bool q38_tokenizer_encode(const q38_tokenizer *t, const char *s, bool add,
                          q38_token_batch *out, char *err, size_t err_len);
void q38_token_batch_free(q38_token_batch *out);
```

Caveat: it requires a model **directory** containing `tokenizer.json` and
`merges.txt` (`q38_tokenizer.c:425/448`). `models/qwen3_32b_bf16/` has **neither**
(only `config.json`, `generation_config.json`, 17 safetensors), so that path is not
directly usable.

**Recommended path:** build the tokenizer from the GGUF itself. `models/Qwen3-32B-Q4_K_M.gguf`
carries the full tokenizer metadata: `tokenizer.ggml.model=gpt2`,
`tokenizer.ggml.pre=qwen2`, `tokenizer.ggml.tokens` (151 936), `tokenizer.ggml.token_type`
(151 936), `tokenizer.ggml.merges` (151 387), eos=151 645, pad/bos=151 643,
`add_bos=0`, plus a chat template. So a Qwen3 tokenizer can be constructed from the
GGUF alone — no `tokenizer.json` needed. Either port `q38_tokenizer.c` and feed it
the GGUF arrays, or write a small `q3_tokenizer` that reads the GGUF KV block.
M4 §27 allows exactly one tokenizer smoke test.

---

## 10. Existing Laya baseline numbers

**N/A.** The donor repo contains **no** `artifacts/` directory and **no** measured
startup/latency/throughput/GPU-memory numbers. The only performance material is a
*methodology* description, "Level D — performance benchmark", at
`_reference/laya.c/docs/LAYA_CUDA_GB10_IMPLEMENTATION.md:1259–1272`, which lists
what to measure (startup/model residency time, single-question latency,
multi-question latency, GPU memory) but reports no values.
`_reference/laya.c/docs/LAYA_JEV_SERVER_MIGRATION.md:1283` states explicitly that
its numbers are placeholders. M4 §45 says to use only really measured Laya numbers
and otherwise write `N/A` — so the comparison table will be `N/A` for every Laya
column.

For reference, sera.c's own measured numbers (not Laya):

| metric | value | source |
|---|---|---|
| resident load, warm median | 4774.3 ms | `artifacts/m0_q4_load/summary.md` |
| prefix prefill (129 tok) | 743.4 ms | `artifacts/m3_batch/ladder.json` |
| B=1 warm | 455.5 ms, 2.20 q/s | `artifacts/m3_batch/ladder.json` |
| B=32 warm | 2219.3 ms, 69.4 ms/q, 14.42 q/s | `artifacts/m3_batch/ladder.json` |
| prefix KV bytes | 67 633 152 B (constant in B) | `artifacts/m3_batch/summary.md` |
| warm cudaMalloc/cudaFree | 0 | `artifacts/m3_batch/summary.md` |

---

## 11. Risks / gotchas

1. **`emit_answers` depends on `option_keys`/`option_labels`.** The donor fills them
   in `laya_infer_result` (`laya_infer.h:26–27`). `q3_systemone_result` must supply
   the same arrays (choice → criteria keys; score → `"0".."N"` + raw criteria for
   the legend; noul → `false`/`true`), otherwise the response serializer must be
   rewritten.
2. **`parse_systemone` keeps the parsed `hd_json *root` alive and references (does
   not copy) question keys/values** (L723–819); `serialize_runtime_request` (L869)
   re-serializes `{"state":<verbatim>,"questions":{…}}`. The `hd_json` lifetime rules
   must be preserved exactly, or the worker will read freed memory.
3. **sera.c has no JSON library.** `src/io/hd_json.{c,h}` must be copied; there is no
   alternative in-tree.
4. **`job_run` re-parses on the worker thread** (L1118–1122) so CUDA is only ever
   touched by that one thread. Do not move parsing to the client thread.
5. **`dispatch_job` overloads `j->total_ms` as the enqueue stamp** (L1238) and
   `job_run` converts it to `queue_wait_ms` (L1112). Easy to break when adding the
   M4 §18 timing fields.
6. **`wait_for_job_or_disconnect` polls every 100 ms and sets `j->cancelled`**
   (L1168); the client thread must never free a job the worker may still be inside.
7. **Model is loaded before the socket is opened** (L1455–1465). A client must never
   see a listening port that cannot serve. Preserve this ordering with the Q3 load
   sequence.
8. **`/health` and `/healthz` are unauthenticated** (L1332–1336) and OPTIONS returns
   204 before auth (L1324–1327). Keep it that way (M4 §21).
9. **`content_length` returns −1 for malformed headers** (L441) and is never treated
   as 0; `read_http_request` drains an oversized body (bounded) before returning 413
   (L483).
10. **`stop_signal_handler` closes the listen fd and `_exit(130)` on a second
    signal** (L68); the main loop polls with a 200 ms timeout (L1477) because a
    signal may land on any thread.
11. **`#ifndef LAYA_SERVER_TEST` guard around `main`** (L1378–1536) — keep it
    (renamed) so the server can be linked into a test binary.
12. **`laya_timing.{c,h}` is not used by the server** — do not copy it; M4 §18/§19
    timing must come from `q3_decide_stats` (`src/q3_decide.h:52–85`) plus
    `wall_ms` deltas.
13. **Uniform-K candidate padding.** `q3_decide_run` requires every item to have the
    same `candidate_count`; pad shorter lists by repeating the last id
    (`src/q3_decide_cli.cu:220`). The wire response must still emit only the real
    candidates.
14. **Runtime token capacity.** Size `q3_forward_create` with
    `fwd_tokens = max_packed` (largest packed suffix batch), not
    `max(state_count, max_packed)` — the M3 MMQ arena sizing bug (checkpoint 047)
    was caused by over-sizing.
15. **`q3_build_decision_suffix(...)` does not exist yet** (only named in
    `docs/M4.md:501`). It is the single central prompt builder and must be created
    in the runtime, not in the server, so the M3 CLI and the server share it.
16. **`q3_options` name collision**: `src/q3.h:34` already defines `q3_options`; the
    ported `laya_options` (`laya_sequence.h:29`) must be renamed to something else
    (e.g. `q3_question_options`).
17. **M4 §48 forbids** continuous batching, cross-request batching, multiple GPU
    workers, async CUDA, server-side prefix cache, paged KV, CUDA Graphs, TLS,
    metrics server, WebSocket, streaming, OpenAI compat. The donor server already
    complies — do not "improve" it.
18. **M4 §49**: if the server adds 1–2 ms, do not optimize; if it adds ~50 ms,
    measure precisely and document, then STOP.
19. **M4 §50 caps the work at 3 commits**: (1) port the server, (2) wire the System
    One API to the Q3 runtime, (3) benchmark end-to-end.
20. **`--debug-stats`** (M4 §19) is a new flag not present in the donor `usage`
    (L1359) — add it, gated, and keep the default response byte-identical to the
    donor's.
