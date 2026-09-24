# sera.c

A from-scratch **Qwen3-32B Q4 inference runtime** for a single NVIDIA GB10
(Blackwell, `sm_121a`, aarch64): GGUF reader, residency planner, resident CUDA
loader, native Q4_K MMQ/MMVQ compute, paged KV with shared-prefix batching, a
decision head, and a System One HTTP server.

No third-party runtime is linked, loaded or executed. The build produces a
static-feeling single binary plus one server binary; all compute is in-tree
C/CUDA.

```
Qwen3-32B Q4 GGUF
      ↓ mmap
metadata + tensor descriptors
      ↓ residency plan
2 × bounded pinned staging
      ↓ cudaMemcpyAsync
final resident CUDA storage
      ↓
forward (RMSNorm, RoPE, attention, MLP, LM head)
      ↓
shared-prefix KV + packed suffix branches
      ↓
candidate logits → typed decision
      ↓
System One HTTP answer
```

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | GGUF mmap, inspect, residency plan, fast resident CUDA load | complete |
| M1 | native Q4_K quantized linear (MMQ/MMVQ) | complete |
| M2 | one-pass forward (embedding → 64 layers → final norm → LM head) | complete |
| M3 | shared-prefix KV, packed suffix batching, decision workload | complete |
| M4 | System One HTTP server + end-to-end benchmark | complete |
| M5 | Q4_K production GGUF, Python Q4 oracle, parity artifacts | complete |

Current accuracy on the frozen 32-question decision workload
(`docs/research/m4/bench_systemone.json`) is **32/32 for both the CLI and the
server**, using the numeric v4 protocol (numbered options, digit candidates).

Open numerical debt is tracked in [`docs/CORRECTNESS_DEBT.md`](docs/CORRECTNESS_DEBT.md).
The binding milestone specifications are [`docs/M0.md`](docs/M0.md) …
[`docs/M4.md`](docs/M4.md); porting rules and donor provenance are in
[`THIRD_PARTY_NOTES.md`](THIRD_PARTY_NOTES.md).

## Requirements

| | |
|---|---|
| GPU | NVIDIA GB10 (Blackwell, compute capability 12.1). Other targets are refused, never silently degraded. |
| OS / arch | Linux, `aarch64` (GB10 SBSA) |
| CUDA toolkit | 13.0 (`nvcc`, `-gencode arch=compute_121a,code=sm_121a`) |
| Host compiler | a C99 compiler (`cc`) and `nvcc` (C++17) |
| RAM / unified memory | ≥ 64 GB free; the Q4 model is ~19.8 GB resident and a full run also maps the file |
| Python (tools only) | 3.10+, `numpy`, `torch` (CUDA), `transformers` — needed only for the artifact pipeline and the oracle, not to build or run the engine |
| Model file | `models/Qwen3-32B-Q4_K_M.gguf` or `models/Qwen3-32B-Q4-production.gguf` (both git-ignored) |

> **Memory warning.** On GB10 the CPU and GPU share one 121.6 GB pool. Never
> run a float32 model path (~131 GB): the host OOM killer will take the process
> down mid-load. Keep `earlyoom` (or similar) armed if you experiment.

## Install

The repository is self-contained: no package to install, no submodule.

```bash
git clone https://github.com/luca-saggese/sera.c
cd sera.c

# The model is not in the repository. Place a Q4 GGUF in models/:
ls models/Qwen3-32B-Q4_K_M.gguf
```

Python tooling (artifact pipeline, oracle, workload tokenizer, benchmark) is
optional and used from the checkout:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy torch transformers
```

## Build

```bash
make            # -> ./q3        (CLI: inspect / load / forward / decisions)
make server     # -> build/q3-server (System One HTTP server)
make -j8        # parallel build
make clean      # removes objects, ./q3, build/, test binaries
```

Everything is one configuration: `-O3 -g` with diagnostics counters enabled,
so `--load-only --json` telemetry is always available. `make clean` also
deletes the test binaries; `make test` rebuilds them.

## Run

### CLI — inspect the model

```bash
./q3 --platform                              # CUDA + host memory probe
./q3 --inspect       models/Qwen3-32B-Q4_K_M.gguf
./q3 --list-tensors  models/Qwen3-32B-Q4_K_M.gguf
./q3 --memory-plan   models/Qwen3-32B-Q4_K_M.gguf
./q3 --load-only     models/Qwen3-32B-Q4_K_M.gguf --json   # load telemetry
```

`--inspect` reports GGUF version, architecture, tensor count, alignment, qtypes
and bytes per qtype; `--load-only --json` reports the residency plan, staging,
H2D bytes, allocation and sync counts, RSS and CUDA memory.

### CLI — correctness workload (32 questions)

```bash
./q3 --bench-decisions models/Qwen3-32B-Q4-production.gguf
# fixture:  docs/research/m4/bench_systemone.json
# accuracy: 32/32
```

This command runs the *same* production entry point as the server
(`q3_systemone_run()`), so tokenization, the `QUESTION/OPTIONS/ANSWER` suffix,
the `1/2/3` candidate digits and the result mapping have exactly one
implementation. Pass `--workload <request.json>` to use another System One
request fixture, and `--verbose` to print the per-question predictions.

Other CLI modes: `--bench-q4-linear <model.gguf>` (Q4_K linear primitive
benchmark) and `--forward <model.gguf> --tokens ids.bin
--candidate-token-ids a,b,c` (one-pass forward over pre-tokenized IDs).

### Server — System One HTTP API

```bash
./build/q3-server \
    --model models/Qwen3-32B-Q4-production.gguf \
    --host 127.0.0.1 --port 8000
```

The model is loaded once, at startup (~5 s); wait for `ready`:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","ready":true,...}
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/systemone`.
Request shape (see [`tests/fixtures/systemone_smoke.json`](tests/fixtures/systemone_smoke.json)):

```json
{
  "model": "q3-local",
  "state": "The cat is gray. The horse is black.",
  "questions": {
    "q01": {
      "type": "choice",
      "instructions": "What color is the cat?",
      "criteria": {"gray": "gray", "black": "black", "red": "red"}
    }
  }
}
```

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/systemone_smoke.json | python3 -m json.tool
```

Question types are `choice` (criteria as an object of `key: label`), `score`
(criteria as an ordered array of levels) and `noul` (yes/no/unknown). Answers
carry the chosen key, per-option probabilities, confidence and token usage. One
request is one shared state prefill plus one packed suffix batch, so up to 32
questions per request cost a single prefix pass.

Server options: `--served-model-name`, `--model-alias`, `--device`, `--cors`,
`--max-body-mb`, `--queue-depth`, `--api-key` (`-h` for the full list).

### Tests

```bash
make test                 # all C/CUDA unit tests (skipped if the model is absent)
Q3_TEST_MODEL=models/Qwen3-32B-Q4-production.gguf make test

python3 tests/systemone_smoke.py --base-url http://127.0.0.1:8000
python3 tools/bench_systemone.py --base-url http://127.0.0.1:8000 \
        --fixture docs/research/m4/bench_systemone.json \
        --out artifacts/m4_e2e
```

## Repository layout

```
Makefile                single-configuration build (CLI + server)
src/q3_gguf.*           GGUF v3 reader: mmap, metadata, tensor descriptors
src/q3_residency*.*     residency plan (spans, totals, overlap checks)
src/q3_binder.*         GGUF tensors -> typed weight descriptors
src/q3_model.*          Qwen3 config extraction and validation
cuda/q3_cuda.cu         CUDA context, errors, timers, memory accounting
cuda/q3_model_loader_cuda.cu  2 × bounded pinned staging -> resident storage
cuda/q3_forward.cu      layers, attention, KV, LM head, workspaces
cuda/q3_decide.cu       shared-prefix branches, packed batched candidate head
src/q3_systemone.cu     the one runtime boundary behind the server
src/q3_tokenizer.c      GGUF-vocab tokenizer + Qwen3 chat template
src/server/q3_server.c  HTTP/JSON server (no CUDA/GGUF types leak in)
src/io/hd_json.*        minimal JSON parser used by server and CLI
src/q3_decide_cli.cu    CLI correctness run, through q3_systemone_run()
src/q3_main.c           CLI modes and option parsing
cuda/mmq/               native Q4_K MMQ/MMVQ kernels (donor lineage)
tools/                  Python: quantize -> write GGUF -> oracle -> benchmarks
docs/                   milestone specs M0-M4, research notes, correctness debt
artifacts/              measured artifacts per milestone (JSON + summary.md)
tests/                  unit tests (C/CUDA) + System One wire smoke test
```

## Artifact pipeline (tools/)

The engine consumes a GGUF; these tools produce and verify it. They need
`torch`, `transformers` and `numpy`, and the FP8/bf16 source checkpoint (kept
outside the repository).

```bash
# 1. quantize once: FP8 checkpoint -> Q4_K payload store (never quantizes twice)
python3 tools/qwen3_quantize_model.py \
    --checkpoint artifacts/m5_parity/hf_checkpoint \
    --outdir     artifacts/m5_parity/q4

# 2. package the frozen Q4_K bytes + F32 norms into a GGUF (never quantizes)
python3 tools/qwen3_write_gguf.py \
    --checkpoint artifacts/m5_parity/hf_checkpoint \
    --q4dir      artifacts/m5_parity/q4 \
    --out        models/Qwen3-32B-Q4-production.gguf \
    --name       Qwen3-32B-Q4-production

# 3. Python Q4 oracle: same bytes, dequantized, run through HF Qwen3
python3 tools/qwen3_q4_oracle.py --out artifacts/m5_parity/golden_logits.json

# 4. tokenizer-only workload artifact for the M3 tests
python3 tools/tokenize_workload.py --gguf models/Qwen3-32B-Q4_K_M.gguf

# 5. end-to-end server benchmark
python3 tools/bench_systemone.py --base-url http://127.0.0.1:8000
```

`tools/qwen3_q4_oracle.py` is the frozen reference: it dequantizes exactly the
bytes written to `artifacts/m5_parity/q4/` and never re-quantizes, so it can be
compared with the C/CUDA path without a precision mismatch.

## License and provenance

MIT. A small number of source files were copied from external projects and
adapted in-tree under the project rule **COPY → RENAME → EDIT**; provenance,
donor commits and licenses are recorded in
[`THIRD_PARTY_NOTES.md`](THIRD_PARTY_NOTES.md).
