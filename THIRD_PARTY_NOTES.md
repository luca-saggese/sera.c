# Third-party notices

sera.c is a self-contained C/CUDA codebase. It does not link against, load
or run any third-party runtime. A small number of source files are copied
from external projects and adapted in-tree; this file records their
provenance and license.

## q38.c — primary donor

| Field    | Value                                                             |
|----------|-------------------------------------------------------------------|
| Source   | https://github.com/luca-saggese/q38.c                             |
| Refs     | `main` (M1 donor), `qwen38-spark-proto` (M0 donor)                |
| Commit   | `c1d4597a80e300b803dc642519718f2c999589da`                        |
| License  | MIT                                                               |

M0 (`src/q3_gguf.*`, `src/q3_residency*.*`, `cuda/q3_cuda.cu`,
`cuda/q3_model_loader_cuda.*`, parts of `src/q3_main.c`) was ported from the
`qwen38-spark-proto` branch of q38.c.

M1 (`cuda/q3_mmq.*`, `src/q3_q4_linear.*`, `src/q3_quant.*`,
`cuda/vendor/*`, `tests/test_q3_q4_linear.cu`) was ported from the `main`
branch of q38.c, in particular from `cuda/mmq/`. The port follows the
project rule COPY -> RENAME -> EDIT: donor files are copied, symbols are
mechanically renamed (`ds4_*` -> `q3_*`), and DS4-specific functionality
that M1 does not need (MoE / expert IDs, Q2_K / IQ2 / MXFP4 / NVFP4 weight
paths, D2R experiments, CUDA-graph experiments, environment tuning hooks,
NVTX instrumentation) is removed.

`_reference/q38.c` and `_reference/q38-main` are local, git-ignored clones
used only as read-only references for the port. Nothing at build or run
time depends on them.

## llama.cpp — upstream of the vendored MMQ kernels

The Q4_K MMQ/MMVQ compute kernels in `cuda/vendor/` were originally ported
by q38.c (`cuda/mmq/VENDOR.md`) from llama.cpp's `ggml-cuda` backend.
sera.c inherits them from q38.c, which is the only direct donor used here.

| Field        | Value                                                                          |
|--------------|--------------------------------------------------------------------------------|
| Source       | https://github.com/ggml-org/llama.cpp                                           |
| Commit       | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`                                      |
| Commit date  | 2026-05-14                                                                      |
| License      | MIT, copyright "2023-2026 The ggml authors"                                     |

Vendored files (verbatim unless noted): `mmq.cuh`, `mma.cuh`, `vecdotq.cuh`,
`quantize.cuh`, `quantize.cu`, `mmvq.cu`, `mmvq.cuh`, `unary.cuh`,
`common.cuh`, `ggml-common.h`, `vendors/cuda.h`. The `ggml.h`,
`ggml-impl.h` and `ggml-cuda.h` copies in `cuda/vendor/` are one-line
redirects to the local shim and contain no upstream code.

## Stubs

`cuda/vendor/q3_ggml_stubs.{h,cu}` are sera.c-local adapters (derived from
q38.c's `ds4_ggml_stubs.{h,cu}`). They provide the handful of macros and
type helpers the vendored kernels reference, so that no ggml runtime,
scheduler, allocator or graph engine is required.