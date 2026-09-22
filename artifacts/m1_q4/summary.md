# M1 - resident Q4_K quantized linear on GB10

| | |
|---|---|
| git commit | `16efa1f6ecfeb3073a8c05f6b5f2d0e1dbfd1dce` |
| model | `models/Qwen3-32B-Q4_K_M.gguf` |
| tensor | `blk.0.ffn_gate.weight` |
| qtype | `Q4_K` |
| N (output features) | 25600 |
| K (input features) | 5120 |
| physical bytes | 73728000 |
| workspace bytes | 5571072 |

## Correctness (vs. CPU Q4_K oracle)

| M | max abs | mean abs | RMSE | max rel | rel L2 | cosine |
|---|---|---|---|---|---|---|
| 1 | 0.01343 | 0.001883 | 0.002479 | 0.00411 | 0.00392 | 0.999992340 |
| 4 | 0.01847 | 0.001963 | 0.002577 | 0.00406 | 0.00414 | 0.999991431 |
| 16 | 0.0489 | 0.00468 | 0.006146 | 0.0107 | 0.00996 | 0.999950439 |

Activations are deterministic sin/cos. The GPU path quantizes activations to
Q8_1, so it is not bit-exact with the FP32 reference; the error above is the
real measured error. Repeated calls are bit-exact against each other.

## Device timings (median of 3 measured runs, 1 warmup)

| M | path | activation quant ms | kernel ms | total ms | tokens/s | effective weight GB/s |
|---|---|---|---|---|---|---|
| 1 | MMVQ | 0.0132 | 0.4921 | 0.5159 | 1938.3 | 142.91 |
| 4 | MMVQ | 0.0122 | 0.6693 | 0.6938 | 5765.4 | 106.27 |
| 16 | MMQ | 0.0286 | 0.5535 | 0.5990 | 26710.8 | 123.08 |
| 32 | MMQ | 0.0237 | 0.6991 | 0.7456 | 42920.3 | 98.89 |

Effective weight GB/s is the physical matrix bytes divided by the median total
device-op time. For M>1 the same matrix is read once and reused across tokens.

## Production invariants

| property | value |
|---|---|
| CUDA allocations during hot call | 0 |
| host syncs inside the primitive | 0 |
| kernel launches per call | 1 |
| persistent dequantized weight mirror | none |
| backend | MMVQ (M<=8) / MMQ (M>8) |

## Donor provenance

The primitive is a COPY -> RENAME -> EDIT port from `luca-saggese/q38.c @ main`
(commit `c1d4597a80e300b803dc642519718f2c999589da`), specifically `cuda/mmq/`.
q38.c in turn vendored the MMQ/MMVQ kernels from
`ggml-org/llama.cpp @ 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT).
See `THIRD_PARTY_NOTES.md`. No ggml/llama.cpp runtime is linked or loaded.
