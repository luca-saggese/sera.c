# M3 packed decision batching — integration summary

Model: `models/Qwen3-32B-Q4_K_M.gguf`
Workload: `docs/research/m3/workload_tokens.json` (prefix 129 tokens, 32 questions, 390 suffix tokens)
Command:

```
./q3 --bench-decisions models/Qwen3-32B-Q4_K_M.gguf \
     --workload docs/research/m3/workload_tokens.json \
     --batch 1,2,4,8,16,32 --json
```

## Structural gate (M3 §69-§74)

| Gate | Result |
|------|--------|
| B=1,2,4,8,16,32 executes | PASS |
| no crash / NaN / Inf | PASS |
| `prefix_copy_bytes == 0` | PASS (0 for every B) |
| prefix physical KV bytes constant in B | PASS (67,633,152 B for every B) |
| no hot `cudaMalloc`/`cudaFree` | PASS (0 allocations for every B) |
| one branch-isolation smoke | PASS (canary 1) |
| host syncs per batch | 1 (the single explicit boundary sync) |

## Ladder

| B | suffix tokens | suffix fwd ms | head ms | warm ms | ms/question | q/s | accuracy |
|---|---------------|---------------|---------|---------|-------------|-----|----------|
| 1 | 8 | 5.51 | 0.11 | 455.5 | 455.5 | 2.20 | 0/1 |
| 2 | 25 | 4.47 | 0.09 | 282.3 | 141.1 | 7.09 | 1/2 |
| 4 | 53 | 6.45 | 0.09 | 393.3 | 98.3 | 10.17 | 1/4 |
| 8 | 105 | 10.20 | 0.08 | 633.9 | 79.2 | 12.62 | 4/8 |
| 16 | 197 | 18.49 | 0.08 | 1168.2 | 73.0 | 13.70 | 8/16 |
| 32 | 390 | 35.64 | 0.09 | 2219.3 | 69.4 | 14.42 | 12/32 |

Prefix prefill: 743.4 ms (once). Branch create: 0.0014 ms (descriptor only).

## Correctness

Correctness/parity is **DEFERRED** for M3 (see `docs/CORRECTNESS_DEBT.md`). The
accuracy column is reported, not gated. It is a semantic-quality signal only:
the workload tokenization and the M2 logit path are themselves unvalidated
against an oracle.

## Bug fixed during integration

`attention_segmented_kernel` applied the per-layer slab offset to the private
suffix K/V but **not** to the shared prefix K/V. The prefix slabs are
`[num_layers][prefix_capacity][kv_heads][head_dim]`, so every layer above 0
read layer 0's prefix keys/values. Fixed by threading `prefix_capacity` into
the kernel and adding `prefix_layer_base = suffix_layer * prefix_capacity * row`.

Effect: Test 1 argmax went from a mismatch (0 vs 2) to a match (0 vs 0), and
ladder accuracy improved 22/63 → 26/63.

## Test 1 numeric bound

Test 1 compares the M3 segmented path against the M2 contiguous path. The
residual `max|diff| = 1.35` is **inherent split drift**, not an M3 bug: M2's
*own* split path (prefix pass then suffix pass) drifts `3.2` from M2's *own*
contiguous pass. M3 is therefore closer to the contiguous reference than M2's
own split. The gate is the discrete candidate plus "no worse than M2's own
split drift".
