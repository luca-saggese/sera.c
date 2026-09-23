# M4 E2E benchmark

commit: be9b1e5cb1788b610e4f3ed524c4a51173d08b81
model: models/Qwen3-32B-Q4_K_M.gguf
model bytes: 19,762,149,024
GPU: NVIDIA GB10 (sm_121a)

startup_ms: 47.6
resident_model_bytes: N/A
GPU memory after load: N/A

samples per Q: 10 (p95 == max)

| Q | p50 ms | p95 ms | q/s | ms/q |
|---|--------|--------|-----|------|
| 1 | 1565.5 | 1699.9 | 0.6 | 1587.26 |
| 2 | 1499.2 | 1706.1 | 1.3 | 772.75 |
| 4 | 1617.2 | 1765.0 | 2.5 | 406.64 |
| 8 | 1939.4 | 2015.0 | 4.1 | 242.23 |
| 16 | 2425.7 | 2524.0 | 6.6 | 151.51 |
| 32 | 3502.4 | 3677.8 | 9.1 | 110.01 |

server overhead: N/A (no --debug-stats in server)

runtime breakdown: N/A (no stats block)

GPU peak: N/A

correctness:
  smoke: PASS
  ladder accuracy: 260/630
  concurrency4: all200=True correct=False

main bottleneck: N/A (no per-phase telemetry; see M4 §46)
