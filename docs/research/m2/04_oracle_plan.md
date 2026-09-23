# M2 Oracle plan

## Guardrail: oracle MUST be Q4, not bf16

User directive: "tutti i confronti vanno fatti q4 non q8" — all comparisons
must use the Q4 GGUF weights, not bf16.

`tools/oracle_qwen3_forward.py` now loads the Q4 GGUF directly:

```python
if os.path.isfile(args.gguf):
    model = AutoModelForCausalLM.from_pretrained(
        args.gguf, dtype=torch.float32, device_map=args.device)
```

Transformers dequantizes the Q4 GGUF weights to fp32, so the reference
carries the SAME Q4 quantization error as the q3 runtime. This allows a
tight tolerance on boundary parity (embedding, norms, attention, MLP).

The bf16 directory (`models/qwen3_32b_bf16`) is only a fallback when no
GGUF is available. `metadata.json` records `weights_source: gguf_q4|bf16`.

## Run command

```bash
python3 tools/oracle_qwen3_forward.py \
    --model models/qwen3_32b_bf16 \
    --gguf models/Qwen3-32B-Q4_K_M.gguf \
    --dump artifacts/oracle_dump \
    --token-ids 785,8251,7578 \
    --candidates 16,17,18,19
```

## Output

- `tokens.json` — token IDs
- `embedding.bin`, `layer_00_*.bin`, `layer_01/31/63_output.bin`,
  `final_norm.bin` — fp32 dumps
- `candidate_logits.json` — candidate logits + argmax
- `metadata.json` — config + weights_source
