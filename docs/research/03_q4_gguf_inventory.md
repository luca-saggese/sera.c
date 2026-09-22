# Research C — Real Q4 GGUF Inventory (Qwen3-32B-Q4_K_M)

Status: **COMPLETE** (download finished, full parse succeeded).
Method: standalone `mmap`-only GGUF v3 parser (`inspect_gguf.py`), no payload/dequant reads; independently
cross-validated against donor `_reference/q38.c/q38_gguf.c` (identical header, counts, offsets, byte sums).

## 1. Container

| Field | Value |
|---|---|
| path | `/mnt/seagate/models/gguf/Qwen3-32B-Q4_K_M.gguf` |
| file size (exact bytes) | `19762149024` |
| GGUF version | `3` |
| `general.architecture` | `qwen3` |
| alignment (`general.alignment`) | `32` |
| metadata (KV) count | `28` |
| tensor count | `707` |
| metadata table ends at | `5932883` |
| tensor-info table ends at | `5974661` |
| `tensor_data_pos` (aligned) | `5974688` |
| `general.file_type` | `15` (Q4_K_M) |
| `general.quantization_version` | `2` |

`tensor_data_pos = align_up(5974661, 32) = 5974688`. (Note: it is aligned after the *tensor-info*
table, not after metadata — metadata alone ends at 5932883, which is **not** the data start.)

## 2. Key metadata

| Key | Value |
|---|---|
| `general.name` | `Qwen3 32B Awq Compatible Instruct` |
| `qwen3.context_length` | `40960` |
| `qwen3.block_count` | `64` |
| `qwen3.embedding_length` | `5120` |
| `qwen3.feed_forward_length` | `25600` |
| `qwen3.attention.head_count` | `64` |
| `qwen3.attention.head_count_kv` | `8` |
| `qwen3.attention.key_length` / `value_length` | `128` / `128` |
| `qwen3.rope.freq_base` | `1000000.0` |
| `qwen3.attention.layer_norm_rms_epsilon` | `9.999999974752427e-07` |

Tokenizer keys present: `tokenizer.ggml.model=gpt2`, `pre`, `tokens`, `merges`, `token_type`,
`bos_token_id=151643`, `eos_token_id=151645`, `padding_token_id=151643`, `chat_template`.

## 3. Quant types actually present

**Not** all-Q4_K. Three distinct types, all present in the donor `q38_gguf.c` table:

| id | name | block elems | block bytes | tensors | element count | total bytes |
|---|---|---|---|---|---|---|
| 0 | `f32` | 1 | 4 | 257 | 676864 | 2707456 |
| 12 | `q4_k` | 256 | 144 | 385 | 27621457920 | 15537070080 |
| 14 | `q6_k` | 256 | 210 | 65 | 5139988480 | 4216396800 |

- **Unknown type ids: none.** `unknown ids: []`.
- **Every** qtype present (`f32`, `q4_k`, `q6_k`) is in the donor type table → donor parser supports
  all of them (`q38_gguf_type_nbytes` returns valid bytes for all 707 tensors;
  donor `unsupported=0`).
- Total tensor bytes: `19756174336` = sum of the three rows above.

## 4. Metadata count / tensor naming — complete grouped summary

Every distinct layer-suffix pattern (14 patterns, each ×64 = 704 block tensors) plus 3 non-layer tensors:

| Tensor pattern | count | shape (out×in, reversed) | qtype | elems | bytes each |
|---|---|---|---|---|---|
| `blk.N.attn_norm.weight` | 64 | 5120 | f32 | 5120 | 20480 |
| `blk.N.attn_q.weight` | 64 | 8192×5120 | q4_k | 41943040 | 23592960 |
| `blk.N.attn_k.weight` | 64 | 1024×5120 | q4_k | 5242880 | 2949120 |
| `blk.N.attn_v.weight` | 64 | 1024×5120 | **mixed q4_k/q6_k** | 5242880 | q4_k 2949120 / q6_k 4300800 |
| `blk.N.attn_output.weight` | 64 | 5120×8192 | q4_k | 41943040 | 23592960 |
| `blk.N.attn_q_norm.weight` | 64 | 128 | f32 | 128 | 512 |
| `blk.N.attn_k_norm.weight` | 64 | 128 | f32 | 128 | 512 |
| `blk.N.ffn_norm.weight` | 64 | 5120 | f32 | 5120 | 20480 |
| `blk.N.ffn_gate.weight` | 64 | 25600×5120 | q4_k | 131072000 | 73728000 |
| `blk.N.ffn_up.weight` | 64 | 25600×5120 | q4_k | 131072000 | 73728000 |
| `blk.N.ffn_down.weight` | 64 | 5120×25600 | **mixed q4_k/q6_k** | 131072000 | q4_k 73728000 / q6_k 107520000 |
| `token_embd.weight` | 1 | 151936×5120 | q4_k | 777912320 | 437575680 |
| `output_norm.weight` | 1 | 5120 | f32 | 5120 | 20480 |
| `output.weight` | 1 | 151936×5120 | q6_k | 777912320 | 638131200 |

### Mixed-type patterns (important for the loader)

Both `attn_v.weight` and `ffn_down.weight` are **q6_k in 32 layers and q4_k in the other 32**:

| qtype | layers |
|---|---|
| `q6_k` | 0–7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 56–63 |
| `q4_k` | 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54 |

The `q6_k` count of 65 = 32 `attn_v` + 32 `ffn_down` + 1 `output.weight`. All other block tensors are
uniform q4_k (weights) or f32 (norms).

## 5. Largest tensors (top 10 by bytes)

| # | name | shape | qtype | offset (abs) | bytes |
|---|---|---|---|---|---|
| 1 | `output.weight` | 151936×5120 | q6_k | 5974688 | 638131200 |
| 2 | `token_embd.weight` | 151936×5120 | q4_k | 644126368 | 437575680 |
| 3 | `blk.0.ffn_down.weight` | 5120×25600 | q6_k | 1136159392 | 107520000 |
| 4 | `blk.1.ffn_down.weight` | 5120×25600 | q6_k | 1445613216 | 107520000 |
| 5 | `blk.2.ffn_down.weight` | 5120×25600 | q6_k | 1755067040 | 107520000 |
| 6 | `blk.3.ffn_down.weight` | 5120×25600 | q6_k | 2064520864 | 107520000 |
| 7 | `blk.4.ffn_down.weight` | 5120×25600 | q6_k | 2373974688 | 107520000 |
| 8 | `blk.5.ffn_down.weight` | 5120×25600 | q6_k | 2683428512 | 107520000 |
| 9 | `blk.6.ffn_down.weight` | 5120×25600 | q6_k | 2992882336 | 107520000 |
| 10 | `blk.7.ffn_down.weight` | 5120×25600 | q6_k | 3302336160 | 107520000 |

Every q6_k `ffn_down` is 107520000 B; the q4_k ones are 73728000 B. Largest tensor overall is
`output.weight` (`638131200` B), matching the donor's `max_tensor=638131200`.

## 6. Embedding / output (lm_head)

| tensor | name | shape | qtype | bytes | abs offset |
|---|---|---|---|---|---|
| token embedding | `token_embd.weight` | 151936×5120 | q4_k | 437575680 | 644126368 |
| output (lm_head) | `output.weight` | 151936×5120 | q6_k | 638131200 | 5974688 |

**Not tied.** A separate `output.weight` tensor is physically present with its own storage; it also uses
a *different* qtype (q6_k) and different bytes than `token_embd.weight` (q4_k), so the two cannot share
one buffer. Loader must treat embedding and output as distinct tensors.

## 7. Offsets and bounds

| Quantity | Value |
|---|---|
| `tensor_data_pos` | `5974688` |
| min absolute tensor offset | `5974688` (`output.weight`) |
| max absolute tensor offset | `19688421024` |
| max (offset + bytes) | `19762149024` |
| file size | `19762149024` |
| max offset + bytes ≤ file size | **true** (exactly equal; 4-byte remainder unused) |
| gaps between consecutive tensors (sorted by offset) | **0** — tensors are fully contiguous |

Block extents: block 0 spans `1081702048 .. 1391155872`; block 63 spans `19452695200 .. 19762149024`.

## 8. Block-geometry check (computed vs expected)

Using donor block geometry `bytes = ceil(elems / block_elems) * block_bytes`:

| tensor | elems | qtype (block elems/bytes) | arithmetic | computed | expected/stored | match |
|---|---|---|---|---|---|---|
| `token_embd.weight` | 777912320 | q4_k (256/144) | (777912320/256)=3038720 × 144 | 437575680 | 437575680 | ✅ |
| `output.weight` | 777912320 | q6_k (256/210) | (777912320/256)=3038720 × 210 | 638131200 | 638131200 | ✅ |
| `blk.0.attn_q.weight` | 41943040 | q4_k (256/144) | (41943040/256)=163840 × 144 | 23592960 | 23592960 | ✅ |
| `blk.0.attn_v.weight` (q6_k) | 5242880 | q6_k (256/210) | (5242880/256)=20480 × 210 | 4300800 | 4300800 | ✅ |
| `blk.8.attn_v.weight` (q4_k) | 5242880 | q4_k (256/144) | (5242880/256)=20480 × 144 | 2949120 | 2949120 | ✅ |
| `blk.0.ffn_down.weight` (q6_k) | 131072000 | q6_k (256/210) | (131072000/256)=512000 × 210 | 107520000 | 107520000 | ✅ |
| `blk.8.ffn_down.weight` (q4_k) | 131072000 | q4_k (256/144) | (131072000/256)=512000 × 144 | 73728000 | 73728000 | ✅ |
| `blk.0.attn_norm.weight` | 5120 | f32 (1/4) | 5120 × 4 | 20480 | 20480 | ✅ |

All shapes divide exactly by block size (no ceil rounding actually needed for this file). Sum of all
computed bytes = `19756174336`, and every tensor passes the donor bounds check
(`abs_offset + bytes <= size`).

## 9. Conclusions for M0 loader

1. Container is **GGUF v3**, 707 tensors, 28 KV, alignment 32, `tensor_data_pos=5974688`.
2. Only three qtypes: **f32 (257), q4_k (385), q6_k (65)** — all supported by the donor type table;
   no unknown/unsupported type ids.
3. `attn_v` and `ffn_down` are **mixed q4_k/q6_k across layers** — a per-tensor qtype dispatch is
   required; do not hardcode "Q4_K_M ⇒ all q4_k".
4. `output.weight` is **q6_k** and the largest single tensor; embeddings are **not tied**.
5. Layout is contiguous with zero gaps; total payload `19756174336` B fits the file exactly.

Scratch parser (`mmap`-only, no payload reads) and its full JSON dump were kept out of the repo
(session workspace only); all figures above were produced by it, then independently re-verified
against the donor C parser `q38_gguf.c` (identical version/counts/offsets/byte sums,
`unsupported=0`, `fit=1`).