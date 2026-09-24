#!/usr/bin/env python3
"""M5: quantize the BF16 rank-2 tensors the FP8 pass could not cover.

``tools/qwen3_quantize_model.py`` only walks F8_E4M3 weights, so the two BF16
rank-2 tensors (token_embd / output) were never turned into Q4_K.  This tool
runs the *same* Torch/CUDA quantizer on them and appends the results to the
existing manifest, so the model ends up all-Q4_K for every rank-2 weight.

No requantization: tensors already present in the manifest are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_q4 as ref
import qwen3_q4_torch as q4t
from qwen3_quantize_model import build_index


def bf16_to_f32(raw, rows, cols):
    a = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return a.view(np.float32).reshape(rows, cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--chunk-rows", type=int, default=256)
    args = p.parse_args()

    idx = build_index(args.checkpoint)
    mpath = os.path.join(args.outdir, "q4_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}

    targets = []
    for name in sorted(idx):
        shard, info = idx[name]
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            continue
        if name in manifest:
            continue
        if name.endswith("weight_scale_inv"):
            continue
        targets.append((name, shard, info))

    print(f"[extra] {len(targets)} BF16 rank-2 tensors to quantize", flush=True)

    tot = 0
    for n, (name, shard, info) in enumerate(targets, 1):
        rows, cols = info["shape"]
        if cols % ref.QK_K:
            raise SystemExit(f"{name}: cols {cols} not multiple of {ref.QK_K}")

        t0 = time.time()
        raw, _, _ = ref.read_safetensors_raw(shard, name)
        w_f32 = bf16_to_f32(raw, rows, cols)
        del raw
        t_bf16 = time.time() - t0

        t0 = time.time()
        q4 = q4t.quantize_q4_k_matrix_torch(w_f32, device="cuda",
                                            chunk_rows=args.chunk_rows)
        t_q4 = time.time() - t0
        del w_f32

        out_name = name.replace("/", "__") + ".q4k"
        with open(os.path.join(args.outdir, out_name), "wb") as fh:
            fh.write(q4)
        tot += len(q4)

        manifest[name] = {
            "file": out_name,
            "shape": [rows, cols],
            "dtype": "Q4_K",
            "block_bytes": ref.Q4_K_BLOCK_BYTES,
            "bytes": len(q4),
            "blocks_per_row": cols // ref.QK_K,
        }
        print(f"[{n}/{len(targets)}] {name} {rows}x{cols} bf16={t_bf16:.2f}s "
              f"q4={t_q4:.2f}s bytes={len(q4)}", flush=True)

    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[extra] manifest now {len(manifest)} tensors, +{tot} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
