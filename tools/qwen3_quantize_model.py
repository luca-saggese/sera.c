#!/usr/bin/env python3
"""Full-model FP8 -> Q4_K quantization (Torch/CUDA path).

Writes the exact Q4_K bytes that BOTH the Python oracle and (later) the GGUF
will consume.  Quantization happens exactly once.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_q4 as ref
import qwen3_q4_torch as q4t


def build_index(ckpt):
    idx = {}
    for shard in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with open(shard, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
        for k, v in hdr.items():
            if k != "__metadata__":
                idx[k] = (shard, v)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--chunk-rows", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    idx = build_index(args.checkpoint)

    weights = sorted(k for k in idx
                     if idx[k][1]["dtype"] == "F8_E4M3")
    print(f"[model] {len(weights)} FP8 weight tensors", flush=True)

    manifest = {}
    t_start = time.time()
    tot_bytes = 0
    for n, name in enumerate(weights, 1):
        shard, info = idx[name]
        rows, cols = info["shape"]
        si_name = name + "_scale_inv"
        if si_name not in idx:
            print(f"[skip] {name}: no scale_inv", flush=True)
            continue
        fp8_bytes, _, _ = ref.read_safetensors_raw(shard, name)
        si_bytes, _, _ = ref.read_safetensors_raw(shard, si_name)

        t0 = time.time()
        w_f32 = ref.fp8_weight_to_f32(fp8_bytes, si_bytes, rows, cols)
        t_fp8 = time.time() - t0

        t0 = time.time()
        q4 = q4t.quantize_q4_k_matrix_torch(w_f32, device="cuda",
                                            chunk_rows=args.chunk_rows)
        t_q4 = time.time() - t0
        del w_f32

        out_name = name.replace("/", "__") + ".q4k"
        out_path = os.path.join(args.outdir, out_name)
        with open(out_path, "wb") as fh:
            fh.write(q4)
        tot_bytes += len(q4)

        manifest[name] = {
            "file": out_name,
            "shape": [rows, cols],
            "dtype": "Q4_K",
            "block_bytes": ref.Q4_K_BLOCK_BYTES,
            "bytes": len(q4),
            "blocks_per_row": cols // ref.QK_K,
        }
        el = time.time() - t_start
        print(f"[{n:3d}/{len(weights)}] {name} {rows}x{cols} "
              f"fp8={t_fp8:.2f}s q4={t_q4:.2f}s bytes={len(q4)} "
              f"elapsed={el/60:.1f}min", flush=True)

    with open(os.path.join(args.outdir, "q4_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] {len(manifest)} tensors, {tot_bytes/1e9:.2f} GB, "
          f"{(time.time()-t_start)/60:.1f} min -> {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
