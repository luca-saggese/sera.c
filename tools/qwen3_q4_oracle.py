#!/usr/bin/env python3
"""M5 Python oracle: run Qwen3-32B forward using the Q4_K blocks we produced.

Semantics are taken unchanged from test_qwen_4.py (same TESTS, same
make_prompt, same Decider, same candidate-token selection).  The only
difference is where the weights come from:

    FP8 checkpoint -> (once, tools/qwen3_quantize_model.py) -> Q4_K bytes
    Q4_K bytes -> dequant -> bf16 torch weight -> Qwen3ForCausalLM

No second quantization: the bytes read here are exactly the bytes written to
artifacts/m5_parity/q4/.
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import qwen3_q4_dequant_torch as dq                      # noqa: E402
from test_qwen_4 import (                                # noqa: E402
    TESTS, Decider, make_prompt, permute_options,
)


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


def load_bf16_raw(path, name):
    with open(path, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
        info = hdr[name]
        start, end = info["data_offsets"]
        fh.seek(8 + hl + start)
        raw = fh.read(end - start)
    if info["dtype"] == "BF16":
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        return t.reshape(info["shape"]).clone()
    raise ValueError(f"{name}: unsupported dtype {info['dtype']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(REPO, "artifacts/m5_parity/hf_checkpoint"))
    p.add_argument("--q4dir", default=os.path.join(REPO, "artifacts/m5_parity/q4"))
    p.add_argument("--out", default=os.path.join(REPO, "artifacts/m5_parity/golden_logits.json"))
    p.add_argument("--modes", default="compact")
    p.add_argument("--variants", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    t_load0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()

    idx = build_index(args.checkpoint)
    dev = torch.device(args.device)

    sd = {}
    t_w0 = time.time()
    keys = list(model.state_dict().keys())
    n_q4 = n_raw = 0
    for i, name in enumerate(keys, 1):
        if name not in idx:
            raise KeyError(f"{name} not in checkpoint index")
        shard, info = idx[name]
        # Policy: rank 2 -> Q4_K, rank 1 -> raw.  Every rank-2 weight (including
        # token_embd / output) is quantized exactly once and consumed from the
        # same artifacts the GGUF embeds, so oracle and native share one payload.
        if len(info["shape"]) == 2:
            rows, cols = info["shape"]
            q4_path = os.path.join(args.q4dir, name + ".q4k")
            with open(q4_path, "rb") as fh:
                raw = fh.read()
            w = dq.dequant_q4_k_matrix_torch(raw, rows, cols, device=args.device)
            sd[name] = w.to(torch.bfloat16).contiguous()
            n_q4 += 1
        else:
            sd[name] = load_bf16_raw(shard, name).to(dev)
            n_raw += 1
        if i % 100 == 0:
            print(f"  loaded {i}/{len(keys)}  ({time.time()-t_w0:.0f}s)", flush=True)

    model.load_state_dict(sd, assign=True)
    del sd
    torch.cuda.empty_cache()
    t_load1 = time.time()
    print(f"[load] {n_q4} Q4_K tensors + {n_raw} raw, {t_load1-t_load0:.1f}s", flush=True)
    print(f"[load] gpu allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    dec = Decider(model, tok)

    tests = TESTS
    if args.limit:
        tests = tests[:args.limit]

    results = []
    t0 = time.time()
    for ci, case in enumerate(tests, 1):
        for mode in modes:
            for variant in range(args.variants):
                opts = permute_options(case, variant, args.seed)
                z = dec.run(case, opts, mode)

                # exact prompt/token ids (deterministic, same call as Decider)
                prompt = make_prompt(tok, case, opts, mode)
                enc = tok(prompt, return_tensors="pt")
                token_ids = enc["input_ids"][0].tolist()

                label_ids = [tok.encode(str(i), add_special_tokens=False)[0]
                             for i in range(1, len(opts) + 1)]

                with torch.inference_mode():
                    out = model(**{k: v.to(dev) for k, v in enc.items()},
                                use_cache=False, logits_to_keep=1)
                    all_logits = out.logits[:, -1, :].float()[0]
                    cids = torch.tensor(label_ids, dtype=torch.long, device=dev)
                    clogits = all_logits.index_select(0, cids)
                    cprobs = torch.softmax(clogits, -1)
                    order = torch.argsort(clogits, descending=True)

                results.append({
                    "case": case.id,
                    "pair": case.pair,
                    "lang": case.lang,
                    "mode": mode,
                    "variant": variant,
                    "prompt": prompt,
                    "token_ids": token_ids,
                    "options": list(opts),
                    "expected": case.expected,
                    "candidate_token_ids": label_ids,
                    "candidate_logits": [float(v) for v in clogits.tolist()],
                    "candidate_probs": [float(v) for v in cprobs.tolist()],
                    "argmax_index": int(order[0]),
                    "argmax_option": opts[int(order[0])],
                    "predicted": z["predicted"],
                    "correct": z["predicted"] == case.expected,
                })
        print(f"[{ci:2d}/{len(tests)}] {case.id} done ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "meta": {
            "model": "Qwen/Qwen3-32B-Fp8",
            "checkpoint_dir": args.checkpoint,
            "q4dir": args.q4dir,
            "source": "Q4_K blocks produced by tools/qwen3_quantize_model.py",
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "attention": "sdpa",
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "seed": args.seed,
            "modes": modes,
            "variants": args.variants,
            "load_s": t_load1 - t_load0,
        },
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[done] {len(results)} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
