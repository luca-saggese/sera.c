#!/usr/bin/env python3
"""M3 workload tokenizer.

Loads ONLY the Qwen3 tokenizer from the GGUF (metadata, no weights, no
dequantization) and produces the token IDs for the M3 decision workload:

    docs/research/m3/workload.json
        -> docs/research/m3/workload_tokens.json

The output file is consumed by the M3 CLI (--bench-decisions) and by the
M3 tests. This is NOT an oracle run: no model forward, no dequantization.
"""
import argparse
import json
import os
import sys


def load_tokenizer(gguf_path):
    from transformers import AutoTokenizer

    if os.path.isfile(gguf_path):
        directory = os.path.dirname(os.path.abspath(gguf_path))
        gguf_file = os.path.basename(gguf_path)
    else:
        directory = gguf_path
        gguf_file = "Qwen3-32B-Q4_K_M.gguf"
    try:
        return AutoTokenizer.from_pretrained(directory, gguf_file=gguf_file)
    except Exception:
        pass
    try:
        return AutoTokenizer.from_pretrained(gguf_path)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, help="Qwen3 GGUF (tokenizer source)")
    p.add_argument("--workload", default="docs/research/m3/workload.json")
    p.add_argument("--out", default="docs/research/m3/workload_tokens.json")
    args = p.parse_args()

    with open(args.workload, "r", encoding="utf-8") as fh:
        workload = json.load(fh)

    tok = load_tokenizer(args.gguf)
    if tok is None:
        print("error: could not load the Qwen3 tokenizer from the GGUF", file=sys.stderr)
        return 1

    def ids(text):
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"][0].tolist()

    state_ids = ids(workload["state_prompt"])
    questions = []
    for q in workload["questions"]:
        opts = "\n".join(
            f"{i+1}. {c}"
            for i, c in enumerate(q["candidates"])
        )

        suffix = (
            f"QUESTION:\n{q['suffix_prompt']}\n\n"
            f"OPTIONS:\n{opts}\n\n"
            f"ANSWER:"
        )

        suffix_ids = ids(suffix)

        cand_ids = []
        for i in range(len(q["candidates"])):
            x = ids(str(i + 1))
            assert len(x) == 1
            cand_ids.append(x[0])
        questions.append({
            "id": q["id"],
            "category": q["category"],
            "suffix_tokens": suffix_ids,
            "candidate_token_ids": cand_ids,
            "expected_index": q["expected_index"],
        })

    out = {
        "state_prompt": workload["state_prompt"],
        "state_tokens": state_ids,
        "questions": questions,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"state tokens: {len(state_ids)}")
    print(f"questions:    {len(questions)}")
    print(f"wrote:        {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())