#!/usr/bin/env python3
"""M2 oracle forward pass for the sera.c project.

Runs the reference (HF bf16) Qwen3-32B model for a single forward pass and dumps
every intermediate activation sera.c's M2 backend needs to compare against.

Usage:
  python tools/oracle_qwen3_forward.py --model <hf_dir> --gguf <gguf_path> \
      --prompt-file <file> --dump <outdir> [--candidates 16,17,18,19] \
      [--token-ids 1,2,3]

All dumped .bin files are little-endian float32, row-major; shapes are recorded
in metadata.json.  The batch dimension is squeezed out (T == 1 sequence).
"""

import argparse
import json
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tensor capture helpers
# ---------------------------------------------------------------------------

def _unwrap(output):
    """Forward hooks may return tuples/lists; take the first tensor."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _to_numpy(tensor):
    """Cast any captured activation to a contiguous float32 numpy array."""
    if tensor is None or not torch.is_tensor(tensor):
        return None
    return tensor.detach().float().cpu().numpy()


def _squeeze_batch(arr):
    """Drop a leading batch dimension of 1 (we only ever run T=1)."""
    if arr is not None and arr.ndim >= 2 and arr.shape[0] == 1:
        return arr.reshape(arr.shape[1:])
    return arr


class Capture:
    """Ordered registry of captured tensors -> dumped as <name>.bin."""

    def __init__(self):
        self.tensors = OrderedDict()   # name (no extension) -> np.ndarray

    def add(self, name, tensor, squeeze=True):
        arr = _to_numpy(tensor)
        if arr is None:
            raise RuntimeError(f"capture '{name}' was None / not a tensor")
        if squeeze:
            arr = _squeeze_batch(arr)
        self.tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)
        return self.tensors[name]

    def add_raw(self, name, arr, squeeze=False):
        arr = np.asarray(arr)
        if squeeze:
            arr = _squeeze_batch(arr)
        self.tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)
        return self.tensors[name]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dump Qwen3-32B reference activations for M2")
    p.add_argument("--model", required=True, help="HF model directory (bf16)")
    p.add_argument("--gguf", required=True, help="GGUF file used for the tokenizer")
    p.add_argument("--prompt-file", default=None, help="file containing the prompt text")
    p.add_argument("--dump", required=True, help="output directory for the dump")
    p.add_argument("--candidates", default=None,
                   help="comma-separated candidate token ids to report logits for")
    p.add_argument("--token-ids", default=None,
                   help="comma-separated token ids; bypasses tokenization entirely")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def parse_id_list(text):
    if text is None:
        return None
    text = text.strip()
    if not text:
        return []
    return [int(tok) for tok in text.split(",") if tok.strip() != ""]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

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
        # Fall back to treating --gguf as a direct tokenizer source.
        return AutoTokenizer.from_pretrained(gguf_path)
    except Exception:
        # The tokenizer is only used for human-readable decode() strings; the
        # numeric token IDs and logits are the source of truth for parity.
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    from transformers import AutoModelForCausalLM
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    outdir = os.path.abspath(args.dump)
    os.makedirs(outdir, exist_ok=True)

    # ---- prompt / token ids -------------------------------------------------
    token_ids = parse_id_list(args.token_ids)
    prompt = None
    if token_ids is None:
        if args.prompt_file is not None:
            with open(args.prompt_file, "r", encoding="utf-8") as fh:
                prompt = fh.read().rstrip("\n")
        else:
            prompt = "The cat sat"
        print(f"[oracle] prompt = {prompt!r}")
        tokenizer = load_tokenizer(args.gguf)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        token_ids = enc["input_ids"][0].tolist()
        print(f"[oracle] token ids = {token_ids}")
    else:
        print(f"[oracle] --token-ids given, bypassing tokenization: {token_ids}")
        tokenizer = load_tokenizer(args.gguf)

    T = len(token_ids)
    if T == 0:
        raise RuntimeError("empty prompt / token id list")
    print(f"[oracle] sequence length T = {T}")

    with open(os.path.join(outdir, "tokens.json"), "w", encoding="utf-8") as fh:
        json.dump({"prompt": prompt, "token_ids": token_ids, "count": T}, fh, indent=2)

    # ---- model --------------------------------------------------------------
    # Load the Q4 GGUF weights directly (transformers dequantizes them to
    # fp32). The reference then carries the SAME Q4 quantization error as the
    # q3 runtime, so boundary parity can use a tight tolerance. The bf16
    # directory is only used as a fallback when no GGUF is available.
    if os.path.isfile(args.gguf):
        # transformers >= 5.x requires a directory + gguf_file= kwarg, not a
        # raw .gguf path, to load GGUF weights.
        directory = os.path.dirname(os.path.abspath(args.gguf))
        gguf_file = os.path.basename(args.gguf)
        # Dequantize to bf16, not fp32: the Q4-dequantized values are the
        # same numbers (the Q4 quantization error is carried either way), but
        # fp32 for 32B params is ~130GB and OOMs this 121GB host. bf16 halves
        # that to ~65GB; the Q4 weight error still dominates bf16 rounding by
        # orders of magnitude, so parity tolerance stays tight.
        print(f"[oracle] loading Q4 GGUF weights from {args.gguf} (bf16 storage) ...")
        model = AutoModelForCausalLM.from_pretrained(
            directory, gguf_file=gguf_file, dtype=torch.bfloat16, device_map=args.device
        )
    else:
        print(f"[oracle] loading model from {args.model} (bf16, {args.device}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=args.device
        )
    model.eval()
    cfg = model.config
    print("[oracle] model loaded")

    layers = model.model.layers
    attn0 = layers[0].self_attn
    mlp0 = layers[0].mlp

    cap = Capture()

    # ---- post-RoPE Q/K: monkey-patch the module-level function --------------
    # apply_rotary_pos_emb is not a module boundary, so the only faithful way to
    # observe its outputs is to wrap the function in modeling_qwen3's globals
    # (Qwen3Attention.forward resolves it as a module global at call time).
    # Qwen3 uses NeoX half-split RoPE: rotate_half over all head_dim dims.
    orig_apply_rope = modeling_qwen3.apply_rotary_pos_emb
    rope_calls = []

    def patched_apply_rope(q, k, cos, sin, unsqueeze_dim=1):
        q_out, k_out = orig_apply_rope(q, k, cos, sin, unsqueeze_dim)
        rope_calls.append((q_out.detach(), k_out.detach()))
        return q_out, k_out

    modeling_qwen3.apply_rotary_pos_emb = patched_apply_rope

    def hook(name, squeeze=True):
        def fn(module, inputs, output):
            cap.add(name, _unwrap(output), squeeze=squeeze)
        return fn

    handles = [
        model.model.embed_tokens.register_forward_hook(hook("embedding")),
        layers[0].input_layernorm.register_forward_hook(hook("layer_00_input_norm")),
        attn0.q_proj.register_forward_hook(hook("layer_00_q")),
        attn0.k_proj.register_forward_hook(hook("layer_00_k")),
        attn0.v_proj.register_forward_hook(hook("layer_00_v")),
        # q_norm/k_norm are invoked on the *reshaped* q/k ([T, heads, head_dim]),
        # so the raw module output already has the shape the M2 spec wants.
        attn0.q_norm.register_forward_hook(hook("layer_00_q_norm")),
        attn0.k_norm.register_forward_hook(hook("layer_00_k_norm")),
        attn0.o_proj.register_forward_hook(hook("layer_00_o_proj")),
        attn0.register_forward_hook(hook("layer_00_attn_out")),
        layers[0].post_attention_layernorm.register_forward_hook(hook("layer_00_mlp_norm")),
        mlp0.gate_proj.register_forward_hook(hook("layer_00_gate")),
        mlp0.up_proj.register_forward_hook(hook("layer_00_up")),
        mlp0.down_proj.register_forward_hook(hook("layer_00_down")),
        layers[0].register_forward_hook(hook("layer_00_output")),
        layers[1].register_forward_hook(hook("layer_01_output")),
        layers[31].register_forward_hook(hook("layer_31_output")),
        layers[63].register_forward_hook(hook("layer_63_output")),
        model.model.norm.register_forward_hook(hook("final_norm")),
    ]

    logits_holder = {}

    def lm_head_hook(module, inputs, output):
        logits_holder["logits"] = _unwrap(output)

    handles.append(model.lm_head.register_forward_hook(lm_head_hook))

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)

    print("[oracle] running forward pass ...")
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()
        modeling_qwen3.apply_rotary_pos_emb = orig_apply_rope

    # ---- post-RoPE tensors --------------------------------------------------
    if len(rope_calls) != len(layers):
        raise RuntimeError(
            f"apply_rotary_pos_emb monkey-patch captured {len(rope_calls)} calls, "
            f"expected {len(layers)}; RoPE capture is not reliable"
        )
    q0, k0 = rope_calls[0]
    if q0 is None or k0 is None:
        raise RuntimeError("post-RoPE Q/K capture returned None")
    # [B, heads, T, head_dim] -> [T, heads, head_dim] to match q_norm/k_norm dumps.
    q0 = _squeeze_batch(q0.permute(0, 2, 1, 3).contiguous().float().cpu().numpy())
    k0 = _squeeze_batch(k0.permute(0, 2, 1, 3).contiguous().float().cpu().numpy())
    cap.add_raw("layer_00_q_rope", q0)
    cap.add_raw("layer_00_k_rope", k0)
    if cap.tensors["layer_00_q_rope"].ndim != 3:
        raise RuntimeError("unexpected post-RoPE Q rank")
    rope_calls.clear()

    # ---- derived tensors ----------------------------------------------------
    # attention residual: input to self_attn + self_attn output
    cap.add_raw(
        "layer_00_attn_resid",
        cap.tensors["layer_00_input_norm"] + cap.tensors["layer_00_attn_out"],
    )
    # silu(gate) * up
    gate = torch.from_numpy(cap.tensors["layer_00_gate"])
    up = torch.from_numpy(cap.tensors["layer_00_up"])
    cap.add_raw("layer_00_silu_up", (torch.nn.functional.silu(gate) * up).numpy())

    # sanity: layer 0 output == attn_resid + down_proj
    cap.add_raw(
        "layer_00_output",
        cap.tensors["layer_00_attn_resid"] + cap.tensors["layer_00_down"],
    )

    # ---- write .bin files ---------------------------------------------------
    written = OrderedDict()
    for name, arr in cap.tensors.items():
        fname = f"{name}.bin"
        arr.tofile(os.path.join(outdir, fname))
        written[name] = {"file": fname, "shape": list(arr.shape), "dtype": "float32"}
    print(f"[oracle] wrote {len(written)} tensor files")

    # ---- candidate logits ---------------------------------------------------
    logits = logits_holder["logits"]
    last = logits[0, -1].float().cpu().numpy()          # [vocab]
    full = logits[0].float().cpu().numpy()              # [T, vocab]
    argmax_id = int(np.argmax(last))
    topk_idx = np.argsort(-last)[:10]

    candidates = parse_id_list(args.candidates) or []
    cand_logits = {int(i): float(last[int(i)]) for i in candidates}
    cand_texts = {
        int(i): tokenizer.decode([int(i)]) if tokenizer is not None else ""
        for i in candidates
    }

    cand_payload = {
        "candidate_token_ids": [int(i) for i in candidates],
        "logits": {str(k): v for k, v in cand_logits.items()},
        "candidate_texts": {str(k): v for k, v in cand_texts.items()},
        "argmax": argmax_id,
        "argmax_text": tokenizer.decode([argmax_id]) if tokenizer is not None else "",
        "last_position_logits_topk": [
            {
                "token_id": int(i),
                "logit": float(last[int(i)]),
                "text": tokenizer.decode([int(i)]) if tokenizer is not None else "",
            }
            for i in topk_idx
        ],
        "logits_shape": list(full.shape),
    }
    with open(os.path.join(outdir, "candidate_logits.json"), "w", encoding="utf-8") as fh:
        json.dump(cand_payload, fh, indent=2)

    # ---- metadata -----------------------------------------------------------
    metadata = {
        "model_dir": os.path.abspath(args.model),
        "gguf": os.path.abspath(args.gguf),
        "weights_source": "gguf_q4" if os.path.isfile(args.gguf) else "bf16",
        "device": args.device,
        "torch_version": torch.__version__,
        "sequence_length": T,
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": float(getattr(cfg, "rope_theta", 0.0)),
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        },
        "tensors": written,
        "notes": {
            "layout": "little-endian float32, row-major, batch dimension squeezed (T == 1)",
            "layer_00_q_rope": "post-RoPE Q, [T, 64, 128] (NeoX half-split rotate_half over head_dim)",
            "layer_00_k_rope": "post-RoPE K, [T, 8, 128]",
            "layer_00_attn_out": "self_attn module output (equals o_proj output in Qwen3)",
            "layer_00_attn_resid": "input_layernorm output + self_attn output",
            "layer_00_silu_up": "silu(gate_proj output) * up_proj output",
            "layer_00_output": "layer 0 decoder output (attn_resid + down_proj output)",
            "rope_capture": "apply_rotary_pos_emb in modeling_qwen3 was monkey-patched during the forward pass",
        },
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # ---- report -------------------------------------------------------------
    print("[oracle] dumped tensors:")
    for name, info in written.items():
        print(f"    {info['file']:<32} {info['shape']}")
    print(f"[oracle] argmax token id = {argmax_id}  text = {cand_payload['argmax_text']!r}")
    for cid in candidates:
        print(f"    candidate {cid:>7} ({cand_texts[cid]!r}): logit = {cand_logits[cid]:.6f}")
    print(f"[oracle] dump directory: {outdir}")
    print("ORACLE OK")
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        sys.stderr.write("ORACLE FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
