#!/usr/bin/env python3
"""M5: package the already-quantized Q4_K bytes plus the F32 norms into a GGUF.

This writer NEVER quantizes.  For every matmul weight the Q4_K payload is
copied byte-for-byte from the artifact produced once by
``tools/qwen3_quantize_model.py`` (``artifacts/m5_parity/q4/<name>.q4k``), which
is the same payload the Python oracle consumes.  Norms / embeddings / LM head
stay in F32, again exactly as the oracle sees them.

GGUF v3 layout, alignment 32, tensors emitted in deterministic (sorted) order.

Geometry convention (matches src/q3_binder.c ``bind_matrix``):
  dim[0] = contiguous row length  = HF columns (input width)
  dim[1] = row count              = HF rows    (output width)
The payload itself is stored verbatim in HF ``[rows, cols]`` order.
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747  # b"GGUF" read as little-endian u32
GGUF_VERSION = 3
ALIGNMENT = 32

(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12
QK_K = 256
Q4_K_BLOCK_BYTES = 144

TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = range(1, 7)

HF_TO_GGUF_SUFFIX = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
HF_TO_GGUF_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def hf_name_to_gguf(name):
    if name in HF_TO_GGUF_GLOBAL:
        return HF_TO_GGUF_GLOBAL[name]
    prefix = "model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        idx, _, tail = rest.partition(".")
        if tail in HF_TO_GGUF_SUFFIX:
            return "blk.%s.%s" % (idx, HF_TO_GGUF_SUFFIX[tail])
    raise KeyError("no GGUF name mapping for %s" % name)


class Writer:
    """Little-endian GGUF primitive writer."""

    def __init__(self):
        self.buf = bytearray()

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def i32(self, v):
        self.buf += struct.pack("<i", v)

    def u64(self, v):
        self.buf += struct.pack("<Q", v)

    def f32(self, v):
        self.buf += struct.pack("<f", v)

    def string(self, s):
        b = s.encode("utf-8")
        self.u64(len(b))
        self.buf += b

    def key(self, name, typ):
        self.string(name)
        self.u32(typ)


def add_u32(w, name, v):
    w.key(name, T_UINT32)
    w.u32(v)


def add_f32(w, name, v):
    w.key(name, T_FLOAT32)
    w.f32(v)


def add_str(w, name, v):
    w.key(name, T_STRING)
    w.string(v)


def add_bool(w, name, v):
    w.key(name, T_BOOL)
    w.buf += bytes([1 if v else 0])


def add_str_array(w, name, items):
    w.key(name, T_ARRAY)
    w.u32(T_STRING)
    w.u64(len(items))
    for s in items:
        w.string(s)


def add_i32_array(w, name, items):
    w.key(name, T_ARRAY)
    w.u32(T_INT32)
    w.u64(len(items))
    for v in items:
        w.i32(int(v))


# --------------------------------------------------------------------------
# safetensors access (header-only index, streamed reads)
# --------------------------------------------------------------------------

def build_index(ckpt_dir):
    index = {}
    for f in sorted(Path(ckpt_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            index[name] = {
                "file": str(f),
                "start": base + info["data_offsets"][0],
                "end": base + info["data_offsets"][1],
                "dtype": info["dtype"],
                "shape": tuple(info["shape"]),
            }
    return index


def read_raw(index, name):
    info = index[name]
    with open(info["file"], "rb") as fh:
        fh.seek(info["start"])
        return fh.read(info["end"] - info["start"])


def to_f32_bytes(index, name):
    info = index[name]
    raw = read_raw(index, name)
    dt = info["dtype"]
    if dt == "F32":
        return raw
    if dt == "BF16":
        a = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return a.view(np.float32).astype("<f4").tobytes()
    raise ValueError("unsupported dtype %s for %s" % (dt, name))


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

def build_tokenizer(ckpt_dir):
    tok = json.load(open(Path(ckpt_dir) / "tokenizer.json"))
    tcfg_path = Path(ckpt_dir) / "tokenizer_config.json"
    tcfg = json.load(open(tcfg_path)) if tcfg_path.exists() else {}

    vocab = tok["model"]["vocab"]
    added = tok.get("added_tokens", [])

    max_id = max(vocab.values())
    if added:
        max_id = max(max_id, max(a["id"] for a in added))
    id_to_token = [None] * (max_id + 1)
    for s, i in vocab.items():
        id_to_token[i] = s
    types = [TT_NORMAL] * (max_id + 1)
    for a in added:
        id_to_token[a["id"]] = a["content"]
        types[a["id"]] = TT_CONTROL if a.get("special") else TT_USER_DEFINED
    holes = [i for i, s in enumerate(id_to_token) if s is None]
    if holes:
        raise SystemExit("vocabulary has holes: %r" % holes[:8])

    merges = []
    for m in tok["model"]["merges"]:
        merges.append(m if isinstance(m, str) else " ".join(m))

    def resolve(key, default=None):
        v = tcfg.get(key, default)
        if isinstance(v, dict):
            return v.get("content")
        return v

    bos = eos = pad = None
    for a in added:
        if a["content"] == "<|endoftext|>":
            bos = pad = a["id"]
        elif a["content"] == "<|im_end|>":
            eos = a["id"]
    if bos is None or eos is None:
        raise SystemExit("missing bos/eos added tokens")

    return {
        "tokens": id_to_token,
        "types": types,
        "merges": merges,
        "bos": bos,
        "eos": eos,
        "pad": pad,
        "add_bos": bool(resolve("add_bos_token", False)),
        "add_eos": bool(resolve("add_eos_token", False)),
        "chat_template": resolve("chat_template"),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--q4dir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Qwen3-32B-Q4-production")
    ap.add_argument("--dry-run", action="store_true",
                    help="print type counts and payload bytes, write nothing")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.load(open(ckpt / "config.json"))
    manifest_path = (Path(args.manifest) if args.manifest
                     else Path(args.q4dir) / "q4_manifest.json")
    q4 = json.load(open(manifest_path))

    print("[gguf] checkpoint %s" % ckpt, flush=True)
    print("[gguf] q4 tensors %d" % len(q4), flush=True)

    index = build_index(ckpt)
    print("[gguf] safetensors entries %d" % len(index), flush=True)

    # Policy: rank 2 -> Q4_K, rank 1 -> F32.  The FP8 ``*_scale_inv`` tensors
    # are dequantization metadata consumed by the quantizer, not model weights,
    # and are therefore never emitted.
    entries = []  # dicts: gguf, hf, type, dims, nbytes, src, off
    skipped = []
    for hf_name in sorted(index):
        if hf_name.endswith("weight_scale_inv"):
            skipped.append(hf_name)
            continue
        info = index[hf_name]
        rank = len(info["shape"])
        if rank not in (1, 2):
            raise SystemExit("%s: unsupported rank %d" % (hf_name, rank))
        gguf_name = hf_name_to_gguf(hf_name)

        if rank == 2:
            if hf_name not in q4:
                raise SystemExit("%s: rank-2 tensor missing from Q4 manifest"
                                 % hf_name)
            qinfo = q4[hf_name]
            rows, cols = qinfo["shape"]
            if (rows, cols) != tuple(info["shape"]):
                raise SystemExit("%s: manifest shape %r != checkpoint %r"
                                 % (hf_name, qinfo["shape"], info["shape"]))
            if cols % QK_K:
                raise SystemExit("%s: cols %d not a multiple of %d"
                                 % (hf_name, cols, QK_K))
            nbytes = rows * (cols // QK_K) * Q4_K_BLOCK_BYTES
            if nbytes != qinfo["bytes"]:
                raise SystemExit("%s: byte mismatch %d != %d"
                                 % (hf_name, nbytes, qinfo["bytes"]))
            path = Path(args.q4dir) / qinfo["file"]
            if not path.exists() or path.stat().st_size != nbytes:
                raise SystemExit("bad q4 payload %s" % path)
            entries.append({
                "gguf": gguf_name, "hf": hf_name, "type": GGML_TYPE_Q4_K,
                "dims": (cols, rows), "nbytes": nbytes,
                "src": ("q4k", str(path)), "off": 0,
            })
        else:
            dims = info["shape"]
            nbytes = int(np.prod(dims)) * 4
            entries.append({
                "gguf": gguf_name, "hf": hf_name, "type": GGML_TYPE_F32,
                "dims": dims, "nbytes": nbytes, "src": ("f32", hf_name), "off": 0,
            })

    entries.sort(key=lambda e: e["gguf"])
    n_q4 = sum(1 for e in entries if e["type"] == GGML_TYPE_Q4_K)
    n_f32 = len(entries) - n_q4
    print("[gguf] %d tensors: Q4_K=%d F32=%d (%d scale_inv excluded)"
          % (len(entries), n_q4, n_f32, len(skipped)), flush=True)

    offset = 0
    for e in entries:
        if offset % ALIGNMENT:
            offset += ALIGNMENT - (offset % ALIGNMENT)
        e["off"] = offset
        offset += e["nbytes"]
    total_data = offset

    if args.dry_run:
        print("[gguf] DRY RUN - nothing written", flush=True)
        print("[gguf] Q4_K tensors: %d" % n_q4, flush=True)
        print("[gguf] F32  tensors: %d" % n_f32, flush=True)
        print("[gguf] total tensors: %d" % len(entries), flush=True)
        print("[gguf] payload bytes: %d (%.2f GiB)"
              % (total_data, total_data / 2**30), flush=True)
        for want in ("token_embd.weight", "output.weight"):
            for e in entries:
                if e["gguf"] == want:
                    print("[gguf] %-20s hf=%-34s type=%s dims=%s bytes=%d"
                          % (e["gguf"], e["hf"],
                             "Q4_K" if e["type"] == GGML_TYPE_Q4_K else "F32",
                             e["dims"], e["nbytes"]), flush=True)
        return

    tok = build_tokenizer(ckpt)
    print("[gguf] vocab %d merges %d" % (len(tok["tokens"]), len(tok["merges"])),
          flush=True)

    w = Writer()
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(entries))

    meta = Writer()
    n_meta = 0

    def M(fn, name, val):
        nonlocal n_meta
        fn(meta, name, val)
        n_meta += 1

    M(add_str, "general.architecture", "qwen3")
    M(add_str, "general.name", args.name)
    M(add_u32, "general.file_type", 15)
    M(add_u32, "general.quantization_version", 2)

    M(add_u32, "qwen3.context_length",
      int(cfg.get("max_position_embeddings", 40960)))
    M(add_u32, "qwen3.block_count", int(cfg["num_hidden_layers"]))
    M(add_u32, "qwen3.embedding_length", int(cfg["hidden_size"]))
    M(add_u32, "qwen3.feed_forward_length", int(cfg["intermediate_size"]))
    M(add_u32, "qwen3.attention.head_count", int(cfg["num_attention_heads"]))
    M(add_u32, "qwen3.attention.head_count_kv", int(cfg["num_key_value_heads"]))
    M(add_u32, "qwen3.attention.key_length", int(cfg["head_dim"]))
    M(add_u32, "qwen3.attention.value_length", int(cfg["head_dim"]))
    M(add_f32, "qwen3.rope.freq_base", float(cfg["rope_theta"]))
    M(add_f32, "qwen3.attention.layer_norm_rms_epsilon",
      float(cfg["rms_norm_eps"]))

    M(add_str, "tokenizer.ggml.model", "gpt2")
    M(add_str, "tokenizer.ggml.pre", "qwen2")
    M(add_str_array, "tokenizer.ggml.tokens", tok["tokens"])
    M(add_i32_array, "tokenizer.ggml.token_type", tok["types"])
    M(add_str_array, "tokenizer.ggml.merges", tok["merges"])
    M(add_u32, "tokenizer.ggml.bos_token_id", tok["bos"])
    M(add_u32, "tokenizer.ggml.eos_token_id", tok["eos"])
    M(add_u32, "tokenizer.ggml.padding_token_id", tok["pad"])
    M(add_bool, "tokenizer.ggml.add_bos_token", tok["add_bos"])
    M(add_bool, "tokenizer.ggml.add_eos_token", tok["add_eos"])
    if tok["chat_template"]:
        M(add_str, "tokenizer.chat_template", tok["chat_template"])

    w.u64(n_meta)
    w.buf += meta.buf

    for e in entries:
        w.string(e["gguf"])
        w.u32(len(e["dims"]))
        for d in e["dims"]:
            w.u64(int(d))
        w.u32(e["type"])
        w.u64(e["off"])

    if len(w.buf) % ALIGNMENT:
        w.buf += b"\x00" * (ALIGNMENT - (len(w.buf) % ALIGNMENT))
    print("[gguf] header %.1f MiB, data %.2f GiB"
          % (len(w.buf) / 2**20, total_data / 2**30), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(w.buf)
        cur = 0
        for i, e in enumerate(entries):
            if e["off"] > cur:
                fh.write(b"\x00" * (e["off"] - cur))
                cur = e["off"]
            kind, src = e["src"]
            if kind == "q4k":
                with open(src, "rb") as sf:
                    while True:
                        chunk = sf.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
            else:
                fh.write(to_f32_bytes(index, src))
            cur += e["nbytes"]
            if (i + 1) % 100 == 0 or i + 1 == len(entries):
                print("[gguf] %d/%d tensors, %.2f GiB"
                      % (i + 1, len(entries), cur / 2**30), flush=True)
    os.replace(tmp, out)
    print("[gguf] wrote %s (%.2f GiB)"
          % (out, out.stat().st_size / 2**30), flush=True)


if __name__ == "__main__":
    main()
