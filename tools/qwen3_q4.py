#!/usr/bin/env python3
"""M5: Qwen3 FP8 -> Q4_K quantization in pure Python.

Pipeline (per tensor):
    FP8 E4M3 bytes + BF16 weight_scale_inv
        -> float32  (w = decode_e4m3(fp8) * scale_inv, per 128x128 block)
        -> Q4_K blocks (144 bytes per 256 elements, layout matches
           src/q3_quant.h q3_q4_k_block and the donor quants.c writer)

The exact Q4_K bytes are saved once to artifacts/m5_parity/q4/ and reused by
both the Python oracle and the GGUF writer.  No second quantization.

Layout reference (q3_q4_k_block, 144 bytes):
    offset 0:  uint16 d     (f16 of max_scale/63)
    offset 2:  uint16 dmin  (f16 of max_min/63)
    offset 4:  uint8  scales[12]
    offset 16: uint8  qs[128]
"""

import argparse
import json
import os
import struct

import numpy as np

QK_K = 256
Q4_K_BLOCK_BYTES = 144
BLOCK_OUT = 128  # FP8 scale block size (rows)
BLOCK_IN = 128   # FP8 scale block size (cols)


# ---------------------------------------------------------------------------
# FP8 E4M3 decode
# ---------------------------------------------------------------------------

def e4m3_to_f32(x):
    """Decode one FP8 E4M3 value to float32 (matches donor e4m3fn_to_f32)."""
    x = int(x) & 0xFF
    abs_x = x & 0x7F
    sign = (x & 0x80) != 0
    if abs_x == 0:
        return -0.0 if sign else 0.0
    if abs_x == 0x7F:
        return 0.0  # NaN -> 0
    exp = (x >> 3) & 0x0F
    man = x & 0x07
    if exp == 0:
        value = np.ldexp(float(man), -9)
    else:
        value = np.ldexp(1.0 + float(man) / 8.0, exp - 7)
    return -value if sign else value


def decode_e4m3_array(data):
    """Decode a byte array of FP8 E4M3 to float32 numpy array."""
    data = np.asarray(data, dtype=np.uint8)
    out = np.empty(data.shape, dtype=np.float32)
    # vectorized decode
    abs_x = data & 0x7F
    sign = (data & 0x80) != 0
    exp = (data >> 3) & 0x0F
    man = data & 0x07
    # exp==0: man * 2^-9 ; else (1 + man/8) * 2^(exp-7)
    value = np.where(
        exp == 0,
        man.astype(np.float32) * np.float32(2.0 ** -9),
        (1.0 + man.astype(np.float32) / 8.0) * np.float32(2.0) ** (exp.astype(np.float32) - 7.0),
    )
    value = np.where(abs_x == 0x7F, 0.0, value)
    value = np.where(abs_x == 0, 0.0, value)
    out = np.where(sign, -value, value)
    return out


def bf16_to_f32(bits):
    """BF16 uint16 -> float32."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """float32 -> BF16 uint16 (round-to-nearest-even, matches donor)."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    # round to nearest even
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
    # NaN handling
    nan_mask = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan_mask, (rounded | np.uint32(64)).astype(np.uint32), rounded)
    return rounded.astype(np.uint16)


def f32_to_f16(x):
    """float32 -> FP16 uint16 (matches donor ds4q_f32_to_f16)."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    sign = bits & np.uint32(0x80000000)
    shl1_w = bits + bits
    bias = shl1_w & np.uint32(0xFF000000)
    bias = np.where(bias < np.uint32(0x71000000), np.uint32(0x71000000), bias)
    base = np.abs(x) * np.float32(2.0 ** 112) * np.float32(2.0 ** -110)
    base = ((bias >> 1) + np.uint32(0x07800000)).view(np.float32) + base
    out = base.view(np.uint32)
    exp_bits = (out >> 13) & np.uint32(0x00007C00)
    mantissa_bits = out & np.uint32(0x00000FFF)
    nonsign = exp_bits + mantissa_bits
    result = (sign >> 16) | np.where(shl1_w > np.uint32(0xFF000000), np.uint16(0x7E00), nonsign.astype(np.uint16))
    return result.astype(np.uint16)


def f16_to_f32(bits):
    """FP16 uint16 -> float32 (matches donor ds4q_f16_to_f32)."""
    bits = np.asarray(bits, dtype=np.uint16)
    w = bits.astype(np.uint32) << 16
    sign = w & np.uint32(0x80000000)
    two_w = w + w
    exp_offset = np.uint32(0xE0) << 23
    exp_scale = np.float32(2.0 ** -112)
    normalized = ((two_w >> 4) + exp_offset).view(np.float32) * exp_scale
    magic_mask = np.uint32(126) << 23
    magic_bias = np.float32(0.5)
    denormalized = ((two_w >> 17) | magic_mask).view(np.float32) - magic_bias
    denormalized_cutoff = np.uint32(1) << 27
    result = sign | np.where(
        two_w < denormalized_cutoff,
        denormalized.view(np.uint32),
        normalized.view(np.uint32),
    )
    return result.view(np.float32)


# ---------------------------------------------------------------------------
# Q4_K quantization (port of donor ds4q_write_q4_k_block_ref)
# ---------------------------------------------------------------------------

def _nearest_int(fval):
    """Port of ds4q_nearest_int (round-half-away via magic constant)."""
    val = fval + 12582912.0
    i = val.view(np.int32)
    return (i & 0x007FFFFF) - 0x00400000


def _make_qkx2_quants(x, weights, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """Port of ds4q_make_qkx2_quants for n=32.

    Returns (L, the_min, scale).
    """
    n = 32
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    L = np.zeros(n, dtype=np.uint8)
    Laux = np.zeros(n, dtype=np.uint8)

    min_v = float(x[0])
    max_v = float(x[0])
    sum_w = float(weights[0])
    sum_x = sum_w * float(x[0])
    for i in range(1, n):
        if x[i] < min_v:
            min_v = float(x[i])
        if x[i] > max_v:
            max_v = float(x[i])
        w = float(weights[i])
        sum_w += w
        sum_x += w * float(x[i])
    if min_v > 0:
        min_v = 0
    if max_v == min_v:
        return L, -min_v, 0.0
    iscale = nmax / (max_v - min_v)
    scale = 1.0 / iscale
    best_error = 0.0
    for i in range(n):
        l = _nearest_int(np.float32(iscale * (x[i] - min_v)))
        l = max(0, min(nmax, l))
        L[i] = l
        diff = scale * l + min_v - x[i]
        best_error += weights[i] * (diff * diff)
    if nstep < 1:
        return L, -min_v, scale
    for istep in range(nstep + 1):
        if max_v <= min_v:
            break
        iscale = (rmin + rdelta * istep + nmax) / (max_v - min_v)
        sum_l = 0.0
        sum_l2 = 0.0
        sum_xl = 0.0
        for i in range(n):
            l = _nearest_int(np.float32(iscale * (x[i] - min_v)))
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = float(weights[i])
            sum_l += w * l
            sum_l2 += w * l * l
            sum_xl += w * l * float(x[i])
        D = sum_w * sum_l2 - sum_l * sum_l
        if D > 0:
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            if this_min > 0:
                this_min = 0
                this_scale = sum_xl / sum_l2
            cur_error = 0.0
            for i in range(n):
                diff = this_scale * Laux[i] + this_min - x[i]
                cur_error += weights[i] * (diff * diff)
            if cur_error < best_error:
                L[:] = Laux
                best_error = cur_error
                scale = this_scale
                min_v = this_min
    return L, -min_v, scale


def _get_scale_min_k4(j, scales):
    """Port of ds4q_get_scale_min_k4."""
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def quantize_q4_k_block(x):
    """Quantize 256 float32 values into one 144-byte Q4_K block.

    Port of donor ds4q_write_q4_k_block_ref.  Returns bytes (144).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    assert x.size == QK_K, f"block must have {QK_K} elements, got {x.size}"

    y = bytearray(Q4_K_BLOCK_BYTES)
    L = np.zeros(QK_K, dtype=np.uint8)
    Laux = np.zeros(32, dtype=np.uint8)
    mins = np.zeros(QK_K // 32, dtype=np.float32)
    scales = np.zeros(QK_K // 32, dtype=np.float32)

    max_scale = 0.0
    max_min = 0.0
    for j in range(QK_K // 32):
        xj = x[32 * j:32 * j + 32]
        sum_x2 = float(np.sum(xj * xj))
        av_x = float(np.sqrt(sum_x2 / 32))
        weights = av_x + np.abs(xj)
        L32, m, s = _make_qkx2_quants(xj, weights)
        L[32 * j:32 * j + 32] = L32
        mins[j] = m
        scales[j] = s
        if s > max_scale:
            max_scale = s
        if m > max_min:
            max_min = m

    inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
    inv_min = 63.0 / max_min if max_min > 0 else 0.0

    scales_out = np.zeros(12, dtype=np.uint8)
    for j in range(QK_K // 32):
        ls = int(round(inv_scale * scales[j]))
        lm = int(round(inv_min * mins[j]))
        ls = min(63, ls)
        lm = min(63, lm)
        if j < 4:
            scales_out[j] = ls
            scales_out[j + 4] = lm
        else:
            scales_out[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4)
            scales_out[j - 4] |= ((ls >> 4) << 6)
            scales_out[j] |= ((lm >> 4) << 6)

    d = f32_to_f16(np.float32(max_scale / 63.0))
    dmin = f32_to_f16(np.float32(max_min / 63.0))
    struct.pack_into("<H", y, 0, int(d))
    struct.pack_into("<H", y, 2, int(dmin))
    y[4:16] = scales_out.tobytes()

    # recompute quantized levels
    d_f = float(f16_to_f32(np.uint16(int(d))))
    dmin_f = float(f16_to_f32(np.uint16(int(dmin))))
    for j in range(QK_K // 32):
        sc, m = _get_scale_min_k4(j, scales_out)
        dd = d_f * sc
        if dd == 0:
            continue
        dm = dmin_f * m
        for ii in range(32):
            l = _nearest_int(np.float32((x[32 * j + ii] + dm) / dd))
            l = max(0, min(15, l))
            L[32 * j + ii] = l

    # pack nibbles: for j in steps of 64, q[l] = L[j+l] | (L[j+l+32] << 4)
    q = np.zeros(128, dtype=np.uint8)
    for j in range(0, QK_K, 64):
        for l in range(32):
            q[(j // 64) * 32 + l] = L[j + l] | (L[j + l + 32] << 4)
    y[16:144] = q.tobytes()
    return bytes(y)


def quantize_q4_k_row(row):
    """Quantize one row (ncols float32) into Q4_K blocks.  ncols % 256 == 0."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    ncols = row.size
    assert ncols % QK_K == 0, f"ncols {ncols} not divisible by {QK_K}"
    nblocks = ncols // QK_K
    out = bytearray(nblocks * Q4_K_BLOCK_BYTES)
    for b in range(nblocks):
        blk = quantize_q4_k_block(row[b * QK_K:(b + 1) * QK_K])
        out[b * Q4_K_BLOCK_BYTES:(b + 1) * Q4_K_BLOCK_BYTES] = blk
    return bytes(out)


def _nearest_int_vec(fval):
    """Vectorized ds4q_nearest_int (round-half-away via magic constant)."""
    shape_in = np.shape(fval)
    val = np.asarray(fval, dtype=np.float32) + np.float32(12582912.0)
    i = val.view(np.int32)
    out = (i & 0x007FFFFF) - 0x00400000
    assert out.shape == shape_in, (shape_in, out.shape)
    return out


def _make_qkx2_quants_vec(x, weights, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """Vectorized ds4q_make_qkx2_quants over batch dim.  x: [..., 32].

    Returns (L, mins, scales) with L shaped like x, mins/scales shaped x[..., 0].
    """
    orig_shape = x.shape
    B = int(np.prod(orig_shape[:-1]))
    x = x.reshape(B, 32)
    weights = weights.reshape(B, 32)
    L = np.zeros((B, 32), dtype=np.uint8)
    Laux = np.zeros((B, 32), dtype=np.uint8)

    min_v = x.min(axis=1)
    max_v = x.max(axis=1)
    sum_w = weights.sum(axis=1)
    sum_x = np.sum(weights * x, axis=1)

    min_v = np.where(min_v > 0, 0.0, min_v)
    eq = max_v == min_v
    max_v_safe = np.where(eq, min_v + 1.0, max_v)

    iscale = nmax / (max_v_safe - min_v)
    scale = 1.0 / iscale
    l = _nearest_int_vec(iscale[:, None] * (x - min_v[:, None]))
    l = np.clip(l, 0, nmax)
    L = l.astype(np.uint8)
    diff = scale[:, None] * l + min_v[:, None] - x
    best_error = np.sum(weights * diff * diff, axis=1)

    for istep in range(nstep + 1):
        iscale = (rmin + rdelta * istep + nmax) / (max_v_safe - min_v)
        iscale = np.asarray(iscale, dtype=np.float32)
        z = iscale[:, None] * (x - min_v[:, None])
        assert z.shape == x.shape, (z.shape, x.shape)
        assert z.shape == weights.shape, (z.shape, weights.shape)
        l = _nearest_int_vec(z)
        assert l.shape == x.shape, (l.shape, x.shape)
        l = np.clip(l, 0, nmax)
        Laux = l.astype(np.uint8)
        sum_l = np.sum(weights * l, axis=1)
        sum_l2 = np.sum(weights * l * l, axis=1)
        sum_xl = np.sum(weights * l * x, axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        Dpos = D > 0
        Dsafe = np.where(Dpos, D, 1.0)
        this_scale = np.where(Dpos, (sum_w * sum_xl - sum_x * sum_l) / Dsafe, 0.0)
        this_min = np.where(Dpos, (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe, 0.0)
        mpos = this_min > 0
        this_min = np.where(mpos, 0.0, this_min)
        this_scale = np.where(mpos, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), this_scale)
        cur_error = np.sum(
            weights * (this_scale[:, None] * Laux + this_min[:, None] - x) ** 2,
            axis=1,
        )
        better = cur_error < best_error
        L = np.where(better[:, None], Laux, L)
        best_error = np.where(better, cur_error, best_error)
        scale = np.where(better, this_scale, scale)
        min_v = np.where(better, this_min, min_v)

    # degenerate blocks (max_v == min_v): L=0, m=-min_v, s=0
    L = np.where(eq[:, None], 0, L)
    scale = np.where(eq, 0.0, scale)
    return L.reshape(orig_shape), -min_v.reshape(orig_shape[:-1]), scale.reshape(orig_shape[:-1])


def _get_scale_min_k4_vec(scales_out):
    """Vectorized ds4q_get_scale_min_k4.  scales_out: [R, B, 12].

    Returns (d, m) each [R, B, 8].
    """
    s = scales_out
    d = np.zeros(s.shape[:2] + (8,), dtype=np.uint8)
    m = np.zeros(s.shape[:2] + (8,), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            d[:, :, j] = s[:, :, j] & 63
            m[:, :, j] = s[:, :, j + 4] & 63
        else:
            d[:, :, j] = (s[:, :, j + 4] & 0x0F) | ((s[:, :, j - 4] >> 6) << 4)
            m[:, :, j] = (s[:, :, j + 4] >> 4) | ((s[:, :, j] >> 6) << 4)
    return d, m


def quantize_q4_k_matrix(mat):
    """Quantize a 2D float32 matrix [rows, cols] -> bytes (rows * nblocks * 144).

    Vectorized over all rows/sub-blocks.  Each row is quantized independently
    (matches the donor row-streaming path).
    """
    mat = np.asarray(mat, dtype=np.float32)
    assert mat.ndim == 2
    rows, ncols = mat.shape
    assert ncols % QK_K == 0, f"ncols {ncols} not divisible by {QK_K}"
    nblocks = ncols // QK_K
    nsub = nblocks * 8

    x = mat.reshape(rows, nsub, 32)
    sum_x2 = np.sum(x * x, axis=2)
    av_x = np.sqrt(sum_x2 / 32.0)
    weights = av_x[:, :, None] + np.abs(x)

    L, mins, scales = _make_qkx2_quants_vec(x, weights)
    # L: [rows, nsub] uint8 ; mins/scales: [rows, nsub]

    max_scale = scales.reshape(rows, nblocks, 8).max(axis=2)
    max_min = mins.reshape(rows, nblocks, 8).max(axis=2)

    inv_scale = np.where(max_scale > 0, 63.0 / max_scale, 0.0)
    inv_min = np.where(max_min > 0, 63.0 / max_min, 0.0)

    sc = scales.reshape(rows, nblocks, 8)
    mn = mins.reshape(rows, nblocks, 8)
    ls = np.round(inv_scale[:, :, None] * sc).astype(np.int32)
    lm = np.round(inv_min[:, :, None] * mn).astype(np.int32)
    ls = np.clip(ls, 0, 63)
    lm = np.clip(lm, 0, 63)

    scales_out = np.zeros((rows, nblocks, 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            scales_out[:, :, j] = ls[:, :, j]
            scales_out[:, :, j + 4] = lm[:, :, j]
        else:
            scales_out[:, :, j + 4] = (ls[:, :, j] & 0x0F) | ((lm[:, :, j] & 0x0F) << 4)
            scales_out[:, :, j - 4] |= (((ls[:, :, j] >> 4) << 6) & 0xC0).astype(np.uint8)
            scales_out[:, :, j] |= (((lm[:, :, j] >> 4) << 6) & 0xC0).astype(np.uint8)

    d = f32_to_f16((max_scale / 63.0).astype(np.float32))  # [rows, nblocks]
    dmin = f32_to_f16((max_min / 63.0).astype(np.float32))

    # recompute quantized levels with the stored (d, dmin, scales)
    d_f = f16_to_f32(d)
    dmin_f = f16_to_f32(dmin)
    sc2, m2 = _get_scale_min_k4_vec(scales_out)  # [rows, nblocks, 8]
    dd = d_f[:, :, None] * sc2
    dm = dmin_f[:, :, None] * m2
    xr = x.reshape(rows, nblocks, 8, 32)
    dd_safe = np.where(dd == 0, 1.0, dd)
    l = _nearest_int_vec((xr + dm[:, :, :, None]) / dd_safe[:, :, :, None])
    l = np.clip(l, 0, 15)
    Lr = np.where(dd[:, :, :, None] == 0, L.reshape(rows, nblocks, 8, 32), l).astype(np.uint8)

    # pack nibbles: q[(j//64)*32 + l] = L[j+l] | (L[j+l+32] << 4)
    Lb = Lr.reshape(rows, nblocks, 8, 32)
    q = np.zeros((rows, nblocks, 128), dtype=np.uint8)
    for pair in range(4):
        s0 = pair * 2
        q[:, :, pair * 32:(pair + 1) * 32] = Lb[:, :, s0] | (Lb[:, :, s0 + 1] << 4)

    out = np.zeros((rows, nblocks, 144), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(rows, nblocks, 2)
    out[:, :, 2:4] = dmin.view(np.uint8).reshape(rows, nblocks, 2)
    out[:, :, 4:16] = scales_out
    out[:, :, 16:144] = q
    return out.tobytes()


# ---------------------------------------------------------------------------
# Q4_K dequant (matches src/q3_quant.c dequant_q4)
# ---------------------------------------------------------------------------

def dequant_q4_k_block(blk):
    """Dequantize one 144-byte Q4_K block to 256 float32 (matches runtime)."""
    d = float(f16_to_f32(np.uint16(struct.unpack_from("<H", blk, 0)[0])))
    min_v = float(f16_to_f32(np.uint16(struct.unpack_from("<H", blk, 2)[0])))
    scales = np.frombuffer(blk, dtype=np.uint8, count=12, offset=4)
    q = np.frombuffer(blk, dtype=np.uint8, count=128, offset=16)
    out = np.zeros(QK_K, dtype=np.float32)
    scale_index = 0
    qpos = 0
    for j in range(0, QK_K, 64):
        sc, m = _get_scale_min_k4(scale_index, scales)
        scale_index += 1
        d1 = d * sc
        m1 = min_v * m
        sc, m = _get_scale_min_k4(scale_index, scales)
        scale_index += 1
        d2 = d * sc
        m2 = min_v * m
        for l in range(32):
            out[j + l] = d1 * (q[qpos + l] & 0x0F) - m1
        for l in range(32):
            out[j + 32 + l] = d2 * (q[qpos + l] >> 4) - m2
        qpos += 32
    return out


def dequant_q4_k_matrix(blk_bytes, rows, ncols):
    """Dequantize Q4_K bytes back to [rows, ncols] float32."""
    nblocks = ncols // QK_K
    out = np.zeros((rows, ncols), dtype=np.float32)
    for r in range(rows):
        for b in range(nblocks):
            off = (r * nblocks + b) * Q4_K_BLOCK_BYTES
            out[r, b * QK_K:(b + 1) * QK_K] = dequant_q4_k_block(blk_bytes[off:off + Q4_K_BLOCK_BYTES])
    return out


# ---------------------------------------------------------------------------
# Raw safetensors reader (F8_E4M3 is not mappable to numpy, read raw bytes)
# ---------------------------------------------------------------------------

def read_safetensors_raw(path, name):
    """Read raw bytes + shape + dtype for one tensor from a safetensors file.

    Returns (bytes, shape, dtype).  dtype is the safetensors dtype string
    (e.g. "F8_E4M3", "BF16").
    """
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        if name not in header:
            raise KeyError(f"{name} not in {path}")
        info = header[name]
        start, end = info["data_offsets"]
        fh.seek(8 + header_len + start)
        data = fh.read(end - start)
        return data, list(info["shape"]), info["dtype"]


# ---------------------------------------------------------------------------
# FP8 tensor -> float32 (Qwen3 semantics: w = decode_e4m3(fp8) * scale_inv)
# ---------------------------------------------------------------------------

def fp8_weight_to_f32(fp8_bytes, scale_inv_bytes, out_rows, in_cols):
    """Dequantize an FP8 E4M3 weight with BF16 weight_scale_inv.

    w = decode_e4m3(fp8) * scale_inv, per 128x128 block.
    scale_inv shape: [out_rows/128, in_cols/128] BF16.
    """
    fp8 = np.frombuffer(fp8_bytes, dtype=np.uint8).reshape(out_rows, in_cols)
    w = decode_e4m3_array(fp8)  # [out, in] float32

    sr = out_rows // BLOCK_OUT
    sc = in_cols // BLOCK_IN
    si = np.frombuffer(scale_inv_bytes, dtype=np.uint16).reshape(sr, sc)
    si_f = bf16_to_f32(si)  # [sr, sc] float32

    # expand to [out, in]
    si_exp = np.repeat(np.repeat(si_f, BLOCK_OUT, axis=0), BLOCK_IN, axis=1)
    return w * si_exp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Qwen3 FP8 -> Q4_K quantization (Python)")
    p.add_argument("--checkpoint", required=True, help="HF FP8 checkpoint dir")
    p.add_argument("--tensor", default=None,
                   help="single tensor name to convert (e.g. model.layers.0.self_attn.q_proj.weight); "
                        "default: all FP8 weights")
    p.add_argument("--outdir", required=True, help="output dir for Q4_K bytes + manifest")
    p.add_argument("--verify", action="store_true",
                   help="dequantize and report max abs error vs FP32 reference")
    args = p.parse_args()

    from safetensors import safe_open  # noqa: F401  (kept for reference)

    os.makedirs(args.outdir, exist_ok=True)
    index_path = os.path.join(args.checkpoint, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    weight_map = index["weight_map"]

    # determine target tensors
    if args.tensor:
        targets = [args.tensor]
    else:
        targets = sorted(
            k for k in weight_map
            if k.endswith(".weight") and not k.endswith("_scale_inv")
        )

    manifest = {}
    for name in targets:
        shard = weight_map[name]
        shard_path = os.path.join(args.checkpoint, shard)
        fp8_bytes, meta, dtype = read_safetensors_raw(shard_path, name)
        if dtype != "F8_E4M3":
            print(f"[skip] {name}: dtype {dtype} (not FP8)")
            continue
        scale_name = name + "_scale_inv"
        if scale_name not in weight_map:
            print(f"[skip] {name}: no weight_scale_inv")
            continue
        si_bytes, si_shape, si_dtype = read_safetensors_raw(shard_path, scale_name)

        rows, cols = meta[0], meta[1]
        print(f"[q4] {name} {rows}x{cols} ...")
        w_f32 = fp8_weight_to_f32(fp8_bytes, si_bytes, rows, cols)
        q4 = quantize_q4_k_matrix(w_f32)
        del w_f32

        out_name = name.replace("/", "__")
        out_path = os.path.join(args.outdir, out_name + ".q4k")
        with open(out_path, "wb") as fh:
            fh.write(q4)

        manifest[name] = {
            "file": out_name + ".q4k",
            "shard": shard,
            "shape": [rows, cols],
            "dtype": "Q4_K",
            "block_bytes": Q4_K_BLOCK_BYTES,
            "bytes": len(q4),
            "blocks_per_row": cols // QK_K,
        }
        print(f"    -> {out_path} ({len(q4)} bytes)")

        if args.verify:
            w_dq = dequant_q4_k_matrix(q4, rows, cols)
            err = np.abs(w_dq - fp8_weight_to_f32(fp8_bytes, si_bytes, rows, cols))
            print(f"    verify: max_abs_err={float(err.max()):.6f} "
                  f"mean_abs_err={float(err.mean()):.6f}")
            del w_dq

    with open(os.path.join(args.outdir, "q4_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] {len(manifest)} tensors -> {args.outdir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
