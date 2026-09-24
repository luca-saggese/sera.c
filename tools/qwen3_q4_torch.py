#!/usr/bin/env python3
"""Torch/CUDA port of the Q4_K quantizer in tools/qwen3_q4.py.

Same mathematics as the NumPy reference (which stays the oracle):
  * ds4q_nearest_int          -> _nearest_int_torch
  * ds4q_make_qkx2_quants     -> _make_qkx2_quants_torch
  * ds4q_get_scale_min_k4     -> _get_scale_min_k4_torch
  * ds4q_write_q4_k_block_ref -> quantize_q4_k_matrix_torch

The 21-step search loop is kept as a Python loop; every iteration issues
large element-wise GPU kernels.  float32 only, no autocast.
"""

from __future__ import annotations

import numpy as np
import torch

from qwen3_q4 import (
    QK_K,
    Q4_K_BLOCK_BYTES,
    f32_to_f16,
    f16_to_f32,
)

NMASK = 0x007FFFFF
NBIAS = 0x00400000
MAGIC = 12582912.0


def _nearest_int_torch(fval: torch.Tensor) -> torch.Tensor:
    """Port of ds4q_nearest_int (round-half-away via magic constant).

    fval must be float32 and |fval| <= 4194303.
    """
    shape_in = fval.shape
    val = fval.contiguous() + MAGIC
    i = val.view(torch.int32)
    out = (i & NMASK) - NBIAS
    assert out.shape == shape_in, (shape_in, out.shape)
    return out


def _make_qkx2_quants_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    nmax: int = 15,
    rmin: float = -1.0,
    rdelta: float = 0.1,
    nstep: int = 20,
):
    """Port of ds4q_make_qkx2_quants.  x, weights: [B, 32] float32.

    Returns (L uint8 [B,32], mins float32 [B], scales float32 [B]).
    """
    orig_shape = x.shape
    B = int(np.prod(orig_shape[:-1]))
    x = x.reshape(B, 32).contiguous()
    weights = weights.reshape(B, 32).contiguous()

    min_v = x.min(dim=1).values
    max_v = x.max(dim=1).values
    sum_w = weights.sum(dim=1)
    sum_x = (weights * x).sum(dim=1)

    min_v = torch.where(min_v > 0, torch.zeros_like(min_v), min_v)
    eq = max_v == min_v
    max_v_safe = torch.where(eq, min_v + 1.0, max_v)

    iscale = nmax / (max_v_safe - min_v)
    scale = 1.0 / iscale
    l = _nearest_int_torch(iscale[:, None] * (x - min_v[:, None]))
    l = torch.clamp(l, 0, nmax)
    L = l.to(torch.uint8)
    diff = scale[:, None] * l + min_v[:, None] - x
    best_error = (weights * diff * diff).sum(dim=1)

    for istep in range(nstep + 1):
        iscale = (rmin + rdelta * istep + nmax) / (max_v_safe - min_v)
        iscale = iscale.to(torch.float32)
        z = iscale[:, None] * (x - min_v[:, None])
        assert z.shape == x.shape, (z.shape, x.shape)
        assert z.shape == weights.shape, (z.shape, weights.shape)
        l = _nearest_int_torch(z)
        assert l.shape == x.shape, (l.shape, x.shape)
        l = torch.clamp(l, 0, nmax)
        Laux = l.to(torch.uint8)
        sum_l = (weights * l).sum(dim=1)
        sum_l2 = (weights * l * l).sum(dim=1)
        sum_xl = (weights * l * x).sum(dim=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        Dpos = D > 0
        Dsafe = torch.where(Dpos, D, torch.ones_like(D))
        this_scale = torch.where(Dpos, (sum_w * sum_xl - sum_x * sum_l) / Dsafe,
                                 torch.zeros_like(D))
        this_min = torch.where(Dpos, (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe,
                               torch.zeros_like(D))
        mpos = this_min > 0
        this_min = torch.where(mpos, torch.zeros_like(this_min), this_min)
        this_scale = torch.where(
            mpos,
            sum_xl / torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2)),
            this_scale,
        )
        cur_error = (
            weights * (this_scale[:, None] * Laux + this_min[:, None] - x) ** 2
        ).sum(dim=1)
        better = cur_error < best_error
        L = torch.where(better[:, None], Laux, L)
        best_error = torch.where(better, cur_error, best_error)
        scale = torch.where(better, this_scale, scale)
        min_v = torch.where(better, this_min, min_v)

    L = torch.where(eq[:, None], torch.zeros_like(L), L)
    scale = torch.where(eq, torch.zeros_like(scale), scale)
    return (
        L.reshape(orig_shape),
        (-min_v).reshape(orig_shape[:-1]),
        scale.reshape(orig_shape[:-1]),
    )


def _get_scale_min_k4_torch(scales_out: torch.Tensor):
    """Port of ds4q_get_scale_min_k4.  scales_out: [R, B, 12] uint8.

    Returns (d, m) each int32 [R, B, 8].
    """
    s = scales_out.to(torch.int32)
    d = torch.zeros(s.shape[:2] + (8,), dtype=torch.int32, device=s.device)
    m = torch.zeros_like(d)
    for j in range(8):
        if j < 4:
            d[:, :, j] = s[:, :, j] & 63
            m[:, :, j] = s[:, :, j + 4] & 63
        else:
            d[:, :, j] = (s[:, :, j + 4] & 0x0F) | ((s[:, :, j - 4] >> 6) << 4)
            m[:, :, j] = (s[:, :, j + 4] >> 4) | ((s[:, :, j] >> 6) << 4)
    return d, m


def quantize_q4_k_matrix_torch(
    mat,
    device: str = "cuda",
    chunk_rows: int = 1024,
) -> bytes:
    """Quantize a 2D float32 matrix [rows, cols] -> bytes (rows*nblocks*144).

    Identical algorithm to qwen3_q4.quantize_q4_k_matrix, executed on GPU in
    row chunks.  ncols % 256 == 0.
    """
    if isinstance(mat, torch.Tensor):
        mat_t = mat.to(device=device, dtype=torch.float32).contiguous()
    else:
        mat_np = np.ascontiguousarray(mat, dtype=np.float32)
        mat_t = torch.from_numpy(mat_np).to(device=device)

    assert mat_t.ndim == 2
    rows, ncols = mat_t.shape
    assert ncols % QK_K == 0, f"ncols {ncols} not divisible by {QK_K}"
    nblocks = ncols // QK_K
    nsub = nblocks * 8

    out_cpu = np.empty((rows, nblocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)

    with torch.inference_mode():
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            m = mat_t[r0:r1].contiguous()
            R = r1 - r0

            x = m.reshape(R, nsub, 32).contiguous()
            sum_x2 = (x * x).sum(dim=2)
            av_x = torch.sqrt(sum_x2 / 32.0)
            weights = av_x[:, :, None] + torch.abs(x)

            L, mins, scales = _make_qkx2_quants_torch(x, weights)
            assert L.shape == (R, nsub, 32), (L.shape, (R, nsub, 32))
            assert mins.shape == (R, nsub)
            assert scales.shape == (R, nsub)

            max_scale = scales.reshape(R, nblocks, 8).max(dim=2).values
            max_min = mins.reshape(R, nblocks, 8).max(dim=2).values

            inv_scale = torch.where(max_scale > 0, 63.0 / max_scale,
                                    torch.zeros_like(max_scale))
            inv_min = torch.where(max_min > 0, 63.0 / max_min,
                                  torch.zeros_like(max_min))

            sc = scales.reshape(R, nblocks, 8)
            mn = mins.reshape(R, nblocks, 8)
            ls = torch.round(inv_scale[:, :, None] * sc).to(torch.int32).clamp(0, 63)
            lm = torch.round(inv_min[:, :, None] * mn).to(torch.int32).clamp(0, 63)

            scales_out = torch.zeros((R, nblocks, 12), dtype=torch.int32,
                                     device=device)
            for j in range(8):
                if j < 4:
                    scales_out[:, :, j] = ls[:, :, j]
                    scales_out[:, :, j + 4] = lm[:, :, j]
                else:
                    scales_out[:, :, j + 4] = (ls[:, :, j] & 0x0F) | \
                                              ((lm[:, :, j] & 0x0F) << 4)
                    scales_out[:, :, j - 4] = scales_out[:, :, j - 4] | \
                                              (((ls[:, :, j] >> 4) << 6) & 0xC0)
                    scales_out[:, :, j] = scales_out[:, :, j] | \
                                          (((lm[:, :, j] >> 4) << 6) & 0xC0)
            scales_u8 = scales_out.to(torch.uint8)

            d = f32_to_f16((max_scale / 63.0).to(torch.float32).cpu().numpy())
            dmin = f32_to_f16((max_min / 63.0).to(torch.float32).cpu().numpy())
            d_t = torch.from_numpy(np.ascontiguousarray(d)).to(device)
            dmin_t = torch.from_numpy(np.ascontiguousarray(dmin)).to(device)

            d_f = torch.from_numpy(f16_to_f32(d)).to(device)
            dmin_f = torch.from_numpy(f16_to_f32(dmin)).to(device)

            sc2, m2 = _get_scale_min_k4_torch(scales_u8)
            sc2f = sc2.to(torch.float32)
            m2f = m2.to(torch.float32)
            dd = d_f[:, :, None] * sc2f
            dm = dmin_f[:, :, None] * m2f
            xr = x.reshape(R, nblocks, 8, 32)
            dd_safe = torch.where(dd == 0, torch.ones_like(dd), dd)
            l = _nearest_int_torch((xr + dm[:, :, :, None]) / dd_safe[:, :, :, None])
            l = torch.clamp(l, 0, 15)
            Lr = torch.where(dd[:, :, :, None] == 0,
                             L.reshape(R, nblocks, 8, 32),
                             l).to(torch.uint8)

            Lb = Lr.reshape(R, nblocks, 8, 32).to(torch.int32)
            q = torch.zeros((R, nblocks, 128), dtype=torch.int32, device=device)
            for pair in range(4):
                s0 = pair * 2
                q[:, :, pair * 32:(pair + 1) * 32] = Lb[:, :, s0] | \
                                                     (Lb[:, :, s0 + 1] << 4)

            blk = torch.zeros((R, nblocks, Q4_K_BLOCK_BYTES), dtype=torch.uint8,
                              device=device)
            d_u8 = d_t.view(torch.uint8).reshape(R, nblocks, 2)
            dmin_u8 = dmin_t.view(torch.uint8).reshape(R, nblocks, 2)
            blk[:, :, 0:2] = d_u8
            blk[:, :, 2:4] = dmin_u8
            blk[:, :, 4:16] = scales_u8
            blk[:, :, 16:144] = q.to(torch.uint8)

            out_cpu[r0:r1] = blk.cpu().numpy()

    return out_cpu.tobytes()
