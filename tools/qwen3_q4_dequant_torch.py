#!/usr/bin/env python3
"""Torch/CUDA dequantizer for Q4_K bytes produced by qwen3_q4_torch.

Bit-exact port of src/q3_quant.c dequant_q4 and of
qwen3_q4.dequant_q4_k_matrix, vectorized over blocks.
"""

from __future__ import annotations

import numpy as np
import torch

from qwen3_q4 import QK_K, Q4_K_BLOCK_BYTES


def _f16_bytes_to_f32(b: torch.Tensor) -> torch.Tensor:
    """b: [..., 2] uint8 little-endian half -> float32."""
    u16 = b[..., 0].to(torch.int32) | (b[..., 1].to(torch.int32) << 8)
    return u16.to(torch.uint16).view(torch.float16).to(torch.float32)


def dequant_q4_k_blocks_torch(blk: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """blk: [N, 144] uint8 -> [N, 256] float32.

    Mirrors the C loop: for each of 8 sub-blocks, scale/min come from
    scale_min_q4(index) and the nibble source advances every two sub-blocks.
    """
    if isinstance(blk, np.ndarray):
        blk = torch.from_numpy(np.ascontiguousarray(blk))
    blk = blk.to(device=device, dtype=torch.uint8).contiguous()
    N = blk.shape[0]
    assert blk.shape[1] == Q4_K_BLOCK_BYTES

    with torch.inference_mode():
        d = _f16_bytes_to_f32(blk[:, 0:2])          # [N]
        dmin = _f16_bytes_to_f32(blk[:, 2:4])       # [N]
        s = blk[:, 4:16].to(torch.int32)            # [N,12]
        qs = blk[:, 16:144].to(torch.int32)         # [N,128]

        sc = torch.zeros((N, 8), dtype=torch.int32, device=device)
        mn = torch.zeros((N, 8), dtype=torch.int32, device=device)
        for j in range(8):
            if j < 4:
                sc[:, j] = s[:, j] & 63
                mn[:, j] = s[:, j + 4] & 63
            else:
                sc[:, j] = (s[:, j + 4] & 0x0F) | ((s[:, j - 4] >> 6) << 4)
                mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)

        out = torch.empty((N, QK_K), dtype=torch.float32, device=device)
        for j in range(8):
            qslice = qs[:, (j // 2) * 32:(j // 2) * 32 + 32]
            qv = (qslice & 0x0F) if (j % 2 == 0) else (qslice >> 4)
            qvf = qv.to(torch.float32)
            d1 = (d * sc[:, j].to(torch.float32))[:, None]
            m1 = (dmin * mn[:, j].to(torch.float32))[:, None]
            out[:, j * 32:(j + 1) * 32] = d1 * qvf - m1
        return out


def dequant_q4_k_matrix_torch(q4_bytes: bytes, rows: int, ncols: int,
                              device: str = "cuda") -> torch.Tensor:
    """Q4_K row-major bytes -> [rows, ncols] float32 on device."""
    nblocks = ncols // QK_K
    arr = np.frombuffer(q4_bytes, dtype=np.uint8)
    assert arr.size == rows * nblocks * Q4_K_BLOCK_BYTES, arr.size
    blk = torch.from_numpy(arr.reshape(-1, Q4_K_BLOCK_BYTES))
    flat = dequant_q4_k_blocks_torch(blk, device=device)
    return flat.reshape(rows, ncols)
