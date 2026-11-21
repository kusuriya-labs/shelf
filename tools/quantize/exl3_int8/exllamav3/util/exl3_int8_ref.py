"""NumPy reference for EXL3-INT8 (doc/exl3_int8.md): mul1 trellis -> int8 grid.

Independent of the HIP extension so that other runtimes (shelf) can use it as a
golden source. Layout facts (verified against ext.reconstruct, see
tests/test_exl3_int8_fixed_grid.py):

* trellis[kt, nt, :] is uint16[16*K]; a 16x16 tile (rows = input index k,
  columns = output index n) of 256 weights, K bits each, as a circular bit
  stream, MSB-first inside every little-endian uint32 word.
* weight t of a tile has the state = 16 bits starting at
  (t*K + K - 16 + 256*K) mod (256*K).
* mul1: x = state * 0x83DCD12D (uint32), B = sum of the 4 bytes of x,
  q = (B - 509) >> 2 (int8 grid, value = q * S_FIX).
* mul24 (24-bit multiplier 0xDCD12D) places t at row = 8 * (t // 128) + t % 8, col = (t // 8) % 16 instead of
  the tensor-core placement below. It is selected by the multiplier (0xDCD12D), see research/codebook.
* tensor-core placement of t inside the tile: lane = t // 8, j = t % 8,
  row = (lane % 4) * 2 + (j & 1) + 8 * ((j >> 1) & 1), col = lane // 4 + 8 * (j >> 2).
"""
import numpy as np

S_FIX = 4.0 * 887.0 / 131072.0


MUL1 = 0x83DCD12D
MUL24 = 0xDCD12D


def tile_positions(mult=MUL1):
    t = np.arange(256)
    lane, j = t // 8, t % 8
    if mult == MUL24:
        return 8 * (lane // 16) + j, lane % 16
    row = (lane % 4) * 2 + (j & 1) + 8 * ((j >> 1) & 1)
    col = lane // 4 + 8 * (j >> 2)
    return row, col


def states(trellis: np.ndarray) -> np.ndarray:
    """trellis uint16[Kt, Nt, 16*K] -> uint16 states [Kt, Nt, 256]"""
    kt, nt, n16 = trellis.shape
    k = n16 // 16
    words = np.ascontiguousarray(trellis.astype(np.uint16)).view(np.uint32).reshape(kt, nt, 8 * k).astype(np.uint64)
    nbits = 256 * k
    t = np.arange(256)
    p0 = (t * k + k - 16 + nbits) % nbits
    wi, off = p0 // 32, p0 % 32
    hi = words[:, :, wi]
    lo = words[:, :, (wi + 1) % (8 * k)]
    merged = (hi << np.uint64(32)) | lo
    shift = (64 - off - 16).astype(np.uint64)
    return ((merged >> shift) & np.uint64(0xFFFF)).astype(np.uint32)


def byte_sum(state: np.ndarray, mult=MUL1) -> np.ndarray:
    x = (state.astype(np.uint64) * np.uint64(mult)) & np.uint64(0xFFFFFFFF)
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)


def decode_q(trellis: np.ndarray, chunk: int = 8, mult=MUL1) -> np.ndarray:
    """trellis uint16[Kt, Nt, 16*K] -> int8 q [out_features, in_features] (row = output n, col = input k)"""
    kt, nt, _ = trellis.shape
    row, col = tile_positions(mult)
    out = np.zeros((nt, 16, kt, 16), dtype=np.int8)
    for k0 in range(0, kt, chunk):
        b = byte_sum(states(trellis[k0:k0 + chunk]), mult).astype(np.int32)
        q = ((b - 509) >> 2).astype(np.int8)
        for t in range(256):
            out[:, col[t], k0:k0 + q.shape[0], row[t]] = q[:, :, t].T
    return out.reshape(nt * 16, kt * 16)
