"""
Tier 3: Cryptographic Block Permutation (AES Block Scrambling).

The ROI is divided into non-overlapping B x B pixel blocks. A pseudorandom
permutation of block indices is generated using AES-128 in CTR mode. The
blocks are spatially rearranged according to this permutation. Pixel values
within each block are preserved; only spatial arrangement is altered.

Nonce derivation:
    A single global key K is fixed across the dataset. The AES-CTR nonce is
    derived by computing SHA-256 over the concatenation of video_id, frame_idx,
    and block_size, then truncating to 8 bytes. This produces a unique
    permutation for every combination of video, frame, and block size.

    The permutation is recomputed independently for each frame. No two frames
    within a clip share the same block arrangement, eliminating temporal block
    correspondence.

Fallback:
    If pycryptodome is not installed, a SHA-256-based CSPRNG fallback is used.
    The fallback produces DIFFERENT permutations from the AES-CTR path. The
    distributed dataset was generated exclusively using AES-CTR. Install
    pycryptodome to reproduce byte-identical outputs.
"""

import hashlib

import cv2
import numpy as np

from .utils import clamp_bbox

# Try to import pycryptodome for canonical AES-CTR output.
try:
    from Crypto.Cipher import AES as _AES
    from Crypto.Util import Counter as _Counter
    _USE_AES = True
except ImportError:
    try:
        from Cryptodome.Cipher import AES as _AES
        from Cryptodome.Util import Counter as _Counter
        _USE_AES = True
    except ImportError:
        _USE_AES = False


def _sha256_csprng(key_hex, nonce, n_bytes):
    """SHA-256-based CSPRNG fallback when pycryptodome is unavailable."""
    key = bytes.fromhex(key_hex)
    out = bytearray()
    ctr = 0
    while len(out) < n_bytes:
        out.extend(hashlib.sha256(key + nonce + ctr.to_bytes(8, "big")).digest())
        ctr += 1
    return bytes(out[:n_bytes])


def _get_permutation(key_hex, video_id, frame_idx, block_size, n_blocks):
    """
    Generate a pseudorandom permutation of n_blocks indices.

    The permutation is unique per (video_id, frame_idx, block_size) triple.
    Uses Fisher-Yates shuffle seeded by AES-CTR keystream (or SHA-256 fallback).

    Args:
        key_hex: Hex-encoded AES-128 key.
        video_id: Video identifier string.
        frame_idx: Frame index within the clip.
        block_size: Block size B.
        n_blocks: Number of blocks to permute.

    Returns:
        numpy array of permuted indices.
    """
    nonce_input = f"{video_id}_{frame_idx}_{block_size}".encode()
    nonce = hashlib.sha256(nonce_input).digest()[:8]
    n_bytes = n_blocks * 4

    if _USE_AES:
        ctr = _Counter.new(64, prefix=nonce, initial_value=0)
        cipher = _AES.new(bytes.fromhex(key_hex), _AES.MODE_CTR, counter=ctr)
        ks = cipher.encrypt(b"\x00" * n_bytes)
    else:
        ks = _sha256_csprng(key_hex, nonce, n_bytes)

    idx = np.arange(n_blocks)
    for i in range(n_blocks - 1, 0, -1):
        off = (n_blocks - 1 - i) * 4
        j = int.from_bytes(ks[off:off + 4], "big") % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]

    return idx


def apply_scramble(frame, bbox, block_size, key_hex, video_id, frame_idx):
    """
    Apply AES block permutation scrambling to the ROI.

    Args:
        frame: Input BGR frame (H, W, 3).
        bbox: Bounding box [x1, y1, x2, y2] or None.
        block_size: Block size B (e.g., 4, 8, 16).
        key_hex: Hex-encoded AES-128 key.
        video_id: Video identifier string.
        frame_idx: Frame index within the clip.

    Returns:
        Scrambled frame (copy; input is not modified).
    """
    if bbox is None:
        return frame.copy()

    out = frame.copy()
    x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return out

    roi = out[y1:y2, x1:x2].copy()
    hb = roi.shape[0] // block_size
    wb = roi.shape[1] // block_size
    if hb == 0 or wb == 0:
        return out

    ht = hb * block_size
    wt = wb * block_size
    n = hb * wb

    perm = _get_permutation(key_hex, video_id, frame_idx, block_size, n)

    trimmed = roi[:ht, :wt]
    c = trimmed.shape[2]
    blocks = trimmed.reshape(hb, block_size, wb, block_size, c)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(-1, block_size, block_size, c)
    shuffled = blocks[perm].reshape(hb, wb, block_size, block_size, c)
    shuffled = shuffled.transpose(0, 2, 1, 3, 4).reshape(ht, wt, c)

    out[y1:y1 + ht, x1:x1 + wt] = shuffled
    return out


def is_aes_available():
    """Return True if pycryptodome is installed (canonical AES-CTR path)."""
    return _USE_AES
