# agent/src/screen_signature.py
from __future__ import annotations
import numpy as np, cv2, math, hashlib
from typing import List, Dict, Any

def phash64(image_bgr: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash -> vector of 0/1 floats length 64."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    # 2D DCT
    dct = cv2.dct(img)
    # take top-left 8x8 (skip [0,0] DC is fine to keep or drop; we keep)
    dct8 = dct[:8, :8].flatten()
    med = np.median(dct8)
    bits = (dct8 > med).astype(np.float32)
    return bits  # shape (64,)

def bow256_from_ocr(ocr_items: List[Dict[str,Any]]) -> np.ndarray:
    """Hashed bag-of-words (length 256) over OCR tokens."""
    v = np.zeros(256, dtype=np.float32)
    for w in ocr_items or []:
        t = (w.get("text","") or "").strip().lower()
        if not t: continue
        # simple tokenization
        for tok in t.split():
            h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=2).hexdigest(), 16)  # 16-bit
            v[h & 0xFF] += 1.0
    # log-scale normalize
    v = np.log1p(v)
    if v.sum() > 0: v /= np.linalg.norm(v)
    return v  # (256,)

def layout_hist_4x4(ocr_items: List[Dict[str,Any]], w: int, h: int) -> np.ndarray:
    """Count words per 4x4 grid cell, normalized (length 16)."""
    bins = np.zeros((4,4), dtype=np.float32)
    if w <= 0 or h <= 0:
        return bins.flatten()
    for witem in ocr_items or []:
        x1,y1,x2,y2 = [int(v) for v in witem.get("box",[0,0,0,0])]
        cx = max(0, min(w-1, (x1+x2)//2)); cy = max(0, min(h-1, (y1+y2)//2))
        gx = min(3, int(4*cx / max(1,w))); gy = min(3, int(4*cy / max(1,h)))
        bins[gy, gx] += 1.0
    v = bins.flatten()
    if v.sum() > 0: v /= v.sum()
    return v  # (16,)

def make_signature(image_bgr: np.ndarray, ocr_items: List[Dict[str,Any]]) -> np.ndarray:
    """Final signature vector ~ 64 + 256 + 16 = 336 dims, L2 normalized."""
    h, w = image_bgr.shape[:2]
    v1 = phash64(image_bgr)              # 64
    v2 = bow256_from_ocr(ocr_items)      # 256
    v3 = layout_hist_4x4(ocr_items, w, h)# 16
    vec = np.concatenate([v1, v2, v3]).astype(np.float32)
    # normalize
    n = np.linalg.norm(vec)
    return vec / max(n, 1e-6)
