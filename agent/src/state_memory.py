# agent/src/state_memory.py
from __future__ import annotations
import os, json, numpy as np, time
from typing import Dict, Any, List, Tuple, Optional

class StateMemory:
    """
    Tiny in-process store of screen state prototypes.
    Each item: {"id": str, "vec": List[float], "tags": [..], "created": ts}
    """
    def __init__(self, path: str, max_items: int = 200):
        self.path = path
        self.max_items = max_items
        self.items: List[Dict[str,Any]] = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.items = data.get("items", [])
        except Exception:
            self.items = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"items": self.items}, f)
        except Exception:
            pass

    def register(self, state_id: str, vec: np.ndarray, tags: Optional[List[str]]=None):
        entry = {"id": state_id, "vec": vec.tolist(), "tags": tags or [], "created": time.time()}
        self.items.append(entry)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self._save()

    def match(self, vec: np.ndarray, topk: int = 3) -> List[Tuple[str, float, List[str]]]:
        out: List[Tuple[str,float,List[str]]] = []
        if not self.items:
            return out
        v = vec.astype(np.float32)
        for it in self.items:
            u = np.asarray(it["vec"], dtype=np.float32)
            # cosine similarity
            sim = float(np.dot(u, v) / max(1e-6, (np.linalg.norm(u)*np.linalg.norm(v))))
            out.append((it["id"], sim, it.get("tags", [])))
        out.sort(key=lambda x: -x[1])
        return out[:topk]
