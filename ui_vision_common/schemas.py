from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OCRBox(BaseModel):
    poly: List[List[float]]
    text: str
    score: float
    lang: Optional[str] = None
    line_id: Optional[int] = None
    source_model: Optional[str] = None


class OCRRequest(BaseModel):
    image_b64: Optional[str] = None
    return_level: Literal["word", "line", "both"] = "both"
    min_score: float = 0.45
    detect_rotation: bool = False
    debug: bool = False


class OCRResponse(BaseModel):
    width: int
    height: int
    words: List[OCRBox] = Field(default_factory=list)
    lines: List[OCRBox] = Field(default_factory=list)
    model_id: Optional[str] = None
    backend: Optional[str] = None
    latency_ms: Optional[int] = None
    debug_artifacts: Dict[str, str] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    id: str
    box: List[int]
    text: str = ""
    score: float = 0.0
    role: Optional[str] = None
    source: Optional[str] = None
    allowed_actions: List[str] = Field(default_factory=list)
    extras: Dict[str, Any] = Field(default_factory=dict)


class GroundingRequest(BaseModel):
    instruction: str
    history: List[str] = Field(default_factory=list)
    screenshot_b64: Optional[str] = None
    screenshot_mime: str = "image/png"
    candidates: List[Candidate] = Field(default_factory=list)
    top_k: int = 5
    mode: Literal["rank_candidates"] = "rank_candidates"
    debug: bool = False


class RankedPrediction(BaseModel):
    candidate_id: str
    action: str
    p: float
    raw_box: Optional[List[int]] = None
    raw_point: Optional[List[int]] = None
    rationale: Optional[str] = None


class GroundingResponse(BaseModel):
    model_id: str
    backend: str
    predictions: List[RankedPrediction] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    debug_artifacts: Dict[str, str] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
