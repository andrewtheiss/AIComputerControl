from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class OCRBox(BaseModel):
    # 4-point polygon in image pixel coords (x,y) * 4
    poly: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str
    score: float
    lang: Optional[str] = None
    line_id: Optional[int] = None

class OCRRequest(BaseModel):
    image_b64: Optional[str] = None
    return_level: Literal["word","line","both"] = "both"
    min_score: float = 0.45
    detect_rotation: bool = False

class OCRResponse(BaseModel):
    width: int
    height: int
    words: List[OCRBox] = Field(default_factory=list)
    lines: List[OCRBox] = Field(default_factory=list)
