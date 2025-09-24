# app/main.py
import io, base64, cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from .schemas import OCRRequest, OCRResponse, OCRBox
from .engine_ppocr import PPOCREngine
from .postproc import to_polys_words, group_lines

app = FastAPI(title="OCR API (PP-OCRv5)")
_engine = PPOCREngine()

def _decode_to_bgr(data: bytes) -> np.ndarray:
    """Try OpenCV first; if it fails (e.g., some PNGs), fall back to Pillow."""
    if not data:
        raise HTTPException(400, "Empty image payload")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Pillow fallback
    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(415, f"Unsupported/undecodable image: {e}")

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(None),
    body: OCRRequest | None = None,
):
    # Accept either base64 JSON body or multipart file
    if body and body.image_b64:
        try:
            data = base64.b64decode(body.image_b64, validate=True)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64: {e}")
        req = body
    elif file is not None:
        data = await file.read()
        req = OCRRequest() if body is None else body
    else:
        raise HTTPException(400, "Provide multipart file or image_b64")

    img = _decode_to_bgr(data)

    h, w = img.shape[:2]
    raw = _engine.infer(img)               # unchanged
    words = to_polys_words(raw, req.min_score)
    lines = group_lines(words)

    resp = OCRResponse(width=w, height=h)
    if req.return_level in ("word","both"):
        resp.words = [
            OCRBox(
                poly=[[d["box"][0], d["box"][1]],
                      [d["box"][2], d["box"][1]],
                      [d["box"][2], d["box"][3]],
                      [d["box"][0], d["box"][3]]],
                text=d["text"],
                score=d["conf"],
            )
            for d in words
        ]
    if req.return_level in ("line","both"):
        resp.lines = [
            OCRBox(
                poly=[[d["box"][0], d["box"][1]],
                      [d["box"][2], d["box"][1]],
                      [d["box"][2], d["box"][3]],
                      [d["box"][0], d["box"][3]]],
                text=d["text"],
                score=d["conf"],
            )
            for d in lines
        ]
    return resp

@app.post("/admin/reload")
def reload_models():
    global _engine
    _engine = PPOCREngine()
    return {"status":"ok"}

@app.get("/health")
def health():
    return {"status":"ok","backend":"ppocr"}
