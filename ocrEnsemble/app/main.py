from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from ui_vision_common.debug_tools import new_artifact_dir, ocr_overlay, write_image, write_json
from ui_vision_common.image_tools import box_to_poly, decode_image_b64, decode_image_bytes, encode_png_b64, poly_to_box
from ui_vision_common.mock_backends import mock_ocr_result
from ui_vision_common.schemas import OCRRequest, OCRResponse


DEFAULT_MODEL_URLS = {
    "ppocr": "mock://ppocr",
    "omniparser": "mock://omniparser",
    "paddleocr-vl": "mock://paddleocr-vl",
    "surya": "mock://surya",
}


def _load_model_urls() -> Dict[str, str]:
    raw = str(os.environ.get("OCR_ENSEMBLE_MODEL_URLS", "") or "").strip()
    if not raw:
        return dict(DEFAULT_MODEL_URLS)
    return {str(key): str(value) for key, value in json.loads(raw).items()}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _center_distance(a: List[int], b: List[int]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _merge_level(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for item in items:
        box = poly_to_box(item.get("poly") or [])
        if not box:
            continue
        text_key = _normalize_text(str(item.get("text", "") or ""))
        score = float(item.get("score", 0.0))
        matched = None
        for group in groups:
            if group["text_key"] != text_key:
                continue
            if _iou(box, group["box"]) >= 0.10 or _center_distance(box, group["box"]) <= 40.0:
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "text_key": text_key,
                    "text": item.get("text", ""),
                    "box": box,
                    "scores": [score],
                    "models": [item.get("source_model", "")],
                }
            )
            continue
        matched["box"] = [
            min(matched["box"][0], box[0]),
            min(matched["box"][1], box[1]),
            max(matched["box"][2], box[2]),
            max(matched["box"][3], box[3]),
        ]
        matched["scores"].append(score)
        matched["models"].append(item.get("source_model", ""))

    merged = []
    for group in groups:
        merged.append(
            {
                "poly": box_to_poly(group["box"]),
                "text": group["text"],
                "score": round(sum(group["scores"]) / max(1, len(group["scores"])), 4),
                "source_model": ",".join(sorted(set(filter(None, group["models"])))),
            }
        )
    merged.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("text", ""))))
    return merged


def _call_model(session: requests.Session, model_id: str, model_url: str, image_b64: str, min_score: float) -> Dict[str, Any]:
    if model_url.startswith("mock://"):
        image_bgr = decode_image_b64(image_b64)
        return mock_ocr_result(model_id=model_id, image_bgr=image_bgr, min_score=min_score)
    resp = session.post(
        model_url.rstrip("/") + "/ocr" if not model_url.rstrip("/").endswith("/ocr") else model_url,
        json={"image_b64": image_b64, "return_level": "both", "min_score": min_score},
        timeout=(3.0, 30.0),
    )
    resp.raise_for_status()
    return resp.json()


def create_app() -> FastAPI:
    debug_dir = str(os.environ.get("DEBUG_ARTIFACT_DIR", "/tmp/ocr-ensemble-debug") or "/tmp/ocr-ensemble-debug")
    model_urls = _load_model_urls()
    session = requests.Session()
    app = FastAPI(title="OCR Ensemble API")

    async def _decode_request(request: Request, file: Optional[UploadFile], body: Optional[OCRRequest]):
        req = body
        if req is None:
            content_type = str(request.headers.get("content-type", "") or "").lower()
            if "application/json" in content_type:
                try:
                    raw_body = await request.json()
                except Exception:
                    raw_body = None
                if raw_body:
                    req = OCRRequest.model_validate(raw_body)
        if req and req.image_b64:
            return decode_image_b64(req.image_b64), req
        if file is not None:
            return decode_image_bytes(await file.read()), req or OCRRequest()
        raise HTTPException(status_code=400, detail="Provide multipart file or image_b64")

    def _artifact_bundle(image_bgr, response: Dict[str, Any], per_model: Dict[str, Any]) -> Dict[str, str]:
        artifact_dir = new_artifact_dir(debug_dir, "ocr-ensemble")
        raw_path = write_image(os.path.join(artifact_dir, "input.png"), image_bgr)
        merged_words = []
        for item in response.get("words", []):
            box = poly_to_box(item.get("poly") or [])
            if box:
                merged_words.append({"box": box, "text": item.get("text", ""), "score": item.get("score", 0.0)})
        overlay_path = write_image(
            os.path.join(artifact_dir, "overlay_merged_words.png"),
            ocr_overlay(image_bgr, merged_words, (255, 255, 255)),
        )
        per_model_path = write_json(os.path.join(artifact_dir, "per_model.json"), per_model)
        response_path = write_json(os.path.join(artifact_dir, "response.json"), response)
        return {"artifact_dir": artifact_dir, "input": raw_path, "overlay": overlay_path, "per_model": per_model_path, "response": response_path}

    def _run(image_bgr, req: OCRRequest, force_debug: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        image_b64 = encode_png_b64(image_bgr)
        per_model: Dict[str, Any] = {}
        all_words: List[Dict[str, Any]] = []
        all_lines: List[Dict[str, Any]] = []
        for model_id, model_url in model_urls.items():
            try:
                result = _call_model(session, model_id=model_id, model_url=model_url, image_b64=image_b64, min_score=req.min_score)
                per_model[model_id] = {
                    "status": "ok",
                    "latency_ms": int(result.get("latency_ms") or 0),
                    "words": list(result.get("words") or []),
                    "lines": list(result.get("lines") or []),
                }
                for item in per_model[model_id]["words"]:
                    item["source_model"] = model_id
                for item in per_model[model_id]["lines"]:
                    item["source_model"] = model_id
                all_words.extend(per_model[model_id]["words"])
                all_lines.extend(per_model[model_id]["lines"])
            except Exception as exc:
                per_model[model_id] = {"status": "error", "error": str(exc), "words": [], "lines": []}

        height, width = image_bgr.shape[:2]
        response = OCRResponse(
            width=width,
            height=height,
            words=_merge_level(all_words),
            lines=_merge_level(all_lines),
            model_id="ocr-ensemble",
            backend="fanout",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            meta={"models": list(model_urls.keys()), "per_model_status": {key: value["status"] for key, value in per_model.items()}},
        ).model_dump()
        if req.return_level == "word":
            response["lines"] = []
        elif req.return_level == "line":
            response["words"] = []
        if req.debug or force_debug:
            response["debug_artifacts"] = _artifact_bundle(image_bgr, response, per_model)
            response["meta"]["per_model"] = per_model
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": "ocr-ensemble", "models": list(model_urls.keys())}

    @app.post("/ocr")
    async def ocr_endpoint(request: Request, file: UploadFile = File(None), body: OCRRequest | None = None):
        image_bgr, req = await _decode_request(request=request, file=file, body=body)
        return _run(image_bgr=image_bgr, req=req, force_debug=False)

    @app.post("/infer/debug")
    async def debug_endpoint(request: Request, file: UploadFile = File(None), body: OCRRequest | None = None):
        image_bgr, req = await _decode_request(request=request, file=file, body=body)
        req.debug = True
        return _run(image_bgr=image_bgr, req=req, force_debug=True)

    @app.post("/admin/selftest")
    def selftest():
        import numpy as np

        image_bgr = np.full((768, 1024, 3), 255, dtype=np.uint8)
        return _run(image_bgr=image_bgr, req=OCRRequest(debug=True), force_debug=True)

    return app


app = create_app()
