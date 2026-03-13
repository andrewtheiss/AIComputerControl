from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from ui_vision_common.debug_tools import new_artifact_dir, ocr_overlay, grounding_overlay, write_image, write_json
from ui_vision_common.image_tools import decode_image_b64, decode_image_bytes, poly_to_box
from ui_vision_common.schemas import GroundingRequest, GroundingResponse, OCRRequest, OCRResponse

from .providers import ProviderError, build_provider


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name, default) or default).strip()


def create_app() -> FastAPI:
    model_id = _env("MODEL_ID", "omniparser")
    role = _env("MODEL_ROLE", "ocr")
    backend = _env("MODEL_BACKEND", "mock")
    remote_url = _env("REMOTE_MODEL_URL", "")
    debug_dir = _env("DEBUG_ARTIFACT_DIR", "/tmp/ui-model-debug")

    app = FastAPI(title=f"UI Model Service ({model_id})")
    provider = build_provider(model_id=model_id, role=role, backend=backend, remote_url=remote_url)

    async def _decode_ocr_request(request: Request, file: Optional[UploadFile], body: Optional[OCRRequest]):
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
            data = decode_image_b64(req.image_b64)
            return data, req
        if file is not None:
            data = await file.read()
            return decode_image_bytes(data), req or OCRRequest()
        raise HTTPException(status_code=400, detail="Provide multipart file or image_b64")

    def _save_ocr_debug(image_bgr, response: OCRResponse) -> Dict[str, str]:
        artifact_dir = new_artifact_dir(debug_dir, model_id)
        raw_path = write_image(os.path.join(artifact_dir, "input.png"), image_bgr)
        word_boxes = []
        for item in response.words:
            box = poly_to_box(item.poly)
            if box:
                word_boxes.append({"box": box, "text": item.text, "score": item.score})
        overlay_path = write_image(
            os.path.join(artifact_dir, "overlay_words.png"),
            ocr_overlay(image_bgr, word_boxes, (0, 220, 255)),
        )
        payload_path = write_json(os.path.join(artifact_dir, "response.json"), response.model_dump())
        return {"artifact_dir": artifact_dir, "input": raw_path, "overlay": overlay_path, "response": payload_path}

    def _save_grounding_debug(image_bgr, payload: GroundingRequest, response: GroundingResponse) -> Dict[str, str]:
        artifact_dir = new_artifact_dir(debug_dir, model_id)
        raw_path = write_image(os.path.join(artifact_dir, "input.png"), image_bgr)
        candidate_scores: Dict[str, Dict[str, float]] = {}
        for pred in response.predictions:
            candidate_scores.setdefault(pred.candidate_id, {})[model_id] = float(pred.p)
        final_id = response.predictions[0].candidate_id if response.predictions else ""
        overlay_path = write_image(
            os.path.join(artifact_dir, "overlay_candidates.png"),
            grounding_overlay(
                image_bgr,
                [candidate.model_dump() for candidate in payload.candidates],
                candidate_scores=candidate_scores,
                final_candidate_id=final_id,
            ),
        )
        request_path = write_json(os.path.join(artifact_dir, "request.json"), payload.model_dump())
        response_path = write_json(os.path.join(artifact_dir, "response.json"), response.model_dump())
        return {
            "artifact_dir": artifact_dir,
            "input": raw_path,
            "overlay": overlay_path,
            "request": request_path,
            "response": response_path,
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model_id": model_id, "role": role, "backend": backend}

    @app.post("/ocr", response_model=OCRResponse)
    async def ocr_endpoint(request: Request, file: UploadFile = File(None), body: OCRRequest | None = None):
        if role != "ocr":
            raise HTTPException(status_code=404, detail="This service is not configured for OCR")
        try:
            image_bgr, req = await _decode_ocr_request(request=request, file=file, body=body)
            result = provider.run_ocr(image_bgr, min_score=req.min_score)
            response = OCRResponse(
                width=int(result["width"]),
                height=int(result["height"]),
                words=list(result.get("words") or []),
                lines=list(result.get("lines") or []),
                model_id=model_id,
                backend=backend,
                latency_ms=int(result.get("latency_ms") or 0),
            )
            if req.debug:
                response.debug_artifacts = _save_ocr_debug(image_bgr, response)
            return response
        except HTTPException:
            raise
        except ProviderError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}") from exc

    @app.post("/infer", response_model=GroundingResponse)
    def infer_endpoint(payload: GroundingRequest):
        if role != "grounding":
            raise HTTPException(status_code=404, detail="This service is not configured for grounding")
        try:
            if not payload.screenshot_b64:
                raise HTTPException(status_code=400, detail="screenshot_b64 is required for grounding")
            image_bgr = decode_image_b64(payload.screenshot_b64)
            result = provider.run_grounding(payload.model_dump())
            response = GroundingResponse(
                model_id=model_id,
                backend=backend,
                predictions=list(result.get("predictions") or []),
                latency_ms=int(result.get("latency_ms") or 0),
            )
            if payload.debug:
                response.debug_artifacts = _save_grounding_debug(image_bgr, payload, response)
            return response
        except HTTPException:
            raise
        except ProviderError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Grounding inference failed: {exc}") from exc

    @app.post("/infer/debug")
    def infer_debug(payload: GroundingRequest):
        payload.debug = True
        return infer_endpoint(payload)

    @app.post("/admin/selftest")
    async def selftest() -> Dict[str, Any]:
        if role == "ocr":
            import numpy as np

            image_bgr = np.full((768, 1024, 3), 255, dtype=np.uint8)
            result = provider.run_ocr(image_bgr, min_score=0.45)
            return {"status": "ok", "model_id": model_id, "checks": {"words": len(result.get("words") or []), "lines": len(result.get("lines") or [])}}

        import numpy as np

        image_bgr = np.full((768, 1024, 3), 255, dtype=np.uint8)
        from ui_vision_common.image_tools import encode_png_b64

        payload = GroundingRequest(
            instruction="click the Sign in button in the top right",
            screenshot_b64=encode_png_b64(image_bgr),
            candidates=[
                {"id": "C1", "box": [600, 120, 720, 170], "text": "Sign in", "score": 0.91, "allowed_actions": ["click"], "role": "button"},
                {"id": "C2", "box": [420, 320, 720, 370], "text": "Continue", "score": 0.70, "allowed_actions": ["click"], "role": "button"},
            ],
        )
        result = provider.run_grounding(payload.model_dump())
        return {"status": "ok", "model_id": model_id, "checks": {"predictions": len(result.get("predictions") or [])}}

    return app


app = create_app()
