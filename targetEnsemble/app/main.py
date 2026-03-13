from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import requests
from fastapi import FastAPI

from ui_vision_common.debug_tools import grounding_overlay, new_artifact_dir, write_image, write_json
from ui_vision_common.image_tools import decode_image_b64
from ui_vision_common.mock_backends import mock_grounding_result
from ui_vision_common.schemas import GroundingRequest


DEFAULT_MODEL_URLS = {
    "groundnext": "mock://groundnext",
    "aria-ui": "mock://aria-ui",
    "phi-ground": "mock://phi-ground",
}

DEFAULT_WEIGHTS = {
    "groundnext": 0.45,
    "aria-ui": 0.25,
    "phi-ground": 0.20,
    "candidate_prior": 0.10,
}


def _load_json_env(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return dict(default)
    return {str(key): value for key, value in json.loads(raw).items()}


def _candidate_prior(candidate: Dict[str, Any]) -> float:
    extras = dict(candidate.get("extras") or {})
    score = float(candidate.get("score", 0.0))
    interactable_score = float(extras.get("interactable_score", score) or score)
    ocr_consensus = float(extras.get("ocr_consensus", min(1.0, score)) or 0.0)
    action_compatibility = float(
        extras.get(
            "action_compatibility",
            1.0 if "click" in (candidate.get("allowed_actions") or []) else 0.5,
        )
        or 0.0
    )
    allowed_actions = candidate.get("allowed_actions") or []
    role = str(candidate.get("role", "") or "")
    prior = 0.50 * interactable_score + 0.30 * ocr_consensus + 0.20 * action_compatibility
    bonus = 0.04 if "click" in allowed_actions else 0.0
    if role in ("button", "link", "textbox"):
        bonus += 0.03
    return min(1.0, max(0.0, prior + bonus))


def _call_model(session: requests.Session, model_id: str, model_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if model_url.startswith("mock://"):
        image_bgr = decode_image_b64(str(payload.get("screenshot_b64", "") or ""))
        _ = image_bgr  # ensures invalid base64 fails consistently
        return mock_grounding_result(
            model_id=model_id,
            instruction=str(payload.get("instruction", "") or ""),
            candidates=list(payload.get("candidates") or []),
            history=list(payload.get("history") or []),
            top_k=int(payload.get("top_k") or 5),
        )
    resp = session.post(model_url.rstrip("/") + "/infer" if not model_url.rstrip("/").endswith("/infer") else model_url, json=payload, timeout=(3.0, 30.0))
    resp.raise_for_status()
    return resp.json()


def create_app() -> FastAPI:
    debug_dir = str(os.environ.get("DEBUG_ARTIFACT_DIR", "/tmp/target-ensemble-debug") or "/tmp/target-ensemble-debug")
    model_urls = _load_json_env("TARGET_ENSEMBLE_MODEL_URLS", DEFAULT_MODEL_URLS)
    weights = _load_json_env("TARGET_ENSEMBLE_WEIGHTS", DEFAULT_WEIGHTS)
    session = requests.Session()
    app = FastAPI(title="Target Ensemble API")

    def _artifacts(image_bgr, response: Dict[str, Any], payload: GroundingRequest) -> Dict[str, str]:
        artifact_dir = new_artifact_dir(debug_dir, "target-ensemble")
        raw_path = write_image(os.path.join(artifact_dir, "input.png"), image_bgr)
        overlay_path = write_image(
            os.path.join(artifact_dir, "overlay_votes.png"),
            grounding_overlay(
                image_bgr,
                [candidate.model_dump() for candidate in payload.candidates],
                candidate_scores=response.get("candidate_scores", {}),
                final_candidate_id=(response.get("final_prediction") or {}).get("candidate_id", ""),
            ),
        )
        request_path = write_json(os.path.join(artifact_dir, "request.json"), payload.model_dump())
        response_path = write_json(os.path.join(artifact_dir, "response.json"), response)
        return {"artifact_dir": artifact_dir, "input": raw_path, "overlay": overlay_path, "request": request_path, "response": response_path}

    def _run(payload: GroundingRequest, force_debug: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        image_bgr = decode_image_b64(payload.screenshot_b64 or "")
        per_model: Dict[str, Any] = {}
        candidate_scores: Dict[str, Dict[str, float]] = {
            candidate.id: {"candidate_prior": _candidate_prior(candidate.model_dump())}
            for candidate in payload.candidates
        }

        for model_id, model_url in model_urls.items():
            try:
                result = _call_model(session, model_id=model_id, model_url=model_url, payload=payload.model_dump())
                per_model[model_id] = result
                for pred in result.get("predictions", []):
                    cid = str(pred.get("candidate_id", ""))
                    candidate_scores.setdefault(cid, {"candidate_prior": 0.0})[model_id] = float(pred.get("p", 0.0))
            except Exception as exc:
                per_model[model_id] = {"status": "error", "error": str(exc), "predictions": []}

        ranked = []
        for candidate in payload.candidates:
            cid = candidate.id
            parts = candidate_scores.get(cid, {"candidate_prior": 0.0})
            final_score = 0.0
            final_score += weights.get("candidate_prior", 0.0) * float(parts.get("candidate_prior", 0.0))
            for model_id in model_urls.keys():
                final_score += float(weights.get(model_id, 0.0)) * float(parts.get(model_id, 0.0))
            ranked.append(
                {
                    "candidate_id": cid,
                    "text": candidate.text,
                    "action": "click" if "click" in (candidate.allowed_actions or ["click"]) else "focus",
                    "score": round(final_score, 6),
                    "box": candidate.box,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)

        response = {
            "model_id": "target-ensemble",
            "backend": "fanout",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "final_prediction": ranked[0] if ranked else None,
            "ranked_candidates": ranked[: max(1, payload.top_k)],
            "candidate_scores": candidate_scores,
            "weights": weights,
            "per_model": per_model if payload.debug or force_debug else {key: {"status": "ok"} for key in model_urls.keys()},
        }
        if payload.debug or force_debug:
            response["debug_artifacts"] = _artifacts(image_bgr, response, payload)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": "target-ensemble", "models": list(model_urls.keys())}

    @app.post("/infer")
    def infer(payload: GroundingRequest):
        return _run(payload, force_debug=False)

    @app.post("/infer/debug")
    def infer_debug(payload: GroundingRequest):
        payload.debug = True
        return _run(payload, force_debug=True)

    @app.post("/admin/selftest")
    def selftest():
        import numpy as np
        from ui_vision_common.image_tools import encode_png_b64

        image_bgr = np.full((768, 1024, 3), 255, dtype=np.uint8)
        payload = GroundingRequest(
            instruction="click the Sign in button in the top right",
            screenshot_b64=encode_png_b64(image_bgr),
            debug=True,
            candidates=[
                {"id": "C1", "box": [700, 120, 820, 170], "text": "Sign in", "score": 0.96, "role": "button", "allowed_actions": ["click"]},
                {"id": "C2", "box": [430, 360, 640, 410], "text": "Continue", "score": 0.75, "role": "button", "allowed_actions": ["click"]},
            ],
        )
        return _run(payload, force_debug=True)

    return app


app = create_app()
