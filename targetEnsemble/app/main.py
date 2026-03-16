from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests
from fastapi import FastAPI

from ui_vision_common.debug_tools import grounding_overlay, new_artifact_dir, write_image, write_json
from ui_vision_common.image_tools import decode_image_b64, encode_png_b64
from ui_vision_common.mock_backends import mock_grounding_result
from ui_vision_common.schemas import Candidate, GroundingRequest


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

DEFAULT_GATING = {
    "top1_threshold": 0.72,
    "margin_threshold": 0.12,
    "min_agreeing_models": 2,
    "cluster_iou_threshold": 0.50,
    "repair_top_n": 3,
    "repair_score_window": 0.18,
    "repair_padding_px": 24,
}

DEFAULT_VALIDATOR = {
    "min_top1_score": 0.70,
    "min_score_margin": 0.08,
    "min_candidate_prior": 0.60,
    "min_ocr_consensus": 0.50,
    "min_agreeing_models": 2,
}


def _load_json_env(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return dict(default)
    return {str(key): value for key, value in json.loads(raw).items()}


def _box_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
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
    return inter / float(max(1, a_area + b_area - inter))


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


def _select_repair_candidates(ranked: List[Dict[str, Any]], gating: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ranked:
        return []
    top1_score = float(ranked[0].get("score", 0.0) or 0.0)
    window = float(gating.get("repair_score_window", 0.18) or 0.18)
    limit = max(1, int(gating.get("repair_top_n", 3) or 3))
    selected = [ranked[0]]
    for item in ranked[1:]:
        if len(selected) >= limit:
            break
        if top1_score - float(item.get("score", 0.0) or 0.0) <= window:
            selected.append(item)
    return selected


def _repair_crop_box(boxes: List[List[int]], width: int, height: int, padding_px: int) -> List[int]:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad = max(int(padding_px or 0), int(0.15 * max(x2 - x1, y2 - y1, 1)))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    ]


def _rebase_candidate_for_crop(candidate: Candidate, crop_box: List[int]) -> Candidate:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    box = [int(v) for v in candidate.box]
    local_box = [box[0] - crop_x1, box[1] - crop_y1, box[2] - crop_x1, box[3] - crop_y1]
    extras = dict(candidate.extras or {})
    extras["repair_original_box"] = box
    extras["repair_crop_box"] = list(crop_box)
    extras["bbox_rel_1000"] = [
        int(round(local_box[0] * 1000.0 / crop_w)),
        int(round(local_box[1] * 1000.0 / crop_h)),
        int(round(local_box[2] * 1000.0 / crop_w)),
        int(round(local_box[3] * 1000.0 / crop_h)),
    ]
    extras["center_abs"] = [
        int(round((local_box[0] + local_box[2]) / 2.0)),
        int(round((local_box[1] + local_box[3]) / 2.0)),
    ]
    return candidate.model_copy(update={"box": local_box, "extras": extras})


def _restore_boxes(rows: List[Dict[str, Any]], candidates: Dict[str, Candidate]) -> List[Dict[str, Any]]:
    restored: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        candidate = candidates.get(str(entry.get("candidate_id", "")))
        if candidate is not None:
            entry["box"] = [int(v) for v in candidate.box]
        restored.append(entry)
    return restored


def create_app() -> FastAPI:
    debug_dir = str(os.environ.get("DEBUG_ARTIFACT_DIR", "/tmp/target-ensemble-debug") or "/tmp/target-ensemble-debug")
    model_urls = _load_json_env("TARGET_ENSEMBLE_MODEL_URLS", DEFAULT_MODEL_URLS)
    weights = _load_json_env("TARGET_ENSEMBLE_WEIGHTS", DEFAULT_WEIGHTS)
    gating = _load_json_env("TARGET_ENSEMBLE_GATING", DEFAULT_GATING)
    final_validator = _load_json_env("TARGET_ENSEMBLE_VALIDATOR", DEFAULT_VALIDATOR)
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

    def _score_payload(payload: GroundingRequest) -> Dict[str, Any]:
        per_model: Dict[str, Any] = {}
        per_model_top1: Dict[str, Dict[str, Any]] = {}
        candidate_scores: Dict[str, Dict[str, float]] = {
            candidate.id: {"candidate_prior": _candidate_prior(candidate.model_dump())}
            for candidate in payload.candidates
        }

        for model_id, model_url in model_urls.items():
            try:
                result = _call_model(session, model_id=model_id, model_url=model_url, payload=payload.model_dump())
                per_model[model_id] = result
                top_preds = list(result.get("predictions") or [])
                if top_preds:
                    per_model_top1[model_id] = dict(top_preds[0])
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
        return {
            "per_model": per_model,
            "per_model_top1": per_model_top1,
            "candidate_scores": candidate_scores,
            "ranked": ranked,
        }

    def _resolve_ranking(payload: GroundingRequest, ranked: List[Dict[str, Any]], per_model_top1: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        candidate_map = {candidate.id: candidate for candidate in payload.candidates}
        top1 = ranked[0] if ranked else None
        top2 = ranked[1] if len(ranked) > 1 else None
        top1_score = float((top1 or {}).get("score", 0.0) or 0.0)
        top2_score = float((top2 or {}).get("score", 0.0) or 0.0)
        score_margin = round(top1_score - top2_score, 6) if top1 else 0.0

        cluster_ids: List[str] = []
        agreeing_models: List[str] = []
        if top1:
            top1_box = list(top1.get("box") or [0, 0, 0, 0])
            cluster_ids = [
                candidate.id
                for candidate in payload.candidates
                if candidate.id == top1["candidate_id"] or _box_iou(candidate.box, top1_box) >= float(gating.get("cluster_iou_threshold", 0.50))
            ]
            for model_id, pred in per_model_top1.items():
                if str(pred.get("candidate_id", "")) in cluster_ids:
                    agreeing_models.append(model_id)

        action_allowed = False
        if top1 and top1["candidate_id"] in candidate_map:
            chosen = candidate_map[top1["candidate_id"]]
            action_allowed = str(top1.get("action", "")) in list(chosen.allowed_actions or ["click"])

        reason_codes: List[str] = []
        if not top1:
            reason_codes.append("no_candidates")
        else:
            if top1_score < float(gating.get("top1_threshold", 0.72)):
                reason_codes.append("score_below_threshold")
            if score_margin < float(gating.get("margin_threshold", 0.12)):
                reason_codes.append("margin_below_threshold")
            if len(agreeing_models) < int(gating.get("min_agreeing_models", 2)):
                reason_codes.append("insufficient_model_agreement")
            if not action_allowed:
                reason_codes.append("action_not_allowed")

        auto_execute = bool(top1 and not reason_codes)
        repair_candidates = _select_repair_candidates(ranked, gating)
        repair_plan = None
        if not auto_execute:
            repair_plan = {
                "status": "needs_repair",
                "strategy": "zoom_rerank_top_candidates",
                "candidate_ids": [item["candidate_id"] for item in repair_candidates],
                "boxes": [item["box"] for item in repair_candidates],
                "reason_codes": reason_codes,
            }
        return {
            "top1": top1,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": score_margin,
            "cluster_ids": cluster_ids,
            "agreeing_models": agreeing_models,
            "action_allowed": action_allowed,
            "reason_codes": reason_codes,
            "auto_execute": auto_execute,
            "repair_candidates": repair_candidates,
            "repair_plan": repair_plan,
            "gating": {
                "top1_threshold": float(gating.get("top1_threshold", 0.72)),
                "margin_threshold": float(gating.get("margin_threshold", 0.12)),
                "min_agreeing_models": int(gating.get("min_agreeing_models", 2)),
                "cluster_iou_threshold": float(gating.get("cluster_iou_threshold", 0.50)),
                "top1_score": top1_score,
                "top2_score": top2_score,
                "score_margin": score_margin,
                "agreeing_models": agreeing_models,
                "agreeing_model_count": len(agreeing_models),
                "cluster_candidate_ids": cluster_ids,
                "action_allowed": action_allowed,
                "reason_codes": reason_codes,
            },
        }

    def _run_repair_rerun(
        payload: GroundingRequest,
        image_bgr,
        repair_candidates: List[Dict[str, Any]],
        expose_per_model: bool,
    ) -> Dict[str, Any] | None:
        if not repair_candidates:
            return None
        candidate_map = {candidate.id: candidate for candidate in payload.candidates}
        selected = [candidate_map[item["candidate_id"]] for item in repair_candidates if item["candidate_id"] in candidate_map]
        if not selected:
            return None
        crop_box = _repair_crop_box(
            boxes=[[int(v) for v in candidate.box] for candidate in selected],
            width=int(image_bgr.shape[1]),
            height=int(image_bgr.shape[0]),
            padding_px=int(gating.get("repair_padding_px", 24) or 24),
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        rerun_payload = payload.model_copy(
            update={
                "screenshot_b64": encode_png_b64(crop_bgr),
                "screenshot_mime": "image/png",
                "candidates": [_rebase_candidate_for_crop(candidate, crop_box) for candidate in selected],
                "top_k": min(max(1, payload.top_k), len(selected)),
            }
        )
        scored = _score_payload(rerun_payload)
        resolution = _resolve_ranking(rerun_payload, scored["ranked"], scored["per_model_top1"])
        rerun_ranked = _restore_boxes(scored["ranked"], candidate_map)
        rerun_top1 = rerun_ranked[0] if rerun_ranked else None
        return {
            "status": "performed",
            "candidate_ids": [candidate.id for candidate in selected],
            "crop_box": crop_box,
            "crop_size": [max(1, crop_x2 - crop_x1), max(1, crop_y2 - crop_y1)],
            "candidate_scores": scored["candidate_scores"],
            "gating": resolution["gating"],
            "auto_execute": resolution["auto_execute"],
            "promoted": resolution["auto_execute"],
            "final_prediction": rerun_top1,
            "ranked_candidates": rerun_ranked,
            "per_model": scored["per_model"] if expose_per_model else {key: {"status": "ok"} for key in model_urls.keys()},
        }

    def _run_final_validator(
        payload: GroundingRequest,
        final_prediction: Dict[str, Any] | None,
        final_gating: Dict[str, Any],
        candidate_scores: Dict[str, Dict[str, float]],
        per_model: Dict[str, Any],
        stage: str,
        repair_candidates: List[Dict[str, Any]],
        repair_rerun: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        candidate_map = {candidate.id: candidate for candidate in payload.candidates}
        thresholds = {
            "min_top1_score": float(final_validator.get("min_top1_score", 0.70)),
            "min_score_margin": float(final_validator.get("min_score_margin", 0.08)),
            "min_candidate_prior": float(final_validator.get("min_candidate_prior", 0.60)),
            "min_ocr_consensus": float(final_validator.get("min_ocr_consensus", 0.50)),
            "min_agreeing_models": int(final_validator.get("min_agreeing_models", 2)),
        }
        validator_result = {
            "status": "skipped",
            "passed": False,
            "stage": stage,
            "candidate_id": str((final_prediction or {}).get("candidate_id", "") or ""),
            "reason_codes": [],
            "thresholds": thresholds,
            "checks": {},
        }
        if not final_prediction:
            validator_result["reason_codes"] = ["no_final_prediction"]
            return validator_result

        candidate_id = str(final_prediction.get("candidate_id", "") or "")
        chosen = candidate_map.get(candidate_id)
        if chosen is None:
            validator_result["reason_codes"] = ["candidate_missing"]
            return validator_result

        parts = dict(candidate_scores.get(candidate_id) or {})
        extras = dict(chosen.extras or {})
        candidate_prior = float(parts.get("candidate_prior", 0.0) or 0.0)
        ocr_consensus = float(extras.get("ocr_consensus", min(1.0, candidate_prior)) or 0.0)
        top1_score = float(final_gating.get("top1_score", 0.0) or 0.0)
        score_margin = float(final_gating.get("score_margin", 0.0) or 0.0)
        agreeing_model_count = int(final_gating.get("agreeing_model_count", 0) or 0)
        model_errors = sorted(
            model_id
            for model_id, model_result in (per_model or {}).items()
            if isinstance(model_result, dict) and model_result.get("status") == "error"
        )
        reason_codes: List[str] = []
        if top1_score < thresholds["min_top1_score"]:
            reason_codes.append("validator_top1_below_threshold")
        if len(repair_candidates) > 1 and score_margin < thresholds["min_score_margin"]:
            reason_codes.append("validator_margin_below_threshold")
        if candidate_prior < thresholds["min_candidate_prior"]:
            reason_codes.append("validator_candidate_prior_below_threshold")
        if ocr_consensus < thresholds["min_ocr_consensus"]:
            reason_codes.append("validator_ocr_consensus_below_threshold")
        if agreeing_model_count < thresholds["min_agreeing_models"]:
            reason_codes.append("validator_insufficient_model_agreement")
        if stage == "repair_rerun":
            repair_ids = {str(item.get("candidate_id", "")) for item in repair_candidates}
            if candidate_id not in repair_ids:
                reason_codes.append("validator_candidate_outside_repair_subset")
            if not bool((repair_rerun or {}).get("promoted")):
                reason_codes.append("validator_rerun_not_promoted")

        validator_result["checks"] = {
            "top1_score": top1_score,
            "score_margin": score_margin,
            "candidate_prior": candidate_prior,
            "ocr_consensus": ocr_consensus,
            "agreeing_model_count": agreeing_model_count,
            "action_allowed": bool(final_gating.get("action_allowed")),
            "model_error_count": len(model_errors),
            "model_errors": model_errors,
        }
        validator_result["reason_codes"] = reason_codes
        validator_result["passed"] = not reason_codes
        validator_result["status"] = "passed" if not reason_codes else "failed"
        return validator_result

    def _run(payload: GroundingRequest, force_debug: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        image_bgr = decode_image_b64(payload.screenshot_b64 or "")
        expose_per_model = bool(payload.debug or force_debug)
        scored = _score_payload(payload)
        resolution = _resolve_ranking(payload, scored["ranked"], scored["per_model_top1"])
        final_prediction = resolution["top1"]
        final_gating = resolution["gating"]
        ranked_candidates = scored["ranked"]
        auto_execute = resolution["auto_execute"]
        resolution_mode = "auto_execute" if auto_execute else "repair"
        repair_plan = resolution["repair_plan"]
        repair_rerun = None
        validator_result = {
            "status": "skipped",
            "passed": False,
            "stage": "skipped",
            "candidate_id": str((final_prediction or {}).get("candidate_id", "") or ""),
            "reason_codes": [],
            "thresholds": {
                "min_top1_score": float(final_validator.get("min_top1_score", 0.70)),
                "min_score_margin": float(final_validator.get("min_score_margin", 0.08)),
                "min_candidate_prior": float(final_validator.get("min_candidate_prior", 0.60)),
                "min_ocr_consensus": float(final_validator.get("min_ocr_consensus", 0.50)),
                "min_agreeing_models": int(final_validator.get("min_agreeing_models", 2)),
            },
            "checks": {},
        }
        if not auto_execute and resolution["repair_candidates"]:
            repair_rerun = _run_repair_rerun(
                payload=payload,
                image_bgr=image_bgr,
                repair_candidates=resolution["repair_candidates"],
                expose_per_model=expose_per_model,
            )
            if repair_plan and repair_rerun:
                repair_plan = dict(repair_plan)
                repair_plan["status"] = "rerun_promoted" if repair_rerun["promoted"] else "rerun_inconclusive"
                repair_plan["crop_box"] = repair_rerun["crop_box"]
                repair_plan["rerun_candidate_ids"] = repair_rerun["candidate_ids"]
            if repair_rerun and repair_rerun["promoted"]:
                final_prediction = repair_rerun["final_prediction"]
                final_gating = repair_rerun["gating"]
                ranked_candidates = repair_rerun["ranked_candidates"]
                auto_execute = True
                resolution_mode = "repair_rerun_auto_execute"
        if auto_execute:
            validator_stage = "repair_rerun" if resolution_mode == "repair_rerun_auto_execute" else "initial"
            validator_result = _run_final_validator(
                payload=payload,
                final_prediction=final_prediction,
                final_gating=final_gating,
                candidate_scores=(repair_rerun or {}).get("candidate_scores", scored["candidate_scores"]) if validator_stage == "repair_rerun" else scored["candidate_scores"],
                per_model=(repair_rerun or {}).get("per_model", scored["per_model"]) if validator_stage == "repair_rerun" else scored["per_model"],
                stage=validator_stage,
                repair_candidates=resolution["repair_candidates"],
                repair_rerun=repair_rerun,
            )
            if not validator_result["passed"]:
                auto_execute = False
                resolution_mode = "repair_rerun_validation_failed" if validator_stage == "repair_rerun" else "repair_validation_failed"
                if repair_plan is None:
                    repair_plan = {
                        "status": "validation_failed",
                        "strategy": "validator_hold",
                        "candidate_ids": [str((final_prediction or {}).get("candidate_id", "") or "")],
                        "boxes": [list((final_prediction or {}).get("box") or [])],
                        "reason_codes": list(validator_result["reason_codes"]),
                    }
                else:
                    repair_plan = dict(repair_plan)
                    repair_plan["status"] = "validation_failed"
                    repair_plan["validator_reason_codes"] = list(validator_result["reason_codes"])
        response = {
            "model_id": "target-ensemble",
            "backend": "fanout",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "final_prediction": final_prediction,
            "ranked_candidates": ranked_candidates[: max(1, payload.top_k)],
            "candidate_scores": scored["candidate_scores"],
            "weights": weights,
            "gating": final_gating,
            "auto_execute": auto_execute,
            "resolution_mode": resolution_mode,
            "repair_candidates": resolution["repair_candidates"],
            "repair_plan": repair_plan,
            "repair_rerun": repair_rerun,
            "validator": validator_result,
            "per_model": scored["per_model"] if expose_per_model else {key: {"status": "ok"} for key in model_urls.keys()},
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
                {"id": "C2", "box": [430, 360, 640, 410], "text": "Cancel", "score": 0.18, "role": "button", "allowed_actions": ["click"]},
            ],
        )
        return _run(payload, force_debug=True)

    return app


app = create_app()
