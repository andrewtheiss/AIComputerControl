from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI

from ui_vision_common.candidate_graph import build_candidate_graph, candidate_graph_to_candidates
from ui_vision_common.debug_tools import grounding_overlay, new_artifact_dir, write_image, write_json
from ui_vision_common.image_tools import decode_image_b64
from ui_vision_common.schemas import Candidate, CandidateGraphRequest, CandidateGraphResponse


def create_app() -> FastAPI:
    debug_dir = str(os.environ.get("DEBUG_ARTIFACT_DIR", "/tmp/candidate-graph-debug") or "/tmp/candidate-graph-debug")
    app = FastAPI(title="Candidate Graph API")

    def _artifacts(payload: CandidateGraphRequest, response: CandidateGraphResponse) -> Dict[str, str]:
        artifact_dir = new_artifact_dir(debug_dir, "candidate-graph")
        request_path = write_json(os.path.join(artifact_dir, "request.json"), payload.model_dump())
        response_path = write_json(os.path.join(artifact_dir, "response.json"), response.model_dump())
        bundle = {"artifact_dir": artifact_dir, "request": request_path, "response": response_path}
        if payload.screenshot_b64:
            image_bgr = decode_image_b64(payload.screenshot_b64)
            raw_path = write_image(os.path.join(artifact_dir, "input.png"), image_bgr)
            overlay_path = write_image(
                os.path.join(artifact_dir, "overlay_candidates.png"),
                grounding_overlay(
                    image_bgr,
                    [candidate.model_dump() for candidate in response.candidates],
                    candidate_scores={},
                    final_candidate_id=response.candidates[0].id if response.candidates else "",
                ),
            )
            bundle["input"] = raw_path
            bundle["overlay"] = overlay_path
        return bundle

    def _run(payload: CandidateGraphRequest, force_debug: bool = False) -> CandidateGraphResponse:
        t0 = time.perf_counter()
        candidate_graph = build_candidate_graph(
            ui_elements=payload.ui_elements,
            limit=int(payload.limit or 80),
            viewport=payload.viewport,
        )
        candidates = [Candidate.model_validate(item) for item in candidate_graph_to_candidates(candidate_graph)]
        response = CandidateGraphResponse(
            model_id="candidate-graph",
            backend="library",
            candidate_graph=candidate_graph,
            candidates=candidates,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            meta={
                "input_element_count": len(payload.ui_elements),
                "graph_count": len(candidate_graph),
                "candidate_count": len(candidates),
                "merged_count": max(0, len(payload.ui_elements) - len(candidate_graph)),
            },
        )
        if payload.debug or force_debug:
            response.debug_artifacts = _artifacts(payload, response)
        return response

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model_id": "candidate-graph", "backend": "library"}

    @app.post("/infer", response_model=CandidateGraphResponse)
    def infer(payload: CandidateGraphRequest):
        return _run(payload, force_debug=False)

    @app.post("/infer/debug", response_model=CandidateGraphResponse)
    def infer_debug(payload: CandidateGraphRequest):
        payload.debug = True
        return _run(payload, force_debug=True)

    @app.post("/admin/selftest")
    def selftest() -> Dict[str, Any]:
        from ui_vision_common.image_tools import encode_png_b64
        import numpy as np

        image_bgr = np.full((768, 1024, 3), 255, dtype=np.uint8)
        response = _run(
            CandidateGraphRequest(
                screenshot_b64=encode_png_b64(image_bgr),
                debug=True,
                viewport={"width": 1024, "height": 768},
                ui_elements=[
                    {"source": "ocr_word", "text": "Sign in", "box": [700, 120, 815, 168], "score": 0.93},
                    {"source": "ax", "text": "Sign in", "box": [696, 116, 820, 172], "score": 0.99, "role": "button"},
                    {"source": "ax", "text": "Email", "box": [120, 210, 320, 252], "score": 0.98, "role": "textbox"},
                ],
            ),
            force_debug=True,
        )
        top = response.candidate_graph[0] if response.candidate_graph else {}
        return {
            "status": "ok",
            "model_id": "candidate-graph",
            "checks": {
                "graph_count": len(response.candidate_graph),
                "candidate_count": len(response.candidates),
                "top_id": top.get("id", ""),
                "top_text": top.get("text", ""),
            },
        }

    return app


app = create_app()
