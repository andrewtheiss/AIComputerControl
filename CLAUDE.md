# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For deeper detail, see `DOCUMENTATION.md` and `README.md`. This file is the orientation layer.

## Stack overview

A multi-container platform that runs desktop UI automation agents in GPU-enabled XFCE/VNC sandboxes. Agents observe a virtual desktop via OCR + optional detection + accessibility, ask a planner for the next action, then dispatch keyboard/mouse via `xdotool` and verify whether the UI actually changed.

The live runtime path is orchestrated by `docker-compose.yml`. Services split into two profiles:

**Always on (bare `docker compose up -d`):**
- `ocr-api` (`ocr/`) — FastAPI wrapper around PaddleOCR PP-OCRv5. Returns word- and line-level boxes. Host port `28020:8020` for smoke tests.
- `rtdetr-api` (`inference/`) — FastAPI wrapper around a TensorRT RT-DETR engine for optional object detection.
- `task-planner` (`taskPlanner/`) — FastAPI service that calls an OpenAI-compatible LLM (configured via `OLLAMA_OPENAI_BASE` / `OLLAMA_MODEL`) and returns a single next action as strict JSON. `POST /v1/actions/next`.

**Behind `--profile ui-vision` (required for the agents themselves):**
- `vnc-instance-1` / `vnc-instance-2` (`agent/`) — Headful XFCE+Firefox containers running `agent/src/agent.py` in `AGENT_MODE=dynamic`. Exposed at `localhost:25901` / `localhost:25902` for VNC. **These agents are in the `ui-vision` profile — bare `up -d` does not start them.**
- `modelService/` generic per-model HTTP wrappers: `omniparser-api` (28101), `paddleocr-vl-api` (28102), `surya-api` (28103), `groundnext-api` (28111), `aria-ui-api` (28112), `phi-ground-api` (28113). Default backend is `mock`.
- `ocr-ensemble-api` (28120) — fans out to ppocr + omniparser + paddleocr-vl + surya, merges. **This is the agent's OCR source now** via `OCR_API_URL=http://ocr-ensemble-api:8000/ocr`.
- `candidate-graph-api` (28125), `target-ensemble-api` (28130) — candidate graph + grounding ensemble. Both consumed by the agent (candidate-graph via `CANDIDATE_GRAPH_API_URL`, target-ensemble via `TARGET_ENSEMBLE_API_URL` for shadow + execution mode).
- Real-weight runtime services are under a separate `ui-vision-real` profile (`omniparser-runtime`, `paddleocr-vl-runtime`, `surya-runtime`, `groundnext-runtime`, `aria-ui-runtime`, `phi-ground-runtime`).

Shared schemas and helpers for the ensemble stack live in `ui_vision_common/`.

GPU split in the sample compose: OCR on GPU 2, RT-DETR on GPU 1, agents on GPU 0 — adjust to local hardware. All services share the `ai-net` Docker network and resolve each other by service name.

## Architecture: dynamic agent loop

`agent/src/agent.py` has two modes (`AGENT_MODE`):

1. **`static`** — `TaskRunner` executes a YAML workflow from `tasks/{AGENT_NAME}.yaml` with hot-reload. Ops: `open_url`, `click_text`, `wait_text`, `wait_detection`, `type_text`, `key_seq`, `for_each`, `for_pages`, `if`, `ocr_extract`, etc.

2. **`dynamic`** — `ActionExecutorDynamic` is the production path. Each step:
   1. `mss` capture → OCR via `ocr-api` (Tesseract fallback) including multi-pass top/bottom interaction-band passes → optional AX nodes via `A11Y_BRIDGE_URL` → optional RT-DETR detections.
   2. Build a state snapshot in `screen_signatures.py` (hash, top texts, semantic tags like `app:browser_like`/`surface:auth_like`/`phase:loading_like`, focus metadata) and classify blockers via `blockers.py` (`browser_session_restore`, `cookie_banner`, `browser_url_suggestion_dropdown`, modal dialogs, etc.).
   3. POST `{goal, current_state, task_history, ocr_results, ui_elements, screenshot?, planner_session_id, available_actions}` to `task-planner`.
   4. Apply executor-side policy: blockers handled before goal progression; typing blocked when AX says focus is not editable; anti-repeat guards prevent the same action against the same unresolved state-signature; brittle planner picks (e.g. "Compose"/"Send") may be intercepted and re-run through the multi-proposer consensus path in `decision.py` (`ComposeProposer`, `SendProposer`, `DismissModalProposer`, `arbitrate`).
   5. Dispatch via `xdotool`, then **verify** post-action (state hash/signature delta, tag delta, blocker cleared, focus changed, typed text now visible, loading indicator). If only dispatched but not verified, the action is recorded as unresolved — *not* success.

Key invariant: **the planner is advisory.** The executor is allowed to override or block planner output, and verification is required before treating an action as successful. When changing executor logic, preserve this — don't introduce code paths that mark actions successful purely on event dispatch.

`ui_elements` sent to the planner are a fusion of OCR words, OCR line sub-boxes, synthesized token boxes, AX nodes, and optional detections — see `perception.py` and `ui_core.py`.

## Branch context

The current branch is `ocroverhaul`. Recent work moves the click-targeting/grounding stack onto a canonical **candidate graph** (`ui_vision_common/candidate_graph.py`, `candidateGraph/`) and a **target ensemble** (`targetEnsemble/`, `agent/src/target_ensemble_shadow_utils.py`) running in **shadow mode** — i.e. the new resolver scores candidates in parallel with the live path and logs results, but does not yet drive clicks. The next planned step (per the latest commit message) is upgrading the resolver with threshold/margin/2-of-3 gating + repair reruns and adopting it on the live click path. Keep new ensemble work behind the same shadow gating until that switchover.

## Common commands

Bring up the default (backend-only) stack — ocr-api + rtdetr-api + task-planner:
```bash
docker compose up -d --build
```

Bring up the full stack with agents and the UI-vision ensemble — this is what you want for live runs:
```bash
docker compose --profile ui-vision up -d --build
```
(The agents `vnc-instance-1` / `vnc-instance-2` are in the `ui-vision` profile, so they only start when that profile is active. The agents' `OCR_API_URL` points at `ocr-ensemble-api`, so the decision path sees fused multi-model OCR, not just ppocr.)

Iteration shortcuts:
```bash
# Rebuild + recreate just planner and both agents
docker compose up -d --build --force-recreate task-planner vnc-instance-1 vnc-instance-2

# One agent only
docker compose up -d --build --force-recreate vnc-instance-1

# Logs
docker compose logs -f task-planner
docker compose logs -f vnc-instance-1
```

VNC into an agent: `localhost:25901` (agent-1), `localhost:25902` (agent-2). Password is the one baked into `agent/passwd` (one-time `vncpasswd passwd` from inside `agent/`).

Tests (Python `unittest`-style under `tests/`, with `tests/support.py` providing harness helpers — many tests spin up real subprocess services, so they're slow and have system deps):
```bash
# All tests
python3 -m pytest tests/

# A single test file or test
python3 -m pytest tests/test_target_ensemble.py
python3 -m pytest tests/test_target_ensemble.py::TestName::test_method
```

Debug helpers (run from repo root):
```bash
# Smoke-test every OCR ingredient in the running stack (ocr-api, the three
# modelService sidecars, and ocr-ensemble-api). Add --grounding to also
# selftest candidate-graph and target-ensemble. See OCR_AUDIT.md.
python3 tools/ocr_smoke.py
python3 tools/ocr_smoke.py --grounding --json

# Open the latest agent trace.html files in Windows Explorer
python3 scripts/open_latest_traces.py

# Side-by-side planner-dump comparison across LLM models
python3 scripts/compare_models.py --dump planner-dumps/<dump>.json --models "modelA,modelB"

# Replay a planner request
python3 scripts/planner_replay.py

# Lifecycle verifier — starts/stops services and runs the safe test suite.
# Unlike ocr_smoke, this owns the lifecycle.
python3 tools/verify_ui_vision_stack.py --skip-real-ocr
```

Each OCR ingredient also writes its own debug artifacts to host-mounted dirs when `debug: true` is set in the request (or for selftest calls):
- `./ui-model-debug/<model_id>/<TS-HEX>/` — per-model inputs, overlays, response JSON
- `./ocr-ensemble-debug/ocr-ensemble/<TS-HEX>/` — merged overlay plus `per_model.json` showing per-upstream status/latency/results
- `./candidate-graph-debug/`, `./target-ensemble-debug/` — request/response JSON + overlays
- `./agent-debug/<agent>-<run_id>/` — full trace.jsonl + trace.html + per-step hierarchy/crop overlays (primary forensic surface)
- `./planner-dumps/` — sanitized planner request/response dumps

## Working in this repo

- Prefer editing inside the existing services; the agent ↔ planner ↔ ocr ↔ ensemble boundaries are HTTP, so swap implementations service-locally rather than refactoring across them.
- New planner actions need three coordinated edits: document in `taskPlanner/tools.py`, add to the `available_actions` list in the planner payload, and implement in `ActionExecutorDynamic`. Missing any of the three causes silent fallbacks.
- New static-mode ops go in `TaskRunner._do_steps` in `agent/src/agent.py`.
- Adding another OCR/grounding model is cheapest as a new `modelService/`-style sidecar plus a merge entry in `ocrEnsemble/` or `targetEnsemble/`; the agent already supports HTTP-first OCR with Tesseract fallback in `agent/src/ocr_client.py`.
- `taskgen-ai` is installed and a `planner_agent` exists, but the live planner path uses direct `AsyncOpenAI` calls in `taskPlanner/main.py`. Don't assume the `taskgen-ai` agent is on the request path.
- Detection labels are sent to the planner as generic `label:<id>` strings — there is no label map yet.
- The committed `docker-compose.yml` contains plaintext example secrets; treat that as known tech debt, don't propagate it into new services.
