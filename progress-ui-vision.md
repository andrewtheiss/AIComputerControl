# UI Vision Progress

Last updated: 2026-03-13

## Executive Summary

The new UI vision work is partially scaffolded and partially verified.

- The new mock model containers are real enough to boot, answer `/health`, and return structured mock predictions over HTTP.
- The replay/harness utility works for pure data-path tests.
- The service verification tests were replaced with container-backed integration tests and now pass cleanly in the local WSL environment.
- The real `ocr-api` and `ocr-ensemble-api` have now been re-verified with a readiness wait and a live-image OCR probe.
- The agent now uses target-ensemble decisions beyond shadow mode for gated auto-execute cases across all current click families: `click_text`, `click_any_text`, `click_near_text`, and `click_box`.
- A one-command verifier now exists at `tools/verify_ui_vision_stack.py`.
- `targetEnsemble` now includes gated resolver output for auto-execute vs repair-mode decisions, plus repair crop reruns that can promote a repair case when the rerun becomes decisive.

## What Was Verified Today

### Verified by live container HTTP checks

These services were started and checked successfully:

- `omniparser-api`
- `paddleocr-vl-api`
- `surya-api`
- `groundnext-api`
- `aria-ui-api`
- `phi-ground-api`
- `target-ensemble-api`
- `ocr-api`
- `ocr-ensemble-api`

Observed results:

- `GET /health` returned `200 OK` for all of the above.
- `POST /admin/selftest` returned valid structured results for:
  - `groundnext-api`
  - `omniparser-api`
  - `target-ensemble-api`

Specific verified responses:

- `groundnext-api /admin/selftest` returned `{"status":"ok","model_id":"groundnext","checks":{"predictions":2}}`
- `omniparser-api /admin/selftest` returned `{"status":"ok","model_id":"omniparser","checks":{"words":4,"lines":2}}`
- `target-ensemble-api /admin/selftest` returned a ranked response with:
  - top candidate `C1`
  - action `click`
  - score `0.71947`
  - per-model votes from `groundnext`, `aria-ui`, and `phi-ground`
- `ocr-api` stayed up long enough to finish model downloads and answered internal health with `{"status":"ok","backend":"ppocr"}`
- `ocr-ensemble-api /health` returned `{"status":"ok","model_id":"ocr-ensemble","models":["ppocr","omniparser","paddleocr-vl","surya"]}`

### Verified by live OCR path checks

- `ocr-api` accepted a live `POST /ocr` request from `ocr-ensemble-api`
- after readiness, `ocr-ensemble-api` returned live `ppocr` output on `texty_image.png`
- the `ppocr` per-model payload contained non-empty `words` and `lines`

Important nuance:

- if `ocr-ensemble-api` is queried immediately after bringing the stack up, `ppocr` may appear as `error`
- after `ocr-api` is actually ready, the same request succeeds
- this is currently a readiness race, not evidence of a hard crash

### Verified by checked-in debug artifacts

- `ocr-ensemble-debug/.../response.json` shows merged OCR output across mock OCR services.
- `target-ensemble-debug/.../response.json` shows per-model votes, final ranking, weights, and debug artifact paths.

### Verified by local test environment

These tests now pass:

- `tests.test_candidate_graph_service`
- `tests.test_model_service`
- `tests.test_model_service_providers`
- `tests.test_ocr_ensemble`
- `tests.test_target_ensemble`
- `tests.test_ui_vision_harness`
- `tests.test_target_ensemble_shadow_utils`
- `tools/verify_ui_vision_stack.py` also completes successfully

Verified command:

```powershell
wsl bash -lc 'cd /home/theiss/AIComputerControl && .venv-ui-vision/bin/python -m unittest tests.test_candidate_graph_service tests.test_model_service tests.test_model_service_providers tests.test_ocr_ensemble tests.test_target_ensemble tests.test_ui_vision_harness tests.test_target_ensemble_shadow_utils'
```

Observed result:

- `Ran 28 tests`
- `OK`

This means the current automated verification path covers:

- standalone candidate graph service boot plus selftest
- mock model-service container boot plus selftest
- model-service `http_proxy` provider behavior
- mock OCR ensemble container boot plus selftest
- mock target ensemble container boot plus selftest
- gated resolver behavior for confident, ambiguous, and repair-rerun-promoted cases
- candidate graph building and merge behavior
- candidate building
- action inference
- instruction derivation
- shadow endpoint normalization and shadow merge/score parity helpers
- gated agent execution overrides for `click_text`, `click_any_text`, `click_near_text`, and `click_box`

Additional direct runtime verification:

- the new final validator stage was exercised by calling the `targetEnsemble` `/infer` endpoint function directly with mock backends
- the container-backed `tests.test_target_ensemble` suite was re-run successfully after disk cleanup
- the standalone `candidate-graph-api` service was verified over HTTP with `/health`, `/infer/debug`, and `/admin/selftest`
- the harness can now call `candidate-graph-api` explicitly when `CANDIDATE_GRAPH_API_URL` is configured
- the generic model sidecars now support real backend routing via `MODEL_BACKEND=http_proxy` plus `REMOTE_MODEL_URL`
- verified pass case:
  - strong `Sign in` candidate returns `auto_execute: true`
  - validator returns `status: "passed"`
- verified fail case:
  - low-prior `Sign in` candidate returns `auto_execute: false`
  - resolution mode becomes `repair_validation_failed`
  - validator returns `validator_candidate_prior_below_threshold`
- verified repair-rerun pass case:
  - rerun promotion returns `resolution_mode: "repair_rerun_auto_execute"`
  - validator returns `status: "passed"` with `stage: "repair_rerun"`

## Reproduced Failures

### 1. Legacy in-process service tests segfault in WSL

The original service-test approach reproducibly crashed with exit code `139`:

- `tests.test_model_service`
- `tests.test_ocr_ensemble`
- `tests.test_target_ensemble`

What is important:

- imports alone are fine
- `TestClient(create_app())` construction is fine
- the segfault happens when the test actually sends a request through `Starlette TestClient`

Observed stack shape from `faulthandler`:

- crash occurs in the `anyio` blocking portal / event loop thread used by `Starlette TestClient`
- main thread is inside `httpx` -> `starlette.testclient` -> `.post(...)`
- this is a native segfault, not a Python exception

Environment fingerprint where this reproduces:

- Python `3.12.3`
- FastAPI `0.135.1`
- Starlette `0.52.1`
- HTTPX `0.28.1`
- NumPy `2.4.3`
- OpenCV `4.13.0`

Current conclusion:

- do not reintroduce `Starlette TestClient` for these services in this environment
- container-backed verification is the currently working replacement

### 2. `ocr-ensemble` can fail cleanly when `ocr-api` is down

Checked artifact evidence shows:

- `ppocr` path returned connection refused
- mock OCR sidecars still returned output

This is not a segfault, but it is an important verification failure mode.

### 3. `ocr-ensemble` can show transient `ppocr` errors if queried before `ocr-api` is ready

Observed behavior:

- immediate post-start ensemble request can report:
  - `ppocr: "error"`
  - `ppocr_word_count: 0`
- after waiting for `ocr-api` readiness, the same flow returns real `ppocr` OCR words and lines

Current conclusion:

- ensemble verification should include a readiness wait for `ocr-api`

### 4. Container exit code `255` after restart is not enough evidence of an app crash

The recently exited containers all share the same shutdown time window around the machine restart.

Likely interpretation:

- those `255` exits are consistent with Docker/WSL shutdown during the hard restart
- they do not by themselves prove that each service crashed internally

## Likely Risk Areas

### Highest confidence current issue

- local WSL `.venv-ui-vision` service tests crash in `Starlette TestClient`

### Likely operational risks

- `ocr-api` is the heaviest runtime in this stack and is the main candidate for GPU/runtime instability
- `ocr-ensemble-api` depends on `ocr-api`, so it should not be used as an early verification step
- the Dockerfile for `ocr-api` defaults to `UVICORN_WORKERS=2`; compose overrides this to `1`, but running the image outside compose without that override is a footgun for GPU model startup
- the generic model sidecars now support `MODEL_BACKEND=http_proxy` and per-service `REMOTE_MODEL_URL` settings, but the default compose configuration still runs them in `mock` mode unless real endpoints are configured

## Current Progress Against The Requested Architecture

### Done

- created a generic `modelService` wrapper for OCR and grounding model endpoints
- created `ocrEnsemble` service
- created `targetEnsemble` service
- created a reusable candidate graph builder
- created shared schemas and debug helpers in `ui_vision_common`
- added `tools/ui_vision_harness.py` for replay/export workflows
- added `tools/verify_ui_vision_stack.py` for one-command verification
- added agent-side shadow plumbing for target ensemble
- added agent-side gated execution integration for decisive target-ensemble results across all current click families
- added agent-side candidate graph service consumption with local fallback
- added harness-side candidate graph service consumption with local fallback
- added compose definitions for the new services
- added gated resolver outputs to `targetEnsemble`
- added repair crop reruns to `targetEnsemble` with regression tests
- added final validator stage to `targetEnsemble`
- added standalone `candidate-graph-api` with `/health`, `/infer`, `/infer/debug`, and `/admin/selftest`
- added configurable real-backend routing for model sidecars via `MODEL_BACKEND=http_proxy` and `REMOTE_MODEL_URL`
- added `.env.ui-vision.example` for real backend wiring
- added a new optional `ui-vision-real` compose profile with heavyweight local runtime services for:
  - OmniParser V2
  - PaddleOCR-VL-1.5
  - Surya
  - GroundNext-7B
  - Aria-UI
  - Phi-Ground
- added `.env.ui-vision.real.example` for local runtime bring-up

### Partially done

- OCR ensemble fan-out and merge exists and is now verified with live `ppocr`, but it still returns merged OCR words/lines rather than a full interactable graph
- target ensemble now includes threshold, margin, agreement gating, repair crop reruns, and a final validator stage
- candidate graph building exists as both a shared library and a standalone service; the agent and harness can now consume the standalone HTTP service directly with local fallbacks
- the stack can now route model sidecars to real remote backends over HTTP, and there is now an initial local-runtime compose profile for the actual upstream models, but the current wrapper `*-api` services are still the stable integration surface
- `omniparser-runtime` is now running locally and `omniparser-api` can call it through a new `MODEL_BACKEND=omniparser_runtime` adapter that converts `/parse/` output into the repo's boxed OCR schema
- `paddleocr-vl-runtime` is now running locally and `paddleocr-vl-api` can call it through a new `MODEL_BACKEND=paddleocr_vl_runtime` adapter that converts `Spotting:` output with `<|LOC_...|>` tokens into the repo's boxed OCR schema
- `surya-runtime` is now running locally and `surya-api` can call it through a new `MODEL_BACKEND=surya_runtime` adapter that converts `text_lines` output into the repo's boxed OCR schema
- `ocr-ensemble-api` has now been re-verified with the real local `omniparser-api`, `paddleocr-vl-api`, and `surya-api` wrappers active alongside `ppocr`
- live agent integration now exists for all current click families, but it is still gated and only takes over when the target ensemble returns a decisive, constraint-compatible auto-execute result
- debug output exists, but not yet at the full artifact granularity originally requested
- harness exists and a one-command verifier exists, but there is not yet a full benchmark/eval suite for every stage

### Not done yet

- adapter-level integration from the remaining new local runtime containers into the existing wrapper `/ocr` and `/infer` contracts
- verification that the new local runtime containers all boot cleanly on this specific machine and GPU setup
- robust benchmark runs over larger real corpora

## Safe Verification Plan

This is the recommended order. Stop after each step and only continue if the pass criteria are met.

### Step 0. Snapshot the state

Purpose:

- confirm exactly what code and containers you are testing

Run:

```powershell
git status --short --branch
git diff --stat main
docker compose ps --all
```

Pass criteria:

- you know whether you are testing uncommitted work
- you know which containers are already up or exited

Stop here if:

- there are unexpected local changes you do not recognize

### Step 1. Verify the mock OCR-model services only

Purpose:

- verify service boot, health checks, and OCR-shaped responses without touching the risky real OCR runtime

Run:

```powershell
docker compose --profile ui-vision up -d omniparser-api paddleocr-vl-api surya-api
curl.exe -s http://localhost:28101/health
curl.exe -s http://localhost:28102/health
curl.exe -s http://localhost:28103/health
curl.exe -s -X POST http://localhost:28101/admin/selftest
```

Pass criteria:

- each `/health` returns `status=ok`
- `omniparser-api /admin/selftest` returns word and line counts

Manual checks:

- `docker compose ps omniparser-api paddleocr-vl-api surya-api`
- `docker logs --tail 100 aicomputercontrol-omniparser-api-1`

Stop here if:

- any service fails health
- any service is restarting

### Step 2. Verify the mock grounding services and target ensemble

Purpose:

- verify that the new three-grounder stack and ensemble transport work end-to-end over HTTP

Run:

```powershell
docker compose --profile ui-vision up -d groundnext-api aria-ui-api phi-ground-api target-ensemble-api
curl.exe -s http://localhost:28111/health
curl.exe -s http://localhost:28112/health
curl.exe -s http://localhost:28113/health
curl.exe -s http://localhost:28130/health
curl.exe -s -X POST http://localhost:28111/admin/selftest
curl.exe -s -X POST http://localhost:28130/admin/selftest
```

Pass criteria:

- all services return `status=ok`
- `target-ensemble-api /admin/selftest` returns:
  - ranked candidates
  - per-model predictions
  - weights
  - debug artifact paths

Manual checks:

- `docker logs --tail 100 aicomputercontrol-target-ensemble-api-1`
- verify that per-model vote data is present in the JSON

Stop here if:

- ensemble health works but selftest does not return per-model vote details

### Step 3. Verify the pure harness path

Purpose:

- verify the automated local safety net

Run:

```powershell
wsl bash -lc 'cd /home/theiss/AIComputerControl && .venv-ui-vision/bin/python -m unittest tests.test_model_service tests.test_ocr_ensemble tests.test_target_ensemble tests.test_ui_vision_harness'
```

Pass criteria:

- all 7 tests pass
- no segfault occurs

Stop here if:

- the suite does not end in `OK`

### Step 4. Do not revive the old in-process request path

Purpose:

- avoid repeating the known segfault path

Legacy bad pattern:

```powershell
from fastapi.testclient import TestClient
```

Current result:

- this old approach was the source of the reproducible WSL segfault path

Recommended replacement:

- keep using the container-backed tests and manual HTTP checks

### Step 5. Verify `ocr-api` in isolation before touching `ocr-ensemble-api`

Purpose:

- test the real OCR runtime alone, because it is the main unstable dependency

Important caution:

- `ocr-api` is not exposed on a host port in the current compose file
- verify it from inside the container or add a temporary port mapping in a separate branch if you want host-side `curl`

Recommended sequence:

```powershell
docker compose up -d ocr-api
docker compose ps ocr-api
docker logs --tail 200 aicomputercontrol-ocr-api-1
docker exec aicomputercontrol-ocr-api-1 python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8020/health').read().decode())"
```

Pass criteria:

- container stays `Up`
- health endpoint returns JSON
- logs do not show repeated restarts or immediate native failure
- model startup completes and `Uvicorn running on http://0.0.0.0:8020` appears in logs

Stop here if:

- the container exits
- Docker reports restart loops
- system instability returns

### Step 6. Only after `ocr-api` is stable, verify `ocr-ensemble-api`

Purpose:

- confirm fan-out behavior with live `ppocr` plus the mock OCR sidecars

Run:

```powershell
docker compose --profile ui-vision up -d ocr-ensemble-api
docker compose ps ocr-ensemble-api
curl.exe -s http://localhost:28120/health
```

Then inspect:

```powershell
docker logs --tail 200 aicomputercontrol-ocr-ensemble-api-1
```

Pass criteria:

- `ocr-ensemble-api` stays up
- health succeeds
- logs do not show persistent `ocr-api` connection refused errors
- a real-image request returns `ppocr` status `ok`

Stop here if:

- `ppocr` is still failing after `ocr-api` health is confirmed

### Step 7. Enable shadow mode only after Steps 1 through 6 are stable

Purpose:

- verify agent integration without letting the new stack drive clicks

Current compose state:

- `TARGET_ENSEMBLE_SHADOW_MODE: "0"`

What to change later:

- set it to `1`
- keep `TARGET_ENSEMBLE_SHADOW_DEBUG: "1"`

Pass criteria:

- the agent runs normally
- trace output contains `target_ensemble_shadow`
- no click execution changes are driven by the new stack yet

## What I Think Caused The Trouble

What I can say with confidence:

- there is a real, reproducible local segfault in the current WSL test workflow when service tests use `Starlette TestClient`
- the machine-restart-era `255` container exits do not by themselves prove those containers individually crashed

What is plausible but not fully proven from this audit:

- verifying too many GPU-heavy services together could have contributed to instability
- `ocr-api` is the most likely runtime stress point once real model execution starts
- if the `ocr-api` image was ever run without the compose override, `UVICORN_WORKERS=2` would be an unnecessary risk

## Recommended Immediate Next Changes

### Before more runtime validation

- keep the new container-backed service tests as the default verification path
- add one dedicated verification script per service:
  - model service
  - OCR ensemble
  - target ensemble
- bring up `ui-vision-real` one service at a time, starting with `omniparser-runtime` and `paddleocr-vl-runtime`
- keep `ocr-api` isolated until it has a known-good health and single-request verification path

### Before integration into the live clicker

- build the real candidate graph service
- add score gating and repair logic to `targetEnsemble`
- enable shadow mode and collect disagreements before allowing any execution changes

## Open Items

- confirm whether `ocr-api` can survive a clean isolated startup after the reboot
- verify `ocr-ensemble-api` against live `ocr-api`
- wire the new local runtime containers into the stable wrapper `*-api` services
- build the full candidate graph and repair loop
