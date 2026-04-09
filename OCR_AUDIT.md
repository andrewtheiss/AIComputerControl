# OCR Stack Audit

Last updated: 2026-04-09
Branch: `ocroverhaul`

## Goal of this audit

Three questions, one per section:

1. **Does each OCR ingredient work in its own context?**
2. **Are the ingredients maximally and cleanly integrated into the dynamic agent that chooses next actions?**
3. **Can each ingredient be debugged while a `docker compose --profile ui-vision up -d` stack is running?**

---

## 1. Ingredient inventory

| # | Service | Image / module | Role | Default MODEL_BACKEND | Host port | Consumed by agent? |
|---|---|---|---|---|---|---|
| 1 | `ocr-api` | `ai-ocr-ppocr` (ocr/) | Primary PPOCR (PaddleOCR PP-OCRv5) | real | **none** (internal) | yes (legacy path) |
| 2 | `omniparser-api` | `ai-ui-model-omniparser` (modelService/) | OCR model wrapper | `mock` (`OMNIPARSER_MODEL_BACKEND`) | 28101 | indirectly, via ocr-ensemble-api |
| 3 | `paddleocr-vl-api` | same | OCR model wrapper | `mock` | 28102 | indirectly, via ocr-ensemble-api |
| 4 | `surya-api` | same | OCR model wrapper | `mock` | 28103 | indirectly, via ocr-ensemble-api |
| 5 | `ocr-ensemble-api` | `ai-ocr-ensemble` (ocrEnsemble/) | Fan-out + merge OCR aggregator | n/a (fanout) | 28120 | **yes after this audit** (was orphaned) |
| 6 | `candidate-graph-api` | `ai-candidate-graph` (candidateGraph/) | Candidate graph builder | n/a (library) | 28125 | yes (`CANDIDATE_GRAPH_API_URL`) |
| 7 | `groundnext-api` / `aria-ui-api` / `phi-ground-api` | modelService/ | Grounding model wrappers | `mock` | 28111–28113 | indirectly, via target-ensemble-api |
| 8 | `target-ensemble-api` | `ai-target-ensemble` (targetEnsemble/) | Weighted voting + repair + gated resolver | n/a (fanout) | 28130 | yes (shadow + execution via `TARGET_ENSEMBLE_API_URL`) |

The agent itself (`vnc-instance-1`, `vnc-instance-2`) is also behind the `ui-vision` compose profile. A bare `docker compose up -d` brings up only `ocr-api`, `rtdetr-api`, and `task-planner`. **You must use `--profile ui-vision` to get the agents and the ensemble stack together.**

All services accept JSON (`image_b64`) or multipart (`file`) for their OCR endpoints via a shared decoder in `modelService/app/main.py:29-46` and `ocrEnsemble/app/main.py:129-144`. All services have `/health` and `/admin/selftest`.

---

## 2. Standalone status per ingredient

Each line below is the one-shot probe that proves the ingredient is alive and producing output in isolation.

Assumes `docker compose --profile ui-vision up -d` has been run and the stack has stabilised.

### 2.1 `ocr-api` (legacy PPOCR)

Before this audit, `ocr-api` had **no host port mapping** — smoke testing from the host required a `docker compose exec` gymnastics. This audit adds a host-reachable port.

- Health (after this audit's port mapping): `curl http://localhost:28020/health` → `{"status":"ok","backend":"ppocr"}`
- Fallback for older checkouts without the new port mapping:
  ```bash
  docker compose exec ocr-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8020/health').read().decode())"
  ```
- Real OCR probe:
  ```bash
  curl -s -X POST http://localhost:28020/ocr -F "file=@texty_image.png" | jq '.words | length, .lines | length'
  ```
- Reload model in place: `curl -X POST http://localhost:28020/admin/reload`
- Debug artifacts: **none** — `ocr-api` does not persist anything to disk. All forensics must come from `docker compose logs -f ocr-api`.

Code entry points: `ocr/app/main.py:27-98`, engine at `ocr/app/engine_ppocr.py`. Known quirks:
- `PP_OCR_MODEL` env var is read but never passed to PaddleOCR (`ocr/app/engine_ppocr.py:8`) — it's cosmetic.
- `detect_rotation` field in `OCRRequest` is accepted but ignored.
- Dockerfile defaults `UVICORN_WORKERS=2`; compose overrides to `1`. If you run the image directly without the override, GPU model init will race.

### 2.2 `modelService/` sidecars — omniparser, paddleocr-vl, surya (OCR role)

Same binary, three different `MODEL_ID` env values. All default to `MODEL_BACKEND=mock` in compose.

```bash
for p in 28101 28102 28103; do
  curl -s http://localhost:$p/health
  curl -s -X POST http://localhost:$p/admin/selftest
done
```

`/admin/selftest` (`modelService/app/main.py:151-174`) runs the provider against a synthetic 1024×768 white frame and returns `{"status":"ok","model_id":"…","checks":{"words":N,"lines":N}}`. With `MODEL_BACKEND=mock`, the hardcoded words are `Browser`, `Sign`, `in`, `Continue` (see `ui_vision_common/mock_backends.py` → `mock_ocr_result` — per-model pixel offsets so the ensemble's merge logic gets exercised).

Real OCR against an image:
```bash
IMG_B64=$(base64 -w0 < texty_image.png)
curl -s -X POST http://localhost:28101/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"'"$IMG_B64"'","return_level":"both","debug":true}' \
| jq '.words | length, .debug_artifacts'
```

Debug artifacts land in the container at `/tmp/ui-model-debug/<model_id>/<YYYYMMDD-HHMMSS-XXXXXXXX>/` and are mounted back to `./ui-model-debug/` on the host (docker-compose.yml:41, 59, 77). Each bundle contains:
- `input.png` — the decoded frame as-received
- `overlay_words.png` — OCR boxes drawn over the input
- `response.json` — full `OCRResponse`

Debug artifacts only land if you set `debug: true` in the request OR call `/infer/debug` (for grounding) OR `/admin/selftest`. Normal `/ocr` calls do not write artifacts.

Switching a sidecar from mock to a real backend:
```bash
# in .env or shell
OMNIPARSER_MODEL_BACKEND=http_proxy
OMNIPARSER_REMOTE_MODEL_URL=http://omniparser-runtime:8000
docker compose --profile ui-vision up -d --force-recreate omniparser-api
```
The `http_proxy` provider (`modelService/app/providers.py:59-72`) re-POSTs the same OCR request to `REMOTE_MODEL_URL/ocr`. There's a parallel `ui-vision-real` profile with `*-runtime` services that provide those real backends.

### 2.3 `ocr-ensemble-api`

Fan-out aggregator that calls all four upstream OCR services via `OCR_ENSEMBLE_MODEL_URLS` (`ocrEnsemble/app/main.py:25-29`). Current wiring:
```json
{"ppocr":"http://ocr-api:8020/ocr","omniparser":"http://omniparser-api:8000/ocr","paddleocr-vl":"http://paddleocr-vl-api:8000/ocr","surya":"http://surya-api:8000/ocr"}
```

Merge behavior (`ocrEnsemble/app/main.py:61-107`): groups items by normalized lowercase text + spatial overlap (IoU ≥ 0.10 or center distance ≤ 40px). Averages scores, concatenates `source_model` comma-separated. Per-level (word/line) independently.

```bash
curl -s http://localhost:28120/health
curl -s -X POST http://localhost:28120/admin/selftest | jq '.meta.per_model_status, (.words | length)'
curl -s -X POST http://localhost:28120/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"$(base64 -w0 < texty_image.png)\",\"return_level\":\"both\",\"debug\":true}" \
| jq '.meta.per_model_status, (.words | length), .debug_artifacts.artifact_dir'
```

Debug artifacts land at `./ocr-ensemble-debug/ocr-ensemble/<TS-HEX>/`:
- `input.png`
- `overlay_merged_words.png` (white boxes per merged group)
- `per_model.json` — per-upstream `{status, latency_ms, words, lines}` including **errors** per upstream
- `response.json` — final merged response with `meta.per_model_status`

The single most useful artifact for debugging is `per_model.json`: if `ppocr` is `error` while the others are `ok`, ocr-api isn't ready/alive, and you'll see this in one file instead of hunting through four log streams.

**Readiness race**: as documented in `progress-ui-vision.md:163-179`, hitting `ocr-ensemble-api` the instant the stack comes up can show `ppocr: error` for a minute or two while PPOCR downloads/initializes the model. This is a known transient, not a crash.

**Mock-drop behavior** (added in this audit): the ensemble now excludes mock-backed upstreams from the merged output by default, gated by `OCR_ENSEMBLE_DROP_MOCK_BACKEND=1` (docker-compose default). A response is classified as mock if its upstream URL starts with `mock://` (the ensemble's own mock scheme) OR its `backend` field is `"mock"` (a modelService sidecar running on `MODEL_BACKEND=mock`). Dropped upstreams remain visible in `meta.per_model` with `status: "mock_skipped"` so debug forensics still work — the only thing that changes is that their placeholder text ("Browser", "Sign in", "Continue") no longer leaks into the merged `words` / `lines` the agent consumes. The response also carries `meta.active_models` (contributors to the merged output) and `meta.dropped_mock_models` (skipped mocks) at the top level.

Set `OCR_ENSEMBLE_DROP_MOCK_BACKEND=0` on the service to restore the legacy merge-everything behavior. The `tests/test_ocr_ensemble.py::OCREnsembleDropMockTest` suite exercises both modes directly via the pure `execute_ocr_fanout` entry point (no HTTP, no subprocess) — run it with:

```bash
.venv-ui-vision/bin/python -m unittest tests.test_ocr_ensemble.OCREnsembleDropMockTest -v
```

### 2.4 `candidate-graph-api`

Library service that applies `ui_vision_common/candidate_graph.py`'s merge/scoring logic over HTTP.

```bash
curl -s http://localhost:28125/health
curl -s -X POST http://localhost:28125/admin/selftest | jq '.checks'
```

Request body is a `CandidateGraphRequest` (`ui_vision_common/schemas.py`): `ui_elements` (the same fused word/line/AX payload the agent builds), `limit`, `viewport`, `screenshot_b64`, `debug`. Response is a ranked `candidate_graph` with `interactable_score`, `ocr_consensus`, `source_mask`, per-candidate `bbox_abs`, and a flat `candidates` list.

Debug artifacts at `./candidate-graph-debug/candidate-graph/<TS-HEX>/`:
- `request.json`, `response.json`
- `input.png` (if `screenshot_b64` was provided)
- `overlay_candidates.png`

The agent posts to this service at every step the dynamic loop runs (see §3.4 below).

### 2.5 Grounding sidecars (`groundnext-api`, `aria-ui-api`, `phi-ground-api`)

Same binary as the OCR sidecars but with `MODEL_ROLE=grounding`, so they expose `/infer` (`modelService/app/main.py:120-149`) instead of `/ocr`.

```bash
for p in 28111 28112 28113; do
  curl -s http://localhost:$p/health
  curl -s -X POST http://localhost:$p/admin/selftest | jq '.checks'
done
```

Selftest synthesizes a frame + two candidates (`C1: Sign in`, `C2: Continue`) and ranks them. With mock backend, `groundnext` is weighted higher so `C1` wins (`ui_vision_common/mock_backends.py` → `MODEL_TEXT_WEIGHTS`).

Debug artifacts at `./ui-model-debug/<model_id>/<TS-HEX>/`:
- `input.png`
- `overlay_candidates.png` — each candidate box with per-model score, final pick drawn in white
- `request.json`, `response.json`

### 2.6 `target-ensemble-api`

Fan-out across the three grounding sidecars, weighted vote, optional repair crop rerun, final validator with threshold/margin/2-of-3 gating (`targetEnsemble/app/main.py:218-447`).

```bash
curl -s http://localhost:28130/health
curl -s -X POST http://localhost:28130/admin/selftest | jq '.final_prediction, .gating, .validator'
```

Env vars that shape its behavior (all set by docker-compose.yml:336-341):
- `TARGET_ENSEMBLE_MODEL_URLS` — where to fan out
- `TARGET_ENSEMBLE_WEIGHTS` — `{groundnext:0.45, aria-ui:0.25, phi-ground:0.20, candidate_prior:0.10}`
- `TARGET_ENSEMBLE_GATING` — threshold/margin/min_agreeing_models for auto-execute gate
- `TARGET_ENSEMBLE_VALIDATOR` — final safety check before emitting an auto-execute

Debug artifacts at `./target-ensemble-debug/target-ensemble/<TS-HEX>/`:
- `input.png`, `overlay_votes.png`
- `request.json`, `response.json` — includes per-model results, `gating` checks, `repair_plan`, `validator_result`

### 2.7 `rtdetr-api` (not strictly OCR, adjacent)

Optional detection sidecar. Agent only calls it if you explicitly wire detections in. Out of scope for this OCR audit, but listed for completeness: `curl http://localhost:<host-mapped-if-any>/health`. No host port mapped currently — it's reached from the agent via the internal `http://rtdetr-api:8000/predict`.

---

## 3. Integration into the dynamic agent

The decision path is `_capture_state` → `_state_snapshot` / `_public_state` → planner POST → `_target_ensemble_shadow` → (optional) `resolve_execution_override` → dispatch → verify. Every step that needs pixels-to-symbols passes through OCR.

### 3.1 OCR capture (entry point)

`agent/src/agent.py:192-193` builds a single `OCRClient` pointing at `OCR_API_URL` (default `http://ocr-api:8020/ocr`). `OCRClient` (`agent/src/ocr_client.py`) sends a multipart PNG to that URL with a `(3s connect, 15s read)` timeout. On any exception it silently falls back to local Tesseract. The response's `source` field is `"http"`, `"tesseract"`, or `"ensemble_http"` (new — see fixes applied below).

`ocr_image_levels` (`agent/src/agent.py:280-282`) wraps `OCRClient.ocr_levels` with the multi-band merge in `_ocr_bands_merge` (`agent/src/agent.py:203-273`). The band pass re-runs OCR on the top 18 % and bottom 40 % of the frame, shifts the boxes back into full-frame coordinates, and merges with IoU-0.5 dedup.

### 3.2 State capture

`ActionExecutorDynamic._capture_state` (`agent/src/agent.py:4053-4092`) calls `ocr_image_levels`, splits into `self.last_ocr_words` / `self.last_ocr_lines`, stamps `_origin` tags, and sorts-and-caps:
- Words: `max(OCR_LIMIT, 2000)` — **`OCR_LIMIT` has no effect on words because of the `max`**
- Lines: `max(40, min(OCR_LIMIT, 160))` — `OCR_LIMIT` is effectively clamped 40–160

That `OCR_LIMIT` mismatch is in `agent/src/agent.py:4067-4068`. It's in the audit notes because it silently makes the environment variable misleading.

`last_ocr_source` is already captured on `self` and logged on `state.captured` — forensics for "am I actually on the HTTP path?" are already available via the agent's structured log. With the fix applied below, the source string now disambiguates between `http_ensemble` and `http_ppocr`, so a single log line tells you whether the ensemble is being consumed.

### 3.3 Planner payload

`_build_planner_payload` (`agent/src/agent.py:4649-4714`) fuses:
- OCR lines → `ui_elements` with `source: ocr_line`, capped at `PLANNER_MAX_OCR_LINE_ELEMENTS` (default 40)
- OCR words → `ui_elements` with `source: ocr_word`, capped at `PLANNER_MAX_OCR_WORD_ELEMENTS` (default 80)
- AX nodes → `source: ax`, capped at 40
- Optional detections → `source: det`, capped at 60
- A separate minimal `ocr_results` list with `{text, box, conf, level}` for direct planner consumption

So the planner sees OCR twice: a rich fused `ui_elements` list (the thing click actions are expected to operate on) and a flat `ocr_results` list (legacy-compatible view). Both are populated from the same underlying `last_ocr_words`/`last_ocr_lines`, so whatever backend is behind `OCR_API_URL` drives both.

### 3.4 Shadow + execution via target-ensemble

`_target_ensemble_shadow` (`agent/src/agent.py:3571-3623`) runs for every `click_text|click_any_text|click_near_text|click_box` action if `TARGET_ENSEMBLE_SHADOW_MODE=1` (default in compose). It:
1. Builds candidates via `CANDIDATE_GRAPH_API_URL` (falls back to local `build_candidate_graph`)
2. POSTs `{instruction, candidates, screenshot_b64, top_k, debug}` to `TARGET_ENSEMBLE_API_URL`
3. Returns the shadow result, logged as `target_ensemble.shadow` into `trace.jsonl`

If `TARGET_ENSEMBLE_EXECUTION_MODE=1` (also default in compose), `resolve_execution_override` in `agent/src/target_ensemble_shadow_utils.py` can upgrade the shadow result to an execution override when gating passes. The executor logs `target_ensemble.execution_override` and swaps `op`/`params`.

### 3.5 Gaps I found, pre-audit

The following ARE actual integration gaps that needed fixing:

1. **`ocr-ensemble-api` was orphaned from the live decision path.** The agent only consumed legacy `ocr-api`. omniparser/paddleocr-vl/surya were being fanned out only for their own local selftests, never into the state the planner sees. After this audit, the `OCR_API_URL` for both `vnc-instance-*` is set to `http://ocr-ensemble-api:8000/ocr`, so `_capture_state` transparently pulls merged multi-model OCR on every step. OCRClient's existing multipart POST + `OCRResponse` parser are drop-in compatible with `ocrEnsemble/app/main.py:210-213`.
2. **`ocr-api` had no host port.** You could not curl it from the host. Fixed by mapping `28020:8020` — preserves the internal DNS name for in-cluster calls while giving you a host-side debug handle.
3. **`depends_on` on agents was incomplete.** Only `candidate-graph-api` was listed. `ocr-ensemble-api` and `target-ensemble-api` were called from the agent but not in `depends_on`, so `up -d` ordering was racy. Added both.
4. **Tesseract fallback was silent.** `ocr_client.py:82` swallowed the HTTP exception with `pass`. You couldn't see from traces when the HTTP OCR path had silently degraded. Added a `logging.warning` with the exception class so a fallback event shows up in agent logs and the operator can grep for it.
5. **`OCRClient` source tagging didn't distinguish ppocr from ensemble.** When `OCR_API_URL` changes between runs, nothing in the trace tells you which backend was in use. Now the client looks at the URL (`ensemble` in the URL → `ensemble_http`, else `http`), and the agent's `state.captured` debug log surfaces that string via `ocr_source`.
6. **No unified host-side smoke test.** `tools/verify_ui_vision_stack.py` is a lifecycle verifier that starts/stops services. It does not answer the live-stack question "is OCR working end-to-end right now?" Added `tools/ocr_smoke.py`, which probes every OCR ingredient in parallel against `texty_image.png` and prints `per_model_status + word_count + latency` as a single JSON report.

Everything else listed in the agent-path exploration (unused `A11Y_BRIDGE_URL` duplicate in `perception.py`, VLM locate box cache comment with no actual cache, `OCR_LIMIT` clamp, blocker classifier not consuming AX) is OCR-adjacent tech debt but not blocking the OCR audit. Left in the tracker, not touched in this pass.

---

## 4. Debug matrix (live `docker compose --profile ui-vision up -d` stack)

This is the cheat sheet for "test this one ingredient right now."

| Ingredient | Health | Selftest | Real OCR probe | Debug artifacts (host path) | Container logs |
|---|---|---|---|---|---|
| `ocr-api` | `curl localhost:28020/health` | n/a (no selftest endpoint) | `curl -X POST localhost:28020/ocr -F "file=@texty_image.png"` | **none** — logs only | `docker compose logs -f ocr-api` |
| `omniparser-api` | `curl localhost:28101/health` | `curl -X POST localhost:28101/admin/selftest` | JSON body with `image_b64` + `debug:true` | `./ui-model-debug/omniparser/<TS-HEX>/` | `docker compose logs -f omniparser-api` |
| `paddleocr-vl-api` | `curl localhost:28102/health` | `curl -X POST localhost:28102/admin/selftest` | same | `./ui-model-debug/paddleocr-vl/<TS-HEX>/` | `docker compose logs -f paddleocr-vl-api` |
| `surya-api` | `curl localhost:28103/health` | `curl -X POST localhost:28103/admin/selftest` | same | `./ui-model-debug/surya/<TS-HEX>/` | `docker compose logs -f surya-api` |
| `ocr-ensemble-api` | `curl localhost:28120/health` | `curl -X POST localhost:28120/admin/selftest` | multipart OR JSON body with `debug:true` → check `per_model_status` | `./ocr-ensemble-debug/ocr-ensemble/<TS-HEX>/` (incl. `per_model.json`) | `docker compose logs -f ocr-ensemble-api` |
| `candidate-graph-api` | `curl localhost:28125/health` | `curl -X POST localhost:28125/admin/selftest` | `POST /infer/debug` with `CandidateGraphRequest` | `./candidate-graph-debug/candidate-graph/<TS-HEX>/` | `docker compose logs -f candidate-graph-api` |
| `groundnext-api` | `curl localhost:28111/health` | `curl -X POST localhost:28111/admin/selftest` | `POST /infer/debug` with `GroundingRequest` | `./ui-model-debug/groundnext/<TS-HEX>/` | `docker compose logs -f groundnext-api` |
| `aria-ui-api` | `curl localhost:28112/health` | `curl -X POST localhost:28112/admin/selftest` | same | `./ui-model-debug/aria-ui/<TS-HEX>/` | — |
| `phi-ground-api` | `curl localhost:28113/health` | `curl -X POST localhost:28113/admin/selftest` | same | `./ui-model-debug/phi-ground/<TS-HEX>/` | — |
| `target-ensemble-api` | `curl localhost:28130/health` | `curl -X POST localhost:28130/admin/selftest` | `POST /infer/debug` with `GroundingRequest` → inspect `per_model`, `gating`, `validator` | `./target-ensemble-debug/target-ensemble/<TS-HEX>/` | `docker compose logs -f target-ensemble-api` |
| agent (live run) | VNC into `localhost:25901` or `25902` | n/a | run the agent with `AGENT_DEBUG=1` (default) | `./agent-debug/<agent-name>-<run_id>/` (trace.jsonl, trace.html, hierarchy/crop overlays) | `docker compose logs -f vnc-instance-1` |

### Two commands worth committing to muscle memory

```bash
# Probe the entire OCR stack against texty_image.png, print JSON report.
python3 tools/ocr_smoke.py

# Open the latest agent trace HTMLs in Windows Explorer (WSL).
python3 scripts/open_latest_traces.py
```

`tools/ocr_smoke.py` is new in this audit. It hits every OCR ingredient in parallel, prints `{service, status, latency_ms, word_count, line_count, first_5_words, error}`, and exits non-zero if any service fails. It is the fastest answer to "is the OCR stack healthy right now?"

`scripts/open_latest_traces.py` already existed and is the best answer to "what did the agent actually see last time it ran?"

### How to tell which OCR source is live in the agent

Search the agent logs for `state.captured` events:
```bash
docker compose logs vnc-instance-1 2>&1 | grep state.captured | tail
```
Each line now carries `ocr_source: ensemble_http | http | tesseract`. If you see `tesseract` at any point, the HTTP OCR degraded — the matching `agent.ocr.fallback` warning line tells you why.

---

## 5. Applied changes (this audit commit)

Minimal, surgical. Each one answers a specific audit finding above.

1. **docker-compose.yml**
   - `ocr-api`: mapped `28020:8020` so it's host-reachable for debug.
   - `vnc-instance-1` and `vnc-instance-2`: set `OCR_API_URL: http://ocr-ensemble-api:8000/ocr`, so the decision-making agent's OCR path goes through the ensemble aggregator by default. Existing `OCRClient` is drop-in compatible.
   - `vnc-instance-1` and `vnc-instance-2`: added `ocr-ensemble-api` and `target-ensemble-api` to `depends_on`.
   - `ocr-ensemble-api`: set `OCR_ENSEMBLE_DROP_MOCK_BACKEND: "1"` explicitly so the intent is documented in-place.

2. **agent/src/ocr_client.py**
   - Added a `logging.warning` on Tesseract fallback with the exception class name. No more silent degradation.
   - Response `source` field now disambiguates `ensemble_http` vs `http` by inspecting the URL. The agent's `state.captured` log already prints `source`, so this change is automatically visible in traces.

3. **ocrEnsemble/app/main.py**
   - New top-level helper `execute_ocr_fanout(image_bgr, req, model_urls, drop_mock_backend, session=None)` encapsulates the full fan-out + merge + mock-drop logic as a pure function. `create_app._run` is now a thin wrapper that delegates to it and handles debug artifact writing. Lets tests exercise the exact production code path with no HTTP, subprocess, or ASGI transport.
   - New env var `OCR_ENSEMBLE_DROP_MOCK_BACKEND` (default `1`): when on, upstream responses whose URL starts with `mock://` or whose `backend` field is `"mock"` are tagged `mock_skipped` in `per_model` and excluded from the merged `words`/`lines`. Response meta now includes `active_models`, `dropped_mock_models`, and `drop_mock_backend`.

4. **tools/ocr_smoke.py** (new)
   - Parallel probes of all six OCR services (`ocr-api`, `omniparser-api`, `paddleocr-vl-api`, `surya-api`, `ocr-ensemble-api`, and optionally the grounding + target ensemble if you pass `--grounding`).
   - Uses `texty_image.png` at the repo root as the probe image.
   - Prints one JSON report, exits non-zero on any failure.

5. **tests/test_ocr_ensemble.py**
   - Replaced the legacy single-assert test with a three-test `OCREnsembleDropMockTest` suite that calls `execute_ocr_fanout` directly: default-drop behavior, legacy merge-everything behavior when disabled, and a mixed real+mock scenario that monkey-patches `_call_model` to prove mock contamination is prevented end-to-end. The docker-compose smoke test is retained under `OCREnsembleComposeTest` but its expectations are updated for the new default.

6. **CLAUDE.md**
   - Corrected a prior inaccuracy: the `ui-vision` profile is required for the agents themselves, not just the ensemble sidecars. Bare `docker compose up -d` does not start the agents.

---

## 6. Known tech debt (out of scope for this pass)

Flagged in the audit but not fixed:

- **`OCR_LIMIT` env var is a no-op for words** (`agent/src/agent.py:4067`). Fix is trivial but behavior-changing.
- **Blocker classifier ignores AX evidence** (`agent/src/agent.py:4060`, `blockers.py`). AX nodes are fetched and passed to the planner but not to `classify_blockers`. Real integration for AX-sourced modal detection is a bigger lift.
- **Duplicate `A11Y_BRIDGE_URL`** (`agent/src/agent.py:59` and `agent/src/perception.py:8`), only one is actually used. `perception.a11y_snapshot` is dead code.
- **`ocr_client.py` filter pass is duplicated** — score filter then empty-text filter on the same list (`ocr_client.py:96-115`). Cosmetic.
- **PPOCR `MODEL` env var is cosmetic** (`ocr/app/engine_ppocr.py:8`). Misleading.
- **`detect_rotation` schema field is ignored** (`ocr/app/main.py`).
- **`candidate-graph-api` defaults to GPU-less library ops** — it's CPU-only, which is correct for a graph service, but worth documenting if you see latency spikes on very large `ui_elements`.
