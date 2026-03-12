AIComputerControl - System Documentation

## Purpose
This repository runs desktop UI automation agents inside GPU-enabled containers. Each agent sees a virtual Linux desktop (XFCE over VNC), understands the UI through OCR, optional accessibility snapshots, and object detection, then acts by sending keyboard and mouse events. Agents can either follow a static YAML task script or run dynamically through a planner-backed observe -> decide -> act loop.

The current production path in `docker-compose.yml` uses:
- `ocr-api` for primary OCR
- `rtdetr-api` for optional object detection
- `task-planner` for next-action planning
- `vnc-instance-*` agents in `AGENT_MODE=dynamic`

## High-level Architecture
- **OCR API (`ocr/`)**: FastAPI service wrapping PaddleOCR PP-OCRv5. Returns both word-level and line-level text boxes.
- **Inference API (`inference/`)**: FastAPI service wrapping a TensorRT RT-DETR engine. Accepts screenshots and returns detection boxes.
- **Agent (`agent/`)**: Headful Linux desktop with Firefox. Captures screen, runs OCR, optionally calls detection and accessibility sources, classifies blockers, executes actions via `xdotool`, and verifies whether actions actually changed the UI.
- **Task Planner (`taskPlanner/`)**: FastAPI service that receives the agent goal, current state summary, OCR results, UI elements, recent history, and optionally a screenshot, then returns a single next action plus parameters.
- **Orchestration (`docker-compose.yml`)**: Brings up all services on a shared Docker network with GPU isolation and agent-specific Firefox profiles mounted from the host.

Dynamic-mode data flow:
1. Agent captures the screen with `mss`.
2. Agent runs OCR through `ocr-api` first; if OCR HTTP is unavailable, the local agent falls back to Tesseract.
3. Agent optionally fetches accessibility nodes and optional RT-DETR detections.
4. Agent builds a state snapshot with tags, blocker classification, focus info, and a signature hash.
5. Agent posts `{goal, current_state, task_history, ocr_results, ui_elements, screenshot?, planner_session_id, available_actions}` to the planner.
6. Planner returns strict JSON for one action.
7. Agent may allow, override, or block that action based on executor-side policies.
8. Agent executes the action and verifies the post-action UI transition before treating it as success.

---

## Services and Components

### OCR API (`ocr/app`)
- Entrypoint: `ocr/app/main.py`
  - Endpoints:
    - `GET /health` -> `{"status":"ok","backend":"ppocr"}`
    - `POST /ocr` -> OCR result with image size plus word and/or line boxes
    - `POST /admin/reload` -> rebuilds the OCR engine in-process
- Request inputs:
  - Multipart image upload via `file`
  - Or JSON body with base64 image payload via `image_b64`
- Output model:
  - `width`, `height`
  - `words[]` and/or `lines[]`
  - Each item contains polygon coordinates, text, and score
- Engine: `ocr/app/engine_ppocr.py`
  - Uses `paddleocr.PaddleOCR`
  - Defaults to `PP-OCRv5` server model
  - Supports HPI toggle via `PP_OCR_ENABLE_HPI`
  - Orientation mode is configurable
- Post-processing: `ocr/app/postproc.py`
  - Normalizes PaddleOCR v2/v3 shapes into common word items
  - Greedily groups words into line-level boxes

Primary OCR behavior in the agent:
- The agent posts PNG screenshots to `OCR_API_URL`
- OCR output is normalized into `words` and `lines`
- If the OCR service is unavailable, the agent falls back to local Tesseract
- The agent also runs a multi-pass OCR strategy over the full frame plus top and bottom "interaction bands" for better browser chrome and footer-button recall

Key OCR-related environment variables:
- `OCR_API_URL`
- `OCR_MIN_SCORE`
- `OCR_BAND_MULTIPASS`
- `OCR_BAND_TOP_FRAC`
- `OCR_BAND_BOTTOM_FRAC`
- `PP_OCR_DEVICE`
- `PP_OCR_MODEL`
- `PP_OCR_DET_SIDE`
- `PP_OCR_ORIENTATION`

### Inference API (`inference/app`)
- Entrypoint: `inference/app/main.py`
  - Endpoints:
    - `GET /health` -> `{"status":"ok"}`
    - `POST /predict` -> `{"detections":[{"box":[x1,y1,x2,y2],"score":float,"label":int}, ...]}`
  - Reads model path from `ENGINE_PATH` (default `/app/models/rtdetr-l.engine`)
- Engine: `inference/app/inference.py` (`RTDETREngine`)
  - Loads TensorRT engine once and serializes inference with a lock
  - Supports both legacy TensorRT bindings and v3 tensor API
  - Preprocess: resize to `640x640`, convert BGR -> RGB, normalize to `[0,1]`, NCHW, CUDA tensor
  - Postprocess:
    - supports single-output `N x 6`
    - supports `boxes + logits`
    - maps normalized center-based boxes back to image pixel coordinates
  - Uses a dedicated CUDA stream
- Utilities:
  - `inference/export_onnx.py` for export-time workflows
  - `inference/requirements_export.txt` for export dependencies

Usage:
- Send an image via multipart `file`
- Response contains detections with boxes in image coordinates

### Agent (`agent/src/agent.py`)
The file contains two execution modes selected by `AGENT_MODE`.

#### 1) Static runner: `TaskRunner`
- Loads YAML from `AGENT_TASK` (default `/tasks/{AGENT_NAME}.yaml`)
- Watches the file for changes and reloads when modified
- Uses:
  - screen capture with `mss`
  - OCR through `ocr-api` with Tesseract fallback
  - detection through `rtdetr-api`
  - input dispatch with `xdotool`
- Supports operations such as:
  - navigation and waits: `open_url`, `wait_text`, `wait_detection`, `detect`, `sleep`
  - OCR interactions: `click_text`, `ocr_extract`
  - detection interactions: `click_detection`
  - input: `type_text`, `key_seq`
  - control flow: `if`, `for_each`, `for_pages`, `for_questions`
  - utilities: `set_var`, `track_series`, `load_json`, `run_llm`, `screenshot`, `debug_dump_ctx`
- Saves debug overlays, screenshots, and step logs under `${AGENT_DEBUG_DIR}/${AGENT_NAME}-${RUN_ID}` when debugging is enabled

#### 2) Dynamic executor: `ActionExecutorDynamic`
This is the core live path used by the compose stack.

Main loop:
1. Capture state
2. Summarize state for the planner
3. Ask the planner for the next action
4. Apply executor-side policy checks and overrides
5. Dispatch the action
6. Verify whether the action actually changed the UI
7. Record the result in history and continue

What the dynamic executor actually does:
- Captures OCR words and OCR lines, capping them separately
- Fetches optional AX nodes from an accessibility bridge when configured
- Builds UI element payloads from:
  - OCR word boxes
  - OCR line boxes
  - optional detections
  - AX nodes
- Builds a state snapshot with:
  - state hash
  - top visible texts
  - semantic tags like `app:browser_like`, `surface:auth_like`, `phase:loading_like`
  - blocker descriptors
  - blocker signature
  - visible blocker resolve targets
  - focus metadata
- Detects blockers with `agent/src/blockers.py`
- Applies blocker-specific policy before executing planner instructions
- Prevents repeated same-family actions against the same unresolved state
- Verifies post-action state transitions rather than assuming clicks or keys worked

Action classes currently offered to the planner:
- `open_url`
- `wait_text`
- `wait_any_text`
- `click_text`
- `click_any_text`
- `click_near_text`
- `click_box`
- `type_text`
- `key_seq`
- `sleep`
- `ocr_extract`
- `end_task`

Important executor behaviors:
- The planner is advisory; the executor can override unsafe or brittle actions
- Typing is blocked when AX says the focused element is not editable
- Browser blockers are handled before task progression
- Repeated unresolved actions trigger anti-repeat guards
- Click targeting can use:
  - raw OCR words
  - OCR line sub-boxes
  - synthesized token boxes
  - AX nodes
  - optional VLM fallback for ambiguous click targets
- Post-action verification uses state similarity, tags, blocker changes, focus changes, and typed-text visibility evidence

Consensus mode:
- Certain brittle planner actions such as "Compose" or "Send" may be intercepted and re-run through a multi-proposer consensus path in `agent/src/decision.py`
- Candidates are ranked by confidence and basic heuristics before being tried in order

Common agent configuration:
- `AGENT_MODE` = `static` | `dynamic`
- `AGENT_GOAL`
- `TASK_PLANNER_URL`
- `PLANNER_API_KEY`
- `RTDETR_API_URL` or `DETECT_API_URL`
- `OCR_API_URL`
- `OCR_MIN_SCORE`
- `AGENT_NAME`
- `AGENT_TASK`
- `MAX_STEPS`
- `HISTORY_WINDOW`
- `OCR_LIMIT`
- `CLICK_ENABLED`
- `AGENT_SCREEN_INDEX`
- `AGENT_DEBUG`
- `AGENT_DEBUG_DIR`
- `AGENT_TRACE_ENABLED`
- `PLANNER_SEND_SCREENSHOT_MODE`
- `PLANNER_SCREENSHOT_JPEG_QUALITY`
- `PLANNER_SCREENSHOT_MAX_DIM`
- `A11Y_BRIDGE_URL`
- `FIREFOX_BIN`
- `FIREFOX_PROFILE_PATH`
- Optional LLM settings for helper/VLM paths:
  - `LLM_API_URL`
  - `LLM_MODEL`
  - `LLM_API_KEY`
  - `LLM_API_MODE`

Agent container and desktop:
- Dockerfile installs XFCE, TigerVNC, Firefox ESR, `xdotool`, Tesseract, and Python dependencies
- VNC password is pre-generated on the host and copied into the image via `agent/passwd`
- On start, VNC launches display `:1`, starts XFCE, and runs `agent.py`
- Compose maps VNC to:
  - `localhost:25901` for `vnc-instance-1`
  - `localhost:25902` for `vnc-instance-2`

### Blocker detection and decision core
The key executor logic is a hybrid of LLM planning plus hardcoded safety rules.

Blockers currently classified from OCR text:
- `browser_session_restore`
- `browser_permission_prompt`
- `cookie_banner`
- `modal_dialog`
- `browser_interstitial_error`
- `browser_url_suggestion_dropdown`

Important behaviors:
- Blockers are ranked by priority and confidence
- Browser suggestion dropdowns typically force `Escape` before page clicks
- Session restore prefers large page-level CTAs like `Start New Session` over tiny chrome targets
- Cookie banners and dialog-like blockers prefer visible resolve targets before generic recovery
- The executor tracks recent recovery attempts per blocker signature and avoids repeating exhausted strategies

Resolve target scoring:
- Lower score is better
- Scores consider:
  - text match quality
  - button-like geometry
  - footer proximity
  - OCR confidence
  - penalties for wide row-like boxes or merged labels
  - area as a tie-breaker

Verification:
- Non-passive actions are not assumed successful
- Success evidence may include:
  - state hash/signature change
  - changed tags
  - blocker cleared or changed
  - focus changed
  - text became visible or disappeared
  - loading indicators appearing after navigation/submission
- If the event was dispatched but the outcome is not verified, the action is recorded as unresolved rather than success

### Task Planner (`taskPlanner/`)
- Entrypoint: `taskPlanner/main.py`
- Endpoint: `POST /v1/actions/next`
- Security: optional bearer token via `PLANNER_API_KEY`
- Request models:
  - goal
  - task history
  - current state
  - OCR results
  - UI elements
  - optional screenshot
  - planner session id
  - available actions
- Response model:
  - `action`
  - `parameters`
  - `reasoning`
  - `completed`

How the planner works today:
- Uses `AsyncOpenAI` against an OpenAI-compatible endpoint configured by:
  - `OLLAMA_OPENAI_BASE`
  - `OLLAMA_MODEL`
- Uses a strong system prompt with guardrails around:
  - blocker clearing
  - focus-aware typing
  - repeated-failure avoidance
  - browser chrome vs page-content distinctions
  - visible resolve-target preference
- Accepts optional screenshots and will attempt multimodal requests first when a screenshot is present
- Falls back to text-only if the backend rejects multimodal input
- Retries the plan call with exponential backoff using `tenacity`
- Parses strict JSON, extracts embedded JSON if needed, and validates actions against `available_actions`
- Normalizes common parameter aliases before returning the result

About `taskgen-ai`:
- `taskgen-ai` is installed and a `planner_agent` object is created
- In the current live request path, planning decisions are produced by direct `AsyncOpenAI` calls rather than by dispatching through the `planner_agent`
- The docstrings in `taskPlanner/tools.py` still serve as the planner's tool contract

Planner environment:
- `PLANNER_API_KEY`
- `OLLAMA_OPENAI_BASE`
- `OLLAMA_MODEL`
- `PLANNER_LOG_REQUESTS`
- `PLANNER_DUMP_REQUESTS_DIR`
- `PLANNER_DUMP_MAX_FILES`
- `PLANNER_LM_SESSION_ENABLED`

---

## Orchestration (`docker-compose.yml`)
The current compose file starts these services:

- `ocr-api`
  - Builds from `./ocr`
  - Uses GPU access
  - Runs PaddleOCR
  - Internal service name: `http://ocr-api:8020` by default from the agent's perspective

- `rtdetr-api`
  - Builds from `./inference`
  - Runs uvicorn on port `8000`
  - Uses GPU device `1`
  - Internal service URL: `http://rtdetr-api:8000/predict`

- `task-planner`
  - Builds from `./taskPlanner`
  - Exposes host port `28000:8000`
  - Internal service URL: `http://task-planner:8000/v1/actions/next`
  - Uses `host.docker.internal` to reach the host LLM server

- `vnc-instance-1`
  - Builds from `./agent`
  - Uses GPU device `0`
  - Exposes `25901:5901`
  - Mounts:
    - `./tasks` -> `/tasks`
    - `./agent-debug` -> `/tmp/agent-debug`
    - host Firefox profile/cache
  - Runs in `AGENT_MODE=dynamic`

- `vnc-instance-2`
  - Uses the same agent image
  - Exposes `25902:5901`
  - Runs in `AGENT_MODE=dynamic`

Shared networking:
- All services join `ai-net`
- Services resolve each other by compose service name

GPU notes:
- Example split in the current compose:
  - OCR on GPU `2`
  - RT-DETR on GPU `1`
  - Agents on GPU `0`
- Adjust device ids to match local hardware

Security note:
- The sample compose currently contains plaintext API keys and example secrets in environment variables
- Prefer moving secrets into `.env`, Docker secrets, or another secure injection mechanism

---

## Configuration and Profiles

### Tasks YAML (`tasks/*.yaml`)
Static mode uses YAML files such as `tasks/agent-1.yaml`.

Common structure:
- `version`
- `agent`
- `loop`
- `steps`

Each step is a single-key mapping like:
```yaml
- click_text:
    regex: "^Compose$"
    nth: 0
```

Variable substitution:
- `${ENV_VAR}` can resolve from environment variables or runtime context values

Known mismatches in example YAML:
- `click_text.fuzzy` is not a supported argument; use `fuzzy_text` and `fuzzy_threshold`
- `ocr_extract.capture` is ignored; `save_as` is the meaningful key
- `extract_grade_table` in `tasks/agent-2.yaml` is not implemented and will log `op.unknown`

### Firefox profiles (`profiles/agent-*`)
- Compose mounts Firefox profile and cache paths into each agent container
- This preserves cookies, sessions, and browser state across restarts

### Secrets
- Agent tasks can use environment values such as `SECRETS_EMAIL` and `SECRETS_PASSWORD`
- Avoid hardcoding secrets in YAML
- Prefer environment injection or a dedicated secret store

---

## APIs

### OCR API
- `POST /ocr`
  - Input:
    - multipart `file`, or
    - JSON with `image_b64`
  - Request options:
    - `return_level`: `word` | `line` | `both`
    - `min_score`
  - Response:
    ```json
    {
      "width": 1280,
      "height": 720,
      "words": [{"poly": [[1,2],[3,2],[3,4],[1,4]], "text": "Inbox", "score": 0.98}],
      "lines": [{"poly": [[1,2],[10,2],[10,4],[1,4]], "text": "Inbox Compose", "score": 0.95}]
    }
    ```
- `GET /health`
- `POST /admin/reload`

### Inference API
- `POST /predict`
  - Multipart form field: `file`
  - Response:
    ```json
    {
      "detections": [
        {"box": [100.0, 150.0, 200.0, 250.0], "score": 0.95, "label": 17}
      ]
    }
    ```
- `GET /health`

### Task Planner API
- `POST /v1/actions/next`
  - Request shape (abridged):
    ```json
    {
      "goal": "Open Firefox and navigate to Gmail",
      "task_history": [{"action":"click_text","parameters":{"regex":"^Compose$"},"result":{"status":"dispatched","outcome_verified":false}}],
      "current_state": {
        "hash": "abc123",
        "tags": ["app:browser_like", "blocker:browser_url_suggestion_dropdown"],
        "blockers": [],
        "focused_role": "entry",
        "focused_editable": true
      },
      "ocr_results": [{"text":"Inbox","box":[1,2,3,4],"conf":97,"level":"word"}],
      "ui_elements": [{"source":"ocr_word","text":"Inbox","box":[1,2,3,4],"score":0.97,"role":null}],
      "screenshot_b64": null,
      "planner_session_id": "agent-1:deadbeef1234",
      "available_actions": ["open_url","wait_text","wait_any_text","click_text","click_any_text","click_near_text","click_box","type_text","key_seq","sleep","ocr_extract","end_task"]
    }
    ```
  - Response:
    ```json
    {
      "action": "click_text",
      "parameters": {"regex": "^Compose$", "nth": 0},
      "reasoning": "Compose is visible and no blocker remains, so opening a new message is the next valid step.",
      "completed": false
    }
    ```

---

## Building, Running, and Scaling

Host prerequisites:
- Windows 11 with WSL2 and Docker Desktop using the WSL backend
- Recent NVIDIA drivers and NVIDIA Container Toolkit
- VNC viewer such as TigerVNC Viewer

One-time VNC password setup:
```bash
sudo apt-get update
sudo apt-get install -y tigervnc-tools
cd agent/
vncpasswd passwd
```

Start the stack:
```bash
docker compose up --build -d
```

Connect to running agents:
- `vnc-instance-1` -> `localhost:25901`
- `vnc-instance-2` -> `localhost:25902`

View logs:
```bash
docker compose logs -f ocr-api
docker compose logs -f rtdetr-api
docker compose logs -f task-planner
docker compose logs -f vnc-instance-1
```

Shutdown:
```bash
docker compose down
```

Scaling note:
- The repository currently defines two named agent services rather than one replicated service
- To scale beyond two agents, duplicate a service block or refactor compose into a templated pattern

---

## Testing and Utilities

### Planner request dumps
- `task-planner` can dump sanitized request JSON plus decoded screenshots to `planner-dumps/`
- This is useful for replay, comparison, and prompt debugging

### Agent traces
- Dynamic mode can save trace frames, overlays, and an HTML trace view under `agent-debug/`
- These traces include planner decision, executed action, verification evidence, and click targeting visuals

### Mock inference server
- `tools/mock_inference_server/` contains sample Triton-style GRPC mock services
- These are not on the main runtime path, which currently uses HTTP FastAPI services

---

## Extending the System

- Add new planner tools:
  - document the tool in `taskPlanner/tools.py`
  - expose it in the planner payload `available_actions`
  - implement it in `ActionExecutorDynamic`
- Add new static ops:
  - extend `TaskRunner._do_steps`
  - normalize arguments consistently
- Add another OCR engine:
  - easiest path is another HTTP OCR sidecar and a merge/fallback layer in `agent/src/ocr_client.py`
  - the current agent already supports HTTP-first OCR with local Tesseract fallback, so a second HTTP source can be added cleanly
- Swap detection models:
  - rebuild the inference container with a new engine file and keep post-processing compatible
- Customize browsers and profiles:
  - change `FIREFOX_BIN`
  - mount per-agent profiles

Recommended second OCR engine if you want to mix another one in:
- **EasyOCR** is the simplest additional engine to add for a second opinion on cropped UI text
- Good fit:
  - easy Python integration
  - works well on many UI screenshots
  - can be run as another sidecar or in-process fallback
- Trade-offs:
  - slower than PaddleOCR
  - typically weaker than PaddleOCR on some tiny browser chrome text

Other candidates:
- **Tesseract**: already present as a local fallback in the agent
- **Surya**: stronger for document-style OCR/layout, heavier integration
- **TrOCR**: useful for crop-level recognition, usually better as a targeted recognizer than as a full-screen OCR replacement

---

## Notes and Limitations

- `extract_grade_table` in `tasks/agent-2.yaml` is still not implemented
- Some example YAML keys are ignored by the current implementation
- `taskgen-ai` is installed, but the live planner path uses direct `AsyncOpenAI` calls
- Detection labels are currently sent to the planner as generic `label:<id>` strings unless you add a label map
- `xdotool` interactions require the correct X11 desktop focus inside the agent container
- Multi-monitor capture depends on `AGENT_SCREEN_INDEX`
- The current compose file includes plaintext secrets and should be hardened before wider use

---

## Repository Map

- `docker-compose.yml`: service orchestration, GPU assignment, profiles, goals, and ports
- `agent/`
  - `Dockerfile`: desktop, VNC, Firefox, system packages, agent runtime
  - `passwd`: pre-generated VNC password file
  - `src/agent.py`: static and dynamic agent runtimes
  - `src/blockers.py`: OCR-driven blocker classifier
  - `src/decision.py`: consensus candidate generation and arbitration
  - `src/ocr_client.py`: OCR HTTP client and Tesseract fallback
  - `src/perception.py`: OCR/detection/AX fusion helpers
  - `src/screen_signatures.py`: state signature and similarity helpers
- `ocr/`
  - `app/main.py`: OCR API endpoints
  - `app/engine_ppocr.py`: PaddleOCR engine wrapper
  - `app/postproc.py`: OCR normalization and line grouping
  - `app/schemas.py`: Pydantic request/response models
- `inference/`
  - `app/main.py`: detection API endpoints
  - `app/inference.py`: TensorRT RT-DETR wrapper
  - `export_onnx.py`: export utility
  - `requirements*.txt`: runtime/export dependencies
- `taskPlanner/`
  - `main.py`: planner API, prompt construction, LLM call path, retries, validation
  - `tools.py`: documented planner tool contract
  - `requirements.txt`: planner dependencies
- `tasks/`: static-mode YAML workflows
- `profiles/`: mounted Firefox profiles and caches
- `planner-dumps/`: optional dumped planner requests
- `agent-debug/`: optional trace/debug output
- `tools/mock_inference_server/`: Triton mock utilities
- `scripts/`: trace helpers, planner replay utilities, and model-comparison scripts

---

## Quickstart
1. Generate the VNC password file in `agent/passwd`.
2. Run `docker compose up --build -d`.
3. Open a VNC viewer to `localhost:25901` or `localhost:25902`.
4. Watch the dynamic agent run the goal defined in `docker-compose.yml`.
5. Inspect `agent-debug/` and `planner-dumps/` when debugging planner or OCR behavior.
