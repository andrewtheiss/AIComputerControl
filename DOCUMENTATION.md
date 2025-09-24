AIComputerControl – System Documentation

## Purpose
This repository runs desktop UI automation agents inside GPU-enabled containers. Each agent sees a virtual Linux desktop (XFCE over VNC), understands the UI via OCR and a centralized object detection API, and acts by sending keyboard/mouse events. Agents can either follow a static YAML task script or operate dynamically by consulting a planning microservice backed by a local LLM (via an OpenAI-compatible API such as Ollama).

## High-level Architecture
- **Inference API (`inference/`)**: FastAPI service wrapping a TensorRT RT-DETR engine. Accepts screenshots and returns detection boxes.
- **Agent (`agent/`)**: Headful Linux desktop with Firefox. Captures screen, runs OCR, calls Inference API, and executes actions via `xdotool`. Two modes:
  - Static: executes YAML steps from `tasks/*.yaml`.
  - Dynamic: consults the planner to decide next actions based on OCR and history.
- **Task Planner (`taskPlanner/`)**: FastAPI service that receives the agent’s goal, OCR results, and history, and returns the next action+parameters. Uses `taskgen-ai` with an OpenAI-compatible endpoint (Ollama by default).
- **Orchestration (`docker-compose.yml`)**: Brings up all services on a shared Docker network with GPU isolation and agent-specific Firefox profiles mounted from the host.

Data flow (dynamic mode):
1) Agent captures screen (mss) → runs OCR (Tesseract) and optionally calls Inference API (RT‑DETR) for object boxes.
2) Agent posts goal, OCR, and recent history to Task Planner.
3) Task Planner returns a single action (e.g., `click_text`, `wait_any_text`, `type_text`).
4) Agent executes action via `xdotool`, records result, loops.

---

## Services and Components

### Inference API (`inference/app`)
- Entrypoint: `app/main.py`
  - Endpoints:
    - `GET /health` → `{ "status": "ok" }`
    - `POST /predict` → `{"detections":[{"box":[x1,y1,x2,y2],"score":float,"label":int}, ...]}`
  - Reads model path from `ENGINE_PATH` (default `/app/models/rtdetr-l.engine`).
- Engine: `app/inference.py` (`RTDETREngine`)
  - Loads TensorRT engine once; thread-safe via a lock.
  - Supports TensorRT legacy (v2) and tensors API (v3).
  - Preprocess: resize to 640×640, RGB, normalize to [0,1], NCHW, CUDA tensor.
  - Postprocess: handles several output layouts (N×6 or boxes+logits); maps normalized center-based boxes to absolute pixel coordinates.
  - Uses its own CUDA stream and synchronizes appropriately.
- Docker: `inference/Dockerfile`
  - Based on NVIDIA PyTorch image, installs FastAPI deps and the app under `/opt/app`.
  - Exposes port 8000; started via uvicorn.
- Models & utilities:
  - Model artifacts: `inference/models/rtdetr-l.engine` (mounted into image on build).
  - Optional exporter: `inference/export_onnx.py` (ONNX export via `ultralytics RTDETR`).
  - Requirements: `inference/requirements.txt` for API runtime, `requirements_export.txt` for export tooling.

Usage: Send a JPEG image via multipart form field `file`. The response contains `detections` with boxes in image pixel coordinates.

### Agent (`agent/src/agent.py`)
Two execution modes (selected by `AGENT_MODE`):

1) Static runner: `TaskRunner`
- Loads YAML from `AGENT_TASK` (default `/tasks/{AGENT_NAME}.yaml`).
- Continuously watches for changes and re-loads the YAML.
- Provides primitives:
  - Navigation and waits: `open_url`, `wait_text`, `wait_detection`, `detect`, `sleep`
  - OCR interactions: `click_text` (regex/fuzzy), `ocr_extract`
  - Detection interactions: `click_detection` (by label/score)
  - Input: `type_text`, `key_seq`
  - Control flow: `if`, `for_each`, `for_pages`, `for_questions`
  - Utilities: `set_var`, `track_series`, `load_json`, `run_llm`, `screenshot`, `debug_dump_ctx`
- Screen capture via `mss` (configurable monitor index `AGENT_SCREEN_INDEX`).
- OCR via Tesseract (`pytesseract`); basic quality filtering.
- Detection via HTTP POST to RT‑DETR API (`RTDETR_API_URL`).
- Input via `xdotool` (guarded by `CLICK_ENABLED`).
- Debugging: saves images/overlays and step logs under `${AGENT_DEBUG_DIR}/${AGENT_NAME}-${RUN_ID}` when `AGENT_DEBUG=1`.

2) Dynamic executor: `ActionExecutorDynamic`
- Observe→Decide→Act loop for up to `MAX_STEPS`.
- Captures OCR results (prioritized by height/confidence and capped by `OCR_LIMIT`).
- Posts to planner: `{ goal, task_history[-HISTORY_WINDOW:], ocr_results, available_actions }`.
- Normalizes planner parameters (e.g., `text`→`regex`, `timeout`→`timeout_s`).
- Executes allowed actions: `open_url`, `wait_text`, `wait_any_text`, `click_text`, `click_any_text`, `click_near_text`, `type_text`, `key_seq`, `sleep`, `ocr_extract`, `end_task`, `run_llm`.
- Robust key normalization (e.g., `CTRL + ENTER` → `ctrl+Return`).

Common agent configuration (environment variables):
- `AGENT_MODE` = `static` | `dynamic`
- `AGENT_GOAL` (dynamic mode): high-level instruction string
- `TASK_PLANNER_URL` (dynamic): planner endpoint (default `http://localhost:8000/v1/actions/next`)
- `PLANNER_API_KEY` (optional): bearer token for planner
- `RTDETR_API_URL`: inference endpoint (default `http://rtdetr-api:8000/predict`)
- `AGENT_NAME`: used to pick default task file
- `AGENT_TASK`: absolute path to YAML (defaults to `/tasks/{AGENT_NAME}.yaml`)
- `CLICK_ENABLED` = `1|0`
- `AGENT_SCREEN_INDEX` (mss monitor index)
- `AGENT_DEBUG` and `AGENT_DEBUG_DIR`
- `FIREFOX_BIN` and `FIREFOX_PROFILE_PATH` to pin the browser and profile
- Optional LLM for `run_llm`: `LLM_API_URL` (OpenAI-compatible), `LLM_MODEL`, `LLM_API_KEY`

Agent container and desktop:
- Dockerfile installs XFCE, TigerVNC, Firefox-ESR, `xdotool`, Tesseract, and Python deps.
- VNC password is pre-generated on host and copied into the image (`agent/passwd`).
- On container start, VNC launches display `:1`, starts XFCE, and runs `agent.py`.
- Connect to the agent via VNC on the mapped host port (e.g., `localhost:5901`).

### Task Planner (`taskPlanner/`)
- Entrypoint: `main.py` (FastAPI) exposes `POST /v1/actions/next`.
- Security: optional bearer token via `PLANNER_API_KEY`.
- Models: strict Pydantic schemas for OCR results, history, and the planner response.
- LLM backend: `taskgen-ai` Agent using `AsyncOpenAI` client pointed at `OLLAMA_OPENAI_BASE` (default `http://host.docker.internal:11434/v1`). Model via `OLLAMA_MODEL`.
- System prompt encodes robust UI planning rules (e.g., Gmail compose recipe, synonym handling, confirmation after clicks).
- Tools: The planner communicates its tool contract via docstrings in `taskPlanner/tools.py` (documentation-only shims). The executor enforces the real behavior.
- Parameter normalization ensures consistency before returning to agent.
- Dockerfile: Python 3.12 slim, installs requirements and `taskgen-ai`, runs uvicorn on 8000.

Planner environment:
- `PLANNER_API_KEY`: if set, required as `Authorization: Bearer {key}`
- `OLLAMA_OPENAI_BASE`, `OLLAMA_MODEL`: OpenAI-compatible base and model name

---

## Orchestration (`docker-compose.yml`)
- `rtdetr-api` (GPU 1 by example): builds from `inference/`, exposes `8000:8000`, sets `ENGINE_PATH`, `NVIDIA_*` envs, and ulimits. Network: `ai-net`.
- `vnc-instance-1` and `vnc-instance-2` (GPU 0): build from `agent/` (image `ai-sandbox`), expose VNC ports (`5901`, `5902`), mount `./tasks` into `/tasks`, and mount host Firefox profile/cache to persist cookies and session data. Key env:
  - `AGENT_MODE=dynamic`
  - `RTDETR_API_URL` → `http://rtdetr-api:8000/predict`
  - `TASK_PLANNER_URL` → `http://task-planner:8000/v1/actions/next`
  - `PLANNER_API_KEY`, `FIREFOX_BIN`, `FIREFOX_PROFILE_PATH`, `AGENT_GOAL`, login secrets
  - `depends_on`: `rtdetr-api`, `task-planner`
- `task-planner`: builds from `taskPlanner/`, exposes `8010:8000`, configured to reach Ollama at `host.docker.internal:11434` with `extra_hosts` for Linux compatibility.
- Shared bridge network `ai-net` connects all services by name (service DNS).

GPU notes: `device_ids` select GPUs per service. Adjust per your hardware. The compose file shows an example split (planner CPU-only; inference on GPU 1; agents on GPU 0 for xdotool and desktop rendering).

---

## Configuration and Profiles

### Tasks YAML (`/tasks/*.yaml`)
Static mode uses YAML files like `tasks/agent-1.yaml`. Structure:
- `version`: optional metadata
- `agent`: human-readable agent id
- `loop`: whether to loop after completing steps
- `steps`: ordered list where each item is a single-key mapping `{op: args}`

Common ops (see Agent section for full list): `open_url`, `wait_text`, `click_text`, `type_text`, `key_seq`, `sleep`, `ocr_extract`, `run_llm`, `if`, `for_each`, `screenshot`, etc.

Variable substitution: `${ENV_VAR}` is supported in many string arguments and resolves from environment or the runtime context `ctx`.

Known divergences in provided examples:
- `tasks/agent-1.yaml`: `click_text` includes `fuzzy` parameter in some steps; the static runner supports `fuzzy_text` and `fuzzy_threshold` instead. The extra key is harmless but ignored.
- `ocr_extract` contains a `capture` key which is not used by the current implementation; `save_as` is honored.
- `tasks/agent-2.yaml` uses `extract_grade_table`, which is not implemented. The agent will log `op.unknown` for this step. Consider implementing a helper or removing the step.

### Firefox profiles (`profiles/agent-*`)
- Compose mounts host Firefox profile and cache into each agent container to persist sessions/cookies.
- If you need to initialize these folders, you can copy them from a running container once and then mount them on subsequent runs.

### Secrets
- Environment variables like `SECRETS_EMAIL` and `SECRETS_PASSWORD` can be passed into agent containers and referenced by YAML using `${SECRETS_EMAIL}`.
- Avoid hardcoding secrets into YAML files; prefer env injection.

---

## APIs

### Inference API
- `POST /predict`
  - Multipart form field: `file` (JPEG/PNG image bytes)
  - Response:
    ```json
    {
      "detections": [
        {"box": [100.0,150.0,200.0,250.0], "score": 0.95, "label": 17}
      ]
    }
    ```
- `GET /health` → `{ "status": "ok" }`

### Task Planner API
- `POST /v1/actions/next` (optionally requires bearer token)
  - Request (abridged):
    ```json
    {
      "goal": "Wait for Gmail inbox and send 5 emails...",
      "task_history": [{"action":"click_text","parameters":{},"result":{"status":"success"}}],
      "ocr_results": [{"text":"Inbox","box":[x1,y1,x2,y2],"conf":97}],
      "available_actions": ["open_url","wait_text", "click_text", "type_text", "key_seq", "sleep", "end_task"]
    }
    ```
  - Response (strict JSON):
    ```json
    {
      "action": "click_text",
      "parameters": {"regex": "^Compose$", "nth": 0},
      "reasoning": "Compose is visible; opening new message.",
      "completed": false
    }
    ```

---

## Building, Running, and Scaling

Prerequisites on host:
- Windows 11 with WSL2 + Docker Desktop (WSL backend).
- Recent NVIDIA drivers and NVIDIA Container Toolkit for GPU access.
- VNC viewer (e.g., TigerVNC Viewer).

One-time VNC password (host):
```bash
sudo apt-get update && sudo apt-get install -y tigervnc-tools
cd agent/
vncpasswd passwd  # creates agent/passwd used by agent Dockerfile
```

Bring everything up:
```bash
docker compose up --build -d
```

Connect to agents:
- `vnc-instance-1` → `localhost:5901`
- `vnc-instance-2` → `localhost:5902`

Scale agents:
```bash
# Example pattern if using a single service with replicas.
# In this repository, two named services are defined; duplicate or templatize as needed.
docker compose up -d --scale vnc-instance-1=1 --scale vnc-instance-2=1
```

View logs:
```bash
docker compose logs -f rtdetr-api
docker compose logs -f vnc-instance-1
docker compose logs -f task-planner
```

Shutdown:
```bash
docker compose down
```

---

## Testing and Utilities

### Inference API test
`test/test_client.py` posts `test/test.jpg` to `http://localhost:8000/predict` and prints the JSON response.

### Mock inference server (Triton gRPC)
`tools/mock_inference_server/` contains a small GRPC service that mocks Triton’s `ModelInfer`. This is not used by the main application (which uses HTTP FastAPI), but can be repurposed if you move to Triton GRPC.

---

## Extending the System

- Add new planner tools: document the function signature and intent in `taskPlanner/tools.py`, then allow the executor to implement it (Dynamic mode) or add a new op in `TaskRunner` (Static mode).
- Add new static ops: extend `TaskRunner._do_steps` with your operation, keep args normalized, and update the documentation.
- Swap models: rebuild the inference container with a new `ENGINE_PATH`. Maintain the expected output shapes or adapt postprocessing in `RTDETREngine`.
- Customize browsers: change `FIREFOX_BIN` and mount specific profiles per agent in compose.

---

## Notes and Limitations

- `extract_grade_table` step in `tasks/agent-2.yaml` is a placeholder and not implemented.
- Some example YAML keys (e.g., `fuzzy` under `click_text`, or `capture` under `ocr_extract`) are ignored by the current implementation. Use `fuzzy_text`/`fuzzy_threshold` instead.
- `xdotool` interactions require X11 focus inside the agent desktop; if clicks are skipped, ensure `CLICK_ENABLED=1` and `xdotool` is present.
- Multi-monitor capture uses `mss` monitor indices; set `AGENT_SCREEN_INDEX` accordingly.

---

## Repository Map

- `docker-compose.yml`: Orchestrates services and networks.
- `agent/`
  - `Dockerfile`: Desktop + VNC + Firefox + agent runtime.
  - `passwd`: Pre-generated VNC password file (copied to `/root/.vnc/passwd`).
  - `src/agent.py`: Static `TaskRunner` and dynamic `ActionExecutorDynamic` implementations.
- `inference/`
  - `Dockerfile`: RT‑DETR FastAPI server.
  - `app/main.py`: API endpoints.
  - `app/inference.py`: TensorRT wrapper.
  - `models/`: TensorRT engine file(s).
  - `requirements*.txt`: Dependencies.
  - `export_onnx.py`: ONNX export helper.
- `taskPlanner/`
  - `Dockerfile`: Planner API service.
  - `main.py`: Planner endpoint and LLM/tooling integration.
  - `tools.py`: Documented tool functions (schema for LLM), not executed here.
  - `requirements.txt`: Dependencies (FastAPI, taskgen-ai, openai client, etc.).
- `tasks/`: YAML tasks per agent (used in static mode).
- `profiles/`: Host-side Firefox profiles and caches to be mounted per agent.
- `tools/mock_inference_server/`: Triton GRPC mock server samples.
- `test/`: Simple Inference API client and sample image.

---

## Quickstart
1) Generate VNC password (`agent/passwd`) as shown above.
2) `docker compose up --build -d`
3) Open `localhost:5901` in a VNC viewer and watch the agent act. Adjust goals/YAML and environment variables in `docker-compose.yml` as needed.


