# task_planner/main.py
import os, json
import orjson
import logging
import base64
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi import Request
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import AsyncOpenAI

from taskgen import Agent, Function   # documented usage
from tools import ALL_TOOLS

load_dotenv()

# --- Logging ---
LOG_LEVEL = os.getenv("PLANNER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("task-planner")

LOG_REQUESTS = os.getenv("PLANNER_LOG_REQUESTS", "0").strip() == "1"
LOG_DECISIONS = os.getenv("PLANNER_LOG_DECISIONS", "1").strip() == "1"
LOG_LLM_PROMPT = os.getenv("PLANNER_LOG_LLM_PROMPT", "0").strip() == "1"   # may include OCR/history
LOG_LLM_OUTPUT = os.getenv("PLANNER_LOG_LLM_OUTPUT", "0").strip() == "1"   # raw model text (truncated)
LOG_TRUNC_CHARS = int(os.getenv("PLANNER_LOG_TRUNC_CHARS", "1200"))

# Optional request dumping (sanitized JSON + decoded screenshot file)
DUMP_DIR = os.getenv("PLANNER_DUMP_REQUESTS_DIR", "").strip()
MAX_DUMPS = int(os.getenv("PLANNER_DUMP_MAX_FILES", "200"))

# --- OpenAI-compatible client (local or remote) ---
# Point to your server base (e.g., http://127.0.0.1:1234/v1) and model (e.g., qwen3.5)
OLLAMA_OPENAI_BASE = os.getenv("OLLAMA_OPENAI_BASE", "http://127.0.0.1:1234/v1")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen3.5")
OPENAI_DUMMY_KEY   = os.getenv("OPENAI_API_KEY", "local")  # many local servers ignore the key but SDK requires it
client = AsyncOpenAI(base_url=OLLAMA_OPENAI_BASE, api_key=OPENAI_DUMMY_KEY)

PLANNER_API_KEY = os.getenv("PLANNER_API_KEY", "")

# --- Security (simple bearer) ---
def verify_key(authorization: Optional[str] = Header(None)):
    if not PLANNER_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1] != PLANNER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# --- Pydantic models for strict contract ---
class OCRResult(BaseModel):
    text: str
    box: List[int]
    conf: float
    level: Optional[str] = None

class HistoryItem(BaseModel):
    action: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]

class UIElement(BaseModel):
    """
    A compact, screen-space UI element description produced by the executor.
    Coordinates are image-relative [x1,y1,x2,y2] for the current screenshot frame.
    """
    source: str  # "ocr_word" | "ocr_line" | "det" | "ax"
    text: str
    box: List[int]
    score: float
    role: Optional[str] = None

class PlannerRequest(BaseModel):
    goal: str
    task_history: List[HistoryItem] = Field(default_factory=list)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    ui_elements: List[UIElement] = Field(default_factory=list)
    screenshot_b64: Optional[str] = None
    screenshot_mime: str = "image/jpeg"
    available_actions: List[str] = Field(default_factory=list)

class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    parameters: Dict[str, Any]
    reasoning: str
    completed: bool = False

# --- TaskGen agent setup ---
# LLM wrapper with TaskGen’s required signature
async def tg_llm(user_prompt: str, system_prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.2,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content

planner_agent = Agent(
    name="UI Task Planner",
    description="Decide the next valid UI action for a high-level goal given OCR and history.",
    llm=tg_llm
)
# Inform the LLM of the tool schema via TaskGen (names + docstrings matter)
planner_agent.assign_functions([Function(external_fn=fn) for fn in ALL_TOOLS])

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are a precise planner for a versatile automation agent running in a full VM environment.
You must output STRICT JSON ONLY with keys: action, parameters, reasoning, completed.

action: one of the provided tool names.
parameters: dict of args for that tool.
reasoning: 1-3 sentences explaining the choice using GOAL, current OCR/state, and last action result.
completed: true only when the full GOAL is done or impossible.

Rules:

Use only the actions announced by the caller.
Leverage the full VM access: open and interact with browsers (e.g., Firefox), terminals, text editors, or any other apps as needed to achieve the GOAL.
Treat any previous action with status=dispatched, status=failure, or outcome_verified=false as unresolved. It is NOT a soft success.
Do not assume a click, keypress, or typing step changed the UI unless history says outcome_verified=true or the current OCR/screenshot clearly proves the new state.
If same_action_same_state_streak is greater than 1 or verification evidence says no_observable_change / OUTCOME_NOT_VERIFIED, do not repeat the same action family again immediately. Force a strategy change.
If current_state.tags or recent history indicates any blocker:* tag, clear that blocker before normal goal progression. Do not continue with task-level actions while blocker tags remain present.
Common blockers include browser suggestion overlays, browser chrome dropdowns, modals, cookie banners, permission prompts, session restore prompts, and similar interruptions.
Distinguish browser page content from browser chrome. Suggestion overlays, URL bar dropdowns, browser dialogs, and session restore are not the destination page.
Use current_state.blockers as the source of truth when available. Respect blocker class, scope, visible resolve_targets, allow_page_click_through, and suggested_strategies.
If a blocker exposes visible resolve_targets, prefer those over generic recovery guesses.
If a blocker's recovery options show prior attempts for a strategy, do not keep reusing that strategy unless the state clearly changed.
If blocker scope is browser_chrome and allow_page_click_through=true, a click on visible page content may be a valid dual-purpose move that both dismisses the chrome blocker and progresses the task.
If blocker scope is browser_page, page_overlay, or modal, do not assume underlying page targets are legal until the blocker is resolved or a visible dual-purpose target explicitly indicates otherwise.
When current_state or history exposes focused_role / focused_name / focused_editable, use that as strong evidence for whether typing or keyboard navigation is appropriate.
Do not type unless focus is likely correct for the intended field. If focus is ambiguous, first refocus or verify it.
Do not press Enter unless focus or the visible state strongly supports submit/search behavior. Avoid Enter when browser chrome or overlays may have focus.
Never reference UI elements not present in OCR or current state.
For web tasks, open browser if not already, navigate URLs, use search bars, click links/buttons via text matches or positions.
For file operations, prefer terminal (e.g., echo, cat, touch) or open a text editor like gedit/vi to create/edit/save files.
Handle interruptions like dialogs, popups, or errors by reading OCR text and responding appropriately (e.g., click 'OK', close window, or adjust strategy).
After major actions (e.g., navigation, app switch, dismissing popups), insert short sleep(0.5-1.5s) and confirm state with wait_any_text() or similar before proceeding.
For comparisons or data collection (e.g., prices), track info mentally across steps; use files or terminal output if needed for persistence.
If matching UI by text, use synonyms/variations with wait_any_text() or click_any_text(); for icons, use click_near_text() with anchors.
If OCR or state is ambiguous, prefer verification or recovery actions over confident progression.
Prioritize efficient paths: use keyboard shortcuts only after confirming focus; switch between apps/windows as required."""

def _tool_specs_for_actions(actions: List[str]) -> str:
    """
    Build a compact tool spec block from docstrings in tools.py for ONLY the
    actions announced by the caller, to reduce hallucinated parameters.
    """
    want = set(actions or [])
    blocks: List[str] = []
    for fn in ALL_TOOLS:
        name = getattr(fn, "__name__", "")
        if name and name in want:
            doc = (getattr(fn, "__doc__", "") or "").strip()
            if doc:
                blocks.append(f"## {name}\n{doc}")
            else:
                blocks.append(f"## {name}\n(No docstring.)")
    return "\n\n".join(blocks) if blocks else "(No tool specs available.)"

def _format_history_item(h: HistoryItem) -> str:
    result = h.result or {}
    status = result.get("status", "unknown")
    verification = result.get("verification") or {}
    before_state = result.get("before_state") or {}
    after_state = result.get("after_state") or {}
    evidence = ", ".join((verification.get("evidence") or [])[:3])
    parts = [f"status={status}"]
    if "event_applied" in result:
        parts.append(f"event_applied={result.get('event_applied')}")
    if "outcome_verified" in result:
        parts.append(f"outcome_verified={result.get('outcome_verified')}")
    if result.get("same_action_same_state_streak") is not None:
        parts.append(f"same_state_streak={result.get('same_action_same_state_streak')}")
    if result.get("blocker_class"):
        parts.append(f"blocker_class={result.get('blocker_class')}")
    if result.get("recovery_strategy"):
        parts.append(f"recovery_strategy={result.get('recovery_strategy')}")
    if result.get("recovery_effect"):
        parts.append(f"recovery_effect={result.get('recovery_effect')}")
    if before_state.get("tags"):
        parts.append(f"before_tags={before_state.get('tags')}")
    if after_state.get("tags"):
        parts.append(f"after_tags={after_state.get('tags')}")
    if before_state.get("focused_role") or before_state.get("focused_name"):
        parts.append(f"before_focus={before_state.get('focused_role') or before_state.get('focused_name')}")
    if after_state.get("focused_role") or after_state.get("focused_name"):
        parts.append(f"after_focus={after_state.get('focused_role') or after_state.get('focused_name')}")
    if verification.get("reason"):
        parts.append(f"verification={verification.get('reason')}")
    if evidence:
        parts.append(f"evidence={evidence}")
    if result.get("error_code"):
        parts.append(f"error_code={result.get('error_code')}")
    return f"- {h.action}({h.parameters}) => " + "; ".join(parts)

def _last_unresolved_history_item(history: List[HistoryItem]) -> str:
    for h in reversed(history or []):
        result = h.result or {}
        if result.get("status") != "success" or result.get("outcome_verified") is False:
            return _format_history_item(h)
    return "None."

def make_user_prompt(req: PlannerRequest) -> str:
    hist = "\n".join(_format_history_item(h) for h in req.task_history[-5:]) or "No prior actions."
    cur_state = req.current_state or {}
    blockers = [str(tag) for tag in (cur_state.get("tags") or []) if str(tag).startswith("blocker:")]
    blocker_details = cur_state.get("blockers") or []
    recovery_options = cur_state.get("recovery_options") or []
    recent_blocker_attempts = cur_state.get("recent_blocker_attempts") or []
    visible_resolve_targets = cur_state.get("visible_resolve_targets") or []
    last_unresolved = _last_unresolved_history_item(req.task_history)
    cur_state_line = (
        f"hash={cur_state.get('hash','')}, blocker_signature={cur_state.get('blocker_signature','')}, "
        f"tags={cur_state.get('tags', [])}, top_texts={cur_state.get('top_texts', [])}, "
        f"focused_role={cur_state.get('focused_role','')}, focused_name={cur_state.get('focused_name','')}, focused_editable={cur_state.get('focused_editable', False)}"
        if cur_state else "No executor state summary."
    )
    ocr = "\n".join(
        f"- [{r.level or 'ocr'}] {r.text} @ {r.box} (conf={int(r.conf)})"
        for r in req.ocr_results[:300]
    ) or "No OCR text."
    elems = "\n".join(
        f"- [{e.source}{'/' + e.role if e.role else ''}] {e.text} @ {e.box} (score={e.score:.2f})"
        for e in req.ui_elements[:200]
    ) or "No UI elements."
    actions = ", ".join(req.available_actions) or "[]"
    tool_specs = _tool_specs_for_actions(req.available_actions)
    has_shot = "yes" if (req.screenshot_b64 and str(req.screenshot_b64).strip()) else "no"
    return (
        f"GOAL:\n{req.goal}\n\n"
        f"AVAILABLE_ACTIONS: {actions}\n\n"
        f"HAS_SCREENSHOT: {has_shot}\n\n"
        f"CURRENT_STATE:\n{cur_state_line}\n\n"
        f"ACTIVE_BLOCKERS:\n{blockers or []}\n\n"
        f"BLOCKER_DETAILS:\n{json.dumps(blocker_details, ensure_ascii=False, indent=2) if blocker_details else '[]'}\n\n"
        f"VISIBLE_RESOLVE_TARGETS:\n{json.dumps(visible_resolve_targets, ensure_ascii=False, indent=2) if visible_resolve_targets else '[]'}\n\n"
        f"RECOVERY_OPTIONS:\n{json.dumps(recovery_options, ensure_ascii=False, indent=2) if recovery_options else '[]'}\n\n"
        f"RECENT_BLOCKER_ATTEMPTS:\n{json.dumps(recent_blocker_attempts, ensure_ascii=False, indent=2) if recent_blocker_attempts else '[]'}\n\n"
        f"LAST_UNRESOLVED_ACTION:\n{last_unresolved}\n\n"
        f"RECENT_HISTORY:\n{hist}\n\n"
        f"CURRENT_OCR:\n{ocr}\n\n"
        f"UI_ELEMENTS (image-relative boxes):\n{elems}\n\n"
        f"TOOL_SPECS:\n{tool_specs}\n\n"
        f"Return strict JSON only."
    )

app = FastAPI(title="TaskPlanner API", version="1.0.0")

@app.middleware("http")
async def log_planner_requests(request: Request, call_next):
    """
    Optional request logging for /v1/actions/next. By default, logs only a safe
    summary (counts + screenshot sizes) and never logs screenshot contents.
    Enable with PLANNER_LOG_REQUESTS=1.
    """
    if request.url.path == "/v1/actions/next" and LOG_REQUESTS:
        try:
            body = await request.body()
            # Ensure downstream can still read the body (FastAPI dependencies / parsing).
            request._body = body  # type: ignore[attr-defined]

            summary: Dict[str, Any] = {"bytes": len(body)}
            try:
                obj = json.loads(body.decode("utf-8", errors="replace"))
                shot = (obj.get("screenshot_b64") or "")
                summary.update({
                    "goal_len": len(obj.get("goal") or ""),
                    "task_history_n": len(obj.get("task_history") or []),
                    "ocr_results_n": len(obj.get("ocr_results") or []),
                    "ui_elements_n": len(obj.get("ui_elements") or []),
                    "available_actions_n": len(obj.get("available_actions") or []),
                    "has_screenshot": bool(shot.strip()),
                    "screenshot_b64_len": len(shot) if isinstance(shot, str) else None,
                    "screenshot_mime": obj.get("screenshot_mime"),
                })

                # Optional dump to disk for replay debugging.
                if DUMP_DIR:
                    dump_dir = Path(DUMP_DIR)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    ts_ms = int(time.time() * 1000)
                    rid = uuid.uuid4().hex[:10]
                    base = dump_dir / f"{ts_ms}_{rid}"

                    # Save decoded screenshot to file (if present).
                    screenshot_path = None
                    if isinstance(shot, str) and shot.strip():
                        try:
                            data_url = shot.strip()
                            mime = (obj.get("screenshot_mime") or "image/jpeg").strip() or "image/jpeg"
                            b64_part = data_url
                            if data_url.startswith("data:"):
                                # data:<mime>;base64,<b64>
                                comma = data_url.find(",")
                                b64_part = data_url[comma + 1 :] if comma != -1 else ""
                            img_bytes = base64.b64decode(b64_part, validate=False)
                            ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png" if "png" in mime else ".img"
                            screenshot_path = str(base.with_suffix(ext).name)
                            (base.with_suffix(ext)).write_bytes(img_bytes)
                        except Exception as e:
                            logger.warning("planner.dump_screenshot_failed %s", {"error": str(e)})

                    # Write a sanitized JSON (no base64 inline).
                    obj_san = dict(obj)
                    if screenshot_path:
                        obj_san["screenshot_path"] = screenshot_path
                    obj_san["screenshot_b64"] = None
                    (base.with_suffix(".json")).write_text(json.dumps(obj_san, indent=2), encoding="utf-8")

                    # Best-effort cleanup of oldest dumps.
                    try:
                        if MAX_DUMPS > 0:
                            files = sorted(dump_dir.glob("*"), key=lambda p: p.stat().st_mtime)
                            if len(files) > MAX_DUMPS:
                                for p in files[: max(0, len(files) - MAX_DUMPS)]:
                                    try:
                                        p.unlink()
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            except Exception as e:
                summary["json_parse_error"] = str(e)

            client_ip = getattr(getattr(request, "client", None), "host", None)
            if client_ip:
                summary["client_ip"] = client_ip
            logger.info("planner.request %s", summary)
        except Exception as e:
            logger.warning("planner.request_log_failed %s", {"error": str(e)})

    return await call_next(request)

def _data_url_from_b64(b64_or_data_url: str, mime: str) -> str:
    s = (b64_or_data_url or "").strip()
    if not s:
        return ""
    if s.startswith("data:"):
        return s
    m = (mime or "image/jpeg").strip() or "image/jpeg"
    return f"data:{m};base64,{s}"

async def call_planner_llm(req: PlannerRequest, system_prompt: str) -> str:
    """
    Call the OpenAI-compatible backend. If a screenshot is provided, use the
    multimodal chat.completions message format (text + image_url). If the backend
    rejects multimodal, fall back to text-only.
    """
    t0 = time.time()
    user_prompt = make_user_prompt(req)
    if LOG_LLM_PROMPT:
        logger.info(
            "planner.user_prompt %s",
            {
                "chars": len(user_prompt),
                "trunc": user_prompt[:LOG_TRUNC_CHARS],
            },
        )

    if req.screenshot_b64 and str(req.screenshot_b64).strip():
        data_url = _data_url_from_b64(req.screenshot_b64, req.screenshot_mime)
        try:
            logger.info("planner.llm_call %s", {"multimodal": True, "model": OLLAMA_MODEL})
            resp = await client.chat.completions.create(
                model=OLLAMA_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            out = resp.choices[0].message.content or ""
            logger.info("planner.llm_done %s", {"multimodal": True, "model": OLLAMA_MODEL, "elapsed_ms": int((time.time() - t0) * 1000), "out_chars": len(out)})
            if LOG_LLM_OUTPUT:
                logger.info("planner.llm_output %s", {"trunc": out[:LOG_TRUNC_CHARS]})
            return out
        except Exception:
            # fall back to text-only below
            pass

    logger.info("planner.llm_call %s", {"multimodal": False, "model": OLLAMA_MODEL})
    resp = await client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = resp.choices[0].message.content or ""
    logger.info("planner.llm_done %s", {"multimodal": False, "model": OLLAMA_MODEL, "elapsed_ms": int((time.time() - t0) * 1000), "out_chars": len(out)})
    if LOG_LLM_OUTPUT:
        logger.info("planner.llm_output %s", {"trunc": out[:LOG_TRUNC_CHARS]})
    return out

def normalize_params(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    if "timeout" in p and "timeout_s" not in p:
        p["timeout_s"] = p.pop("timeout")
    if action == "click_text" and "text" in p and "regex" not in p:
        p["regex"] = p.pop("text")
    if action in ("wait_any_text", "click_any_text") and "texts" in p and "patterns" not in p:
        p["patterns"] = p.pop("texts")
    if action == "click_near_text" and "anchor" in p and "anchor_regex" not in p:
        p["anchor_regex"] = p.pop("anchor")
    if action == "click_box" and "bbox" in p and "box" not in p:
        p["box"] = p.pop("bbox")
    return p

@retry(wait=wait_exponential(min=0.5, max=4), stop=stop_after_attempt(3))
async def _decide(req: PlannerRequest) -> PlannerResponse:
    out = await call_planner_llm(req, SYSTEM_PROMPT)

    # Attempt strict JSON parse; if not JSON, try to extract a JSON object
    try:
        obj = json.loads(out)
    except json.JSONDecodeError:
        start = out.find("{"); end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(out[start:end+1])
        else:
            obj = {"action":"end_task",
                   "parameters":{"reason":"LLM did not return JSON"},
                   "reasoning":"Parser fallback",
                   "completed": True}

    # Validate action
    invalid_action = obj.get("action")
    if invalid_action not in req.available_actions:
        obj = {"action":"end_task",
               "parameters":{"reason":f"Invalid action {invalid_action}"},
               "reasoning":"Planner filtered to safe action set",
               "completed": True}

    # Planner-side parameter normalization to reduce executor work
    act = obj.get("action")
    obj["parameters"] = normalize_params(act, obj.get("parameters", {}))

    resp = PlannerResponse(**obj)
    if LOG_DECISIONS:
        logger.info(
            "planner.decision %s",
            {
                "action": resp.action,
                "parameters": resp.parameters,
                "completed": resp.completed,
                "reasoning_trunc": (resp.reasoning or "")[:400],
                "has_screenshot": bool((req.screenshot_b64 or "").strip()),
                "ocr_results_n": len(req.ocr_results or []),
                "ui_elements_n": len(req.ui_elements or []),
                "available_actions_n": len(req.available_actions or []),
            },
        )
    return resp

@app.post("/v1/actions/next", response_model=PlannerResponse, dependencies=[Depends(verify_key)])
async def next_action(req: PlannerRequest):
    try:
        return await _decide(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
