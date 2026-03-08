# scripts/agent.py
import os, re, time, json, threading, subprocess, io, uuid, shutil, traceback, hashlib
from ui_core import UIElement, Observation
from perception import fuse_observation
from decision import ComposeProposer, SendProposer, DismissModalProposer, arbitrate, Candidate
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from screen_signatures import make_signature
from blockers import classify_blockers

import requests
import yaml
import mss
import cv2
import numpy as np
import base64
from ocr_client import OCRClient
from rapidfuzz import fuzz, process as fuzz_process

# -----------------------
# Config
# -----------------------
# --- New config (top of file) ---
AGENT_MODE       = os.environ.get("AGENT_MODE", "static")
AGENT_GOAL       = os.environ.get("AGENT_GOAL", "").strip()
TASK_PLANNER_URL = os.environ.get("TASK_PLANNER_URL", "http://localhost:8000/v1/actions/next")
PLANNER_API_KEY  = os.environ.get("PLANNER_API_KEY", "")      # optional, see §3.4
MAX_STEPS        = int(os.environ.get("MAX_STEPS", "80"))
HISTORY_WINDOW   = int(os.environ.get("HISTORY_WINDOW", "10"))
OCR_LIMIT        = int(os.environ.get("OCR_LIMIT", "400"))     # cap tokens

DETECT_API_URL = os.environ.get("DETECT_API_URL") or os.environ.get("RTDETR_API_URL", "http://rtdetr-api:8000/predict")
AGENT_NAME = os.environ.get("AGENT_NAME", "agent-1")
TASK_FILE = os.environ.get("AGENT_TASK", f"/tasks/{AGENT_NAME}.yaml")
CLICK_ENABLED = os.environ.get("CLICK_ENABLED", "1") == "1"
SCREEN_INDEX = int(os.environ.get("AGENT_SCREEN_INDEX", "1"))  # mss monitor index
DEBUG_ENABLED = os.environ.get("AGENT_DEBUG", "1") == "1"
DEBUG_DIR = os.environ.get("AGENT_DEBUG_DIR", "/tmp/agent-debug")
CLICK_CROP_DEBUG = os.environ.get("AGENT_DUMP_CLICK_CROPS", "1") == "1"
TRACE_ENABLED = os.environ.get("AGENT_TRACE_ENABLED", "1") == "1"
TRACE_MAX = int(os.environ.get("AGENT_TRACE_MAX", "100"))
TRACE_FRAME_QUALITY = int(os.environ.get("AGENT_TRACE_FRAME_QUALITY", "70"))  # 25..95
POST_ACTION_VERIFY_TIMEOUT_S = float(os.environ.get("POST_ACTION_VERIFY_TIMEOUT_S", "2.2"))
POST_ACTION_VERIFY_POLL_S = float(os.environ.get("POST_ACTION_VERIFY_POLL_S", "0.35"))
POST_ACTION_VERIFY_STABLE_HITS = max(1, int(os.environ.get("POST_ACTION_VERIFY_STABLE_HITS", "2")))
NOOP_SIMILARITY_THRESHOLD = float(os.environ.get("AGENT_NOOP_SIMILARITY_THRESHOLD", "0.985"))
A11Y_BRIDGE_URL = os.environ.get("A11Y_BRIDGE_URL", "").strip()
A11Y_FETCH_TIMEOUT_S = float(os.environ.get("A11Y_FETCH_TIMEOUT_S", "0.45"))

# Optional LLM (OpenAI-compatible or local)
# Configure one of the following:
# - LLM_API_URL: Either a full endpoint (…/v1/chat/completions or …/v1/responses) OR the base URL (…/v1)
# - LLM_MODEL: Target model name (e.g., "qwen3.5" or vendor-specific alias)
# - LLM_API_KEY: Optional bearer token if your server requires it
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:1234/v1").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_API_MODE = os.environ.get("LLM_API_MODE", "auto").strip()  # "auto" | "responses" | "chat"

# -----------------------
# Utilities
# -----------------------
def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj))

def which(cmd: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        fp = os.path.join(d, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None

def redact_if_confidential(s: str, confidential: bool) -> str:
    return "[REDACTED]" if confidential else s


# -----------------------
# Screen capture
# -----------------------
class ScreenGrabber:
    """
    Wrap mss with cached monitor geometry so we can map image-space -> absolute desktop coords.
    """
    def __init__(self, screen_index: int = 1):
        self.screen_index = screen_index
        self.last_mon: Optional[Dict[str, int]] = None
        self._mss = mss.mss()

    def capture(self) -> Tuple[np.ndarray, Dict[str, int]]:
        mon = self._mss.monitors[self.screen_index]
        self.last_mon = mon
        # BGRA -> BGR
        raw = np.array(self._mss.grab(mon))
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return img, mon

    def to_abs(self, x_rel: int, y_rel: int) -> Tuple[int, int]:
        """
        Convert image (monitor-relative) coords -> absolute desktop coords
        """
        if not self.last_mon:
            raise RuntimeError("Screen geometry not captured yet.")
        return int(self.last_mon["left"] + x_rel), int(self.last_mon["top"] + y_rel)

# -----------------------
# OCR (HTTP-first with fallback)
# -----------------------
OCR_API_URL = os.environ.get("OCR_API_URL", "http://ocr-api:8020/ocr").strip()
_ocr_client = OCRClient(url=OCR_API_URL, min_score=float(os.environ.get("OCR_MIN_SCORE", "0.45")))

def ocr_image(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Return OCR words with boxes using OCRClient (HTTP preferred, Tesseract fallback)."""
    return _ocr_client.ocr(image_bgr)

def ocr_image_levels(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Return OCR word + line boxes using OCRClient (HTTP preferred, Tesseract fallback)."""
    return _ocr_client.ocr_levels(image_bgr)

def _score_boldish_height(box: List[int]) -> int:
    return box[3] - box[1]

def _text_target_prefers_lines(*targets: Optional[Union[str, List[str]]]) -> bool:
    for target in targets:
        if isinstance(target, list):
            if any(_text_target_prefers_lines(item) for item in target):
                return True
            continue
        s = str(target or "")
        if not s:
            continue
        if any(token in s for token in (" ", r"\s", r"\s+", r"\W")):
            return True
    return False

def _find_text_box_in_items(
    items: List[Dict[str, Any]],
    *,
    default_level: str,
    regex: Optional[str] = None,
    any_regex: Optional[List[str]] = None,
    fuzzy_text: Optional[str] = None,
    fuzzy_threshold: Optional[float] = None,
    prefer_bold: bool = False,
    nth: int = 0,
) -> Optional[Dict[str, Any]]:
    patterns = []
    if regex:
        patterns.append(re.compile(regex, re.I))
    if any_regex:
        patterns += [re.compile(r, re.I) for r in any_regex]

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for item in items or []:
        s = str(item.get("text", "") or "")
        match_score = None

        if patterns:
            if any(p.search(s) for p in patterns):
                match_score = 100.0
        elif fuzzy_text:
            score = fuzz.partial_ratio(fuzzy_text, s)
            if fuzzy_threshold is None or score >= float(fuzzy_threshold):
                match_score = float(score)

        if match_score is None:
            continue

        matched = dict(item)
        matched.setdefault("level", default_level)
        bonus = _score_boldish_height(matched["box"]) if prefer_bold else 0
        candidates.append((match_score + bonus, matched))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    idx = min(nth, len(candidates) - 1)
    return candidates[idx][1]

def find_text_box(
    ocr_items: List[Dict[str, Any]],
    line_items: Optional[List[Dict[str, Any]]] = None,
    regex: Optional[str] = None,
    any_regex: Optional[List[str]] = None,
    fuzzy_text: Optional[str] = None,
    fuzzy_threshold: Optional[float] = None,
    prefer_bold: bool = False,
    nth: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Find a text box by regex or fuzzy text.
    - For multi-word targets, line OCR is preferred and word OCR is used as fallback.
    - For single-word targets, word OCR is preferred and line OCR is used as fallback.
    - regex / any_regex: case-insensitive compiled patterns
    - fuzzy_text + fuzzy_threshold (0..100): RapidFuzz partial_ratio scoring
    """
    prefer_lines = _text_target_prefers_lines(regex, any_regex or [], fuzzy_text)
    search_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    if prefer_lines and line_items:
        search_groups.append(("line", line_items))
    if ocr_items:
        search_groups.append(("word", ocr_items))
    if not prefer_lines and line_items:
        search_groups.append(("line", line_items))

    for level, items in search_groups:
        match = _find_text_box_in_items(
            items,
            default_level=level,
            regex=regex,
            any_regex=any_regex,
            fuzzy_text=fuzzy_text,
            fuzzy_threshold=fuzzy_threshold,
            prefer_bold=prefer_bold,
            nth=nth,
        )
        if match:
            return match
    return None

# -----------------------
# RT-DETR client with retries
# -----------------------
def encode_jpeg_bgr(image_bgr: np.ndarray, q: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def rtdetr_detect(session: requests.Session, image_bgr: np.ndarray, timeout=(3.0, 6.0), retries: int = 2) -> List[Dict[str, Any]]:
    jpg = encode_jpeg_bgr(image_bgr)
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = session.post(DETECT_API_URL, files={"file": ("screen.jpg", jpg, "image/jpeg")}, timeout=timeout)
            r.raise_for_status()
            return r.json().get("detections", [])
        except Exception as e:
            last_err = e
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"RT-DETR request failed after {retries+1} attempts: {last_err}")

# -----------------------
# Input helpers (typing/clicking)
# -----------------------
def _have_xdotool() -> bool:
    return which("xdotool") is not None

def _safe_run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        pass

def type_text(text: str):
    # Use xdotool type; for multiline, split to be safe
    lines = text.splitlines()
    for i, line in enumerate(lines):
        _safe_run(["xdotool", "type", "--clearmodifiers", line])
        # Only add Return for explicit \n lines; do not append one at the end if the user didn't include it
        if i < len(lines) - 1:
            _safe_run(["xdotool", "key", "Return"])

def send_keys(keys: List[str]):
    for k in keys:
        _safe_run(["xdotool", "key", "--clearmodifiers", k])

def open_url(url: str):
    firefox_bin = os.environ.get("FIREFOX_BIN") or which("firefox-esr") or which("firefox")
    if not firefox_bin:
        if which("xdg-open"):
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        subprocess.Popen(["firefox", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    args = [firefox_bin]
    profile = os.environ.get("FIREFOX_PROFILE_PATH", "").strip()
    if profile:
        # Auto-create the Firefox profile directory if missing (ESR sometimes requires it)
        if not os.path.isdir(profile):
            try:
                subprocess.run(
                    [firefox_bin, "-CreateProfile", f"agent {profile}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
            except Exception:
                pass
        # Ensure we don't attach to an existing process and use the specified profile
        args += ["--no-remote", "--new-instance", "--profile", profile]
    args += ["--new-window", url]
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -----------------------
# LLM helpers (OpenAI-compatible v1: responses or chat.completions)
# -----------------------
def _derive_endpoints(url: str) -> Tuple[str, str]:
    """
    Accept either a base URL (…/v1) or a full endpoint URL.
    Return (responses_url, chat_url).
    """
    u = (url or "").rstrip("/")
    if u.endswith("/responses"):
        base = u[: -len("/responses")]
        return u, f"{base}/chat/completions"
    if u.endswith("/chat/completions"):
        base = u[: -len("/chat/completions")]
        return f"{base}/responses", u
    # assume base …/v1
    return f"{u}/responses", f"{u}/chat/completions"


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout=(5, 60)) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_text_from_openai_like(resp: Dict[str, Any]) -> str:
    """
    Normalize textual output for both responses API and chat.completions.
    """
    # chat.completions shape
    if "choices" in resp and resp.get("choices"):
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            pass
    # responses API variants
    # Some servers expose "output_text"; others have "output" with text segments.
    if "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"]
    if "output" in resp and isinstance(resp["output"], list):
        texts: List[str] = []
        for item in resp["output"]:
            # common shapes: {"type":"output_text", "text":"..."} or {"content":[{"type":"output_text","text":"..."}]}
            if isinstance(item, dict):
                if item.get("type") in ("message", "output_text") and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif "content" in item and isinstance(item["content"], list):
                    for c in item["content"]:
                        if isinstance(c, dict) and c.get("type") in ("message", "output_text") and isinstance(c.get("text"), str):
                            texts.append(c["text"])
        if texts:
            return "\n".join(t for t in texts if t)
    # fallback to any string field named "content"
    if isinstance(resp.get("content"), str):
        return resp["content"]
    return ""


# -----------------------
# LLM optional (text-only)
# -----------------------
def run_llm(system: str, prompt: str) -> str:
    if not LLM_API_URL:
        return "[LLM disabled] " + prompt[:300]

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    responses_url, chat_url = _derive_endpoints(LLM_API_URL)

    # Build both request bodies
    chat_body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    responses_body = {
        "model": LLM_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{system}\n\n{prompt}"}
                ],
            }
        ],
        "temperature": 0.2,
    }

    # Try preferred mode first, then fallback
    try_order: List[Tuple[str, Dict[str, Any]]] = []
    mode = LLM_API_MODE.lower()
    if mode == "responses":
        try_order = [(responses_url, responses_body), (chat_url, chat_body)]
    elif mode == "chat":
        try_order = [(chat_url, chat_body), (responses_url, responses_body)]
    else:
        try_order = [(responses_url, responses_body), (chat_url, chat_body)]

    last_err: Optional[Exception] = None
    for url, body in try_order:
        try:
            j = _post_json(url, headers, body)
            text = _extract_text_from_openai_like(j)
            if text:
                return text
            # If empty, still return something useful
            return json.dumps(j)[:2000]
        except Exception as e:
            last_err = e
            continue
    return f"[LLM error: {last_err}] {prompt[:300]}"


# -----------------------
# LLM vision (image + text → text)
# -----------------------
def run_llm_vision(image_bgr: np.ndarray, system: str, prompt: str) -> str:
    """
    Send a screenshot/image with text instructions to an OpenAI-compatible server.
    Supports both /v1/responses and /v1/chat/completions formats.
    """
    if not LLM_API_URL:
        return "[LLM disabled] " + prompt[:300]

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    # Encode image to base64 data URL
    jpg_bytes = encode_jpeg_bgr(image_bgr, q=90)
    b64 = base64.b64encode(jpg_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    responses_url, chat_url = _derive_endpoints(LLM_API_URL)

    # Chat Completions style (multi-part message content)
    chat_body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    # Responses API style (input_text + input_image)
    responses_body = {
        "model": LLM_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "temperature": 0.2,
    }

    try_order: List[Tuple[str, Dict[str, Any]]] = []
    mode = LLM_API_MODE.lower()
    if mode == "responses":
        try_order = [(responses_url, responses_body), (chat_url, chat_body)]
    elif mode == "chat":
        try_order = [(chat_url, chat_body), (responses_url, responses_body)]
    else:
        try_order = [(responses_url, responses_body), (chat_url, chat_body)]

    last_err: Optional[Exception] = None
    for url, body in try_order:
        try:
            j = _post_json(url, headers, body)
            text = _extract_text_from_openai_like(j)
            if text:
                return text
            return json.dumps(j)[:2000]
        except Exception as e:
            last_err = e
            continue
    return f"[LLM error: {last_err}] {prompt[:300]}"

def sub_env(s: str, ctx: Dict[str, Any]) -> str:
    # ${VAR} from env or ctx
    def repl(m):
        key = m.group(1)
        return str(os.environ.get(key, ctx.get(key, "")))
    return re.sub(r"\$\{([^}]+)\}", repl, s)

# -----------------------
# Task Runner
# -----------------------
class TaskRunner:
    def __init__(self, task_path: str):
        self.task_path = task_path
        self.session = requests.Session()
        self.ctx: Dict[str, Any] = {"series": {}}
        self._last_mtime = 0
        self.task = None

        # Screen / cache
        self.grabber = ScreenGrabber(screen_index=SCREEN_INDEX)
        self.last_img: Optional[np.ndarray] = None
        self.last_ocr: Optional[List[Dict[str, Any]]] = None
        self.last_ocr_words: Optional[List[Dict[str, Any]]] = None
        self.last_ocr_lines: Optional[List[Dict[str, Any]]] = None
        self.last_dets: Optional[List[Dict[str, Any]]] = None

        # Debug / run folder
        self.debug_enabled = DEBUG_ENABLED
        self.run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        self.run_dir = os.path.join(DEBUG_DIR, f"{AGENT_NAME}-{self.run_id}")
        if self.debug_enabled:
            ensure_dir(self.run_dir)
        self.log_path = os.path.join(self.run_dir, "steps.jsonl") if self.debug_enabled else None
        self.summary_path = os.path.join(self.run_dir, "summary.json") if self.debug_enabled else None

        self._xdotool_ok = _have_xdotool()
        self._log(
            level="info",
            msg="agent.start",
            extra={"agent": AGENT_NAME, "task_file": self.task_path, "api_url": DETECT_API_URL, "click_enabled": CLICK_ENABLED,
                   "screen_index": SCREEN_INDEX, "xdotool": self._xdotool_ok, "debug_dir": self.run_dir},
        )

    # ------------- Debug logging -------------
    def _log(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
        rec = {"ts": now_utc_iso(), "level": level, "msg": msg}
        if extra:
            rec.update(extra)
        line = safe_json_dumps(rec)
        print(line, flush=True)  # console
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _save_summary(self):
        if not self.summary_path:
            return
        summary = {
            "agent": AGENT_NAME,
            "run_id": self.run_id,
            "task": self.task,
            "ctx_keys": list(self.ctx.keys()),
            "debug_dir": self.run_dir,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ------------- Screen capture / cache -------------
    def _capture(self, force: bool = False) -> np.ndarray:
        # If we already have a frame for this step, reuse unless force
        if force or self.last_img is None:
            img, _mon = self.grabber.capture()
            self.last_img = img
            self.last_ocr = None
            self.last_ocr_words = None
            self.last_ocr_lines = None
            self.last_dets = None
            if self.debug_enabled:
                raw_path = os.path.join(self.run_dir, f"{int(time.time()*1000)}_raw.jpg")
                cv2.imwrite(raw_path, img)
                self._log("debug", "screenshot.captured", {"path": raw_path, "w": int(img.shape[1]), "h": int(img.shape[0])})
        return self.last_img

    # ------------- Annotate helpers -------------
    def _annotate_and_save(self, img: np.ndarray, save_name: str, with_ocr: bool = True, with_dets: bool = True):
        if not self.debug_enabled:
            return
        canvas = img.copy()
        if with_dets and self.last_dets:
            for d in self.last_dets:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange
                label = f'id={d.get("label","?")} s={d.get("score",0):.2f}'
                cv2.putText(canvas, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
        if with_ocr and self.last_ocr:
            for w in self.last_ocr:
                x1, y1, x2, y2 = map(int, w["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)  # green
                cv2.putText(canvas, w["text"], (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        out = os.path.join(self.run_dir, save_name)
        cv2.imwrite(out, canvas)
        self._log("debug", "overlay.saved", {"path": out})

    # ------------- OCR / DET wrappers -------------
    def _do_ocr(self) -> List[Dict[str, Any]]:
        if self.last_ocr is None:
            img = self._capture()
            ocr_levels = ocr_image_levels(img)
            self.last_ocr_words = list(ocr_levels.get("words") or [])
            self.last_ocr_lines = list(ocr_levels.get("lines") or [])
            self.last_ocr = self.last_ocr_words
            self._log("debug", "ocr.done", {"words": len(self.last_ocr_words), "lines": len(self.last_ocr_lines)})
        return self.last_ocr

    def _do_detect(self, save_overlay: bool = False) -> List[Dict[str, Any]]:
        if self.last_dets is None:
            img = self._capture()
            try:
                self.last_dets = rtdetr_detect(self.session, img)
            except Exception as e:
                self._log("error", "detect.error", {"error": str(e)})
                self.last_dets = []
            self.ctx["detections"] = self.last_dets
            self._log("debug", "detect.done", {"n": len(self.last_dets)})
        if save_overlay:
            self._annotate_and_save(self.last_img, f"{int(time.time()*1000)}_overlay.jpg", with_ocr=True, with_dets=True)
        return self.last_dets

    # ------------- Click helpers -------------
    def _click_abs(self, x_abs: int, y_abs: int):
        if not CLICK_ENABLED:
            self._log("info", "click.skipped", {"x": x_abs, "y": y_abs, "reason": "CLICK_ENABLED=0"})
            return
        if not self._xdotool_ok:
            self._log("warn", "click.skipped", {"x": x_abs, "y": y_abs, "reason": "xdotool missing"})
            return
        _safe_run(["xdotool", "mousemove", "--sync", str(x_abs), str(y_abs)])
        _safe_run(["xdotool", "click", "1"])
        self._log("info", "click.done", {"x": x_abs, "y": y_abs})

    def _click_box_image_space(self, box: List[float]):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x_abs, y_abs = self.grabber.to_abs(cx, cy)
        self._click_abs(x_abs, y_abs)

    # ------------- Legacy helpers (kept/backcompat) -------------
    def _set_var(self, name, value):
        self.ctx[name] = value

    def _track_series(self, key: str, value: Any, window: int = 3):
        s = self.ctx["series"].setdefault(key, [])
        try:
            v = float(value)
        except:
            return
        s.append(v)
        if len(s) > window:
            del s[0]

    def _open_student_profile(self, name: str):
        # Heuristic: search, click matching name via OCR
        self._click_text(regex=name)

    # ------------- OCR interactions -------------
    def _wait_text(self, regex=None, any_regex=None, fuzzy_text=None, fuzzy_threshold=None, timeout_s=30):
        t0 = time.time()
        while time.time() - t0 <= timeout_s:
            self._capture(force=True)
            ocr = self._do_ocr()
            w = find_text_box(
                ocr, line_items=self.last_ocr_lines, regex=regex, any_regex=any_regex,
                fuzzy_text=fuzzy_text, fuzzy_threshold=fuzzy_threshold
            )
            if w:
                self._log("info", "wait_text.found", {"text": w["text"], "box": w["box"]})
                return True
            time.sleep(0.5)
        self._log("warn", "wait_text.timeout", {"timeout_s": timeout_s, "regex": regex, "any_regex": any_regex, "fuzzy_text": fuzzy_text})
        return False

    def _click_text(self, regex=None, any_regex=None, nth=0, prefer_bold=False, fuzzy_text=None, fuzzy_threshold=None):
        self._capture(force=True)
        ocr = self._do_ocr()
        w = find_text_box(
            ocr,
            line_items=self.last_ocr_lines,
            regex=regex, any_regex=any_regex,
            fuzzy_text=fuzzy_text, fuzzy_threshold=fuzzy_threshold,
            prefer_bold=prefer_bold, nth=nth
        )
        if not w:
            self._log("warn", "click_text.not_found", {"regex": regex, "any_regex": any_regex, "fuzzy_text": fuzzy_text})
            return False
        self._click_box_image_space(w["box"])
        return True

    # ------------- Detection interactions -------------
    def _wait_detection(self, labels: Optional[List[int]], min_score: float = 0.5, timeout_s: float = 30.0) -> bool:
        t0 = time.time()
        labels_set = set(labels) if labels else None
        while time.time() - t0 <= timeout_s:
            self._capture(force=True)
            dets = self._do_detect(save_overlay=False)
            filtered = [d for d in dets if d.get("score", 0) >= min_score and (labels_set is None or int(d.get("label", -1)) in labels_set)]
            if filtered:
                self._log("info", "wait_detection.found", {"count": len(filtered), "labels": list(labels_set) if labels_set else "any"})
                return True
            time.sleep(0.4)
        self._log("warn", "wait_detection.timeout", {"timeout_s": timeout_s, "labels": labels})
        return False

    def _pick_detection(
        self,
        labels: Optional[List[int]] = None,
        min_score: float = 0.5,
        nth: int = 0,
        nearest_to_text: Optional[str] = None,
        fuzzy_threshold: float = 70.0
    ) -> Optional[Dict[str, Any]]:
        dets = self._do_detect(save_overlay=False)
        if labels:
            dets = [d for d in dets if int(d.get("label", -1)) in set(labels)]
        dets = [d for d in dets if d.get("score", 0) >= min_score]
        if not dets:
            return None

        if nearest_to_text:
            # Compute distances from each detection center to the best fuzzy-matched OCR word
            ocr = self._do_ocr()
            if ocr:
                # best fuzzy target
                scored = [(fuzz.partial_ratio(nearest_to_text, w["text"]), w) for w in ocr]
                scored.sort(key=lambda x: -x[0])
                if scored and scored[0][0] >= fuzzy_threshold:
                    target_w = scored[0][1]
                    tx = (target_w["box"][0] + target_w["box"][2]) / 2.0
                    ty = (target_w["box"][1] + target_w["box"][3]) / 2.0
                    dets.sort(key=lambda d: (( ( (d["box"][0]+d["box"][2])/2.0 - tx ) ** 2 + ( (d["box"][1]+d["box"][3])/2.0 - ty ) ** 2 )))
                else:
                    # fallback to score sort if no good fuzzy target found
                    dets.sort(key=lambda d: -float(d.get("score", 0)))
            else:
                dets.sort(key=lambda d: -float(d.get("score", 0)))
        else:
            dets.sort(key=lambda d: -float(d.get("score", 0)))

        if nth >= len(dets):
            return None
        return dets[nth]

    def _click_detection(self, labels: Optional[List[int]], min_score: float = 0.5, nth: int = 0,
                         nearest_to_text: Optional[str] = None):
        self._capture(force=True)
        det = self._pick_detection(labels=labels, min_score=min_score, nth=nth, nearest_to_text=nearest_to_text)
        if not det:
            self._log("warn", "click_detection.not_found", {"labels": labels, "min_score": min_score})
            return False
        self._click_box_image_space(det["box"])
        return True

    # ------------- Misc helpers -------------
    def _screenshot_save(self, overlay: bool = False, note: Optional[str] = None):
        img = self._capture(force=True)
        ts = int(time.time() * 1000)
        base = f"{ts}"
        if note:
            base += f"_{re.sub(r'[^a-zA-Z0-9_.-]+','-', note)[:40]}"
        raw_path = os.path.join(self.run_dir, base + "_raw.jpg")
        cv2.imwrite(raw_path, img)
        info = {"raw": raw_path}
        if overlay:
            self._do_ocr()
            self._do_detect()
            self._annotate_and_save(img, base + "_overlay.jpg", with_ocr=True, with_dets=True)
            info["overlay"] = os.path.join(self.run_dir, base + "_overlay.jpg")
        self._log("info", "screenshot.saved", info)

    # ------------- Task loading -------------
    def _load_if_changed(self):
        try:
            st = os.stat(self.task_path)
            if st.st_mtime != self._last_mtime or self.task is None:
                with open(self.task_path, "r", encoding="utf-8") as f:
                    self.task = yaml.safe_load(f) or {}
                self._last_mtime = st.st_mtime
                self._log("info", "task.loaded", {"path": self.task_path})
        except FileNotFoundError:
            pass

    # ------------- Steps engine -------------
    def _do_steps(self, steps: List[Dict[str, Any]]):
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            (op, args), = step.items()
            start_ts = time.time()
            ok = True
            err: Optional[str] = None

            # Normalize args
            _args = args
            if isinstance(_args, str):
                _args = {"value": _args}
            _args = _args or {}

            try:
                # --- Ops ---
                if op == "open_url":
                    url = _args.get("url") or _args.get("") or _args.get("value")
                    open_url(url)

                elif op == "wait_text":
                    self._wait_text(regex=_args.get("regex"), any_regex=_args.get("any_regex"),
                                    fuzzy_text=_args.get("fuzzy_text"), fuzzy_threshold=_args.get("fuzzy_threshold"),
                                    timeout_s=float(_args.get("timeout_s", 30)))

                elif op == "click_text":
                    self._click_text(regex=_args.get("regex"), any_regex=_args.get("any_regex"),
                                     nth=int(_args.get("nth", 0)), prefer_bold=bool(_args.get("prefer_bold", False)),
                                     fuzzy_text=_args.get("fuzzy_text"), fuzzy_threshold=_args.get("fuzzy_threshold"))

                elif op == "type_text":
                    # _args is always a dict (strings normalized to {"value": ...})
                    s = sub_env(_args.get("value") or _args.get("") or _args.get("text", ""), self.ctx)
                    type_text(s)

                elif op == "key_seq":
                    seq = _args if isinstance(_args, list) else _args.get("", []) or _args.get("keys", [])
                    send_keys(seq)

                elif op in ("sleep", "wait"):
                    time.sleep(float(_args.get("seconds", _args.get("", 1.0))))

                elif op == "detect":
                    save_overlay = bool(_args.get("save_overlay", False))
                    self._do_detect(save_overlay=save_overlay)

                elif op == "wait_detection":
                    labels = _args.get("labels")
                    if labels is None and "label" in _args:
                        labels = [_args.get("label")]
                    min_score = float(_args.get("min_score", 0.5))
                    timeout_s = float(_args.get("timeout_s", 30))
                    self._wait_detection(labels=labels, min_score=min_score, timeout_s=timeout_s)

                elif op == "click_detection":
                    labels = _args.get("labels")
                    if labels is None and "label" in _args:
                        labels = [_args.get("label")]
                    min_score = float(_args.get("min_score", 0.5))
                    nth = int(_args.get("nth", 0))
                    nearest_to_text = _args.get("nearest_to_text")
                    self._click_detection(labels=labels, min_score=min_score, nth=nth, nearest_to_text=nearest_to_text)

                elif op == "ocr_extract":
                    save_as = _args.get("save_as", "text_blob")
                    self._capture(force=True)
                    items = self._do_ocr()
                    self.ctx[save_as] = " ".join([w["text"] for w in items])

                elif op == "set_var":
                    self._set_var(_args["name"], _args["value"])

                elif op == "track_series":
                    key = sub_env(_args["key"], self.ctx)
                    val = sub_env(str(_args["value"]), self.ctx)
                    self._track_series(key, val, window=int(_args.get("window", 3)))

                elif op == "open_student_profile":
                    self._open_student_profile(sub_env(_args["name"], self.ctx))

                elif op == "load_json":
                    with open(sub_env(_args["path"], self.ctx), "r", encoding="utf-8") as f:
                        self.ctx[_args["save_as"]] = json.load(f)

                elif op == "run_llm":
                    sys = sub_env(_args.get("system", ""), self.ctx)
                    prompt = sub_env(_args.get("prompt", ""), self.ctx)
                    out = run_llm(sys, prompt)
                    self.ctx[_args.get("var_out", "llm_out")] = out

                elif op == "if":
                    cond = _args.get("condition", "")
                    try:
                        ok_cond = bool(eval(cond, {"__builtins__": {}}, {"ctx": self.ctx}))
                    except Exception as e:
                        self._log("error", "if.condition.error", {"error": str(e), "cond": cond})
                        ok_cond = False
                    self._do_steps(_args.get("then", []) if ok_cond else _args.get("else", []))

                elif op == "for_each":
                    lst = self.ctx.get(_args.get("list_var", ""), [])
                    as_var = _args.get("as", "item")
                    for item in lst:
                        self.ctx[as_var] = item
                        self._do_steps(_args.get("do", []))
                    self.ctx.pop(as_var, None)

                elif op == "for_pages":
                    nxt_rx = _args.get("next_button_regex", "Next|Continue")
                    until_rx = _args.get("until_regex", "Submit|Finish")
                    while True:
                        if self._wait_text(regex=until_rx, timeout_s=1):
                            break
                        self._do_steps(_args.get("do", []))
                        self._click_text(regex=nxt_rx, prefer_bold=True)
                        time.sleep(0.5)

                elif op == "for_questions":
                    self._do_steps(_args.get("do", []))

                elif op == "choose_option_from_key":
                    key = self.ctx.get(_args.get("key", "answer_key"), {})
                    self._capture(force=True)
                    words = self._do_ocr()
                    qids = [w for w in words if re.match(r"^Q\d+$", w["text"], re.I)]
                    for q in qids:
                        qid = q["text"].upper()
                        ans = key.get(qid)
                        if ans:
                            self._click_text(regex=f"^{re.escape(ans)}\\)", nth=0)

                elif op == "screenshot":
                    self._screenshot_save(overlay=bool(_args.get("overlay", False)), note=_args.get("note"))

                elif op == "log":
                    msg = _args.get("msg") or _args.get("") or _args.get("value") or ""
                    self._log("info", "user.log", {"message": sub_env(str(msg), self.ctx)})

                elif op == "debug_dump_ctx":
                    if self.debug_enabled:
                        p = os.path.join(self.run_dir, "ctx.json")
                        with open(p, "w", encoding="utf-8") as f:
                            json.dump(self.ctx, f, indent=2, ensure_ascii=False, default=str)
                        self._log("info", "ctx.saved", {"path": p})

                else:
                    self._log("warn", "op.unknown", {"op": op})
                    ok = False

            except Exception as e:
                ok = False
                err = f"{e.__class__.__name__}: {e}"
                tb = traceback.format_exc(limit=4)
                self._log("error", "step.error", {"op": op, "args": _args, "error": err, "trace": tb})

            # Per-step trace record
            self._log("info", "step.done", {
                "op": op,
                "ok": ok,
                "ms": int((time.time() - start_ts) * 1000),
            })

    def run(self):
        self._log("info", "agent.loop.start", {})
        while True:
            self._load_if_changed()
            if not self.task:
                time.sleep(1.0)
                continue
            loop = bool(self.task.get("loop", True))
            steps = self.task.get("steps", [])
            self._do_steps(steps)
            self._save_summary()
            if not loop:
                break
            time.sleep(0.25)

#------------------------
# ActionExecutorDynamic
#------------------------
class ActionExecutorDynamic:
    """Dynamic observe→decide→act executor."""
    def __init__(self):
        self.session = requests.Session()
        self.ctx: Dict[str, Any] = {}           # user-visible context (ocr_extract etc.)
        self.history: List[Dict[str, Any]] = [] # rolling action history
        self.grabber = ScreenGrabber(screen_index=SCREEN_INDEX)
        self.last_img: Optional[np.ndarray] = None
        self.last_ocr: List[Dict[str, Any]] = []
        self.last_ocr_words: List[Dict[str, Any]] = []
        self.last_ocr_lines: List[Dict[str, Any]] = []
        self.last_ocr_source: str = ""
        self.last_ax_nodes: List[Dict[str, Any]] = []
        self._xdotool_ok = which("xdotool") is not None

        # Debug folder (persist crops/overlays if host mounts AGENT_DEBUG_DIR)
        self.debug_enabled = DEBUG_ENABLED
        self.run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        self.run_dir = os.path.join(DEBUG_DIR, f"{AGENT_NAME}-{self.run_id}")
        if self.debug_enabled:
            ensure_dir(self.run_dir)

        # Trace (numbered actions + HTML grid)
        self.trace_enabled = bool(self.debug_enabled and TRACE_ENABLED)
        self.trace_max = max(10, min(500, int(TRACE_MAX)))
        self.trace_idx = 0
        self.trace_path = os.path.join(self.run_dir, "trace.jsonl")
        self.trace_html_path = os.path.join(self.run_dir, "trace.html")
        self._trace_tail: List[Dict[str, Any]] = []
        self._last_click_debug: Optional[Dict[str, Any]] = None

    # --- Logging, capture, OCR ---
    def _log(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
        rec = {"ts": now_utc_iso(), "level": level, "msg": msg}
        if extra: rec.update(extra)
        line = safe_json_dumps(rec)
        print(line, flush=True)

    def _debug_save_click_crop(
        self,
        *,
        box: List[int],
        click_rel: Tuple[int, int],
        action: str,
        note: Optional[str] = None,
        pad: int = 60,
    ) -> Optional[Dict[str, Any]]:
        """
        Save a cropped image around the intended click region, plus an overlay that
        shows the box and the click point. This is the fastest way to diagnose
        small systematic x/y offsets.
        """
        if not (self.debug_enabled and CLICK_CROP_DEBUG):
            return None
        if self.last_img is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in (box or [0, 0, 0, 0])]
            cx, cy = int(click_rel[0]), int(click_rel[1])
            h, w = self.last_img.shape[:2]
            # Clamp crop bounds
            left = max(0, min(w - 1, min(x1, x2) - pad))
            top = max(0, min(h - 1, min(y1, y2) - pad))
            right = max(1, min(w, max(x1, x2) + pad))
            bottom = max(1, min(h, max(y1, y2) + pad))
            if right <= left or bottom <= top:
                return None

            crop = self.last_img[top:bottom, left:right].copy()
            overlay = crop.copy()
            # Draw the original box and click point in crop coordinates
            bx1, by1, bx2, by2 = x1 - left, y1 - top, x2 - left, y2 - top
            px, py = cx - left, cy - top
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 200, 0), 2)  # green box
            cv2.circle(overlay, (px, py), 6, (0, 0, 255), -1)              # red dot
            cv2.drawMarker(overlay, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

            ts_ms = int(time.time() * 1000)
            base = f"{ts_ms}_{action}"
            raw_name = os.path.join(self.run_dir, f"{base}_crop.jpg")
            ov_name = os.path.join(self.run_dir, f"{base}_crop_overlay.jpg")
            cv2.imwrite(raw_name, crop)
            cv2.imwrite(ov_name, overlay)

            meta = {
                "raw_path": raw_name,
                "overlay_path": ov_name,
                "action": action,
                "note": note or "",
                "box": [x1, y1, x2, y2],
                "click_rel": [cx, cy],
                "crop_bounds": [left, top, right, bottom],
            }
            self._last_click_debug = meta
            self._log("info", "click.debug.crop_saved", meta)
            return meta
        except Exception as e:
            self._log("warn", "click.debug.crop_failed", {"error": str(e), "action": action})
            return None

    def _trace_append(self, rec: Dict[str, Any]) -> None:
        if not self.trace_enabled:
            return
        try:
            # persist
            line = safe_json_dumps(rec)
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            # tail
            self._trace_tail.append(rec)
            if len(self._trace_tail) > self.trace_max:
                self._trace_tail = self._trace_tail[-self.trace_max :]
        except Exception as e:
            self._log("warn", "trace.append_failed", {"error": str(e)})

    def _trace_render_html(self) -> None:
        if not self.trace_enabled:
            return
        try:
            items = list(self._trace_tail)[-self.trace_max :]
            cards = []
            for it in reversed(items):
                idx = it.get("idx")
                ts = it.get("ts")
                planner_action = (it.get("decision") or {}).get("action", "")
                planner_params = json.dumps((it.get("decision") or {}).get("parameters", {}), ensure_ascii=False)[:400]
                why = (it.get("decision") or {}).get("reasoning", "")[:260]
                result = it.get("result") or {}
                executed_action = result.get("executed_action") or planner_action
                executed_params = json.dumps(result.get("executed_parameters") or (it.get("decision") or {}).get("parameters", {}), ensure_ascii=False)[:400]
                status = result.get("status", "")
                verification = result.get("verification") or {}
                before_state = result.get("before_state") or {}
                after_state = result.get("after_state") or {}
                focus_before = before_state.get("focused_role") or before_state.get("focused_name") or ""
                focus_after = after_state.get("focused_role") or after_state.get("focused_name") or ""
                verification_reason = verification.get("reason", "")
                verification_evidence = ", ".join((verification.get("evidence") or [])[:4])
                event_applied = result.get("event_applied")
                outcome_verified = result.get("outcome_verified")
                targeting_source = result.get("targeting_source") or result.get("ocr_level") or result.get("anchor_level") or ""
                executor_override_reason = result.get("executor_override_reason") or ""
                blocker_class = result.get("blocker_class") or ""
                recovery_strategy = result.get("recovery_strategy") or ""
                recovery_effect = result.get("recovery_effect") or ""
                frame = it.get("frame_path") or ""
                post_frame = it.get("post_frame_path") or ""
                crop = it.get("crop_overlay_path") or ""
                imgs = []
                if crop:
                    imgs.append(f'<figure><img src="{os.path.basename(crop)}" alt="crop overlay"/><figcaption>target</figcaption></figure>')
                if frame:
                    imgs.append(f'<figure><img src="{os.path.basename(frame)}" alt="frame"/><figcaption>before</figcaption></figure>')
                if post_frame:
                    imgs.append(f'<figure><img src="{os.path.basename(post_frame)}" alt="post frame"/><figcaption>after</figcaption></figure>')
                img_html = "".join(imgs) if imgs else '<div class="noimg">no image</div>'
                card_cls = "card"
                if status == "success" and outcome_verified:
                    card_cls += " ok"
                elif status == "dispatched" or (event_applied and not outcome_verified):
                    card_cls += " warn"
                else:
                    card_cls += " bad"
                cards.append(
                    f"""
<div class="{card_cls}">
  <div class="hdr">
    <div class="idx">#{idx}</div>
    <div class="meta">{ts} · <b>{executed_action}</b> · {status}</div>
  </div>
  <div class="imgs">{img_html}</div>
  <div class="txt"><b>planner</b> {planner_action} {planner_params}</div>
  <div class="txt"><b>executed</b> {executed_action} {executed_params}</div>
  <div class="txt"><b>why</b> {why}</div>
  <div class="txt"><b>verification</b> applied={event_applied} verified={outcome_verified} reason={verification_reason}</div>
  <div class="txt"><b>blocker</b> {blocker_class}</div>
  <div class="txt"><b>recovery</b> strategy={recovery_strategy} effect={recovery_effect}</div>
  <div class="txt"><b>targeting</b> {targeting_source}</div>
  <div class="txt"><b>override</b> {executor_override_reason}</div>
  <div class="txt"><b>evidence</b> {verification_evidence}</div>
  <div class="txt"><b>before tags</b> {before_state.get("tags", [])}</div>
  <div class="txt"><b>after tags</b> {after_state.get("tags", [])}</div>
  <div class="txt"><b>focus</b> {focus_before} -> {focus_after}</div>
</div>
"""
                )
            html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Agent trace</title>
<style>
body {{ font-family: system-ui, Arial, sans-serif; margin: 16px; }}
.grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:10px; background:#fff; }}
.card.ok {{ border-color:#9ad0a0; background:#f5fff6; }}
.card.warn {{ border-color:#e9c46a; background:#fffaf0; }}
.card.bad {{ border-color:#ef9a9a; background:#fff5f5; }}
.hdr {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
.idx {{ font-weight:700; }}
.meta {{ font-size:12px; color:#333; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.imgs {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }}
figure {{ margin:0; }}
figcaption {{ font-size:11px; color:#666; margin-top:4px; text-align:center; }}
img {{ width:100%; height:auto; border-radius:8px; background:#f6f6f6; }}
.txt {{ font-size:12px; color:#222; margin-top:8px; word-break:break-word; }}
.noimg {{ font-size:12px; color:#777; padding:18px; text-align:center; border:1px dashed #ccc; border-radius:8px; }}
</style></head>
<body>
<h2>Last {len(items)} actions (auto-updated)</h2>
<div class="grid">
{''.join(cards)}
</div>
</body></html>"""
            with open(self.trace_html_path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            self._log("warn", "trace.render_failed", {"error": str(e)})

    def _save_trace_frame(self, op: str, suffix: str) -> str:
        if not (self.trace_enabled and self.last_img is not None):
            return ""
        try:
            q = max(25, min(95, int(TRACE_FRAME_QUALITY)))
            path = os.path.join(self.run_dir, f"{self.trace_idx:05d}_{op}_{suffix}.jpg")
            cv2.imwrite(path, self.last_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return path
        except Exception as e:
            self._log("warn", "trace.frame_save_failed", {"error": str(e), "suffix": suffix, "action": op})
            return ""

    def _normalize_visible_text(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _current_text_items(self) -> List[Dict[str, Any]]:
        return self.last_ocr_lines or self.last_ocr_words or self.last_ocr or []

    def _active_blockers(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        blockers = snapshot.get("blockers")
        return blockers if isinstance(blockers, list) else []

    def _primary_blocker(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        blockers = self._active_blockers(snapshot)
        return blockers[0] if blockers else None

    def _blocker_tags(self, snapshot: Dict[str, Any]) -> List[str]:
        tags = [str(tag) for tag in (snapshot.get("tags") or []) if str(tag).startswith("blocker:")]
        for blocker in self._active_blockers(snapshot):
            cls = str(blocker.get("class", "") or "").strip()
            if cls:
                tag = f"blocker:{cls}"
                if tag not in tags:
                    tags.append(tag)
        return tags

    def _blocker_signature(self, snapshot: Dict[str, Any]) -> str:
        return str(snapshot.get("blocker_signature", "") or "")

    def _box_iou(self, a: Optional[List[int]], b: Optional[List[int]]) -> float:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return float(inter / max(1, area_a + area_b - inter))

    def _chrome_bottom_estimate(self, snapshot: Dict[str, Any], blocker: Optional[Dict[str, Any]] = None) -> int:
        h = int(self.last_img.shape[0]) if self.last_img is not None else 900
        estimate = min(max(96, int(h * 0.16)), 180)
        if blocker and blocker.get("scope") == "browser_chrome":
            bbox = blocker.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                estimate = max(estimate, int(bbox[3]) + 16)
        return estimate

    def _target_scope_from_box(
        self,
        box: Optional[List[int]],
        snapshot: Dict[str, Any],
        blocker: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not box or len(box) != 4:
            return "unknown"
        primary = blocker or self._primary_blocker(snapshot)
        if primary and self._box_iou(box, primary.get("bbox")) >= 0.25:
            return str(primary.get("scope") or "blocker_surface")
        chrome_bottom = self._chrome_bottom_estimate(snapshot, primary)
        if int(box[3]) <= chrome_bottom:
            return "browser_chrome"
        return "page_content"

    def _resolve_action_target(
        self,
        action: str,
        params: Dict[str, Any],
        snapshot: Dict[str, Any],
        blocker: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        target: Optional[Dict[str, Any]] = None
        if action == "click_text":
            target = find_text_box(
                self.last_ocr_words or self.last_ocr,
                line_items=self.last_ocr_lines,
                regex=params.get("regex"),
                nth=int(params.get("nth", 0)),
                prefer_bold=bool(params.get("prefer_bold", False)),
                fuzzy_text=params.get("fuzzy_text"),
                fuzzy_threshold=params.get("fuzzy_threshold"),
            )
        elif action == "click_any_text":
            patterns = params.get("patterns") or []
            for pat in patterns:
                target = find_text_box(
                    self.last_ocr_words or self.last_ocr,
                    line_items=self.last_ocr_lines,
                    regex=pat,
                    nth=int(params.get("nth", 0)),
                    prefer_bold=bool(params.get("prefer_bold", True)),
                )
                if target:
                    break
        elif action == "click_near_text":
            target = find_text_box(
                self.last_ocr_words or self.last_ocr,
                line_items=self.last_ocr_lines,
                regex=params.get("anchor_regex"),
            )
        elif action == "click_box" and isinstance(params.get("box"), list):
            target = {"box": [int(v) for v in params.get("box", [0, 0, 0, 0])], "text": "", "level": "box"}

        if not target:
            return None

        target_box = target.get("box") if isinstance(target.get("box"), list) else None
        info = {
            "text": str(target.get("text", "") or ""),
            "box": [int(v) for v in target_box] if target_box else None,
            "level": str(target.get("level", "") or ""),
            "scope": self._target_scope_from_box(target_box, snapshot, blocker),
        }
        return info

    def _page_click_away_box(self, snapshot: Dict[str, Any], blocker: Optional[Dict[str, Any]]) -> Optional[List[int]]:
        if self.last_img is None:
            return None
        h, w = self.last_img.shape[:2]
        cx = w // 2
        cy = max(int(h * 0.55), self._chrome_bottom_estimate(snapshot, blocker) + 60)
        if blocker and isinstance(blocker.get("bbox"), list) and len(blocker.get("bbox")) == 4:
            bx1, by1, bx2, by2 = [int(v) for v in blocker.get("bbox", [0, 0, 0, 0])]
            if by2 + 80 < h:
                cy = max(cy, by2 + 80)
            if bx1 <= cx <= bx2:
                cx = min(w - 40, max(40, bx2 + 80))
        return [max(0, cx - 12), max(0, cy - 12), min(w, cx + 12), min(h, cy + 12)]

    def _match_resolve_target(self, blocker: Dict[str, Any], target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not target:
            return None
        target_text = self._normalize_visible_text(str(target.get("text", "") or ""))
        target_box = target.get("box")
        for candidate in blocker.get("resolve_targets") or []:
            cand_text = self._normalize_visible_text(str(candidate.get("text", "") or candidate.get("label", "") or ""))
            cand_box = candidate.get("box")
            if target_text and cand_text and target_text == cand_text:
                return candidate
            if self._box_iou(target_box, cand_box) >= 0.45:
                return candidate
        return None

    def _action_strategy(self, action: str, params: Dict[str, Any], target: Optional[Dict[str, Any]], blocker: Optional[Dict[str, Any]]) -> str:
        if action in ("click_text", "click_any_text", "click_near_text", "click_box"):
            match = self._match_resolve_target(blocker or {}, target)
            if match is not None:
                return "click_visible_resolve_target"
            scope = str((target or {}).get("scope") or "")
            if scope == "page_content":
                return "click_visible_page_target"
            if scope == "browser_chrome":
                return "click_browser_chrome"
            if scope:
                return "click_blocker_surface"
            return "click"
        if action == "type_text":
            return "type"
        if action == "wait_text":
            return "wait_text"
        if action == "wait_any_text":
            return "wait_any_text"
        if action == "open_url":
            return "open_url"
        if action == "sleep":
            return "sleep"
        if action == "ocr_extract":
            return "ocr_extract"
        if action == "key_seq":
            keys = self._normalize_key_sequence((params or {}).get("keys", []))
            keys_blob = ",".join(str(k).lower() for k in keys)
            if "escape" in keys_blob:
                return "escape"
            if "ctrl+l" in keys_blob:
                return "refocus_urlbar"
            if "tab" in keys_blob:
                return "tab"
            if "return" in keys_blob or "enter" in keys_blob:
                return "submit_key"
            return "key_seq"
        return action

    def _action_family(self, action: str, params: Optional[Dict[str, Any]] = None) -> str:
        if action in ("click_text", "click_any_text", "click_near_text", "click_box"):
            return "click"
        if action == "type_text":
            return "type"
        if action == "wait_text":
            return "wait_text"
        if action == "wait_any_text":
            return "wait_any_text"
        if action == "open_url":
            return "open_url"
        if action == "sleep":
            return "sleep"
        if action == "ocr_extract":
            return "ocr_extract"
        if action == "run_llm":
            return "run_llm"
        if action == "key_seq":
            keys = self._normalize_key_sequence((params or {}).get("keys", []))
            keys_blob = ",".join(str(k).lower() for k in keys)
            if "escape" in keys_blob:
                return "escape"
            if "ctrl+l" in keys_blob:
                return "refocus_urlbar"
            if "tab" in keys_blob:
                return "tab"
            if "return" in keys_blob or "enter" in keys_blob:
                return "submit_key"
            return "key_seq"
        return action

    def _is_blocker_sensitive_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        family = self._action_family(action, params)
        return family in {"click", "type", "submit_key", "open_url"}

    def _recent_blocker_attempts(self, blocker_signature: str) -> List[Dict[str, Any]]:
        if not blocker_signature:
            return []
        attempts: List[Dict[str, Any]] = []
        for item in self.history[-16:]:
            result = item.get("result") or {}
            before_state = result.get("before_state") or {}
            after_state = result.get("after_state") or {}
            if blocker_signature not in {
                str(before_state.get("blocker_signature", "") or ""),
                str(after_state.get("blocker_signature", "") or ""),
            }:
                continue
            attempts.append(
                {
                    "action": item.get("action"),
                    "parameters": item.get("parameters") or {},
                    "status": result.get("status"),
                    "strategy": result.get("recovery_strategy") or self._action_family(item.get("action", ""), item.get("parameters") or {}),
                    "effect": result.get("recovery_effect") or result.get("verification", {}).get("reason", ""),
                    "verified": result.get("outcome_verified"),
                }
            )
        return attempts

    def _attempt_counts(self, attempts: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for attempt in attempts:
            strategy = str(attempt.get("strategy", "") or "")
            if not strategy:
                continue
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts

    def _recovery_options(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocker = self._primary_blocker(snapshot)
        if not blocker:
            return []
        counts = self._attempt_counts(self._recent_blocker_attempts(str(blocker.get("signature", "") or "")))
        options: List[Dict[str, Any]] = []
        for target in blocker.get("resolve_targets") or []:
            options.append(
                {
                    "strategy": "click_visible_resolve_target",
                    "label": str(target.get("label", "") or target.get("text", "") or ""),
                    "box": target.get("box"),
                    "dual_purpose": bool(target.get("dual_purpose", False)),
                    "attempts": counts.get("click_visible_resolve_target", 0),
                }
            )
        if blocker.get("class") == "browser_url_suggestion_dropdown":
            options.extend(
                [
                    {"strategy": "escape", "label": "Press Escape", "attempts": counts.get("escape", 0)},
                    {"strategy": "click_away_page", "label": "Click page content or blank area", "attempts": counts.get("click_away_page", 0)},
                    {"strategy": "refocus_urlbar", "label": "Refocus URL bar", "attempts": counts.get("refocus_urlbar", 0)},
                    {"strategy": "open_url", "label": "Open target URL in new window", "attempts": counts.get("open_url", 0)},
                ]
            )
        else:
            options.extend(
                [
                    {"strategy": s, "label": s.replace("_", " "), "attempts": counts.get(s, 0)}
                    for s in blocker.get("suggested_strategies") or []
                    if s != "click_visible_resolve_target"
                ]
            )
        return options[:8]

    def _recovery_effect(self, before_snapshot: Dict[str, Any], after_snapshot: Dict[str, Any]) -> str:
        before_sig = self._blocker_signature(before_snapshot)
        after_sig = self._blocker_signature(after_snapshot)
        if before_sig and not after_sig:
            return "blocker_cleared"
        if before_sig and after_sig and before_sig != after_sig:
            return "blocker_changed"
        if before_sig and after_sig and before_sig == after_sig:
            return "blocker_unchanged"
        return "no_blocker"

    def _default_blocker_recovery_hint(self, before_snapshot: Dict[str, Any], after_snapshot: Optional[Dict[str, Any]] = None) -> str:
        blocker = self._primary_blocker(after_snapshot or before_snapshot) or self._primary_blocker(before_snapshot)
        blocker_class = str((blocker or {}).get("class", "") or "")
        if blocker_class == "browser_url_suggestion_dropdown":
            return "use_page_click_or_open_url_not_repeated_escape"
        if blocker_class == "browser_session_restore":
            return "click_visible_session_restore_target_or_open_url"
        if blocker_class:
            return "clear_blocker_before_progressing"
        return "verify_state_before_continuing"

    def _blocker_policy_directive(
        self,
        action: str,
        params: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        blocker = self._primary_blocker(snapshot)
        if blocker is None:
            return {"decision": "allow", "action": action, "params": params, "strategy": self._action_family(action, params)}

        target = self._resolve_action_target(action, params, snapshot, blocker)
        strategy = self._action_strategy(action, params, target, blocker)
        attempts = self._recent_blocker_attempts(str(blocker.get("signature", "") or ""))
        counts = self._attempt_counts(attempts)
        blocker_class = str(blocker.get("class", "") or "")
        focus = snapshot.get("ax") or {}

        def allow(reason: str) -> Dict[str, Any]:
            return {
                "decision": "allow",
                "action": action,
                "params": params,
                "reason": reason,
                "strategy": strategy,
                "target": target,
                "blocker": blocker,
            }

        def override(new_action: str, new_params: Dict[str, Any], reason: str, override_strategy: str) -> Dict[str, Any]:
            return {
                "decision": "override",
                "action": new_action,
                "params": new_params,
                "reason": reason,
                "strategy": override_strategy,
                "target": target,
                "blocker": blocker,
            }

        def block(reason: str) -> Dict[str, Any]:
            return {
                "decision": "block",
                "action": action,
                "params": params,
                "reason": reason,
                "strategy": strategy,
                "target": target,
                "blocker": blocker,
            }

        if blocker_class == "browser_url_suggestion_dropdown":
            if strategy == "click_visible_page_target":
                return allow("page_target_click_can_clear_chrome_dropdown")
            if strategy == "open_url":
                return allow("open_url_bypasses_browser_dropdown")
            if strategy == "escape" and counts.get("escape", 0) < 1:
                return allow("first_escape_attempt_allowed_for_browser_dropdown")
            if strategy == "click_browser_chrome" and counts.get("click_browser_chrome", 0) < 1:
                return allow("single_browser_chrome_refocus_attempt_allowed")
            if strategy == "refocus_urlbar" and counts.get("refocus_urlbar", 0) < 1:
                return allow("single_urlbar_refocus_attempt_allowed")
            if strategy == "type" and focus.get("focused_editable"):
                return allow("typing_allowed_once_focus_is_editable")
            if strategy == "submit_key" and focus.get("focused_editable") and counts.get("submit_key", 0) < 1:
                return allow("submit_allowed_when_editable_focus_confirms_intent")
            if counts.get("escape", 0) < 1:
                return override("key_seq", {"keys": ["Escape"]}, "browser_dropdown_first_try_escape", "escape")
            click_away_box = self._page_click_away_box(snapshot, blocker)
            if click_away_box and counts.get("click_away_page", 0) < 1:
                return override("click_box", {"box": click_away_box}, "browser_dropdown_try_click_away", "click_away_page")
            if counts.get("refocus_urlbar", 0) < 1:
                return override("key_seq", {"keys": ["Ctrl+l"]}, "browser_dropdown_refocus_urlbar", "refocus_urlbar")
            return block("browser_dropdown_requires_page_target_or_open_url_not_more_escape")

        if strategy == "click_visible_resolve_target":
            return allow("visible_blocker_resolve_target_selected")
        if strategy == "open_url" and blocker_class in ("browser_session_restore", "browser_interstitial_error"):
            return allow("open_url_can_bypass_browser_page_interstitial")
        if blocker_class in ("browser_permission_prompt", "modal_dialog", "cookie_banner"):
            if strategy == "escape" and counts.get("escape", 0) < 1:
                return allow("single_escape_attempt_allowed_for_dialog_like_blocker")
            if strategy == "click_away_page" and blocker_class == "cookie_banner" and counts.get("click_away_page", 0) < 1:
                return allow("single_click_away_allowed_for_cookie_banner")

        resolve_targets = blocker.get("resolve_targets") or []
        if resolve_targets and counts.get("click_visible_resolve_target", 0) < 2:
            primary = resolve_targets[0]
            return override(
                "click_box",
                {"box": [int(v) for v in primary.get("box", [0, 0, 0, 0])]},
                f"use_visible_{blocker_class}_resolve_target",
                "click_visible_resolve_target",
            )
        if blocker_class in ("browser_permission_prompt", "modal_dialog", "cookie_banner") and counts.get("escape", 0) < 1:
            return override("key_seq", {"keys": ["Escape"]}, f"fallback_escape_for_{blocker_class}", "escape")
        return block(f"{blocker_class}_blocks_requested_action_until_resolved")

    def _executor_guard_result(
        self,
        action: str,
        params: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        ax = snapshot.get("ax") or {}
        if action == "type_text" and ax.get("available") and not ax.get("focused_editable"):
            return {
                "status": "failure",
                "error_code": "FOCUS_NOT_EDITABLE",
                "error_message": f"Focused AX node is not editable ({ax.get('focused_role') or 'unknown'})",
                "event_applied": False,
                "recovery_hint": "refocus_editable_field",
            }

        blocker_policy = self._blocker_policy_directive(action, params, snapshot)
        if blocker_policy.get("decision") == "block":
            blocker = blocker_policy.get("blocker") or {}
            return {
                "status": "failure",
                "error_code": "BLOCKER_POLICY_BLOCKED",
                "error_message": str(blocker_policy.get("reason", "Blocked by blocker policy")),
                "event_applied": False,
                "recovery_hint": self._default_blocker_recovery_hint(snapshot),
                "blocker_class": blocker.get("class"),
                "blocker_signature": blocker.get("signature"),
                "recovery_strategy": blocker_policy.get("strategy"),
                "recovery_options": self._recovery_options(snapshot),
            }

        if not self.history:
            return None

        last = self.history[-1]
        last_result = last.get("result") or {}
        last_unresolved = last_result.get("status") == "dispatched" or last_result.get("outcome_verified") is False
        if not last_unresolved:
            return None

        current_hash = snapshot.get("hash", "")
        last_after_hash = ((last_result.get("after_state") or {}).get("hash") or "")
        last_before_hash = ((last_result.get("before_state") or {}).get("hash") or "")
        same_state = bool(current_hash) and current_hash in {last_after_hash, last_before_hash}
        same_family = self._action_family(last.get("action", ""), last.get("parameters") or {}) == self._action_family(action, params)
        last_streak = int(last_result.get("same_action_same_state_streak") or 1)
        if same_state and same_family and last_streak > 1:
            return {
                "status": "failure",
                "error_code": "ANTI_REPEAT_GUARD",
                "error_message": "Repeated same-family action on the same unresolved state was blocked",
                "event_applied": False,
                "recovery_hint": "change_action_family",
                "guard_reason": "same_family_repeated_after_unverified_state",
            }
        return None

    def _fetch_ax_nodes(self) -> List[Dict[str, Any]]:
        if not A11Y_BRIDGE_URL:
            return []
        try:
            r = requests.get(f"{A11Y_BRIDGE_URL}/ax/snapshot", timeout=A11Y_FETCH_TIMEOUT_S)
            r.raise_for_status()
            nodes = r.json()
            return nodes if isinstance(nodes, list) else []
        except Exception:
            return []

    def _ax_state_tokens(self, node: Dict[str, Any]) -> set:
        tokens = set()
        for key in ("state", "states", "flags"):
            raw = node.get(key)
            if isinstance(raw, str):
                tokens |= {tok.strip().lower() for tok in re.split(r"[,| ]+", raw) if tok.strip()}
            elif isinstance(raw, list):
                tokens |= {str(tok).strip().lower() for tok in raw if str(tok).strip()}
            elif isinstance(raw, dict):
                for name, enabled in raw.items():
                    if enabled:
                        tokens.add(str(name).strip().lower())
        return tokens

    def _ax_flag(self, node: Dict[str, Any], *names: str) -> bool:
        names_norm = {str(name).strip().lower() for name in names}
        for key, value in node.items():
            key_norm = str(key).strip().lower()
            if key_norm in names_norm:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "focused", "editable", "selected"):
                    return True
        tokens = self._ax_state_tokens(node)
        return any(name in tokens for name in names_norm)

    def _ax_text(self, node: Dict[str, Any]) -> str:
        return str(node.get("name") or node.get("label") or node.get("description") or "").strip()

    def _ax_value_text(self, node: Dict[str, Any]) -> str:
        return str(node.get("value") or node.get("text") or "").strip()

    def _ax_focus_summary(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "available": bool(nodes),
            "focus_hash": "",
            "focused_role": "",
            "focused_name": "",
            "focused_value": "",
            "focused_value_norm": "",
            "focused_editable": False,
            "focused_box": None,
        }
        if not nodes:
            return summary

        focused = None
        for node in nodes:
            if self._ax_flag(node, "focused", "active"):
                focused = node
                break
        if focused is None:
            return summary

        role = str(focused.get("role") or "").strip().lower()
        name = self._ax_text(focused)
        value = self._ax_value_text(focused)
        editable = self._ax_flag(focused, "editable") or role in ("textbox", "searchbox", "textarea", "entry", "combobox", "input")
        box = focused.get("box")
        focus_key = json.dumps({
            "role": role,
            "name": name,
            "value": value,
            "box": box,
            "editable": editable,
        }, ensure_ascii=False, sort_keys=True)
        summary.update({
            "focus_hash": hashlib.sha1(focus_key.encode("utf-8")).hexdigest()[:12],
            "focused_role": role,
            "focused_name": name,
            "focused_value": value,
            "focused_value_norm": self._normalize_visible_text(value),
            "focused_editable": bool(editable),
            "focused_box": box,
        })
        return summary

    def _state_snapshot(self) -> Dict[str, Any]:
        texts: List[str] = []
        top_texts: List[str] = []
        seen = set()
        text_items = self._current_text_items()
        for w in text_items:
            t = str(w.get("text", "") or "").strip()
            if not t:
                continue
            texts.append(t)
            key = self._normalize_visible_text(t)
            if key and key not in seen and len(top_texts) < 10:
                seen.add(key)
                top_texts.append(t)

        text_blob = " ".join(texts)
        text_blob_norm = self._normalize_visible_text(text_blob)
        tags: List[str] = []
        blockers = classify_blockers(text_items)
        blocker_signature = str(blockers[0].get("signature", "") or "") if blockers else ""

        def add_tag(tag: str) -> None:
            if tag not in tags:
                tags.append(tag)

        if any(tok in text_blob_norm for tok in ("firefox", "new tab", "search with google", "switch to tab", "gmail", "ebay", "amazon", "session restore")):
            add_tag("app:browser_like")
        for blocker in blockers:
            blocker_class = str(blocker.get("class", "") or "").strip()
            if blocker_class:
                add_tag(f"blocker:{blocker_class}")
            for legacy_tag in blocker.get("legacy_tags") or []:
                add_tag(str(legacy_tag))
        if any(tok in text_blob_norm for tok in ("loading", "please wait", "just a moment", "opening", "working")):
            add_tag("phase:loading_like")
        if any(tok in text_blob_norm for tok in ("search", "results", "sort", "filter", "price", "buy now", "add to cart", "cart")):
            add_tag("surface:results_like")
        if any(tok in text_blob_norm for tok in ("compose", "new message", "send", "subject", "recipients", "inbox")):
            add_tag("surface:mail_like")
        if any(tok in text_blob_norm for tok in ("sign in", "log in", "password", "username", "email")):
            add_tag("surface:auth_like")

        ax_summary = self._ax_focus_summary(self.last_ax_nodes or [])
        if ax_summary.get("available"):
            add_tag("sensor:ax")
        if ax_summary.get("focused_role"):
            add_tag(f"focus:{ax_summary.get('focused_role')}")
        if ax_summary.get("focused_editable"):
            add_tag("focus:editable")
        if not tags:
            add_tag("state:unclassified")

        sig_vec = None
        state_hash = ""
        if self.last_img is not None:
            try:
                sig_vec = make_signature(self.last_img, self.last_ocr_words or self.last_ocr or [])
                state_hash = hashlib.sha1(sig_vec.tobytes()).hexdigest()[:12]
            except Exception as e:
                self._log("warn", "state.signature_failed", {"error": str(e)})

        return {
            "hash": state_hash,
            "tags": tags,
            "top_texts": top_texts,
            "ocr_count": len(self.last_ocr or []),
            "blockers": blockers,
            "blocker_signature": blocker_signature,
            "visible_resolve_targets": [target for blocker in blockers for target in (blocker.get("resolve_targets") or [])][:8],
            "text_blob": text_blob,
            "text_blob_norm": text_blob_norm,
            "signature_vec": sig_vec,
            "ax": ax_summary,
        }

    def _public_state_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ax = snapshot.get("ax") or {}
        blockers = []
        for blocker in self._active_blockers(snapshot)[:3]:
            blockers.append(
                {
                    "class": blocker.get("class"),
                    "scope": blocker.get("scope"),
                    "confidence": blocker.get("confidence"),
                    "bbox": blocker.get("bbox"),
                    "evidence": list(blocker.get("evidence") or [])[:3],
                    "allow_page_click_through": bool(blocker.get("allow_page_click_through", False)),
                    "resolve_targets": list(blocker.get("resolve_targets") or [])[:4],
                    "suggested_strategies": list(blocker.get("suggested_strategies") or [])[:5],
                }
            )
        return {
            "hash": snapshot.get("hash", ""),
            "tags": list(snapshot.get("tags", []))[:8],
            "top_texts": list(snapshot.get("top_texts", []))[:8],
            "ocr_count": int(snapshot.get("ocr_count", 0)),
            "blocker_signature": str(snapshot.get("blocker_signature", "") or ""),
            "blockers": blockers,
            "visible_resolve_targets": list(snapshot.get("visible_resolve_targets", []))[:8],
            "focused_role": ax.get("focused_role", ""),
            "focused_name": ax.get("focused_name", ""),
            "focused_editable": bool(ax.get("focused_editable", False)),
        }

    def _snapshot_similarity(self, before: Dict[str, Any], after: Dict[str, Any]) -> Optional[float]:
        vb = before.get("signature_vec")
        va = after.get("signature_vec")
        if vb is None or va is None:
            return None
        try:
            return float(np.dot(vb, va) / max(1e-6, (np.linalg.norm(vb) * np.linalg.norm(va))))
        except Exception:
            return None

    def _same_action_same_state_streak(self, action: str, before_hash: str) -> int:
        if not before_hash:
            return 1
        streak = 1
        for item in reversed(self.history[-6:]):
            if item.get("action") != action:
                break
            prev_hash = ((item.get("result") or {}).get("before_state") or {}).get("hash", "")
            if prev_hash != before_hash:
                break
            streak += 1
        return streak

    def _typed_text_visible(self, snapshot: Dict[str, Any], text: str) -> bool:
        snippet = self._normalize_visible_text(text)
        if not snippet:
            return False
        snippet = snippet[:48]
        return snippet in snapshot.get("text_blob_norm", "")

    def _assess_post_action(
        self,
        op: str,
        params: Dict[str, Any],
        dispatch_result: Dict[str, Any],
        before_snapshot: Dict[str, Any],
        after_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        before_tags = set(before_snapshot.get("tags", []))
        after_tags = set(after_snapshot.get("tags", []))
        before_blocker_sig = self._blocker_signature(before_snapshot)
        after_blocker_sig = self._blocker_signature(after_snapshot)
        before_blocker = self._primary_blocker(before_snapshot) or {}
        after_blocker = self._primary_blocker(after_snapshot) or {}
        before_ax = before_snapshot.get("ax") or {}
        after_ax = after_snapshot.get("ax") or {}
        similarity = self._snapshot_similarity(before_snapshot, after_snapshot)
        screen_changed = similarity is not None and similarity < NOOP_SIMILARITY_THRESHOLD
        tags_changed = before_tags != after_tags
        blocker_cleared = bool(before_blocker_sig) and not bool(after_blocker_sig)
        blocker_changed = bool(before_blocker_sig and after_blocker_sig and before_blocker_sig != after_blocker_sig)
        blocker_persisted = bool(before_blocker_sig and after_blocker_sig and before_blocker_sig == after_blocker_sig)
        focus_changed = bool(before_ax.get("focus_hash")) and bool(after_ax.get("focus_hash")) and before_ax.get("focus_hash") != after_ax.get("focus_hash")
        focus_became_editable = bool(after_ax.get("focused_editable")) and not bool(before_ax.get("focused_editable"))

        evidence: List[str] = []
        if screen_changed and similarity is not None:
            evidence.append(f"screen_changed:{similarity:.3f}")
        if tags_changed:
            evidence.append("state_tags_changed")
        if blocker_cleared:
            evidence.append("blocking_overlay_cleared")
        if blocker_changed:
            evidence.append("blocker_changed")
        if blocker_persisted:
            evidence.append("blocker_still_present")
        if str(after_blocker.get("class", "") or "") == "browser_url_suggestion_dropdown" and blocker_persisted:
            evidence.append("browser_overlay_still_present")
        if focus_changed:
            evidence.append("ax_focus_changed")
        if focus_became_editable:
            evidence.append("ax_focus_became_editable")

        verified_candidate = False
        if op == "open_url":
            if "phase:loading_like" in after_tags:
                evidence.append("loading_after_open")
            verified_candidate = (
                screen_changed
                or tags_changed
                or "phase:loading_like" in after_tags
                or ("app:browser_like" in after_tags and "app:browser_like" not in before_tags)
            )
        elif op == "type_text":
            typed_text = "" if params.get("confidential") else str(params.get("text", "") or "")
            if typed_text and typed_text[:48] and typed_text[:48].lower() in (after_ax.get("focused_value_norm") or ""):
                evidence.append("ax_value_contains_typed")
                verified_candidate = True
            if typed_text and self._typed_text_visible(after_snapshot, typed_text):
                evidence.append("typed_text_visible")
                verified_candidate = True
            elif params.get("confidential"):
                verified_candidate = screen_changed or tags_changed
                if verified_candidate:
                    evidence.append("confidential_input_changed_state")
        elif op == "key_seq":
            keys = dispatch_result.get("keys") or params.get("keys") or []
            keys_blob = ",".join(str(k).lower() for k in keys)
            if "return" in keys_blob or "enter" in keys_blob:
                if "phase:loading_like" in after_tags:
                    evidence.append("loading_after_submit")
                verified_candidate = screen_changed or tags_changed or blocker_cleared or "phase:loading_like" in after_tags
            elif "tab" in keys_blob:
                verified_candidate = screen_changed or tags_changed or focus_changed
                if focus_changed:
                    evidence.append("tab_changed_focus")
                elif verified_candidate:
                    evidence.append("focus_or_layout_shift_after_tab")
            else:
                verified_candidate = screen_changed or tags_changed or focus_changed
        else:
            clicked_text = self._normalize_visible_text(str(dispatch_result.get("clicked", "") or ""))
            if clicked_text and clicked_text in before_snapshot.get("text_blob_norm", "") and clicked_text not in after_snapshot.get("text_blob_norm", ""):
                evidence.append("clicked_text_disappeared")
                verified_candidate = True
            if focus_changed:
                evidence.append("click_changed_focus")
                verified_candidate = True
            verified_candidate = verified_candidate or screen_changed or tags_changed or blocker_cleared or blocker_changed

        if blocker_persisted and self._is_blocker_sensitive_action(op, params):
            verified_candidate = False

        return {
            "verified_candidate": bool(verified_candidate),
            "similarity": similarity,
            "evidence": evidence,
            "score": len(evidence) + (1 if verified_candidate else 0),
            "reason": evidence[0] if evidence else "no_observable_change",
        }

    def _verify_post_action(
        self,
        op: str,
        params: Dict[str, Any],
        dispatch_result: Dict[str, Any],
        before_snapshot: Dict[str, Any],
        timeout_s: float = POST_ACTION_VERIFY_TIMEOUT_S,
    ) -> Dict[str, Any]:
        deadline = time.time() + max(0.2, float(timeout_s))
        stable_hits = 0
        best_assessment: Optional[Dict[str, Any]] = None
        last_snapshot = before_snapshot

        while time.time() <= deadline:
            self._capture_state()
            after_snapshot = self._state_snapshot()
            last_snapshot = after_snapshot
            assessment = self._assess_post_action(op, params, dispatch_result, before_snapshot, after_snapshot)
            if best_assessment is None or assessment.get("score", 0) > best_assessment.get("score", 0):
                best_assessment = dict(assessment, after_snapshot=after_snapshot)
            if assessment.get("verified_candidate"):
                stable_hits += 1
                if stable_hits >= POST_ACTION_VERIFY_STABLE_HITS:
                    return {
                        "verified": True,
                        "reason": assessment.get("reason", "observable_transition"),
                        "evidence": assessment.get("evidence", []),
                        "similarity": assessment.get("similarity"),
                        "after_snapshot": after_snapshot,
                    }
            else:
                stable_hits = 0
            time.sleep(POST_ACTION_VERIFY_POLL_S)

        best = best_assessment or {
            "reason": "outcome_not_verified",
            "evidence": [],
            "similarity": self._snapshot_similarity(before_snapshot, last_snapshot),
            "after_snapshot": last_snapshot,
        }
        return {
            "verified": False,
            "reason": best.get("reason", "outcome_not_verified"),
            "evidence": best.get("evidence", []),
            "similarity": best.get("similarity"),
            "after_snapshot": best.get("after_snapshot", last_snapshot),
        }

    def _finalize_action_result(
        self,
        op: str,
        params: Dict[str, Any],
        raw_result: Dict[str, Any],
        before_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(raw_result or {})
        result.setdefault("recovery_strategy", self._action_family(op, params))
        result["before_state"] = self._public_state_snapshot(before_snapshot)
        result["same_action_same_state_streak"] = self._same_action_same_state_streak(op, before_snapshot.get("hash", ""))

        if result.get("status") != "success":
            after_snapshot = self._state_snapshot()
            result["after_state"] = self._public_state_snapshot(after_snapshot)
            result["event_applied"] = bool(result.get("event_applied", False))
            result["outcome_verified"] = False
            result["blocker_class"] = (self._primary_blocker(after_snapshot) or self._primary_blocker(before_snapshot) or {}).get("class")
            result["blocker_signature"] = self._blocker_signature(after_snapshot) or self._blocker_signature(before_snapshot)
            result["recovery_effect"] = self._recovery_effect(before_snapshot, after_snapshot)
            failure_reason = str(result.get("error_code") or "event_dispatch_failed")
            result["verification"] = {
                "verified": False,
                "reason": failure_reason.lower(),
                "evidence": [result["recovery_effect"]] if result.get("recovery_effect") else [],
                "similarity": self._snapshot_similarity(before_snapshot, after_snapshot),
            }
            if not result.get("recovery_hint"):
                result["recovery_hint"] = self._default_blocker_recovery_hint(before_snapshot, after_snapshot)
            return result

        passive_ops = {"wait_text", "wait_any_text", "sleep", "ocr_extract", "run_llm"}
        if op in passive_ops:
            after_snapshot = self._state_snapshot()
            result["after_state"] = self._public_state_snapshot(after_snapshot)
            result["event_applied"] = True
            result["outcome_verified"] = True
            result["verification"] = {
                "verified": True,
                "reason": "passive_action_completed",
                "evidence": [],
                "similarity": self._snapshot_similarity(before_snapshot, after_snapshot),
            }
            return result

        verification = self._verify_post_action(op, params, result, before_snapshot)
        after_snapshot = verification.get("after_snapshot", before_snapshot)
        result["after_state"] = self._public_state_snapshot(after_snapshot)
        result["event_applied"] = True
        result["outcome_verified"] = bool(verification.get("verified"))
        result["blocker_class"] = (self._primary_blocker(after_snapshot) or self._primary_blocker(before_snapshot) or {}).get("class")
        result["blocker_signature"] = self._blocker_signature(after_snapshot) or self._blocker_signature(before_snapshot)
        result["recovery_effect"] = self._recovery_effect(before_snapshot, after_snapshot)
        result["verification"] = {
            "verified": bool(verification.get("verified")),
            "reason": verification.get("reason", ""),
            "evidence": verification.get("evidence", []),
            "similarity": verification.get("similarity"),
        }
        if verification.get("verified"):
            result["status"] = "success"
        else:
            result["status"] = "dispatched"
            result["error_code"] = "OUTCOME_NOT_VERIFIED"
            if result.get("same_action_same_state_streak", 1) > 1 and not self._blocker_signature(after_snapshot):
                result["recovery_hint"] = "change_action_family"
            else:
                result["recovery_hint"] = self._default_blocker_recovery_hint(before_snapshot, after_snapshot)
        return result

    def _apply_candidate(self, cand: Candidate) -> Dict[str,Any]:
        if cand.action == "click_box":
            self._capture_state()  # refresh last_img so to_abs works
            x1,y1,x2,y2 = cand.params["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
            x_abs,y_abs = self.grabber.to_abs(cx,cy)
            if not CLICK_ENABLED or not self._xdotool_ok:
                return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
            _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
            _safe_run(["xdotool","click","1"])
            return {"status":"success","applied":"click_box","box":cand.params["box"],"why":cand.why}

        if cand.action == "keys":
            seq = self._normalize_key_sequence(cand.params.get("keys",[]))
            for k in seq:
                _safe_run(["xdotool","key","--clearmodifiers",k])
            return {"status":"success","applied":"keys","keys":seq,"why":cand.why}

        return {"status":"failure","error_code":"UNKNOWN_CAND_ACTION"}

    # This is the heart of the "several opinions"
    def _consensus_step(self, intent: str, verify_patterns: List[str], timeout_after_click: float = 2.0):
        """
        intent: "compose" | "send" | "dismiss" (extend freely)
        verify_patterns: text regexes that must appear after a successful action.
        """
        # 1) Observe full perception (OCR + det + A11y if available)
        img, _ = self.grabber.capture()
        img_h, img_w = img.shape[:2]
        ocr_min = [{"text": w["text"], "box": w["box"], "conf": w["conf"]} for w in self.last_ocr] if self.last_ocr else []
        # Pull detections without overlay drawings
        dets = []
        try:
            dets = rtdetr_detect(self.session, img, timeout=(2.0,4.0), retries=1)
        except Exception:
            pass
        obs = fuse_observation(img_w=img_w, img_h=img_h, ocr_items=ocr_min, dets=dets, label_map=None)

        # 2) Generate candidates from multiple proposers
        proposers = []
        if intent == "compose":
            proposers = [ComposeProposer(), DismissModalProposer()]
        elif intent == "send":
            proposers = [SendProposer(), DismissModalProposer()]
        else:
            proposers = [DismissModalProposer()]

        cands = []
        for p in proposers:
            cands.extend(p.propose(intent, obs))

        # 3) Try in ranked order; verify each
        for cand in arbitrate(cands, k=6):
            applied = self._apply_candidate(cand)
            self._log("info","consensus.try",{"intent": intent, "candidate": cand.__dict__, "result": applied})
            if applied.get("status") != "success":
                continue
            # small wait for UI to react
            self._action_sleep(timeout_after_click)
            # verification
            r = self._action_wait_any_text(verify_patterns, timeout_s=4)
            if r.get("status") == "success":
                return {"status": "success", "picked": cand.__dict__}
        return {"status": "failure", "reason":"no candidate verified"}

    def _detect_brittle_intent(self, action: str, params: Dict[str, Any]) -> Optional[Tuple[str, List[str]]]:
        """
        Detect planner-decided steps that are brittle (text/icon dependent) and
        map them to a higher-level intent that can be handled by consensus.

        Returns (intent, verify_patterns) or None if not applicable.
        """
        try:
            # Normalize candidate patterns from params regardless of action flavor
            patterns: List[str] = []
            if action == "click_text" and isinstance(params.get("regex"), str):
                patterns = [params.get("regex")]  # single regex
            elif action == "click_any_text" and isinstance(params.get("patterns"), list):
                patterns = [p for p in params.get("patterns") if isinstance(p, str)]
            elif action == "click_near_text" and isinstance(params.get("anchor_regex"), str):
                patterns = [params.get("anchor_regex")]  # anchor-based

            # Intent: compose
            compose_rx = re.compile(r"compose|new (mail|message)|write( message)?", re.I)
            if patterns and any(compose_rx.search(p) for p in patterns):
                return ("compose", [r"^To$", r"^Recipients?$", r"^Subject$"])

            # Intent: send
            send_rx = re.compile(r"^send( now| & archive)?$", re.I)
            if patterns and any(send_rx.search(p) for p in patterns):
                # Conservative verify: look for ephemeral sent toast or compose window closing cues
                return ("send", [r"^Message sent", r"^Undo$", r"^Inbox$"])

            # Keyboard-only intents
            if action == "key_seq":
                keys = params.get("keys", []) if isinstance(params.get("keys", []), list) else []
                joined = ",".join([str(k) for k in keys]).lower()
                if "ctrl+return" in joined or "ctrl+kp_enter" in joined:
                    return ("send", [r"^Message sent", r"^Undo$", r"^Inbox$"])
        except Exception:
            pass
        return None

    def _canonicalize_params(self, action: str, params: dict) -> dict:
        """Map common LLM synonyms to the executor's canonical arg names and drop unknowns."""
        p = dict(params or {})

        # Global aliases
        if "timeout" in p and "timeout_s" not in p:
            p["timeout_s"] = p.pop("timeout")
        if "seconds" in p and "timeout_s" not in p and action == "sleep":
            # keep seconds as-is for sleep, don't rename
            pass

        if action == "click_text":
            # Accept 'text' as alias for 'regex'
            if "text" in p and "regex" not in p:
                p["regex"] = p.pop("text")
            # keep only valid keys
            allow = {"regex", "nth", "prefer_bold", "fuzzy_text", "fuzzy_threshold"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "wait_text":
            allow = {"regex", "timeout_s"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "wait_any_text":
            # 'texts' -> 'patterns'
            if "texts" in p and "patterns" not in p:
                p["patterns"] = p.pop("texts")
            allow = {"patterns", "timeout_s"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "click_any_text":
            # 'texts' -> 'patterns'
            if "texts" in p and "patterns" not in p:
                p["patterns"] = p.pop("texts")
            allow = {"patterns", "nth", "prefer_bold"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "click_near_text":
            # 'anchor' -> 'anchor_regex'
            if "anchor" in p and "anchor_regex" not in p:
                p["anchor_regex"] = p.pop("anchor")
            allow = {"anchor_regex", "dx", "dy"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "click_box":
            if "bbox" in p and "box" not in p:
                p["box"] = p.pop("bbox")
            allow = {"box"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "type_text":
            allow = {"text", "confidential"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "key_seq":
            allow = {"keys"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "sleep":
            allow = {"seconds"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "ocr_extract":
            allow = {"save_as"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "open_url":
            allow = {"url"}
            return {k: v for k, v in p.items() if k in allow}

        if action == "end_task":
            allow = {"reason"}
            return {k: v for k, v in p.items() if k in allow}

        # Default: pass through
        return p

    def _capture_state(self):
        img, _ = self.grabber.capture()
        self.last_img = img
        ocr_levels = ocr_image_levels(img)
        self.last_ocr_words = list(ocr_levels.get("words") or [])
        self.last_ocr_lines = list(ocr_levels.get("lines") or [])
        self.last_ocr_source = str(ocr_levels.get("source") or "")
        self.last_ax_nodes = self._fetch_ax_nodes()
        # Keep words for precise targeting and lines for multi-word targeting / planner context.
        self.last_ocr_words = sorted(self.last_ocr_words, key=lambda w: (-(w["box"][3]-w["box"][1]), -float(w.get("conf", 100))))[:OCR_LIMIT]
        self.last_ocr_lines = sorted(self.last_ocr_lines, key=lambda w: (-(w["box"][3]-w["box"][1]), -float(w.get("conf", 100))))[: max(40, min(OCR_LIMIT, 160))]
        self.last_ocr = self.last_ocr_words
        self._log(
            "debug",
            "state.captured",
            {
                "ocr_words": len(self.last_ocr_words),
                "ocr_lines": len(self.last_ocr_lines),
                "ocr_source": self.last_ocr_source,
                "ax_nodes": len(self.last_ax_nodes),
            },
        )

    def _action_wait_any_text(self, patterns, timeout_s: int = 20):
        t0 = time.time()
        while time.time() - t0 <= timeout_s:
            self._capture_state()
            found = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, any_regex=patterns)
            if found:
                return {
                    "status": "success",
                    "match": {"text": found["text"], "box": found["box"]},
                    "ocr_level": found.get("level", "word"),
                    "targeting_source": f"ocr_{found.get('level', 'word')}",
                }
            time.sleep(0.4)
        return {"status":"failure","error_code":"TIMEOUT_ANY","error_message": f"None matched in {patterns}"}

    def _action_click_any_text(self, patterns, nth: int = 0, prefer_bold: bool = True):
        # try each regex in order until one has a match
        import re
        for pat in patterns:
            w = find_text_box(
                self.last_ocr_words or self.last_ocr,
                line_items=self.last_ocr_lines,
                regex=pat,
                nth=nth,
                prefer_bold=prefer_bold
            )
            if w:
                x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
                self._debug_save_click_crop(
                    box=[int(x1), int(y1), int(x2), int(y2)],
                    click_rel=(int(cx), int(cy)),
                    action="click_any_text",
                    note=f"pattern_used={pat} level={w.get('level', 'word')}",
                )
                x_abs,y_abs = self.grabber.to_abs(cx,cy)
                if not CLICK_ENABLED or not self._xdotool_ok:
                    return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
                _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
                _safe_run(["xdotool","click","1"])
                return {
                    "status": "success",
                    "clicked": w["text"],
                    "box": w["box"],
                    "pattern_used": pat,
                    "ocr_level": w.get("level", "word"),
                    "targeting_source": f"ocr_{w.get('level', 'word')}",
                }
        return {"status":"failure","error_code":"ELEMENT_NOT_FOUND","error_message": f"No pattern matched: {patterns}"}

    def _action_click_near_text(self, anchor_regex: str, dx: int = 0, dy: int = 0):
        import re
        w = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, regex=anchor_regex)
        if not w:
            return {"status":"failure","error_code":"ANCHOR_NOT_FOUND","error_message": anchor_regex}
        x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
        # Guard against header/account anchors (e.g., email in header)
        if "@" in w.get("text", "") and cy < 80:
            return {"status":"failure","error_code":"ANCHOR_REJECTED_HEADER","error_message": w.get("text", "")}
        click_x, click_y = int(cx + dx), int(cy + dy)
        # For near-text clicks, define a small box around the click point for cropping.
        self._debug_save_click_crop(
            box=[click_x - 12, click_y - 12, click_x + 12, click_y + 12],
            click_rel=(click_x, click_y),
            action="click_near_text",
            note=f"anchor={w.get('text','')} level={w.get('level', 'word')} offset=[{dx},{dy}]",
        )
        x_abs,y_abs = self.grabber.to_abs(click_x, click_y)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
        _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
        _safe_run(["xdotool","click","1"])
        return {
            "status": "success",
            "anchor": w["text"],
            "anchor_box": w["box"],
            "offset": [dx, dy],
            "anchor_level": w.get("level", "word"),
            "targeting_source": f"ocr_{w.get('level', 'word')}",
        }

    def _action_click_box(self, box):
        try:
            x1, y1, x2, y2 = [int(v) for v in (box or [0, 0, 0, 0])]
        except Exception:
            return {"status": "failure", "error_code": "BAD_BOX", "error_message": str(box)}
        # Basic sanity check
        if x2 <= x1 or y2 <= y1:
            return {"status": "failure", "error_code": "BAD_BOX", "error_message": str(box)}
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self._debug_save_click_crop(
            box=[x1, y1, x2, y2],
            click_rel=(int(cx), int(cy)),
            action="click_box",
        )
        x_abs, y_abs = self.grabber.to_abs(cx, cy)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status": "failure", "error_code": "CLICK_DISABLED_OR_MISSING_XDOTOOL"}
        _safe_run(["xdotool", "mousemove", "--sync", str(x_abs), str(y_abs)])
        _safe_run(["xdotool", "click", "1"])
        return {"status": "success", "box": [x1, y1, x2, y2], "clicked_abs": [x_abs, y_abs]}

    def _action_sleep(self, seconds: float = 0.8):
        time.sleep(float(seconds))
        return {"status":"success","slept_seconds": float(seconds)}

    # --- Action impls (call your existing primitives) ---
    from typing import Optional
    def _action_click_text(
        self,
        regex: Optional[str] = None, nth: int = 0, prefer_bold: bool = False,
        fuzzy_text: Optional[str] = None, fuzzy_threshold: Optional[float] = None
    ):
        if not (self.last_ocr_words or self.last_ocr_lines or self.last_ocr):
            return {"status": "failure", "error_code": "NO_OCR", "error_message": "No OCR results yet"}
        w = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, regex=regex, nth=nth, prefer_bold=prefer_bold,
                          fuzzy_text=fuzzy_text, fuzzy_threshold=fuzzy_threshold)
        if not w:
            return {"status": "failure", "error_code": "ELEMENT_NOT_FOUND",
                    "error_message": f"Regex '{regex}' not found"}
        x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
        self._debug_save_click_crop(
            box=[int(x1), int(y1), int(x2), int(y2)],
            click_rel=(int(cx), int(cy)),
            action="click_text",
            note=f"level={w.get('level', 'word')}",
        )
        x_abs,y_abs = self.grabber.to_abs(cx,cy)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status": "failure", "error_code": "CLICK_DISABLED_OR_MISSING_XDOTOOL",
                    "error_message": "CLICK_ENABLED=0 or xdotool missing"}
        _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
        _safe_run(["xdotool","click","1"])
        return {
            "status": "success",
            "clicked": w["text"],
            "box": w["box"],
            "ocr_level": w.get("level", "word"),
            "targeting_source": f"ocr_{w.get('level', 'word')}",
        }

    def _action_type_text(self, text: str, confidential: bool = False):
        ax = self._state_snapshot().get("ax") or {}
        if ax.get("available") and not ax.get("focused_editable"):
            return {
                "status": "failure",
                "error_code": "FOCUS_NOT_EDITABLE",
                "error_message": f"Focused AX node is not editable ({ax.get('focused_role') or 'unknown'})",
                "event_applied": False,
                "recovery_hint": "refocus_editable_field",
            }
        # Allow ${VAR} substitution from env/ctx
        def repl(m): 
            key = m.group(1)
            return str(os.environ.get(key, self.ctx.get(key, "")))
        s = re.sub(r"\$\{([^}]+)\}", repl, text)
        # Use your robust multi-line typer
        lines = s.splitlines()
        for i, line in enumerate(lines):
            _safe_run(["xdotool","type","--clearmodifiers", line])
            if i < len(lines) - 1:
                _safe_run(["xdotool","key","Return"])
        return {"status": "success", "typed_chars": len(s), "preview": redact_if_confidential(s[:8]+"...", confidential)}

    def _normalize_key_sequence(self, keys: list) -> list:
        """
        Turn ["CTRL","ENTER"] into ["ctrl+Return"], normalize case, keep existing combos like "ctrl+Shift+Tab".
        """
        out = []
        i = 0
        to_lower = lambda s: s.lower().replace(" ", "")

        while i < len(keys):
            k = keys[i]
            k_low = to_lower(k)

            # ctrl + enter / return / kp_enter
            if k_low in ("ctrl","control") and i + 1 < len(keys):
                nxt = to_lower(keys[i+1])
                if nxt in ("enter","return","kp_enter"):
                    out.append("ctrl+Return" if nxt != "kp_enter" else "ctrl+KP_Enter")
                    i += 2
                    continue

            # ctrl + shift + enter, common for "Send & archive"
            if k_low in ("ctrl","control") and i + 2 < len(keys):
                nxt = to_lower(keys[i+1]); nxt2 = to_lower(keys[i+2])
                if nxt in ("shift") and nxt2 in ("enter","return","kp_enter"):
                    out.append("ctrl+Shift+Return" if nxt2 != "kp_enter" else "ctrl+Shift+KP_Enter")
                    i += 3
                    continue

            # If already a combo like "ctrl+Return" or "shift+tab", keep as-is
            if "+" in k:
                out.append(k)
            else:
                # normalize some common names to xdotool names
                mapping = {
                    "enter": "Return",
                    "return": "Return",
                    "esc": "Escape",
                    "escape": "Escape",
                    "space": "space",
                    "tab": "Tab",
                }
                out.append(mapping.get(k_low, k))
            i += 1

        return out


    def _action_key_seq(self, keys: List[str]):
        seq = self._normalize_key_sequence(keys)
        for k in seq:
            _safe_run(["xdotool", "key", "--clearmodifiers", k])
        return {"status": "success", "keys": seq}

    def _action_wait_text(self, regex: str, timeout_s: int = 20):
        t0 = time.time()
        while time.time() - t0 <= timeout_s:
            self._capture_state()
            match = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, regex=regex)
            if match:
                return {
                    "status": "success",
                    "regex": regex,
                    "ocr_level": match.get("level", "word"),
                    "targeting_source": f"ocr_{match.get('level', 'word')}",
                }
            time.sleep(0.4)
        return {"status": "failure", "error_code": "TIMEOUT", "error_message": f"wait_text timeout for '{regex}'"}

    def _action_ocr_extract(self, save_as: str):
        text_items = self.last_ocr_lines or self.last_ocr_words or self.last_ocr
        text = "\n".join(w["text"] for w in text_items)
        self.ctx[save_as] = text
        return {"status": "success", "var": save_as, "len": len(text)}

    def _action_open_url(self, url: str):
        firefox_bin = os.environ.get("FIREFOX_BIN") or which("firefox-esr") or which("firefox")
        if not firefox_bin:
            if which("xdg-open"):
                subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"status": "success", "url": url, "launcher": "xdg-open"}
            subprocess.Popen(["firefox", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"status": "success", "url": url, "launcher": "firefox"}

        args = [firefox_bin]
        profile = os.environ.get("FIREFOX_PROFILE_PATH", "").strip()
        if profile:
            # Auto-create the Firefox profile directory if missing (ESR sometimes requires it)
            if not os.path.isdir(profile):
                try:
                    subprocess.run(
                        [firefox_bin, "-CreateProfile", f"agent {profile}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                except Exception:
                    pass
            args += ["--no-remote", "--new-instance", "--profile", profile]
        args += ["--new-window", url]
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"status": "success", "url": url}

    # --- Planner request/response ---
    def _post_to_planner(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        headers = {"Content-Type":"application/json"}
        if PLANNER_API_KEY:
            headers["Authorization"] = f"Bearer {PLANNER_API_KEY}"
        try:
            r = self.session.post(TASK_PLANNER_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self._log("error","planner.request.failed",{"error": str(e)})
            return None

                
    # Optional LLM helper (if you want replies drafted by LLM)
    def _action_run_llm(self, system: str, prompt: str, var_out: str = "draft"):
        out = run_llm(system, prompt)  # uses your existing helper & env
        self.ctx[var_out] = out
        return {"status": "success", "var_out": var_out, "chars": len(out)}


    def run(self):
        if not AGENT_GOAL:
            self._log("error", "executor.abort", {"reason": "AGENT_GOAL empty"})
            return

        step = 0
        while step < MAX_STEPS:
            # Monotonic trace id for HTML grid + dumps
            self.trace_idx += 1
            self._last_click_debug = None
            # 1) Observe
            self._capture_state()
            pre_action_snapshot = self._state_snapshot()
            public_state = self._public_state_snapshot(pre_action_snapshot)
            public_state["recovery_options"] = self._recovery_options(pre_action_snapshot)
            public_state["recent_blocker_attempts"] = self._recent_blocker_attempts(public_state.get("blocker_signature", ""))[-6:]
            ocr_min: List[Dict[str, Any]] = []

            # 2) Report
            # Optionally attach a compressed screenshot so the planner can use a vision model.
            # Keep payload bounded to avoid ballooning latency/tokens.
            screenshot_b64 = None
            screenshot_mime = "image/jpeg"
            ui_elements: List[Dict[str, Any]] = []
            try:
                if os.environ.get("PLANNER_SEND_SCREENSHOT", "1").strip() != "0" and self.last_img is not None:
                    q = int(os.environ.get("PLANNER_SCREENSHOT_JPEG_QUALITY", "60"))
                    q = max(25, min(90, q))
                    jpg = encode_jpeg_bgr(self.last_img, q=q)
                    screenshot_b64 = base64.b64encode(jpg).decode("ascii")
            except Exception as e:
                self._log("warn", "planner.screenshot.encode_failed", {"error": str(e)})

            try:
                # OCR elements (boxes are image-relative)
                max_ocr_line_elems = int(os.environ.get("PLANNER_MAX_OCR_LINE_ELEMENTS", "80"))
                max_ocr_word_elems = int(os.environ.get("PLANNER_MAX_OCR_WORD_ELEMENTS", "140"))
                max_ocr_line_elems = max(0, min(200, max_ocr_line_elems))
                max_ocr_word_elems = max(0, min(300, max_ocr_word_elems))
                for w in self.last_ocr_lines[:max_ocr_line_elems]:
                    ocr_min.append({"text": w["text"], "box": w["box"], "conf": w["conf"], "level": "line"})
                    ui_elements.append({
                        "source": "ocr_line",
                        "text": w.get("text", ""),
                        "box": w.get("box", [0, 0, 0, 0]),
                        "score": float(w.get("conf", 0.0)) / 100.0 if float(w.get("conf", 0.0)) > 1.0 else float(w.get("conf", 0.0)),
                        "role": None,
                    })
                for w in self.last_ocr_words[:max_ocr_word_elems]:
                    ocr_min.append({"text": w["text"], "box": w["box"], "conf": w["conf"], "level": "word"})
                    ui_elements.append({
                        "source": "ocr_word",
                        "text": w.get("text", ""),
                        "box": w.get("box", [0, 0, 0, 0]),
                        "score": float(w.get("conf", 0.0)) / 100.0 if float(w.get("conf", 0.0)) > 1.0 else float(w.get("conf", 0.0)),
                        "role": None,
                    })

                # Optional detections (icon-only affordances). Disabled by default.
                if os.environ.get("PLANNER_SEND_DETECTIONS", "0").strip() == "1" and self.last_img is not None:
                    max_det = int(os.environ.get("PLANNER_MAX_DETECTIONS", "60"))
                    max_det = max(0, min(200, max_det))
                    dets = rtdetr_detect(self.session, self.last_img, timeout=(2.0, 4.0), retries=1) or []
                    dets = sorted(dets, key=lambda d: -float(d.get("score", 0.0)))[:max_det]
                    for d in dets:
                        ui_elements.append({
                            "source": "det",
                            "text": f"label:{int(d.get('label', -1))}",
                            "box": d.get("box", [0, 0, 0, 0]),
                            "score": float(d.get("score", 0.0)),
                            "role": None,
                        })

                if self.last_ax_nodes:
                    max_ax_elems = int(os.environ.get("PLANNER_MAX_AX_ELEMENTS", "80"))
                    max_ax_elems = max(0, min(200, max_ax_elems))
                    for node in self.last_ax_nodes[:max_ax_elems]:
                        box = node.get("box", [0, 0, 0, 0])
                        if not isinstance(box, list) or len(box) != 4:
                            box = [0, 0, 0, 0]
                        ui_elements.append({
                            "source": "ax",
                            "text": self._ax_text(node) or self._ax_value_text(node) or str(node.get("role", "") or ""),
                            "box": [int(v) for v in box],
                            "score": float(node.get("score", 0.95) or 0.95),
                            "role": str(node.get("role", "") or None) if node.get("role") else None,
                        })
            except Exception as e:
                self._log("warn", "planner.ui_elements.build_failed", {"error": str(e)})

            payload = {
                "goal": AGENT_GOAL,
                "task_history": self.history[-HISTORY_WINDOW:],
                "current_state": public_state,
                "ocr_results": ocr_min,
                "ui_elements": ui_elements,
                "screenshot_b64": screenshot_b64,
                "screenshot_mime": screenshot_mime,
                "available_actions": [
                    "open_url","wait_text","wait_any_text",
                    "click_text","click_any_text","click_near_text","click_box",
                    "type_text","key_seq","sleep","ocr_extract","end_task"
                ]
            }

            if os.environ.get("AGENT_LOG_PLANNER_SUMMARY", "1").strip() == "1":
                self._log(
                    "info",
                    "planner.payload.summary",
                    {
                        "step": step,
                        "goal_len": len(AGENT_GOAL or ""),
                        "task_history_n": len(payload.get("task_history") or []),
                        "ocr_results_n": len(payload.get("ocr_results") or []),
                        "ocr_line_results_n": len(self.last_ocr_lines),
                        "ocr_word_results_n": len(self.last_ocr_words),
                        "ui_elements_n": len(payload.get("ui_elements") or []),
                        "has_screenshot": bool(payload.get("screenshot_b64")),
                        "screenshot_b64_len": len(payload.get("screenshot_b64") or "") if payload.get("screenshot_b64") else 0,
                        "available_actions_n": len(payload.get("available_actions") or []),
                    },
                )
            resp = self._post_to_planner(payload)
            if not resp:
                time.sleep(2)
                continue

            op = resp.get("action")
            params = resp.get("args", {}) or resp.get("parameters", {}) or {}
            params = self._canonicalize_params(op, params)
            reasoning = resp.get("reasoning","")
            completed = bool(resp.get("completed", False))
            self._log("info","planner.decision",{"action": op, "params": params, "why": reasoning})
            planner_op = op
            planner_params = dict(params)

            # Trace: save the pre-action frame (what the planner saw)
            frame_path = ""
            if self.trace_enabled and self.last_img is not None:
                frame_path = self._save_trace_frame(op, "frame")

            # 3) Act
            result = {"status":"failure","error_code":"UNKNOWN_ACTION","error_message": op}
            try:
                policy = self._blocker_policy_directive(op, params, pre_action_snapshot)
                if policy.get("decision") == "override":
                    op = str(policy.get("action") or op)
                    params = self._canonicalize_params(op, policy.get("params") or {})
                    override_reason = str(policy.get("reason") or "")
                    executed_strategy = str(policy.get("strategy") or self._action_family(op, params))
                    policy_blocker = policy.get("blocker") or {}
                    self._log(
                        "info",
                        "executor.override",
                        {
                            "from_action": planner_op,
                            "to_action": op,
                            "reason": override_reason,
                            "blocker_class": policy_blocker.get("class"),
                            "strategy": executed_strategy,
                        },
                    )
                else:
                    override_reason = ""
                    executed_strategy = str(policy.get("strategy") or self._action_family(op, params))
                    policy_blocker = policy.get("blocker") or {}

                guard_result = self._executor_guard_result(op, params, pre_action_snapshot)
                if guard_result is not None:
                    result = self._finalize_action_result(op, params, guard_result, pre_action_snapshot)
                else:
                    # Intercept brittle planner steps with a consensus step first
                    brittle = self._detect_brittle_intent(op, params)
                    if brittle:
                        intent, verify_patterns = brittle
                        self._log("info","consensus.invoke", {"intent": intent, "verify": verify_patterns})
                        res = self._consensus_step(intent=intent, verify_patterns=verify_patterns)
                        if res.get("status") == "success":
                            after_snapshot = self._state_snapshot()
                            result = {
                                "status":"success",
                                "via":"consensus",
                                "picked": res.get("picked"),
                                "event_applied": True,
                                "outcome_verified": True,
                                "planner_requested_action": planner_op,
                                "planner_requested_parameters": planner_params,
                                "executed_action": f"consensus:{intent}",
                                "executed_parameters": {"verify": verify_patterns},
                                "recovery_strategy": "consensus",
                                "before_state": self._public_state_snapshot(pre_action_snapshot),
                                "after_state": self._public_state_snapshot(after_snapshot),
                                "same_action_same_state_streak": self._same_action_same_state_streak(f"consensus:{intent}", pre_action_snapshot.get("hash", "")),
                                "verification": {
                                    "verified": True,
                                    "reason": "consensus_verified_transition",
                                    "evidence": list(verify_patterns),
                                    "similarity": self._snapshot_similarity(pre_action_snapshot, after_snapshot),
                                },
                            }
                            # After success, optionally insert a short sleep to stabilize UI
                            time.sleep(0.3)
                            post_frame_path = self._save_trace_frame(f"consensus_{intent}", "post")
                            # record and continue to next loop without invoking the brittle action itself
                            self.history.append({"action": f"consensus:{intent}", "parameters": {"verify": verify_patterns}, "result": result})
                            self._log("info","action.result", {"action": f"consensus:{intent}", "result": result})
                            crop_overlay = (self._last_click_debug or {}).get("overlay_path") if self._last_click_debug else ""
                            self._trace_append(
                                {
                                    "idx": self.trace_idx,
                                    "ts": now_utc_iso(),
                                    "decision": {"action": f"consensus:{intent}", "parameters": {"verify": verify_patterns}, "reasoning": reasoning, "completed": completed},
                                    "result": result,
                                    "frame_path": frame_path,
                                    "post_frame_path": post_frame_path,
                                    "crop_overlay_path": crop_overlay,
                                    "ocr_results_n": len(ocr_min),
                                    "ocr_line_results_n": len(self.last_ocr_lines),
                                    "ocr_word_results_n": len(self.last_ocr_words),
                                    "ui_elements_n": len(ui_elements),
                                    "screenshot_b64_len": len(screenshot_b64) if screenshot_b64 else 0,
                                }
                            )
                            self._trace_render_html()
                            step += 1
                            time.sleep(0.6)
                            continue
                        else:
                            self._log("warn","consensus.failed", {"intent": intent, "reason": res.get("reason")})

                    if op == "done" or op == "end_task":
                        self._log("info", "task.complete", {"reason": params.get("reason","planner requested end")})
                        break
                    if op == "open_url":         result = self._action_open_url(**params)
                    elif op == "wait_text":      result = self._action_wait_text(**params)
                    elif op == "wait_any_text":  result = self._action_wait_any_text(**params)
                    elif op == "click_text":     result = self._action_click_text(**params)
                    elif op == "click_any_text": result = self._action_click_any_text(**params)
                    elif op == "click_near_text":result = self._action_click_near_text(**params)
                    elif op == "click_box":      result = self._action_click_box(**params)
                    elif op == "type_text":      result = self._action_type_text(**params)
                    elif op == "key_seq":        result = self._action_key_seq(**params)
                    elif op == "sleep":          result = self._action_sleep(**params)
                    elif op == "ocr_extract":    result = self._action_ocr_extract(**params)
                    elif op == "run_llm":        result = self._action_run_llm(**params)
                    result = self._finalize_action_result(op, params, result, pre_action_snapshot)
                    # Debounce after dispatched or verified clicks
                    if op in ("click_text","click_any_text","click_near_text") and result.get("event_applied"):
                        time.sleep(0.25)

                result["planner_requested_action"] = planner_op
                result["planner_requested_parameters"] = planner_params
                result["executed_action"] = op
                result["executed_parameters"] = params
                result["recovery_strategy"] = result.get("recovery_strategy") or executed_strategy
                if override_reason:
                    result["executor_override_reason"] = override_reason
                    result["executor_override_from"] = {"action": planner_op, "parameters": planner_params}
                if policy_blocker:
                    result.setdefault("blocker_class", policy_blocker.get("class"))
                    result.setdefault("blocker_signature", policy_blocker.get("signature"))
            except Exception as e:
                result = self._finalize_action_result(
                    op,
                    params,
                    {"status":"failure","error_code":"EXCEPTION","error_message": str(e), "event_applied": False},
                    pre_action_snapshot,
                )
                result["planner_requested_action"] = planner_op
                result["planner_requested_parameters"] = planner_params
                result["executed_action"] = op
                result["executed_parameters"] = params

            # 4) Record
            self.history.append({"action": op, "parameters": params, "result": result})
            self._log("info","action.result",{"action": op, "result": result})

            # 5) Trace record + regenerate HTML (last N)
            post_frame_path = self._save_trace_frame(op, "post")
            crop_overlay = (self._last_click_debug or {}).get("overlay_path") if self._last_click_debug else ""
            self._trace_append(
                {
                    "idx": self.trace_idx,
                    "ts": now_utc_iso(),
                    "decision": {"action": planner_op, "parameters": planner_params, "reasoning": reasoning, "completed": completed},
                    "result": result,
                    "frame_path": frame_path,
                    "post_frame_path": post_frame_path,
                    "crop_overlay_path": crop_overlay,
                    "ocr_results_n": len(ocr_min),
                    "ocr_line_results_n": len(self.last_ocr_lines),
                    "ocr_word_results_n": len(self.last_ocr_words),
                    "ui_elements_n": len(ui_elements),
                    "screenshot_b64_len": len(screenshot_b64) if screenshot_b64 else 0,
                }
            )
            self._trace_render_html()
            step += 1
            time.sleep(0.8)
 
# -----------------------
# Main
# -----------------------
# --- Entrypoint tweak: choose static vs dynamic ---
if __name__ == "__main__":
    if AGENT_MODE == "dynamic":
        ActionExecutorDynamic().run()
    else:
        TaskRunner(TASK_FILE).run()   # your original YAML path

       