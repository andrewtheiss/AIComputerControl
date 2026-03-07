# scripts/agent.py
import os, re, time, json, threading, subprocess, io, uuid, shutil, traceback
from ui_core import UIElement, Observation
from perception import fuse_observation
from decision import ComposeProposer, SendProposer, DismissModalProposer, arbitrate, Candidate
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

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

def _score_boldish_height(box: List[int]) -> int:
    return box[3] - box[1]

def find_text_box(
    ocr_items: List[Dict[str, Any]],
    regex: Optional[str] = None,
    any_regex: Optional[List[str]] = None,
    fuzzy_text: Optional[str] = None,
    fuzzy_threshold: Optional[float] = None,
    prefer_bold: bool = False,
    nth: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Find a word box by regex or fuzzy text.
    - regex / any_regex: case-insensitive compiled patterns
    - fuzzy_text + fuzzy_threshold (0..100): RapidFuzz partial_ratio scoring
    """
    patterns = []
    if regex:
        patterns.append(re.compile(regex, re.I))
    if any_regex:
        patterns += [re.compile(r, re.I) for r in any_regex]

    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for w in ocr_items:
        s = w["text"]
        match_score = None

        if patterns:
            ok = any(p.search(s) for p in patterns)
            if ok:
                match_score = 100.0
        elif fuzzy_text:
            # partial_ratio works well for UI snippets
            score = fuzz.partial_ratio(fuzzy_text, s)
            if fuzzy_threshold is None or score >= float(fuzzy_threshold):
                match_score = float(score)

        if match_score is None:
            continue

        bonus = _score_boldish_height(w["box"]) if prefer_bold else 0
        candidates.append((match_score + bonus, w))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    idx = min(nth, len(candidates) - 1)
    return candidates[idx][1]

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
            self.last_ocr = ocr_image(img)
            self._log("debug", "ocr.done", {"n": len(self.last_ocr)})
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
                ocr, regex=regex, any_regex=any_regex,
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
                action = (it.get("decision") or {}).get("action", "")
                params = json.dumps((it.get("decision") or {}).get("parameters", {}), ensure_ascii=False)[:400]
                why = (it.get("decision") or {}).get("reasoning", "")[:260]
                status = (it.get("result") or {}).get("status", "")
                frame = it.get("frame_path") or ""
                crop = it.get("crop_overlay_path") or ""
                imgs = []
                if crop:
                    imgs.append(f'<img src="{os.path.basename(crop)}" alt="crop overlay"/>')
                if frame:
                    imgs.append(f'<img src="{os.path.basename(frame)}" alt="frame"/>')
                img_html = "".join(imgs) if imgs else '<div class="noimg">no image</div>'
                cards.append(
                    f"""
<div class="card">
  <div class="hdr">
    <div class="idx">#{idx}</div>
    <div class="meta">{ts} · <b>{action}</b> · {status}</div>
  </div>
  <div class="imgs">{img_html}</div>
  <div class="txt"><b>params</b> {params}</div>
  <div class="txt"><b>why</b> {why}</div>
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
.hdr {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
.idx {{ font-weight:700; }}
.meta {{ font-size:12px; color:#333; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.imgs {{ display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }}
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
        self.last_ocr = ocr_image(img)
        # truncate OCR to reduce token size (keep bigger words first)
        self.last_ocr = sorted(self.last_ocr, key=lambda w: (-(w["box"][3]-w["box"][1]), -float(w.get("conf", 100))))[:OCR_LIMIT]
        self._log("debug", "state.captured", {"ocr_items": len(self.last_ocr)})

    def _action_wait_any_text(self, patterns, timeout_s: int = 20):
        import re, time
        compiled = [re.compile(pat, re.I) for pat in patterns]
        t0 = time.time()
        while time.time() - t0 <= timeout_s:
            self._capture_state()
            found = None
            for w in self.last_ocr:
                for rx in compiled:
                    if rx.search(w["text"]):
                        found = {"text": w["text"], "box": w["box"]}
                        break
                if found:
                    break
            if found:
                return {"status":"success","match": found}
            time.sleep(0.4)
        return {"status":"failure","error_code":"TIMEOUT_ANY","error_message": f"None matched in {patterns}"}

    def _action_click_any_text(self, patterns, nth: int = 0, prefer_bold: bool = True):
        # try each regex in order until one has a match
        import re
        for pat in patterns:
            w = find_text_box(
                self.last_ocr,
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
                    note=f"pattern_used={pat}",
                )
                x_abs,y_abs = self.grabber.to_abs(cx,cy)
                if not CLICK_ENABLED or not self._xdotool_ok:
                    return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
                _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
                _safe_run(["xdotool","click","1"])
                return {"status":"success","clicked": w["text"], "box": w["box"], "pattern_used": pat}
        return {"status":"failure","error_code":"ELEMENT_NOT_FOUND","error_message": f"No pattern matched: {patterns}"}

    def _action_click_near_text(self, anchor_regex: str, dx: int = 0, dy: int = 0):
        import re
        w = find_text_box(self.last_ocr, regex=anchor_regex)
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
            note=f"anchor={w.get('text','')} offset=[{dx},{dy}]",
        )
        x_abs,y_abs = self.grabber.to_abs(click_x, click_y)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
        _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
        _safe_run(["xdotool","click","1"])
        return {"status":"success","anchor": w["text"], "anchor_box": w["box"], "offset": [dx,dy]}

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
        if not self.last_ocr:
            return {"status": "failure", "error_code": "NO_OCR", "error_message": "No OCR results yet"}
        w = find_text_box(self.last_ocr, regex=regex, nth=nth, prefer_bold=prefer_bold,
                          fuzzy_text=fuzzy_text, fuzzy_threshold=fuzzy_threshold)
        if not w:
            return {"status": "failure", "error_code": "ELEMENT_NOT_FOUND",
                    "error_message": f"Regex '{regex}' not found"}
        x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
        x_abs,y_abs = self.grabber.to_abs(cx,cy)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status": "failure", "error_code": "CLICK_DISABLED_OR_MISSING_XDOTOOL",
                    "error_message": "CLICK_ENABLED=0 or xdotool missing"}
        _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
        _safe_run(["xdotool","click","1"])
        return {"status": "success", "clicked": w["text"], "box": w["box"]}

    def _action_type_text(self, text: str, confidential: bool = False):
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
            if find_text_box(self.last_ocr, regex=regex):
                return {"status": "success", "regex": regex}
            time.sleep(0.4)
        return {"status": "failure", "error_code": "TIMEOUT", "error_message": f"wait_text timeout for '{regex}'"}

    def _action_ocr_extract(self, save_as: str):
        text = " ".join(w["text"] for w in self.last_ocr)
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
            # 1) Observe
            self._capture_state()
            ocr_min = [{"text": w["text"], "box": w["box"], "conf": w["conf"]} for w in self.last_ocr]

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
                max_ocr_elems = int(os.environ.get("PLANNER_MAX_OCR_ELEMENTS", "200"))
                max_ocr_elems = max(0, min(400, max_ocr_elems))
                for w in self.last_ocr[:max_ocr_elems]:
                    ui_elements.append({
                        "source": "ocr",
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
            except Exception as e:
                self._log("warn", "planner.ui_elements.build_failed", {"error": str(e)})

            payload = {
                "goal": AGENT_GOAL,
                "task_history": self.history[-HISTORY_WINDOW:],
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

            # Trace: save the pre-action frame (what the planner saw)
            frame_path = ""
            if self.trace_enabled and self.last_img is not None:
                try:
                    q = max(25, min(95, int(TRACE_FRAME_QUALITY)))
                    frame_path = os.path.join(self.run_dir, f"{self.trace_idx:05d}_{op}_frame.jpg")
                    cv2.imwrite(frame_path, self.last_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                except Exception as e:
                    self._log("warn", "trace.frame_save_failed", {"error": str(e)})

            # 3) Act
            result = {"status":"failure","error_code":"UNKNOWN_ACTION","error_message": op}
            try:
                # Intercept brittle planner steps with a consensus step first
                brittle = self._detect_brittle_intent(op, params)
                if brittle:
                    intent, verify_patterns = brittle
                    self._log("info","consensus.invoke", {"intent": intent, "verify": verify_patterns})
                    res = self._consensus_step(intent=intent, verify_patterns=verify_patterns)
                    if res.get("status") == "success":
                        result = {"status":"success","via":"consensus","picked": res.get("picked")}
                        # After success, optionally insert a short sleep to stabilize UI
                        time.sleep(0.3)
                        # record and continue to next loop without invoking the brittle action itself
                        self.history.append({"action": f"consensus:{intent}", "parameters": {"verify": verify_patterns}, "result": result})
                        self._log("info","action.result", {"action": f"consensus:{intent}", "result": result})
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
                # Debounce after successful clicks
                if op in ("click_text","click_any_text","click_near_text") and result.get("status") == "success":
                    time.sleep(0.25)
            except Exception as e:
                result = {"status":"failure","error_code":"EXCEPTION","error_message": str(e)}

            # 4) Record
            self.history.append({"action": op, "parameters": params, "result": result})
            self._log("info","action.result",{"action": op, "result": result})

            # 5) Trace record + regenerate HTML (last N)
            crop_overlay = (self._last_click_debug or {}).get("overlay_path") if self._last_click_debug else ""
            self._trace_append(
                {
                    "idx": self.trace_idx,
                    "ts": now_utc_iso(),
                    "decision": {"action": op, "parameters": params, "reasoning": reasoning, "completed": completed},
                    "result": result,
                    "frame_path": frame_path,
                    "crop_overlay_path": crop_overlay,
                    "ocr_results_n": len(ocr_min),
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

       