# scripts/agent.py
import os, re, time, json, threading, subprocess, io, uuid, shutil, traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import requests
import yaml
import mss
import cv2
import numpy as np
import pytesseract
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

API_URL = os.environ.get("RTDETR_API_URL", "http://rtdetr-api:8000/predict")
AGENT_NAME = os.environ.get("AGENT_NAME", "agent-1")
TASK_FILE = os.environ.get("AGENT_TASK", f"/tasks/{AGENT_NAME}.yaml")
CLICK_ENABLED = os.environ.get("CLICK_ENABLED", "1") == "1"
SCREEN_INDEX = int(os.environ.get("AGENT_SCREEN_INDEX", "1"))  # mss monitor index
DEBUG_ENABLED = os.environ.get("AGENT_DEBUG", "1") == "1"
DEBUG_DIR = os.environ.get("AGENT_DEBUG_DIR", "/tmp/agent-debug")

# Optional LLM (OpenAI-compatible or local): set LLM_API_URL + LLM_MODEL + LLM_API_KEY
LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

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
# OCR
# -----------------------
def ocr_image(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Return OCR words with boxes: [{'text': str, 'box':[x1,y1,x2,y2], 'conf': float}]"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
        if conf < 50:  # basic quality gate
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"text": txt, "box": [x, y, x + w, y + h], "conf": conf})
    return out

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
            r = session.post(API_URL, files={"file": ("screen.jpg", jpg, "image/jpeg")}, timeout=timeout)
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
        _safe_run(["xdotool", "key", k])

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
# LLM optional
# -----------------------
def run_llm(system: str, prompt: str) -> str:
    if not LLM_API_URL:
        return "[LLM disabled] " + prompt[:300]
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        r = requests.post(LLM_API_URL, headers=headers, json=body, timeout=(5, 60))
        r.raise_for_status()
        j = r.json()
        # OpenAI-compatible
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error: {e}] {prompt[:300]}"

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
            extra={"agent": AGENT_NAME, "task_file": self.task_path, "api_url": API_URL, "click_enabled": CLICK_ENABLED,
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
                    s = sub_env(str(_args), self.ctx) if isinstance(args, str) else sub_env(_args.get("", "") or _args.get("text", ""), self.ctx)
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

    # --- Logging, capture, OCR ---
    def _log(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
        rec = {"ts": now_utc_iso(), "level": level, "msg": msg}
        if extra: rec.update(extra)
        line = safe_json_dumps(rec)
        print(line, flush=True)

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
            allow = {"regex", "nth", "prefer_bold"}
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
        self.last_ocr = sorted(self.last_ocr, key=lambda w: (-(w["box"][3]-w["box"][1]), -w["conf"]))[:OCR_LIMIT]
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
        x_abs,y_abs = self.grabber.to_abs(cx+dx, cy+dy)
        if not CLICK_ENABLED or not self._xdotool_ok:
            return {"status":"failure","error_code":"CLICK_DISABLED_OR_MISSING_XDOTOOL"}
        _safe_run(["xdotool","mousemove","--sync",str(x_abs),str(y_abs)])
        _safe_run(["xdotool","click","1"])
        return {"status":"success","anchor": w["text"], "anchor_box": w["box"], "offset": [dx,dy]}

    def _action_sleep(self, seconds: float = 0.8):
        time.sleep(float(seconds))
        return {"status":"success","slept_seconds": float(seconds)}

    # --- Action impls (call your existing primitives) ---
    def _action_click_text(self, regex: str, nth: int = 0, prefer_bold: bool = False):
        if not self.last_ocr:
            return {"status": "failure", "error_code": "NO_OCR", "error_message": "No OCR results yet"}
        w = find_text_box(self.last_ocr, regex=regex, nth=nth, prefer_bold=prefer_bold)
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
            # 1) Observe
            self._capture_state()
            ocr_min = [{"text": w["text"], "box": w["box"], "conf": w["conf"]} for w in self.last_ocr]

            # 2) Report
            payload = {
                "goal": AGENT_GOAL,
                "task_history": self.history[-HISTORY_WINDOW:],
                "ocr_results": ocr_min,
                "available_actions": [
                    "open_url","wait_text","wait_any_text",
                    "click_text","click_any_text","click_near_text",
                    "type_text","key_seq","sleep","ocr_extract","end_task"
                ]
            }
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

            # 3) Act
            result = {"status":"failure","error_code":"UNKNOWN_ACTION","error_message": op}
            try:
                if op == "done" or op == "end_task":
                    self._log("info", "task.complete", {"reason": params.get("reason","planner requested end")})
                    break
                if op == "open_url":         result = self._action_open_url(**params)
                elif op == "wait_text":      result = self._action_wait_text(**params)
                elif op == "wait_any_text":  result = self._action_wait_any_text(**params)
                elif op == "click_text":     result = self._action_click_text(**params)
                elif op == "click_any_text": result = self._action_click_any_text(**params)
                elif op == "click_near_text":result = self._action_click_near_text(**params)
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

       