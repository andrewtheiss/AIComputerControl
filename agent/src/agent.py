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
HISTORY_WINDOW   = int(os.environ.get("HISTORY_WINDOW", "6"))
OCR_LIMIT        = int(os.environ.get("OCR_LIMIT", "400"))     # cap tokens
PLANNER_SEND_SCREENSHOT_MODE = os.environ.get("PLANNER_SEND_SCREENSHOT_MODE", "").strip().lower() or ("always" if os.environ.get("PLANNER_SEND_SCREENSHOT", "1").strip() != "0" else "never")
PLANNER_SCREENSHOT_JPEG_QUALITY = int(os.environ.get("PLANNER_SCREENSHOT_JPEG_QUALITY", "50"))
PLANNER_SCREENSHOT_MAX_DIM = int(os.environ.get("PLANNER_SCREENSHOT_MAX_DIM", "1280"))
PLANNER_MAX_OCR_LINE_ELEMENTS = max(0, min(200, int(os.environ.get("PLANNER_MAX_OCR_LINE_ELEMENTS", "40"))))
PLANNER_MAX_OCR_WORD_ELEMENTS = max(0, min(300, int(os.environ.get("PLANNER_MAX_OCR_WORD_ELEMENTS", "80"))))
PLANNER_MAX_AX_ELEMENTS = max(0, min(200, int(os.environ.get("PLANNER_MAX_AX_ELEMENTS", "40"))))

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
TARGET_ENSEMBLE_API_URL = os.environ.get("TARGET_ENSEMBLE_API_URL", "").strip()
TARGET_ENSEMBLE_SHADOW_MODE = os.environ.get("TARGET_ENSEMBLE_SHADOW_MODE", "0") == "1"
TARGET_ENSEMBLE_SHADOW_TOP_K = max(1, min(10, int(os.environ.get("TARGET_ENSEMBLE_SHADOW_TOP_K", "5"))))
TARGET_ENSEMBLE_SHADOW_TIMEOUT_S = float(os.environ.get("TARGET_ENSEMBLE_SHADOW_TIMEOUT_S", "8.0"))
TARGET_ENSEMBLE_SHADOW_DEBUG = os.environ.get("TARGET_ENSEMBLE_SHADOW_DEBUG", "1") == "1"

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

    Scale factors (CLICK_SCALE_X / CLICK_SCALE_Y env vars, default 1.0):
      On HiDPI displays mss captures at physical resolution while xdotool uses
      logical coordinates.  Set e.g. CLICK_SCALE_X=0.5 CLICK_SCALE_Y=0.5 for 2×
      HiDPI, or run with CLICK_SCALE_AUTO=1 to detect automatically via xrandr.
    """
    def __init__(self, screen_index: int = 1):
        self.screen_index = screen_index
        self.last_mon: Optional[Dict[str, int]] = None
        self._mss = mss.mss()
        self.scale_x, self.scale_y = self._init_scale()

    def _init_scale(self) -> Tuple[float, float]:
        env_x = os.environ.get("CLICK_SCALE_X", "").strip()
        env_y = os.environ.get("CLICK_SCALE_Y", "").strip()
        auto  = os.environ.get("CLICK_SCALE_AUTO", "0").strip() not in ("0", "", "false", "no")
        if env_x and env_y:
            try:
                sx, sy = float(env_x), float(env_y)
                print(json.dumps({"ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                   "level": "info", "msg": "screen.scale.env",
                                   "scale_x": sx, "scale_y": sy}), flush=True)
                return sx, sy
            except ValueError:
                pass
        if auto:
            detected = self._detect_scale_xrandr()
            if detected:
                sx, sy = detected
                print(json.dumps({"ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                   "level": "info", "msg": "screen.scale.auto",
                                   "scale_x": sx, "scale_y": sy}), flush=True)
                return sx, sy
        return 1.0, 1.0

    def _detect_scale_xrandr(self) -> Optional[Tuple[float, float]]:
        """
        Compare the mss monitor dimensions against xrandr's reported logical resolution.
        Returns (scale_x, scale_y) or None if detection fails.
        """
        try:
            import subprocess as _sp
            out = _sp.check_output(["xrandr", "--current"], text=True, timeout=3)
            # Match lines like "  1920x1080+0+0" (active mode line)
            m = re.search(r'\bconnected\b.*?(\d{3,5})x(\d{3,5})\+\d+\+\d+', out)
            if not m:
                return None
            logical_w, logical_h = int(m.group(1)), int(m.group(2))
            mon = self._mss.monitors[self.screen_index]
            if mon["width"] > 0 and mon["height"] > 0:
                sx = logical_w / mon["width"]
                sy = logical_h / mon["height"]
                if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                    return sx, sy
        except Exception:
            pass
        return None

    def capture(self) -> Tuple[np.ndarray, Dict[str, int]]:
        mon = self._mss.monitors[self.screen_index]
        self.last_mon = mon
        # BGRA -> BGR
        raw = np.array(self._mss.grab(mon))
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return img, mon

    def to_abs(self, x_rel: int, y_rel: int) -> Tuple[int, int]:
        """
        Convert image (monitor-relative) coords -> absolute desktop coords,
        applying any configured scale factors.
        """
        if not self.last_mon:
            raise RuntimeError("Screen geometry not captured yet.")
        return (
            int(self.last_mon["left"] + x_rel * self.scale_x),
            int(self.last_mon["top"]  + y_rel * self.scale_y),
        )

# -----------------------
# OCR (HTTP-first with fallback)
# -----------------------
OCR_API_URL = os.environ.get("OCR_API_URL", "http://ocr-api:8020/ocr").strip()
_ocr_client = OCRClient(url=OCR_API_URL, min_score=float(os.environ.get("OCR_MIN_SCORE", "0.45")))

# Multi-pass OCR on "interaction bands" (top chrome + bottom footer) for better quality
# on small UI text without running full-frame OCR at higher resolution.
# Band sizes: top = top 18% of image, bottom = bottom 40%.
OCR_BAND_MULTIPASS = os.environ.get("OCR_BAND_MULTIPASS", "1").strip() == "1"
OCR_BAND_TOP_FRAC   = float(os.environ.get("OCR_BAND_TOP_FRAC",    "0.18"))
OCR_BAND_BOTTOM_FRAC = float(os.environ.get("OCR_BAND_BOTTOM_FRAC", "0.40"))


def _ocr_bands_merge(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    OCR tweak: run OCR on the full image plus the top-chrome and bottom-footer bands
    separately, then merge results (deduplicating by box IOU > 0.5).

    This improves detection of small UI text (address bar, dialog buttons) without
    increasing global DET side length.  Only runs if OCR_BAND_MULTIPASS=1.
    """
    full = _ocr_client.ocr_levels(image_bgr)
    if not OCR_BAND_MULTIPASS:
        return full

    h, w = image_bgr.shape[:2]
    top_h    = max(1, int(h * OCR_BAND_TOP_FRAC))
    bot_y    = max(0, h - int(h * OCR_BAND_BOTTOM_FRAC))

    extra_words: List[Dict[str, Any]] = []
    extra_lines: List[Dict[str, Any]] = []

    def _merge_into(src_items: List[Dict[str, Any]], extra_list: List[Dict[str, Any]],
                    existing: List[Dict[str, Any]], y_offset: int) -> None:
        for item in src_items:
            b = item.get("box")
            if not isinstance(b, list) or len(b) != 4:
                continue
            # Shift box back to full-image coordinates
            adjusted = dict(item, box=[b[0], b[1] + y_offset, b[2], b[3] + y_offset])
            # Skip if highly overlapping with an existing result
            duplicate = False
            for ex in existing:
                ex_b = ex.get("box")
                if not isinstance(ex_b, list) or len(ex_b) != 4:
                    continue
                ax1, ay1, ax2, ay2 = [int(v) for v in adjusted["box"]]
                bx1, by1, bx2, by2 = [int(v) for v in ex_b]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter > 0:
                    area_a = max(1, (ax2-ax1)*(ay2-ay1))
                    iou = inter / area_a
                    if iou > 0.5:
                        duplicate = True
                        break
            if not duplicate:
                extra_list.append(adjusted)

    # Top band
    if top_h > 10:
        top_crop = image_bgr[:top_h, :, :]
        try:
            top_ocr = _ocr_client.ocr_levels(top_crop)
            _merge_into(top_ocr.get("words", []), extra_words, full.get("words", []), 0)
            _merge_into(top_ocr.get("lines", []), extra_lines, full.get("lines", []), 0)
        except Exception:
            pass

    # Bottom band
    if bot_y < h - 10:
        bot_crop = image_bgr[bot_y:, :, :]
        try:
            bot_ocr = _ocr_client.ocr_levels(bot_crop)
            _merge_into(bot_ocr.get("words", []), extra_words, full.get("words", []) + extra_words, bot_y)
            _merge_into(bot_ocr.get("lines", []), extra_lines, full.get("lines", []) + extra_lines, bot_y)
        except Exception:
            pass

    merged_words = full.get("words", []) + extra_words
    merged_lines = full.get("lines", []) + extra_lines
    return {"words": merged_words, "lines": merged_lines, "source": full.get("source", "http")}


def ocr_image(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Return OCR words with boxes using OCRClient (HTTP preferred, Tesseract fallback)."""
    return _ocr_client.ocr(image_bgr)

def ocr_image_levels(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Return OCR word + line boxes (with multi-pass band merging if enabled)."""
    return _ocr_bands_merge(image_bgr)

def _score_boldish_height(box: List[int]) -> int:
    return box[3] - box[1]


# Fix L — provenance origins (ordered from most to least trustworthy)
# raw_word   : native OCR word-level box
# line_item  : native OCR line-level box (whole-line match)
# line_subbox: sub-box derived from line via character-proportional span (Fix F)
# synth_word : synthetic token box from proportional line-split (Fix G)
# a11y       : Accessibility-tree node coordinate
# vlm        : VLM-derived coordinate
BOX_ORIGINS_SYNTHETIC = frozenset({"synth_word", "line_subbox"})
BOX_ORIGINS_TRUSTED   = frozenset({"raw_word", "line_item", "a11y", "vlm"})


def _subbox_for_match(text: str, box: List[int], match_span: Tuple[int, int], pad: int = 8) -> List[int]:
    """
    Fix F: Given a full-line OCR box and the (start, end) character span of the
    matched substring, return a tighter x-range for just that portion of the text.

    Uses proportional character-width estimation.  The y-range is kept as-is.
    Padding of `pad` px is added on both sides and then clamped to the line box.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    line_w = max(1, x2 - x1)
    n = max(1, len(text))
    start, end = match_span
    start = max(0, min(start, n))
    end   = max(start, min(end, n))
    # proportional x-slice
    sx1 = x1 + int(line_w * start / n) - pad
    sx2 = x1 + int(line_w * end   / n) + pad
    # clamp to parent box
    sx1 = max(x1, sx1)
    sx2 = min(x2, max(sx1 + 8, sx2))
    return [sx1, y1, sx2, y2]


def _synthesize_word_boxes(line_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fix G: Synthesize per-token boxes from a line OCR item whose text spans the
    whole line box.  Tokens are split on whitespace; widths are allocated
    proportional to character length (spaces count as 0.6 chars each).

    Returns a list of word-level dicts with the same keys as normal OCR words.
    Returns an empty list if the item already looks word-level or has no text.
    """
    text = str(line_item.get("text", "") or "").strip()
    if not text:
        return []
    box = line_item.get("box", [])
    if not isinstance(box, list) or len(box) != 4:
        return []

    # Skip if it's already a word-level item (rough heuristic: no space in text)
    level = str(line_item.get("level", "line") or "line")
    tokens = text.split(" ")
    if len(tokens) <= 1 and level != "line":
        return []

    x1, y1, x2, y2 = [int(v) for v in box]
    line_w = max(1, x2 - x1)
    conf = int(line_item.get("conf", 50) or 50)

    # Compute unit widths: each character = 1 unit, each space gap = 0.6
    token_units = [max(1, len(t)) for t in tokens]
    gap_units = [0.6] * max(0, len(tokens) - 1)
    total_units = sum(token_units) + sum(gap_units)
    if total_units <= 0:
        return []

    words = []
    cursor = 0.0
    for i, (tok, units) in enumerate(zip(tokens, token_units)):
        if not tok:
            cursor += units + (gap_units[i] if i < len(gap_units) else 0)
            continue
        wx1 = x1 + int(line_w * cursor / total_units)
        wx2 = x1 + int(line_w * (cursor + units) / total_units)
        wx2 = max(wx1 + 4, wx2)
        wx2 = min(x2, wx2)
        words.append({
            "text": tok,
            "box": [wx1, y1, wx2, y2],
            "conf": conf,
            "level": "word_synth",
            "_origin": "synth_word",          # Fix L: provenance
            "_parent_box": [x1, y1, x2, y2],  # Fix L: parent line box for refinement
        })
        cursor += units + (gap_units[i] if i < len(gap_units) else 0)
    return words


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
    apply_subbox: bool = True,
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
        matched_span: Optional[Tuple[int, int]] = None

        if patterns:
            for p in patterns:
                m = p.search(s)
                if m:
                    match_score = 100.0
                    matched_span = (m.start(), m.end())
                    break
        elif fuzzy_text:
            score = fuzz.partial_ratio(fuzzy_text, s)
            if fuzzy_threshold is None or score >= float(fuzzy_threshold):
                match_score = float(score)
                # Approximate span for fuzzy: use full string
                matched_span = (0, len(s))

        if match_score is None:
            continue

        matched = dict(item)
        matched.setdefault("level", default_level)

        # Fix F: if the match is a substring of a line-level item, tighten the box.
        box = matched.get("box")
        level = str(matched.get("level", default_level) or "")
        if (
            apply_subbox
            and matched_span is not None
            and isinstance(box, list) and len(box) == 4
            and level in ("line", "line_synth", "line_ocr")
            and matched_span != (0, len(s))          # skip if whole-string match
            and len(s) > 0
        ):
            sub = _subbox_for_match(s, box, matched_span, pad=8)
            matched = dict(matched, box=sub, _subbox_applied=True, _original_box=box)
            # Fix L: provenance — this is a derived sub-box, not a raw OCR box
            matched["_origin"] = "line_subbox"
            matched["_parent_box"] = list(box)
        else:
            # Fix L: stamp origin based on level
            existing_origin = matched.get("_origin", "")
            if not existing_origin:
                lvl = str(matched.get("level", default_level) or "")
                if lvl == "word_synth":
                    matched["_origin"] = "synth_word"
                elif lvl in ("line", "line_synth", "line_ocr"):
                    matched["_origin"] = "line_item"
                else:
                    matched["_origin"] = "raw_word"

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

    Search order:
    - Multi-word / multi-token targets: line OCR first (with Fix F subbox refinement),
      then synthesized word boxes from lines (Fix G), then raw word OCR.
    - Single-word targets: word OCR first, then line OCR (with Fix F).

    Fixes applied:
    - Fix F: when a regex matches a substring of a line box, the returned box is
      tightened to the matched character span (proportional width estimate).
    - Fix G: line items are expanded into per-token boxes so that single-word
      searches against line OCR return tight token-level boxes.
    """
    prefer_lines = _text_target_prefers_lines(regex, any_regex or [], fuzzy_text)

    # Fix G: synthesise word-level boxes from each line item (cheap, proportional split)
    synth_words: List[Dict[str, Any]] = []
    for li in (line_items or []):
        synth_words.extend(_synthesize_word_boxes(li))

    search_groups: List[Tuple[str, List[Dict[str, Any]], bool]] = []
    # (level_label, item_list, apply_subbox)
    if prefer_lines and line_items:
        search_groups.append(("line", line_items, True))         # Fix F active
    if synth_words:
        search_groups.append(("word_synth", synth_words, False)) # already word-level
    if ocr_items:
        search_groups.append(("word", ocr_items, False))         # native word boxes
    if not prefer_lines and line_items:
        search_groups.append(("line", line_items, True))         # Fix F active

    for level, items, do_subbox in search_groups:
        match = _find_text_box_in_items(
            items,
            default_level=level,
            regex=regex,
            any_regex=any_regex,
            fuzzy_text=fuzzy_text,
            fuzzy_threshold=fuzzy_threshold,
            prefer_bold=prefer_bold,
            nth=nth,
            apply_subbox=do_subbox,
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

def encode_jpeg_bgr_resized(image_bgr: np.ndarray, q: int = 85, max_dim: int = 1280) -> bytes:
    img = image_bgr
    if max_dim and max_dim > 0:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = float(max_dim) / float(longest)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return encode_jpeg_bgr(img, q=q)

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
        self.planner_session_id = f"{AGENT_NAME}:{hashlib.sha1((AGENT_GOAL or self.run_id).encode('utf-8')).hexdigest()[:12]}"
        # Hard cap: tracks how many transparent keyboard escalations the anti-repeat guard
        # has fired per blocker signature. Resets naturally when the blocker signature changes.
        self._guard_kb_escalations: Dict[str, int] = {}
        # Max number of silent keyboard escalations we allow before surfacing to the planner.
        self._GUARD_KB_ESCALATION_CAP = 2
        # Last scored resolve candidates (set by _preferred_resolve_target) for trace visualization.
        self._last_resolve_candidates: List[Dict[str, Any]] = []

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

    def _vlm_locate_box(self, label: str, context: str = "") -> Optional[List[int]]:
        """
        Fix J: Ask the VLM (run_llm_vision) to locate a UI element by label in the
        current screenshot.  Returns [x1, y1, x2, y2] in image-space, or None.

        Only called when geometry is ambiguous (wide box, low confidence, or
        repeated blocker failures).  Results are cached per (state_hash, label)
        to avoid redundant calls.
        """
        if not LLM_API_URL or self.last_img is None:
            return None
        try:
            h, w = self.last_img.shape[:2]
            ctx_hint = f" Context: {context}." if context else ""
            system = (
                "You are a precise UI element locator. "
                "Respond ONLY with a JSON object: {\"box\": [x1, y1, x2, y2]}. "
                "Coordinates are in pixels (origin top-left). "
                "Do not include any other text."
            )
            prompt = (
                f"Image size: {w}x{h} px.{ctx_hint} "
                f"Find the bounding box of the clickable UI element labeled: \"{label}\". "
                "Return the tightest box around just that element, not the whole dialog."
            )
            resp = run_llm_vision(self.last_img, system, prompt)
            # Extract JSON from response (may have surrounding prose)
            m = re.search(r'\{[^}]*"box"\s*:\s*\[([^\]]+)\][^}]*\}', resp, re.I | re.S)
            if not m:
                # try bare array
                m = re.search(r'\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]', resp)
            if m:
                nums = [int(x.strip()) for x in m.group(1).split(",")]
                if len(nums) == 4:
                    x1, y1, x2, y2 = nums
                    # Clamp to image bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 4, min(x2, w))
                    y2 = max(y1 + 4, min(y2, h))
                    self._log("info", "vlm.locate_box", {
                        "label": label, "box": [x1, y1, x2, y2],
                        "raw_resp": resp[:200],
                    })
                    return [x1, y1, x2, y2]
            self._log("warn", "vlm.locate_box.parse_failed", {
                "label": label, "raw_resp": resp[:200]
            })
        except Exception as e:
            self._log("warn", "vlm.locate_box.failed", {"label": label, "error": str(e)})
        return None

    def _box_is_ambiguous(self, box: Optional[List[int]], text: str = "") -> bool:
        """
        Return True when a box is geometrically ambiguous (too wide for its text
        length, or extreme aspect ratio) and the VLM should be consulted as fallback.
        """
        if not isinstance(box, list) or len(box) != 4:
            return True
        bw = max(1, int(box[2]) - int(box[0]))
        bh = max(1, int(box[3]) - int(box[1]))
        n_chars = max(1, len(str(text or "").strip()))
        w_per_char = bw / n_chars
        aspect = bw / max(1, bh)
        return w_per_char > 18 or aspect > 20

    @staticmethod
    def _draw_labeled_box(
        img: np.ndarray,
        box: List[int],
        color: Tuple[int, int, int],
        thickness: int,
        label: str,
        font_scale: float = 0.48,
        font_thickness: int = 1,
    ) -> None:
        """Draw a rectangle with a label above it, clamped to image bounds."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            return
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            tx = max(0, x1c)
            ty = max(th + 2, y1c - 4)
            # Dark backing for readability
            cv2.rectangle(img, (tx - 1, ty - th - 2), (tx + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, font_thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_crosshair(img: np.ndarray, cx: int, cy: int) -> None:
        cv2.circle(img, (cx, cy), 9, (0, 0, 255), -1)
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 32, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 1, cv2.LINE_AA)

    @staticmethod
    def _legend_block(
        img: np.ndarray,
        entries: List[Tuple[Tuple[int,int,int], str]],
    ) -> None:
        """Draw a color-coded legend in the top-right corner."""
        h, w = img.shape[:2]
        row_h, pad, swatch = 18, 6, 14
        lw = max(160, max(len(lbl) for _, lbl in entries) * 7 + swatch + pad * 2 + 6)
        lh = row_h * len(entries) + pad * 2
        ox = w - lw - 4
        oy = 4
        # Semi-transparent black background
        sub = img[oy : oy + lh, ox : ox + lw]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, black, 0.65, 0, sub)
        img[oy : oy + lh, ox : ox + lw] = sub
        for i, (color, lbl) in enumerate(entries):
            ry = oy + pad + i * row_h
            rx = ox + pad
            cv2.rectangle(img, (rx, ry), (rx + swatch, ry + swatch - 2), color, -1)
            cv2.putText(img, lbl, (rx + swatch + 4, ry + swatch - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _crop_level(
        img: np.ndarray,
        box: List[int],
        label: str,
        box_color: Tuple[int,int,int],
        click_pt: Optional[Tuple[int,int]] = None,
        pad_factor: float = 0.4,
        scale_up: int = 2,
    ) -> np.ndarray:
        """
        Return a cropped view around `box` with the box drawn and optional click point.
        Adds a header bar with `label`.
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_x = max(20, int(bw * pad_factor))
        pad_y = max(20, int(bh * pad_factor))
        left   = max(0, x1 - pad_x)
        top    = max(0, y1 - pad_y)
        right  = min(w, x2 + pad_x)
        bottom = min(h, y2 + pad_y)
        crop = img[top:bottom, left:right].copy()
        # Scale up small crops for readability
        ch, cw = crop.shape[:2]
        if scale_up > 1 and max(cw, ch) < 400:
            crop = cv2.resize(crop, None, fx=scale_up, fy=scale_up,
                              interpolation=cv2.INTER_CUBIC)
        s = scale_up if (max(cw, ch) < 400) else 1
        # Draw box in crop-local coords
        lx1 = max(0, (x1 - left) * s)
        ly1 = max(0, (y1 - top) * s)
        lx2 = min(crop.shape[1] - 1, (x2 - left) * s)
        ly2 = min(crop.shape[0] - 1, (y2 - top) * s)
        cv2.rectangle(crop, (lx1, ly1), (lx2, ly2), box_color, max(1, s + 1))
        if click_pt:
            lcx = (click_pt[0] - left) * s
            lcy = (click_pt[1] - top) * s
            if 0 <= lcx < crop.shape[1] and 0 <= lcy < crop.shape[0]:
                cv2.circle(crop, (lcx, lcy), max(4, s * 4), (0, 0, 255), -1)
                cv2.drawMarker(crop, (lcx, lcy), (0, 0, 255), cv2.MARKER_CROSS,
                               max(12, s * 12), max(1, s), cv2.LINE_AA)
        # Header bar
        bar_h = 20
        bar = np.zeros((bar_h, crop.shape[1], 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        box_coords = f"[{x1},{y1},{x2},{y2}]"
        cv2.putText(bar, f"{label}  {box_coords}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
        return np.vstack([bar, crop])

    def _save_click_visual_debug(
        self,
        *,
        ocr_box: List[int],
        final_box: List[int],
        click_rel: Tuple[int, int],
        action: str,
        subbox_applied: bool = False,
        parent_box: Optional[List[int]] = None,
        origin: str = "",
        candidates_debug: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        Save annotated full-frame images for each click action.

        Images produced:
        1. ``*_hierarchy.jpg``    – comprehensive single screenshot with ALL box levels
                                    drawn (screen border → parent → OCR match → final target)
                                    plus click point and legend. This is the primary debug view.
        2. ``*_ocr_box.jpg``      – raw OCR-matched box (cyan) + all scored candidates.
        3. ``*_click_target.jpg`` – final click box (green/magenta) + click crosshair.
        4. ``*_level_N.jpg``      – one zoomed crop per box level in the hierarchy.

        Returns dict with all path keys (empty dict on error).
        """
        if self.last_img is None or not (self.debug_enabled and self.trace_enabled):
            return {}
        try:
            ts_ms = int(time.time() * 1000)
            paths: Dict[str, str] = {}
            img_h, img_w = self.last_img.shape[:2]
            cx, cy = int(click_rel[0]), int(click_rel[1])

            ox1, oy1, ox2, oy2 = [int(v) for v in ocr_box]
            fx1, fy1, fx2, fy2 = [int(v) for v in final_box]

            # ── Build the box hierarchy (from widest to narrowest) ───────────
            # Each entry: (box, color_BGR, label, thickness)
            is_vlm = action.endswith("_vlm")
            final_color = (255, 0, 255) if is_vlm else (0, 230, 60)

            # Determine distinct hierarchy levels
            hierarchy: List[Tuple[List[int], Tuple[int,int,int], str, int]] = []

            # Level 0: full screen boundary
            screen_box = [0, 0, img_w - 1, img_h - 1]
            hierarchy.append((screen_box, (180, 180, 180), "screen", 1))

            # Level 1: parent line box (if present and different from ocr_box)
            if (parent_box and isinstance(parent_box, list) and len(parent_box) == 4
                    and parent_box != ocr_box):
                hierarchy.append(([int(v) for v in parent_box], (0, 140, 255),
                                   "parent line", 2))

            # Level 2: OCR matched box (if different from final)
            if [ox1, oy1, ox2, oy2] != [fx1, fy1, fx2, fy2]:
                ocr_lbl = "OCR subbox" if subbox_applied else "OCR match"
                hierarchy.append(([ox1, oy1, ox2, oy2], (0, 220, 255), ocr_lbl, 2))

            # Level 3: final click target
            final_lbl = f"VLM({origin})" if is_vlm else ("subbox" if subbox_applied else f"target({origin})")
            hierarchy.append(([fx1, fy1, fx2, fy2], final_color, final_lbl, 3))

            # ── Image 1: comprehensive hierarchy view ─────────────────────────
            hier_img = self.last_img.copy()

            # Background candidates in dark yellow (thinnest, drawn first)
            for ci, cand in enumerate(candidates_debug or []):
                cb = cand.get("box")
                if isinstance(cb, list) and len(cb) == 4:
                    cbx1, cby1, cbx2, cby2 = [int(v) for v in cb]
                    cv2.rectangle(hier_img, (cbx1, cby1), (cbx2, cby2), (0, 170, 220), 1)
                    stxt = (f"c{ci+1}:{cand['_score']:.0f}" if isinstance(cand.get("_score"), float)
                            else f"c{ci+1}")
                    cv2.putText(hier_img, stxt, (cbx1, max(9, cby1 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 170, 220), 1, cv2.LINE_AA)

            # Draw hierarchy from widest → narrowest (so small boxes render on top)
            for (hbox, hcol, hlbl, hthick) in hierarchy:
                self._draw_labeled_box(hier_img, hbox, hcol, hthick, hlbl,
                                       font_scale=0.45 if hthick <= 1 else 0.50)

            # Click crosshair on top
            self._draw_crosshair(hier_img, cx, cy)
            cv2.putText(hier_img, f"click({cx},{cy})",
                        (max(0, cx + 12), max(14, cy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Legend
            legend_entries: List[Tuple[Tuple[int,int,int], str]] = [
                ((180, 180, 180), "screen boundary"),
            ]
            if len(hierarchy) > 2:
                legend_entries.append(((0, 140, 255), "parent line box"))
            legend_entries.append(((0, 220, 255), "OCR match box"))
            legend_entries.append((final_color, "final target"))
            legend_entries.append(((0, 170, 220), "candidates"))
            legend_entries.append(((0, 0, 255), "click point"))
            self._legend_block(hier_img, legend_entries)

            p_hier = os.path.join(self.run_dir, f"{ts_ms}_{action}_hierarchy.jpg")
            cv2.imwrite(p_hier, hier_img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            paths["hierarchy_path"] = p_hier

            # ── Image 2: OCR match + scored candidates ────────────────────────
            img1 = self.last_img.copy()
            for ci, cand in enumerate(candidates_debug or []):
                cb = cand.get("box")
                if isinstance(cb, list) and len(cb) == 4:
                    cbx1, cby1, cbx2, cby2 = [int(v) for v in cb]
                    cv2.rectangle(img1, (cbx1, cby1), (cbx2, cby2), (0, 200, 255), 1)
                    stxt = (f"#{ci+1} s={cand['_score']:.1f}" if isinstance(cand.get("_score"), float)
                            else f"#{ci+1}")
                    cv2.putText(img1, stxt, (cbx1, max(10, cby1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
            ocr_lbl = "OCR subbox" if subbox_applied else "OCR match"
            self._draw_labeled_box(img1, [ox1, oy1, ox2, oy2], (0, 220, 255), 3, ocr_lbl)
            p1 = os.path.join(self.run_dir, f"{ts_ms}_{action}_ocr_box.jpg")
            cv2.imwrite(p1, img1, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            paths["ocr_box_path"] = p1

            # ── Image 3: final click target + click point ─────────────────────
            img2 = self.last_img.copy()
            for cand in (candidates_debug or []):
                cb = cand.get("box")
                if isinstance(cb, list) and len(cb) == 4:
                    cbx1, cby1, cbx2, cby2 = [int(v) for v in cb]
                    cv2.rectangle(img2, (cbx1, cby1), (cbx2, cby2), (0, 200, 255), 1)
            if [ox1, oy1, ox2, oy2] != [fx1, fy1, fx2, fy2]:
                cv2.rectangle(img2, (ox1, oy1), (ox2, oy2), (0, 220, 255), 2)
                if subbox_applied:
                    cv2.putText(img2, "original", (ox1, max(14, oy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            box_lbl = "VLM target" if is_vlm else ("subbox" if subbox_applied else "target")
            self._draw_labeled_box(img2, [fx1, fy1, fx2, fy2], final_color, 3, box_lbl)
            self._draw_crosshair(img2, cx, cy)
            cv2.putText(img2, f"({cx},{cy})", (max(0, cx + 12), max(14, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            p2 = os.path.join(self.run_dir, f"{ts_ms}_{action}_click_target.jpg")
            cv2.imwrite(p2, img2, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            paths["click_target_path"] = p2

            # ── Images 4+: one zoomed crop per meaningful hierarchy level ─────
            level_paths: List[str] = []
            click_pt_tuple = (cx, cy)
            zoom_levels = [
                # (box, color, label, pad_factor, draw_click)
                (screen_box,              (180, 180, 180), "L0 screen",      0.0,  False),
            ]
            if parent_box and isinstance(parent_box, list) and len(parent_box) == 4:
                zoom_levels.append(([int(v) for v in parent_box], (0, 140, 255),
                                    "L1 parent line", 0.35, False))
            if [ox1, oy1, ox2, oy2] != [fx1, fy1, fx2, fy2]:
                zoom_levels.append(([ox1, oy1, ox2, oy2], (0, 220, 255),
                                    "L2 OCR box", 0.40, False))
            zoom_levels.append(([fx1, fy1, fx2, fy2], final_color,
                                 "L3 target", 0.60, True))

            for li, (zbox, zcol, zlbl, zpad, zdraw_click) in enumerate(zoom_levels):
                # For the full screen level, just use a downscaled version of hierarchy
                if li == 0:
                    thumb = cv2.resize(hier_img, (min(img_w, 640),
                                                   min(img_h, int(img_h * 640 / img_w))),
                                       interpolation=cv2.INTER_AREA)
                    bar = np.zeros((20, thumb.shape[1], 3), dtype=np.uint8)
                    bar[:] = (40, 40, 40)
                    cv2.putText(bar, f"L0 screen  [0,0,{img_w},{img_h}]",
                                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (180, 180, 180), 1, cv2.LINE_AA)
                    lvl_img = np.vstack([bar, thumb])
                else:
                    lvl_img = self._crop_level(
                        self.last_img,
                        zbox, zlbl, zcol,
                        click_pt=click_pt_tuple if zdraw_click else None,
                        pad_factor=zpad,
                    )
                lp = os.path.join(self.run_dir,
                                  f"{ts_ms}_{action}_level{li}.jpg")
                cv2.imwrite(lp, lvl_img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                level_paths.append(lp)

            paths["level_paths"] = level_paths  # type: ignore[assignment]
            return paths
        except Exception as e:
            self._log("warn", "click.visual_debug.failed", {"error": str(e), "action": action})
            return {}

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
                box_origin = result.get("box_origin") or ""
                frame = it.get("frame_path") or ""
                post_frame = it.get("post_frame_path") or ""
                crop = it.get("crop_overlay_path") or ""
                hierarchy_ov   = it.get("hierarchy_overlay_path") or ""
                ocr_box_ov     = it.get("ocr_box_overlay_path") or ""
                click_tgt_ov   = it.get("click_target_overlay_path") or ""
                level_paths    = it.get("level_overlay_paths") or []
                shadow_targeting = it.get("target_ensemble_shadow") or {}

                subbox_applied = bool(result.get("subbox_applied", False))
                vlm_used       = bool(result.get("vlm_used", False))
                synth_failed   = bool(result.get("synthetic_refine_failed", False))
                subbox_lbl = ' <span class="tag tag-subbox">subbox</span>' if subbox_applied else ""
                vlm_lbl    = ' <span class="tag tag-vlm">VLM</span>'       if vlm_used       else ""
                synth_lbl  = ' <span class="tag tag-synth">synth!</span>'  if synth_failed   else ""
                origin_lbl = f' <span class="tag tag-origin">{box_origin}</span>' if box_origin else ""

                ocr_box_val  = result.get("ocr_box") or ""
                click_abs_val = result.get("click_abs") or result.get("clicked_abs") or ""
                shadow_final = (shadow_targeting.get("final_prediction") or {}) if isinstance(shadow_targeting, dict) else {}
                shadow_desc = ""
                if shadow_final:
                    shadow_desc = f"{shadow_final.get('candidate_id', '')} score={shadow_final.get('score', '')}"
                elif shadow_targeting.get("error"):
                    shadow_desc = f"error={shadow_targeting.get('error')}"

                is_click_action = executed_action in (
                    "click_text", "click_any_text", "click_near_text",
                    "click_box", "click_text_vlm", "click_box_biased",
                )

                # ── Image sections ────────────────────────────────────────────
                # Section A: hierarchy + drill-down levels (click actions only)
                hier_html = ""
                if is_click_action and hierarchy_ov:
                    # Wide hierarchy composite (spans full card width)
                    hier_html += (
                        f'<div class="hier-row">'
                        f'<figure class="hier-fig">'
                        f'<img src="{os.path.basename(hierarchy_ov)}" alt="hierarchy"/>'
                        f'<figcaption>&#9660; box hierarchy (all levels)</figcaption>'
                        f'</figure></div>'
                    )
                    # Drill-down level crops in a separate row
                    if level_paths:
                        level_figs = "".join(
                            f'<figure><img src="{os.path.basename(lp)}" alt="L{li}"/>'
                            f'<figcaption>{'L0 screen' if li==0 else f'L{li} zoom'}</figcaption></figure>'
                            for li, lp in enumerate(level_paths) if lp
                        )
                        n_lvls = len([lp for lp in level_paths if lp])
                        hier_html += (
                            f'<div class="level-row" style="grid-template-columns:repeat({n_lvls},1fr)">'
                            f'{level_figs}</div>'
                        )

                # Section B: standard before/after frames + OCR/target overlays
                std_imgs = []
                if ocr_box_ov:
                    std_imgs.append(
                        f'<figure><img src="{os.path.basename(ocr_box_ov)}" alt="ocr box"/>'
                        f'<figcaption>OCR candidates</figcaption></figure>'
                    )
                if click_tgt_ov:
                    std_imgs.append(
                        f'<figure><img src="{os.path.basename(click_tgt_ov)}" alt="click target"/>'
                        f'<figcaption>click target</figcaption></figure>'
                    )
                if crop and not (ocr_box_ov or click_tgt_ov):
                    std_imgs.append(
                        f'<figure><img src="{os.path.basename(crop)}" alt="crop"/>'
                        f'<figcaption>target (crop)</figcaption></figure>'
                    )
                if frame:
                    std_imgs.append(
                        f'<figure><img src="{os.path.basename(frame)}" alt="before"/>'
                        f'<figcaption>before</figcaption></figure>'
                    )
                if post_frame:
                    std_imgs.append(
                        f'<figure><img src="{os.path.basename(post_frame)}" alt="after"/>'
                        f'<figcaption>after</figcaption></figure>'
                    )
                std_cols = max(2, min(4, len(std_imgs)))
                std_html = ""
                if std_imgs:
                    std_html = (
                        f'<div class="imgs" style="grid-template-columns:repeat({std_cols},1fr)">'
                        + "".join(std_imgs) + "</div>"
                    )
                elif not hier_html:
                    std_html = '<div class="noimg">no image</div>'

                card_cls = "card"
                if status == "success" and outcome_verified:
                    card_cls += " ok"
                elif status == "dispatched" or (event_applied and not outcome_verified):
                    card_cls += " warn"
                else:
                    card_cls += " bad"
                if is_click_action:
                    card_cls += " click-card"

                cards.append(
                    f"""
<div class="{card_cls}">
  <div class="hdr">
    <div class="idx">#{idx}</div>
    <div class="meta">{ts} · <b>{executed_action}</b> · {status}{subbox_lbl}{vlm_lbl}{synth_lbl}{origin_lbl}</div>
  </div>
  {hier_html}
  {std_html}
  <div class="txt"><b>planner</b> {planner_action} {planner_params}</div>
  <div class="txt"><b>executed</b> {executed_action} {executed_params}</div>
  <div class="txt"><b>why</b> {why}</div>
  <div class="txt"><b>verification</b> applied={event_applied} verified={outcome_verified} reason={verification_reason}</div>
  <div class="txt"><b>blocker</b> {blocker_class}</div>
  <div class="txt"><b>recovery</b> strategy={recovery_strategy} effect={recovery_effect}</div>
  <div class="txt"><b>targeting</b> {targeting_source}{subbox_lbl}{vlm_lbl}{synth_lbl}</div>
  <div class="txt"><b>shadow ensemble</b> {shadow_desc}</div>
  <div class="txt"><b>ocr box</b> {ocr_box_val}  <b>click abs</b> {click_abs_val}  <b>origin</b> {box_origin}</div>
  <div class="txt"><b>override</b> {executor_override_reason}</div>
  <div class="txt"><b>evidence</b> {verification_evidence}</div>
  <div class="txt"><b>before tags</b> {before_state.get("tags", [])}</div>
  <div class="txt"><b>after tags</b> {after_state.get("tags", [])}</div>
  <div class="txt"><b>focus</b> {focus_before} → {focus_after}</div>
</div>
"""
                )
            html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Agent trace</title>
<style>
body {{ font-family: system-ui, Arial, sans-serif; margin: 16px; background:#f0f2f5; }}
.grid {{ display:grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
@media (min-width:1600px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
/* Click cards are wider — they span 2 columns on the 3-col layout */
@media (min-width:1600px) {{ .click-card {{ grid-column: span 2; }} }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:10px; background:#fff;
         box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.card.ok   {{ border-color:#9ad0a0; background:#f5fff6; }}
.card.warn {{ border-color:#e9c46a; background:#fffaf0; }}
.card.bad  {{ border-color:#ef9a9a; background:#fff5f5; }}
.hdr {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; margin-bottom:6px; }}
.idx {{ font-weight:700; font-size:16px; }}
.meta {{ font-size:12px; color:#333; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
/* Hierarchy composite — full card width */
.hier-row {{ margin:4px 0 6px; }}
.hier-fig {{ margin:0; width:100%; }}
.hier-fig img {{ width:100%; height:auto; border-radius:6px; border:2px solid #aaa; cursor:zoom-in; }}
.hier-fig figcaption {{ font-size:11px; font-weight:600; color:#555; text-align:center;
                         margin-top:3px; letter-spacing:.04em; }}
/* Drill-down level crops */
.level-row {{ display:grid; gap:4px; margin:4px 0 6px; }}
.level-row figure {{ margin:0; }}
.level-row figcaption {{ font-size:10px; color:#888; text-align:center; margin-top:2px; }}
.level-row img {{ width:100%; height:auto; border-radius:4px; border:1px solid #ccc; cursor:zoom-in; }}
/* Standard images grid */
.imgs {{ display:grid; gap:6px; margin:4px 0 6px; }}
figure {{ margin:0; }}
figcaption {{ font-size:11px; color:#666; margin-top:3px; text-align:center; }}
img {{ width:100%; height:auto; border-radius:6px; background:#f6f6f6; cursor:zoom-in; }}
img:hover {{ outline:2px solid #4a90d9; }}
.txt {{ font-size:12px; color:#222; margin-top:5px; word-break:break-word; }}
.noimg {{ font-size:12px; color:#777; padding:18px; text-align:center; border:1px dashed #ccc; border-radius:8px; }}
.tag {{ display:inline-block; font-size:10px; font-weight:700; padding:1px 5px; border-radius:3px; margin-left:4px; vertical-align:middle; }}
.tag-subbox  {{ background:#e0f4ff; color:#0077bb; border:1px solid #99d0f0; }}
.tag-vlm     {{ background:#f4e0ff; color:#8800cc; border:1px solid #d0a0f0; }}
.tag-synth   {{ background:#fff0d0; color:#aa5500; border:1px solid #f0c060; }}
.tag-origin  {{ background:#e8ffe8; color:#226622; border:1px solid #88cc88; font-size:9px; }}
/* lightbox */
#lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.88); z-index:9999;
       align-items:center; justify-content:center; cursor:zoom-out; flex-direction:column; gap:8px; }}
#lb.open {{ display:flex; }}
#lb img {{ max-width:96vw; max-height:90vh; border-radius:8px; box-shadow:0 4px 32px rgba(0,0,0,.7); }}
#lb-caption {{ color:#eee; font-size:13px; font-family:monospace; }}
</style></head>
<body>
<h2 style="margin-bottom:12px">Last {len(items)} actions
  <span style="font-size:13px;font-weight:400;color:#888">(auto-updated · click any image to zoom)</span>
</h2>
<div id="lb">
  <img id="lb-img" src="" alt=""/>
  <div id="lb-caption"></div>
</div>
<div class="grid">
{''.join(cards)}
</div>
<script>
var lb=document.getElementById('lb'),
    lbi=document.getElementById('lb-img'),
    lbc=document.getElementById('lb-caption');
document.querySelectorAll('img').forEach(function(img){{
  img.addEventListener('click',function(e){{
    e.stopPropagation();
    lbi.src=img.src;
    var cap=img.closest('figure');
    lbc.textContent=cap ? (cap.querySelector('figcaption')||{{}}).textContent||'' : '';
    lb.classList.add('open');
  }});
}});
lb.addEventListener('click',function(){{lb.classList.remove('open');lbi.src='';lbc.textContent='';}});
document.addEventListener('keydown',function(e){{
  if(e.key==='Escape'){{lb.classList.remove('open');lbi.src='';lbc.textContent='';}}
}});
</script>
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

    # Words that appear exclusively on safe/confirm dismiss buttons.
    _BUTTON_EXACT_RE = re.compile(
        r'^(ok|okay|close|yes|continue|dismiss|accept|allow|done|got it|'
        r'close firefox|open new session|start new session|confirm|apply|update|later)$',
        re.I
    )
    # Words that can appear on dismiss buttons but may also occur in body text.
    _BUTTON_PARTIAL_RE = re.compile(
        r'\b(ok|okay|close|yes|no|cancel|continue|dismiss|accept|allow|deny|'
        r'try again|later|update|apply|confirm|submit|done|got it|close firefox|'
        r'open new session|start new session|quit)\b',
        re.I
    )

    def _preferred_resolve_target(self, snapshot: Dict[str, Any], blocker: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Pick the best resolve target to click in order to dismiss a blocker.

        Uses a continuous numeric score (lower = better candidate) derived from:

          1. OCR text match quality
             • Exact-match on known button words → 0
             • Partial-match                     → 10
             • No match                          → 30

          2. Clickability geometry (must pass minimum dimensions to qualify):
             min width ≥ 40px, min height ≥ 16px,
             aspect ratio 1.5 ≤ w/h ≤ 15,
             area < 25 000 px² and height < 60px  → strong button bonus (-15)

          3. Proximity to modal footer: boxes whose Y-centre is in the lower 30%
             of the screen image get a -8 bonus.

          4. OCR confidence: each 10-point confidence above 50 removes 1 from score.

          5. Area (as a tie-breaker on a log scale): smaller is better, but artifacts
             (area < 500 or w < 40 or h < 16) are penalised +50 to filter them out.

        Reverses the old "largest area wins" order that caused the agent to click
        dialog body text instead of the dismiss button.
        """
        if blocker is None:
            return None

        # Fix K: For browser-chrome blockers (permission prompts, session restore),
        # prefer A11y button nodes over OCR boxes — they give precise hit-target coords.
        blocker_class_k = str(blocker.get("class", "") or "")
        if A11Y_BRIDGE_URL and self.last_ax_nodes:
            # Collect the expected button label words from all resolve targets
            ax_labels = []
            for rt in (blocker.get("resolve_targets") or []):
                lbl = str(rt.get("label", "") or rt.get("text", "") or "")
                if lbl:
                    ax_labels.append(lbl)
            # Add class-specific fallback labels
            if blocker_class_k == "browser_session_restore":
                ax_labels += ["Start New Session", "Restore Session", "start new", "restore"]
            elif blocker_class_k == "browser_permission_prompt":
                ax_labels += ["Allow", "Block", "Don't Allow", "Not Now"]
            elif blocker_class_k in ("modal_dialog", "cookie_banner"):
                ax_labels += ["OK", "Close", "Accept", "Dismiss", "Allow", "Got it"]
            if ax_labels:
                ax_btn = self._ax_best_button_for_labels(ax_labels)
                if ax_btn:
                    self._last_resolve_candidates = [{"box": ax_btn["box"], "text": ax_btn["text"], "_score": -100.0}]
                    return ax_btn

        img_h = int(self.last_img.shape[0]) if self.last_img is not None else 900

        candidates: List[Dict[str, Any]] = []
        for target in blocker.get("resolve_targets") or []:
            box = target.get("box") if isinstance(target.get("box"), list) else None
            scope = self._target_scope_from_box(box, snapshot, blocker)
            if box and scope == "page_content":
                bw   = max(1, int(box[2]) - int(box[0]))
                bh   = max(1, int(box[3]) - int(box[1]))
                area = bw * bh
                cy   = (int(box[1]) + int(box[3])) / 2.0
                candidates.append(dict(target, scope=scope, area=area, _bw=bw, _bh=bh, _cy=cy))
        if not candidates:
            return None

        import math

        def _score(c: Dict[str, Any]) -> float:
            text  = str(c.get("text", "") or c.get("label", "") or "")
            area  = int(c.get("area", 0))
            bh    = int(c.get("_bh", 999))
            bw    = int(c.get("_bw", 9999))
            cy    = float(c.get("_cy", img_h / 2))
            conf  = int(c.get("conf", 50))          # OCR confidence 0-100

            score = 0.0

            # ── 1. Text quality ──────────────────────────────────────────────
            text_stripped = text.strip()
            if self._BUTTON_EXACT_RE.match(text_stripped):
                score += 0.0          # best: exact button label
            elif self._BUTTON_PARTIAL_RE.search(text):
                score += 10.0         # partial match in longer text
            else:
                score += 30.0         # no button keyword found

            # ── 2. Geometry / clickability ───────────────────────────────────
            # Hard artifact filter (penalise, don't discard so we always return something)
            if area < 500 or bw < 40 or bh < 16:
                score += 50.0
            else:
                aspect = bw / max(1, bh)
                good_aspect = 1.5 <= aspect <= 15.0
                good_size   = bh < 60 and bw < 500 and area < 25_000
                if good_size and good_aspect:
                    score -= 15.0     # strong button-shape bonus
                elif good_size or good_aspect:
                    score -= 5.0      # partial geometry match

            # ── 3. Proximity to modal footer ─────────────────────────────────
            if cy >= img_h * 0.65:
                score -= 8.0          # near-bottom bonus

            # ── 4. OCR confidence ────────────────────────────────────────────
            score -= max(0.0, (conf - 50) / 10.0)

            # ── 5. Fix H: Line-box ambiguity penalties ───────────────────────
            # Penalise boxes that are wide relative to their text length (OCR row
            # boxes rather than tight button hit-targets).
            n_chars = max(1, len(text.strip()))
            w_per_char = bw / n_chars
            if w_per_char > 18:
                # Wide-per-character: classic row/line box (e.g. 800 px for 5 chars)
                score += min(30.0, (w_per_char - 18) * 1.5)

            aspect = bw / max(1, bh)
            if aspect > 20:
                # Extreme horizontal aspect ratio → almost certainly a row strip
                score += 20.0

            # Multi-action label: if the text contains ≥2 button words it's probably
            # a merged line like "Start New Session  Restore Session"
            n_action_words = len(self._BUTTON_PARTIAL_RE.findall(text))
            if n_action_words >= 2:
                score += 15.0

            # ── 6. Area tie-breaker (log scale, smaller preferred) ───────────
            score += math.log1p(area) * 0.4

            return score

        # Attach computed score to each candidate (for trace/debug visualization)
        for c in candidates:
            c["_score"] = _score(c)
        candidates.sort(key=lambda c: c["_score"])
        best = candidates[0]
        # Expose scored candidates as a class-level attribute for visual debug
        self._last_resolve_candidates = [
            {"box": c.get("box"), "text": c.get("text", ""), "_score": c["_score"]}
            for c in candidates
        ]
        # Strip internal scoring keys before returning
        return {k: v for k, v in best.items() if not k.startswith("_")}

    # Words on-screen that indicate the default action for Enter could be destructive
    # (e.g. delete, discard, quit). In those dialogs Escape should be tried first.
    _DESTRUCTIVE_DIALOG_RE = re.compile(
        r'\b(delete|remove|discard|quit|leave|uninstall|format|erase|wipe|'
        r'permanently|lose|overwrite)\b',
        re.I
    )
    # Words that confirm this dialog is a safe session-restore / continue dialog
    # where Enter is the right first move.
    _SAFE_RESTORE_RE = re.compile(
        r'\b(restore|reopen|continue|session|tabs?|windows?)\b',
        re.I
    )

    def _keyboard_escalation_key_order(self, blocker_class: str) -> List[str]:
        """
        Return [first_key, second_key] for keyboard escalation.

        Logic:
        • session-restore dialogs → Enter is always safe (restores tabs, not destructive).
        • Any dialog whose visible OCR text contains destructive words (delete / quit …)
          → try Escape first, then Enter only as a last resort.
        • Default (unknown / safe) → Enter first, then Escape.
        """
        if blocker_class == "browser_session_restore":
            return ["Return", "Escape"]

        # Collect all visible OCR text from the current frame.
        ocr_texts: List[str] = []
        for w in (self.last_ocr_words or []):
            t = str(w.get("text", "") or "")
            if t:
                ocr_texts.append(t)
        if not ocr_texts:
            for w in (self.last_ocr or []):
                t = str(w.get("text", "") or "")
                if t:
                    ocr_texts.append(t)
        joined = " ".join(ocr_texts)

        if self._DESTRUCTIVE_DIALOG_RE.search(joined) and not self._SAFE_RESTORE_RE.search(joined):
            return ["Escape", "Return"]
        return ["Return", "Escape"]

    def _crop_ocr_refine_target(
        self,
        target: Dict[str, Any],
        *,
        regex: Optional[str] = None,
        any_regex: Optional[List[str]] = None,
        fuzzy_text: Optional[str] = None,
        fuzzy_threshold: Optional[float] = None,
        prefer_bold: bool = False,
        nth: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if self.last_img is None:
            return None
        box = target.get("box")
        if not isinstance(box, list) or len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad = max(24, min(120, int(max(w, h) * 2.2)))
        img_h, img_w = self.last_img.shape[:2]
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(img_w, x2 + pad)
        bottom = min(img_h, y2 + pad)
        if right - left < 8 or bottom - top < 8:
            return None

        crop = self.last_img[top:bottom, left:right].copy()
        scale = 2 if max(w, h) <= 80 else 1
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop_levels = ocr_image_levels(crop)
        crop_words = list(crop_levels.get("words") or [])
        crop_lines = list(crop_levels.get("lines") or [])
        refined = find_text_box(
            crop_words,
            line_items=crop_lines,
            regex=regex,
            any_regex=any_regex,
            fuzzy_text=fuzzy_text,
            fuzzy_threshold=fuzzy_threshold,
            prefer_bold=prefer_bold,
            nth=nth,
        )
        if not refined:
            return None

        rx1, ry1, rx2, ry2 = [int(v) for v in refined.get("box", [0, 0, 0, 0])]
        mapped_box = [
            int(left + rx1 / scale),
            int(top + ry1 / scale),
            int(left + rx2 / scale),
            int(top + ry2 / scale),
        ]
        return {
            "text": refined.get("text", target.get("text", "")),
            "box": mapped_box,
            "level": f"{refined.get('level', 'word')}_crop",
            "scope": target.get("scope", ""),
        }

    def _refine_synthetic_target(
        self,
        target: Dict[str, Any],
        *,
        regex: Optional[str] = None,
        any_regex: Optional[List[str]] = None,
        fuzzy_text: Optional[str] = None,
        fuzzy_threshold: Optional[float] = None,
        prefer_bold: bool = False,
        nth: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Fix M: When a match originated from a synthetic sub-box (synth_word or
        line_subbox), the proportional geometry may be wrong for grid/tile UIs.

        Strategy: crop around the *parent* line box (not the tiny sub-box), upscale
        2-3×, run real OCR, re-run find_text_box().  Map the result back to image
        space.  If a real word box is found, return it; otherwise return None so the
        caller can try VLM.
        """
        if self.last_img is None:
            return None
        origin = str(target.get("_origin", "") or "")
        if origin not in BOX_ORIGINS_SYNTHETIC:
            return None  # Only refine synthetic origins

        # Use the parent line box as the crop region (much wider than the sub-box)
        parent_box = target.get("_parent_box")
        crop_box = parent_box if isinstance(parent_box, list) and len(parent_box) == 4 else target.get("box")
        if not isinstance(crop_box, list) or len(crop_box) != 4:
            return None

        px1, py1, px2, py2 = [int(v) for v in crop_box]
        img_h, img_w = self.last_img.shape[:2]
        # Add generous vertical padding so text above/below the line is included
        v_pad = max(20, int((py2 - py1) * 0.5))
        h_pad = max(24, int((px2 - px1) * 0.1))
        left   = max(0, px1 - h_pad)
        top    = max(0, py1 - v_pad)
        right  = min(img_w, px2 + h_pad)
        bottom = min(img_h, py2 + v_pad)
        if right - left < 16 or bottom - top < 8:
            return None

        crop = self.last_img[top:bottom, left:right].copy()
        # Upscale if the crop is small (tile grids often have small text)
        crop_h, crop_w = crop.shape[:2]
        scale = 3 if crop_h < 80 else (2 if crop_h < 160 else 1)
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        crop_levels = ocr_image_levels(crop)
        crop_words = list(crop_levels.get("words") or [])
        crop_lines = list(crop_levels.get("lines") or [])

        refined = find_text_box(
            crop_words,
            line_items=crop_lines,
            regex=regex,
            any_regex=any_regex,
            fuzzy_text=fuzzy_text,
            fuzzy_threshold=fuzzy_threshold,
            prefer_bold=prefer_bold,
            nth=nth,
        )
        if not refined:
            return None

        # Accept only if the refined result is not also synthetic
        refined_origin = str(refined.get("_origin", "") or "")
        if refined_origin in BOX_ORIGINS_SYNTHETIC:
            return None

        rx1, ry1, rx2, ry2 = [int(v) for v in refined.get("box", [0, 0, 0, 0])]
        mapped_box = [
            int(left + rx1 / scale),
            int(top  + ry1 / scale),
            int(left + rx2 / scale),
            int(top  + ry2 / scale),
        ]
        self._log("info", "refine_synthetic.found", {
            "original_origin": origin,
            "refined_origin": refined_origin,
            "parent_box": crop_box,
            "refined_box": mapped_box,
            "text": refined.get("text", ""),
        })
        return {
            "text": refined.get("text", target.get("text", "")),
            "box": mapped_box,
            "level": f"{refined.get('level', 'word')}_synth_refined",
            "_origin": "raw_word",  # promoted to trusted after real OCR
            "scope": target.get("scope", ""),
            "_subbox_applied": False,
        }

    def _validate_click_box(
        self,
        box: List[int],
        desired_text: str,
        *,
        pad_factor: float = 1.8,
    ) -> bool:
        """
        Fix N: Cheap pre-click validation.  Crops a window around the final box,
        runs OCR, and checks that the desired text is present inside the crop.

        Returns True if text is confirmed (or if validation cannot be run), so
        callers can treat True as "proceed" and False as "targeting failure".
        """
        if self.last_img is None or not desired_text:
            return True  # Can't validate — assume OK
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            img_h, img_w = self.last_img.shape[:2]
            pad_x = max(16, int(bw * (pad_factor - 1) / 2))
            pad_y = max(16, int(bh * (pad_factor - 1) / 2))
            left   = max(0, x1 - pad_x)
            top    = max(0, y1 - pad_y)
            right  = min(img_w, x2 + pad_x)
            bottom = min(img_h, y2 + pad_y)
            if right - left < 8 or bottom - top < 8:
                return True
            crop = self.last_img[top:bottom, left:right].copy()
            scale = 2 if max(bw, bh) <= 100 else 1
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            crop_levels = ocr_image_levels(crop)
            crop_words = list(crop_levels.get("words") or []) + list(crop_levels.get("lines") or [])
            all_text = " ".join(str(w.get("text", "") or "") for w in crop_words).lower()
            target_lower = desired_text.strip().lower()
            # Fuzzy: require at least 70% of the target chars to appear in the crop text
            found = target_lower in all_text
            if not found:
                score = fuzz.partial_ratio(target_lower, all_text)
                found = score >= 70
            self._log("debug", "validate_click_box", {
                "box": box, "desired": desired_text,
                "crop_text_snippet": all_text[:120], "found": found,
            })
            return found
        except Exception as e:
            self._log("warn", "validate_click_box.error", {"error": str(e)})
            return True  # On error, proceed rather than block

    def _maybe_refine_click_target(
        self,
        action: str,
        params: Dict[str, Any],
        snapshot: Dict[str, Any],
        blocker: Optional[Dict[str, Any]],
        target: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not target:
            return target
        box = target.get("box")
        if not isinstance(box, list) or len(box) != 4:
            return target

        scope = str(target.get("scope", "") or "")
        origin = str(target.get("_origin", "") or "")
        width  = max(1, int(box[2]) - int(box[0]))
        height = max(1, int(box[3]) - int(box[1]))

        regex     = params.get("regex")
        any_regex = params.get("patterns")
        if action == "click_near_text":
            regex     = params.get("anchor_regex")
            any_regex = None
        kwargs = dict(
            regex=regex,
            any_regex=any_regex if isinstance(any_regex, list) else None,
            fuzzy_text=params.get("fuzzy_text"),
            fuzzy_threshold=params.get("fuzzy_threshold"),
            prefer_bold=bool(params.get("prefer_bold", False)),
            nth=int(params.get("nth", 0)),
        )

        # Fix M: Synthetic origin on page_content → mandatory real OCR refinement
        if (
            action in ("click_text", "click_any_text", "click_near_text")
            and origin in BOX_ORIGINS_SYNTHETIC
            and scope != "browser_chrome"  # browser_chrome handled below
        ):
            refined = self._refine_synthetic_target(target, **kwargs)
            if refined:
                return refined
            # Could not refine — caller will invoke VLM fallback

        # Original chrome refinement logic (unchanged)
        if scope == "browser_chrome":
            tiny_target = width <= 140 and height <= 36
            if action in ("click_text", "click_any_text", "click_near_text") and tiny_target:
                refined = self._crop_ocr_refine_target(target, **kwargs)
                return refined or target

        return target

    def _should_send_planner_screenshot(self, snapshot: Dict[str, Any]) -> bool:
        mode = (PLANNER_SEND_SCREENSHOT_MODE or "auto").lower()
        if mode in ("0", "never", "false", "off"):
            return False
        if mode in ("1", "always", "true", "on"):
            return True

        blockers = self._active_blockers(snapshot)
        if not (self.last_ocr_words or self.last_ocr_lines):
            return True
        if "state:unclassified" in set(snapshot.get("tags", [])):
            return True
        if blockers:
            if any(not (blocker.get("resolve_targets") or []) for blocker in blockers):
                return True
            return False
        unresolved_recent = [
            item for item in self.history[-3:]
            if (item.get("result") or {}).get("outcome_verified") is False or (item.get("result") or {}).get("status") == "dispatched"
        ]
        if len(unresolved_recent) >= 2:
            return True
        if len(self.last_ocr_words) < 6 and len(self.last_ocr_lines) < 3:
            return True
        return False

    def _ax_best_button_for_labels(self, labels: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fix K: Search the A11y tree for a node with role 'button' (or similar clickable
        role) whose accessible name matches one of the provided label patterns.

        Returns a dict with 'box', 'text', and 'source'='ax' or None.
        Used to prefer precise A11y coords for browser-chrome blockers over OCR line boxes.
        """
        if not self.last_ax_nodes:
            return None
        clickable_roles = {"button", "link", "menuitem", "option", "checkbox", "radio",
                           "tab", "treeitem", "listitem", "menuitemcheckbox", "menuitemradio"}
        label_pats = [re.compile(lb, re.I) for lb in labels]
        best: Optional[Dict[str, Any]] = None
        for node in self.last_ax_nodes:
            role = str(node.get("role", "") or "").lower()
            if role not in clickable_roles:
                continue
            name = str(node.get("name", "") or node.get("value", "") or "")
            if not any(p.search(name) for p in label_pats):
                continue
            box = node.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            bw = max(1, int(box[2]) - int(box[0]))
            bh = max(1, int(box[3]) - int(box[1]))
            area = bw * bh
            if area < 16:
                continue
            # Prefer the smallest matching button (tightest hit target)
            if best is None or area < (max(1, (best["box"][2]-best["box"][0]) * (best["box"][3]-best["box"][1]))):
                best = {"box": [int(v) for v in box], "text": name, "source": "ax", "role": role}
        if best:
            self._log("info", "ax.button_found", {"text": best["text"], "box": best["box"], "role": best["role"]})
        return best

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
            # Fix P: Deterministic dismissal before any page-content click.
            # A click on page content while the dropdown is open is often "consumed"
            # by the dropdown and does NOT activate the underlying target.
            # Strategy: always send Esc first (up to 2 times); only then allow the
            # page click through. This is more reliable than "click and hope".
            esc_count = counts.get("escape", 0)

            # Explicit escape action always allowed up to 2 attempts
            if strategy == "escape" and esc_count < 2:
                return allow("dropdown_escape_allowed")

            # open_url bypasses the whole dropdown problem entirely
            if strategy == "open_url":
                return allow("open_url_bypasses_browser_dropdown")

            # Type/submit only allowed after dropdown is already dismissed
            if strategy == "type" and focus.get("focused_editable"):
                return allow("typing_allowed_once_focus_is_editable")
            if strategy == "submit_key" and focus.get("focused_editable") and counts.get("submit_key", 0) < 1:
                return allow("submit_allowed_when_editable_focus_confirms_intent")

            # Fix P: If we need to click a page target or browser chrome but haven't
            # sent Esc yet, send Esc first.  After Esc is confirmed (esc_count >= 1),
            # the next cycle will re-detect the state; if dropdown is gone, the
            # original action will be allowed normally.
            if strategy in ("click_visible_page_target", "click_browser_chrome"):
                if esc_count < 1:
                    return override(
                        "key_seq", {"keys": ["Escape"]},
                        "fix_p_esc_before_page_click_to_dismiss_dropdown", "escape",
                    )
                if esc_count < 2:
                    # Second Esc attempt if dropdown persisted after first
                    return override(
                        "key_seq", {"keys": ["Escape"]},
                        "fix_p_second_esc_dropdown_still_present", "escape",
                    )
                # Dropdown survived two Esc presses — allow the click as last resort
                return allow("dropdown_survived_two_esc_allow_page_click")

            # For anything else: send Esc if not yet tried
            if esc_count < 1:
                return override("key_seq", {"keys": ["Escape"]}, "browser_dropdown_first_try_escape", "escape")
            if strategy == "refocus_urlbar" and counts.get("refocus_urlbar", 0) < 1:
                return allow("single_urlbar_refocus_attempt_allowed")
            click_away_box = self._page_click_away_box(snapshot, blocker)
            if click_away_box and counts.get("click_away_page", 0) < 1:
                return override("click_box", {"box": click_away_box}, "browser_dropdown_try_click_away", "click_away_page")
            if counts.get("refocus_urlbar", 0) < 1:
                return override("key_seq", {"keys": ["Ctrl+l"]}, "browser_dropdown_refocus_urlbar", "refocus_urlbar")
            return block("browser_dropdown_requires_page_target_or_open_url_not_more_escape")

        if blocker_class == "browser_session_restore":
            preferred_target = self._preferred_resolve_target(snapshot, blocker)

            # Allow planner's explicit click_text resolve that landed on page content.
            if strategy == "click_visible_resolve_target":
                matched = self._match_resolve_target(blocker or {}, target)
                if matched is not None and str((target or {}).get("scope", "")) == "page_content":
                    return allow("page_level_session_restore_resolve_target_selected")

            # FIX B: If the planner sent an explicit click_box with a more precise target
            # than our preferred CTA, trust the planner and skip the override.
            # Priority order for checks:
            #   1. Containment: planner box fully inside preferred → always allow (more specific)
            #   2. Absolute size: area ≤ 25 000 px² AND h ≤ 80 px → looks like a real button
            #   3. Relative area: planner box ≤ 25% of preferred AND h ≤ 80 px
            if action == "click_box" and target:
                planner_box = (target or {}).get("box")
                if isinstance(planner_box, list) and len(planner_box) == 4:
                    px1, py1, px2, py2 = [int(v) for v in planner_box]
                    p_w    = max(1, px2 - px1)
                    p_h    = max(1, py2 - py1)
                    p_area = p_w * p_h
                    pref_box = (preferred_target or {}).get("box") if preferred_target else None
                    if isinstance(pref_box, list) and len(pref_box) == 4:
                        rx1, ry1, rx2, ry2 = [int(v) for v in pref_box]
                        ref_area = max(1, (rx2 - rx1) * (ry2 - ry1))
                        # Check 1: containment (planner inside preferred → allow)
                        contained = (px1 >= rx1 and py1 >= ry1 and px2 <= rx2 and py2 <= ry2)
                    else:
                        ref_area  = float("inf")
                        contained = False
                        rx1 = ry1 = rx2 = ry2 = 0
                    # Check 2: absolute button size
                    abs_button = p_area <= 25_000 and p_h <= 80
                    # Check 3: relative area ratio
                    rel_small  = p_area <= ref_area * 0.25 and p_h < 80
                    if contained or abs_button or rel_small:
                        return allow("planner_chose_more_precise_target_than_cta_override")

            # Override to the preferred (button-shaped) resolve target while attempts remain.
            if preferred_target and counts.get("click_visible_resolve_target", 0) < 2:
                return override(
                    "click_box",
                    {"box": [int(v) for v in preferred_target.get("box", [0, 0, 0, 0])]},
                    "prefer_session_restore_cta_button",
                    "click_visible_resolve_target",
                )

            # FIX C: After click attempts are exhausted, escalate to keyboard before giving up.
            # Key order is determined by OCR context (destructive vs safe dialog).
            _kb_order = self._keyboard_escalation_key_order("browser_session_restore")
            if counts.get("keyboard_confirm", 0) < 1:
                return override(
                    "key_seq", {"keys": [_kb_order[0]]},
                    f"session_restore_keyboard_confirm_key_{_kb_order[0].lower()}_after_click_exhausted",
                    "keyboard_confirm",
                )
            if counts.get("escape", 0) < 1:
                return override(
                    "key_seq", {"keys": [_kb_order[1]]},
                    f"session_restore_keyboard_fallback_key_{_kb_order[1].lower()}",
                    "escape",
                )

            # FIX I: Address-bar bypass — after all clicks and keyboard attempts are
            # exhausted, try navigating away via Ctrl+L (focus URL bar) + Return.
            # This sidesteps the restore UI entirely.  Sequence:
            #   Step 1 (addressbar_bypass count=0): Ctrl+L to focus the address bar.
            #   Step 2 (addressbar_bypass count=1): Return to navigate to the current
            #           URL (whatever was already in the bar) or the planner will issue
            #           an open_url in the next cycle.
            if counts.get("addressbar_bypass", 0) == 0:
                return override(
                    "key_seq", {"keys": ["ctrl+l"]},
                    "session_restore_addressbar_bypass_step1_focus",
                    "addressbar_bypass",
                )
            if counts.get("addressbar_bypass", 0) == 1:
                return override(
                    "key_seq", {"keys": ["Return"]},
                    "session_restore_addressbar_bypass_step2_navigate",
                    "addressbar_bypass",
                )

            if strategy == "open_url":
                return allow("open_url_bypasses_session_restore_page")

            return block("session_restore_requires_large_page_resolve_target_or_open_url")

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
            # Use the button-preferring sort for the generic case too.
            preferred = self._preferred_resolve_target(snapshot, blocker)
            primary = preferred if preferred else resolve_targets[0]
            return override(
                "click_box",
                {"box": [int(v) for v in primary.get("box", [0, 0, 0, 0])]},
                f"use_visible_{blocker_class}_resolve_target",
                "click_visible_resolve_target",
            )
        # FIX C (generic): keyboard escalation after click attempts exhausted.
        # Key order respects whether the visible dialog text looks destructive.
        _kb_order_g = self._keyboard_escalation_key_order(blocker_class)
        if counts.get("keyboard_confirm", 0) < 1:
            return override(
                "key_seq", {"keys": [_kb_order_g[0]]},
                f"keyboard_confirm_key_{_kb_order_g[0].lower()}_fallback_for_{blocker_class}",
                "keyboard_confirm",
            )
        if counts.get("escape", 0) < 1:
            return override(
                "key_seq", {"keys": [_kb_order_g[1]]},
                f"keyboard_key_{_kb_order_g[1].lower()}_fallback_for_{blocker_class}",
                "escape",
            )
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
            # FIX D: Before hard-failing, check whether keyboard dismissal has been tried
            # for the current blocker.  If not, escalate proactively so the caller can
            # act on the suggestion rather than just seeing a bare failure.
            # Key order is determined by OCR context (destructive vs safe dialog).
            current_blocker = self._primary_blocker(snapshot)
            if current_blocker:
                bsig  = str(current_blocker.get("signature", "") or "")
                bclass = str(current_blocker.get("class", "") or "")
                b_counts = self._attempt_counts(self._recent_blocker_attempts(bsig))
                _kb_order_guard = self._keyboard_escalation_key_order(bclass)
                if b_counts.get("keyboard_confirm", 0) < 1:
                    _k0 = _kb_order_guard[0]
                    return {
                        "status": "failure",
                        "error_code": "ANTI_REPEAT_GUARD_KEYBOARD_ESCALATION",
                        "error_message": f"Anti-repeat guard: escalating to {_k0} to dismiss blocker",
                        "event_applied": False,
                        "recovery_hint": f"use_key_seq_{_k0}_to_dismiss_blocker",
                        "suggested_action": "key_seq",
                        "suggested_params": {"keys": [_k0]},
                        "guard_reason": "same_family_repeated_escalating_to_keyboard",
                        "blocker_class": bclass,
                    }
                if b_counts.get("escape", 0) < 1:
                    _k1 = _kb_order_guard[1]
                    return {
                        "status": "failure",
                        "error_code": "ANTI_REPEAT_GUARD_KEYBOARD_ESCALATION",
                        "error_message": f"Anti-repeat guard: escalating to {_k1} to dismiss blocker",
                        "event_applied": False,
                        "recovery_hint": f"use_key_seq_{_k1}_to_dismiss_blocker",
                        "suggested_action": "key_seq",
                        "suggested_params": {"keys": [_k1]},
                        "guard_reason": "same_family_repeated_escalating_to_escape",
                        "blocker_class": bclass,
                    }
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

    def _shadow_candidate_actions(self, element: Dict[str, Any]) -> List[str]:
        role = str(element.get("role", "") or "").lower()
        text = str(element.get("text", "") or "").strip()
        source = str(element.get("source", "") or "").lower()
        if role in ("textbox", "entry", "textarea", "input"):
            return ["click", "type", "focus"]
        if role in ("button", "link", "tab", "menuitem", "checkbox", "radio"):
            return ["click", "hover"]
        if source == "ax" and role:
            return ["click", "focus"]
        if text:
            return ["click"]
        return ["click"]

    def _build_target_ensemble_candidates(self, ui_elements: List[Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for idx, element in enumerate(ui_elements[:limit], start=1):
            box = element.get("box") or [0, 0, 0, 0]
            if not isinstance(box, list) or len(box) != 4:
                continue
            candidates.append(
                {
                    "id": f"C{idx:03d}",
                    "box": [int(v) for v in box],
                    "text": str(element.get("text", "") or ""),
                    "score": float(element.get("score", 0.0) or 0.0),
                    "role": element.get("role"),
                    "source": element.get("source"),
                    "allowed_actions": self._shadow_candidate_actions(element),
                    "extras": {},
                }
            )
        return candidates

    def _build_target_ensemble_instruction(self, action: str, params: Dict[str, Any]) -> str:
        if action == "click_text":
            target = params.get("regex") or params.get("fuzzy_text") or params.get("any_regex")
            if target:
                return f"click the UI element matching {target}"
        if action == "click_any_text":
            patterns = params.get("patterns") or params.get("texts") or []
            if isinstance(patterns, list) and patterns:
                return f"click the first matching UI element among {patterns[0]}"
        if action == "click_near_text":
            anchor = params.get("anchor_regex") or params.get("anchor")
            if anchor:
                return f"click near the UI element matching {anchor}"
        if action == "click_box":
            return "click the intended target inside the provided bounding box"
        return AGENT_GOAL or f"resolve the next {action} step"

    def _target_ensemble_shadow(
        self,
        action: str,
        params: Dict[str, Any],
        ui_elements: List[Dict[str, Any]],
        screenshot_b64: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not TARGET_ENSEMBLE_SHADOW_MODE or not TARGET_ENSEMBLE_API_URL:
            return None
        if action not in ("click_text", "click_any_text", "click_near_text", "click_box"):
            return None
        candidates = self._build_target_ensemble_candidates(ui_elements)
        if not candidates:
            return None
        shot = (screenshot_b64 or "").strip()
        if not shot and self.last_img is not None:
            try:
                jpg = encode_jpeg_bgr_resized(self.last_img, q=max(25, min(90, int(PLANNER_SCREENSHOT_JPEG_QUALITY))), max_dim=PLANNER_SCREENSHOT_MAX_DIM)
                shot = base64.b64encode(jpg).decode("ascii")
            except Exception:
                shot = ""
        if not shot:
            return None
        payload = {
            "instruction": self._build_target_ensemble_instruction(action, params),
            "history": [str((item or {}).get("action", "")) for item in self.history[-4:]],
            "screenshot_b64": shot,
            "screenshot_mime": "image/jpeg",
            "candidates": candidates,
            "top_k": TARGET_ENSEMBLE_SHADOW_TOP_K,
            "debug": bool(self.trace_enabled and TARGET_ENSEMBLE_SHADOW_DEBUG),
        }
        endpoint = TARGET_ENSEMBLE_API_URL.rstrip("/") + ("/infer/debug" if payload["debug"] and not TARGET_ENSEMBLE_API_URL.rstrip("/").endswith("/infer/debug") else "")
        if not payload["debug"] and not endpoint.endswith("/infer"):
            endpoint = TARGET_ENSEMBLE_API_URL.rstrip("/") + ("" if TARGET_ENSEMBLE_API_URL.rstrip("/").endswith("/infer") else "/infer")
        try:
            resp = self.session.post(endpoint, json=payload, timeout=(2.0, TARGET_ENSEMBLE_SHADOW_TIMEOUT_S))
            resp.raise_for_status()
            data = resp.json()
            final_pred = data.get("final_prediction") or {}
            self._log(
                "info",
                "target_ensemble.shadow",
                {
                    "requested_action": action,
                    "instruction": payload["instruction"],
                    "candidate_count": len(candidates),
                    "final_candidate_id": final_pred.get("candidate_id"),
                    "final_score": final_pred.get("score"),
                },
            )
            return data
        except Exception as e:
            self._log("warn", "target_ensemble.shadow_failed", {"action": action, "error": str(e)})
            return {"status": "error", "error": str(e), "requested_action": action}

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
        # Fix O: Separate caps per OCR level.
        # - Words: sorted by confidence first (highest conf = best word detection),
        #   then by box height to keep taller glyphs. Hard cap raised to 2000 to
        #   prevent dropping real word boxes and forcing synthetic fallback.
        # - Lines: sorted by height desc (large regions first), capped at a smaller
        #   limit since there are naturally fewer line-level items.
        _word_cap = max(OCR_LIMIT, 2000)
        _line_cap = max(40, min(OCR_LIMIT, 160))
        # Stamp raw_word origin on word items that have no origin yet
        for _w in self.last_ocr_words:
            _w.setdefault("_origin", "raw_word")
        for _l in self.last_ocr_lines:
            _l.setdefault("_origin", "line_item")
        self.last_ocr_words = sorted(
            self.last_ocr_words,
            key=lambda w: (-float(w.get("conf", 100)), -(w["box"][3] - w["box"][1])),
        )[:_word_cap]
        self.last_ocr_lines = sorted(
            self.last_ocr_lines,
            key=lambda w: (-(w["box"][3] - w["box"][1]), -float(w.get("conf", 100))),
        )[:_line_cap]
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
        snapshot = self._state_snapshot()
        blocker = self._primary_blocker(snapshot)
        for pat in patterns:
            w = find_text_box(
                self.last_ocr_words or self.last_ocr,
                line_items=self.last_ocr_lines,
                regex=pat,
                nth=nth,
                prefer_bold=prefer_bold
            )
            if w:
                ocr_box = list(w.get("_original_box") or w["box"])
                subbox_applied = bool(w.get("_subbox_applied", False))
                w_origin = str(w.get("_origin", "") or "")
                w = self._maybe_refine_click_target("click_any_text", {"patterns": patterns, "nth": nth, "prefer_bold": prefer_bold}, snapshot, blocker, w) or w
                x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
                vis_paths = self._save_click_visual_debug(
                    ocr_box=ocr_box,
                    final_box=[int(x1), int(y1), int(x2), int(y2)],
                    click_rel=(int(cx), int(cy)),
                    action="click_any_text",
                    subbox_applied=subbox_applied,
                    parent_box=w.get("_parent_box"),
                    origin=w_origin,
                    candidates_debug=self._last_resolve_candidates,
                )
                self._debug_save_click_crop(
                    box=[int(x1), int(y1), int(x2), int(y2)],
                    click_rel=(int(cx), int(cy)),
                    action="click_any_text",
                    note=f"pattern_used={pat} level={w.get('level', 'word')} subbox={subbox_applied}",
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
                    "ocr_box": ocr_box,
                    "pattern_used": pat,
                    "ocr_level": w.get("level", "word"),
                    "targeting_source": f"ocr_{w.get('level', 'word')}",
                    "box_origin": w_origin,
                    "subbox_applied": subbox_applied,
                    "click_abs": [x_abs, y_abs],
                    "hierarchy_overlay_path": vis_paths.get("hierarchy_path", ""),
                    "ocr_box_overlay_path": vis_paths.get("ocr_box_path", ""),
                    "click_target_overlay_path": vis_paths.get("click_target_path", ""),
                    "level_overlay_paths": vis_paths.get("level_paths") or [],
                }
        return {"status":"failure","error_code":"ELEMENT_NOT_FOUND","error_message": f"No pattern matched: {patterns}"}

    def _action_click_near_text(self, anchor_regex: str, dx: int = 0, dy: int = 0):
        import re
        snapshot = self._state_snapshot()
        blocker = self._primary_blocker(snapshot)
        w = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, regex=anchor_regex)
        if not w:
            return {"status":"failure","error_code":"ANCHOR_NOT_FOUND","error_message": anchor_regex}
        ocr_box = list(w["box"])
        w_origin = str(w.get("_origin", "") or "")
        w = self._maybe_refine_click_target("click_near_text", {"anchor_regex": anchor_regex, "dx": dx, "dy": dy}, snapshot, blocker, w) or w
        x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2
        # Guard against header/account anchors (e.g., email in header)
        if "@" in w.get("text", "") and cy < 80:
            return {"status":"failure","error_code":"ANCHOR_REJECTED_HEADER","error_message": w.get("text", "")}
        click_x, click_y = int(cx + dx), int(cy + dy)
        click_final_box = [click_x - 12, click_y - 12, click_x + 12, click_y + 12]
        vis_paths = self._save_click_visual_debug(
            ocr_box=ocr_box,
            final_box=click_final_box,
            click_rel=(click_x, click_y),
            action="click_near_text",
            parent_box=w.get("_parent_box"),
            origin=w_origin,
            candidates_debug=self._last_resolve_candidates,
        )
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
            "ocr_box": ocr_box,
            "offset": [dx, dy],
            "anchor_level": w.get("level", "word"),
            "targeting_source": f"ocr_{w.get('level', 'word')}",
            "box_origin": w_origin,
            "click_abs": [x_abs, y_abs],
            "hierarchy_overlay_path": vis_paths.get("hierarchy_path", ""),
            "ocr_box_overlay_path": vis_paths.get("ocr_box_path", ""),
            "click_target_overlay_path": vis_paths.get("click_target_path", ""),
            "level_overlay_paths": vis_paths.get("level_paths") or [],
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

        # FIX E (refined): Bottom-band bias for large dialog boxes under an active blocker.
        #
        # The bias is SKIPPED when:
        #   1. The box is small / button-sized → likely already targeting the right element.
        #   2. A preferred button target from Fix A is available with a small bbox → click
        #      that instead (the caller — _blocker_policy_directive — should have already done
        #      this, but guard here as a belt-and-suspenders check).
        #
        # When bias IS applied:
        #   • cy is set to the footer band [0.70h, 0.92h] within the box.
        #   • cx is biased toward the right half of the box (where most UI buttons live
        #     in LTR layouts) but kept within [x1+32, x2-32].
        #   • The bias only fires if OCR confirms a button-like word is in the lower third
        #     of the image, OR if the box is so large it clearly spans the entire dialog.
        bottom_biased = False
        bottom_bias_reason = ""
        bw = x2 - x1
        bh = y2 - y1
        _is_large_dialog_body = bw > 400 and bh > 120
        if _is_large_dialog_body and self.last_img is not None:
            snap = self._state_snapshot()
            active_blocker = self._primary_blocker(snap)
            if active_blocker:
                img_h = int(self.last_img.shape[0])
                # Check 1: do we already have a precise button bbox from the blocker?
                preferred_tgt = self._preferred_resolve_target(snap, active_blocker)
                p_box = (preferred_tgt or {}).get("box")
                if isinstance(p_box, list) and len(p_box) == 4:
                    p_bh = max(1, int(p_box[3]) - int(p_box[1]))
                    p_bw = max(1, int(p_box[2]) - int(p_box[0]))
                    if p_bh < 60 and p_bw < 500:
                        # A precise button target is known; skip the bias entirely
                        _is_large_dialog_body = False

                if _is_large_dialog_body:
                    # Check 2: confirm a dismiss word is visible in the lower 35% of the image
                    _footer_ocr_hit = False
                    footer_y_threshold = img_h * 0.65
                    for w in (self.last_ocr_words or self.last_ocr or []):
                        w_box = w.get("box") or []
                        if len(w_box) >= 4:
                            w_cy = (int(w_box[1]) + int(w_box[3])) / 2.0
                            if w_cy >= footer_y_threshold and self._BUTTON_PARTIAL_RE.search(str(w.get("text", ""))):
                                _footer_ocr_hit = True
                                break
                    # Also fire if the box is extremely large (spans > 60% of screen height)
                    _very_large = bh > img_h * 0.45
                    if _footer_ocr_hit or _very_large:
                        # Apply footer-band bias: cy in [70%, 92%] of bh
                        cy = y1 + int(bh * 0.80)
                        cy = max(y1 + int(bh * 0.70), min(cy, y1 + int(bh * 0.92)))
                        cy = max(y1 + 4, min(cy, y2 - 4))
                        # Bias cx toward right side (most LTR button rows are right-aligned)
                        cx_right_bias = x1 + int(bw * 0.70)
                        cx = max(x1 + 32, min(cx_right_bias, x2 - 32))
                        bottom_biased = True
                        bottom_bias_reason = "footer_ocr_hit" if _footer_ocr_hit else "very_large_box"

        box_action_tag = "click_box_biased" if bottom_biased else "click_box"
        vis_paths = self._save_click_visual_debug(
            ocr_box=[x1, y1, x2, y2],
            final_box=[x1, y1, x2, y2],
            click_rel=(int(cx), int(cy)),
            action=box_action_tag,
            candidates_debug=self._last_resolve_candidates,
        )
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
        return {
            "status": "success",
            "box": [x1, y1, x2, y2],
            "clicked_abs": [x_abs, y_abs],
            "bottom_biased": bottom_biased,
            "bottom_bias_reason": bottom_bias_reason,
            "hierarchy_overlay_path": vis_paths.get("hierarchy_path", ""),
            "ocr_box_overlay_path": vis_paths.get("ocr_box_path", ""),
            "click_target_overlay_path": vis_paths.get("click_target_path", ""),
            "level_overlay_paths": vis_paths.get("level_paths") or [],
        }

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
        snapshot = self._state_snapshot()
        blocker = self._primary_blocker(snapshot)
        w = find_text_box(self.last_ocr_words or self.last_ocr, line_items=self.last_ocr_lines, regex=regex, nth=nth, prefer_bold=prefer_bold,
                          fuzzy_text=fuzzy_text, fuzzy_threshold=fuzzy_threshold)
        if not w:
            return {"status": "failure", "error_code": "ELEMENT_NOT_FOUND",
                    "error_message": f"Regex '{regex}' not found"}
        ocr_box = list(w.get("_original_box") or w["box"])  # original pre-subbox box for visual
        subbox_applied = bool(w.get("_subbox_applied", False))
        origin_before_refine = str(w.get("_origin", "") or "")
        refine_params = {"regex": regex, "nth": nth, "prefer_bold": prefer_bold,
                         "fuzzy_text": fuzzy_text, "fuzzy_threshold": fuzzy_threshold}
        # Fix M: if origin is synthetic, _maybe_refine_click_target will attempt a real
        # OCR pass on the parent line box. If it returns None the target is still synthetic
        # and we need VLM (handled below).
        refined_w = self._maybe_refine_click_target("click_text", refine_params, snapshot, blocker, w)
        synthetic_refine_failed = (
            origin_before_refine in BOX_ORIGINS_SYNTHETIC
            and (refined_w is None or str(refined_w.get("_origin", "") or "") in BOX_ORIGINS_SYNTHETIC)
        )
        w = refined_w or w
        x1,y1,x2,y2 = w["box"]; cx,cy = (x1+x2)//2, (y1+y2)//2

        # Fix J + Fix L/M: VLM fallback when:
        #   • the chosen box is geometrically ambiguous, OR
        #   • the same blocker has failed ≥ 2 times, OR
        #   • Fix M refinement failed (origin is still synthetic)
        targeting_source = f"ocr_{w.get('level', 'word')}"
        vlm_used = False
        if LLM_API_URL:
            blocker_failure_count = 0
            if blocker:
                bsig = str(blocker.get("signature", "") or "")
                b_counts = self._attempt_counts(self._recent_blocker_attempts(bsig))
                blocker_failure_count = b_counts.get("click_visible_resolve_target", 0)
            should_try_vlm = (
                synthetic_refine_failed
                or self._box_is_ambiguous(w["box"], w.get("text", ""))
                or blocker_failure_count >= 2
            )
            if should_try_vlm:
                label = w.get("text", "") or str(regex or fuzzy_text or "")
                ctx_hint = f"blocker={blocker.get('class', '')}" if blocker else ""
                vlm_box = self._vlm_locate_box(label, ctx_hint)
                if vlm_box:
                    vlm_bw = max(1, vlm_box[2] - vlm_box[0])
                    vlm_bh = max(1, vlm_box[3] - vlm_box[1])
                    # Use VLM result if it's tighter than current box (or if refine failed)
                    cur_area = (x2 - x1) * (y2 - y1)
                    vlm_area = vlm_bw * vlm_bh
                    if vlm_area < cur_area * 0.9 or synthetic_refine_failed:
                        x1, y1, x2, y2 = vlm_box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        vlm_used = True
                        targeting_source = "vlm_locate"
                        self._log("info", "click_text.vlm_fallback_used", {
                            "original_box": w["box"], "vlm_box": vlm_box,
                            "origin_before": origin_before_refine,
                            "synthetic_refine_failed": synthetic_refine_failed,
                            "regex": regex, "label": label,
                        })

        # Fix N: Pre-click validation gate — confirm the desired text is in the crop.
        # Skip for VLM-derived boxes (they already went through visual confirmation)
        # and for boxes from a11y (trusted).
        desired_text = w.get("text", "") or str(regex or fuzzy_text or "")
        w_origin = str(w.get("_origin", "") or "")
        if not vlm_used and w_origin not in ("a11y", "vlm"):
            valid = self._validate_click_box([x1, y1, x2, y2], desired_text)
            if not valid:
                self._log("warn", "click_text.validation_failed", {
                    "box": [x1, y1, x2, y2], "desired": desired_text,
                    "origin": w_origin, "level": w.get("level", ""),
                })
                # Escalate to VLM if available; otherwise return failure so planner can re-plan
                if LLM_API_URL:
                    label = desired_text
                    ctx_hint = f"blocker={blocker.get('class', '')}" if blocker else ""
                    vlm_box = self._vlm_locate_box(label, ctx_hint)
                    if vlm_box:
                        x1, y1, x2, y2 = vlm_box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        vlm_used = True
                        targeting_source = "vlm_locate_validation_escalation"
                    else:
                        return {
                            "status": "failure",
                            "error_code": "VALIDATION_FAILED",
                            "error_message": f"Text '{desired_text}' not found in crop around box {[x1,y1,x2,y2]}",
                            "box": [x1, y1, x2, y2],
                            "origin": w_origin,
                        }
                else:
                    return {
                        "status": "failure",
                        "error_code": "VALIDATION_FAILED",
                        "error_message": f"Text '{desired_text}' not found in crop around box {[x1,y1,x2,y2]}",
                        "box": [x1, y1, x2, y2],
                        "origin": w_origin,
                    }

        action_tag = "click_text_vlm" if vlm_used else "click_text"
        vis_paths = self._save_click_visual_debug(
            ocr_box=ocr_box,
            final_box=[int(x1), int(y1), int(x2), int(y2)],
            click_rel=(int(cx), int(cy)),
            action=action_tag,
            subbox_applied=subbox_applied,
            parent_box=w.get("_parent_box"),
            origin=w_origin,
            candidates_debug=self._last_resolve_candidates,
        )
        self._debug_save_click_crop(
            box=[int(x1), int(y1), int(x2), int(y2)],
            click_rel=(int(cx), int(cy)),
            action="click_text",
            note=f"level={w.get('level', 'word')} subbox={subbox_applied} vlm={vlm_used}",
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
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "ocr_box": ocr_box,
            "ocr_level": w.get("level", "word"),
            "targeting_source": targeting_source,
            "box_origin": w_origin,
            "subbox_applied": subbox_applied,
            "synthetic_refine_failed": synthetic_refine_failed,
            "vlm_used": vlm_used,
            "click_abs": [x_abs, y_abs],
            "hierarchy_overlay_path": vis_paths.get("hierarchy_path", ""),
            "ocr_box_overlay_path": vis_paths.get("ocr_box_path", ""),
            "click_target_overlay_path": vis_paths.get("click_target_path", ""),
            "level_overlay_paths": vis_paths.get("level_paths") or [],
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
                if self._should_send_planner_screenshot(pre_action_snapshot) and self.last_img is not None:
                    q = max(25, min(90, int(PLANNER_SCREENSHOT_JPEG_QUALITY)))
                    jpg = encode_jpeg_bgr_resized(self.last_img, q=q, max_dim=PLANNER_SCREENSHOT_MAX_DIM)
                    screenshot_b64 = base64.b64encode(jpg).decode("ascii")
            except Exception as e:
                self._log("warn", "planner.screenshot.encode_failed", {"error": str(e)})

            try:
                # OCR elements (boxes are image-relative)
                for w in self.last_ocr_lines[:PLANNER_MAX_OCR_LINE_ELEMENTS]:
                    ocr_min.append({"text": w["text"], "box": w["box"], "conf": w["conf"], "level": "line"})
                    ui_elements.append({
                        "source": "ocr_line",
                        "text": w.get("text", ""),
                        "box": w.get("box", [0, 0, 0, 0]),
                        "score": float(w.get("conf", 0.0)) / 100.0 if float(w.get("conf", 0.0)) > 1.0 else float(w.get("conf", 0.0)),
                        "role": None,
                    })
                for w in self.last_ocr_words[:PLANNER_MAX_OCR_WORD_ELEMENTS]:
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
                    for node in self.last_ax_nodes[:PLANNER_MAX_AX_ELEMENTS]:
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
                "planner_session_id": self.planner_session_id,
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
            shadow_targeting = None
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

                shadow_targeting = self._target_ensemble_shadow(op, params, ui_elements, screenshot_b64)
                guard_result = self._executor_guard_result(op, params, pre_action_snapshot)
                if guard_result is not None:
                    # FIX D: If the guard suggests a keyboard action, execute it directly
                    # instead of surfacing the failure to the planner — up to the hard cap.
                    suggested_action = guard_result.get("suggested_action")
                    suggested_params = guard_result.get("suggested_params") or {}
                    _is_kb_escalation = (
                        suggested_action is not None
                        and guard_result.get("error_code") == "ANTI_REPEAT_GUARD_KEYBOARD_ESCALATION"
                    )
                    if _is_kb_escalation:
                        # Use OCR-context key ordering regardless of what _executor_guard_result chose
                        _blocker_for_key_order = self._primary_blocker(pre_action_snapshot) or {}
                        _bclass_for_ko = str((_blocker_for_key_order.get("class") if isinstance(_blocker_for_key_order, dict) else "") or "")
                        _kb_order_d = self._keyboard_escalation_key_order(_bclass_for_ko)
                        _bsig_d = str(
                            (_blocker_for_key_order.get("signature") if isinstance(_blocker_for_key_order, dict) else "")
                            or "unknown"
                        )
                        _escalation_count = self._guard_kb_escalations.get(_bsig_d, 0)
                        if _escalation_count >= self._GUARD_KB_ESCALATION_CAP:
                            # Cap reached: surface to planner so it can change strategy
                            _is_kb_escalation = False
                            self._log("warn", "guard.keyboard_escalation.cap_reached", {
                                "blocker_sig": _bsig_d,
                                "cap": self._GUARD_KB_ESCALATION_CAP,
                                "count": _escalation_count,
                                "surfacing_to_planner": True,
                            })
                        else:
                            # Pick the correct key based on context-aware order
                            _chosen_key = _kb_order_d[_escalation_count % len(_kb_order_d)]
                            suggested_params = {"keys": [_chosen_key]}
                            self._guard_kb_escalations[_bsig_d] = _escalation_count + 1
                    if _is_kb_escalation:
                        self._log("info", "guard.keyboard_escalation", {
                            "original_action": op,
                            "escalating_to": suggested_action,
                            "keys": suggested_params.get("keys"),
                            "reason": guard_result.get("guard_reason"),
                            "escalation_n": self._guard_kb_escalations.get(_bsig_d, 1),
                            "cap": self._GUARD_KB_ESCALATION_CAP,
                        })
                        escalated_params = self._canonicalize_params(suggested_action, suggested_params)
                        escalated_result = self._action_key_seq(**escalated_params) if suggested_action == "key_seq" else None
                        if escalated_result is not None:
                            escalated_result = self._finalize_action_result(
                                suggested_action, escalated_params, escalated_result, pre_action_snapshot
                            )
                            escalated_result["planner_requested_action"] = planner_op
                            escalated_result["planner_requested_parameters"] = planner_params
                            escalated_result["executed_action"] = suggested_action
                            escalated_result["executed_parameters"] = escalated_params
                            escalated_result["executor_override_reason"] = guard_result.get("guard_reason", "anti_repeat_keyboard_escalation")
                            escalated_result["recovery_strategy"] = "keyboard_confirm"
                            result = escalated_result
                            self.history.append({"action": suggested_action, "parameters": escalated_params, "result": result})
                            self._log("info", "action.result", {"action": suggested_action, "result": result})
                            post_frame_path = self._save_trace_frame(suggested_action, "post")
                            self._trace_append({
                                "idx": self.trace_idx,
                                "ts": now_utc_iso(),
                                "decision": {"action": planner_op, "parameters": planner_params, "reasoning": reasoning, "completed": completed},
                                "result": result,
                                "target_ensemble_shadow": shadow_targeting or {},
                                "frame_path": frame_path,
                                "post_frame_path": post_frame_path,
                                "crop_overlay_path": "",
                                "hierarchy_overlay_path": result.get("hierarchy_overlay_path") or "",
                                "ocr_box_overlay_path": "",
                                "click_target_overlay_path": "",
                                "level_overlay_paths": result.get("level_overlay_paths") or [],
                                "ocr_results_n": len(ocr_min),
                                "ocr_line_results_n": len(self.last_ocr_lines),
                                "ocr_word_results_n": len(self.last_ocr_words),
                                "ui_elements_n": len(ui_elements),
                                "screenshot_b64_len": len(screenshot_b64) if screenshot_b64 else 0,
                            })
                            self._trace_render_html()
                            step += 1
                            time.sleep(0.8)
                            continue
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
                                    "target_ensemble_shadow": shadow_targeting or {},
                                    "frame_path": frame_path,
                                    "post_frame_path": post_frame_path,
                                    "crop_overlay_path": crop_overlay,
                                    "hierarchy_overlay_path": result.get("hierarchy_overlay_path") or "",
                                    "ocr_box_overlay_path": result.get("ocr_box_overlay_path") or "",
                                    "click_target_overlay_path": result.get("click_target_overlay_path") or "",
                                    "level_overlay_paths": result.get("level_overlay_paths") or [],
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
                    "target_ensemble_shadow": shadow_targeting or {},
                    "frame_path": frame_path,
                    "post_frame_path": post_frame_path,
                    "crop_overlay_path": crop_overlay,
                    "hierarchy_overlay_path": result.get("hierarchy_overlay_path") or "",
                    "ocr_box_overlay_path": result.get("ocr_box_overlay_path") or "",
                    "click_target_overlay_path": result.get("click_target_overlay_path") or "",
                    "level_overlay_paths": result.get("level_overlay_paths") or [],
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

       