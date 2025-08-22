import os
import cv2
import json
import time
import mss
import numpy as np
import subprocess
import requests
from typing import List, Dict, Any, Optional, Tuple

API_URL = os.environ.get("RTDETR_API_URL", "http://rtdetr-api:8000/predict")
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.50"))
MIN_BOX_AREA = int(os.environ.get("MIN_BOX_AREA", "800"))  # ignore tiny detections
CLICK_ENABLED = os.environ.get("CLICK_ENABLED", "1") == "1"
SAVE_DEBUG = os.environ.get("SAVE_DEBUG", "0") == "1"
SLEEP_BETWEEN_LOOPS = float(os.environ.get("LOOP_SLEEP", "0.25"))  # seconds

def capture_screen() -> np.ndarray:
    """Capture the primary monitor of the VNC display (:1). Returns RGB image."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary display
        img = np.array(sct.grab(monitor))  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # to BGR
        return img

def encode_jpeg_bgr(image_bgr: np.ndarray) -> bytes:
    """Encode to JPEG; keep full resolution to preserve coordinate space."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def call_rtdetr_api(session: requests.Session, image_bgr: np.ndarray, timeout: Tuple[float, float]=(1.5, 3.0)) -> List[Dict[str, Any]]:
    """
    POST screenshot to the FastAPI server. Returns list of detections:
      [{"box":[x1,y1,x2,y2],"score":float,"label":int}, ...]
    """
    jpg = encode_jpeg_bgr(image_bgr)
    files = {"file": ("screen.jpg", jpg, "image/jpeg")}
    r = session.post(API_URL, files=files, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("detections", [])

def choose_box(dets: List[Dict[str, Any]], conf_thresh: float, min_area: int) -> Optional[Dict[str, Any]]:
    """Pick the highest-confidence box that passes basic filters."""
    best = None
    best_score = -1.0
    for d in dets:
        score = float(d.get("score", 0.0))
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(float, d.get("box", [0, 0, 0, 0]))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue
        if score > best_score:
            best = d
            best_score = score
    return best

def perform_click(box: Dict[str, Any]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box["box"]
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    # Move + click (sync to reduce race conditions)
    subprocess.run(["xdotool", "mousemove", "--sync", str(cx), str(cy)], check=False)
    subprocess.run(["xdotool", "click", "1"], check=False)
    return cx, cy

def draw_debug(image_bgr: np.ndarray, box: Dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box["box"])
    out = image_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    txt = f"id={box.get('label')} conf={box.get('score'):.2f}"
    cv2.putText(out, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return out

def main():
    # Optional: launch Firefox for a simple manual test page
    subprocess.Popen(["firefox-esr", "--no-sandbox"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

    # HTTP keep-alive for lower latency
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    print(f"[agent] Using API_URL={API_URL}")
    lat_hist: List[float] = []

    try:
        while True:
            t0 = time.perf_counter()
            img = capture_screen()
            t1 = time.perf_counter()

            try:
                dets = call_rtdetr_api(session, img)
            except Exception as e:
                print(f"[agent] API error: {e}")
                time.sleep(SLEEP_BETWEEN_LOOPS)
                continue

            t2 = time.perf_counter()

            chosen = choose_box(dets, CONF_THRESH, MIN_BOX_AREA)
            if chosen:
                if CLICK_ENABLED:
                    cx, cy = perform_click(chosen)
                    print(f"[agent] Clicked at ({cx},{cy})")
                else:
                    print("[agent] CLICK_DISABLED; would click:", chosen)

                if SAVE_DEBUG:
                    dbg = draw_debug(img, chosen)
                    outp = f"/tmp/last_detection_{int(time.time())}.jpg"
                    cv2.imwrite(outp, dbg)

            loop_latency_ms = (time.perf_counter() - t0) * 1000.0
            capture_ms = (t1 - t0) * 1000.0
            net_infer_ms = (t2 - t1) * 1000.0
            lat_hist.append(loop_latency_ms)
            if len(lat_hist) > 60:
                lat_hist.pop(0)
            avg_ms = sum(lat_hist) / len(lat_hist)

            print(f"[agent] capture={capture_ms:.1f} ms, net+infer={net_infer_ms:.1f} ms, loop={loop_latency_ms:.1f} ms (avg={avg_ms:.1f} ms)")

            time.sleep(SLEEP_BETWEEN_LOOPS)

    except KeyboardInterrupt:
        print("[agent] Exiting.")

if __name__ == "__main__":
    main()
