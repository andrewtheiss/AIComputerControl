import base64, io, os
import cv2, numpy as np
from paddleocr import PaddleOCR

def make_ppocr():
    lang         = os.getenv("PP_OCR_LANG", "en")
    enable_hpi   = os.getenv("PP_OCR_ENABLE_HPI", "0") == "1"
    model_tag    = os.getenv("PP_OCR_MODEL", "ppocrv5_server")   # server/mobile
    det_side_len = int(os.getenv("PP_OCR_DET_SIDE", "1280"))
    device       = os.getenv("PP_OCR_DEVICE", "gpu:0")
    orient_mode  = os.getenv("PP_OCR_ORIENTATION", "textline").strip().lower()
    # enforce exclusivity by passing only one flag to PaddleOCR
    use_angle_cls = (orient_mode == "angle_cls")
    use_textline_orientation = (orient_mode == "textline")

    # PaddleOCR 3.x: default pipeline is PP-OCRv5_server; enable_hpi toggles high-perf inference
    # HPI on CUDA12.6 uses ORT/OpenVINO; CUDA11.8 can use TRT 8.6.1.6 per docs.
    # (We don't hardcode model URLs; PaddleOCR manages downloads/caching.)
    kwargs = dict(
        lang=lang,
        enable_hpi=enable_hpi,
        # Text det preprocess tuning for high‑DPI screens
        text_det_limit_type="min",
        text_det_limit_side_len=det_side_len,
        device=device,
    )
    if use_angle_cls:
        kwargs["use_angle_cls"] = True
    elif use_textline_orientation:
        kwargs["use_textline_orientation"] = True

    try:
        ocr = PaddleOCR(**kwargs)
    except Exception as e:
        # Graceful fallback if HPI plugin is unavailable
        if enable_hpi:
            print("[ppocr] HPI unavailable, retrying with enable_hpi=0: " + str(e), flush=True)
            kwargs["enable_hpi"] = False
            ocr = PaddleOCR(**kwargs)
        else:
            raise
    return ocr

class PPOCREngine:
    def __init__(self):
        self.ocr = make_ppocr()

    def infer(self, bgr_img: np.ndarray):
        # PaddleOCR 3.x unified call:
        # returns a dict with dt_polys/rec_texts/rec_scores etc. (pipeline API)
        orient_mode  = os.getenv("PP_OCR_ORIENTATION", "textline").strip().lower()
        cls_flag = (orient_mode == "angle_cls")
        try:
            return self.ocr.ocr(bgr_img, cls=False)  # 2.x style
        except TypeError:
            return self.ocr.ocr(bgr_img)             # 3.x style
        return result
