# CUDA 12.6 stack for Blackwell
FROM paddlepaddle/paddle:3.1.0-gpu-cuda12.6-cudnn9.5

WORKDIR /opt/ocr

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /opt/ocr/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /opt/ocr/app

# Default config (overridden by env at runtime)
ENV OCR_BACKEND=ppocr \
    PP_OCR_LANG=en \
    PP_OCR_ENABLE_HPI=0 \
    PP_OCR_MODEL=ppocrv5_server \
    PP_OCR_TEXT_SCORE_THRESH=0.45 \
    PP_OCR_DET_SIDE=1280 \
    UVICORN_WORKERS=2

EXPOSE 8020
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "2"]
