FROM python:3.12-slim

WORKDIR /app

COPY modelService/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY modelService /app/modelService
COPY ui_vision_common /app/ui_vision_common

ENV PYTHONPATH=/app \
    MODEL_ID=omniparser \
    MODEL_ROLE=ocr \
    MODEL_BACKEND=mock \
    DEBUG_ARTIFACT_DIR=/tmp/ui-model-debug

EXPOSE 8000
CMD ["uvicorn", "modelService.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
