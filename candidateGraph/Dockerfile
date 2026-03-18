FROM python:3.12-slim

WORKDIR /app

COPY candidateGraph/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY candidateGraph /app/candidateGraph
COPY ui_vision_common /app/ui_vision_common

ENV PYTHONPATH=/app \
    DEBUG_ARTIFACT_DIR=/tmp/candidate-graph-debug

EXPOSE 8000
CMD ["uvicorn", "candidateGraph.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
