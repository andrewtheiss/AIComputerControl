ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

WORKDIR /app

RUN python3 -m pip install --no-cache-dir \
    "transformers==4.45.0" \
    "accelerate==0.34.1" \
    "sentencepiece==0.2.0"

COPY realModels/aria-ui/app.py /app/app.py

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
