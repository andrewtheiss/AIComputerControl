# task_planner/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# You rarely need build-essential here; keep slim unless you add compiled deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install taskgen-ai --upgrade

COPY . .

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]

