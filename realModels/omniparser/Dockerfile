FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    OMNIPARSER_REPO_DIR=/opt/OmniParser

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/microsoft/OmniParser.git "${OMNIPARSER_REPO_DIR}"

COPY realModels/omniparser/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY realModels/omniparser/start.sh /usr/local/bin/start-omniparser
RUN chmod +x /usr/local/bin/start-omniparser

EXPOSE 8000
CMD ["start-omniparser"]
