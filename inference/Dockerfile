# Dockerfile
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Work in a stable path and create a non-root user
WORKDIR /opt/app
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /opt/app

# Python deps (TensorRT already in base image)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -rf /root/.cache/pip
# opencv's recent wheels pull numpy>=2; pin numpy<2 to avoid ABI hiccups in some bases
RUN pip install --no-cache-dir 'numpy<2.0'

# Convenience symlink so /app works in code and docs
RUN ln -s /opt/app /app

# Copy code and model
USER appuser
COPY --chown=appuser:appuser app/ /opt/app/app/
COPY --chown=appuser:appuser models/ /opt/app/models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
