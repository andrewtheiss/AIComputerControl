# Rebuild docker
docker build -t ai-sandbox .

# Run manually outside of dockerconainer
docker run -d --gpus 'device=0' --name vnc-instance-1 --network ai-net \
  -p 5901:5901 \
  -v $(pwd)/tasks:/tasks \
  -e AGENT_NAME="agent-1" \
  -e AGENT_TASK="/tasks/agent-1.yaml" \
  -e RTDETR_API_URL="http://rtdetr-api:8000/predict" \
  ai-sandbox




  # Setup local cache for agent.py
  # Step 1:
  python3 - <<'PY'
import os, configparser, json
ini = os.path.expanduser('~/snap/firefox/common/.mozilla/firefox/profiles.ini')
cfg = configparser.ConfigParser()
cfg.read(ini)
out = {}
for s in cfg.sections():
    if cfg.get(s,'Default',fallback='0')=='1':
        path = cfg.get(s,'Path')
        rel  = cfg.getboolean(s,'IsRelative',fallback=True)
        prof = os.path.join(os.path.expanduser('~/snap/firefox/common/.mozilla/firefox'), path) if rel else path
        cache = prof.replace('/.mozilla/firefox/', '/.cache/mozilla/firefox/')
        out['PROFILE_DIR'] = prof
        out['CACHE_DIR']   = cache
        break
print(json.dumps(out, indent=2))
PY

## Step 2:
2) Mount the host profile+cache into the container


# Rebuilding this container:
docker compose stop ocr-api
docker compose build --no-cache ocr-api
docker compose up -d --force-recreate --no-deps ocr-api
docker compose logs -f ocr-api
