#!/usr/bin/env bash
set -euo pipefail

OMNIPARSER_REPO_DIR="${OMNIPARSER_REPO_DIR:-/opt/OmniParser}"
OMNIPARSER_WEIGHTS_DIR="${OMNIPARSER_WEIGHTS_DIR:-${OMNIPARSER_REPO_DIR}/weights}"
OMNIPARSER_DEVICE="${OMNIPARSER_DEVICE:-cuda}"
OMNIPARSER_HOST="${OMNIPARSER_HOST:-0.0.0.0}"
OMNIPARSER_PORT="${OMNIPARSER_PORT:-8000}"
OMNIPARSER_BOX_THRESHOLD="${OMNIPARSER_BOX_THRESHOLD:-0.05}"
OMNIPARSER_MODEL_ID="${OMNIPARSER_MODEL_ID:-microsoft/OmniParser-v2.0}"

mkdir -p "${OMNIPARSER_WEIGHTS_DIR}"

if [ ! -f "${OMNIPARSER_WEIGHTS_DIR}/icon_detect/model.pt" ] || [ ! -d "${OMNIPARSER_WEIGHTS_DIR}/icon_caption_florence" ]; then
  rm -rf \
    "${OMNIPARSER_WEIGHTS_DIR}/icon_detect" \
    "${OMNIPARSER_WEIGHTS_DIR}/icon_caption" \
    "${OMNIPARSER_WEIGHTS_DIR}/icon_caption_florence"

  python - <<PY
from huggingface_hub import snapshot_download

repo_id = "${OMNIPARSER_MODEL_ID}"
local_dir = "${OMNIPARSER_WEIGHTS_DIR}"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=["icon_caption/*"],
)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=["icon_detect/*"],
)
PY

  if [ -d "${OMNIPARSER_WEIGHTS_DIR}/icon_caption" ] && [ ! -d "${OMNIPARSER_WEIGHTS_DIR}/icon_caption_florence" ]; then
    mv "${OMNIPARSER_WEIGHTS_DIR}/icon_caption" "${OMNIPARSER_WEIGHTS_DIR}/icon_caption_florence"
  fi
fi

NEEDS_FLORENCE_PIN="$(python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

def get_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""

transformers_version = get_version("transformers")
tokenizers_version = get_version("tokenizers")

if transformers_version != "4.49.0":
    print("1")
elif tokenizers_version:
    major_minor = tuple(int(part) for part in tokenizers_version.split(".")[:2])
    print("1" if major_minor >= (0, 22) else "0")
else:
    print("1")
PY
)"

if [ "${NEEDS_FLORENCE_PIN}" = "1" ]; then
  pip install --no-cache-dir "transformers==4.49.0" "tokenizers<0.22"
fi

cd "${OMNIPARSER_REPO_DIR}/omnitool/omniparserserver"

python - <<PY
from pathlib import Path

utils_path = Path("${OMNIPARSER_REPO_DIR}/util/utils.py")
text = utils_path.read_text()
old = """paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)"""
new = "paddle_ocr = None  # hotfix: avoid eager PaddleOCR init on incompatible package versions"
if old in text:
    utils_path.write_text(text.replace(old, new, 1))
PY

exec python -m omniparserserver \
  --som_model_path "${OMNIPARSER_WEIGHTS_DIR}/icon_detect/model.pt" \
  --caption_model_name florence2 \
  --caption_model_path "${OMNIPARSER_WEIGHTS_DIR}/icon_caption_florence" \
  --device "${OMNIPARSER_DEVICE}" \
  --BOX_TRESHOLD "${OMNIPARSER_BOX_THRESHOLD}" \
  --host "${OMNIPARSER_HOST}" \
  --port "${OMNIPARSER_PORT}"
