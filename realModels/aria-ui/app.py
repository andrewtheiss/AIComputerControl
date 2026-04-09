from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, Tuple

from fastapi import FastAPI, HTTPException
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name, default) or default).strip()


MODEL_ID = _env("MODEL_ID", "Aria-UI/Aria-UI-base")
SERVED_MODEL_NAME = _env("SERVED_MODEL_NAME", "aria-ui")
MODEL_DTYPE = _env("MODEL_DTYPE", "bfloat16")
MAX_TOKENS = int(_env("MAX_TOKENS", "64") or "64")
IMAGE_MAX_SIZE = int(_env("IMAGE_MAX_SIZE", "980") or "980")
SPLIT_IMAGE = _env("SPLIT_IMAGE", "true").lower() not in {"0", "false", "no"}


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(name.strip().lower(), torch.bfloat16)


TORCH_DTYPE = _torch_dtype(MODEL_DTYPE)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

app = FastAPI(title="Aria-UI Runtime")


def _decode_image(data_url: str) -> Image.Image:
    payload = str(data_url or "")
    if "base64," in payload:
        payload = payload.split("base64,", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_prompt_and_image(messages: list[dict[str, Any]]) -> Tuple[str, Image.Image]:
    if not messages:
        raise HTTPException(status_code=400, detail="messages are required")
    content_items = list((messages[-1] or {}).get("content") or [])
    prompt = ""
    image: Image.Image | None = None
    for item in content_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            prompt = str(item.get("text", "") or "").strip()
        elif item.get("type") == "image_url":
            image_url = dict(item.get("image_url") or {}).get("url")
            image = _decode_image(str(image_url or ""))
    if not prompt or image is None:
        raise HTTPException(status_code=400, detail="Provide text and image_url content")
    return prompt, image


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_id": SERVED_MODEL_NAME, "backend": "aria_ui_runtime"}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt, image = _extract_prompt_and_image(list(payload.get("messages") or []))
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(TORCH_DTYPE)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        if str(model.device).startswith("cuda"):
            with torch.amp.autocast("cuda", dtype=TORCH_DTYPE):
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(payload.get("max_tokens") or MAX_TOKENS),
                    do_sample=False,
                )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=int(payload.get("max_tokens") or MAX_TOKENS),
                do_sample=False,
            )

    output_ids = output[0][inputs["input_ids"].shape[1] :]
    content = processor.decode(output_ids, skip_special_tokens=True)
    for stop in list(payload.get("stop") or ["<|im_end|>"]):
        if stop:
            content = content.split(str(stop), 1)[0]
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(payload.get("model") or SERVED_MODEL_NAME),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content.strip(),
                },
            }
        ],
    }
