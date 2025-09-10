# task_planner/main.py
import os, json
import orjson
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import AsyncOpenAI

from taskgen import Agent, Function   # documented usage
from tools import ALL_TOOLS

load_dotenv()

# --- Ollama (OpenAI-compatible) client ---
OLLAMA_OPENAI_BASE = os.getenv("OLLAMA_OPENAI_BASE", "http://host.docker.internal:11434/v1")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_DUMMY_KEY   = os.getenv("OPENAI_API_KEY", "ollama")  # not used by Ollama, but OpenAI client requires it
client = AsyncOpenAI(base_url=OLLAMA_OPENAI_BASE, api_key=OPENAI_DUMMY_KEY)

PLANNER_API_KEY = os.getenv("PLANNER_API_KEY", "")

# --- Security (simple bearer) ---
def verify_key(authorization: Optional[str] = Header(None)):
    if not PLANNER_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1] != PLANNER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# --- Pydantic models for strict contract ---
class OCRResult(BaseModel):
    text: str
    box: List[int]
    conf: float

class HistoryItem(BaseModel):
    action: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]

class PlannerRequest(BaseModel):
    goal: str
    task_history: List[HistoryItem] = Field(default_factory=list)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)

class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    parameters: Dict[str, Any]
    reasoning: str
    completed: bool = False

# --- TaskGen agent setup ---
# LLM wrapper with TaskGen’s required signature
async def tg_llm(user_prompt: str, system_prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.2,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content

planner_agent = Agent(
    name="UI Task Planner",
    description="Decide the next valid UI action for a high-level goal given OCR and history.",
    llm=tg_llm
)
# Inform the LLM of the tool schema via TaskGen (names + docstrings matter)
planner_agent.assign_functions([Function(external_fn=fn) for fn in ALL_TOOLS])

SYSTEM_PROMPT = """You are a precise planner for a UI automation agent.
You must output STRICT JSON ONLY with keys: action, parameters, reasoning, completed.
- action: one of the provided tool names.
- parameters: dict of args for that tool.
- reasoning: 1-3 sentences explaining the choice using GOAL, OCR and last action result.
- completed: true only when the full GOAL is done or impossible.

Rules:
- Use only the actions announced by the caller.
- If a previous action failed (see last history.result), try a different strategy (wait, alternate regex, etc.).
- Never reference UI elements not present in OCR.
- Prefer generic auth patterns: find Login/Sign in, then username/email then password then submit, then wait for Inbox.
"""

def make_user_prompt(req: PlannerRequest) -> str:
    hist = "\n".join(
        f"- {h.action}({h.parameters}) => {h.result.get('status')}"
        for h in req.task_history[-5:]
    ) or "No prior actions."
    ocr = "\n".join(f"- {r.text} @ {r.box} (conf={int(r.conf)})" for r in req.ocr_results[:300]) or "No OCR text."
    actions = ", ".join(req.available_actions) or "[]"
    return (
        f"GOAL:\n{req.goal}\n\n"
        f"AVAILABLE_ACTIONS: {actions}\n\n"
        f"RECENT_HISTORY:\n{hist}\n\n"
        f"CURRENT_OCR:\n{ocr}\n\n"
        f"Return strict JSON only."
    )

app = FastAPI(title="TaskPlanner API", version="1.0.0")

@retry(wait=wait_exponential(min=0.5, max=4), stop=stop_after_attempt(3))
async def _decide(req: PlannerRequest) -> PlannerResponse:
    user_prompt = make_user_prompt(req)
    out = await planner_agent.llm(user_prompt, SYSTEM_PROMPT)

    # Attempt strict JSON parse; if not JSON, try to extract a JSON object
    try:
        obj = json.loads(out)
    except json.JSONDecodeError:
        start = out.find("{"); end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(out[start:end+1])
        else:
            obj = {"action":"end_task",
                   "parameters":{"reason":"LLM did not return JSON"},
                   "reasoning":"Parser fallback",
                   "completed": True}

    # Validate action
    if obj.get("action") not in req.available_actions:
        obj = {"action":"end_task",
               "parameters":{"reason":f"Invalid action {obj.get('action')}"},
               "reasoning":"Planner filtered to safe action set",
               "completed": True}

    return PlannerResponse(**obj)

@app.post("/v1/actions/next", response_model=PlannerResponse, dependencies=[Depends(verify_key)])
async def next_action(req: PlannerRequest):
    try:
        return await _decide(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
