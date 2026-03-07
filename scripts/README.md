## Planner replay scripts

### `planner_replay.py`

Replays a single `POST /v1/actions/next` request against the planner, optionally embedding a screenshot.

Example (from host, using the exposed port):

```bash
python3 scripts/planner_replay.py \
  --url "http://localhost:28000/v1/actions/next" \
  --api-key "taskPlannerApiSecret" \
  --payload /path/to/payload.json \
  --screenshot /path/to/screen.jpg
```

Notes:
- `--screenshot` embeds the image as base64; the planner will then attempt a multimodal LLM call.
- If your backend prefers a `data:<mime>;base64,...` URL, add `--as-data-url`.

