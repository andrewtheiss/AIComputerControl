# task_planner/tools.py
from typing import List

def open_url(url: str) -> str:
    """
    Open a URL in a new browser window.

    Args:
      url: Absolute URL, e.g. "https://www.mail.com/".

    Returns:
      Success/failure is reported by the executor; planner should assume success
      only after subsequent waits confirm target text is visible.
    """
    ...

def wait_text(regex: str, timeout_s: int = 20) -> str:
    """
    Wait until any OCR word on screen matches the given case-insensitive regex.

    Args:
      regex: A safe, anchored or unanchored regex. Prefer short, distinctive tokens
             like "Inbox|Sign in|Log in|Reply".
      timeout_s: Max seconds to wait before failing.

    Notes:
      - Use this after navigation or form submit to confirm the new state.
      - If the UI uses synonyms, prefer wait_any_text().
    """
    ...

def wait_any_text(patterns: List[str], timeout_s: int = 20) -> str:
    """
    Wait until ANY of patterns appear.
    Parameters: patterns (alias: texts), timeout_s (alias: timeout).

    Args:
      patterns: A list of regex strings. Example: ["Inbox", "Mailbox", "New mail"].
      timeout_s: Max seconds to wait.

    Notes:
      - Use for synonymy and i18n variants (e.g., "Sign in|Log in|Anmelden").
    """
    ...

def click_text(regex: str, nth: int = 0, prefer_bold: bool = False) -> str:
    """
    Click element whose OCR matches regex.
    Parameters: regex (alias: text), nth, prefer_bold.

    Args:
      regex: Case-insensitive regex for the visible text.
      nth: If multiple matches, choose the nth (0-based) sorted by score.
      prefer_bold: If True, slightly prefer taller/bolder text (buttons/nav).

    Notes:
      - If you have multiple candidate strings, prefer click_any_text().
      - Use concise regex without leading/trailing whitespace.
    """
    ...

def click_any_text(patterns: List[str], nth: int = 0, prefer_bold: bool = True) -> str:
    """
    Click first match among patterns.
    Parameters: patterns (alias: texts), nth, prefer_bold.

    Args:
      patterns: Ordered list of regex choices, highest priority first.
      nth: If the chosen regex has multiple matches, pick the nth.
      prefer_bold: Prefer bold/taller words (good for primary buttons).

    Notes:
      - Ideal for "Sign in|Log in|Continue" flows.
    """
    ...

def click_near_text(anchor_regex: str, dx: int = 0, dy: int = 0) -> str:
    """
    Click at an offset from the center of the FIRST OCR word matching anchor_regex.
    Parameters: anchor_regex (alias: anchor), dx, dy.
    Guidance:
      - Use field labels as anchors (e.g., '^To$','^Recipients$','^Subject$').
      - DO NOT anchor on the account email (contains '@') or app header chrome.
    Args:
      anchor_regex: Regex to locate the anchor word/label.
      dx: Horizontal offset in pixels (positive = right).
      dy: Vertical offset in pixels (positive = down).

    Constraints:
      - Use only when the clickable icon has no text but is directly adjacent
        (e.g., a reply icon 24–48 px to the right of the word "Reply").
      - Keep |dx|,|dy| <= 120 unless absolutely necessary.
    """
    ...

def type_text(text: str, confidential: bool = False) -> str:
    """
    Type text into the currently focused field.

    Args:
      text: Raw text. Supports ${ENV_VAR} substitution by the executor.
      confidential: If True, the executor will redact the value in logs.

    Notes:
      - Planner should first focus the target field (via click_text) or tab to it.
      - For passwords, set confidential=True.
    """
    ...

def key_seq(keys: List[str]) -> str:
    """
    Send a sequence of keys (xdotool names), e.g., ["Tab","Enter"].

    Args:
      keys: Valid xdotool key names (Tab, Enter, BackSpace, Ctrl+l, etc.).

    Notes:
      - Use to navigate forms, submit (Enter), or select fields.
    """
    ...

def sleep(seconds: float = 0.8) -> str:
    """
    Idle for a short period.
    Parameters: seconds.

    Args:
      seconds: Fractional seconds to sleep.

    Notes:
      - Use to debounce dynamic UIs between actions.
    """
    ...

def ocr_extract(save_as: str) -> str:
    """
    Save the concatenated OCR text of the current screen to a context variable.

    Args:
      save_as: Name of the context var (e.g., "email_page_text").

    Notes:
      - The executor stores text in its local ctx for downstream rules.
    """
    ...

def end_task(reason: str) -> str:
    """
    Finish the task.

    Args:
      reason: Short explanation why the goal is complete or cannot proceed.
    """
    ...

def run_llm(system: str, prompt: str, var_out: str = "draft") -> str:
    """
    Ask the executor to call its LLM and store the result in a context variable.

    Args:
      system: System prompt.
      prompt: User prompt.
      var_out: Context var name (e.g., "draft").
    """
    ...
 

ALL_TOOLS = [
    open_url,
    wait_text,
    wait_any_text,
    click_text,
    click_any_text,
    click_near_text,
    type_text,
    key_seq,
    sleep,
    ocr_extract,
    end_task,
    run_llm,
]
