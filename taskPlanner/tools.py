# task_planner/tools.py
from typing import List

def open_url(url: str) -> str:
    """Open a URL in a new browser window. Args: url (string)."""
    ...

def wait_text(regex: str, timeout_s: int = 20) -> str:
    """Wait until text matching regex appears. Case-insensitive."""
    ...

def click_text(regex: str, nth: int = 0, prefer_bold: bool = False) -> str:
    """Click UI element with text matching regex. Prefer bold for nav/buttons if True."""
    ...

def type_text(text: str, confidential: bool = False) -> str:
    """Type text in the focused field. Set confidential=True to avoid logging secrets."""
    ...

def key_seq(keys: List[str]) -> str:
    """Send key sequence like ["Tab","Enter"]."""
    ...

def ocr_extract(save_as: str) -> str:
    """Save full-screen OCR text into a context var name 'save_as'."""
    ...

def end_task(reason: str) -> str:
    """Finish when goal is complete or impossible. Explain why in 'reason'."""
    ...

ALL_TOOLS = [open_url, wait_text, click_text, type_text, key_seq, ocr_extract, end_task]
