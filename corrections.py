"""
Manager correction mechanism.
Corrections are saved to corrections.json and used in future analyses.
"""
import json
import os
from datetime import datetime

CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), "corrections.json")


def load_corrections() -> list:
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_correction(correction_type: str, original: str, corrected: str, context: str = ""):
    corrections = load_corrections()
    corrections.append({
        "type":      correction_type,   # product / branch / speaker / transcript
        "original":  original,
        "corrected": corrected,
        "context":   context,
        "timestamp": datetime.now().isoformat(),
    })
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def get_corrections_as_context() -> str:
    """Return saved corrections as a text block to inject into prompts."""
    corrections = load_corrections()
    if not corrections:
        return ""
    lines = ["**תיקונים ידועים ממנהלים:**"]
    for c in corrections[-30:]:  # last 30 only
        lines.append(f"- [{c['type']}] '{c['original']}' → '{c['corrected']}'")
    return "\n".join(lines)
