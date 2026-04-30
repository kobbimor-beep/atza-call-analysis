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
    """Return manager corrections as a text block to inject into the Claude prompt.
    Pulls from Supabase feedback table first, falls back to local JSON."""
    lines = ["**תיקונים והנחיות ממנהלים (חובה להתחשב בהם בניתוח):**"]
    found = False

    # Primary: Supabase feedback table
    try:
        import db
        if db.is_configured():
            rows = db.load_feedback_corrections(limit=50)
            for r in rows:
                field   = r.get("field_corrected") or ""
                orig    = r.get("original_value") or ""
                corr    = r.get("corrected_value") or ""
                notes   = r.get("notes") or ""
                ftype   = r.get("feedback_type", "correction")
                if ftype == "correction" and orig and corr:
                    label = f"[{field}] " if field else ""
                    lines.append(f"- {label}'{orig}' → '{corr}'" + (f" ({notes})" if notes else ""))
                    found = True
                elif ftype == "comment" and notes:
                    lines.append(f"- הערת מנהל: {notes}")
                    found = True
    except Exception:
        pass

    # Fallback: local corrections.json
    local = load_corrections()
    for c in local[-20:]:
        lines.append(f"- [{c['type']}] '{c['original']}' → '{c['corrected']}'")
        found = True

    return "\n".join(lines) if found else ""
