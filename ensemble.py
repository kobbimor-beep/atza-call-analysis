import anthropic
import json
import os

from transcriber import transcribe_call
from whisper_transcriber import transcribe_with_whisper

CONFIDENCE_THRESHOLD = 0.80


def _merge_with_claude(aai_result: dict, whisper_text: str) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    aai_text = "\n".join(
        f"[{u['speaker']}]: {u['text']}"
        for u in aai_result["utterances"]
    )

    prompt = f"""יש לך שני תמלולים של אותה שיחת מוקד בעברית.

תמלול 1 — AssemblyAI (עם זיהוי דוברים נציג/לקוח):
{aai_text}

תמלול 2 — Whisper (ללא זיהוי דוברים, טקסט רצוף):
{whisper_text}

המשימה:
- השווה בין השניים ומצא מילים או משפטים שAssemblyAI טעה בהם
- תקן את תמלול AssemblyAI לפי Whisper היכן שWhisper נראה יותר הגיוני
- שמור בדיוק על מבנה הדוברים (נציג/לקוח) מתמלול 1

החזר JSON בדיוק בפורמט הזה (ללא טקסט נוסף):
{{
  "utterances": [
    {{"speaker": "נציג או לקוח", "text": "הטקסט המתוקן"}}
  ],
  "corrections_made": 3,
  "confidence_boost": 0.08,
  "notes": "תיאור קצר של ההבדלים העיקריים שנמצאו"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    merged = json.loads(raw)

    # Rebuild utterances preserving original timing/confidence fields
    original = {i: u for i, u in enumerate(aai_result["utterances"])}
    merged_utterances = []
    for i, u in enumerate(merged.get("utterances", [])):
        orig = original.get(i, {})
        merged_utterances.append({
            "speaker":    u.get("speaker", orig.get("speaker", "נציג")),
            "text":       u.get("text", orig.get("text", "")),
            "start_ms":   orig.get("start_ms", 0),
            "end_ms":     orig.get("end_ms", 0),
            "confidence": orig.get("confidence", 0.0),
        })

    result = aai_result.copy()
    result["utterances"]              = merged_utterances
    result["used_ensemble"]           = True
    result["ensemble_corrections"]    = merged.get("corrections_made", 0)
    result["ensemble_notes"]          = merged.get("notes", "")
    result["transcription_confidence"] = min(
        1.0,
        aai_result["transcription_confidence"] + merged.get("confidence_boost", 0.0)
    )
    result["transcription_quality"]   = "גבוהה (Ensemble)"

    return result


def transcribe_ensemble(audio_path: str) -> tuple[dict, list[str]]:
    """
    Returns (transcript_data, steps) where steps is a list of
    human-readable status messages for the UI.
    """
    steps = []

    steps.append("📤 מעלה קובץ ומתמלל עם AssemblyAI...")
    result = transcribe_call(audio_path)
    conf   = result["transcription_confidence"]
    steps.append(
        f"✅ AssemblyAI הושלם — {int(result['duration_seconds'])} שניות, "
        f"דיוק {int(conf * 100)}%"
    )

    if conf < CONFIDENCE_THRESHOLD:
        steps.append(f"⚠️ דיוק נמוך ({int(conf*100)}%) — מפעיל Whisper לאימות...")
        whisper_text = transcribe_with_whisper(audio_path)
        steps.append("✅ Whisper הושלם — ממזג עם Claude...")
        result = _merge_with_claude(result, whisper_text)
        steps.append(
            f"✅ מיזוג הושלם — {result['ensemble_corrections']} תיקונים, "
            f"דיוק סופי {int(result['transcription_confidence']*100)}%"
        )
    else:
        steps.append(f"✅ דיוק גבוה ({int(conf*100)}%) — Whisper לא נדרש")
        result["used_ensemble"] = False

    return result, steps
