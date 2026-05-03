import assemblyai as aai
import os
import re


# Atza-specific vocabulary to boost recognition accuracy
ATZA_WORD_BOOST = [
    # Brand / general
    "אצה", "סושי", "משלוח", "איסוף",
    # Branches
    "עכו", "חיפה", "נהריה", "קריית אתא", "קריות", "חדרה", "נתניה",
    # Rolls
    "ביסלי", "סלופי", "ג'מבו", "ברוקלין", "קארי", "ספיידר", "קליפורניה",
    "אאוט קראנץ'", "קראנץ'", "ווג'י", "טמפורה", "ספייסי", "טאקה",
    "יאמאזקי", "מאנגה", "אצה רול", "סלמון גריל",
    # Nigiri / sashimi
    "ניגירי", "סאשימי", "סלמון", "טונה", "אבוקדו",
    # Sandwiches
    "סנדוויץ'", "קריספי", "מעושן",
    # Noodles
    "פאד תאי", "סמוקי", "צ'ופ סואי", "נודלס", "קוקוס", "אסאדו",
    # Rice
    "מוקפץ", "בקר", "קשיו",
    # Snacks
    "אדממה", "גיוזה", "כרובית", "פנקו", "כנפיים",
    # Sauces
    "טריאקי", "יוזו", "ספייסי מיונז",
    # Misc
    "צ'ופסטיקס", "קומבינאצה", "פוקי",
]


def _clean_utterance(text: str) -> str:
    """Remove repeated phrases caused by AssemblyAI hallucination loops."""
    # Split into sentences/clauses
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, cleaned = [], []
    for part in parts:
        key = part.strip().lower()
        if key not in seen:
            seen.append(key)
            cleaned.append(part)
    result = " ".join(cleaned)

    # Also deduplicate repeated short tokens (e.g. "כן. כן. כן.")
    result = re.sub(r'\b(.{2,30}?)\b(?:\s*[.,]?\s*\1){3,}', r'\1', result)
    return result.strip()


def _avg_confidence(words) -> float:
    if not words:
        return 0.0
    scores = [w.confidence for w in words if hasattr(w, "confidence") and w.confidence is not None]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "גבוהה"
    if score >= 0.70:
        return "בינונית"
    return "נמוכה"


# Phrases that strongly indicate the speaker is the call-center agent
_AGENT_OPENERS = [
    "אצה", "שלום אצה", "אצה שלום", "אצה סושי",
    "במה אפשר", "מה אפשר", "אפשר לעזור", "איך אפשר",
    "הזמנות", "מוקד", "שירות לקוחות",
]


def _infer_agent_speaker(utterances_raw) -> tuple[str, float]:
    """
    Return (speaker_label, confidence) for the agent.
    Checks first two utterances for known agent opening phrases.
    Falls back to first-speaker convention (agents pick up the phone first).
    """
    if not utterances_raw:
        return "A", 0.50

    for i, u in enumerate(utterances_raw[:2]):
        text_lower = (u.text or "").lower()
        for phrase in _AGENT_OPENERS:
            if phrase in text_lower:
                return u.speaker, 0.93

    # Default: first speaker = agent (call-center convention)
    return utterances_raw[0].speaker, 0.72


def transcribe_call(audio_path: str) -> dict:
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    config = aai.TranscriptionConfig(
        language_code="he",
        speech_model=aai.SpeechModel.slam_1,
        speaker_labels=True,
        sentiment_analysis=False,
        disfluencies=False,
        word_boost=ATZA_WORD_BOOST,
        boost_param="high",
    )

    transcriber = aai.Transcriber()
    transcript  = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"שגיאת תמלול: {transcript.error}")

    agent_label, speaker_conf = _infer_agent_speaker(transcript.utterances or [])

    utterances = []
    for u in (transcript.utterances or []):
        cleaned = _clean_utterance(u.text)
        conf    = round(u.confidence, 3) if u.confidence else 0.0
        if u.speaker == agent_label:
            speaker = "נציג"
        elif u.speaker in ("A", "B", "C", "D"):
            speaker = "לקוח"
        else:
            speaker = "לא ידוע"
        utterances.append({
            "speaker":     speaker,
            "speaker_raw": u.speaker,
            "text":        cleaned,
            "start_ms":    u.start,
            "end_ms":      u.end,
            "confidence":  conf,
        })

    avg_conf = _avg_confidence(transcript.words or [])
    duration = transcript.audio_duration or 0

    return {
        "full_text":                transcript.text or "",
        "utterances":               utterances,
        "sentiments":               [],
        "duration_seconds":         duration,
        "words_count":              len(transcript.words or []),
        "transcription_confidence": avg_conf,
        "transcription_quality":    _confidence_label(avg_conf),
        "speaker_confidence":       round(speaker_conf, 2),
    }
