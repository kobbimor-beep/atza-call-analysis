import anthropic
import json
import os
from menu import get_menu_context, normalize_order_items, ATZA_BRANCHES
from corrections import get_corrections_as_context

ATZA_SCRIPT_CHECKPOINTS = [
    "ברכה ופתיחת שיחה",
    "זיהוי הלקוח / בדיקת כתובת למשלוח",
    "לקיחת פרטי ההזמנה",
    "הצעת תוספות / רטבים / שדרוגים",
    "חזרה על ההזמנה ללקוח לאישור",
    "אישור זמן אספקה משוער",
    "סיום שיחה מקצועי",
]

ANALYSIS_PROMPT = """אתה מומחה לניתוח איכות שיחות במוקד שירות לרשת מסעדות סושי "אצה".

{menu_context}

{corrections_context}

{branch_context}
**סניפי אצה הקיימים:** {branches}

**תסריט חובה של הנציג:**
{checkpoints}

**נתוני שיחה:**
- משך: {duration} שניות | מילים: {words_count}

**תמלול השיחה (עם חותמות זמן):**
{transcript}

נתח את השיחה והחזר JSON בפורמט הבא (ללא טקסט נוסף, ללא markdown):

{{
  "call_type": "order/service/inquiry/failed",
  "call_type_reasoning": "הסבר קצר מדוע סווגה השיחה כך",
  "missed_conversion": true/false,
  "missed_conversion_reason": "מה הנציג היה יכול לעשות אחרת כדי להמיר להזמנה",
  "early_transfer": true/false,
  "improper_disconnect": true/false,
  "branch_detection": {{
    "branch_name": "שם הסניף שזוהה, או null",
    "confidence": "גבוהה/בינונית/נמוכה/לא זוהה",
    "evidence": "ציטוט מהתמלול שממנו הוסק הסניף",
    "requires_manual_review": true/false
  }},
  "speaker_analysis": {{
    "agent_is_speaker_a": true/false,
    "confidence": "גבוהה/בינונית/נמוכה",
    "reasoning": "מדוע הוחלט מי הנציג ומי הלקוח"
  }},
  "order": {{
    "raw_items": ["פריטים כפי שנאמרו בשיחה, מילה במילה"],
    "items": ["פריטים לאחר נרמול לשמות מהתפריט"],
    "sauces": ["רטבים שהוזמנו"],
    "quantity_notes": "הערות על כמויות",
    "cutlery": "צ'ופסטיקס/מזלג/לא הוזכר",
    "address": "כתובת המשלוח",
    "delivery_or_pickup": "משלוח/איסוף/לא ברור",
    "special_requests": "בקשות מיוחדות",
    "confirmed_by_agent": true/false,
    "repeated_back_to_customer": true/false,
    "missing_details": ["פרטים שלא נאספו"]
  }},
  "agent_performance": {{
    "overall_score": 0,
    "script_compliance": {{
      "ברכה ופתיחת שיחה": true/false,
      "זיהוי הלקוח / בדיקת כתובת למשלוח": true/false,
      "לקיחת פרטי ההזמנה": true/false,
      "הצעת תוספות / רטבים / שדרוגים": true/false,
      "חזרה על ההזמנה ללקוח לאישור": true/false,
      "אישור זמן אספקה משוער": true/false,
      "סיום שיחה מקצועי": true/false
    }},
    "missed_checkpoints": [],
    "professionalism_notes": "",
    "strong_points": [],
    "improvement_areas": []
  }},
  "customer_satisfaction": {{
    "is_reliable": true/false,
    "reliability_reason": "מדוע הניתוח אמין או לא",
    "overall_score": 0,
    "sentiment": "חיובי/שלילי/נייטרלי",
    "frustration_indicators": [],
    "satisfaction_indicators": [],
    "notes": ""
  }},
  "call_quality": {{
    "duration_seconds": {duration},
    "words_per_minute": 0,
    "pace_assessment": "מהיר מדי/אטי מדי/תקין",
    "clarity_score": 0,
    "notes": ""
  }},
  "dispute_analysis": {{
    "order_stated_by_customer": [],
    "order_corrections": [],
    "agent_verified_order": true/false,
    "liability_assessment": ""
  }},
  "flags": {{
    "low_transcription_quality": false,
    "unknown_products_detected": false,
    "speaker_identity_uncertain": false,
    "branch_undetected": false,
    "manual_review_required": false,
    "manual_review_reasons": []
  }}
}}"""


def _build_transcript_text(utterances: list) -> str:
    lines = []
    for u in utterances:
        start = u.get("start_ms", 0) // 1000
        end   = u.get("end_ms", 0) // 1000
        ts    = f"[{start//60:02d}:{start%60:02d}–{end//60:02d}:{end%60:02d}]"
        lines.append(f"{ts} [{u['speaker']}]: {u['text']}")
    return "\n".join(lines)


def _validate_customer_satisfaction(cust: dict) -> dict:
    """Return cust as-is if reliable, else return a safe fallback."""
    if not cust.get("is_reliable", True):
        return {
            "is_reliable": False,
            "reliability_reason": cust.get("reliability_reason", "לא זוהה בביטחון מספק"),
            "overall_score": None,
            "sentiment": "לא זוהה",
            "frustration_indicators": [],
            "satisfaction_indicators": [],
            "notes": "לא זוהה בביטחון מספק — נדרשת בדיקה ידנית",
        }
    # Basic sanity check
    score = cust.get("overall_score")
    if score is None or not isinstance(score, (int, float)) or score < 0 or score > 10:
        cust["is_reliable"] = False
        cust["notes"] = "לא זוהה בביטחון מספק — נדרשת בדיקה ידנית"
    return cust


def _call_claude(client, prompt: str, max_tokens: int = 4000) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def analyze_call(transcript_data: dict, branch_hint: dict | None = None) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    transcript_text  = _build_transcript_text(transcript_data["utterances"]) or transcript_data.get("full_text", "")
    checkpoints_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(ATZA_SCRIPT_CHECKPOINTS))

    # Build branch context from deterministic detection (if available)
    branch_context_str = ""
    if branch_hint and branch_hint.get("branch_name"):
        method_heb = {"filename": "שם קובץ", "transcript": "תמלול", "llm": "AI"}.get(
            branch_hint.get("method", ""), "זיהוי אוטומטי"
        )
        branch_context_str = (
            f"**זיהוי סניף מוקדם ({method_heb}):** {branch_hint['branch_name']} "
            f"— ביטחון {int(branch_hint.get('confidence', 0) * 100)}%\n"
            f"אנא אמת ואשר זיהוי זה בשדה branch_detection."
        )

    corrections_context = get_corrections_as_context()
    prompt = ANALYSIS_PROMPT.format(
        menu_context        = get_menu_context(),
        corrections_context = corrections_context,
        branch_context      = branch_context_str,
        branches            = ", ".join(ATZA_BRANCHES),
        checkpoints         = checkpoints_text,
        duration            = int(transcript_data["duration_seconds"]),
        words_count         = transcript_data["words_count"],
        transcript          = transcript_text,
    )

    raw = _call_claude(client, prompt)

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        # Retry once with a stricter instruction
        retry_prompt = prompt + "\n\nחשוב: החזר JSON בלבד. ללא שום טקסט לפני או אחרי."
        raw = _call_claude(client, retry_prompt)
        analysis = json.loads(raw)

    # Validate customer satisfaction
    if "customer_satisfaction" in analysis:
        analysis["customer_satisfaction"] = _validate_customer_satisfaction(
            analysis["customer_satisfaction"]
        )

    # Normalize order items against menu
    if "order" in analysis:
        raw_items = analysis["order"].get("raw_items") or analysis["order"].get("items", [])
        analysis["order"]["normalized_items"] = normalize_order_items(raw_items)
        unknown = [x for x in analysis["order"]["normalized_items"] if not x["is_known"]]
        if unknown:
            analysis["flags"]["unknown_products_detected"] = True
            analysis["flags"]["manual_review_required"]    = True
            analysis["flags"]["manual_review_reasons"].append(
                f"פריטים לא מזוהים בתפריט: {', '.join(x['raw'] for x in unknown)}"
            )

    # Override branch_detection with deterministic result if reliable
    if branch_hint and branch_hint.get("confidence", 0) >= 0.80:
        conf_heb = "גבוהה" if branch_hint["confidence"] >= 0.85 else "בינונית"
        analysis["branch_detection"] = {
            "branch_name":          branch_hint["branch_name"],
            "confidence":           conf_heb,
            "evidence":             branch_hint.get("evidence", ""),
            "method":               branch_hint.get("method", ""),
            "requires_manual_review": False,
        }

    # Branch flags
    branch = analysis.get("branch_detection", {})
    if branch.get("confidence") in ("נמוכה", "לא זוהה") or not branch.get("branch_name"):
        analysis["flags"]["branch_undetected"]       = True
        analysis["flags"]["manual_review_required"]  = True
        analysis["flags"]["manual_review_reasons"].append("סניף לא זוהה בביטחון מספק")
    else:
        # Clear stale branch flag if deterministic detection succeeded
        analysis["flags"]["branch_undetected"] = False
        analysis["flags"]["manual_review_reasons"] = [
            r for r in analysis["flags"].get("manual_review_reasons", [])
            if "סניף" not in r
        ]

    # Transcription quality flag
    if transcript_data.get("transcription_confidence", 1.0) < 0.80:
        analysis["flags"]["low_transcription_quality"] = True
        analysis["flags"]["manual_review_required"]    = True
        analysis["flags"]["manual_review_reasons"].append(
            f"דיוק תמלול נמוך: {int(transcript_data.get('transcription_confidence',0)*100)}%"
        )

    return analysis
