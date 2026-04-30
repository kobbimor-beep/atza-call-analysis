"""
Generates a WhatsApp-ready Hebrew text summary from a call analysis dict.
The output is plain text formatted for easy copy-paste to WhatsApp.
"""

_CALL_TYPE_HEB = {
    "order":   "הזמנה",
    "service": "שירות / בעיה בהזמנה",
    "inquiry": "פנייה / שאלה",
    "failed":  "טיפול כושל",
    "unknown": "לא זוהה",
}

_SENTIMENT_HEB = {
    "חיובי":   "😊 חיובי",
    "שלילי":   "😠 שלילי",
    "נייטרלי": "😐 נייטרלי",
    "לא זוהה": "—",
}


def _score_bar(score) -> str:
    if score is None:
        return "—"
    try:
        s = int(score)
        filled  = "█" * s
        empty   = "░" * (10 - s)
        return f"{filled}{empty} {s}/10"
    except (TypeError, ValueError):
        return "—"


def generate_whatsapp_summary(analysis: dict, agent_name: str = "") -> str:
    """
    Returns a formatted Hebrew string ready to be sent via WhatsApp.
    """
    branch  = analysis.get("branch_detection", {}).get("branch_name") or "לא זוהה"
    c_type  = _CALL_TYPE_HEB.get(analysis.get("call_type", "unknown"), "לא זוהה")
    order   = analysis.get("order", {})
    a_perf  = analysis.get("agent_performance", {})
    cust    = analysis.get("customer_satisfaction", {})
    flags   = analysis.get("flags", {})
    quality = analysis.get("call_quality", {})

    agent_score    = a_perf.get("overall_score")
    customer_score = cust.get("overall_score") if cust.get("is_reliable") else None
    sentiment      = _SENTIMENT_HEB.get(cust.get("sentiment", ""), "—")

    # Missed checkpoints
    script   = a_perf.get("script_compliance", {})
    missed   = [k for k, v in script.items() if v is False]
    missed_str = "\n".join(f"  ❌ {m}" for m in missed) if missed else "  ✅ הכל בוצע"

    # Strong points
    strong = a_perf.get("strong_points", [])
    strong_str = "\n".join(f"  ✅ {s}" for s in strong) if strong else "  —"

    # Improvement areas
    improve = a_perf.get("improvement_areas", [])
    improve_str = "\n".join(f"  🔧 {i}" for i in improve) if improve else "  —"

    # Behavior flags
    flag_items = []
    if flags.get("low_transcription_quality"):
        flag_items.append("⚠️ תמלול באיכות נמוכה")
    if flags.get("unknown_products_detected"):
        flag_items.append("⚠️ פריטים לא מזוהים בתפריט")
    if flags.get("branch_undetected"):
        flag_items.append("⚠️ סניף לא זוהה")
    reasons = flags.get("manual_review_reasons", [])
    for r in reasons:
        if r not in flag_items:
            flag_items.append(f"⚠️ {r}")
    flags_str = "\n".join(f"  {f}" for f in flag_items) if flag_items else "  ✅ אין"

    # Order summary
    items = order.get("normalized_items") or []
    items_names = [x.get("matched") or x.get("raw", "") for x in items] if items else order.get("items", [])
    order_str = ", ".join(items_names) if items_names else "לא זוהתה הזמנה"

    conversion = "✅ כן" if order.get("confirmed_by_agent") else "❌ לא"
    address    = order.get("address") or "לא נמסרה"

    # Duration
    dur_s  = int(quality.get("duration_seconds") or 0)
    dur_str = f"{dur_s // 60}:{dur_s % 60:02d} דקות"

    name_line = f"*שם נציג:* {agent_name}" if agent_name else "*שם נציג:* —"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🍣 *דוח שיחה — אצה {branch}*",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        name_line,
        f"*סוג שיחה:* {c_type}",
        f"*משך:* {dur_str}",
        f"*המרה לכן:* {conversion}",
        "",
        "📋 *פרטי הזמנה*",
        f"  פריטים: {order_str}",
        f"  כתובת: {address}",
        "",
        "📊 *ציונים*",
        f"  נציג:   {_score_bar(agent_score)}",
        f"  לקוח:   {_score_bar(customer_score)}",
        f"  סנטימנט: {sentiment}",
        "",
        "✅ *נקודות חוזק*",
        strong_str,
        "",
        "❌ *שלבים שלא בוצעו*",
        missed_str,
        "",
        "🔧 *המלצות לשיפור*",
        improve_str,
        "",
        "🚩 *דגלים*",
        flags_str,
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    return "\n".join(lines)
