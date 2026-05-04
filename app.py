import streamlit as st
import tempfile
import json
import os
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

# On Streamlit Cloud, secrets live in st.secrets — inject them into env
try:
    import streamlit as _st
    for _k in ("ASSEMBLYAI_API_KEY", "ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"):
        if _k in _st.secrets and not os.getenv(_k):
            os.environ[_k] = _st.secrets[_k]
except Exception:
    pass

from ensemble import transcribe_ensemble
from analyzer import analyze_call
from corrections import save_correction, load_corrections
from branch_detector import detect_branch
from auth import require_login, logout, current_user_name
from whatsapp_summary import generate_whatsapp_summary
import db

st.set_page_config(
    page_title="אצה AI - ניתוח שיחות",
    page_icon="🍣",
    layout="wide",
)

# ── Single CSS block ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Heebo', sans-serif !important;
    direction: rtl;
    color: #0A0A0A;
}
.stApp { background: #F5F5F5; direction: rtl; text-align: right; }
.stMarkdown { direction: rtl; text-align: right; }
/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none !important; }

/* Primary button */
div.stButton > button[kind="primary"] {
    background: #E31C3D; border: none; color: #FFFFFF;
    font-family: 'Heebo', sans-serif; font-weight: 700;
    font-size: 1.1rem; border-radius: 10px; padding: 0.6rem 1.5rem;
    transition: background 0.2s;
}
div.stButton > button[kind="primary"]:hover { background: #B7152F; }

/* ── HERO ── */
.hero {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2.8rem 2rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border-bottom: 3px solid #E31C3D;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.hero-accent-bar { display: none; }
.hero h1 {
    font-size: 4rem;
    font-weight: 900;
    color: #0A0A0A;
    margin: 0 0 0.1rem 0;
    letter-spacing: -2px;
    line-height: 1;
}
.hero h1 .brand-pink { color: #E31C3D; }
.hero-sub {
    font-size: 1.05rem;
    color: #555555;
    margin: 0.4rem 0 1rem 0;
    font-weight: 400;
}
.hero-tag {
    display: inline-block;
    background: #FFFFFF;
    color: #0A0A0A;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    margin: 0.15rem;
    border: 1.5px solid #0A0A0A;
}
.hero-tag-mint {
    display: inline-block;
    background: #E31C3D;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    margin: 0.15rem;
}

/* Cards */
.section-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0A0A0A;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E31C3D;
}

/* Score */
.score-big { font-size: 3.5rem; font-weight: 900; line-height: 1; color: #E31C3D; }
.score-label { font-size: 0.9rem; color: #666666; margin-top: 0.3rem; }

/* Checkpoints */
.checkpoint-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid #F5F5F5; font-size: 0.95rem;
}
.checkpoint-row:last-child { border-bottom: none; }
.check-yes { font-size: 1rem; font-weight: 900; color: #4ED7C5; }
.check-no  { font-size: 1rem; font-weight: 900; color: #E31C3D; }

/* Badges */
.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600; margin: 0.15rem;
}
.badge-red    { background: #fff0f2; color: #E31C3D; }
.badge-green  { background: #e6faf7; color: #0e9e86; }
.badge-yellow { background: #FFD400; color: #0A0A0A; }
.badge-gray   { background: #F5F5F5; color: #666666; }

/* Sentiment */
.tag-positive { background: #e6faf7; color: #0e9e86; }
.tag-negative { background: #fff0f2; color: #E31C3D; }
.tag-neutral  { background: #F5F5F5; color: #666666; }

/* Order items */
.order-item {
    background: #fff5f6; border-right: 3px solid #E31C3D;
    padding: 0.4rem 0.8rem; margin: 0.3rem 0;
    border-radius: 0 8px 8px 0; font-size: 0.95rem;
}
.order-item-unknown {
    background: #fffde7; border-right: 3px solid #FFD400;
    padding: 0.4rem 0.8rem; margin: 0.3rem 0;
    border-radius: 0 8px 8px 0; font-size: 0.95rem;
}

/* Transcript */
.utterance-agent {
    background: #e8f9f7; border-right: 3px solid #4ED7C5;
    border-radius: 0 10px 10px 0; padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.9rem;
}
.utterance-customer {
    background: #F5F5F5; border-right: 3px solid #6ED3F5;
    border-radius: 0 10px 10px 0; padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.9rem;
}
.utterance-unknown {
    background: #fff9f0; border-right: 3px solid #FFD400;
    border-radius: 0 10px 10px 0; padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.9rem;
}
.speaker-label { font-weight: 700; font-size: 0.78rem; margin-bottom: 0.2rem; color: #666666; }
.timestamp { font-size: 0.75rem; color: #aaa; margin-bottom: 0.2rem; font-family: monospace; }

/* Liability */
.liability-box {
    margin-top: 0.6rem; background: #FFD40022; border-right: 3px solid #FFD400;
    border-radius: 0 8px 8px 0; padding: 0.7rem; font-size: 0.88rem;
}

/* Flag banner */
.flag-banner {
    background: #fff0f2; border: 2px solid #E31C3D; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 1.2rem;
}

/* Report header */
.report-header {
    background: #FFFFFF; border-radius: 14px; padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem; box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border-top: 4px solid #E31C3D;
}
.report-stats {
    display: flex; gap: 2.5rem; flex-wrap: wrap; direction: rtl;
    border-top: 1px solid #F5F5F5; padding-top: 0.8rem; margin-top: 1rem;
}
.stat-label { font-size: 0.78rem; color: #666666; }
.stat-value { font-size: 1.2rem; font-weight: 700; color: #0A0A0A; }

div[data-testid="stFileUploader"] { direction: rtl; }
</style>
""", unsafe_allow_html=True)


# ── Authentication gate ───────────────────────────────────────────────────────
if not require_login():
    st.stop()

# Sidebar — user info + logout
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'Heebo',sans-serif;direction:rtl;padding:0.5rem 0;">
        <div style="font-weight:700;font-size:1rem;color:#0A0A0A;">👤 {current_user_name()}</div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:1rem;">מנהל מחובר</div>
    </div>
    """, unsafe_allow_html=True)

    if db.is_configured():
        cost_data = db.load_total_cost()
        total_usd = cost_data["total_usd"]
        total_ils = total_usd * 3.7
        n_calls   = cost_data["call_count"]
        st.markdown(f"""
        <div style="font-family:'Heebo',sans-serif;direction:rtl;
                    background:#fff0f2;border-radius:10px;padding:0.8rem 1rem;
                    margin-bottom:1rem;border-right:3px solid #E31C3D;">
            <div style="font-size:0.75rem;color:#888;margin-bottom:0.2rem;">עלות כוללת ({n_calls} שיחות)</div>
            <div style="font-size:1.6rem;font-weight:900;color:#E31C3D;line-height:1.1;">${total_usd:.4f}</div>
            <div style="font-size:0.85rem;color:#666;">≈ ₪{total_ils:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚪 התנתקות", use_container_width=True):
        logout()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms_to_ts(ms: int) -> str:
    s = ms // 1000
    return f"{s//60:02d}:{s%60:02d}"


def _conf_color(conf: float) -> str:
    if conf >= 0.85: return "#4ED7C5"
    if conf >= 0.70: return "#FFD400"
    return "#E31C3D"


def _check(val: bool) -> str:
    return '<span class="check-yes">✓</span>' if val else '<span class="check-no">✗</span>'


def render_badges(items, badge_class):
    return "".join(f'<span class="badge {badge_class}">{i}</span>' for i in items) if items else ""


# ── Section renderers ─────────────────────────────────────────────────────────

def render_flags(flags: dict):
    reasons = flags.get("manual_review_reasons", [])
    if not flags.get("manual_review_required") or not reasons:
        return
    items = "".join(f"<li>{r}</li>" for r in reasons)
    st.markdown(f"""
    <div class="flag-banner">
        <strong>⚠️ נדרשת בדיקה ידנית:</strong>
        <ul style="margin:0.4rem 0 0 1rem; padding:0;">{items}</ul>
    </div>
    """, unsafe_allow_html=True)


def render_branch(branch: dict):
    name       = branch.get("branch_name") or "לא זוהה"
    conf       = branch.get("confidence", "לא זוהה")
    evidence   = branch.get("evidence", "")
    method     = branch.get("method", "")
    conf_color = {"גבוהה": "#4ED7C5", "בינונית": "#FFD400", "נמוכה": "#E31C3D"}.get(conf, "#E31C3D")
    method_label = {
        "filename":   "📁 שם קובץ",
        "transcript": "📝 תמלול",
        "llm":        "🤖 AI",
    }.get(method, "")
    review = '<span style="color:#E31C3D;font-weight:700;"> · נדרשת בדיקה ידנית</span>' if branch.get("requires_manual_review") else ""
    method_html = f'<span style="font-size:0.78rem;color:#888;margin-right:0.5rem;">{method_label}</span>' if method_label else ""
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">📍 זיהוי סניף</div>
        <div style="font-size:1.4rem;font-weight:900;color:#0A0A0A;">{name}</div>
        <div style="margin-top:0.4rem;font-size:0.88rem;color:{conf_color};font-weight:700;">
            ביטחון: {conf}{review} {method_html}
        </div>
        {f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#666666;font-style:italic;">{evidence}</div>' if evidence else ''}
    </div>
    """, unsafe_allow_html=True)


def render_score_card(title, score, subtitle=""):
    display = score if score is not None else "—"
    st.markdown(f"""
    <div class="section-card" style="text-align:center; border-top:4px solid #E31C3D;">
        <div style="font-size:0.9rem;color:#666666;margin-bottom:0.3rem;">{title}</div>
        <div class="score-big">{display}<span style="font-size:1.5rem;color:#666666;">/10</span></div>
        {f'<div class="score-label">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_checkpoints(compliance: dict):
    rows = "".join(
        f'<div class="checkpoint-row"><span>{name}</span>{_check(done)}</div>'
        for name, done in compliance.items()
    )
    st.markdown(
        f'<div class="section-card"><div class="section-title">📋 תסריט נציג</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def render_order(order: dict):
    normalized = order.get("normalized_items", [])
    if normalized:
        items_html = ""
        for item in normalized:
            css = "order-item" if item["is_known"] else "order-item-unknown"
            flag = "" if item["is_known"] else " ⚠️ לא מזוהה בתפריט"
            label = item["canonical"] if item["canonical"] != item["raw"] else item["raw"]
            items_html += f'<div class="{css}">• {label}{flag}</div>'
    else:
        items_html = "".join(f'<div class="order-item">• {i}</div>' for i in order.get("items", []))

    sauces = order.get("sauces", [])
    sauces_html = ""
    if sauces:
        sauces_html = f'<div style="margin-top:0.6rem;font-size:0.9rem;"><strong>רטבים:</strong> {", ".join(sauces)}</div>'

    cutlery = order.get("cutlery", "")
    cutlery_html = f'<div style="margin-top:0.3rem;font-size:0.9rem;"><strong>כלים:</strong> {cutlery}</div>' if cutlery and cutlery != "לא הוזכר" else ""

    missing = order.get("missing_details", [])
    missing_html = ""
    if missing:
        missing_html = f'<div style="margin-top:0.6rem;font-size:0.85rem;color:#E31C3D;">⚠️ חסר: {", ".join(missing)}</div>'

    special = f'<div style="margin-top:0.6rem;font-size:0.9rem;color:#666666;">🗒️ {order.get("special_requests","")}</div>' if order.get("special_requests") else ""

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">🛒 פרטי ההזמנה</div>
        {items_html}
        {sauces_html}
        {cutlery_html}
        {missing_html}
        <div style="margin-top:0.8rem;font-size:0.95rem;">📍 <strong>כתובת:</strong> {order.get("address","—")}</div>
        <div style="font-size:0.9rem;color:#666666;">🚚 {order.get("delivery_or_pickup","—")}</div>
        {special}
        <div style="margin-top:0.8rem;font-size:0.9rem;color:#666666;">
            נציג חזר על הזמנה: {_check(order.get("repeated_back_to_customer",False))}
            &nbsp;|&nbsp;
            הזמנה אושרה: {_check(order.get("confirmed_by_agent",False))}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_customer(cust: dict):
    if not cust.get("is_reliable", True):
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">😊 שביעות רצון לקוח</div>
            <div style="color:#666666;font-style:italic;padding:0.5rem 0;">
                ⚠️ לא זוהה בביטחון מספק — נדרשת בדיקה ידנית<br>
                <small>{cust.get("reliability_reason","")}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    sentiment_map = {"חיובי": ("🟢","tag-positive"), "שלילי": ("🔴","tag-negative"), "נייטרלי": ("🟡","tag-neutral")}
    icon, tag   = sentiment_map.get(cust.get("sentiment",""), ("⚪","tag-neutral"))
    frustration = render_badges(cust.get("frustration_indicators",[]), "badge badge-red")
    satisfaction= render_badges(cust.get("satisfaction_indicators",[]), "badge badge-green")
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">😊 שביעות רצון לקוח</div>
        <div style="margin-bottom:0.8rem;">
            <span class="badge {tag}" style="font-size:1rem;">{icon} {cust.get("sentiment","")}</span>
        </div>
        {f'<div style="margin-bottom:0.5rem;font-size:0.88rem;font-weight:600;color:#E31C3D;">סימני תסכול:</div><div style="margin-bottom:0.8rem;">{frustration}</div>' if frustration else ''}
        {f'<div style="margin-bottom:0.5rem;font-size:0.88rem;font-weight:600;color:#0e9e86;">סימני שביעות רצון:</div><div style="margin-bottom:0.8rem;">{satisfaction}</div>' if satisfaction else ''}
        {f'<div style="font-size:0.88rem;color:#666666;border-top:1px solid #F5F5F5;padding-top:0.6rem;">{cust.get("notes","")}</div>' if cust.get("notes") else ''}
    </div>
    """, unsafe_allow_html=True)


def render_dispute(dispute: dict):
    stated    = "".join(f'<div class="order-item">• {i}</div>' for i in dispute.get("order_stated_by_customer",[]))
    corrections = dispute.get("order_corrections",[])
    corr_html = ""
    if corrections:
        corr_items = "".join(f'<div style="padding:0.3rem 0;font-size:0.9rem;color:#b45309;">⚠️ {c}</div>' for c in corrections)
        corr_html  = f'<div style="margin-top:0.8rem;font-weight:700;font-size:0.9rem;">תיקונים במהלך השיחה:</div>{corr_items}'
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">⚖️ ניתוח מחלוקות</div>
        <div style="font-weight:700;font-size:0.9rem;margin-bottom:0.4rem;">מה הלקוח ביקש:</div>
        {stated}
        {corr_html}
        <div style="margin-top:0.8rem;font-size:0.9rem;">נציג אימת הזמנה: {_check(dispute.get("agent_verified_order",False))}</div>
        <div class="liability-box">⚖️ {dispute.get("liability_assessment","—")}</div>
    </div>
    """, unsafe_allow_html=True)


def render_transcript(utterances):
    if not utterances:
        return
    with st.expander("📄 תמלול השיחה המלא עם חותמות זמן"):
        for u in utterances:
            speaker  = u.get("speaker", "לא ידוע")
            start_ts = _ms_to_ts(u.get("start_ms", 0))
            end_ts   = _ms_to_ts(u.get("end_ms",   0))
            css      = {"נציג": "utterance-agent", "לקוח": "utterance-customer"}.get(speaker, "utterance-unknown")
            label    = {"נציג": "🎧 נציג", "לקוח": "👤 לקוח"}.get(speaker, "❓ לא ידוע")
            conf     = u.get("confidence", 0)
            conf_str = f" · {int(conf*100)}%" if conf else ""
            st.markdown(f"""
            <div class="{css}">
                <div class="timestamp">[{start_ts}–{end_ts}]{conf_str}</div>
                <div class="speaker-label">{label}</div>
                {u["text"]}
            </div>
            """, unsafe_allow_html=True)


def render_corrections_ui():
    with st.expander("✏️ תיקון ידני (מנהל)"):
        st.markdown("**הוסף תיקון לשיחה זו:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            c_type = st.selectbox("סוג תיקון", ["product", "branch", "speaker", "transcript"])
        with col2:
            original  = st.text_input("מה היה שגוי")
        with col3:
            corrected = st.text_input("מה נכון")
        if st.button("שמור תיקון"):
            if original and corrected:
                save_correction(c_type, original, corrected)
                st.success("התיקון נשמר ✓")


def display_report(analysis, transcript_data, filename=""):
    cq   = analysis.get("call_quality", {})
    dur  = int(cq.get("duration_seconds", 0))
    mins, secs = dur // 60, dur % 60
    conf = transcript_data.get("transcription_confidence", 0)
    ensemble_used = transcript_data.get("used_ensemble", False)

    # Flags banner
    render_flags(analysis.get("flags", {}))

    # Report header
    branch_name = analysis.get("branch_detection", {}).get("branch_name") or "לא זוהה"
    ensemble_badge = ' <span style="background:#4ED7C5;color:#0A0A0A;font-size:0.75rem;font-weight:700;padding:0.1rem 0.5rem;border-radius:10px;">Ensemble</span>' if ensemble_used else ""
    st.markdown(f"""
    <div class="report-header">
        <div style="font-size:1.3rem;font-weight:700;">
            📞 דוח ניתוח שיחה &nbsp;|&nbsp;
            <span style="color:#E31C3D;">סניף: {branch_name}</span>
        </div>
        <div class="report-stats">
            <div><div class="stat-label">משך שיחה</div><div class="stat-value">{mins}:{secs:02d}</div></div>
            <div><div class="stat-label">מילים/דקה</div><div class="stat-value">{cq.get("words_per_minute","—")}</div></div>
            <div><div class="stat-label">קצב</div><div class="stat-value" style="font-size:1rem;">{cq.get("pace_assessment","—")}</div></div>
            <div><div class="stat-label">בהירות</div><div class="stat-value">{cq.get("clarity_score","—")}/10</div></div>
            <div><div class="stat-label">דיוק תמלול</div>
                 <div class="stat-value" style="color:{_conf_color(conf)};">
                     {int(conf*100)}%{ensemble_badge}
                 </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Branch
    render_branch(analysis.get("branch_detection", {}))

    # Scores
    agent = analysis.get("agent_performance", {})
    cust  = analysis.get("customer_satisfaction", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        render_score_card("ביצועי נציג", agent.get("overall_score"),
                          f'{len(agent.get("missed_checkpoints",[]))} תחנות חסרות')
    with col2:
        score = cust.get("overall_score") if cust.get("is_reliable", True) else None
        render_score_card("שביעות רצון לקוח", score,
                          cust.get("sentiment","") if cust.get("is_reliable", True) else "לא זוהה")
    with col3:
        render_score_card("בהירות שיחה", cq.get("clarity_score"))

    col_left, col_right = st.columns(2)
    with col_left:
        render_checkpoints(agent.get("script_compliance", {}))
        areas_html  = render_badges(agent.get("improvement_areas",[]), "badge badge-yellow")
        strong_html = render_badges(agent.get("strong_points",[]),     "badge badge-green")
        if areas_html or strong_html:
            st.markdown(f"""
            <div class="section-card">
                <div class="section-title">💡 משוב לנציג</div>
                {f'<div style="margin-bottom:0.5rem;font-size:0.85rem;font-weight:600;color:#0e9e86;">נקודות חוזק:</div><div style="margin-bottom:0.8rem;">{strong_html}</div>' if strong_html else ''}
                {f'<div style="margin-bottom:0.5rem;font-size:0.85rem;font-weight:600;color:#9a7800;">לשיפור:</div><div>{areas_html}</div>' if areas_html else ''}
                {f'<div style="margin-top:0.8rem;font-size:0.88rem;color:#666666;border-top:1px solid #F5F5F5;padding-top:0.6rem;">{agent.get("professionalism_notes","")}</div>' if agent.get("professionalism_notes") else ''}
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        render_order(analysis.get("order", {}))
        render_customer(cust)

    render_dispute(analysis.get("dispute_analysis", {}))
    render_transcript(transcript_data.get("utterances", []))
    render_corrections_ui()

    # Call type badge
    call_type = analysis.get("call_type", "unknown")
    call_type_heb = {"order": "📦 הזמנה", "service": "🔧 שירות", "inquiry": "❓ פנייה", "failed": "⛔ טיפול כושל"}.get(call_type, "❓ לא זוהה")
    missed_conv = analysis.get("missed_conversion", False)
    missed_html = f'<span class="badge badge-red" style="margin-right:0.5rem;">⚠️ הזדמנות המרה שהוחמצה</span>' if missed_conv else ""
    early_transfer = '<span class="badge badge-red">📞 העברה מוקדמת</span>' if analysis.get("early_transfer") else ""
    st.markdown(f"""
    <div class="section-card" style="border-top:4px solid #E31C3D;">
        <div class="section-title">🏷️ סיווג שיחה</div>
        <span class="badge badge-gray" style="font-size:1rem;">{call_type_heb}</span>
        {missed_html}{early_transfer}
        <div style="margin-top:0.6rem;font-size:0.85rem;color:#666;">{analysis.get("call_type_reasoning","")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Cost breakdown
    cost = analysis.get("_cost", {})
    if cost:
        st.markdown(f"""
        <div class="section-card" style="border-top:2px solid #0A0A0A;">
            <div class="section-title">💰 עלות שיחה זו</div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:0.95rem;">
                <div><span style="color:#666;">Claude AI</span>&nbsp; <strong>${cost.get('cost_claude_usd', 0):.4f}</strong></div>
                <div><span style="color:#666;">תמלול (AssemblyAI)</span>&nbsp; <strong>${cost.get('cost_aai_usd', 0):.4f}</strong></div>
                <div style="border-right:2px solid #E31C3D;padding-right:1rem;">
                    <span style="color:#E31C3D;font-weight:700;">סה״כ</span>&nbsp;
                    <strong style="color:#E31C3D;">${cost.get('total_cost_usd', 0):.4f}</strong>
                </div>
                <div style="font-size:0.8rem;color:#aaa;">
                    טוקנים: {cost.get('tokens_in',0):,} קלט / {cost.get('tokens_out',0):,} פלט
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # JSON export
    full_export = {"analysis": analysis, "transcript": transcript_data, "file": filename}
    st.download_button(
        label="⬇️ ייצוא JSON מלא",
        data=json.dumps(full_export, ensure_ascii=False, indent=2),
        file_name=f"atza_analysis_{Path(filename).stem}.json",
        mime="application/json",
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-accent-bar"></div>
    <h1><span class="brand-pink">ATZA</span> ניתוח שיחות</h1>
    <div class="hero-sub">מערכת AI לניתוח אוטומטי של שיחות מוקד</div>
    <div>
        <span class="hero-tag">🤖 תמלול + Ensemble</span>
        <span class="hero-tag-mint">📍 זיהוי סניף אוטומטי</span>
        <span class="hero-tag">🛒 זיהוי הזמנה</span>
        <span class="hero-tag-mint">⭐ ניקוד נציג</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_history, tab_dashboard = st.tabs(["🎙️ ניתוח שיחה", "📚 היסטוריה", "📊 ביצועים"])

# ── Tab: Analyze ──────────────────────────────────────────────────────────────
with tab_analyze:
    uploaded = st.file_uploader(
        "גרור לכאן קובץ שיחה או לחץ לבחירה",
        type=["wav", "mp3", "m4a", "mp4", "ogg", "flac"],
        label_visibility="visible",
    )

    if uploaded:
        st.success(f"✅ הקובץ נטען: **{uploaded.name}**")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🎙️ נתח שיחה", type="primary", use_container_width=True):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            try:
                with st.status("מעבד את השיחה...", expanded=True) as status:
                    transcript_data, steps = transcribe_ensemble(tmp_path)
                    for step in steps:
                        st.write(step)
                    if transcript_data.get("used_ensemble"):
                        st.write(f"💡 הערות מיזוג: {transcript_data.get('ensemble_notes','')}")

                    branch_result = detect_branch(uploaded.name, transcript_data["utterances"])
                    if branch_result["branch_name"]:
                        method_heb = {"filename": "שם קובץ", "transcript": "תמלול"}.get(
                            branch_result["method"], "אוטומטי"
                        )
                        st.write(f"📍 סניף זוהה ({method_heb}): **{branch_result['branch_name']}**")

                    st.write("🤖 מנתח עם Claude AI...")
                    analysis = analyze_call(transcript_data, branch_hint=branch_result)

                    wa_summary = generate_whatsapp_summary(analysis)
                    analysis["_whatsapp_summary"] = wa_summary

                    if db.is_configured():
                        st.write("💾 שומר ל-DB...")
                        db.save_call(transcript_data, analysis, uploaded.name, wa_summary)

                    st.write("✅ ניתוח הושלם")
                    status.update(label="✅ הדוח מוכן!", state="complete", expanded=False)

                # Save to session state so report survives widget interactions
                st.session_state["last_analysis"]        = analysis
                st.session_state["last_transcript"]      = transcript_data
                st.session_state["last_filename"]        = uploaded.name
                st.session_state["last_wa_summary"]      = wa_summary

            except Exception as e:
                st.error(f"שגיאה: {e}")
                raise
            finally:
                os.unlink(tmp_path)

        # Render from session state — persists through reruns
        if st.session_state.get("last_analysis") and st.session_state.get("last_filename") == uploaded.name:
            analysis       = st.session_state["last_analysis"]
            transcript_data = st.session_state["last_transcript"]
            wa_summary     = st.session_state["last_wa_summary"]

            st.markdown("---")
            display_report(analysis, transcript_data, uploaded.name)

            st.markdown("### 📱 סיכום לשליחה ב-WhatsApp")
            st.text_area(
                label="העתק והדבק בווצאפ לנציג",
                value=wa_summary,
                height=400,
                label_visibility="visible",
            )

# ── Tab: History ──────────────────────────────────────────────────────────────
with tab_history:
    if not db.is_configured():
        st.info("Supabase not configured — set SUPABASE_URL and SUPABASE_KEY.")
    else:
        if st.button("🔄 Refresh"):
            st.rerun()

        all_rows = db.load_call_history(limit=200)
        if not all_rows:
            st.info("No calls saved yet.")
        else:
            call_type_heb = {"order": "הזמנה", "service": "שירות", "inquiry": "פנייה", "failed": "כישלון", "unknown": "—"}

            # ── Filters ───────────────────────────────────────────────────────
            branches_available = sorted({r.get("branch_name") or "—" for r in all_rows})
            ctypes_available   = sorted({call_type_heb.get(r.get("call_type",""), "—") for r in all_rows})

            with st.expander("🔍 Filters", expanded=False):
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    f_branches = st.multiselect("Branch", branches_available)
                with fc2:
                    f_ctypes = st.multiselect("Call type", ctypes_available)
                with fc3:
                    f_score = st.slider("Min agent score", 0, 10, 0)
                with fc4:
                    f_date_from = st.date_input("From", value=date.today() - timedelta(days=30))
                    f_date_to   = st.date_input("To",   value=date.today())

            def _passes_filters(row):
                an     = (row.get("analyses") or [{}])[0]
                branch = row.get("branch_name") or "—"
                ctype  = call_type_heb.get(row.get("call_type",""), "—")
                score  = an.get("agent_score") or 0
                d_str  = (row.get("created_at") or "")[:10]
                try:
                    d = date.fromisoformat(d_str)
                except Exception:
                    d = date.today()
                if f_branches and branch not in f_branches:
                    return False
                if f_ctypes and ctype not in f_ctypes:
                    return False
                if score < f_score:
                    return False
                if not (f_date_from <= d <= f_date_to):
                    return False
                return True

            rows = [r for r in all_rows if _passes_filters(r)]

            # ── Cost summary ──────────────────────────────────────────────────
            total_cost = sum(
                ((r.get("analyses") or [{}])[0].get("full_analysis") or {}).get("_cost", {}).get("total_cost_usd", 0)
                for r in rows
            )
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:1rem;
                        border-right:4px solid #E31C3D;box-shadow:0 2px 8px rgba(0,0,0,0.06);
                        display:flex;gap:2rem;align-items:center;">
                <div>
                    <div style="font-size:0.8rem;color:#666;">Showing</div>
                    <div style="font-size:1.2rem;font-weight:900;">{len(rows)} / {len(all_rows)} calls</div>
                </div>
                {'<div><div style="font-size:0.8rem;color:#666;">Total cost</div><div style="font-size:1.2rem;font-weight:900;color:#E31C3D;">${:.4f} ≈ ₪{:.2f}</div></div>'.format(total_cost, total_cost*3.7) if total_cost > 0 else ''}
            </div>
            """, unsafe_allow_html=True)

            if not rows:
                st.info("No calls match the current filters.")

            # ── Full report viewer ────────────────────────────────────────────
            if st.session_state.get("history_view_id"):
                view_id  = st.session_state["history_view_id"]
                view_row = next((r for r in all_rows if r.get("id") == view_id), None)
                if view_row:
                    view_an  = (view_row.get("analyses") or [{}])[0]
                    view_fa  = view_an.get("full_analysis") or {}
                    view_fn  = view_row.get("filename") or "—"
                    stub_transcript = {
                        "utterances":               [],
                        "transcription_confidence": view_row.get("transcription_confidence", 0),
                        "used_ensemble":            False,
                        "duration_seconds":         view_row.get("duration_seconds", 0),
                        "words_count":              view_row.get("words_count", 0),
                    }
                    st.markdown("---")
                    bcol1, bcol2 = st.columns([1, 8])
                    with bcol1:
                        if st.button("✕ Close"):
                            st.session_state.pop("history_view_id", None)
                            st.rerun()
                    with bcol2:
                        st.markdown(f"**Full report — {view_fn}**")
                    display_report(view_fa, stub_transcript, view_fn)

                    call_id = view_row.get("id","")
                    if call_id:
                        st.markdown("---")
                        st.markdown("**Add feedback:**")
                        fb_col1, fb_col2 = st.columns(2)
                        with fb_col1:
                            fb_type  = st.selectbox("Type", ["comment","correction","flag"], key=f"vfbt_{call_id}")
                            fb_field = st.text_input("Field (optional)", key=f"vfbf_{call_id}")
                        with fb_col2:
                            fb_orig = st.text_input("Original value", key=f"vfbo_{call_id}")
                            fb_corr = st.text_input("Corrected value", key=f"vfbc_{call_id}")
                        fb_notes = st.text_area("Notes", key=f"vfbn_{call_id}", height=80)
                        if st.button("💾 Save feedback", key=f"vsave_fb_{call_id}"):
                            ok = db.save_feedback(call_id, current_user_name(), fb_type, fb_field, fb_orig, fb_corr, fb_notes)
                            if ok:
                                st.success("Feedback saved.")
                    st.markdown("---")

            # ── Call list ─────────────────────────────────────────────────────
            for row in rows:
                analyses  = row.get("analyses") or [{}]
                an        = analyses[0] if analyses else {}
                branch    = row.get("branch_name") or "—"
                ctype     = call_type_heb.get(row.get("call_type",""), "—")
                dur       = int(row.get("duration_seconds") or 0)
                a_score   = an.get("agent_score")
                date_str  = (row.get("created_at") or "")[:10]
                filename  = row.get("filename") or "—"
                fa        = an.get("full_analysis") or {}
                call_cost = (fa.get("_cost") or {}).get("total_cost_usd", 0)
                cost_str  = f"  |  ${call_cost:.4f}" if call_cost else ""
                score_str = f"  |  {a_score}/10" if a_score is not None else ""

                with st.expander(f"📞 {filename}  |  {branch}  |  {ctype}  |  {date_str}{score_str}{cost_str}"):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    col1.metric("Branch", branch)
                    col2.metric("Agent score", f"{a_score}/10" if a_score else "—")
                    col3.metric("Duration", f"{dur//60}:{dur%60:02d}")
                    with col4:
                        if st.button("📋 Full report", key=f"view_{row['id']}"):
                            st.session_state["history_view_id"] = row["id"]
                            st.rerun()

                    wa = an.get("whatsapp_summary","")
                    if wa:
                        st.text_area("WhatsApp summary", value=wa, height=200, key=f"wa_{row['id']}")

                    flags   = an.get("flags") or {}
                    reasons = (flags.get("manual_review_reasons") or []) if isinstance(flags, dict) else []
                    if reasons:
                        st.warning("⚠️ " + " | ".join(reasons))

                    call_id = row.get("id","")
                    if call_id:
                        st.markdown("**Add feedback:**")
                        fb_col1, fb_col2 = st.columns(2)
                        with fb_col1:
                            fb_type  = st.selectbox("Type", ["comment","correction","flag"], key=f"fbt_{call_id}")
                            fb_field = st.text_input("Field (optional)", key=f"fbf_{call_id}")
                        with fb_col2:
                            fb_orig = st.text_input("Original value", key=f"fbo_{call_id}")
                            fb_corr = st.text_input("Corrected value", key=f"fbc_{call_id}")
                        fb_notes = st.text_area("Notes", key=f"fbn_{call_id}", height=80)
                        if st.button("💾 Save feedback", key=f"save_fb_{call_id}"):
                            ok = db.save_feedback(call_id, current_user_name(), fb_type, fb_field, fb_orig, fb_corr, fb_notes)
                            if ok:
                                st.success("Feedback saved.")

# ── Tab: Dashboard ────────────────────────────────────────────────────────────
with tab_dashboard:
    if not db.is_configured():
        st.info("Supabase not configured.")
    else:
        dash_rows = db.load_call_history(limit=500)
        if not dash_rows:
            st.info("No data yet.")
        else:
            # Build flat records
            records = []
            for r in dash_rows:
                an    = (r.get("analyses") or [{}])[0]
                fa    = an.get("full_analysis") or {}
                agent = fa.get("agent_performance", {})
                cust  = fa.get("customer_satisfaction", {})
                records.append({
                    "branch":     r.get("branch_name") or "Unknown",
                    "call_type":  r.get("call_type") or "unknown",
                    "date":       (r.get("created_at") or "")[:10],
                    "dur":        int(r.get("duration_seconds") or 0),
                    "a_score":    an.get("agent_score"),
                    "c_score":    an.get("customer_score"),
                    "missed":     agent.get("missed_checkpoints") or [],
                    "missed_conv": fa.get("missed_conversion", False),
                    "flagged":    (fa.get("flags") or {}).get("manual_review_required", False),
                })

            total      = len(records)
            scored     = [r for r in records if r["a_score"] is not None]
            avg_score  = round(sum(r["a_score"] for r in scored) / len(scored), 1) if scored else 0
            conv_rate  = round(sum(1 for r in records if r["call_type"] == "order") / total * 100, 1)
            missed_conv_pct = round(sum(1 for r in records if r["missed_conv"]) / total * 100, 1)
            flagged_pct     = round(sum(1 for r in records if r["flagged"]) / total * 100, 1)

            # ── KPI row ───────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total calls", total)
            k2.metric("Avg agent score", f"{avg_score}/10")
            k3.metric("Order rate", f"{conv_rate}%")
            k4.metric("Missed conversions", f"{missed_conv_pct}%")
            k5.metric("Flagged for review", f"{flagged_pct}%")

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            # ── Per-branch breakdown ──────────────────────────────────────────
            with col_left:
                st.markdown('<div class="section-card"><div class="section-title">📍 Performance by branch</div>', unsafe_allow_html=True)
                branch_stats = {}
                for r in records:
                    b = r["branch"]
                    if b not in branch_stats:
                        branch_stats[b] = {"calls": 0, "scores": [], "orders": 0, "missed_conv": 0}
                    branch_stats[b]["calls"] += 1
                    if r["a_score"] is not None:
                        branch_stats[b]["scores"].append(r["a_score"])
                    if r["call_type"] == "order":
                        branch_stats[b]["orders"] += 1
                    if r["missed_conv"]:
                        branch_stats[b]["missed_conv"] += 1

                branch_rows = sorted(
                    branch_stats.items(),
                    key=lambda x: -(sum(x[1]["scores"]) / len(x[1]["scores"]) if x[1]["scores"] else 0)
                )
                for branch, s in branch_rows:
                    avg = round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else None
                    bar_w = int((avg or 0) / 10 * 100)
                    color = "#4ED7C5" if (avg or 0) >= 8 else "#FFD400" if (avg or 0) >= 6 else "#E31C3D"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.9rem;font-weight:600;">
                            <span>{branch}</span>
                            <span style="color:{color};">{avg}/10</span>
                        </div>
                        <div style="background:#F5F5F5;border-radius:4px;height:8px;margin:0.2rem 0;">
                            <div style="width:{bar_w}%;background:{color};height:8px;border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.78rem;color:#888;">
                            {s['calls']} calls &nbsp;·&nbsp; {s['orders']} orders &nbsp;·&nbsp; {s['missed_conv']} missed conversions
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Missed checkpoints breakdown ──────────────────────────────────
            with col_right:
                st.markdown('<div class="section-card"><div class="section-title">📋 Most missed checkpoints</div>', unsafe_allow_html=True)
                checkpoint_counts = {}
                for r in records:
                    for cp in r["missed"]:
                        checkpoint_counts[cp] = checkpoint_counts.get(cp, 0) + 1

                if checkpoint_counts:
                    sorted_cp = sorted(checkpoint_counts.items(), key=lambda x: -x[1])
                    for cp, count in sorted_cp:
                        pct = round(count / total * 100)
                        st.markdown(f"""
                        <div style="margin-bottom:0.7rem;">
                            <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                                <span>{cp}</span>
                                <span style="color:#E31C3D;font-weight:700;">{pct}%</span>
                            </div>
                            <div style="background:#F5F5F5;border-radius:4px;height:6px;margin-top:0.2rem;">
                                <div style="width:{pct}%;background:#E31C3D;height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No missed checkpoints recorded yet.")
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Call type distribution ────────────────────────────────────────
            st.markdown('<div class="section-card"><div class="section-title">🏷️ Call type distribution</div>', unsafe_allow_html=True)
            type_labels = {"order": "Order", "service": "Service", "inquiry": "Inquiry", "failed": "Failed", "unknown": "Unknown"}
            type_counts = {}
            for r in records:
                t = type_labels.get(r["call_type"], r["call_type"])
                type_counts[t] = type_counts.get(t, 0) + 1

            dist_cols = st.columns(len(type_counts))
            for i, (t, cnt) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
                pct = round(cnt / total * 100)
                dist_cols[i].metric(t, f"{cnt} ({pct}%)")
            st.markdown('</div>', unsafe_allow_html=True)
