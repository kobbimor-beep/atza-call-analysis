"""
Supabase integration — calls, analyses, feedback.
Env vars required: SUPABASE_URL, SUPABASE_KEY
"""
import os
import json
from datetime import datetime, timezone

_client = None


def _get_client():
    global _client
    if _client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY חסרים ב-.env")
        _client = create_client(url, key)
    return _client


# ── Schema SQL (run once in Supabase SQL editor) ──────────────────────────────
SCHEMA_SQL = """
-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Calls
create table if not exists calls (
    id                      uuid primary key default uuid_generate_v4(),
    created_at              timestamptz default now(),
    filename                text,
    branch_name             text,
    duration_seconds        numeric,
    words_count             int,
    transcription_confidence numeric,
    transcription_quality   text,
    speaker_confidence      numeric,
    call_type               text,
    audio_storage_path      text
);

-- Analyses
create table if not exists analyses (
    id              uuid primary key default uuid_generate_v4(),
    call_id         uuid references calls(id) on delete cascade,
    created_at      timestamptz default now(),
    full_analysis   jsonb,
    agent_score     numeric,
    customer_score  numeric,
    overall_score   numeric,
    whatsapp_summary text,
    flags           jsonb,
    cost_usd        numeric default 0
);

-- Migration: add cost_usd if upgrading from older schema
-- alter table analyses add column if not exists cost_usd numeric default 0;

-- Manager feedback / corrections
create table if not exists feedback (
    id               uuid primary key default uuid_generate_v4(),
    call_id          uuid references calls(id) on delete cascade,
    created_at       timestamptz default now(),
    manager_username text,
    feedback_type    text,   -- 'correction' | 'comment' | 'flag'
    field_corrected  text,   -- e.g. 'call_type', 'order.items', 'agent_score'
    original_value   text,
    corrected_value  text,
    notes            text
);

-- Row-level security (optional — enable if you want per-user isolation)
-- alter table calls enable row level security;
"""


# ── Write ─────────────────────────────────────────────────────────────────────

def save_call(
    transcript_data: dict,
    analysis: dict,
    filename: str,
    whatsapp_summary: str = "",
    audio_storage_path: str = "",
) -> str | None:
    """
    Insert call + analysis into DB. Returns the call UUID or None on error.
    """
    try:
        sb = _get_client()

        branch   = analysis.get("branch_detection", {}).get("branch_name")
        call_type = analysis.get("call_type", "unknown")
        a_perf   = analysis.get("agent_performance", {})
        cust     = analysis.get("customer_satisfaction", {})
        flags    = analysis.get("flags", {})

        call_row = {
            "filename":                 filename,
            "branch_name":              branch,
            "duration_seconds":         transcript_data.get("duration_seconds"),
            "words_count":              transcript_data.get("words_count"),
            "transcription_confidence": transcript_data.get("transcription_confidence"),
            "transcription_quality":    transcript_data.get("transcription_quality"),
            "speaker_confidence":       transcript_data.get("speaker_confidence"),
            "call_type":                call_type,
            "audio_storage_path":       audio_storage_path,
        }

        resp     = sb.table("calls").insert(call_row).execute()
        call_id  = resp.data[0]["id"]

        agent_score    = a_perf.get("overall_score")
        customer_score = cust.get("overall_score") if cust.get("is_reliable") else None
        overall_score  = None
        if agent_score is not None and customer_score is not None:
            overall_score = round((agent_score + customer_score) / 2, 1)

        cost_usd = (analysis.get("_cost") or {}).get("total_cost_usd", 0)

        analysis_row = {
            "call_id":         call_id,
            "full_analysis":   analysis,
            "agent_score":     agent_score,
            "customer_score":  customer_score,
            "overall_score":   overall_score,
            "whatsapp_summary": whatsapp_summary,
            "flags":           flags,
            "cost_usd":        cost_usd,
        }

        sb.table("analyses").insert(analysis_row).execute()
        return call_id

    except Exception as e:
        import streamlit as st
        st.warning(f"DB: לא ניתן לשמור שיחה — {e}")
        return None


def save_feedback(
    call_id: str,
    manager_username: str,
    feedback_type: str,
    field_corrected: str = "",
    original_value: str = "",
    corrected_value: str = "",
    notes: str = "",
) -> bool:
    try:
        sb = _get_client()
        sb.table("feedback").insert({
            "call_id":          call_id,
            "manager_username": manager_username,
            "feedback_type":    feedback_type,
            "field_corrected":  field_corrected,
            "original_value":   str(original_value),
            "corrected_value":  str(corrected_value),
            "notes":            notes,
        }).execute()
        return True
    except Exception as e:
        import streamlit as st
        st.warning(f"DB: לא ניתן לשמור פידבק — {e}")
        return False


# ── Read ──────────────────────────────────────────────────────────────────────

def load_call_history(limit: int = 50) -> list[dict]:
    """Returns list of {call, analysis} dicts ordered by newest first."""
    try:
        sb   = _get_client()
        rows = (
            sb.table("calls")
            .select("*, analyses(agent_score, customer_score, overall_score, whatsapp_summary, flags, full_analysis)")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return rows.data or []
    except Exception:
        return []


def load_call_detail(call_id: str) -> dict | None:
    try:
        sb  = _get_client()
        row = (
            sb.table("calls")
            .select("*, analyses(*), feedback(*)")
            .eq("id", call_id)
            .single()
            .execute()
        )
        return row.data
    except Exception:
        return None


def load_feedback_corrections(limit: int = 50) -> list[dict]:
    """Returns correction-type feedback rows for prompt injection."""
    try:
        sb = _get_client()
        rows = (
            sb.table("feedback")
            .select("field_corrected, original_value, corrected_value, notes, feedback_type")
            .in_("feedback_type", ["correction", "comment"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return rows.data or []
    except Exception:
        return []


def load_total_cost() -> dict:
    """Returns {total_usd, call_count} from the analyses table."""
    try:
        sb   = _get_client()
        rows = sb.table("analyses").select("cost_usd").execute()
        costs = [r["cost_usd"] or 0 for r in (rows.data or [])]
        return {"total_usd": round(sum(costs), 4), "call_count": len(costs)}
    except Exception:
        return {"total_usd": 0.0, "call_count": 0}


def is_configured() -> bool:
    """True if Supabase env vars are present."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
