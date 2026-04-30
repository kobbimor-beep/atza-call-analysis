"""
Deterministic branch detection — runs before LLM.
Priority: filename → transcript phrases → LLM fallback.
"""
import os
from menu import ATZA_BRANCHES

_ALIASES: dict[str, list[str]] = {
    "עכו":         ["עכו", "akko", "acre"],
    "חיפה":        ["חיפה", "haifa"],
    "נהריה":       ["נהריה", "nahariya", "nahariyya"],
    "קריית אתא":   ["קריית אתא", "קריית", "אתא", "kiryat ata", "kiryat-ata"],
    "קריות":       ["קריות", "krayot"],
    "חדרה":        ["חדרה", "hadera"],
    "נתניה":       ["נתניה", "netanya", "netanya"],
    "תל אביב":     ["תל אביב", "תל-אביב", "tel aviv", "telaviv"],
    "רמת גן":      ["רמת גן", "רמת-גן", "ramat gan", "ramat-gan"],
    "פתח תקווה":   ["פתח תקווה", "פתח-תקווה", "petah tikva", "petah-tikva"],
    "ראשון לציון": ["ראשון לציון", "ראשון", "rishon", "rishon lezion"],
    "אשדוד":       ["אשדוד", "ashdod"],
    "באר שבע":     ["באר שבע", "beer sheva", "beersheba"],
    "ירושלים":     ["ירושלים", "jerusalem"],
    "רחובות":      ["רחובות", "rehovot"],
    "הרצליה":      ["הרצליה", "herzliya"],
    "כפר סבא":     ["כפר סבא", "kfar saba"],
    "רעננה":       ["רעננה", "raanana"],
    "הוד השרון":   ["הוד השרון", "הוד", "hod hasharon"],
    "נס ציונה":    ["נס ציונה", "נס-ציונה", "nes ziona"],
    "מודיעין":     ["מודיעין", "modiin"],
    "אילת":        ["אילת", "eilat"],
}


def _build_lookup() -> list[tuple[str, str]]:
    pairs = []
    for branch in ATZA_BRANCHES:
        aliases = _ALIASES.get(branch, [branch])
        if branch not in aliases:
            aliases = [branch] + aliases
        for alias in aliases:
            pairs.append((alias.lower(), branch))
    # Longest alias first — avoids "ראשון" matching before "ראשון לציון"
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


_LOOKUP = _build_lookup()


def _find_branch(text: str) -> str | None:
    text_lower = text.lower()
    for alias, canonical in _LOOKUP:
        if alias in text_lower:
            return canonical
    return None


def detect_branch_from_filename(filename: str) -> dict | None:
    if not filename:
        return None
    match = _find_branch(os.path.basename(filename))
    if match:
        return {
            "branch_name":          match,
            "confidence":           0.95,
            "method":               "filename",
            "evidence":             f"שם קובץ: {os.path.basename(filename)}",
            "requires_manual_review": False,
        }
    return None


def detect_branch_from_transcript(utterances: list, max_utterances: int = 20) -> dict | None:
    for u in utterances[:max_utterances]:
        text = u.get("text", "")
        match = _find_branch(text)
        if match:
            return {
                "branch_name":          match,
                "confidence":           0.88,
                "method":               "transcript",
                "evidence":             f'"{text[:80]}"',
                "requires_manual_review": False,
            }
    return None


def detect_branch(filename: str, utterances: list, llm_result: dict | None = None) -> dict:
    """
    Full branch detection pipeline.
    Returns best result — always includes branch_name, confidence, method, evidence, requires_manual_review.
    """
    result = detect_branch_from_filename(filename)
    if result:
        return result

    result = detect_branch_from_transcript(utterances)
    if result:
        return result

    if llm_result and llm_result.get("branch_name"):
        conf_map = {"גבוהה": 0.85, "בינונית": 0.60, "נמוכה": 0.30, "לא זוהה": 0.0}
        conf = conf_map.get(llm_result.get("confidence", ""), 0.0)
        if conf >= 0.50:
            return {
                "branch_name":          llm_result["branch_name"],
                "confidence":           conf,
                "method":               "llm",
                "evidence":             llm_result.get("evidence", ""),
                "requires_manual_review": conf < 0.80,
            }

    return {
        "branch_name":          None,
        "confidence":           0.0,
        "method":               "unknown",
        "evidence":             "",
        "requires_manual_review": True,
    }
