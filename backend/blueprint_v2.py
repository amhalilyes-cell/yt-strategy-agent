# backend/blueprint_v2.py
from __future__ import annotations

import re
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.db_read import get_video_metrics, recent_channel_metrics
from backend.scoring import score_from_db
from backend.features import basic_video_features


# -----------------------
# Helpers safe
# -----------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _baseline_proxy_features(hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not hist:
        return basic_video_features({"title": "", "published_at": None, "duration_seconds": None})
    anchor = hist[0]
    return basic_video_features(
        {
            "title": anchor.get("title") or "",
            "published_at": anchor.get("published_at"),
            "duration_seconds": anchor.get("duration_seconds"),
        }
    )


def detect_format_shift(outlier_feats: Dict[str, Any], baseline_feats: Dict[str, Any]) -> List[str]:
    keys = ["duration_bucket", "is_short", "hour_utc", "weekday"]
    shifts: List[str] = []
    for k in keys:
        if outlier_feats.get(k) != baseline_feats.get(k):
            shifts.append(k)
    return shifts


def _hook_examples_v2(fmt: str, vertical: str = "saas", lang: str = "fr") -> List[str]:
    vertical = (vertical or "saas").lower()
    if lang.startswith("fr"):
        if fmt == "short":
            return [
                "« STOP — tu perds des leads à cause de ça. »",
                "« 1 tweak = +30% de réponses (sans envoyer plus). »",
                "« Si tu fais ça en cold email, t’es mort. »",
            ]
        return [
            "« Je te montre comment scaler tes leads en 30 jours (sans ads). »",
            "« Le système simple que j’utilise pour convertir en B2B. »",
            "« Pourquoi ta prospection échoue (et comment corriger). »",
        ]
    # EN fallback
    if fmt == "short":
        return ["STOP — this kills replies.", "One tweak = +30% replies.", "You’re doing cold email wrong."]
    return ["How to scale leads in 30 days.", "My simple B2B system.", "Why your outreach fails."]


def _ideas_v2(duration_target: Tuple[int, int], fmt: str, vertical: str, goal: str, lang: str) -> List[Dict[str, Any]]:
    # V2: idées plus “business-ready”, + templates
    if lang.startswith("fr"):
        return [
            {
                "idea": "Erreur fréquente + fix (ultra concret)",
                "hook_template": "« STOP — {ERREUR} te fait perdre {RESULTAT}. Fais ça à la place. »",
                "format": fmt,
                "duration_target_seconds": [duration_target[0], duration_target[1]],
            },
            {
                "idea": "Script / template à copier",
                "hook_template": "« Copie-colle ce message pour {GOAL}. »",
                "format": fmt,
                "duration_target_seconds": [duration_target[0], duration_target[1]],
            },
            {
                "idea": "Étude de cas mini (preuve rapide)",
                "hook_template": "« Comment j’ai obtenu {CHIFFRE} en {DELAI} (sans {OBJECTION}). »",
                "format": fmt,
                "duration_target_seconds": [duration_target[0], duration_target[1]],
            },
        ]
    return [
        {"idea": "Common mistake + fix", "hook_template": "STOP — {MISTAKE} costs you {RESULT}. Do this instead.", "format": fmt,
         "duration_target_seconds": [duration_target[0], duration_target[1]]},
        {"idea": "Copy-paste template", "hook_template": "Copy-paste this message to {GOAL}.", "format": fmt,
         "duration_target_seconds": [duration_target[0], duration_target[1]]},
        {"idea": "Mini case study", "hook_template": "How I got {NUMBER} in {TIME} (without {OBJECTION}).", "format": fmt,
         "duration_target_seconds": [duration_target[0], duration_target[1]]},
    ]


def build_blueprint_v2(
    video_id: str,
    baseline_lookback: int = 30,
    *,
    vertical: str = "saas",
    goal: str = "leads",
    language: str = "fr",
) -> Dict[str, Any]:
    vid = (video_id or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id is required")

    vm = get_video_metrics(vid)
    if not vm:
        raise HTTPException(status_code=404, detail="video_id not found in video_metrics (run trend/bootstrap first)")

    scored = score_from_db(video_ids=[vid], limit=1, baseline_lookback=int(baseline_lookback))
    item = (scored.get("items") or [None])[0]
    if not item:
        raise HTTPException(status_code=500, detail="scoring failed")

    feats = basic_video_features(
        {
            "title": vm.get("title") or "",
            "published_at": vm.get("published_at"),
            "duration_seconds": vm.get("duration_seconds"),
        }
    )

    channel_id = vm.get("channel_id")
    hist = recent_channel_metrics(channel_id, limit=int(baseline_lookback), exclude_video_id=vid)
    baseline_feats = _baseline_proxy_features(hist)

    fmt = "short" if feats.get("is_short") is True else "long"
    dur_s = _safe_int(vm.get("duration_seconds"), 0)

    # V2: targets un peu plus “agressifs” / optimisés business
    if fmt == "short":
        duration_target = (12, 20)
        structure = [
            "0–1s: pattern break (phrase choc / claim)",
            "1–3s: contexte (1 phrase max)",
            "3–10s: preuve (screen / chiffre / exemple)",
            "10–16s: action (1 étape simple)",
            "16–20s: CTA (commentaire / DM / lead magnet)",
        ]
        avoid = ["intro", "bonjour", "trop de contexte", "explication longue"]
        spacing_hours = [6, 18]
        videos_to_post = 7
        timeframe_days = 7
    else:
        duration_target = (420, 900)  # 7–15 min
        structure = [
            "0–15s: promesse + tension (pourquoi ça compte)",
            "15–45s: erreur fréquente + conséquence",
            "45s–70%: 3 leviers (démonstration / exemples)",
            "70%–90%: plan d’action (checklist)",
            "90%–100%: CTA orienté offre / ressource",
        ]
        avoid = ["intro trop longue", "bla bla", "monologue sans preuve"]
        spacing_hours = [48, 96]
        videos_to_post = 3
        timeframe_days = 14

    baseline_vph = _safe_float(item.get("baseline_vph"), 0.0) or _safe_float(item.get("baseline_vph_clamped"), 0.0)

    # V2: kill rules plus “clair”
    kill_conditions = {
        "views_after_30min": f"< baseline_vph × 1.3 (baseline_vph={round(baseline_vph, 3)})",
        "views_after_2h": f"< baseline_vph × 1.8 (baseline_vph={round(baseline_vph, 3)})",
        "action": "change hook + repost variation (même sujet)",
    }

    format_shift_dims = detect_format_shift(feats, baseline_feats)
    warning_signals = []
    if len(format_shift_dims) >= 2:
        warning_signals.append("format_shift_detected")

    # V2: title “rules” plus utiles
    title_rules = {
        "pattern": "Promesse claire + bénéfice + contrainte (sans blabla)",
        "examples_fr": [
            "Cold email : 1 tweak = +30% réponses (sans envoyer plus)",
            "Le script exact pour booker des calls (même débutant)",
            "3 erreurs qui tuent tes leads (et quoi faire à la place)",
        ],
    }

    baseline_proxy = {
        "title_length": len((vm.get("title") or "").strip()),
        "has_number": bool(re.search(r"\d", (vm.get("title") or ""))),
        "weekday": feats.get("weekday"),
        "hour_utc": feats.get("hour_utc"),
        "duration_seconds": dur_s,
        "duration_bucket": feats.get("duration_bucket"),
        "is_short": feats.get("is_short"),
    }

    return {
        "ok": True,
        "version": "v2",
        "video_id": vid,
        "title": vm.get("title"),
        "channel_id": channel_id,
        "channel_title": vm.get("channel_title"),
        "business": {"vertical": vertical, "goal": goal, "language": language},
        "format": fmt,
        "duration_target_seconds": [duration_target[0], duration_target[1]],
        "hook_structure": {
            "type": "business_pattern_break" if fmt == "short" else "business_promise_plus_tension",
            "first_1s_goal": "shock+benefit" if fmt == "short" else "clarity+stakes",
            "avoid": avoid,
            "examples": _hook_examples_v2(fmt, vertical=vertical, lang=language),
        },
        "title_rules": title_rules,
        "content_structure": structure,
        "posting_strategy": {
            "videos_to_post": videos_to_post,
            "timeframe_days": timeframe_days,
            "spacing_hours": spacing_hours,
            "strategy": "replicate_then_iterate",
        },
        "kill_conditions": kill_conditions,
        "confidence": item.get("confidence", "low"),
        "derived_from": {
            "baseline_n": _safe_int(item.get("baseline_n"), 0),
            "baseline_vph": baseline_vph,
            "outlier_ratio": _safe_float(item.get("outlier_ratio"), 0.0),
            "vph": _safe_float(item.get("vph"), 0.0),
        },
        "baseline_proxy_features": baseline_proxy,
        "format_shift": {"detected": len(format_shift_dims) >= 2, "dimensions": format_shift_dims},
        "warning_signals": warning_signals,
        "ideas": _ideas_v2(duration_target, fmt, vertical=vertical, goal=goal, lang=language),
    }
