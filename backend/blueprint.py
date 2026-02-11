# backend/blueprint.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.scoring import score_from_db
from backend.features import basic_video_features
from backend.db_read import get_video_metrics, recent_channel_metrics


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


def _bucket_duration_for_short(duration_seconds: Optional[int]) -> List[int]:
    # fallback si duration inconnue
    if duration_seconds is None:
        return [15, 25]
    if duration_seconds <= 12:
        return [10, 12]
    if duration_seconds <= 18:
        return [14, 18]
    if duration_seconds <= 25:
        return [18, 25]
    if duration_seconds <= 40:
        return [28, 40]
    return [45, 60]


def _hook_type(feats: Dict[str, Any]) -> Dict[str, Any]:
    is_short = feats.get("is_short") is True
    dur = feats.get("duration_seconds")
    dur_i = _safe_int(dur, default=0) if dur is not None else None

    if is_short:
        if dur_i is not None and dur_i <= 18:
            return {
                "type": "instant_pattern_break",
                "first_1s_goal": "surprise",
                "avoid": ["intro", "context", "bonjour", "explication longue"],
                "examples": [
                    "« STOP — si tu fais ça, tu perds de l’argent. »",
                    "« La banque ne te le dira jamais : fais ça maintenant. »",
                    "« 1 erreur = 1 mois d’épargne perdu. »",
                ],
            }
        if dur_i is not None and dur_i <= 40:
            return {
                "type": "quick_list",
                "first_1s_goal": "curiosity",
                "avoid": ["storytime long", "mise en place"],
                "examples": [
                    "« 3 astuces rapides pour économiser aujourd’hui : #1… »",
                    "« 2 pièges qui te ruinent sans que tu le voies… »",
                ],
            }
        return {
            "type": "mini_story",
            "first_1s_goal": "tension",
            "avoid": ["intro longue"],
            "examples": [
                "« J’ai perdu 200€ en 30 secondes… voilà comment éviter. »",
                "« On m’a arnaqué comme ça — regarde bien. »",
            ],
        }

    # long form
    return {
        "type": "clear_promise",
        "first_10s_goal": "authority + clarity",
        "avoid": ["blabla", "digressions"],
        "examples": [
            "« Aujourd’hui je te montre un plan simple pour économiser 200€ par mois. »",
            "« Voici les 5 arnaques les plus fréquentes et comment les éviter. »",
        ],
    }


def _content_structure(feats: Dict[str, Any]) -> List[str]:
    is_short = feats.get("is_short") is True
    if is_short:
        return [
            "pattern break (0–1s)",
            "micro-context (≤2s)",
            "proof / demo (3–8s)",
            "payoff (1 phrase claire)",
            "hard stop + CTA (commentaire / follow)",
        ]
    return [
        "hook (0–10s) : promesse claire",
        "credibility (10–25s) : pourquoi tu sais",
        "framework (25–60s) : 3 points / 5 étapes",
        "examples (60–80%) : cas concrets",
        "recap + CTA (fin) : action simple à faire",
    ]


def _kill_conditions(baseline_vph: float, is_short: bool) -> Dict[str, Any]:
    # baseline_vph peut être None → on clamp
    b = baseline_vph if baseline_vph and baseline_vph > 0 else 1.0
    if is_short:
        return {
            "views_after_30min": f"< baseline_vph × 1.5 (baseline_vph={b:.3f})",
            "views_after_2h": f"< baseline_vph × 2.0 (baseline_vph={b:.3f})",
            "action": "kill / re-edit hook / repost variation",
        }
    return {
        "views_after_2h": f"< baseline_vph × 1.2 (baseline_vph={b:.3f})",
        "views_after_24h": f"< baseline_vph × 1.5 (baseline_vph={b:.3f})",
        "action": "keep if retention ok / otherwise change packaging",
    }


def _posting_strategy(is_short: bool, outlier_ratio: float, confidence: str) -> Dict[str, Any]:
    # plus c'est confiant + outlier élevé, plus on pousse
    if confidence == "high" and outlier_ratio >= 5:
        if is_short:
            return {"videos_to_post": 7, "timeframe_days": 7, "spacing_hours": [6, 12], "strategy": "replicate_fast"}
        return {"videos_to_post": 3, "timeframe_days": 10, "spacing_days": [3, 4], "strategy": "replicate_strong"}

    if confidence == "high":
        if is_short:
            return {"videos_to_post": 5, "timeframe_days": 7, "spacing_hours": [8, 16], "strategy": "replicate"}
        return {"videos_to_post": 2, "timeframe_days": 14, "spacing_days": [5, 7], "strategy": "validate"}

    # low confidence
    if is_short:
        return {"videos_to_post": 3, "timeframe_days": 7, "spacing_hours": [12, 24], "strategy": "test_variations"}
    return {"videos_to_post": 1, "timeframe_days": 14, "spacing_days": [7], "strategy": "test_once"}


def build_video_blueprint(video_id: str, baseline_lookback: int = 30) -> Dict[str, Any]:
    vid = (video_id or "").strip()
    if not vid:
        return {"ok": False, "error": "video_id is required"}

    vm = get_video_metrics(vid)
    if not vm:
        return {"ok": False, "error": "video_id not found in video_metrics (run trend/bootstrap first)"}

    scored = score_from_db(video_ids=[vid], limit=1, baseline_lookback=int(baseline_lookback))
    item = (scored.get("items") or [None])[0]
    if not item:
        return {"ok": False, "error": "scoring failed or video not found in scoring"}

    feats = basic_video_features({
        "title": vm.get("title") or "",
        "published_at": vm.get("published_at"),
        "duration_seconds": vm.get("duration_seconds"),
    })

    is_short = feats.get("is_short") is True
    duration_s = _safe_int(vm.get("duration_seconds"), default=0) if vm.get("duration_seconds") is not None else None

    confidence = str(item.get("confidence") or "low")
    outlier_ratio = _safe_float(item.get("outlier_ratio"), default=1.0)
    baseline_vph = _safe_float(item.get("baseline_vph"), default=1.0)
    baseline_n = _safe_int(item.get("baseline_n"), default=0)

    # Baseline context (proxy)
    channel_id = vm.get("channel_id")
    hist = recent_channel_metrics(channel_id, limit=int(baseline_lookback), exclude_video_id=vid) if channel_id else []
    baseline_proxy = None
    if hist:
        anchor = hist[0]
        baseline_proxy = basic_video_features({
            "title": anchor.get("title") or "",
            "published_at": anchor.get("published_at"),
            "duration_seconds": anchor.get("duration_seconds"),
        })

    # Duration targets
    if is_short:
        dur_target = _bucket_duration_for_short(duration_s)
    else:
        # long form heuristic: 6–12 minutes si on ne sait pas
        if duration_s is None:
            dur_target = [360, 720]
        elif duration_s < 300:
            dur_target = [300, 600]
        elif duration_s > 1200:
            dur_target = [480, 900]
        else:
            dur_target = [max(300, int(duration_s * 0.8)), min(1200, int(duration_s * 1.2))]

    hook = _hook_type({**feats, "duration_seconds": duration_s})
    structure = _content_structure(feats)
    posting = _posting_strategy(is_short=is_short, outlier_ratio=outlier_ratio, confidence=confidence)
    kill = _kill_conditions(baseline_vph=baseline_vph, is_short=is_short)

    # “Ideas” data-driven (pas LLM) : templates + variables
    if is_short:
        ideas = [
            {
                "idea": "Arnaque / piège en 1 phrase + solution",
                "hook_template": "« STOP — {PIEGE} te coûte {MONTANT}€ / mois. »",
                "format": "short",
                "duration_target_seconds": dur_target,
            },
            {
                "idea": "Checklist ultra rapide",
                "hook_template": "« 3 choses à faire avant de payer {CHOSE}… »",
                "format": "short",
                "duration_target_seconds": dur_target,
            },
            {
                "idea": "Mythe à débunker",
                "hook_template": "« Non, {CROYANCE}. Voici la vérité. »",
                "format": "short",
                "duration_target_seconds": dur_target,
            },
        ]
    else:
        ideas = [
            {
                "idea": "Plan étape par étape",
                "hook_template": "« En {X} étapes, tu peux {RESULTAT} sans {DOULEUR}. »",
                "format": "long",
                "duration_target_seconds": dur_target,
            },
            {
                "idea": "Analyse / explication simple",
                "hook_template": "« {SUJET} expliqué simplement : ce que ça change pour toi. »",
                "format": "long",
                "duration_target_seconds": dur_target,
            },
        ]

    return {
        "ok": True,
        "video_id": vid,
        "title": vm.get("title"),
        "channel_id": channel_id,
        "channel_title": vm.get("channel_title"),
        "format": "short" if is_short else "long",
        "duration_target_seconds": dur_target,
        "hook_structure": hook,
        "content_structure": structure,
        "posting_strategy": posting,
        "kill_conditions": kill,
        "confidence": confidence,
        "derived_from": {
            "baseline_n": baseline_n,
            "baseline_vph": baseline_vph,
            "outlier_ratio": outlier_ratio,
            "vph": _safe_float(item.get("vph"), default=0.0),
        },
        "baseline_proxy_features": baseline_proxy,
        "ideas": ideas,
    }