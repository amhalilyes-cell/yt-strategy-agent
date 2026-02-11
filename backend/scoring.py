# backend/scoring.py
from __future__ import annotations

import os
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "app.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def hours_since(published_at: str) -> Optional[float]:
    dt = parse_dt(published_at)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 3600.0, 0.25)


def trending_score(view_count: int, like_count: int, comment_count: int, published_at: str) -> float:
    h = hours_since(published_at) or 72.0
    views = max(int(view_count or 0), 1)
    likes = max(int(like_count or 0), 0)
    comments = max(int(comment_count or 0), 0)

    vph = views / h
    engagement = (likes + 2 * comments) / views
    return float(vph * (1.0 + 3.0 * engagement))


def _median(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def fetch_metrics(video_ids: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()

    if video_ids:
        qmarks = ",".join(["?"] * len(video_ids))
        cur.execute(f"""
            SELECT video_id, channel_id, published_at, views, likes, comments, duration_seconds, fetched_at
            FROM video_metrics
            WHERE video_id IN ({qmarks})
        """, video_ids)
    else:
        cur.execute("""
            SELECT video_id, channel_id, published_at, views, likes, comments, duration_seconds, fetched_at
            FROM video_metrics
            ORDER BY fetched_at DESC
            LIMIT ?
        """, (int(limit),))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def channel_baseline_vph(
    channel_id: str,
    exclude_video_id: Optional[str] = None,
    lookback: int = 30,
) -> Dict[str, Any]:
    """
    Baseline robuste: médiane des VPH des dernières vidéos de la chaîne.
    Retourne aussi n = nb de points.
    """
    if not channel_id:
        return {"baseline_vph": None, "n": 0}

    conn = _conn()
    cur = conn.cursor()

    if exclude_video_id:
        cur.execute("""
            SELECT published_at, views
            FROM video_metrics
            WHERE channel_id = ? AND video_id != ?
            ORDER BY fetched_at DESC
            LIMIT ?
        """, (channel_id, exclude_video_id, int(lookback)))
    else:
        cur.execute("""
            SELECT published_at, views
            FROM video_metrics
            WHERE channel_id = ?
            ORDER BY fetched_at DESC
            LIMIT ?
        """, (channel_id, int(lookback)))

    vals: List[float] = []
    for r in cur.fetchall():
        h = hours_since(r["published_at"])
        if not h:
            continue
        views = max(int(r["views"] or 0), 1)
        vals.append(views / h)

    conn.close()
    return {"baseline_vph": _median(vals), "n": len(vals)}


def confidence_label(n: int) -> str:
    if n >= 20:
        return "high"
    if n >= 8:
        return "medium"
    return "low"


def score_from_db(
    video_ids: Optional[List[str]] = None,
    limit: int = 50,
    baseline_lookback: int = 30,
    min_baseline_vph: float = 1.0,   # ✅ clamp anti ratio débile
) -> Dict[str, Any]:
    rows = fetch_metrics(video_ids=video_ids, limit=limit)
    if not rows:
        return {"ok": False, "error": "No metrics found", "items": []}

    scored: List[Dict[str, Any]] = []

    for r in rows:
        vid = r.get("video_id")
        ch = r.get("channel_id")
        pub = r.get("published_at") or ""
        views = int(r.get("views") or 0)
        likes = int(r.get("likes") or 0)
        comments = int(r.get("comments") or 0)

        h = hours_since(pub) or 72.0
        vph = views / h
        eng = (likes + 2 * comments) / max(views, 1)

        base_info = channel_baseline_vph(ch, exclude_video_id=vid, lookback=baseline_lookback)
        base = base_info["baseline_vph"]
        base_n = int(base_info["n"] or 0)

        base_clamped = None
        out_ratio = None
        out_log = None

        if base is not None:
            base_clamped = max(float(base), float(min_baseline_vph))
            out_ratio = vph / base_clamped
            out_log = math.log10(max(out_ratio, 1e-9))

        scored.append({
            "video_id": vid,
            "channel_id": ch,
            "published_at": pub,
            "views": views,
            "likes": likes,
            "comments": comments,
            "duration_seconds": r.get("duration_seconds"),

            "vph": round(vph, 3),
            "engagement": round(eng, 4),
            "trending_score": round(trending_score(views, likes, comments, pub), 3),

            "baseline_vph": round(base, 3) if base is not None else None,
            "baseline_n": base_n,
            "confidence": confidence_label(base_n),

            "baseline_vph_clamped": round(base_clamped, 3) if base_clamped is not None else None,
            "outlier_ratio": round(out_ratio, 3) if out_ratio is not None else None,
            "outlier_log10": round(out_log, 4) if out_log is not None else None,
        })

    # growth = trending_score
    by_growth = sorted(scored, key=lambda x: x["trending_score"], reverse=True)

    # outliers = outlier_log10 d'abord, puis confidence
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    def out_key(x: Dict[str, Any]):
        # None à la fin
        if x["outlier_log10"] is None:
            return (1, 0, 999)
        return (0, -(x["outlier_log10"] or 0), conf_rank.get(x["confidence"], 9))

    by_outlier = sorted(scored, key=out_key)

    return {
        "ok": True,
        "count": len(scored),
        "top": {
            "growth": by_growth[:10],
            "outliers": by_outlier[:10],
        },
        "items": scored,
    }