# backend/snapshots.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from backend.youtube import get_videos, YouTubeAPIError
from backend.db import insert_video_snapshot


def _parse_iso_dt(s: str) -> datetime:
    # YouTube returns ISO like "2026-02-11T10:12:33Z"
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        # fallback hard fail
        raise HTTPException(status_code=500, detail=f"invalid_publishedAt:{s}")


def _bucket_hours_from_hours(hours_since_publish: float) -> int:
    """
    Buckets simples: 1h, 12h, 24h, 72h, 168h (7d)
    On choisit le bucket "le plus proche" sans être trop strict.
    """
    targets = [1, 12, 24, 72, 168]
    h = max(0.0, float(hours_since_publish))
    # nearest target
    return int(min(targets, key=lambda t: abs(t - h)))


async def collect_snapshot_for_video_id(
    yt_key: str,
    video_id: str,
) -> Dict[str, Any]:
    """
    Fetch video stats, compute timing, insert into video_snapshots.
    """
    vid = (video_id or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id_required")

    try:
        items = await get_videos(yt_key, [vid])
    except YouTubeAPIError as e:
        # keep same spirit as main.py
        if e.status_code == 403 and e.reason == "quotaExceeded":
            raise HTTPException(status_code=429, detail="YouTube quota exceeded for this API key.")
        if e.status_code == 403:
            raise HTTPException(status_code=403, detail=f"YouTube API forbidden. Reason={e.reason}.")
        raise HTTPException(status_code=502, detail=f"YouTube API error: {str(e)}")

    if not items:
        raise HTTPException(status_code=404, detail="video_not_found")

    it = items[0] or {}
    sn = it.get("snippet", {}) or {}
    st = it.get("statistics", {}) or {}

    channel_id = (sn.get("channelId") or "").strip()
    published_at = (sn.get("publishedAt") or "").strip()
    if not channel_id or not published_at:
        raise HTTPException(status_code=500, detail="missing_snippet_fields")

    views = int(st.get("viewCount", 0) or 0)
    likes = st.get("likeCount", None)
    comments = st.get("commentCount", None)

    likes_i = int(likes) if likes is not None else None
    comments_i = int(comments) if comments is not None else None

    now_utc = datetime.now(timezone.utc)
    pub_dt = _parse_iso_dt(published_at)

    hours_since = (now_utc - pub_dt).total_seconds() / 3600.0
    bucket_hours = _bucket_hours_from_hours(hours_since)

    snapshot_at = now_utc.isoformat().replace("+00:00", "Z")

    inserted_id = insert_video_snapshot(
        video_id=vid,
        channel_id=channel_id,
        snapshot_at=snapshot_at,
        published_at=published_at,
        hours_since_publish=float(hours_since),
        bucket_hours=int(bucket_hours),
        views=int(views),
        likes=likes_i,
        comments=comments_i,
    )

    return {
        "ok": True,
        "inserted_id": inserted_id,
        "video_id": vid,
        "channel_id": channel_id,
        "published_at": published_at,
        "snapshot_at": snapshot_at,
        "hours_since_publish": round(float(hours_since), 4),
        "bucket_hours": int(bucket_hours),
        "views": int(views),
        "likes": likes_i,
        "comments": comments_i,
    }


async def collect_snapshots_for_video_ids(
    yt_key: str,
    video_ids: List[str],
) -> Dict[str, Any]:
    ids = [v.strip() for v in (video_ids or []) if isinstance(v, str) and v.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="video_ids_required")

    results = []
    errors = []

    for vid in ids[:50]:
        try:
            results.append(await collect_snapshot_for_video_id(yt_key, vid))
        except HTTPException as e:
            errors.append({"video_id": vid, "status_code": e.status_code, "detail": e.detail})
        except Exception as e:
            errors.append({"video_id": vid, "status_code": 500, "detail": f"unexpected_error:{str(e)[:200]}"})

    return {
        "ok": True,
        "count_ok": len(results),
        "count_errors": len(errors),
        "results": results,
        "errors": errors,
    }