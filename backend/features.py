from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

_NUM_RE = re.compile(r"\d")
_VS_RE = re.compile(r"\bvs\b", re.IGNORECASE)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")

def parse_iso_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def duration_bucket(seconds: Optional[int]) -> str:
    if seconds is None:
        return "unknown"
    s = int(seconds)
    if s <= 30:
        return "0-30s"
    if s <= 60:
        return "31-60s"
    if s <= 180:
        return "1-3m"
    if s <= 600:
        return "3-10m"
    return "10m+"

def is_short(seconds: Optional[int]) -> Optional[bool]:
    if seconds is None:
        return None
    return int(seconds) <= 60

def title_features(title: str) -> Dict[str, Any]:
    t = title or ""
    return {
        "title_length": len(t),
        "has_number": bool(_NUM_RE.search(t)),
        "has_vs": bool(_VS_RE.search(t)),
        "has_allcaps_word": bool(_ALLCAPS_RE.search(t)),
        "ends_with_exclamation": t.strip().endswith("!"),
        "question_mark": "?" in t,
    }

def publish_features(published_at: str) -> Dict[str, Any]:
    dt = parse_iso_dt(published_at)
    if not dt:
        return {"weekday": None, "hour_utc": None}
    return {"weekday": dt.weekday(), "hour_utc": dt.hour}

def basic_video_features(video: Dict[str, Any]) -> Dict[str, Any]:
    title = video.get("title") or ""
    published_at = video.get("published_at") or ""
    dur_s = video.get("duration_seconds")

    feats: Dict[str, Any] = {}
    feats.update(title_features(title))
    feats.update(publish_features(published_at))
    feats["duration_seconds"] = dur_s
    feats["duration_bucket"] = duration_bucket(dur_s)
    feats["is_short"] = is_short(dur_s)
    return feats
