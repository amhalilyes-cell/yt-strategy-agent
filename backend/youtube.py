from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import hashlib
import httpx

from backend.db import cache_get, cache_set, upsert_video_metrics

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Cache TTL logique (on stocke pas l'expiration en DB ici; c'est un cache "best effort")
# Si tu veux du TTL strict: on l'ajoute dans db.py ensuite (created_at + check âge).
CACHE_VERSION = "v1"


class YouTubeAPIError(RuntimeError):
    def __init__(self, status_code: int, reason: str, message: str):
        self.status_code = status_code
        self.reason = reason
        self.message = message
        super().__init__(f"YouTube API error {status_code} ({reason}): {message}")


def _parse_yt_error(resp: httpx.Response) -> YouTubeAPIError:
    """
    Essaie de parser le JSON d'erreur YouTube:
    {
      "error": {
        "code": 403,
        "message": "...",
        "errors": [{"reason":"quotaExceeded", ...}]
      }
    }
    """
    status = resp.status_code
    reason = "unknown"
    msg = resp.text

    try:
        data = resp.json()
        err = (data or {}).get("error", {})
        msg = err.get("message") or msg
        errs = err.get("errors") or []
        if errs and isinstance(errs, list):
            reason = errs[0].get("reason") or reason
    except Exception:
        pass

    return YouTubeAPIError(status_code=status, reason=reason, message=str(msg))


def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(prefix: str, params: Dict[str, Any]) -> str:
    return f"yt:{CACHE_VERSION}:{prefix}:{_stable_hash(params)}"


def _iso8601_duration_to_seconds(duration: Optional[str]) -> Optional[int]:
    """
    Parse ISO 8601 duration like "PT1H2M10S" to seconds.
    Minimal parser (YouTube returns PT...).
    """
    if not duration or not isinstance(duration, str):
        return None
    if not duration.startswith("PT"):
        return None

    # PT#H#M#S
    h = m = s = 0
    num = ""
    for ch in duration[2:]:
        if ch.isdigit():
            num += ch
            continue
        if not num:
            continue
        val = int(num)
        num = ""
        if ch == "H":
            h = val
        elif ch == "M":
            m = val
        elif ch == "S":
            s = val
    return h * 3600 + m * 60 + s


async def search_videos(
    api_key: str,
    query: str,
    region: str = "US",
    relevance_language: Optional[str] = "en",
    max_results: int = 50,
    order: str = "date",
    safe_search: str = "none",
    page_token: Optional[str] = None,
) -> List[str]:
    """
    Retourne une liste de videoIds via /search.
    - Ajoute cache DB (global)
    """
    if not api_key:
        raise ValueError("Missing YouTube API key")
    if not query:
        return []

    max_results = max(1, min(50, int(max_results)))

    params: Dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": order,
        "regionCode": region,
        "safeSearch": safe_search,
        "key": api_key,
    }
    if relevance_language:
        params["relevanceLanguage"] = relevance_language
    if page_token:
        params["pageToken"] = page_token

    ck = _cache_key("search", {k: v for k, v in params.items() if k != "key"})
    cached = cache_get(ck)
    if isinstance(cached, list):
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{YOUTUBE_API_BASE}/search", params=params)

    if r.status_code >= 400:
        raise _parse_yt_error(r)

    data = r.json() or {}
    items = data.get("items", []) or []

    out: List[str] = []
    for it in items:
        vid = ((it.get("id") or {}).get("videoId")) if isinstance(it.get("id"), dict) else None
        if vid:
            out.append(vid)

    cache_set(ck, out)
    return out


async def get_videos(api_key: str, video_ids: List[str]) -> List[Dict[str, Any]]:
    """
    YouTube API: /videos accepte jusqu’à 50 IDs par requête.
    - Cache DB par chunk
    - Upsert metrics (video_metrics) pour capitaliser et réduire quota futur
    """
    if not api_key:
        raise ValueError("Missing YouTube API key")
    if not video_ids:
        return []

    out: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]

            params = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk),
                "key": api_key,
            }

            ck = _cache_key("videos", {"part": params["part"], "id": params["id"]})
            cached = cache_get(ck)
            if isinstance(cached, dict) and isinstance(cached.get("items"), list):
                items = cached["items"]
            else:
                r = await client.get(f"{YOUTUBE_API_BASE}/videos", params=params)
                if r.status_code >= 400:
                    raise _parse_yt_error(r)
                data = r.json() or {}
                items = data.get("items", []) or []
                cache_set(ck, {"items": items})

            # Capitalisation DB (metrics brutes)
            for it in items:
                try:
                    video_id = it.get("id")
                    snippet = it.get("snippet") or {}
                    stats = it.get("statistics") or {}
                    content = it.get("contentDetails") or {}

                    duration_seconds = _iso8601_duration_to_seconds(content.get("duration"))

                    views = stats.get("viewCount")
                    likes = stats.get("likeCount")
                    comments = stats.get("commentCount")

                    metrics = {
                        "video_id": video_id,
                        "channel_id": snippet.get("channelId"),
                        "published_at": snippet.get("publishedAt"),
                        "views": int(views) if views is not None else None,
                        "likes": int(likes) if likes is not None else None,
                        "comments": int(comments) if comments is not None else None,
                        "duration_seconds": duration_seconds,
                    }
                    if video_id:
                        upsert_video_metrics(metrics)
                except Exception:
                    # On ne casse pas la requête pour un seul item mal formé
                    pass

            out.extend(items)

    return out