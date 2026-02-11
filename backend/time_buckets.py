# backend/time_buckets.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utcnow_iso_z() -> str:
    """
    Return current UTC time as ISO string with 'Z' suffix.
    Example: '2026-02-11T12:34:56Z'
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_to_utc(dt_str: str) -> datetime:
    """
    Parse ISO8601 datetime string (supports trailing 'Z' or offsets) into aware UTC datetime.
    """
    s = (dt_str or "").strip()
    if not s:
        raise ValueError("empty datetime string")

    # Accept 'Z'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Treat naive as UTC (safe default for our use-case)
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def hours_since(published_at_iso: str, snapshot_at_iso: str) -> float:
    """
    Compute hours elapsed between published_at and snapshot_at.
    Both inputs are ISO strings (e.g., YouTube publishedAt + our utcnow_iso_z()).
    """
    pub = _parse_iso_to_utc(published_at_iso)
    snap = _parse_iso_to_utc(snapshot_at_iso)

    delta = snap - pub
    return max(0.0, delta.total_seconds() / 3600.0)


def bucket_for(hours_since_publish: float) -> int:
    """
    Map hours since publish to the nearest bucket used in DB:
      0–2h  -> 1
      2–6h  -> 3
      6–12h -> 8
      12–24h -> 16
      24–48h -> 36
      48–96h -> 72
      96h+ -> 168
    """
    h = float(hours_since_publish or 0.0)

    if h < 2:
        return 1
    if h < 6:
        return 3
    if h < 12:
        return 8
    if h < 24:
        return 16
    if h < 48:
        return 36
    if h < 96:
        return 72
    return 168