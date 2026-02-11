# backend/db_read.py
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

# IMPORTANT: on lit la même DB que backend/db.py
DB_PATH = os.getenv("DB_PATH", "backend/app.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_video_metrics(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Retourne une ligne de video_metrics.
    Compatible avec l'ancien schéma (sans title/channel_title) ET le nouveau.
    """
    conn = _conn()
    try:
        row = conn.execute(
            """
            SELECT
                video_id,
                channel_id,
                channel_title,
                title,
                published_at,
                views,
                likes,
                comments,
                duration_seconds,
                fetched_at
            FROM video_metrics
            WHERE video_id=?
            """,
            (video_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        # fallback si colonnes pas encore migrées
        row = conn.execute(
            """
            SELECT
                video_id,
                channel_id,
                published_at,
                views,
                likes,
                comments,
                duration_seconds,
                fetched_at
            FROM video_metrics
            WHERE video_id=?
            """,
            (video_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        d.setdefault("title", None)
        d.setdefault("channel_title", None)
        return d

    conn.close()
    return dict(row) if row else None


def recent_channel_metrics(
    channel_id: str,
    limit: int = 30,
    exclude_video_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retourne les dernières videos d'une chaîne (pour baseline).
    Compatible ancien/nouveau schéma.
    """
    conn = _conn()
    lim = max(1, int(limit))

    try:
        if exclude_video_id:
            rows = conn.execute(
                """
                SELECT
                    video_id,
                    channel_id,
                    channel_title,
                    title,
                    published_at,
                    views,
                    likes,
                    comments,
                    duration_seconds,
                    fetched_at
                FROM video_metrics
                WHERE channel_id=? AND video_id!=?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (channel_id, exclude_video_id, lim),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    video_id,
                    channel_id,
                    channel_title,
                    title,
                    published_at,
                    views,
                    likes,
                    comments,
                    duration_seconds,
                    fetched_at
                FROM video_metrics
                WHERE channel_id=?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (channel_id, lim),
            ).fetchall()

        conn.close()
        return [dict(r) for r in rows]

    except sqlite3.OperationalError:
        # fallback si colonnes pas encore migrées
        if exclude_video_id:
            rows = conn.execute(
                """
                SELECT
                    video_id,
                    channel_id,
                    published_at,
                    views,
                    likes,
                    comments,
                    duration_seconds,
                    fetched_at
                FROM video_metrics
                WHERE channel_id=? AND video_id!=?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (channel_id, exclude_video_id, lim),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    video_id,
                    channel_id,
                    published_at,
                    views,
                    likes,
                    comments,
                    duration_seconds,
                    fetched_at
                FROM video_metrics
                WHERE channel_id=?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (channel_id, lim),
            ).fetchall()

        conn.close()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d.setdefault("title", None)
            d.setdefault("channel_title", None)
            out.append(d)
        return out


def get_baseline_median_views(
    channel_id: str,
    is_short: bool,
    bucket_hours: int,
    limit: int = 10,
) -> float:
    """
    Baseline robuste (médiane) des vues d'une chaîne, par format (short/long)
    et bucket temporel (1/12/24/72/168h).
    On prend les N derniers snapshots correspondant (par défaut 10).

    ⚠️ Requiert la table `video_snapshots`.
    """
    ch = (channel_id or "").strip()
    if not ch:
        return 0.0

    conn = _conn()
    cur = conn.cursor()

    # Join sur video_metrics pour distinguer short vs long via duration_seconds
    cur.execute(
        """
        SELECT vs.views
        FROM video_snapshots vs
        JOIN video_metrics vm ON vm.video_id = vs.video_id
        WHERE
            vs.channel_id = ?
            AND vs.bucket_hours = ?
            AND (
                (? = 1 AND COALESCE(vm.duration_seconds, 0) <= 60)
                OR
                (? = 0 AND COALESCE(vm.duration_seconds, 0) > 60)
            )
        ORDER BY vs.snapshot_at DESC
        LIMIT ?
        """,
        (ch, int(bucket_hours), int(is_short), int(is_short), int(limit)),
    )

    rows = [r[0] for r in cur.fetchall() if r and r[0] is not None]
    conn.close()

    if not rows:
        return 0.0

    rows.sort()
    n = len(rows)
    mid = n // 2

    if n % 2 == 1:
        return float(rows[mid])
    return float((rows[mid - 1] + rows[mid]) / 2)