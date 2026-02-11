# backend/db.py
import os
import sqlite3
import secrets
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple

from fastapi import HTTPException

# =========================================================
# CONFIG
# =========================================================
DB_PATH = os.getenv("DB_PATH", "backend/app.db")


# =========================================================
# HELPERS
# =========================================================
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def _now_ts() -> int:
    return int(datetime.utcnow().timestamp())


def _today_str() -> str:
    return date.today().isoformat()


# =========================================================
# INIT + MIGRATIONS
# =========================================================
def init_db() -> None:
    """
    Initialise + migre automatiquement la DB.
    """
    conn = _conn()
    cur = conn.cursor()

    # ---------------- meta
    if not _table_exists(conn, "meta"):
        cur.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '1')"
        )

    # ---------------- users
    if not _table_exists(conn, "users"):
        cur.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            created_at INTEGER NOT NULL,
            daily_quota INTEGER NOT NULL DEFAULT 50,
            youtube_api_key TEXT
        )
        """)
    else:
        cols = _table_columns(conn, "users")
        if "daily_quota" not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN daily_quota INTEGER NOT NULL DEFAULT 50")
        if "youtube_api_key" not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN youtube_api_key TEXT")

    # ---------------- usage
    if not _table_exists(conn, "usage"):
        cur.execute("""
        CREATE TABLE usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_token TEXT NOT NULL,
            day TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL,
            UNIQUE(user_token, day)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_day ON usage(user_token, day)")
    else:
        cols = _table_columns(conn, "usage")
        if "count" not in cols:
            cur.execute("ALTER TABLE usage ADD COLUMN count INTEGER NOT NULL DEFAULT 0")
        if "updated_at" not in cols:
            cur.execute("ALTER TABLE usage ADD COLUMN updated_at INTEGER NOT NULL DEFAULT 0")

    # ---------------- cache
    if not _table_exists(conn, "cache"):
        cur.execute("""
        CREATE TABLE cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """)
    else:
        cols = _table_columns(conn, "cache")
        if "value" not in cols or "created_at" not in cols:
            cur.execute("DROP TABLE cache")
            cur.execute("""
            CREATE TABLE cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """)

    # ---------------- video_metrics
    if not _table_exists(conn, "video_metrics"):
        cur.execute("""
        CREATE TABLE video_metrics (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            channel_title TEXT,
            title TEXT,
            published_at TEXT,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            duration_seconds INTEGER,
            fetched_at INTEGER NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_metrics_channel ON video_metrics(channel_id)")
    else:
        cols = _table_columns(conn, "video_metrics")
        if "channel_title" not in cols:
            cur.execute("ALTER TABLE video_metrics ADD COLUMN channel_title TEXT")
        if "title" not in cols:
            cur.execute("ALTER TABLE video_metrics ADD COLUMN title TEXT")

    # ---------------- video_features (optionnelle)
    if not _table_exists(conn, "video_features"):
        cur.execute("""
        CREATE TABLE video_features (
            video_id TEXT PRIMARY KEY,
            language TEXT,
            title_length INTEGER,
            has_number_in_title INTEGER,
            hook_strength REAL,
            pacing_score REAL,
            format_type TEXT,
            inferred_market TEXT,
            computed_at INTEGER NOT NULL
        )
        """)

    conn.commit()
    conn.close()


# =========================================================
# USERS / AUTH
# =========================================================
def create_user(daily_quota: int = 50) -> Dict[str, Any]:
    token = secrets.token_urlsafe(24)
    conn = _conn()
    conn.execute(
        "INSERT INTO users (token, created_at, daily_quota) VALUES (?, ?, ?)",
        (token, _now_ts(), int(daily_quota)),
    )
    conn.commit()
    conn.close()
    return {"token": token, "daily_quota": int(daily_quota)}


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    row = conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_user_youtube_api_key(token: str, youtube_api_key: Optional[str]) -> bool:
    conn = _conn()
    cur = conn.execute(
        "UPDATE users SET youtube_api_key=? WHERE token=?",
        (youtube_api_key, token),
    )
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def get_user_youtube_api_key(token: str) -> Optional[str]:
    u = get_user_by_token(token)
    return u.get("youtube_api_key") if u else None


# aliases compat main.py
def set_user_youtube_key(token: str, youtube_api_key: Optional[str]) -> bool:
    return set_user_youtube_api_key(token, youtube_api_key)


def get_user_youtube_key(token: str) -> Optional[str]:
    return get_user_youtube_api_key(token)


# =========================================================
# USAGE / QUOTAS
# =========================================================
def get_daily_usage(token: str, day: Optional[str] = None) -> int:
    day = day or _today_str()
    conn = _conn()
    row = conn.execute(
        "SELECT count FROM usage WHERE user_token=? AND day=?",
        (token, day),
    ).fetchone()
    conn.close()
    return int(row["count"]) if row else 0


def increment_usage(
    token: str,
    amount: int = 1,
    day: Optional[str] = None,
) -> Tuple[bool, str]:
    day = day or _today_str()
    user = get_user_by_token(token)
    if not user:
        return False, "invalid_token"

    quota = int(user.get("daily_quota", 0))
    used = get_daily_usage(token, day)

    if used + amount > quota:
        return False, "quota_exceeded"

    conn = _conn()
    cur = conn.cursor()

    row = cur.execute(
        "SELECT count FROM usage WHERE user_token=? AND day=?",
        (token, day),
    ).fetchone()

    if row:
        cur.execute(
            "UPDATE usage SET count=?, updated_at=? WHERE user_token=? AND day=?",
            (used + amount, _now_ts(), token, day),
        )
    else:
        cur.execute(
            "INSERT INTO usage (user_token, day, count, updated_at) VALUES (?, ?, ?, ?)",
            (token, day, int(amount), _now_ts()),
        )

    conn.commit()
    conn.close()
    return True, "ok"


def incr_daily_usage_or_raise(
    token: str,
    amount: int = 1,
    day: Optional[str] = None,
) -> None:
    ok, reason = increment_usage(token, amount, day)
    if ok:
        return
    if reason == "invalid_token":
        raise HTTPException(status_code=401, detail="Invalid user token")
    if reason == "quota_exceeded":
        raise HTTPException(status_code=429, detail="Daily quota exceeded")
    raise HTTPException(status_code=400, detail=f"Usage error: {reason}")


# =========================================================
# CACHE
# =========================================================
def cache_get(key: str, ttl_seconds: int = 21600) -> Optional[Any]:
    conn = _conn()
    row = conn.execute(
        "SELECT value, created_at FROM cache WHERE key=?",
        (key,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    if _now_ts() - int(row["created_at"]) > ttl_seconds:
        return None

    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]


def cache_set(key: str, value: Any) -> None:
    conn = _conn()
    conn.execute(
        "REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
        (key, json.dumps(value), _now_ts()),
    )
    conn.commit()
    conn.close()


# =========================================================
# VIDEO METRICS
# =========================================================
def upsert_video_metrics(metrics: Dict[str, Any]) -> None:
    conn = _conn()
    conn.execute(
        """
        INSERT INTO video_metrics (
            video_id, channel_id, channel_title, title,
            published_at, views, likes, comments,
            duration_seconds, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            channel_id=excluded.channel_id,
            channel_title=excluded.channel_title,
            title=excluded.title,
            published_at=excluded.published_at,
            views=excluded.views,
            likes=excluded.likes,
            comments=excluded.comments,
            duration_seconds=excluded.duration_seconds,
            fetched_at=excluded.fetched_at
        """,
        (
            metrics.get("video_id"),
            metrics.get("channel_id"),
            metrics.get("channel_title"),
            metrics.get("title"),
            metrics.get("published_at"),
            metrics.get("views"),
            metrics.get("likes"),
            metrics.get("comments"),
            metrics.get("duration_seconds"),
            _now_ts(),
        ),
    )
    conn.commit()
    conn.close()


# =========================================================
# VIDEO SNAPSHOTS
# =========================================================
def insert_video_snapshot(
    video_id: str,
    channel_id: str,
    snapshot_at: str,
    published_at: str,
    hours_since_publish: float,
    bucket_hours: int,
    views: int,
    likes: Optional[int],
    comments: Optional[int],
) -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO video_snapshots
        (video_id, channel_id, snapshot_at, published_at,
         hours_since_publish, bucket_hours, views, likes, comments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            channel_id,
            snapshot_at,
            published_at,
            hours_since_publish,
            bucket_hours,
            views,
            likes,
            comments,
        ),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid