# backend/main.py
import os
import re
import sqlite3
import math
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
import uuid

from typing import List, Literal, Optional, Dict, Any, Tuple


from dotenv import load_dotenv

from pydantic import BaseModel, Field

from backend.youtube import search_videos, get_videos, YouTubeAPIError, YOUTUBE_API_BASE
from backend.llm import make_insights, make_pack
from backend.scoring import score_from_db
from backend.features import basic_video_features
from backend.db_read import get_video_metrics, recent_channel_metrics

from backend.db import (
    init_db,
    create_user,
    get_user_by_token,
    set_user_youtube_key,
    get_user_youtube_key,
    get_daily_usage,
    incr_daily_usage_or_raise,
    cache_get,
    cache_set,
    upsert_video_metrics,
    DB_PATH,
)

# ✅ simulation premium (modes + monte carlo)
from backend.simulate import PlaybookSignals, simulate_playbook

# ✅ snapshots collector
from backend.snapshots import collect_snapshots_for_video_ids

# ✅ BLUEPRINT V2 builder
from backend.blueprint_v2 import build_blueprint_v2

from fastapi.middleware.cors import CORSMiddleware

# ----- ENV -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# ✅ FIX: force .env to override any already-exported env vars
load_dotenv(ENV_PATH, override=True)

ADMIN_TOKEN = os.getenv("API_AUTH_TOKEN")
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "demo-public")
DEFAULT_YT_KEY = os.getenv("YOUTUBE_API_KEY")
CACHE_TTL = int(os.getenv("YT_CACHE_TTL_SECONDS", str(60 * 60 * 6)))  # 6h

# ✅ IMPORTANT: app doit exister AVANT add_middleware
app = FastAPI(title="YT Strategy Agent (Multi-User)")

# ✅ CORS (pour que Vercel puisse appeler Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yt-strategy-agent-yvvz.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    # optionnel mais très utile pour previews Vercel :
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    init_db()

# =========================
# SAAS AUTH (EMAIL + TRIAL 7J) — AJOUT ONLY
# =========================
SESSION_COOKIE = "ysa_session"
TRIAL_DAYS = 7

SESSIONS: Dict[str, str] = {}
USERS_TRIAL: Dict[str, Dict[str, Any]] = {}

def saas_now():
    return datetime.utcnow()

class MagicLoginReq(BaseModel):
    email: str
    
@app.post("/auth/magic/request")
def auth_magic_request(payload: MagicLoginReq, response: Response):
    email = (payload.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email is required")

    if email not in USERS_TRIAL:
        USERS_TRIAL[email] = {
            "email": email,
            "created_at": saas_now(),
            "trial_until": saas_now() + timedelta(days=TRIAL_DAYS),
        }

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = email

    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="none",
        secure=True,   # en prod HTTPS ok
    )

    # ✅ au lieu de "lien envoyé", on dit "ok + redirect"
    return {"ok": True, "redirect_to": "/app"}



@app.get("/auth/me")
def auth_me(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or session_id not in SESSIONS:
        return {"authenticated": False}

    email = SESSIONS[session_id]
    user = USERS_TRIAL.get(email)
    if not user:
        return {"authenticated": False}

    trial_active = saas_now() < user["trial_until"]

    return {
        "authenticated": True,
        "email": email,
        "trial_active": trial_active,
        "trial_until": user["trial_until"].isoformat(),
    }


@app.post("/auth/logout")
def auth_logout(request: Request, response: Response):
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        SESSIONS.pop(session_id, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}

# -----------------------
# Small safe helpers (IMPORTANT: prevent None crashes)
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------
# Auth helpers
# -----------------------
def _norm_token(s: str | None) -> str:
    """
    Normalise un token venant de header ou .env:
    - strip espaces / \\n
    - enlève guillemets "..." ou '...'
    """
    if not s:
        return ""
    return s.strip().strip('"').strip("'").strip()


def require_user(x_auth_token: str | None):
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="Missing X-Auth-Token")

    token = _norm_token(x_auth_token)
    admin = _norm_token(ADMIN_TOKEN)

    # ✅ ADMIN TOKEN = toujours valide
    if admin and token == admin:
        return {
            "token": token,
            "daily_quota": 999999,
            "plan": "admin",
        }

    # ✅ DEMO TOKEN = accès limité sans DB (utile pour landing)
    demo = _norm_token(DEMO_TOKEN)
    if demo and token == demo:
        return {
            "token": "demo",
            "daily_quota": 3,  # ✅ limite/jour demo (mets 3 ou 5)
            "plan": "demo",
        }

    # USER normal (DB)
    u = get_user_by_token(token)
    if not u:
        raise HTTPException(status_code=401, detail="Invalid user token")

    return u


def require_admin(x_auth_token: str | None):
    admin = _norm_token(ADMIN_TOKEN)
    if not admin:
        raise HTTPException(status_code=500, detail="API_AUTH_TOKEN manquant dans .env (admin)")

    token = _norm_token(x_auth_token)
    if not token or token != admin:
        raise HTTPException(status_code=401, detail="Unauthorized (admin)")


def pick_youtube_key(user_token: str) -> str:
    user_key = get_user_youtube_key(user_token)
    if user_key:
        return user_key
    if DEFAULT_YT_KEY:
        return DEFAULT_YT_KEY
    raise HTTPException(
        status_code=500,
        detail="No YOUTUBE_API_KEY available. User must set one via /user/set_youtube_key (BYO key).",
    )


# -----------------------
# ✅ SaaS plan inference (IMPORTANT)
# -----------------------
def infer_plan_from_user(u: Dict[str, Any]) -> str:
    """
    Si ta DB ne stocke pas "plan", on infère depuis le quota.
    - admin => admin
    - gros quota => premium/agency
    - sinon free
    """
    # admin
    tok = _norm_token(u.get("token"))
    if tok and _norm_token(ADMIN_TOKEN) and tok == _norm_token(ADMIN_TOKEN):
        return "admin"
    if u.get("plan") == "admin":
        return "admin"

    dq = int(u.get("daily_quota") or 0)
    if dq >= 200:
        return "premium"
    if dq >= 100:
        return "agency"
    return "free"


# -----------------------
# Duration parser
# -----------------------
_DUR_RE = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def iso8601_duration_to_seconds(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str):
        return None
    m = _DUR_RE.match(s.strip())
    if not m:
        return None
    h = int(m.group(1) or 0)
    mn = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return h * 3600 + mn * 60 + sec


# -----------------------
# Outlier / playbook helpers
# -----------------------
def compute_replicability_score(item: Dict[str, Any], feats: Dict[str, Any]) -> int:
    """
    Score 0-100 (simple et robuste).
    """
    score = 0

    if item.get("confidence") == "high":
        score += 30
    if _safe_float(item.get("outlier_ratio"), 0.0) >= 10:
        score += 25
    if feats.get("is_short") is True:
        score += 15
    if feats.get("duration_bucket") == "31-60s":
        score += 10
    if _safe_int(item.get("baseline_n"), 0) >= 20:
        score += 10

    return min(100, int(score))


def detect_format_shift(outlier_feats: Dict[str, Any], baseline_feats: Dict[str, Any]) -> List[str]:
    """
    Compare des dimensions simples. Si >=2 changes → "format shift".
    """
    keys = ["duration_bucket", "is_short", "hour_utc", "weekday"]
    shifts: List[str] = []
    for k in keys:
        if outlier_feats.get(k) != baseline_feats.get(k):
            shifts.append(k)
    return shifts


def _baseline_proxy_features(hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Proxy baseline: on prend une vidéo "récente" comme ancre (simple),
    car on n'a pas encore de stats de médiane en DB.
    """
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


def _recent_channel_ids_from_db(limit: int = 10) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT channel_id
        FROM video_metrics
        WHERE channel_id IS NOT NULL AND channel_id != ''
        ORDER BY fetched_at DESC
        LIMIT ?
    """,
        (int(limit),),
    )
    out = [r[0] for r in cur.fetchall() if r and r[0]]
    conn.close()
    return out


# -----------------------
# MODELS
# -----------------------
class AdminCreateUserReq(BaseModel):
    name: Optional[str] = None
    plan: str = "free"
    daily_limit: int = 30


class SetYTKeyReq(BaseModel):
    youtube_api_key: str


class TrendToPackReq(BaseModel):
    niche: str
    objectif: Literal["visibilite", "leads", "autorite", "vente", "lancement"]
    niveau: Literal["debutant", "intermediaire", "avance"]
    freq: int = 3
    region: str = "US"
    relevance_language: Optional[str] = "en"
    base_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    pool_size: int = 40
    max_age_days: Optional[int] = 90
    max_queries: int = 6


class ScoreReq(BaseModel):
    video_ids: List[str] = Field(default_factory=list)
    baseline_lookback: int = 30


class BootstrapChannelsReq(BaseModel):
    channel_ids: List[str] = Field(default_factory=list)
    per_channel_videos: int = 30
    region: str = "US"
    relevance_language: Optional[str] = "en"


class ExplainOutlierReq(BaseModel):
    video_id: str
    baseline_lookback: int = 30


class PlaybookReq(BaseModel):
    video_id: str
    baseline_lookback: int = 30


class SimulatePlaybookReq(BaseModel):
    video_id: str
    n_videos: int = Field(default=5, ge=1, le=200)
    timeframe_days: int = Field(default=7, ge=1, le=365)
    mode: Optional[Literal["safe", "base", "aggressive"]] = "base"


class SimulatePortfolioReq(BaseModel):
    video_ids: List[str] = Field(default_factory=list)
    n_videos: int = Field(default=5, ge=1, le=200)
    timeframe_days: int = Field(default=7, ge=1, le=365)
    mode: Optional[Literal["safe", "base", "aggressive"]] = "base"
    runs: int = Field(default=2000, ge=200, le=10000)


# ✅ blueprint endpoints
class BlueprintReq(BaseModel):
    video_id: str
    baseline_lookback: int = 30


class BlueprintBatchReq(BaseModel):
    video_ids: List[str] = Field(default_factory=list)
    baseline_lookback: int = 30
    max_items: int = Field(default=50, ge=1, le=200)


# ✅ snapshots collect
class SnapshotsCollectReq(BaseModel):
    video_ids: List[str] = Field(default_factory=list)


# ✅ BLUEPRINT V2 req
class BlueprintV2Req(BaseModel):
    video_id: str
    baseline_lookback: int = 30
    vertical: str = "saas"
    goal: Literal["visibilite", "leads", "autorite", "vente", "lancement"] = "leads"
    language: str = "fr"


# -----------------------
# BASIC
# -----------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
    admin_clean = _norm_token(ADMIN_TOKEN)
    return {
        "main_file": __file__,
        "env_loaded_from": ENV_PATH,
        "has_admin_token": bool(admin_clean),
        "admin_token_len": len(admin_clean),
        "admin_token_tail": admin_clean[-6:] if admin_clean else None,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_default_youtube_key": bool(DEFAULT_YT_KEY),
        "cache_ttl_seconds": CACHE_TTL,
        "db_path": DB_PATH,
    }


@app.get("/debug/auth")
def debug_auth(x_auth_token: str | None = Header(default=None)):
    tok = _norm_token(x_auth_token)
    adm = _norm_token(ADMIN_TOKEN)
    return {
        "got_header": bool(x_auth_token),
        "token_len": len(tok),
        "token_tail": tok[-6:] if tok else None,
        "admin_len": len(adm),
        "admin_tail": adm[-6:] if adm else None,
        "equals_admin": bool(tok and adm and tok == adm),
    }


# -----------------------
# ADMIN
# -----------------------
@app.post("/admin/create_user")
def admin_create_user(req: AdminCreateUserReq, x_auth_token: str | None = Header(default=None)):
    require_admin(x_auth_token)
    u = create_user(daily_quota=int(req.daily_limit))
    return {"user_token": u["token"], "daily_limit": u["daily_quota"], "plan": req.plan, "name": req.name}


# -----------------------
# USER
# -----------------------
@app.post("/user/set_youtube_key")
def user_set_youtube_key(req: SetYTKeyReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    set_user_youtube_key(token, req.youtube_api_key.strip())
    return {"ok": True}


@app.get("/usage")
def usage(x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    limit = int(u.get("daily_quota", 999999))
    plan_inferred = u.get("plan") or u.get("plan_inferred") or infer_plan_from_user(u)
    return {"today_count": get_daily_usage(token), "daily_limit": limit, "plan_inferred": plan_inferred}


@app.get("/user/me")
def user_me(x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    plan = u.get("plan") or u.get("plan_inferred") or infer_plan_from_user(u)
    return {
        "ok": True,
        "token_tail": token[-6:] if token else None,
        "daily_limit": int(u.get("daily_quota") or 0),
        "today_count": get_daily_usage(token),
        "plan_inferred": plan,
        "has_youtube_key": bool(get_user_youtube_key(token) or DEFAULT_YT_KEY),
    }


# -----------------------
# SCORING
# -----------------------
@app.get("/score")
def score_get(
    limit: int = Query(default=50, ge=1, le=500),
    baseline_lookback: int = Query(default=30, ge=3, le=200),
    x_auth_token: str | None = Header(default=None),
):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)
    return score_from_db(video_ids=None, limit=int(limit), baseline_lookback=int(baseline_lookback))


@app.post("/score")
def score_post(req: ScoreReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)
    ids = [x for x in (req.video_ids or []) if isinstance(x, str) and x.strip()]
    return score_from_db(video_ids=ids if ids else None, limit=50, baseline_lookback=int(req.baseline_lookback))


# -----------------------
# EXPLAIN OUTLIER
# -----------------------
@app.post("/outlier/explain")
def explain_outlier(req: ExplainOutlierReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    vid = (req.video_id or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id is required")

    vm = get_video_metrics(vid)
    if not vm:
        raise HTTPException(status_code=404, detail="video_id not found in video_metrics (run trend/bootstrap first)")

    scored = score_from_db(video_ids=[vid], limit=10, baseline_lookback=int(req.baseline_lookback))
    item = (scored.get("items") or [None])[0]
    if not item:
        raise HTTPException(status_code=500, detail="scoring failed")

    channel_id = vm.get("channel_id")
    hist = recent_channel_metrics(channel_id, limit=int(req.baseline_lookback), exclude_video_id=vid)

    v_feats = basic_video_features(
        {
            "title": vm.get("title") or "",
            "published_at": vm.get("published_at"),
            "duration_seconds": vm.get("duration_seconds"),
        }
    )

    baseline_summary = {
        "baseline_vph": item.get("baseline_vph"),
        "baseline_n": item.get("baseline_n"),
        "confidence": item.get("confidence"),
        "note": "Title-based explanations available once you re-bootstrap/trend after migration.",
    }

    actions = []
    if item.get("confidence") == "low":
        actions.append("Increase baseline confidence: bootstrap more videos for this channel (raise baseline_n).")
    if _safe_float(item.get("outlier_ratio"), 0.0) >= 10:
        actions.append("This is a strong outlier vs channel baseline: reverse-engineer format + timing, then replicate.")
    if v_feats.get("is_short") is True:
        actions.append("Short detected: compare against channel Shorts baseline; test 3 variations of hook in first 1s.")
    elif v_feats.get("is_short") is False:
        actions.append("Long-form detected: compare intro pacing and structure vs channel’s median performers.")

    return {
        "ok": True,
        "video_id": vid,
        "channel_id": channel_id,
        "title": vm.get("title"),
        "channel_title": vm.get("channel_title"),
        "scoring": {
            "vph": item.get("vph"),
            "engagement": item.get("engagement"),
            "trending_score": item.get("trending_score"),
            "baseline_vph": item.get("baseline_vph"),
            "baseline_vph_clamped": item.get("baseline_vph_clamped"),
            "outlier_ratio": item.get("outlier_ratio"),
            "outlier_log10": item.get("outlier_log10"),
            "baseline_n": item.get("baseline_n"),
            "confidence": item.get("confidence"),
        },
        "features": v_feats,
        "baseline_summary": baseline_summary,
        "history_sample_size": len(hist),
        "actions": actions,
    }


# -----------------------
# OUTLIER PLAYBOOK
# -----------------------
@app.post("/outlier/playbook")
def outlier_playbook(req: PlaybookReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    vid = (req.video_id or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id is required")

    vm = get_video_metrics(vid)
    if not vm:
        raise HTTPException(status_code=404, detail="video_id not found in DB (run trend/bootstrap first)")

    scored = score_from_db(video_ids=[vid], limit=1, baseline_lookback=int(req.baseline_lookback))
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
    hist = recent_channel_metrics(channel_id, limit=int(req.baseline_lookback), exclude_video_id=vid)
    baseline_feats = _baseline_proxy_features(hist)

    format_shift_dims = detect_format_shift(feats, baseline_feats)
    replicability = compute_replicability_score(item, feats)

    is_short = feats.get("is_short") is True
    hour = feats.get("hour_utc")
    if isinstance(hour, int):
        post_time_utc = [max(0, hour - 1), min(23, hour + 1)]
    else:
        post_time_utc = [14, 17]

    playbook = {
        "strategy": "format_replication" if replicability >= 70 else "test_and_validate",
        "replicability_score": replicability,
        "recommended_frequency": "1–2 Shorts / day" if is_short else "2–3 videos / week",
        "format_rules": {
            "duration_seconds": [45, 60] if is_short else [300, 900],
            "post_time_utc": post_time_utc,
            "duration_bucket": feats.get("duration_bucket"),
            "title_style_hint": "reaction/mockery + 1 emoji (if Shorts)" if is_short else "clear promise + specificity",
        },
        "kill_conditions": {
            "vph_after_1h": f"< baseline × 2 (baseline_vph={item.get('baseline_vph')})",
            "confidence_required": "high",
        },
        "notes": [
            "This playbook is data-driven. For Shorts, watch time / completion matters more than likes/comments.",
            "If format_shift detected: replicate the format dimensions first (duration/timing/type).",
        ],
    }

    return {
        "ok": True,
        "video_id": vid,
        "channel_id": channel_id,
        "title": vm.get("title"),
        "channel_title": vm.get("channel_title"),
        "replicability_score": replicability,
        "format_shift": {
            "detected": len(format_shift_dims) >= 2,
            "dimensions": format_shift_dims,
            "baseline_proxy": baseline_feats,
            "outlier": feats,
        },
        "scoring": {
            "vph": item.get("vph"),
            "baseline_vph": item.get("baseline_vph"),
            "baseline_n": item.get("baseline_n"),
            "confidence": item.get("confidence"),
            "outlier_ratio": item.get("outlier_ratio"),
            "outlier_log10": item.get("outlier_log10"),
        },
        "playbook": playbook,
    }


# -----------------------
# PREMIUM: quantiles helper
# -----------------------
def _quantiles(xs: List[float], ps=(0.10, 0.50, 0.90)) -> Dict[str, float]:
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        if n == 0:
            return 0.0
        if n == 1:
            return float(s[0])
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(s[lo])
        w = idx - lo
        return float(s[lo] * (1 - w) + s[hi] * w)

    return {"min": q(ps[0]), "median": q(ps[1]), "max": q(ps[2])}


# -----------------------
# PREMIUM: /playbook/simulate  (cost=2)
# -----------------------
@app.post("/playbook/simulate")
def playbook_simulate(req: SimulatePlaybookReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=2)

    vid = (req.video_id or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="video_id is required")

    scored = score_from_db(video_ids=[vid], limit=1, baseline_lookback=30)
    item = (scored.get("items") or [None])[0]
    if not item:
        raise HTTPException(status_code=404, detail="video_id not found in scoring (ensure in video_metrics)")

    vm = get_video_metrics(vid)
    if not vm:
        raise HTTPException(status_code=404, detail="video_id not found in video_metrics (run trend/bootstrap first)")

    feats = basic_video_features(
        {
            "title": vm.get("title") or "",
            "published_at": vm.get("published_at"),
            "duration_seconds": vm.get("duration_seconds"),
        }
    )

    channel_id = vm.get("channel_id")
    hist = recent_channel_metrics(channel_id, limit=30, exclude_video_id=vid)
    baseline_feats = _baseline_proxy_features(hist)

    format_shift_dims = detect_format_shift(feats, baseline_feats)
    format_shift_detected = len(format_shift_dims) >= 2
    replicability = compute_replicability_score(item, feats)

    signals = PlaybookSignals(
        replicability_score=float(replicability),
        format_shift=bool(format_shift_detected),
        confidence=str(item.get("confidence", "low")),
        outlier_ratio=_safe_float(item.get("outlier_ratio"), 1.0),
        baseline_vph=_safe_float(item.get("baseline_vph"), 0.0),
        video_vph=_safe_float(item.get("vph"), 0.0),
        baseline_n=_safe_int(item.get("baseline_n"), 0),
    )

    return simulate_playbook(
        signals=signals,
        n_videos=int(req.n_videos),
        timeframe_days=int(req.timeframe_days),
        mode=req.mode or "base",
        runs=2000,
        return_runs=False,
    )


# -----------------------
# PREMIUM: /playbook/simulate/portfolio  (cost=2*N)
# -----------------------
@app.post("/playbook/simulate/portfolio")
def playbook_simulate_portfolio(req: SimulatePortfolioReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]

    video_ids = [v.strip() for v in (req.video_ids or []) if isinstance(v, str) and v.strip()]
    if not video_ids:
        raise HTTPException(status_code=400, detail="video_ids is required (non-empty list)")
    if len(video_ids) > 50:
        video_ids = video_ids[:50]

    amount = 2 * len(video_ids)
    incr_daily_usage_or_raise(token, amount=amount)

    runs = int(req.runs or 2000)

    items: List[Dict[str, Any]] = []
    runs_totals_matrix: List[List[float]] = []
    global_warnings = set()

    for vid in video_ids:
        scored = score_from_db(video_ids=[vid], limit=1, baseline_lookback=30)
        item = (scored.get("items") or [None])[0]
        if not item:
            raise HTTPException(status_code=404, detail=f"video_id not found in scoring:{vid}")

        vm = get_video_metrics(vid)
        if not vm:
            raise HTTPException(status_code=404, detail=f"video_id not found in video_metrics:{vid}")

        feats = basic_video_features(
            {
                "title": vm.get("title") or "",
                "published_at": vm.get("published_at"),
                "duration_seconds": vm.get("duration_seconds"),
            }
        )

        channel_id = vm.get("channel_id")
        hist = recent_channel_metrics(channel_id, limit=30, exclude_video_id=vid)
        baseline_feats = _baseline_proxy_features(hist)

        format_shift_dims = detect_format_shift(feats, baseline_feats)
        format_shift_detected = len(format_shift_dims) >= 2
        replicability = compute_replicability_score(item, feats)

        signals = PlaybookSignals(
            replicability_score=float(replicability),
            format_shift=bool(format_shift_detected),
            confidence=str(item.get("confidence", "low")),
            outlier_ratio=_safe_float(item.get("outlier_ratio"), 1.0),
            baseline_vph=_safe_float(item.get("baseline_vph"), 0.0),
            video_vph=_safe_float(item.get("vph"), 0.0),
            baseline_n=_safe_int(item.get("baseline_n"), 0),
        )

        res = simulate_playbook(
            signals=signals,
            n_videos=int(req.n_videos),
            timeframe_days=int(req.timeframe_days),
            mode=req.mode or "base",
            runs=runs,
            return_runs=True,
        )

        item_runs = (res.get("_runs") or {}).get("views_totals") or []
        if not item_runs:
            raise HTTPException(status_code=500, detail=f"simulation_missing_runs:{vid}")
        runs_totals_matrix.append(item_runs)

        res.pop("_runs", None)

        for w in (res.get("warning_signals") or []):
            global_warnings.add(w)

        items.append({"video_id": vid, **res})

    portfolio_totals: List[float] = []
    for r in range(runs):
        portfolio_totals.append(sum(row[r] for row in runs_totals_matrix))

    portfolio_q = _quantiles(portfolio_totals, (0.10, 0.50, 0.90))

    risks = [int(it.get("risk_score", 100)) for it in items]
    portfolio_risk = int(max(risks)) if risks else 100

    recs = [int(it.get("recommended_quota", 2)) for it in items]
    portfolio_rec = int(max(recs)) if recs else 2

    return {
        "portfolio_expected_views_total": portfolio_q,
        "portfolio_risk_score": portfolio_risk,
        "portfolio_recommended_quota": portfolio_rec,
        "items": items,
        "warning_signals": sorted(global_warnings),
        "assumptions": {
            "mode": req.mode or "base",
            "runs": runs,
            "n_items": len(items),
            "n_videos": int(req.n_videos),
            "timeframe_days": int(req.timeframe_days),
            "quota_cost_amount": amount,
        },
        "debug": {"aggregation": {"method": "runwise_sum", "runs": runs}},
    }


# -----------------------
# TRENDING (cached)
# -----------------------
async def compute_trending(
    yt_key: str,
    query: str,
    region: str,
    relevance_language: Optional[str],
    pool_size: int,
    max_age_days: Optional[int],
):
    cache_key = f"trend::{region}::{relevance_language}::{pool_size}::{max_age_days}::{query}".lower()
    cached = cache_get(cache_key, ttl_seconds=CACHE_TTL)
    if cached:
        return cached

    max_results = min(50, max(1, pool_size))

    try:
        video_ids = await search_videos(
            api_key=yt_key,
            query=query,
            region=region,
            relevance_language=relevance_language,
            max_results=max_results,
            order="date",
            safe_search="none",
            page_token=None,
        )
        video_ids = list(dict.fromkeys(video_ids))[:pool_size]
        items = await get_videos(yt_key, video_ids)

    except YouTubeAPIError as e:
        if e.status_code == 403 and e.reason == "quotaExceeded":
            raise HTTPException(status_code=429, detail="YouTube quota exceeded for this API key.")
        if e.status_code == 403:
            raise HTTPException(status_code=403, detail=f"YouTube API forbidden. Reason={e.reason}.")
        raise HTTPException(status_code=502, detail=f"YouTube API error: {str(e)}")

    top_videos = []
    db_written = 0

    for it in items:
        sn = it.get("snippet", {}) or {}
        st = it.get("statistics", {}) or {}
        cd = it.get("contentDetails", {}) or {}

        views = int(st.get("viewCount", 0) or 0)
        likes = int(st.get("likeCount", 0) or 0)
        comments = int(st.get("commentCount", 0) or 0)
        score = (likes * 1.0) + (comments * 2.5) + (views * 0.0001)

        vid = it.get("id")
        duration_iso = cd.get("duration")
        dur_s = iso8601_duration_to_seconds(duration_iso)

        if vid:
            upsert_video_metrics(
                {
                    "video_id": vid,
                    "channel_id": sn.get("channelId"),
                    "channel_title": sn.get("channelTitle"),
                    "title": sn.get("title"),
                    "published_at": sn.get("publishedAt"),
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "duration_seconds": dur_s,
                }
            )
            db_written += 1

        top_videos.append(
            {
                "video_id": vid,
                "title": sn.get("title"),
                "channel": sn.get("channelTitle"),
                "channel_id": sn.get("channelId"),
                "published_at": sn.get("publishedAt"),
                "duration": duration_iso,
                "duration_seconds": dur_s,
                "views": views,
                "likes": likes,
                "comments": comments,
                "score": round(score, 2),
            }
        )

    top_videos.sort(key=lambda x: x["score"], reverse=True)

    payload = {
        "query": query,
        "region": region,
        "relevance_language": relevance_language,
        "db_written": db_written,
        "top_videos": top_videos[: min(20, len(top_videos))],
    }

    cache_set(cache_key, payload)
    return payload


# -----------------------
# CHANNELS BOOTSTRAP
# -----------------------
@app.post("/channels/bootstrap")
async def channels_bootstrap(req: BootstrapChannelsReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    yt_key = pick_youtube_key(token)

    per_channel = max(1, min(50, int(req.per_channel_videos)))
    channel_ids = [c.strip() for c in (req.channel_ids or []) if c and c.strip()]
    if not channel_ids:
        channel_ids = _recent_channel_ids_from_db(limit=10)
    channel_ids = channel_ids[:10]

    total_upserts = 0
    details: List[Dict[str, Any]] = []

    import httpx

    for ch in channel_ids:
        params = {
            "part": "snippet",
            "channelId": ch,
            "order": "date",
            "type": "video",
            "maxResults": per_channel,
            "regionCode": req.region,
            "key": yt_key,
        }
        if req.relevance_language:
            params["relevanceLanguage"] = req.relevance_language

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{YOUTUBE_API_BASE}/search", params=params)

        if r.status_code >= 400:
            details.append({"channel_id": ch, "ok": False, "error": r.text[:200]})
            continue

        data = r.json() or {}
        items = data.get("items", []) or []
        video_ids = []
        for it in items:
            vid = ((it.get("id") or {}).get("videoId")) if isinstance(it.get("id"), dict) else None
            if vid:
                video_ids.append(vid)
        video_ids = list(dict.fromkeys(video_ids))[:per_channel]

        if not video_ids:
            details.append({"channel_id": ch, "ok": True, "videos": 0, "upserts": 0})
            continue

        try:
            vids = await get_videos(yt_key, video_ids)
        except YouTubeAPIError as e:
            if e.status_code == 403 and e.reason == "quotaExceeded":
                raise HTTPException(status_code=429, detail="YouTube quota exceeded while bootstrapping channels.")
            details.append({"channel_id": ch, "ok": False, "error": str(e)})
            continue

        upserts = 0
        for it in vids:
            sn = it.get("snippet", {}) or {}
            st = it.get("statistics", {}) or {}
            cd = it.get("contentDetails", {}) or {}

            views = int(st.get("viewCount", 0) or 0)
            likes = int(st.get("likeCount", 0) or 0)
            comments = int(st.get("commentCount", 0) or 0)

            vid = it.get("id")
            dur_s = iso8601_duration_to_seconds(cd.get("duration"))

            if vid:
                upsert_video_metrics(
                    {
                        "video_id": vid,
                        "channel_id": sn.get("channelId"),
                        "channel_title": sn.get("channelTitle"),
                        "title": sn.get("title"),
                        "published_at": sn.get("publishedAt"),
                        "views": views,
                        "likes": likes,
                        "comments": comments,
                        "duration_seconds": dur_s,
                    }
                )
                upserts += 1

        total_upserts += upserts
        details.append({"channel_id": ch, "ok": True, "videos": len(video_ids), "upserts": upserts})

    return {"ok": True, "channels": channel_ids, "total_upserts": total_upserts, "details": details}


# -----------------------
# SNAPSHOTS COLLECT
# -----------------------
@app.post("/snapshots/collect")
async def snapshots_collect(req: SnapshotsCollectReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    yt_key = pick_youtube_key(token)
    return await collect_snapshots_for_video_ids(yt_key=yt_key, video_ids=req.video_ids)


# -----------------------
# TREND → PACK
# -----------------------
@app.post("/trend_to_pack")
async def trend_to_pack(req: TrendToPackReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    daily_limit = int(u.get("daily_quota", 999999))

    incr_daily_usage_or_raise(token, amount=1)
    yt_key = pick_youtube_key(token)

    queries = [req.base_query] + [q for q in (req.expanded_queries or []) if q.strip()]
    queries = queries[: max(1, req.max_queries)]

    all_top = []
    db_written_total = 0

    for q in queries:
        part = await compute_trending(
            yt_key=yt_key,
            query=q,
            region=req.region,
            relevance_language=req.relevance_language,
            pool_size=req.pool_size,
            max_age_days=req.max_age_days,
        )
        db_written_total += int(part.get("db_written", 0) or 0)
        all_top.extend(part.get("top_videos", []))

    seen = set()
    uniq = []
    for v in sorted(all_top, key=lambda x: x.get("score", 0), reverse=True):
        vid = v.get("video_id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        uniq.append(v)
    top_videos = uniq[:20]

    videos_dump = "\n".join(
        f"- {v.get('title','')} | {v.get('channel','')} | views={v.get('views','')} likes={v.get('likes','')} comments={v.get('comments','')} | published={v.get('published_at','')}"
        for v in top_videos[:20]
    )

    insights = make_insights(
        niche=req.niche,
        objectif=req.objectif,
        niveau=req.niveau,
        freq=req.freq,
        videos_dump=videos_dump,
    )

    pack = make_pack(
        niche=req.niche,
        objectif=req.objectif,
        niveau=req.niveau,
        freq=req.freq,
        insights=insights,
    )

    return {
        "query": req.base_query,
        "region": req.region,
        "relevance_language": req.relevance_language,
        "top_videos": top_videos,
        "db_write": {"video_metrics_upserts": db_written_total},
        "insights": insights,
        "pack": pack,
        "usage": {"today_count": get_daily_usage(token), "daily_limit": daily_limit},
    }


# -----------------------
# ✅ BLUEPRINT (single + batch) V1
# -----------------------
def _blueprint_hook_examples() -> List[str]:
    return [
        "« STOP — si tu fais ça, tu perds de l’argent. »",
        "« La banque ne te le dira jamais : fais ça maintenant. »",
        "« 1 erreur = 1 mois d’épargne perdu. »",
    ]


def _blueprint_ideas(duration_target: Tuple[int, int], fmt: str) -> List[Dict[str, Any]]:
    return [
        {
            "idea": "Arnaque / piège en 1 phrase + solution",
            "hook_template": "« STOP — {PIEGE} te coûte {MONTANT}€ / mois. »",
            "format": fmt,
            "duration_target_seconds": [duration_target[0], duration_target[1]],
        },
        {
            "idea": "Checklist ultra rapide",
            "hook_template": "« 3 choses à faire avant de payer {CHOSE}… »",
            "format": fmt,
            "duration_target_seconds": [duration_target[0], duration_target[1]],
        },
        {
            "idea": "Mythe à débunker",
            "hook_template": "« Non, {CROYANCE}. Voici la vérité. »",
            "format": fmt,
            "duration_target_seconds": [duration_target[0], duration_target[1]],
        },
    ]


def build_blueprint(video_id: str, baseline_lookback: int) -> Dict[str, Any]:
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

    if fmt == "short":
        duration_target = (14, 18)
        avoid = ["intro", "context", "bonjour", "explication longue"]
        content_structure = [
            "pattern break (0–1s)",
            "micro-context (≤2s)",
            "proof / demo (3–8s)",
            "payoff (1 phrase claire)",
            "hard stop + CTA (commentaire / follow)",
        ]
    else:
        duration_target = (360, 720)
        avoid = ["intro trop longue", "mise en contexte inutile", "monologue sans preuve"]
        content_structure = [
            "hook (0–15s) : promesse claire + tension",
            "setup (15–45s) : pourquoi ça te concerne",
            "3 points (45s–80%) : preuve / exemple / action",
            "résumé (80–95%) : plan clair à appliquer",
            "CTA (95–100%) : commentaire / subscribe / lead magnet",
        ]

    spacing_hours = [8, 16] if fmt == "short" else [24, 72]
    videos_to_post = 5 if fmt == "short" else 3
    timeframe_days = 7

    baseline_vph = _safe_float(item.get("baseline_vph"), 0.0)
    if baseline_vph <= 0:
        baseline_vph = _safe_float(item.get("baseline_vph_clamped"), 0.0)

    kill_conditions = {
        "views_after_30min": f"< baseline_vph × 1.5 (baseline_vph={round(baseline_vph, 3)})",
        "views_after_2h": f"< baseline_vph × 2.0 (baseline_vph={round(baseline_vph, 3)})",
        "action": "kill / re-edit hook / repost variation",
    }

    format_shift_dims = detect_format_shift(feats, baseline_feats)
    warning_signals = []
    if len(format_shift_dims) >= 2:
        warning_signals.append("format_shift_detected")

    baseline_proxy_features = {
        "title_length": len((vm.get("title") or "").strip()),
        "has_number": bool(re.search(r"\d", (vm.get("title") or ""))),
        "has_vs": " vs " in (vm.get("title") or "").lower(),
        "has_allcaps_word": bool(re.search(r"\b[A-Z]{3,}\b", (vm.get("title") or ""))),
        "ends_with_exclamation": (vm.get("title") or "").strip().endswith("!"),
        "question_mark": "?" in (vm.get("title") or ""),
        "weekday": feats.get("weekday"),
        "hour_utc": feats.get("hour_utc"),
        "duration_seconds": dur_s,
        "duration_bucket": feats.get("duration_bucket"),
        "is_short": feats.get("is_short"),
    }

    return {
        "ok": True,
        "video_id": vid,
        "title": vm.get("title"),
        "channel_id": channel_id,
        "channel_title": vm.get("channel_title"),
        "format": fmt,
        "duration_target_seconds": [duration_target[0], duration_target[1]],
        "hook_structure": {
            "type": "instant_pattern_break" if fmt == "short" else "promise_plus_tension",
            "first_1s_goal": "surprise" if fmt == "short" else "clarity",
            "avoid": avoid,
            "examples": _blueprint_hook_examples(),
        },
        "content_structure": content_structure,
        "posting_strategy": {
            "videos_to_post": videos_to_post,
            "timeframe_days": timeframe_days,
            "spacing_hours": spacing_hours,
            "strategy": "replicate",
        },
        "kill_conditions": kill_conditions,
        "confidence": item.get("confidence", "low"),
        "derived_from": {
            "baseline_n": _safe_int(item.get("baseline_n"), 0),
            "baseline_vph": baseline_vph,
            "outlier_ratio": _safe_float(item.get("outlier_ratio"), 0.0),
            "vph": _safe_float(item.get("vph"), 0.0),
        },
        "baseline_proxy_features": baseline_proxy_features,
        "format_shift": {"detected": len(format_shift_dims) >= 2, "dimensions": format_shift_dims},
        "warning_signals": warning_signals,
        "ideas": _blueprint_ideas(duration_target, fmt),
    }


@app.post("/playbook/blueprint")
def playbook_blueprint(req: BlueprintReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)
    return build_blueprint(req.video_id, req.baseline_lookback)


@app.post("/playbook/blueprint/batch")
def playbook_blueprint_batch(req: BlueprintBatchReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    ids = [v.strip() for v in (req.video_ids or []) if isinstance(v, str) and v.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="video_ids is required (non-empty list)")

    max_items = int(req.max_items or 50)
    ids = ids[:max_items]
    lookback = int(req.baseline_lookback or 30)

    blueprints: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for vid in ids:
        try:
            blueprints.append(build_blueprint(vid, lookback))
        except HTTPException as e:
            errors.append({"video_id": vid, "status_code": e.status_code, "detail": e.detail})
        except Exception as e:
            errors.append({"video_id": vid, "status_code": 500, "detail": f"unexpected_error:{str(e)[:120]}"})

    def _rank(bp: Dict[str, Any]) -> Tuple[int, float, float]:
        conf = bp.get("confidence", "low")
        conf_score = 1 if conf == "high" else 0
        outlier = _safe_float((bp.get("derived_from") or {}).get("outlier_ratio"), 0.0)
        vph = _safe_float((bp.get("derived_from") or {}).get("vph"), 0.0)
        return (conf_score, outlier, vph)

    best_picks = sorted(blueprints, key=_rank, reverse=True)[: min(10, len(blueprints))]

    return {
        "ok": True,
        "count": len(blueprints),
        "baseline_lookback": lookback,
        "blueprints": blueprints,
        "best_picks": best_picks,
        "errors": errors,
        "assumptions": {"max_items": max_items, "quota_cost_amount": 1},
    }


# -----------------------
# ✅ BLUEPRINT V2 endpoint
# -----------------------
@app.post("/playbook/blueprint/v2")
def playbook_blueprint_v2(req: BlueprintV2Req, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]
    incr_daily_usage_or_raise(token, amount=1)

    return build_blueprint_v2(
        video_id=req.video_id,
        baseline_lookback=req.baseline_lookback,
        vertical=req.vertical,
        goal=req.goal,
        language=req.language,
    )

# =========================
# SAAS ENDPOINT — COOKIE + TRIAL (AJOUT ONLY)
# =========================
@app.post("/saas/playbook/blueprint/v2")
def saas_playbook_blueprint_v2(req: BlueprintV2Req, request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated")

    email = SESSIONS[session_id]
    user = USERS_TRIAL[email]

    if saas_now() > user["trial_until"]:
        raise HTTPException(status_code=402, detail="Trial expired")

    return build_blueprint_v2(
        video_id=req.video_id,
        baseline_lookback=req.baseline_lookback,
        vertical=req.vertical,
        goal=req.goal,
        language=req.language,
    )

# =========================================================
# ✅ NEW: Premium "All-in-one" Pack (7/30/90 jours + business vertical)
# =========================================================
class BusinessProfile(BaseModel):
    vertical: str = Field(default="saas")  # saas / agency / ecom / creator / infoproduct / local
    goal: Literal["visibilite", "leads", "autorite", "vente", "lancement"] = Field(default="leads")
    audience: str = Field(default="founders")
    language: str = Field(default="fr")  # fr / en etc.
    region: str = Field(default="FR")  # FR / US etc.


class CadenceProfile(BaseModel):
    shorts_per_week: int = Field(default=5, ge=0, le=40)
    longs_per_month: int = Field(default=4, ge=0, le=30)


class PremiumPackReq(BaseModel):
    seed_video_id: str
    timeframe_days: int = Field(default=30, ge=7, le=365)
    baseline_lookback: int = Field(default=30, ge=3, le=200)
    business: BusinessProfile = Field(default_factory=BusinessProfile)
    cadence: CadenceProfile = Field(default_factory=CadenceProfile)


def _spread_days(total_days: int, count: int) -> List[int]:
    if count <= 0:
        return []
    step = total_days / float(count)
    days = []
    for i in range(count):
        d = int(i * step) + 1
        if d > total_days:
            d = total_days
        days.append(d)
    return days


def _hooks_for_vertical(vertical: str, lang: str = "fr") -> List[str]:
    v = (vertical or "").strip().lower()

    if lang.lower().startswith("fr"):
        if v == "saas":
            return [
                "STOP — ton SaaS perd des clients à cause de ça.",
                "Si ton churn est haut, regarde ça (c’est souvent le vrai bug).",
                "Le hack le plus simple pour doubler tes leads B2B.",
                "Ta landing page tue tes conversions : voilà pourquoi.",
                "Fais ça avant de lancer une feature (sinon tu perds 1 mois).",
            ]
        if v == "agency":
            return [
                "STOP — tu vends mal ton offre (corrige ça).",
                "Si tes prospects ghostent : fais exactement ça.",
                "Le script simple pour closer sans forcer.",
                "Tu underprice ton service : voici le test.",
                "L’erreur n°1 en prospection (et comment la corriger).",
            ]
        if v == "ecom":
            return [
                "STOP — tu perds des ventes sur ta fiche produit (voilà où).",
                "Ton taux d’ajout au panier est bas ? check ça.",
                "3 images produit qui vendent (pas besoin d’ads).",
                "La promo qui marche même sans audience.",
                "Le détail qui augmente le panier moyen.",
            ]
        if v == "local":
            return [
                "STOP — ton business local rate des clients chaque semaine.",
                "Le truc le plus simple pour remplir ton planning.",
                "Si personne t’appelle : change ça sur Google.",
                "La preuve sociale qui fait exploser les RDV.",
                "Le message à envoyer pour convertir en 5 minutes.",
            ]
        return [
            "STOP — fais pas ça.",
            "Personne te dit ça, mais…",
            "3 erreurs qui te bloquent.",
            "Le truc simple qui change tout.",
            "Tu vas gagner du temps avec ça.",
        ]

    return [
        "STOP — you're doing this wrong.",
        "This is why you're not growing.",
        "Do this today (seriously).",
        "The mistake everyone makes.",
        "Here’s the simple fix.",
    ]


def _cta_for_goal(goal: str, lang: str = "fr") -> List[str]:
    g = (goal or "").strip().lower()
    if lang.lower().startswith("fr"):
        if g == "leads":
            return ["Commente DEMO", "Écris PLAN", "DM « GO »"]
        if g == "vente":
            return ["Lien en bio", "Commente PRIX", "DM « OFFRE »"]
        if g == "lancement":
            return ["Commente LISTE", "Active la cloche", "DM « LAUNCH »"]
        return ["Abonne-toi", "Like pour la suite", "Commente GO"]
    else:
        if g == "leads":
            return ["Comment DEMO", "Type PLAN", "DM 'GO'"]
        return ["Follow", "Like for part 2", "Comment GO"]


def _short_script_template(hook: str, cta: str) -> List[str]:
    return [
        f"0–1s: {hook}",
        "1–3s: micro-contexte (1 phrase max)",
        "3–8s: preuve / démo (exemple concret)",
        "8–13s: payoff (1 phrase actionnable)",
        f"13–16s: hard stop + CTA: {cta}",
    ]


def _long_outline_template(vertical: str, goal: str, cta: str, lang: str = "fr") -> List[str]:
    v = (vertical or "business").upper()
    if lang.lower().startswith("fr"):
        return [
            "0–15s: cold open (promesse + tension)",
            f"15–45s: le problème typique en {v}",
            "45s–70%: 3 leviers (avec exemples)",
            "70%–90%: plan d’action (checklist)",
            f"90%–100%: CTA: {cta}",
        ]
    return [
        "0–15s: cold open (promise + tension)",
        f"15–45s: the common problem in {v}",
        "45s–70%: 3 levers (with examples)",
        "70%–90%: action plan (checklist)",
        f"90%–100%: CTA: {cta}",
    ]


@app.post("/premium/pack")
def premium_pack(req: PremiumPackReq, x_auth_token: str | None = Header(default=None)):
    u = require_user(x_auth_token)
    token = u["token"]

    plan = u.get("plan") or u.get("plan_inferred") or infer_plan_from_user(u)
    if plan not in ["premium", "agency", "admin"]:
        raise HTTPException(status_code=403, detail="Premium plan required")

    incr_daily_usage_or_raise(token, amount=2)

    seed = (req.seed_video_id or "").strip()
    if not seed:
        raise HTTPException(status_code=400, detail="seed_video_id is required")

    seed_bp = build_blueprint(seed, int(req.baseline_lookback))

    timeframe = int(req.timeframe_days)
    weeks = int(math.ceil(timeframe / 7.0))

    shorts_count = weeks * int(req.cadence.shorts_per_week)
    longs_count = max(0, int(round((timeframe / 30.0) * int(req.cadence.longs_per_month))))

    if timeframe >= 30 and req.cadence.longs_per_month > 0 and longs_count == 0:
        longs_count = 1

    hook_bank = _hooks_for_vertical(req.business.vertical, req.business.language)
    cta_bank = _cta_for_goal(req.business.goal, req.business.language)

    short_days = _spread_days(timeframe, shorts_count)
    long_days = _spread_days(timeframe, longs_count)

    shorts: List[Dict[str, Any]] = []
    for i, d in enumerate(short_days):
        hook = hook_bank[i % len(hook_bank)]
        cta = cta_bank[i % len(cta_bank)]
        shorts.append(
            {
                "day": d,
                "format": "short",
                "hook": hook,
                "duration_target_seconds": seed_bp.get("duration_target_seconds", [14, 18]),
                "script": _short_script_template(hook, cta),
                "style_rules": seed_bp.get("hook_structure", {}),
            }
        )

    longs: List[Dict[str, Any]] = []
    for i, d in enumerate(long_days):
        cta = cta_bank[i % len(cta_bank)]
        title_fr = f"{req.business.vertical.upper()} : le plan simple pour {req.business.goal}"
        title_en = f"{req.business.vertical.upper()}: the simple plan for {req.business.goal}"
        title = title_fr if req.business.language.lower().startswith("fr") else title_en

        longs.append(
            {
                "day": d,
                "format": "long",
                "title": title,
                "duration_target_seconds": [360, 720],
                "outline": _long_outline_template(req.business.vertical, req.business.goal, cta, req.business.language),
            }
        )

    ideas = []
    for t in seed_bp.get("ideas", [])[:3]:
        ideas.append(t)
    ideas.append(
        {
            "idea": "Variation Hook x3 (même sujet)",
            "hook_variations": random.sample(hook_bank, k=min(3, len(hook_bank))),
            "note": "Post 3 variantes sur 48h, garde celle qui dépasse baseline ×2 en 2h.",
        }
    )

    return {
        "ok": True,
        "offer": {"type": "premium_monthly_all_access", "note": "Tout est inclus. Seule limite: quota/jour (fair-use)."},
        "seed": {"video_id": seed, "blueprint": seed_bp},
        "business": req.business.model_dump(),
        "cadence": req.cadence.model_dump(),
        "timeframe_days": timeframe,
        "calendar": {"shorts": shorts, "longs": longs},
        "ideas": ideas,
        "usage": {"today_count": get_daily_usage(token), "daily_limit": int(u.get("daily_quota", 999999))},
    }


# ===============================
# 🌐 STATIC SITE (LANDING MVP)
# ===============================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

STATIC_DIR = os.path.join(BASE_DIR, "backend", "static")

# sécurité : évite crash si dossier manquant
os.makedirs(STATIC_DIR, exist_ok=True)

# route landing page
@app.get("/", response_class=HTMLResponse)
def landing():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h1>YT Strategy Agent</h1><p>Landing page not deployed yet.</p>",
            status_code=200,
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# permet aussi /static/xxx si tu ajoutes CSS plus tard
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# (optionnel) run local — DOIT rester tout en bas
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
