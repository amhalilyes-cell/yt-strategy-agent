# backend/simulate.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Literal, Optional, Tuple
import math
import random

Mode = Literal["safe", "base", "aggressive"]

@dataclass(frozen=True)
class PlaybookSignals:
    replicability_score: float   # 0-100
    format_shift: bool
    confidence: str              # "low"|"medium"|"high"
    outlier_ratio: float
    baseline_vph: float
    video_vph: float
    baseline_n: int

@dataclass(frozen=True)
class SimulationModeConfig:
    rtm_strength: float
    sigma_base: float
    format_shift_multiplier: float
    tail_risk_prob: float
    tail_risk_impact: float

MODE_CONFIG: Dict[Mode, SimulationModeConfig] = {
    "safe": SimulationModeConfig(0.85, 0.35, 0.75, 0.08, 0.55),
    "base": SimulationModeConfig(0.65, 0.50, 0.85, 0.06, 0.60),
    "aggressive": SimulationModeConfig(0.45, 0.70, 0.92, 0.04, 0.70),
}

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _conf_multiplier(conf: str) -> float:
    c = (conf or "").lower().strip()
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.90
    return 0.78

def _rep_factor(rep_score: float) -> float:
    rep = _clamp((rep_score or 0.0) / 100.0, 0.0, 1.0)
    return rep ** 0.8

def _rtm_alpha(rep_factor: float, conf_mult: float, rtm_strength: float) -> float:
    base_alpha = _clamp(0.20 + 0.75 * rep_factor * conf_mult, 0.05, 0.95)
    k = 0.60
    alpha = base_alpha * (1.0 - k * rtm_strength)
    return _clamp(alpha, 0.03, 0.90)

def _sigma(rep_factor: float, conf_mult: float, outlier_ratio: float, sigma_base: float) -> float:
    r = max(1.0, float(outlier_ratio or 1.0))
    outlier_boost = _clamp(math.log10(r + 1e-9) / 2.0, 0.0, 1.0)
    stability = _clamp(0.55 + 0.45 * rep_factor * conf_mult, 0.30, 1.00)
    sig = sigma_base * (1.0 + 0.70 * outlier_boost) * (1.0 / stability)
    return _clamp(sig, 0.15, 1.50)

def _quantiles(xs: List[float], ps: Tuple[float, float, float]) -> Dict[str, float]:
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

def _risk_score(
    views_totals: List[float],
    baseline_total: float,
    confidence: str,
    rep_score: float,
    format_shift: bool
):
    if not views_totals:
        return 100, {"downside_prob": 1.0, "dispersion": 1.0, "confidence": 1.0, "format_shift": 1.0, "replicability": 1.0}, 1.0

    p_under = sum(1 for x in views_totals if x < baseline_total) / len(views_totals)
    q = _quantiles(views_totals, (0.10, 0.50, 0.90))
    p10, p50, p90 = q["min"], q["median"], q["max"]
    dispersion = (p90 - p10) / max(p50, 1.0)
    dispersion = _clamp(dispersion, 0.0, 3.0) / 3.0

    c = (confidence or "").lower().strip()
    conf_pen = 1.0 if c == "low" else 0.6 if c == "medium" else 0.35
    rep = _clamp((rep_score or 0.0) / 100.0, 0.0, 1.0)
    rep_pen = (1.0 - rep)
    shift_pen = 0.75 if format_shift else 0.25

    raw = (
        0.40 * p_under +
        0.25 * dispersion +
        0.15 * conf_pen +
        0.12 * shift_pen +
        0.08 * rep_pen
    )
    score = int(round(_clamp(raw, 0.0, 1.0) * 100))
    comps = {
        "downside_prob": p_under,
        "dispersion": dispersion,
        "confidence": conf_pen,
        "format_shift": shift_pen,
        "replicability": rep_pen,
    }
    return score, comps, p_under

def simulate_playbook(
    signals: PlaybookSignals,
    n_videos: int,
    timeframe_days: int,
    mode: Mode = "base",
    runs: int = 2000,
    seed: Optional[int] = None,
    return_runs: bool = False
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    cfg = MODE_CONFIG.get(mode, MODE_CONFIG["base"])
    n_videos = int(max(1, n_videos))
    timeframe_days = int(max(1, timeframe_days))
    runs = int(_clamp(float(runs), 200, 10000))

    baseline_vph = float(max(0.0, signals.baseline_vph or 0.0))
    outlier_ratio = float(max(0.0, signals.outlier_ratio or 1.0))
    rep_score = float(_clamp(signals.replicability_score or 0.0, 0.0, 100.0))

    conf_mult = _conf_multiplier(signals.confidence)
    rep_factor = _rep_factor(rep_score)

    alpha = _rtm_alpha(rep_factor, conf_mult, cfg.rtm_strength)
    ratio_rtm = 1.0 + (outlier_ratio - 1.0) * alpha
    ratio_rtm = max(0.05, ratio_rtm)

    if signals.format_shift:
        ratio_rtm *= cfg.format_shift_multiplier

    sig = _sigma(rep_factor, conf_mult, outlier_ratio, cfg.sigma_base)
    baseline_total = baseline_vph * 24.0 * timeframe_days * n_videos

    views_totals: List[float] = []
    views_per_video: List[float] = []

    for _ in range(runs):
        noise = random.lognormvariate(0.0, sig)
        if random.random() < cfg.tail_risk_prob:
            noise *= cfg.tail_risk_impact

        sim_vph = max(0.0, baseline_vph * ratio_rtm * noise)
        v_video = sim_vph * 24.0 * timeframe_days
        v_total = v_video * n_videos
        views_per_video.append(v_video)
        views_totals.append(v_total)

    q_total = _quantiles(views_totals, (0.10, 0.50, 0.90))
    q_video = _quantiles(views_per_video, (0.10, 0.50, 0.90))

    risk, risk_components, p_under = _risk_score(
        views_totals=views_totals,
        baseline_total=baseline_total,
        confidence=signals.confidence,
        rep_score=rep_score,
        format_shift=signals.format_shift
    )

    recommended_quota = int(_clamp(round(2 + risk / 18), 2, 12))

    warning_signals: List[str] = []
    if (signals.confidence or "").lower().strip() == "low":
        warning_signals.append("low_confidence_baseline")
    if signals.format_shift:
        warning_signals.append("format_shift_detected")
    if rep_score < 40:
        warning_signals.append("low_replicability")
    if outlier_ratio > 10:
        warning_signals.append("extreme_outlier_ratio")
    if int(signals.baseline_n or 0) < 12:
        warning_signals.append("weak_baseline_sample")

    result: Dict[str, Any] = {
        "expected_views_total": q_total,
        "expected_views_per_video": q_video,
        "risk_score": int(_clamp(risk, 0, 100)),
        "recommended_quota": recommended_quota,
        "confidence_level": (signals.confidence or "low").lower().strip(),
        "warning_signals": warning_signals,
        "assumptions": {
            "mode": mode,
            "runs": runs,
            "timeframe_days": timeframe_days,
            "n_videos": n_videos,
            "regression_alpha": alpha,
            "ratio_rtm_deterministic": ratio_rtm,
        },
        "debug": {
            "baseline_n": int(signals.baseline_n or 0),
            "ratio_rtm_median": ratio_rtm,
            "sigma": sig,
            "p_under_baseline": p_under,
            "risk_components": risk_components,
            "config_used": asdict(cfg),
        }
    }

    if return_runs:
        result["_runs"] = {"views_totals": views_totals}

    return result