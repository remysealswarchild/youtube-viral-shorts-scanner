# scoring.py
# Aggregation + ranking utilities for YouTube Shorts Virality Scanner
#
# Provides:
# - niche_leaderboard(videos): ranks niches by virality score (robust, outlier-resistant)
# - channel_table(videos, channels): aggregates channels within each niche
# - optional helpers: add_urls, top_videos_by_niche, emerging_niches_from_history
#
# Expected `videos` columns (from scanner.py / DB snapshot):
#   - video_id, niche, views, views_per_day
#   - channel_id, channel_title
# Optional:
#   - shorts_score, published_at, duration_s
#
# Expected `channels` columns (from scanner.fetch_channels):
#   - channel_id, subscribers (optional), channel_title (optional)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class ViralityWeights:
    w_median_views: float = 0.35
    w_p90_views: float = 0.25
    w_median_vpd: float = 0.30
    w_volume: float = 0.10


# -----------------------------
# Core scoring
# -----------------------------

def _p90(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return 0.0
    return float(np.percentile(x.to_numpy(), 90))

def _safe_log10(x: pd.Series | float) -> pd.Series | float:
    # log10(x + 1) safe for zeros
    return np.log10(np.asarray(x) + 1)

def compute_virality_score(
    shorts_count: pd.Series,
    median_views: pd.Series,
    p90_views: pd.Series,
    median_views_per_day: pd.Series,
    weights: ViralityWeights = ViralityWeights(),
) -> pd.Series:
    """
    Vectorized score used by niche_leaderboard.
    """
    return (
        weights.w_median_views * _safe_log10(median_views) +
        weights.w_p90_views * _safe_log10(p90_views) +
        weights.w_median_vpd * _safe_log10(median_views_per_day) +
        weights.w_volume * _safe_log10(shorts_count)
    )


def niche_leaderboard(
    videos: pd.DataFrame,
    weights: ViralityWeights = ViralityWeights(),
    min_shorts: int = 5,
) -> pd.DataFrame:
    """
    Aggregate per-niche virality metrics and compute a robust virality score.

    Returns columns:
      niche, shorts_count, median_views, p90_views, median_views_per_day,
      top_video_views, pct_over_100k, pct_over_1m, score
    """
    if videos is None or videos.empty:
        return pd.DataFrame(columns=[
            "niche","shorts_count","median_views","p90_views","median_views_per_day",
            "top_video_views","pct_over_100k","pct_over_1m","score"
        ])

    df = videos.copy()

    # Ensure numeric
    df["views"] = pd.to_numeric(df.get("views"), errors="coerce").fillna(0).astype(int)
    df["views_per_day"] = pd.to_numeric(df.get("views_per_day"), errors="coerce").fillna(0.0).astype(float)

    agg = (df.groupby("niche")
           .agg(
               shorts_count=("video_id", "count"),
               median_views=("views", "median"),
               p90_views=("views", _p90),
               median_views_per_day=("views_per_day", "median"),
               top_video_views=("views", "max"),
               over_100k=("views", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) >= 100_000).mean())),
               over_1m=("views", lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) >= 1_000_000).mean())),
           )
           .reset_index())

    agg = agg.rename(columns={"over_100k": "pct_over_100k", "over_1m": "pct_over_1m"})

    # Filter low-sample niches (optional, keeps leaderboard stable)
    if min_shorts and min_shorts > 1:
        agg = agg[agg["shorts_count"] >= int(min_shorts)].copy()

    # Score
    agg["score"] = compute_virality_score(
        shorts_count=agg["shorts_count"],
        median_views=agg["median_views"],
        p90_views=agg["p90_views"],
        median_views_per_day=agg["median_views_per_day"],
        weights=weights,
    )

    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    return agg


# -----------------------------
# Channel aggregation
# -----------------------------

def channel_table(
    videos: pd.DataFrame,
    channels: Optional[pd.DataFrame] = None,
    min_shorts_per_channel: int = 1,
) -> pd.DataFrame:
    """
    Aggregate channels within each niche using the sampled Shorts.

    Returns columns:
      niche, channel_id, channel_title, shorts_in_sample,
      sample_views_sum, sample_views_median, sample_views_per_day_median, subscribers
    """
    if videos is None or videos.empty:
        return pd.DataFrame(columns=[
            "niche","channel_id","channel_title","shorts_in_sample",
            "sample_views_sum","sample_views_median","sample_views_per_day_median","subscribers"
        ])

    df = videos.copy()
    df["views"] = pd.to_numeric(df.get("views"), errors="coerce").fillna(0).astype(int)
    df["views_per_day"] = pd.to_numeric(df.get("views_per_day"), errors="coerce").fillna(0.0).astype(float)

    cagg = (df.groupby(["niche", "channel_id", "channel_title"])
            .agg(
                shorts_in_sample=("video_id", "count"),
                sample_views_sum=("views", "sum"),
                sample_views_median=("views", "median"),
                sample_views_per_day_median=("views_per_day", "median"),
            )
            .reset_index())

    if min_shorts_per_channel and min_shorts_per_channel > 1:
        cagg = cagg[cagg["shorts_in_sample"] >= int(min_shorts_per_channel)].copy()

    # Merge subscriber counts if provided
    cagg["subscribers"] = None
    if channels is not None and not channels.empty and "channel_id" in channels.columns:
        merge_cols = ["channel_id"]
        if "subscribers" in channels.columns:
            merge_cols.append("subscribers")
        if "channel_title" in channels.columns:
            merge_cols.append("channel_title")

        ctmp = channels[merge_cols].drop_duplicates("channel_id").copy()
        # Only keep 'subscribers' from channels to avoid overriding sampled channel_title
        if "subscribers" in ctmp.columns:
            cagg = cagg.merge(ctmp[["channel_id", "subscribers"]], on="channel_id", how="left", suffixes=("", "_ch"))
            # prefer merged subscribers
            cagg["subscribers"] = cagg["subscribers_ch"]
            cagg = cagg.drop(columns=["subscribers_ch"], errors="ignore")

    cagg = cagg.sort_values(["niche", "sample_views_sum"], ascending=[True, False]).reset_index(drop=True)
    return cagg


# -----------------------------
# Convenience helpers
# -----------------------------

def add_urls(videos: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure video_url and channel_url columns exist.
    """
    if videos is None or videos.empty:
        return videos

    df = videos.copy()
    if "video_url" not in df.columns and "video_id" in df.columns:
        df["video_url"] = "https://www.youtube.com/watch?v=" + df["video_id"].astype(str)
    if "channel_url" not in df.columns and "channel_id" in df.columns:
        df["channel_url"] = "https://www.youtube.com/channel/" + df["channel_id"].astype(str)
    return df


def top_videos_by_niche(
    videos: pd.DataFrame,
    niche: str,
    top_n: int = 50,
    sort_by: str = "views",
) -> pd.DataFrame:
    """
    Return top videos for a selected niche.
    """
    if videos is None or videos.empty:
        return pd.DataFrame()

    df = videos[videos["niche"] == niche].copy()
    if df.empty:
        return df

    if sort_by not in df.columns:
        sort_by = "views"

    df = df.sort_values(sort_by, ascending=False).head(int(top_n)).reset_index(drop=True)
    return df


def emerging_niches_from_history(
    stats_history: pd.DataFrame,
    window_days: int = 7,
    weights: ViralityWeights = ViralityWeights(),
) -> pd.DataFrame:
    """
    Compute emerging niches from a historical daily stats dataframe.

    Required columns in stats_history:
      - date (YYYY-MM-DD or datetime)
      - niche
      - views
      - views_per_day

    Returns:
      niche, score_last, score_prev, delta, ranges
    """
    if stats_history is None or stats_history.empty:
        return pd.DataFrame()

    df = stats_history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["views_per_day"] = pd.to_numeric(df["views_per_day"], errors="coerce").fillna(0.0).astype(float)

    max_dt = df["date"].max()
    last_start = max_dt - pd.Timedelta(days=window_days - 1)
    prev_end = last_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=window_days - 1)

    last = df[(df["date"] >= last_start) & (df["date"] <= max_dt)].copy()
    prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)].copy()

    def _score(dfx: pd.DataFrame) -> pd.DataFrame:
        if dfx.empty:
            return pd.DataFrame(columns=["niche", "score"])
        agg = (dfx.groupby("niche")
               .agg(
                   shorts_count=("views", "count"),
                   median_views=("views", "median"),
                   p90_views=("views", _p90),
                   median_views_per_day=("views_per_day", "median"),
               )
               .reset_index())
        agg["score"] = compute_virality_score(
            shorts_count=agg["shorts_count"],
            median_views=agg["median_views"],
            p90_views=agg["p90_views"],
            median_views_per_day=agg["median_views_per_day"],
            weights=weights,
        )
        return agg[["niche","score"]]

    last_s = _score(last).rename(columns={"score": "score_last"})
    prev_s = _score(prev).rename(columns={"score": "score_prev"})

    merged = last_s.merge(prev_s, on="niche", how="left")
    merged["score_prev"] = merged["score_prev"].fillna(0.0)
    merged["delta"] = merged["score_last"] - merged["score_prev"]
    merged = merged.sort_values("delta", ascending=False).reset_index(drop=True)

    merged["last_range"] = f"{last_start.date().isoformat()} → {max_dt.isoformat()}"
    merged["prev_range"] = f"{prev_start.date().isoformat()} → {prev_end.date().isoformat()}"
    return merged
