# app.py
# Streamlit Dashboard for YouTube Shorts Virality Scanner
#
# Features:
# - Global scan (last N days) via YouTube Data API v3
# - Niche leaderboard (virality score)
# - Video explorer (filters, links)
# - Channel explorer (aggregated by niche)
# - Persistence-aware:
#   - If DB_URL is set and SQLAlchemy installed, the app can load latest snapshot from SQLite
#   - Manual "Run scan now" persists today's snapshot (one per day per video via PK constraint)
# - Emerging niches view (last 7 days vs previous 7 days) when DB has enough history
#
# Requirements:
# - streamlit, pandas, numpy, plotly, python-dotenv, google-api-python-client, isodate
# - sqlalchemy (optional, only if DB persistence used)
#
# Env (.env):
# - YOUTUBE_API_KEY=...
# - DB_URL=sqlite:///yt_shorts.db   (optional but recommended)
#
# Companion files:
# - niches.py with QUERY_PACKS
# - scanner.py with run_scan, ScanBudget, load_latest_snapshot
# - scoring.py with niche_leaderboard, channel_table (or you can keep the built-ins below)

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from niches import QUERY_PACKS
from scanner import run_scan, ScanBudget, load_latest_snapshot

# If you already have scoring.py, prefer that. Otherwise use these local functions.
try:
    from scoring import niche_leaderboard as _niche_leaderboard
    from scoring import channel_table as _channel_table
except Exception:
    def _niche_leaderboard(videos: pd.DataFrame) -> pd.DataFrame:
        if videos.empty:
            return pd.DataFrame(columns=[
                "niche","shorts_count","median_views","p90_views","median_views_per_day","top_video_views","score"
            ])

        def p90(x): return float(np.percentile(x, 90))

        agg = (videos.groupby("niche")
               .agg(
                   shorts_count=("video_id","count"),
                   median_views=("views","median"),
                   p90_views=("views", p90),
                   median_views_per_day=("views_per_day","median"),
                   top_video_views=("views","max"),
               )
               .reset_index())

        agg["score"] = (
            0.35*np.log10(agg["median_views"]+1) +
            0.25*np.log10(agg["p90_views"]+1) +
            0.30*np.log10(agg["median_views_per_day"]+1) +
            0.10*np.log10(agg["shorts_count"]+1)
        )
        return agg.sort_values("score", ascending=False).reset_index(drop=True)

    def _channel_table(videos: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
        if videos.empty:
            return pd.DataFrame(columns=[
                "niche","channel_id","channel_title","shorts_in_sample",
                "sample_views_sum","sample_views_median","subscribers"
            ])
        cagg = (videos.groupby(["niche","channel_id","channel_title"])
                .agg(
                    shorts_in_sample=("video_id","count"),
                    sample_views_sum=("views","sum"),
                    sample_views_median=("views","median"),
                )
                .reset_index())
        if not channels.empty and "subscribers" in channels.columns:
            cagg = cagg.merge(channels[["channel_id","subscribers"]], on="channel_id", how="left")
        else:
            cagg["subscribers"] = None
        return cagg.sort_values(["niche","sample_views_sum"], ascending=[True, False]).reset_index(drop=True)


# -----------------------------
# DB helper for "emerging niches"
# -----------------------------

def _sqlalchemy_available() -> bool:
    try:
        import sqlalchemy  # noqa: F401
        return True
    except Exception:
        return False

def _load_emerging_niches(db_url: str, window_days: int = 7) -> pd.DataFrame:
    """
    Emerging niches: compare last N days vs previous N days using a niche-level score.
    Requires video_stats_daily table in SQLite (created by scanner.py persistence).
    """
    if not db_url or not _sqlalchemy_available():
        return pd.DataFrame()

    from sqlalchemy import create_engine, text

    engine = create_engine(db_url, future=True)

    with engine.begin() as con:
        max_date = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        if not max_date:
            return pd.DataFrame()

        # Determine date ranges (UTC dates stored as YYYY-MM-DD strings)
        max_dt = pd.to_datetime(max_date).date()
        last_start = max_dt - timedelta(days=window_days - 1)
        prev_end = last_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=window_days - 1)

        # Pull relevant rows
        df = pd.read_sql(text("""
            SELECT date, niche, views, views_per_day
            FROM video_stats_daily
            WHERE date BETWEEN :prev_start AND :max_date
        """), con, params={
            "prev_start": prev_start.isoformat(),
            "max_date": max_dt.isoformat()
        })

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    last_mask = (df["date"] >= last_start) & (df["date"] <= max_dt)
    prev_mask = (df["date"] >= prev_start) & (df["date"] <= prev_end)

    def _score_frame(dfx: pd.DataFrame) -> pd.DataFrame:
        if dfx.empty:
            return pd.DataFrame(columns=["niche","shorts_count","median_views","p90_views","median_views_per_day","score"])

        def p90(x): return float(np.percentile(x, 90))

        agg = (dfx.groupby("niche")
               .agg(
                   shorts_count=("views","count"),
                   median_views=("views","median"),
                   p90_views=("views", p90),
                   median_views_per_day=("views_per_day","median"),
               )
               .reset_index())
        agg["score"] = (
            0.35*np.log10(agg["median_views"]+1) +
            0.25*np.log10(agg["p90_views"]+1) +
            0.30*np.log10(agg["median_views_per_day"]+1) +
            0.10*np.log10(agg["shorts_count"]+1)
        )
        return agg

    last = _score_frame(df[last_mask].copy()).rename(columns={"score": "score_last"})
    prev = _score_frame(df[prev_mask].copy()).rename(columns={"score": "score_prev"})

    merged = last.merge(prev[["niche","score_prev"]], on="niche", how="left")
    merged["score_prev"] = merged["score_prev"].fillna(0.0)
    merged["delta"] = merged["score_last"] - merged["score_prev"]
    merged = merged.sort_values("delta", ascending=False).reset_index(drop=True)

    merged["window"] = f"{window_days}d"
    merged["last_range"] = f"{last_start.isoformat()} → {max_dt.isoformat()}"
    merged["prev_range"] = f"{prev_start.isoformat()} → {prev_end.isoformat()}"
    return merged


# -----------------------------
# App
# -----------------------------

load_dotenv()

st.set_page_config(page_title="YouTube Shorts Virality Scanner", layout="wide")

API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
DB_URL = os.getenv("DB_URL", "").strip() or None

st.title("YouTube Shorts Virality Scanner")
st.caption("Global discovery of viral niches from YouTube Shorts (last 30 days by default).")

with st.sidebar:
    st.header("Run Configuration")

    mode = st.selectbox(
        "Quota mode",
        options=["Default (daily)", "Ultra-safe (frequent)", "Deep-scan (weekly)"],
        index=0
    )

    if mode == "Ultra-safe (frequent)":
        default_qpn, default_mrp, default_cap = 3, 15, 80
    elif mode == "Deep-scan (weekly)":
        default_qpn, default_mrp, default_cap = 6, 35, 180
    else:
        default_qpn, default_mrp, default_cap = 5, 25, 120

    scan_days = st.slider("Time window (days)", min_value=7, max_value=90, value=30, step=1)
    queries_per_niche = st.slider("Queries per niche", 1, 6, default_qpn, 1)
    max_results_per_query = st.slider("Max results per query", 5, 50, default_mrp, 5)

    shorts_threshold = st.slider("Shorts score threshold", 0.40, 0.90, 0.60, 0.05)
    max_videos_per_niche = st.slider("Max videos per niche (post-filter cap)", 40, 250, default_cap, 10)

    st.divider()

    st.header("Filters (Display)")
    min_views = st.number_input("Min views", min_value=0, value=0, step=1000)
    min_views_per_day = st.number_input("Min views/day", min_value=0.0, value=0.0, step=100.0)
    only_top_niches = st.slider("Show leaderboard top N", 10, 50, 30, 5)

    st.divider()

    st.header("Actions")

    use_db = st.checkbox("Use SQLite persistence (DB_URL)", value=bool(DB_URL))
    if use_db and not DB_URL:
        st.warning("DB_URL not set. Add DB_URL=sqlite:///yt_shorts.db to your .env to enable persistence.")
    if use_db and DB_URL and not _sqlalchemy_available():
        st.warning("SQLAlchemy not installed. Run: pip install sqlalchemy")

    run_btn = st.button("Run scan now", type="primary", disabled=(API_KEY == ""))

    load_latest_btn = st.button("Load latest snapshot from DB", disabled=(not (use_db and DB_URL and _sqlalchemy_available())))

    st.divider()
    st.header("About")
    st.write("Niches:", len(QUERY_PACKS))
    st.write("DB persistence:", "Enabled" if (use_db and DB_URL) else "Disabled")

if API_KEY == "":
    st.error("Missing YOUTUBE_API_KEY. Set it in your .env file.")
    st.stop()


def _ensure_urls(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos
    if "video_url" not in videos.columns:
        videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
    if "channel_url" not in videos.columns:
        videos["channel_url"] = "https://www.youtube.com/channel/" + videos["channel_id"].astype(str)
    return videos


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _cached_run_scan(budget_kwargs: dict, db_url: Optional[str]):
    budget = ScanBudget(**budget_kwargs)
    videos, channels = run_scan(
        api_key=API_KEY,
        query_packs=QUERY_PACKS,
        budget=budget,
        db_url=db_url,
    )
    return videos, channels


@st.cache_data(ttl=60 * 5, show_spinner=False)
def _cached_load_latest(db_url: str):
    return load_latest_snapshot(db_url)


# Action handling
videos = pd.DataFrame()
channels = pd.DataFrame()

if run_btn:
    st.cache_data.clear()
    with st.spinner("Running scan (YouTube API)..."):
        budget_kwargs = dict(
            queries_per_niche=queries_per_niche,
            max_results_per_query=max_results_per_query,
            scan_days=scan_days,
            shorts_score_threshold=shorts_threshold,
            max_videos_per_niche=max_videos_per_niche,
        )
        db_to_use = DB_URL if (use_db and DB_URL and _sqlalchemy_available()) else None
        videos, channels = _cached_run_scan(budget_kwargs, db_to_use)
        videos = _ensure_urls(videos)

elif load_latest_btn and use_db and DB_URL and _sqlalchemy_available():
    with st.spinner("Loading latest snapshot from DB..."):
        videos = _cached_load_latest(DB_URL)
        videos = _ensure_urls(videos)
        channels = pd.DataFrame()  # channel aggregation can be derived; optional

else:
    # Default: if DB is enabled and has data, load snapshot; otherwise run a cached scan
    if use_db and DB_URL and _sqlalchemy_available():
        snap = _cached_load_latest(DB_URL)
        if not snap.empty:
            videos = _ensure_urls(snap)
            channels = pd.DataFrame()
        else:
            budget_kwargs = dict(
                queries_per_niche=queries_per_niche,
                max_results_per_query=max_results_per_query,
                scan_days=scan_days,
                shorts_score_threshold=shorts_threshold,
                max_videos_per_niche=max_videos_per_niche,
            )
            db_to_use = DB_URL if (use_db and DB_URL and _sqlalchemy_available()) else None
            videos, channels = _cached_run_scan(budget_kwargs, db_to_use)
            videos = _ensure_urls(videos)
    else:
        budget_kwargs = dict(
            queries_per_niche=queries_per_niche,
            max_results_per_query=max_results_per_query,
            scan_days=scan_days,
            shorts_score_threshold=shorts_threshold,
            max_videos_per_niche=max_videos_per_niche,
        )
        videos, channels = _cached_run_scan(budget_kwargs, None)
        videos = _ensure_urls(videos)


# Header stats
colA, colB, colC, colD = st.columns(4)
colA.metric("Videos (Shorts)", int(len(videos)))
colB.metric("Niches represented", int(videos["niche"].nunique()) if not videos.empty and "niche" in videos.columns else 0)
colC.metric("Channels", int(videos["channel_id"].nunique()) if not videos.empty and "channel_id" in videos.columns else 0)
colD.metric("Data source", "SQLite snapshot" if (use_db and DB_URL and _sqlalchemy_available() and "date" in videos.columns) else "Live scan")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Niche Leaderboard", "Videos Explorer", "Channels Explorer", "Emerging Niches"])

# -----------------------------
# Tab 1: Leaderboard
# -----------------------------
with tab1:
    st.subheader("Niche Leaderboard")

    if videos.empty:
        st.info("No data to display. Try increasing max results per query or lowering Shorts threshold.")
    else:
        lb = _niche_leaderboard(videos)
        if lb.empty:
            st.info("No niches computed from current dataset.")
        else:
            lb_view = lb.head(only_top_niches).copy()

            fig = px.bar(lb_view, x="niche", y="score")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(lb, use_container_width=True, height=360)

            st.caption("Tip: Use Videos Explorer to click into a niche and inspect top Shorts and channels.")


# -----------------------------
# Tab 2: Videos Explorer
# -----------------------------
with tab2:
    st.subheader("Videos Explorer")

    if videos.empty:
        st.info("No videos available.")
    else:
        # Niche selection
        niches_sorted = sorted(videos["niche"].dropna().unique().tolist())
        selected_niche = st.selectbox("Filter by niche", options=["(All)"] + niches_sorted)

        vf = videos.copy()
        if selected_niche != "(All)":
            vf = vf[vf["niche"] == selected_niche].copy()

        # Apply display filters
        if "views" in vf.columns:
            vf = vf[vf["views"] >= int(min_views)]
        if "views_per_day" in vf.columns:
            vf = vf[vf["views_per_day"] >= float(min_views_per_day)]

        # Sort preference
        sort_by = st.selectbox("Sort videos by", options=["views", "views_per_day", "shorts_score", "published_at"], index=0)
        ascending = st.checkbox("Ascending sort", value=False)

        if sort_by in vf.columns:
            vf = vf.sort_values(sort_by, ascending=ascending)

        # Output columns
        cols = [c for c in [
            "niche", "title", "channel_title", "views", "views_per_day",
            "shorts_score", "duration_s", "published_at", "video_url", "channel_url", "source_query"
        ] if c in vf.columns]

        st.dataframe(vf[cols], use_container_width=True, height=520)

        st.caption("Links: copy/paste video_url or channel_url into your browser. Streamlit tables allow easy copying.")


# -----------------------------
# Tab 3: Channels Explorer
# -----------------------------
with tab3:
    st.subheader("Channels Explorer")

    if videos.empty:
        st.info("No channels available.")
    else:
        # If channels enrichment isn't available (DB snapshot path), build aggregation from videos alone.
        if channels is None or channels.empty:
            channels_df = pd.DataFrame(columns=["channel_id","channel_title","subscribers"])
        else:
            channels_df = channels.copy()

        ct = _channel_table(videos, channels_df)

        niches_sorted = sorted(ct["niche"].dropna().unique().tolist()) if not ct.empty else []
        selected_niche_c = st.selectbox("Niche (channels)", options=["(All)"] + niches_sorted, key="channels_niche_select")

        cf = ct.copy()
        if selected_niche_c != "(All)":
            cf = cf[cf["niche"] == selected_niche_c].copy()

        # Basic sort
        ch_sort = st.selectbox("Sort channels by", options=["sample_views_sum", "shorts_in_sample", "sample_views_median", "subscribers"], index=0)
        ch_asc = st.checkbox("Ascending (channels)", value=False)

        if ch_sort in cf.columns:
            cf = cf.sort_values(ch_sort, ascending=ch_asc)

        st.dataframe(cf, use_container_width=True, height=520)

        st.caption("Channel table is based on the sampled Shorts discovered by the scanner (not the entire channel catalog).")


# -----------------------------
# Tab 4: Emerging niches
# -----------------------------
with tab4:
    st.subheader("Emerging Niches (trend acceleration)")

    if not (use_db and DB_URL and _sqlalchemy_available()):
        st.info("Enable SQLite persistence (DB_URL + SQLAlchemy) to compute emerging niches.")
    else:
        window_days = st.slider("Emergence window (days)", 3, 14, 7, 1)
        em = _load_emerging_niches(DB_URL, window_days=window_days)

        if em.empty:
            st.info("Not enough DB history yet. Run scans for at least a few days to compute deltas.")
        else:
            st.write(f"Comparing score over {em['window'].iloc[0]} windows:")
            st.write(f"Last: {em['last_range'].iloc[0]}")
            st.write(f"Prev: {em['prev_range'].iloc[0]}")

            fig = px.bar(em.head(20), x="niche", y="delta")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                em[["niche","delta","score_last","score_prev","shorts_count","median_views","p90_views","median_views_per_day"]],
                use_container_width=True,
                height=420
            )

            st.caption("Delta is score_last - score_prev. High positive delta indicates a niche accelerating in virality.")


# Footer
st.divider()
with st.expander("Operational notes (API & quotas)"):
    st.write(
        """
- `search.list` is the main quota driver; increasing queries per niche or max results per query increases quota usage.
- If you run many times per day, use Ultra-safe mode or enable DB and rely on snapshots for browsing.
- The tool is a discovery scanner; it estimates virality from sampled Shorts, not an exhaustive market census.
        """
    )
