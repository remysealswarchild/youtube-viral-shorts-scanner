# app.py
# Streamlit Dashboard for YouTube Shorts Virality Scanner
#
# Best-practice operational approach (implemented):
# - DB-first always: UI loads and browses from SQLite snapshot (no API calls on load)
# - Daily scan gate: scan can run at most once per UTC day (button disabled after that)
# - Quota-safe: if scan fails (quota/API), UI falls back to latest DB snapshot automatically
# - First-run friendly: if quota is exceeded and DB is empty, show clear "how to seed" guidance
# - Seed DB mode: optional ultra-safe first-run scan budget to minimize quota usage
#
# Requirements:
# - streamlit, pandas, numpy, plotly, python-dotenv
# - sqlalchemy (required for DB-first/daily-scan mode)
#
# Env (.env):
# - YOUTUBE_API_KEY=...
# - DB_URL=sqlite:///yt_shorts.db  (recommended)

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from niches import QUERY_PACKS
from scanner import run_scan, ScanBudget, load_latest_snapshot

# Prefer scoring.py if present; fallback implementations provided otherwise.
try:
    from scoring import niche_leaderboard as _niche_leaderboard
    from scoring import channel_table as _channel_table
except Exception:
    def _niche_leaderboard(videos: pd.DataFrame) -> pd.DataFrame:
        if videos.empty:
            return pd.DataFrame(columns=[
                "niche", "shorts_count", "median_views", "p90_views",
                "median_views_per_day", "top_video_views", "score"
            ])

        def p90(x): return float(np.percentile(x, 90))

        agg = (videos.groupby("niche")
               .agg(
                   shorts_count=("video_id", "count"),
                   median_views=("views", "median"),
                   p90_views=("views", p90),
                   median_views_per_day=("views_per_day", "median"),
                   top_video_views=("views", "max"),
               )
               .reset_index())

        agg["score"] = (
            0.35*np.log10(agg["median_views"] + 1) +
            0.25*np.log10(agg["p90_views"] + 1) +
            0.30*np.log10(agg["median_views_per_day"] + 1) +
            0.10*np.log10(agg["shorts_count"] + 1)
        )
        return agg.sort_values("score", ascending=False).reset_index(drop=True)

    def _channel_table(videos: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
        if videos.empty:
            return pd.DataFrame(columns=[
                "niche", "channel_id", "channel_title", "shorts_in_sample",
                "sample_views_sum", "sample_views_median", "subscribers"
            ])
        cagg = (videos.groupby(["niche", "channel_id", "channel_title"])
                .agg(
                    shorts_in_sample=("video_id", "count"),
                    sample_views_sum=("views", "sum"),
                    sample_views_median=("views", "median"),
                )
                .reset_index())
        if not channels.empty and "subscribers" in channels.columns:
            cagg = cagg.merge(channels[["channel_id", "subscribers"]], on="channel_id", how="left")
        else:
            cagg["subscribers"] = None
        return cagg.sort_values(["niche", "sample_views_sum"], ascending=[True, False]).reset_index(drop=True)


# -----------------------------
# SQLAlchemy availability + DB utilities
# -----------------------------

def _sqlalchemy_available() -> bool:
    try:
        import sqlalchemy  # noqa: F401
        return True
    except Exception:
        return False

def _db_engine(db_url: str):
    from sqlalchemy import create_engine
    return create_engine(db_url, future=True)

def _has_snapshot_for_date(db_url: str, day_yyyy_mm_dd: str) -> bool:
    if not (db_url and _sqlalchemy_available()):
        return False
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    engine = _db_engine(db_url)
    try:
        with engine.begin() as con:
            n = con.execute(
                text("SELECT COUNT(1) FROM video_stats_daily WHERE date = :d"),
                {"d": day_yyyy_mm_dd},
            ).scalar()
        return int(n or 0) > 0
    except OperationalError:
        # Table not created yet (no successful scan persisted)
        return False
    except Exception:
        return False

def _get_latest_date(db_url: str) -> Optional[str]:
    if not (db_url and _sqlalchemy_available()):
        return None
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    engine = _db_engine(db_url)
    try:
        with engine.begin() as con:
            d = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        return str(d) if d else None
    except OperationalError:
        return None
    except Exception:
        return None


# -----------------------------
# DB helper for "emerging niches"
# -----------------------------

def _load_emerging_niches(db_url: str, window_days: int = 7) -> pd.DataFrame:
    if not db_url or not _sqlalchemy_available():
        return pd.DataFrame()

    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
    engine = create_engine(db_url, future=True)

    with engine.begin() as con:
        try:
            max_date = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        except OperationalError:
            return pd.DataFrame()

        if not max_date:
            return pd.DataFrame()

        max_dt = pd.to_datetime(max_date).date()
        last_start = max_dt - timedelta(days=window_days - 1)
        prev_end = last_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=window_days - 1)

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
            return pd.DataFrame(columns=[
                "niche", "shorts_count", "median_views", "p90_views", "median_views_per_day", "score"
            ])

        def p90(x): return float(np.percentile(x, 90))

        agg = (dfx.groupby("niche")
               .agg(
                   shorts_count=("views", "count"),
                   median_views=("views", "median"),
                   p90_views=("views", p90),
                   median_views_per_day=("views_per_day", "median"),
               )
               .reset_index())
        agg["score"] = (
            0.35*np.log10(agg["median_views"] + 1) +
            0.25*np.log10(agg["p90_views"] + 1) +
            0.30*np.log10(agg["median_views_per_day"] + 1) +
            0.10*np.log10(agg["shorts_count"] + 1)
        )
        return agg

    last = _score_frame(df[last_mask].copy()).rename(columns={"score": "score_last"})
    prev = _score_frame(df[prev_mask].copy()).rename(columns={"score": "score_prev"})

    merged = last.merge(prev[["niche", "score_prev"]], on="niche", how="left")
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
st.caption("Best-practice mode: browse DB snapshots; scan at most once per day (UTC).")

def _ensure_urls(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos
    if "video_url" not in videos.columns and "video_id" in videos.columns:
        videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
    if "channel_url" not in videos.columns and "channel_id" in videos.columns:
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

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_load_latest(db_url: str):
    return load_latest_snapshot(db_url)

today_utc = datetime.utcnow().date().isoformat()

with st.sidebar:
    st.header("Daily Scan Configuration")

    mode = st.selectbox(
        "Scan depth",
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
    st.header("Display Filters")
    min_views = st.number_input("Min views", min_value=0, value=0, step=1000)
    min_views_per_day = st.number_input("Min views/day", min_value=0.0, value=0.0, step=100.0)
    only_top_niches = st.slider("Show leaderboard top N", 10, 50, 30, 5)

    st.divider()
    st.header("Storage & Actions")

    use_db = st.checkbox("Use SQLite persistence (DB_URL)", value=bool(DB_URL))

    db_ready = bool(use_db and DB_URL and _sqlalchemy_available())
    if use_db and not DB_URL:
        st.warning("DB_URL not set. Add DB_URL=sqlite:///yt_shorts.db to your .env to enable persistence.")
    if use_db and DB_URL and not _sqlalchemy_available():
        st.warning("SQLAlchemy not installed. Run: pip install sqlalchemy")

    # Seed mode for first run or quota-sensitive environments
    seed_mode = st.checkbox("Seed DB (Ultra-safe, first run)", value=True)
    st.caption("Seed mode reduces API calls dramatically (recommended until DB has snapshots).")

    already_scanned_today = _has_snapshot_for_date(DB_URL, today_utc) if db_ready else False
    latest_date = _get_latest_date(DB_URL) if db_ready else None

    run_btn = st.button(
        f"Run daily scan ({today_utc})",
        type="primary",
        disabled=(not db_ready) or already_scanned_today or (API_KEY == "")
    )

    if not db_ready:
        st.info("Daily scan mode requires DB_URL + SQLAlchemy (recommended).")
    elif API_KEY == "":
        st.warning("YOUTUBE_API_KEY missing. You can still browse existing DB snapshots.")
    elif already_scanned_today:
        st.success(f"Daily scan already completed for {today_utc}.")
    else:
        st.caption("Scan is allowed once per day. All browsing uses DB snapshots.")

    load_latest_btn = st.button("Reload latest DB snapshot", disabled=(not db_ready))

    st.divider()
    st.header("About")
    st.write("Niches:", len(QUERY_PACKS))
    st.write("DB persistence:", "Enabled" if db_ready else "Disabled")
    if latest_date:
        st.write("Latest snapshot date:", latest_date)

# Base budget
budget_kwargs = dict(
    queries_per_niche=queries_per_niche,
    max_results_per_query=max_results_per_query,
    scan_days=scan_days,
    shorts_score_threshold=shorts_threshold,
    max_videos_per_niche=max_videos_per_niche,
)

# Apply seed-mode budget override (ultra conservative)
effective_budget_kwargs = dict(budget_kwargs)
if seed_mode:
    effective_budget_kwargs["queries_per_niche"] = 1
    effective_budget_kwargs["max_results_per_query"] = min(int(effective_budget_kwargs["max_results_per_query"]), 15)
    effective_budget_kwargs["max_videos_per_niche"] = min(int(effective_budget_kwargs["max_videos_per_niche"]), 60)

# -----------------------------
# Core data flow (DB-first)
# -----------------------------

videos = pd.DataFrame()
channels = pd.DataFrame()
data_source_label = "None"

def _load_db_or_empty() -> pd.DataFrame:
    if not db_ready:
        return pd.DataFrame()
    try:
        snap = _cached_load_latest(DB_URL)
        return snap if isinstance(snap, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _render_empty_db_quota_message(e: Exception) -> None:
    st.error(
        "YouTube API quota is exceeded and the database is empty, so there is no snapshot to display.\n\n"
        "How to fix:\n"
        "1) Wait for quota to reset, then click 'Run daily scan' once to seed the database.\n"
        "2) Keep 'Seed DB (Ultra-safe, first run)' enabled to minimize quota usage.\n"
        "3) Or request higher quota in Google Cloud Console.\n\n"
        "After the first successful scan, the UI will always work from DB snapshots—even when quota is exhausted.\n\n"
        f"Details: {e}"
    )

if run_btn:
    st.cache_data.clear()
    if not db_ready:
        st.error("DB is not ready. Enable DB_URL + SQLAlchemy to run daily scans.")
    elif _has_snapshot_for_date(DB_URL, today_utc):
        st.info(f"Daily scan already ran for {today_utc}. Loading DB snapshot.")
        snap = _load_db_or_empty()
        videos = _ensure_urls(snap)
        channels = pd.DataFrame()
        data_source_label = "SQLite snapshot"
    else:
        with st.spinner("Running daily scan (YouTube API) and persisting to DB..."):
            try:
                v, c = _cached_run_scan(effective_budget_kwargs, DB_URL)
                videos, channels = _ensure_urls(v), c
                data_source_label = "Live scan"
                st.success(f"Daily scan completed and stored for {today_utc}.")
            except Exception as e:
                snap = _load_db_or_empty()
                if not snap.empty:
                    st.warning(
                        "Scan failed (quota/API issue). Showing latest saved DB snapshot instead.\n\n"
                        f"Details: {e}"
                    )
                    videos = _ensure_urls(snap)
                    channels = pd.DataFrame()
                    data_source_label = "SQLite snapshot"
                else:
                    _render_empty_db_quota_message(e)
                    videos = pd.DataFrame()
                    channels = pd.DataFrame()
                    data_source_label = "None"

elif load_latest_btn:
    with st.spinner("Reloading latest DB snapshot..."):
        snap = _load_db_or_empty()
        if snap.empty:
            st.warning("No snapshot found in DB yet. Run the daily scan once to populate the database.")
            videos, channels, data_source_label = pd.DataFrame(), pd.DataFrame(), "None"
        else:
            videos, channels, data_source_label = _ensure_urls(snap), pd.DataFrame(), "SQLite snapshot"

else:
    # Default: always use DB snapshot (no scanning on page load)
    snap = _load_db_or_empty()
    if not db_ready:
        st.warning("DB persistence is disabled. Enable DB_URL + SQLAlchemy to keep the UI usable without API quota.")
        videos, channels, data_source_label = pd.DataFrame(), pd.DataFrame(), "None"
    elif snap.empty:
        st.warning("Database is empty. Run the daily scan once to populate it.")
        videos, channels, data_source_label = pd.DataFrame(), pd.DataFrame(), "None"
    else:
        st.info("Showing latest saved snapshot from database. Scans run at most once per day.")
        videos, channels, data_source_label = _ensure_urls(snap), pd.DataFrame(), "SQLite snapshot"

# -----------------------------
# Header stats
# -----------------------------

colA, colB, colC, colD = st.columns(4)
colA.metric("Videos (Shorts)", int(len(videos)))
colB.metric("Niches represented", int(videos["niche"].nunique()) if not videos.empty and "niche" in videos.columns else 0)
colC.metric("Channels", int(videos["channel_id"].nunique()) if not videos.empty and "channel_id" in videos.columns else 0)
colD.metric("Data source", data_source_label)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Niche Leaderboard", "Videos Explorer", "Channels Explorer", "Emerging Niches"])

# -----------------------------
# Tab 1: Leaderboard
# -----------------------------
with tab1:
    st.subheader("Niche Leaderboard")
    if videos.empty:
        st.info("No data to display. Run the daily scan once to populate the DB.")
    else:
        lb = _niche_leaderboard(videos)
        if lb.empty:
            st.info("No niches computed from current dataset.")
        else:
            lb_view = lb.head(only_top_niches).copy()
            fig = px.bar(lb_view, x="niche", y="score")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(lb, use_container_width=True, height=360)

# -----------------------------
# Tab 2: Videos Explorer
# -----------------------------
with tab2:
    st.subheader("Videos Explorer")
    if videos.empty:
        st.info("No videos available.")
    else:
        niches_sorted = sorted(videos["niche"].dropna().unique().tolist())
        selected_niche = st.selectbox("Filter by niche", options=["(All)"] + niches_sorted)

        vf = videos.copy()
        if selected_niche != "(All)":
            vf = vf[vf["niche"] == selected_niche].copy()

        if "views" in vf.columns:
            vf = vf[vf["views"] >= int(min_views)]
        if "views_per_day" in vf.columns:
            vf = vf[vf["views_per_day"] >= float(min_views_per_day)]

        sort_by = st.selectbox("Sort videos by", options=["views", "views_per_day", "shorts_score", "published_at"], index=0)
        ascending = st.checkbox("Ascending sort", value=False)

        if sort_by in vf.columns:
            vf = vf.sort_values(sort_by, ascending=ascending)

        cols = [c for c in [
            "niche", "title", "channel_title", "views", "views_per_day",
            "shorts_score", "duration_s", "published_at", "video_url", "channel_url", "source_query"
        ] if c in vf.columns]

        st.dataframe(vf[cols], use_container_width=True, height=520)
        st.caption("Copy/paste video_url or channel_url into your browser.")

# -----------------------------
# Tab 3: Channels Explorer
# -----------------------------
with tab3:
    st.subheader("Channels Explorer")
    if videos.empty:
        st.info("No channels available.")
    else:
        channels_df = pd.DataFrame(columns=["channel_id", "channel_title", "subscribers"])
        ct = _channel_table(videos, channels_df)

        niches_sorted = sorted(ct["niche"].dropna().unique().tolist()) if not ct.empty else []
        selected_niche_c = st.selectbox("Niche (channels)", options=["(All)"] + niches_sorted, key="channels_niche_select")

        cf = ct.copy()
        if selected_niche_c != "(All)":
            cf = cf[cf["niche"] == selected_niche_c].copy()

        ch_sort = st.selectbox("Sort channels by", options=["sample_views_sum", "shorts_in_sample", "sample_views_median", "subscribers"], index=0)
        ch_asc = st.checkbox("Ascending (channels)", value=False)

        if ch_sort in cf.columns:
            cf = cf.sort_values(ch_sort, ascending=ch_asc)

        st.dataframe(cf, use_container_width=True, height=520)

# -----------------------------
# Tab 4: Emerging niches
# -----------------------------
with tab4:
    st.subheader("Emerging Niches (trend acceleration)")
    if not db_ready:
        st.info("Enable DB_URL + SQLAlchemy to compute emerging niches.")
    else:
        window_days = st.slider("Emergence window (days)", 3, 14, 7, 1)
        em = _load_emerging_niches(DB_URL, window_days=window_days)
        if em.empty:
            st.info("Not enough DB history yet. Run daily scans for a few days to compute deltas.")
        else:
            st.write(f"Comparing score over {em['window'].iloc[0]} windows:")
            st.write(f"Last: {em['last_range'].iloc[0]}")
            st.write(f"Prev: {em['prev_range'].iloc[0]}")

            fig = px.bar(em.head(20), x="niche", y="delta")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                em[["niche", "delta", "score_last", "score_prev", "shorts_count", "median_views", "p90_views", "median_views_per_day"]],
                use_container_width=True,
                height=420
            )

st.divider()
with st.expander("Operational notes (API & quotas)"):
    st.write(
        f"""
- This UI is DB-first: it loads saved snapshots and does not consume API quota while browsing.
- Daily scans are allowed once per UTC day ({today_utc}). After that, the scan button is disabled.
- If a scan fails due to quota/API limits, the UI automatically falls back to the latest DB snapshot.
- If the DB is empty and quota is exceeded, keep Seed DB mode enabled and run once after quota resets.
        """
    )
