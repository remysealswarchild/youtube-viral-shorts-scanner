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
# Updates in this version (implemented):
# - Channel metadata surfaced (created_at, subscribers, video_count, total_views)
# - Channel Profile persists in st.sidebar (not right column) while scrolling
# - Adds "Niche Deep Scan" (API) for a selected niche:
#     - Runs niche-specific YouTube API scan for viral Shorts (quota permitting)
#     - Persists results into DB via scanner.run_niche_scan()
#     - DB-first browsing of the latest niche deep scan via scanner.load_latest_niche_scan()
#     - Run history table via scanner.load_niche_scan_runs()
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
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from niches import QUERY_PACKS
from scanner import (
    run_scan,
    run_niche_scan,
    ScanBudget,
    load_latest_snapshot,
    load_latest_niche_scan,
    load_niche_scan_runs,
    load_channels_table,
)

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
        """
        Sample-based channel table:
        - Aggregates discovered Shorts per niche/channel
        - Enriches with channel metadata when available
        """
        if videos.empty:
            return pd.DataFrame(columns=[
                "niche", "channel_id", "channel_title",
                "shorts_in_sample", "sample_views_sum", "sample_views_median",
                "created_at", "subscribers", "video_count", "total_views"
            ])

        cagg = (videos.groupby(["niche", "channel_id", "channel_title"])
                .agg(
                    shorts_in_sample=("video_id", "count"),
                    sample_views_sum=("views", "sum"),
                    sample_views_median=("views", "median"),
                )
                .reset_index())

        if channels is not None and not channels.empty and "channel_id" in channels.columns:
            keep = [c for c in ["channel_id", "created_at", "subscribers", "video_count", "total_views"]
                    if c in channels.columns]
            if keep:
                cagg = cagg.merge(channels[keep], on="channel_id", how="left")
        else:
            cagg["created_at"] = None
            cagg["subscribers"] = None
            cagg["video_count"] = None
            cagg["total_views"] = None

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

def _has_snapshot_for_date(db_url: str, day_yyyy_mm_dd: str) -> bool:
    if not (db_url and _sqlalchemy_available()):
        return False
    from sqlalchemy import text, create_engine
    from sqlalchemy.exc import OperationalError
    engine = create_engine(db_url, future=True)
    try:
        with engine.begin() as con:
            n = con.execute(
                text("SELECT COUNT(1) FROM video_stats_daily WHERE date = :d"),
                {"d": day_yyyy_mm_dd},
            ).scalar()
        return int(n or 0) > 0
    except OperationalError:
        return False
    except Exception:
        return False

def _get_latest_date(db_url: str) -> Optional[str]:
    if not (db_url and _sqlalchemy_available()):
        return None
    from sqlalchemy import text, create_engine
    from sqlalchemy.exc import OperationalError
    engine = create_engine(db_url, future=True)
    try:
        with engine.begin() as con:
            d = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        return str(d) if d else None
    except OperationalError:
        return None
    except Exception:
        return None


# -----------------------------
# DB helper for "emerging niches"  (FIX: included)
# -----------------------------

def _load_emerging_niches(db_url: str, window_days: int = 7) -> pd.DataFrame:
    """
    Emerging niches: compare last N days vs previous N days using a niche-level score.
    Requires video_stats_daily table in SQLite.
    """
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

        df = pd.read_sql(
            text("""
                SELECT date, niche, views, views_per_day
                FROM video_stats_daily
                WHERE date BETWEEN :prev_start AND :max_date
            """),
            con,
            params={
                "prev_start": prev_start.isoformat(),
                "max_date": max_dt.isoformat(),
            },
        )

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
# App setup
# -----------------------------

load_dotenv()
st.set_page_config(page_title="YouTube Shorts Virality Scanner", layout="wide")

API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
DB_URL = os.getenv("DB_URL", "").strip() or None

st.title("YouTube Shorts Virality Scanner")
st.caption("DB-first mode: browse DB snapshots; run scans intentionally to manage quota.")

today_utc = datetime.utcnow().date().isoformat()

def _ensure_urls(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos
    out = videos.copy()
    if "video_url" not in out.columns and "video_id" in out.columns:
        out["video_url"] = "https://www.youtube.com/watch?v=" + out["video_id"].astype(str)
    if "channel_url" not in out.columns and "channel_id" in out.columns:
        out["channel_url"] = "https://www.youtube.com/channel/" + out["channel_id"].astype(str)
    return out

def _normalize_snapshot_channel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    scanner.load_latest_snapshot may return channel_* columns from DB join.
    Normalize them to expected names used by UI and drawer.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    rename_map = {}
    if "channel_created_at" in out.columns and "created_at" not in out.columns:
        rename_map["channel_created_at"] = "created_at"
    if "channel_subscribers" in out.columns and "subscribers" not in out.columns:
        rename_map["channel_subscribers"] = "subscribers"
    if "channel_video_count" in out.columns and "video_count" not in out.columns:
        rename_map["channel_video_count"] = "video_count"
    if "channel_total_views" in out.columns and "total_views" not in out.columns:
        rename_map["channel_total_views"] = "total_views"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out

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

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_load_channels_db(db_url: str) -> pd.DataFrame:
    return load_channels_table(db_url)

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_load_latest_niche_scan(db_url: str, niche: str) -> Tuple[Optional[str], pd.DataFrame]:
    return load_latest_niche_scan(db_url, niche)

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_load_niche_runs(db_url: str, niche: Optional[str] = None) -> pd.DataFrame:
    return load_niche_scan_runs(db_url, niche=niche)


# -----------------------------
# Drawer renderer (sidebar)
# -----------------------------

def _render_channel_profile_sidebar(
    selected_channel_id: str,
    videos_df: pd.DataFrame,
    channels_df: pd.DataFrame,
    title: str = "Channel Profile",
) -> None:
    if not selected_channel_id:
        st.info("Select a channel to view its profile.")
        return

    vch = videos_df[videos_df["channel_id"] == selected_channel_id].copy() if (not videos_df.empty and "channel_id" in videos_df.columns) else pd.DataFrame()
    chrow = pd.DataFrame()
    if channels_df is not None and not channels_df.empty and "channel_id" in channels_df.columns:
        chrow = channels_df[channels_df["channel_id"] == selected_channel_id].head(1)

    # Identity fields
    if not chrow.empty and "channel_title" in chrow.columns:
        channel_title = chrow["channel_title"].iloc[0]
    elif not vch.empty and "channel_title" in vch.columns:
        channel_title = vch["channel_title"].iloc[0]
    else:
        channel_title = selected_channel_id

    channel_url = f"https://www.youtube.com/channel/{selected_channel_id}"

    st.subheader(title)
    st.markdown(f"**{channel_title}**")
    st.markdown(f"Channel URL: {channel_url}")

    created_at = chrow["created_at"].iloc[0] if (not chrow.empty and "created_at" in chrow.columns) else None
    subscribers = chrow["subscribers"].iloc[0] if (not chrow.empty and "subscribers" in chrow.columns) else None
    video_count = chrow["video_count"].iloc[0] if (not chrow.empty and "video_count" in chrow.columns) else None
    total_views = chrow["total_views"].iloc[0] if (not chrow.empty and "total_views" in chrow.columns) else None

    m1, m2 = st.columns(2)
    m1.metric("Subscribers", "Hidden" if pd.isna(subscribers) else f"{int(subscribers):,}")
    m2.metric("Total channel views", "—" if pd.isna(total_views) else f"{int(total_views):,}")

    m3, m4 = st.columns(2)
    m3.metric("Total uploaded videos", "—" if pd.isna(video_count) else f"{int(video_count):,}")
    m4.metric("Channel created", "—" if not created_at else str(created_at)[:10])

    st.divider()

    if vch.empty:
        st.info("No sampled Shorts for this channel in the current dataset/snapshot.")
        return

    sampled_count = len(vch)
    sampled_views_sum = int(vch["views"].sum()) if "views" in vch.columns else 0
    sampled_views_med = float(vch["views"].median()) if "views" in vch.columns else 0.0
    sampled_vpd_med = float(vch["views_per_day"].median()) if "views_per_day" in vch.columns else 0.0

    s1, s2 = st.columns(2)
    s1.metric("Sampled Shorts (scanner)", f"{sampled_count:,}")
    s2.metric("Sampled views (sum)", f"{sampled_views_sum:,}")

    s3, s4 = st.columns(2)
    s3.metric("Sampled views (median)", f"{int(sampled_views_med):,}")
    s4.metric("Sampled views/day (median)", f"{int(sampled_vpd_med):,}")

    if "niche" in vch.columns and not vch["niche"].isna().all():
        niche_counts = (vch["niche"].value_counts().head(10).reset_index())
        niche_counts.columns = ["niche", "shorts_in_sample"]
        fig = px.bar(niche_counts, x="niche", y="shorts_in_sample")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top sampled Shorts (by views/day)**")
    sort_key = "views_per_day" if "views_per_day" in vch.columns else ("views" if "views" in vch.columns else None)
    if sort_key:
        vch = vch.sort_values(sort_key, ascending=False)

    cols = [c for c in ["niche", "title", "views", "views_per_day", "shorts_score", "published_at", "video_url"] if c in vch.columns]
    st.dataframe(vch[cols].head(25), use_container_width=True, height=420)


# -----------------------------
# Sidebar: controls + persistent profile
# -----------------------------

with st.sidebar:
    st.header("Storage & Mode")

    use_db = st.checkbox("Use SQLite persistence (DB_URL)", value=bool(DB_URL))
    db_ready = bool(use_db and DB_URL and _sqlalchemy_available())

    if use_db and not DB_URL:
        st.warning("DB_URL not set. Add DB_URL=sqlite:///yt_shorts.db to your .env to enable persistence.")
    if use_db and DB_URL and not _sqlalchemy_available():
        st.warning("SQLAlchemy not installed. Run: pip install sqlalchemy")

    st.divider()
    st.header("Daily Scan (global)")

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
        st.info("Daily scan requires DB_URL + SQLAlchemy.")
    elif API_KEY == "":
        st.warning("YOUTUBE_API_KEY missing. You can still browse existing DB snapshots.")
    elif already_scanned_today:
        st.success(f"Daily scan already completed for {today_utc}.")
    else:
        st.caption("Scan is allowed once per UTC day.")

    load_latest_btn = st.button("Reload latest DB snapshot", disabled=(not db_ready))

    st.divider()
    st.header("Niche Deep Scan (API)")

    niche_list = sorted(list(QUERY_PACKS.keys()))
    selected_deep_niche = st.selectbox("Pick a niche", options=niche_list, index=0)

    deep_scan_days = st.slider("Deep scan window (days)", 7, 180, 45, 1)
    deep_queries_per_niche = st.slider("Deep scan: queries per niche", 1, 6, 3, 1)
    deep_max_results_per_query = st.slider("Deep scan: max results/query", 5, 50, 25, 5)
    deep_shorts_threshold = st.slider("Deep scan: shorts threshold", 0.40, 0.90, 0.60, 0.05)
    deep_max_videos_per_niche = st.slider("Deep scan: max videos (post-filter cap)", 40, 400, 200, 10)

    run_deep_btn = st.button(
        "Run niche deep scan (API)",
        type="primary",
        disabled=(not db_ready) or (API_KEY == "")
    )

    st.caption("Deep scans are niche-specific and can run multiple times (until quota is exhausted).")

    st.divider()
    st.header("Display Filters")
    min_views = st.number_input("Min views", min_value=0, value=0, step=1000)
    min_views_per_day = st.number_input("Min views/day", min_value=0.0, value=0.0, step=100.0)
    only_top_niches = st.slider("Show leaderboard top N", 10, 50, 30, 5)

    st.divider()
    st.header("Channel Profile (persistent)")

    if "selected_channel_id" not in st.session_state:
        st.session_state["selected_channel_id"] = ""

    st.caption("Select a channel in Channels Explorer or Niche Deep Dive, then it persists here while you scroll.")


# -----------------------------
# Budgets
# -----------------------------

budget_kwargs = dict(
    queries_per_niche=queries_per_niche,
    max_results_per_query=max_results_per_query,
    scan_days=scan_days,
    shorts_score_threshold=shorts_threshold,
    max_videos_per_niche=max_videos_per_niche,
)

effective_budget_kwargs = dict(budget_kwargs)
if seed_mode:
    effective_budget_kwargs["queries_per_niche"] = 1
    effective_budget_kwargs["max_results_per_query"] = min(int(effective_budget_kwargs["max_results_per_query"]), 15)
    effective_budget_kwargs["max_videos_per_niche"] = min(int(effective_budget_kwargs["max_videos_per_niche"]), 60)

deep_budget = ScanBudget(
    queries_per_niche=int(deep_queries_per_niche),
    max_results_per_query=int(deep_max_results_per_query),
    scan_days=int(deep_scan_days),
    shorts_score_threshold=float(deep_shorts_threshold),
    max_videos_per_niche=int(deep_max_videos_per_niche),
)


# -----------------------------
# Core data flow (DB-first global snapshot)
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

def _load_channels_db_or_empty() -> pd.DataFrame:
    if not db_ready:
        return pd.DataFrame()
    try:
        df = _cached_load_channels_db(DB_URL)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
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

# Run daily global scan
if run_btn:
    st.cache_data.clear()
    if not db_ready:
        st.error("DB is not ready. Enable DB_URL + SQLAlchemy to run daily scans.")
    elif _has_snapshot_for_date(DB_URL, today_utc):
        st.info(f"Daily scan already ran for {today_utc}. Loading DB snapshot.")
        snap = _normalize_snapshot_channel_cols(_load_db_or_empty())
        videos = _ensure_urls(snap)
        channels = _load_channels_db_or_empty()
        data_source_label = "SQLite snapshot"
    else:
        with st.spinner("Running daily scan (YouTube API) and persisting to DB..."):
            try:
                v, c = _cached_run_scan(effective_budget_kwargs, DB_URL)
                videos, channels = _ensure_urls(v), (c if isinstance(c, pd.DataFrame) else pd.DataFrame())
                data_source_label = "Live scan"
                st.success(f"Daily scan completed and stored for {today_utc}.")
            except Exception as e:
                snap = _load_db_or_empty()
                if not snap.empty:
                    st.warning(
                        "Scan failed (quota/API issue). Showing latest saved DB snapshot instead.\n\n"
                        f"Details: {e}"
                    )
                    snap = _normalize_snapshot_channel_cols(snap)
                    videos = _ensure_urls(snap)
                    channels = _load_channels_db_or_empty()
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
            snap = _normalize_snapshot_channel_cols(snap)
            videos = _ensure_urls(snap)
            channels = _load_channels_db_or_empty()
            data_source_label = "SQLite snapshot"

else:
    snap = _load_db_or_empty()
    if not db_ready:
        st.warning("DB persistence is disabled. Enable DB_URL + SQLAlchemy to keep the UI usable without API quota.")
        videos, channels, data_source_label = pd.DataFrame(), pd.DataFrame(), "None"
    elif snap.empty:
        st.warning("Database is empty. Run the daily scan once to populate it.")
        videos, channels, data_source_label = pd.DataFrame(), pd.DataFrame(), "None"
    else:
        st.info("Showing latest saved snapshot from database. Scans run at most once per day.")
        snap = _normalize_snapshot_channel_cols(snap)
        videos = _ensure_urls(snap)
        channels = _load_channels_db_or_empty()
        data_source_label = "SQLite snapshot"


# -----------------------------
# Niche deep scan (API) trigger + DB-first loader
# -----------------------------

deep_run_id: Optional[str] = None
deep_videos = pd.DataFrame()

if run_deep_btn:
    st.cache_data.clear()
    if not db_ready:
        st.error("DB is not ready. Enable DB_URL + SQLAlchemy to run niche deep scans.")
    elif API_KEY == "":
        st.error("Missing YOUTUBE_API_KEY. Cannot run niche deep scan.")
    else:
        with st.spinner(f"Running niche deep scan for '{selected_deep_niche}' and persisting to DB..."):
            try:
                v, c, rid = run_niche_scan(
                    api_key=API_KEY,
                    niche=selected_deep_niche,
                    queries=QUERY_PACKS.get(selected_deep_niche, []),
                    budget=deep_budget,
                    db_url=DB_URL,
                )
                deep_run_id = rid
                deep_videos = _ensure_urls(_normalize_snapshot_channel_cols(v))
                channels = _load_channels_db_or_empty()
                st.success(f"Niche deep scan completed for '{selected_deep_niche}'. run_id={rid}")
            except Exception as e:
                st.error(f"Niche deep scan failed (likely quota/API issue).\n\nDetails: {e}")

# Always attempt to load latest deep scan for the selected niche (DB-first)
if db_ready and selected_deep_niche:
    rid2, df2 = _cached_load_latest_niche_scan(DB_URL, selected_deep_niche)
    if rid2 and isinstance(df2, pd.DataFrame) and not df2.empty:
        deep_run_id = rid2
        deep_videos = _ensure_urls(_normalize_snapshot_channel_cols(df2))


# -----------------------------
# Sidebar: render persistent channel profile based on ACTIVE dataset
# -----------------------------

active_videos_for_profile = deep_videos if (deep_videos is not None and not deep_videos.empty) else videos
active_channels_for_profile = channels if (channels is not None and not channels.empty) else _load_channels_db_or_empty()

with st.sidebar:
    st.divider()
    _render_channel_profile_sidebar(
        selected_channel_id=st.session_state.get("selected_channel_id", ""),
        videos_df=active_videos_for_profile if isinstance(active_videos_for_profile, pd.DataFrame) else pd.DataFrame(),
        channels_df=active_channels_for_profile if isinstance(active_channels_for_profile, pd.DataFrame) else pd.DataFrame(),
        title="Channel Profile",
    )


# -----------------------------
# Header stats
# -----------------------------

colA, colB, colC, colD = st.columns(4)
colA.metric("Videos (Shorts)", int(len(videos)))
colB.metric("Niches represented", int(videos["niche"].nunique()) if not videos.empty and "niche" in videos.columns else 0)
colC.metric("Channels", int(videos["channel_id"].nunique()) if not videos.empty and "channel_id" in videos.columns else 0)
colD.metric("Data source", data_source_label)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Niche Leaderboard", "Videos Explorer", "Channels Explorer", "Emerging Niches", "Niche Deep Dive"]
)

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

        sort_by = st.selectbox(
            "Sort videos by",
            options=[c for c in ["views", "views_per_day", "shorts_score", "published_at"] if c in vf.columns],
            index=0
        )
        ascending = st.checkbox("Ascending sort", value=False)

        if sort_by in vf.columns:
            vf = vf.sort_values(sort_by, ascending=ascending)

        cols = [c for c in [
            "niche", "title", "channel_title", "views", "views_per_day",
            "shorts_score", "duration_s", "published_at", "video_url", "channel_url", "source_query",
            "created_at", "subscribers", "video_count", "total_views"
        ] if c in vf.columns]

        st.dataframe(vf[cols], use_container_width=True, height=520)
        st.caption("Copy/paste video_url or channel_url into your browser.")

# -----------------------------
# Tab 3: Channels Explorer (selection drives sidebar profile)
# -----------------------------
with tab3:
    st.subheader("Channels Explorer")

    if videos.empty:
        st.info("No channels available.")
    else:
        channels_df = channels.copy() if (channels is not None and not channels.empty) else _load_channels_db_or_empty()

        ct = _channel_table(videos, channels_df)

        niches_sorted = sorted(ct["niche"].dropna().unique().tolist()) if not ct.empty else []
        selected_niche_c = st.selectbox("Niche (channels)", options=["(All)"] + niches_sorted, key="channels_niche_select")

        cf = ct.copy()
        if selected_niche_c != "(All)":
            cf = cf[cf["niche"] == selected_niche_c].copy()

        ch_sort = st.selectbox(
            "Sort channels by",
            options=[c for c in ["sample_views_sum", "shorts_in_sample", "sample_views_median", "subscribers", "total_views", "video_count"] if c in cf.columns],
            index=0
        )
        ch_asc = st.checkbox("Ascending (channels)", value=False)

        if ch_sort in cf.columns:
            cf = cf.sort_values(ch_sort, ascending=ch_asc)

        st.markdown("**Channel list (sample-based)**")
        preferred_cols = [
            "niche", "channel_title",
            "subscribers", "total_views", "video_count", "created_at",
            "shorts_in_sample", "sample_views_sum", "sample_views_median",
            "channel_id"
        ]
        show_cols = [c for c in preferred_cols if c in cf.columns]
        st.dataframe(cf[show_cols], use_container_width=True, height=520)

        st.divider()
        st.markdown("**Select a channel to pin to the sidebar profile**")

        channel_options = []
        if not cf.empty and "channel_id" in cf.columns and "channel_title" in cf.columns:
            channel_options = [
                f"{r['channel_title']} | {r['channel_id']}"
                for _, r in cf[["channel_title", "channel_id"]].drop_duplicates().head(300).iterrows()
            ]

        selected = st.selectbox(
            "Channel (pins profile in sidebar)",
            options=["(Select)"] + channel_options,
            key="pin_channel_select_global"
        )

        if selected != "(Select)":
            st.session_state["selected_channel_id"] = selected.split("|")[-1].strip()
            st.success("Pinned. Scroll anywhere—the profile stays in the sidebar.")

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

# -----------------------------
# Tab 5: Niche Deep Dive (DB-first + optional API)
# -----------------------------
with tab5:
    st.subheader("Niche Deep Dive")

    if not db_ready:
        st.info("Enable DB_URL + SQLAlchemy to use niche deep dive (DB-first).")
    else:
        st.write(f"Selected niche: **{selected_deep_niche}**")
        if deep_run_id:
            st.caption(f"Latest deep scan run_id: {deep_run_id}")
        else:
            st.caption("No deep scan found yet for this niche. Use the sidebar to run one (API).")

        runs = _cached_load_niche_runs(DB_URL, niche=selected_deep_niche)
        if runs is not None and not runs.empty:
            st.markdown("**Deep scan run history**")
            st.dataframe(runs, use_container_width=True, height=220)

        if deep_videos is None or deep_videos.empty:
            st.info("No niche deep scan results available yet.")
        else:
            df = deep_videos.copy()
            if "views" in df.columns:
                df = df[df["views"] >= int(min_views)]
            if "views_per_day" in df.columns:
                df = df[df["views_per_day"] >= float(min_views_per_day)]

            sort_by = st.selectbox(
                "Sort deep scan videos by",
                options=[c for c in ["views_per_day", "views", "shorts_score", "published_at"] if c in df.columns],
                index=0,
                key="deep_sort"
            )
            asc = st.checkbox("Ascending (deep)", value=False, key="deep_asc")
            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=asc)

            st.markdown("**Viral Shorts (niche deep scan)**")
            cols = [c for c in [
                "title", "channel_title", "views", "views_per_day", "shorts_score",
                "published_at", "video_url", "channel_url",
                "source_query",
                "created_at", "subscribers", "video_count", "total_views",
            ] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, height=520)

            st.divider()
            st.markdown("**Pin a channel from deep scan to sidebar profile**")
            if "channel_id" in df.columns and "channel_title" in df.columns:
                channel_options = [
                    f"{r['channel_title']} | {r['channel_id']}"
                    for _, r in df[["channel_title", "channel_id"]].drop_duplicates().head(300).iterrows()
                ]
                sel = st.selectbox("Channel (deep scan)", options=["(Select)"] + channel_options, key="pin_channel_select_deep")
                if sel != "(Select)":
                    st.session_state["selected_channel_id"] = sel.split("|")[-1].strip()
                    st.success("Pinned. The profile stays visible in the sidebar.")

st.divider()
with st.expander("Operational notes (API & quotas)"):
    st.write(
        f"""
- DB-first browsing: opening the app does not call the YouTube API.
- Global daily scan is limited to once per UTC day ({today_utc}).
- Niche deep scans are on-demand (until quota is exhausted) and persist separately in DB.
- Channel Profile persists in the sidebar and updates when you “pin” a channel.
        """
    )
