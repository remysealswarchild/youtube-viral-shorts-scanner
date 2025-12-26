# scanner.py
# YouTube Shorts Virality Scanner (Data Collector)
#
# Updates (implemented):
# - Explicit quota/API failure detection (HttpError 403 quotaExceeded/dailyLimitExceeded/rateLimitExceeded)
# - Bubble up quota/API failures so app.py can fall back to DB snapshot
# - Robust error handling (keeps non-quota errors contained where appropriate)
# - Better Shorts detection (<=75s + heuristic scoring)
# - De-duplication across niches/queries (global_seen_video_ids)
# - Optional SQLite persistence (SQLAlchemy) with helpful indices
# - Channel enrichment expanded:
#     - channel creation date (snippet.publishedAt)
#     - subscribers (statistics.subscriberCount, may be hidden)
#     - total uploaded videos (statistics.videoCount)
#     - total channel views (statistics.viewCount)
# - DB schema updated to persist expanded channel metadata
# - Snapshot loader joins in channel metadata for DB-first UI (channel drawer + tables)
#
# NEW (per your latest request: niche-specific API deep scan):
# - Adds "niche deep scan" persistence that does NOT conflict with daily global snapshot:
#     - niche_scan_runs: run-level metadata (one row per deep scan execution)
#     - niche_video_stats_daily: per-video metrics keyed by (video_id, date, niche)
# - Adds APIs for app.py:
#     - run_niche_scan(...)
#     - load_latest_niche_scan(db_url, niche)
#     - load_niche_scan_runs(db_url, niche=None)
#     - load_channels_table(db_url)
#
# Notes:
# - Global daily snapshots continue to write to video_stats_daily with PK(video_id, date).
# - Niche deep scans write to niche_video_stats_daily with PK(video_id, date, niche),
#   so you can run multiple niche scans per day without clobbering global or other niches.

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Optional persistence (SQLite via SQLAlchemy)
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False
    OperationalError = Exception  # type: ignore


# -----------------------------
# Error helpers (quota detection)
# -----------------------------

def _http_error_text(e: HttpError) -> str:
    try:
        return str(e)
    except Exception:
        return "HttpError"

def _is_quota_error(e: HttpError) -> bool:
    """
    Detect quota/rate-limit issues that should trigger UI fallback to DB.
    Some quota errors appear as 403 with reason quotaExceeded / userRateLimitExceeded.
    """
    msg = _http_error_text(e)
    tokens = ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded")
    if any(t in msg for t in tokens):
        return True
    if getattr(e, "status_code", None) == 403:
        return True
    if "HttpError 403" in msg:
        return True
    return False

def _raise_if_quota(e: HttpError) -> None:
    if _is_quota_error(e):
        raise e


# -----------------------------
# Shorts heuristics
# -----------------------------

SHORTS_KEYWORDS = re.compile(r"(#shorts|ytshorts|\bshorts\b|\bshort\b)", re.IGNORECASE)

def parse_duration_seconds(duration_iso: str) -> int:
    """Parse ISO 8601 duration like 'PT59S' into seconds."""
    try:
        return int(isodate.parse_duration(duration_iso).total_seconds())
    except Exception:
        return 0

def shorts_confidence(
    duration_s: int,
    title: Optional[str],
    desc: Optional[str],
    query_hint: Optional[str],
) -> float:
    """
    Heuristic Shorts confidence score in [~ -0.5, 1.1].
    Recommended keep threshold ~0.60.
    """
    score = 0.0

    # Duration prior
    if duration_s <= 60:
        score += 0.70
    elif duration_s <= 75:
        score += 0.40
    elif duration_s <= 90:
        score += 0.10
    else:
        score -= 0.50

    # Text cues
    text_blob = f"{title or ''}\n{desc or ''}"
    if SHORTS_KEYWORDS.search(text_blob):
        score += 0.25

    # Query hint prior
    if query_hint and "short" in query_hint.lower():
        score += 0.10

    return score


# -----------------------------
# Persistence (SQLite)
# -----------------------------

def _get_engine(db_url: str):
    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError("SQLAlchemy is not available; install sqlalchemy to use DB persistence.")
    return create_engine(db_url, future=True)

def _init_db(engine):
    """
    Create tables (if not exist) and indices.

    If your DB was created before channel schema expansion, SQLite will not auto-migrate.
    Best practice: delete the old yt_shorts.db and let it recreate, OR add manual ALTER TABLE.
    """
    with engine.begin() as con:
        # Videos table
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            channel_id TEXT,
            channel_title TEXT,
            published_at TEXT,
            duration_s INTEGER
        );
        """))

        # Expanded channel metadata
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_title TEXT,
            created_at TEXT,
            subscribers INTEGER,
            video_count INTEGER,
            total_views INTEGER,
            updated_at TEXT
        );
        """))

        # Daily global snapshot stats
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS video_stats_daily (
            video_id TEXT,
            date TEXT,
            views INTEGER,
            views_per_day REAL,
            niche TEXT,
            shorts_score REAL,
            source_query TEXT,
            PRIMARY KEY (video_id, date)
        );
        """))

        # NEW: Niche deep scan run metadata
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS niche_scan_runs (
            run_id TEXT PRIMARY KEY,
            niche TEXT,
            started_at TEXT,
            finished_at TEXT,
            scan_days INTEGER,
            queries_per_niche INTEGER,
            max_results_per_query INTEGER,
            shorts_score_threshold REAL,
            max_videos_per_niche INTEGER,
            status TEXT,
            error TEXT
        );
        """))

        # NEW: Niche deep scan per-video stats (allows multiple niches per day)
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS niche_video_stats_daily (
            run_id TEXT,
            video_id TEXT,
            date TEXT,
            niche TEXT,
            views INTEGER,
            views_per_day REAL,
            shorts_score REAL,
            source_query TEXT,
            PRIMARY KEY (video_id, date, niche)
        );
        """))

        # Indices for faster UI queries
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_date ON video_stats_daily(date);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_niche ON video_stats_daily(niche);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_video ON video_stats_daily(video_id);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_channels_updated ON channels(updated_at);"))

        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_runs_niche ON niche_scan_runs(niche);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_runs_started ON niche_scan_runs(started_at);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_stats_date ON niche_video_stats_daily(date);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_stats_niche ON niche_video_stats_daily(niche);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_stats_run ON niche_video_stats_daily(run_id);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_niche_stats_video ON niche_video_stats_daily(video_id);"))

def _upsert_videos(engine, df: pd.DataFrame):
    if df.empty:
        return
    cols = ["video_id", "title", "description", "channel_id", "channel_title", "published_at", "duration_s"]
    df = df[cols].drop_duplicates("video_id").copy()
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0).astype(int)

    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
            INSERT INTO videos(video_id,title,description,channel_id,channel_title,published_at,duration_s)
            VALUES(:video_id,:title,:description,:channel_id,:channel_title,:published_at,:duration_s)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                channel_id=excluded.channel_id,
                channel_title=excluded.channel_title,
                published_at=excluded.published_at,
                duration_s=excluded.duration_s
            """), r.to_dict())

def _upsert_channels(engine, df: pd.DataFrame):
    if df.empty:
        return

    df = df.copy()
    expected = ["channel_id", "channel_title", "created_at", "subscribers", "video_count", "total_views", "updated_at"]
    for c in expected:
        if c not in df.columns:
            df[c] = None

    df["subscribers"] = pd.to_numeric(df["subscribers"], errors="coerce").astype("Int64")
    df["video_count"] = pd.to_numeric(df["video_count"], errors="coerce").astype("Int64")
    df["total_views"] = pd.to_numeric(df["total_views"], errors="coerce").astype("Int64")

    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
            INSERT INTO channels(channel_id,channel_title,created_at,subscribers,video_count,total_views,updated_at)
            VALUES(:channel_id,:channel_title,:created_at,:subscribers,:video_count,:total_views,:updated_at)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_title=excluded.channel_title,
                created_at=excluded.created_at,
                subscribers=excluded.subscribers,
                video_count=excluded.video_count,
                total_views=excluded.total_views,
                updated_at=excluded.updated_at
            """), r.to_dict())

def _insert_daily_stats(engine, df_stats: pd.DataFrame):
    if df_stats.empty:
        return
    cols = ["video_id", "date", "views", "views_per_day", "niche", "shorts_score", "source_query"]
    df_stats = df_stats[cols].copy()
    df_stats["views"] = pd.to_numeric(df_stats["views"], errors="coerce").fillna(0).astype(int)
    df_stats["views_per_day"] = pd.to_numeric(df_stats["views_per_day"], errors="coerce").fillna(0.0).astype(float)
    df_stats["shorts_score"] = pd.to_numeric(df_stats["shorts_score"], errors="coerce").fillna(0.0).astype(float)

    with engine.begin() as con:
        for _, r in df_stats.iterrows():
            con.execute(text("""
            INSERT OR IGNORE INTO video_stats_daily(video_id,date,views,views_per_day,niche,shorts_score,source_query)
            VALUES(:video_id,:date,:views,:views_per_day,:niche,:shorts_score,:source_query)
            """), r.to_dict())

def _insert_niche_run(engine, run_row: dict):
    with engine.begin() as con:
        con.execute(text("""
        INSERT INTO niche_scan_runs(
            run_id,niche,started_at,finished_at,scan_days,queries_per_niche,max_results_per_query,
            shorts_score_threshold,max_videos_per_niche,status,error
        ) VALUES (
            :run_id,:niche,:started_at,:finished_at,:scan_days,:queries_per_niche,:max_results_per_query,
            :shorts_score_threshold,:max_videos_per_niche,:status,:error
        )
        ON CONFLICT(run_id) DO UPDATE SET
            niche=excluded.niche,
            started_at=excluded.started_at,
            finished_at=excluded.finished_at,
            scan_days=excluded.scan_days,
            queries_per_niche=excluded.queries_per_niche,
            max_results_per_query=excluded.max_results_per_query,
            shorts_score_threshold=excluded.shorts_score_threshold,
            max_videos_per_niche=excluded.max_videos_per_niche,
            status=excluded.status,
            error=excluded.error
        """), run_row)

def _insert_niche_stats_daily(engine, df_stats: pd.DataFrame):
    """
    Expected columns:
      run_id, video_id, date, niche, views, views_per_day, shorts_score, source_query
    PK: (video_id, date, niche) so the same video can be stored for multiple niches per day.
    """
    if df_stats.empty:
        return

    cols = ["run_id", "video_id", "date", "niche", "views", "views_per_day", "shorts_score", "source_query"]
    df_stats = df_stats[cols].copy()
    df_stats["views"] = pd.to_numeric(df_stats["views"], errors="coerce").fillna(0).astype(int)
    df_stats["views_per_day"] = pd.to_numeric(df_stats["views_per_day"], errors="coerce").fillna(0.0).astype(float)
    df_stats["shorts_score"] = pd.to_numeric(df_stats["shorts_score"], errors="coerce").fillna(0.0).astype(float)

    with engine.begin() as con:
        for _, r in df_stats.iterrows():
            con.execute(text("""
            INSERT OR IGNORE INTO niche_video_stats_daily(
                run_id, video_id, date, niche, views, views_per_day, shorts_score, source_query
            ) VALUES (
                :run_id, :video_id, :date, :niche, :views, :views_per_day, :shorts_score, :source_query
            )
            """), r.to_dict())


def load_latest_snapshot(db_url: str) -> pd.DataFrame:
    """
    Load the latest daily global snapshot (max date) joined with video + channel metadata.
    Returns empty DataFrame if no DB or no data.

    Used by DB-first UI and the Channel Profile drawer.
    """
    if not db_url or not SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()

    engine = _get_engine(db_url)
    with engine.begin() as con:
        try:
            max_date = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        except OperationalError:
            return pd.DataFrame()

        if not max_date:
            return pd.DataFrame()

        df = pd.read_sql(text("""
            SELECT
                s.video_id,
                s.date,
                s.views,
                s.views_per_day,
                s.niche,
                s.shorts_score,
                s.source_query,

                v.title,
                v.description,
                v.channel_id,
                v.channel_title,
                v.published_at,
                v.duration_s,

                c.created_at AS channel_created_at,
                c.subscribers AS channel_subscribers,
                c.video_count AS channel_video_count,
                c.total_views AS channel_total_views
            FROM video_stats_daily s
            JOIN videos v ON v.video_id = s.video_id
            LEFT JOIN channels c ON c.channel_id = v.channel_id
            WHERE s.date = :d
        """), con, params={"d": max_date})

    return df

def load_channels_table(db_url: str) -> pd.DataFrame:
    """
    Loads channel metadata from DB for UI (channel drawer, channel tables).
    """
    if not db_url or not SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()

    engine = _get_engine(db_url)
    with engine.begin() as con:
        try:
            df = pd.read_sql(text("""
                SELECT channel_id, channel_title, created_at, subscribers, video_count, total_views, updated_at
                FROM channels
            """), con)
        except OperationalError:
            return pd.DataFrame()

    return df


# -----------------------------
# Niche deep scan DB loaders
# -----------------------------

def load_niche_scan_runs(db_url: str, niche: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a table of niche scan runs (latest first).
    """
    if not db_url or not SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()

    engine = _get_engine(db_url)
    with engine.begin() as con:
        try:
            if niche:
                df = pd.read_sql(text("""
                    SELECT run_id, niche, started_at, finished_at, status, error,
                           scan_days, queries_per_niche, max_results_per_query,
                           shorts_score_threshold, max_videos_per_niche
                    FROM niche_scan_runs
                    WHERE niche = :n
                    ORDER BY started_at DESC
                """), con, params={"n": niche})
            else:
                df = pd.read_sql(text("""
                    SELECT run_id, niche, started_at, finished_at, status, error,
                           scan_days, queries_per_niche, max_results_per_query,
                           shorts_score_threshold, max_videos_per_niche
                    FROM niche_scan_runs
                    ORDER BY started_at DESC
                """), con)
        except OperationalError:
            return pd.DataFrame()

    return df

def load_latest_niche_scan(db_url: str, niche: str) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Loads the latest niche deep scan for a given niche.
    Returns (run_id, df). If none exists: (None, empty_df)
    """
    if not db_url or not SQLALCHEMY_AVAILABLE:
        return None, pd.DataFrame()

    engine = _get_engine(db_url)
    with engine.begin() as con:
        try:
            run_id = con.execute(
                text("""
                    SELECT run_id
                    FROM niche_scan_runs
                    WHERE niche = :n AND status = 'success'
                    ORDER BY started_at DESC
                    LIMIT 1
                """),
                {"n": niche},
            ).scalar()
        except OperationalError:
            return None, pd.DataFrame()

        if not run_id:
            return None, pd.DataFrame()

        # Join niche stats -> videos -> channels for a rich UI table
        df = pd.read_sql(text("""
            SELECT
                ns.run_id,
                ns.video_id,
                ns.date,
                ns.niche,
                ns.views,
                ns.views_per_day,
                ns.shorts_score,
                ns.source_query,

                v.title,
                v.description,
                v.channel_id,
                v.channel_title,
                v.published_at,
                v.duration_s,

                c.created_at AS channel_created_at,
                c.subscribers AS channel_subscribers,
                c.video_count AS channel_video_count,
                c.total_views AS channel_total_views
            FROM niche_video_stats_daily ns
            JOIN videos v ON v.video_id = ns.video_id
            LEFT JOIN channels c ON c.channel_id = v.channel_id
            WHERE ns.run_id = :rid
        """), con, params={"rid": run_id})

    return str(run_id), df


# -----------------------------
# API collection
# -----------------------------

def _iso_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=int(days))
    return dt.isoformat().replace("+00:00", "Z")

def fetch_candidates(
    youtube,
    query: str,
    published_after_iso: str,
    max_results: int = 25,
) -> List[str]:
    """
    Uses search.list to fetch candidate video IDs.
    order=viewCount biases toward high views (virality scanning).
    """
    try:
        resp = youtube.search().list(
            part="id",
            q=query,
            type="video",
            order="viewCount",
            publishedAfter=published_after_iso,
            maxResults=max_results,
        ).execute()
    except HttpError as e:
        _raise_if_quota(e)
        return []
    except Exception:
        return []

    ids: List[str] = []
    for it in resp.get("items", []):
        vid = (it.get("id") or {}).get("videoId")
        if vid:
            ids.append(vid)
    return ids

def fetch_videos(youtube, video_ids: List[str]) -> pd.DataFrame:
    """
    Fetches snippet + contentDetails + statistics for a list of video IDs.
    Batched up to 50.
    """
    if not video_ids:
        return pd.DataFrame(columns=[
            "video_id", "title", "description", "channel_id", "channel_title",
            "published_at", "duration_s", "views"
        ])

    rows: List[dict] = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            _raise_if_quota(e)
            continue
        except Exception:
            continue

        for it in resp.get("items", []):
            sn = it.get("snippet", {}) or {}
            stt = it.get("statistics", {}) or {}
            cd = it.get("contentDetails", {}) or {}

            duration_iso = cd.get("duration", "PT0S")
            dur_s = parse_duration_seconds(duration_iso)

            rows.append({
                "video_id": it.get("id"),
                "title": sn.get("title"),
                "description": sn.get("description"),
                "channel_id": sn.get("channelId"),
                "channel_title": sn.get("channelTitle"),
                "published_at": sn.get("publishedAt"),
                "duration_s": int(dur_s),
                "views": int(stt.get("viewCount", 0) or 0),
            })

    df = pd.DataFrame(rows).dropna(subset=["video_id"])
    return df

def fetch_channels(youtube, channel_ids: List[str]) -> pd.DataFrame:
    """
    Fetch channel stats:
      - created_at (snippet.publishedAt)
      - subscribers (statistics.subscriberCount; may be hidden)
      - video_count (statistics.videoCount)
      - total_views (statistics.viewCount)
    Batched up to 50.
    """
    if not channel_ids:
        return pd.DataFrame(columns=[
            "channel_id", "channel_title", "created_at",
            "subscribers", "video_count", "total_views", "updated_at"
        ])

    rows: List[dict] = []
    uniq = list(dict.fromkeys([c for c in channel_ids if c]))

    for i in range(0, len(uniq), 50):
        batch = uniq[i:i+50]
        try:
            resp = youtube.channels().list(
                part="snippet,statistics",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            _raise_if_quota(e)
            continue
        except Exception:
            continue

        for it in resp.get("items", []):
            sn = it.get("snippet", {}) or {}
            stt = it.get("statistics", {}) or {}

            subs = stt.get("subscriberCount")  # may be hidden
            vidc = stt.get("videoCount")
            viewc = stt.get("viewCount")

            rows.append({
                "channel_id": it.get("id"),
                "channel_title": sn.get("title"),
                "created_at": sn.get("publishedAt"),
                "subscribers": int(subs) if subs is not None else None,
                "video_count": int(vidc) if vidc is not None else None,
                "total_views": int(viewc) if viewc is not None else None,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["updated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        df = pd.DataFrame(columns=[
            "channel_id", "channel_title", "created_at",
            "subscribers", "video_count", "total_views", "updated_at"
        ])
    return df


# -----------------------------
# Scan budget configuration
# -----------------------------

@dataclass
class ScanBudget:
    queries_per_niche: int = 5
    max_results_per_query: int = 25
    scan_days: int = 30
    shorts_score_threshold: float = 0.60
    max_videos_per_niche: int = 120


# -----------------------------
# Core scan logic helper (reused)
# -----------------------------

def _scan_one_niche(
    youtube,
    niche: str,
    queries: List[str],
    budget: ScanBudget,
    published_after: str,
) -> pd.DataFrame:
    """
    Scans a single niche and returns a videos DF (not persisted here).
    Quota errors bubble up (HttpError).
    """
    if not queries:
        return pd.DataFrame()

    qlist = queries[: budget.queries_per_niche]

    found_by: Dict[str, str] = {}
    niche_video_ids: List[str] = []

    for q in qlist:
        ids = fetch_candidates(
            youtube,
            query=q,
            published_after_iso=published_after,
            max_results=budget.max_results_per_query,
        )
        for vid in ids:
            if vid in found_by:
                continue
            found_by[vid] = q
            niche_video_ids.append(vid)

    if not niche_video_ids:
        return pd.DataFrame()

    dfv = fetch_videos(youtube, niche_video_ids)
    if dfv.empty:
        return pd.DataFrame()

    dfv["niche"] = niche
    dfv["source_query"] = dfv["video_id"].map(found_by)

    dfv["shorts_score"] = dfv.apply(
        lambda r: shorts_confidence(
            int(r.get("duration_s", 0) or 0),
            r.get("title"),
            r.get("description"),
            r.get("source_query"),
        ),
        axis=1,
    )

    dfv = dfv[dfv["shorts_score"] >= budget.shorts_score_threshold].copy()
    if dfv.empty:
        return pd.DataFrame()

    dfv = dfv.sort_values("views", ascending=False).head(budget.max_videos_per_niche).copy()

    now = datetime.now(timezone.utc)
    pub = pd.to_datetime(dfv["published_at"], utc=True, errors="coerce")
    age_days = (now - pub).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0.25)
    dfv["views_per_day"] = dfv["views"] / age_days

    return dfv


# -----------------------------
# Main entry point: Global scan
# -----------------------------

def run_scan(
    api_key: str,
    query_packs: Dict[str, List[str]],
    budget: Optional[ScanBudget] = None,
    db_url: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a scan over the provided query packs.

    Behavior:
    - If quota is exceeded or rate limited, raises HttpError so UI can fall back to DB.
    - If db_url is provided and SQLAlchemy is available, persists results to SQLite.

    Returns:
      videos_df, channels_df
    """
    if budget is None:
        budget = ScanBudget()

    youtube = build("youtube", "v3", developerKey=api_key)
    published_after = _iso_days_ago(budget.scan_days)

    engine = None
    if db_url:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("DB_URL provided but SQLAlchemy is not installed. pip install sqlalchemy")
        engine = _get_engine(db_url)
        _init_db(engine)

    all_videos: List[pd.DataFrame] = []
    global_seen_video_ids: set[str] = set()

    for niche, queries in query_packs.items():
        if not queries:
            continue

        # Scan one niche (may raise HttpError on quota)
        dfv = _scan_one_niche(youtube, niche, queries, budget, published_after)
        if dfv is None or dfv.empty:
            continue

        # global dedupe (first niche owns it for this run)
        dfv = dfv[~dfv["video_id"].isin(global_seen_video_ids)].copy()
        global_seen_video_ids.update(dfv["video_id"].tolist())

        if dfv.empty:
            continue

        all_videos.append(dfv)

    videos = (
        pd.concat(all_videos, ignore_index=True)
        if all_videos
        else pd.DataFrame(columns=[
            "video_id", "title", "description", "channel_id", "channel_title", "published_at",
            "duration_s", "views", "niche", "source_query", "shorts_score", "views_per_day"
        ])
    )

    # Channel enrichment (quota errors bubble up)
    channels = fetch_channels(youtube, videos["channel_id"].dropna().unique().tolist())

    # Join channel metadata into videos for UI convenience (works in live scans)
    if not videos.empty and not channels.empty:
        ch_keep = ["channel_id", "created_at", "subscribers", "video_count", "total_views"]
        videos = videos.merge(channels[ch_keep], on="channel_id", how="left")

    # Add convenient URLs for UI
    if not videos.empty:
        videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
        videos["channel_url"] = "https://www.youtube.com/channel/" + videos["channel_id"].astype(str)

    # Persist (global snapshot)
    if engine is not None and not videos.empty:
        _upsert_videos(engine, videos)
        if not channels.empty:
            _upsert_channels(engine, channels)

        stats = videos[["video_id", "views", "views_per_day", "niche", "shorts_score", "source_query"]].copy()
        stats["date"] = date.today().isoformat()
        _insert_daily_stats(engine, stats)

    return videos, channels


# -----------------------------
# NEW: Niche deep scan (API) + persistence
# -----------------------------

def run_niche_scan(
    api_key: str,
    niche: str,
    queries: List[str],
    budget: Optional[ScanBudget] = None,
    db_url: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Runs a niche-specific deep scan (API) and persists results into:
      - videos, channels (upserts)
      - niche_scan_runs (run metadata)
      - niche_video_stats_daily (per-video stats keyed by video_id/date/niche)

    Returns:
      videos_df, channels_df, run_id

    Quota errors (HttpError) bubble up to the caller.
    """
    if budget is None:
        budget = ScanBudget()

    if not db_url:
        raise ValueError("run_niche_scan requires db_url (DB-first design).")

    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError("SQLAlchemy is not installed. pip install sqlalchemy")

    engine = _get_engine(db_url)
    _init_db(engine)

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Pre-insert run row as 'running'
    _insert_niche_run(engine, {
        "run_id": run_id,
        "niche": niche,
        "started_at": started_at,
        "finished_at": None,
        "scan_days": int(budget.scan_days),
        "queries_per_niche": int(budget.queries_per_niche),
        "max_results_per_query": int(budget.max_results_per_query),
        "shorts_score_threshold": float(budget.shorts_score_threshold),
        "max_videos_per_niche": int(budget.max_videos_per_niche),
        "status": "running",
        "error": None,
    })

    youtube = build("youtube", "v3", developerKey=api_key)
    published_after = _iso_days_ago(budget.scan_days)

    try:
        videos = _scan_one_niche(youtube, niche, queries, budget, published_after)
        if videos is None or videos.empty:
            videos = pd.DataFrame(columns=[
                "video_id", "title", "description", "channel_id", "channel_title", "published_at",
                "duration_s", "views", "niche", "source_query", "shorts_score", "views_per_day"
            ])

        channels = fetch_channels(youtube, videos["channel_id"].dropna().unique().tolist())

        # Join channel metadata into videos for UI convenience
        if not videos.empty and not channels.empty:
            ch_keep = ["channel_id", "created_at", "subscribers", "video_count", "total_views"]
            videos = videos.merge(channels[ch_keep], on="channel_id", how="left")

        if not videos.empty:
            videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
            videos["channel_url"] = "https://www.youtube.com/channel/" + videos["channel_id"].astype(str)

        # Persist
        if not videos.empty:
            _upsert_videos(engine, videos)
        if not channels.empty:
            _upsert_channels(engine, channels)

        # Niche stats persistence (date = today UTC)
        if not videos.empty:
            stats = videos[["video_id", "views", "views_per_day", "niche", "shorts_score", "source_query"]].copy()
            stats["run_id"] = run_id
            stats["date"] = date.today().isoformat()
            # Ensure niche column is correct even if caller passed mismatched DF
            stats["niche"] = niche
            _insert_niche_stats_daily(engine, stats[[
                "run_id", "video_id", "date", "niche", "views", "views_per_day", "shorts_score", "source_query"
            ]])

        finished_at = datetime.now(timezone.utc).isoformat()
        _insert_niche_run(engine, {
            "run_id": run_id,
            "niche": niche,
            "started_at": started_at,
            "finished_at": finished_at,
            "scan_days": int(budget.scan_days),
            "queries_per_niche": int(budget.queries_per_niche),
            "max_results_per_query": int(budget.max_results_per_query),
            "shorts_score_threshold": float(budget.shorts_score_threshold),
            "max_videos_per_niche": int(budget.max_videos_per_niche),
            "status": "success",
            "error": None,
        })

        return videos, channels, run_id

    except HttpError as e:
        # Bubble quota errors, but record run status first
        finished_at = datetime.now(timezone.utc).isoformat()
        _insert_niche_run(engine, {
            "run_id": run_id,
            "niche": niche,
            "started_at": started_at,
            "finished_at": finished_at,
            "scan_days": int(budget.scan_days),
            "queries_per_niche": int(budget.queries_per_niche),
            "max_results_per_query": int(budget.max_results_per_query),
            "shorts_score_threshold": float(budget.shorts_score_threshold),
            "max_videos_per_niche": int(budget.max_videos_per_niche),
            "status": "failed",
            "error": _http_error_text(e),
        })
        _raise_if_quota(e)
        # Non-quota HttpError: raise anyway (caller decides)
        raise

    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        _insert_niche_run(engine, {
            "run_id": run_id,
            "niche": niche,
            "started_at": started_at,
            "finished_at": finished_at,
            "scan_days": int(budget.scan_days),
            "queries_per_niche": int(budget.queries_per_niche),
            "max_results_per_query": int(budget.max_results_per_query),
            "shorts_score_threshold": float(budget.shorts_score_threshold),
            "max_videos_per_niche": int(budget.max_videos_per_niche),
            "status": "failed",
            "error": str(e),
        })
        raise


# -----------------------------
# Convenience CLI (optional)
# -----------------------------

if __name__ == "__main__":
    from niches import QUERY_PACKS

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing YOUTUBE_API_KEY env var.")

    db_url = os.getenv("DB_URL", "").strip() or None

    budget = ScanBudget(
        queries_per_niche=int(os.getenv("QUERIES_PER_NICHE", "5")),
        max_results_per_query=int(os.getenv("MAX_RESULTS_PER_QUERY", "25")),
        scan_days=int(os.getenv("SCAN_DAYS", "30")),
        shorts_score_threshold=float(os.getenv("SHORTS_SCORE_THRESHOLD", "0.60")),
        max_videos_per_niche=int(os.getenv("MAX_VIDEOS_PER_NICHE", "120")),
    )

    # Global scan
    try:
        videos_df, channels_df = run_scan(
            api_key=api_key,
            query_packs=QUERY_PACKS,
            budget=budget,
            db_url=db_url,
        )
        print(f"[GLOBAL] Videos: {len(videos_df)} | Channels: {len(channels_df)}")
    except HttpError as e:
        print("[GLOBAL] API scan failed (likely quota/rate limit).")
        print(str(e))

    # Example niche scan (optional)
    if db_url and "sports" in QUERY_PACKS:
        try:
            v2, c2, rid = run_niche_scan(
                api_key=api_key,
                niche="sports",
                queries=QUERY_PACKS["sports"],
                budget=ScanBudget(queries_per_niche=2, max_results_per_query=15, scan_days=30, max_videos_per_niche=150),
                db_url=db_url,
            )
            print(f"[NICHE] run_id={rid} | Videos: {len(v2)} | Channels: {len(c2)}")
        except HttpError as e:
            print("[NICHE] API scan failed (likely quota/rate limit).")
            print(str(e))

    if db_url and SQLALCHEMY_AVAILABLE:
        snap = load_latest_snapshot(db_url)
        print(f"[DB] Latest global snapshot rows: {len(snap)}")
        runs = load_niche_scan_runs(db_url)
        print(f"[DB] Niche scan runs: {len(runs)}")
