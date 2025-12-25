# scanner.py
# YouTube Shorts Virality Scanner (Data Collector)
#
# Updates (per your request):
# - Explicit quota/API failure detection (HttpError 403 quotaExceeded/dailyLimitExceeded/rateLimitExceeded)
# - Bubble up quota/API failures so app.py can fall back to DB snapshot
# - More robust error handling (keeps non-quota errors contained where appropriate)
# - Adds helpful DB indices for faster trend queries

from __future__ import annotations

import os
import re
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
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


# -----------------------------
# Error helpers (quota detection)
# -----------------------------

def _http_error_text(e: HttpError) -> str:
    try:
        # googleapiclient HttpError often includes JSON details in str(e)
        return str(e)
    except Exception:
        return "HttpError"

def _is_quota_error(e: HttpError) -> bool:
    """
    Detect quota/rate-limit issues that should trigger UI fallback to DB.
    """
    msg = _http_error_text(e)
    tokens = ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded")
    return any(t in msg for t in tokens) or getattr(e, "status_code", None) == 403

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

def shorts_confidence(duration_s: int, title: Optional[str], desc: Optional[str], query_hint: Optional[str]) -> float:
    """
    Heuristic Shorts confidence score in [~ -0.5, 1.1].
    Recommended keep threshold ~0.60.
    """
    score = 0.0

    if duration_s <= 60:
        score += 0.70
    elif duration_s <= 75:
        score += 0.40
    elif duration_s <= 90:
        score += 0.10
    else:
        score -= 0.50

    text = f"{title or ''}\n{desc or ''}"
    if SHORTS_KEYWORDS.search(text):
        score += 0.25

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
    with engine.begin() as con:
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
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_title TEXT,
            subscribers INTEGER,
            updated_at TEXT
        );
        """))
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

        # indices for faster UI queries
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_date ON video_stats_daily(date);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_niche ON video_stats_daily(niche);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);"))

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
    if "subscribers" in df.columns:
        df["subscribers"] = pd.to_numeric(df["subscribers"], errors="coerce").astype("Int64")

    with engine.begin() as con:
        for _, r in df.iterrows():
            con.execute(text("""
            INSERT INTO channels(channel_id,channel_title,subscribers,updated_at)
            VALUES(:channel_id,:channel_title,:subscribers,:updated_at)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_title=excluded.channel_title,
                subscribers=excluded.subscribers,
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

def load_latest_snapshot(db_url: str) -> pd.DataFrame:
    """
    Load the latest daily snapshot (max date) joined with video metadata.
    Returns empty DataFrame if no DB or no data.
    """
    if not db_url or not SQLALCHEMY_AVAILABLE:
        return pd.DataFrame()
    engine = _get_engine(db_url)
    with engine.begin() as con:
        max_date = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
        if not max_date:
            return pd.DataFrame()
        df = pd.read_sql(text("""
            SELECT s.video_id, s.date, s.views, s.views_per_day, s.niche, s.shorts_score, s.source_query,
                   v.title, v.description, v.channel_id, v.channel_title, v.published_at, v.duration_s
            FROM video_stats_daily s
            JOIN videos v ON v.video_id = s.video_id
            WHERE s.date = :d
        """), con, params={"d": max_date})
    return df


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
    rows = []
    if not video_ids:
        return pd.DataFrame(columns=[
            "video_id", "title", "description", "channel_id", "channel_title",
            "published_at", "duration_s", "views"
        ])

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
            st = it.get("statistics", {}) or {}
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
                "views": int(st.get("viewCount", 0) or 0),
            })

    df = pd.DataFrame(rows).dropna(subset=["video_id"])
    return df

def fetch_channels(youtube, channel_ids: List[str]) -> pd.DataFrame:
    """
    Fetch channel stats (subscriberCount may be hidden for some channels).
    Batched up to 50.
    """
    if not channel_ids:
        return pd.DataFrame(columns=["channel_id", "channel_title", "subscribers", "updated_at"])

    rows = []
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
            st = it.get("statistics", {}) or {}
            subs = st.get("subscriberCount")
            rows.append({
                "channel_id": it.get("id"),
                "channel_title": sn.get("title"),
                "subscribers": int(subs) if subs is not None else None,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["updated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        df = pd.DataFrame(columns=["channel_id", "channel_title", "subscribers", "updated_at"])
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
# Main entry point
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

    all_videos = []
    global_seen_video_ids = set()

    for niche, queries in query_packs.items():
        if not queries:
            continue

        qlist = queries[: budget.queries_per_niche]

        found_by: Dict[str, str] = {}
        niche_video_ids: List[str] = []

        for q in qlist:
            # If quota is exceeded here, HttpError will bubble up to app.py
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
            continue

        niche_video_ids = [vid for vid in niche_video_ids if vid not in global_seen_video_ids]
        global_seen_video_ids.update(niche_video_ids)

        if not niche_video_ids:
            continue

        # If quota is exceeded here, HttpError will bubble up to app.py
        dfv = fetch_videos(youtube, niche_video_ids)
        if dfv.empty:
            continue

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
            continue

        dfv = dfv.sort_values("views", ascending=False).head(budget.max_videos_per_niche).copy()

        now = datetime.now(timezone.utc)
        pub = pd.to_datetime(dfv["published_at"], utc=True, errors="coerce")
        age_days = (now - pub).dt.total_seconds() / 86400.0
        age_days = age_days.clip(lower=0.25)
        dfv["views_per_day"] = dfv["views"] / age_days

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

    # Add convenient URLs for UI
    if not videos.empty:
        videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
        videos["channel_url"] = "https://www.youtube.com/channel/" + videos["channel_id"].astype(str)

    # Persist
    if engine is not None and not videos.empty:
        _upsert_videos(engine, videos)
        if not channels.empty:
            _upsert_channels(engine, channels)

        stats = videos[["video_id", "views", "views_per_day", "niche", "shorts_score", "source_query"]].copy()
        stats["date"] = date.today().isoformat()
        _insert_daily_stats(engine, stats)

    return videos, channels


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

    try:
        videos_df, channels_df = run_scan(
            api_key=api_key,
            query_packs=QUERY_PACKS,
            budget=budget,
            db_url=db_url,
        )
        print(f"Videos: {len(videos_df)} | Channels: {len(channels_df)}")
    except HttpError as e:
        print("API scan failed (likely quota/rate limit).")
        print(str(e))

    if db_url and SQLALCHEMY_AVAILABLE:
        snap = load_latest_snapshot(db_url)
        print(f"Latest DB snapshot rows: {len(snap)}")
