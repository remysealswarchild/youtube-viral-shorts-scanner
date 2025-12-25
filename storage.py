# storage.py
# SQLite persistence for YouTube Shorts Virality Scanner
#
# Stores:
# - videos: stable metadata per video_id
# - channels: stable metadata per channel_id (subscribers may be hidden)
# - video_stats_daily: daily snapshot of views + derived metrics per video_id and niche
#
# Design goals:
# - Idempotent daily writes: PRIMARY KEY(video_id, date) prevents duplicates
# - Fast reads for dashboard: load_latest_snapshot joins stats + metadata
# - Safe upserts for metadata changes over time

from __future__ import annotations

from typing import Iterable, Optional, Set

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# -----------------------------
# Engine / schema
# -----------------------------

def get_engine(db_url: str) -> Engine:
    if not db_url:
        raise ValueError("db_url is required (e.g., sqlite:///yt_shorts.db)")
    return create_engine(db_url, future=True)

def init_db(engine: Engine) -> None:
    with engine.begin() as con:
        # Video metadata (stable-ish)
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

        # Channel metadata (subscribers may be null/hidden)
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_title TEXT,
            subscribers INTEGER,
            updated_at TEXT
        );
        """))

        # Daily snapshot: views & derived virality measures at time of scan
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS video_stats_daily (
            video_id TEXT NOT NULL,
            date TEXT NOT NULL,                  -- YYYY-MM-DD (UTC)
            views INTEGER,
            views_per_day REAL,
            niche TEXT,
            shorts_score REAL,
            source_query TEXT,
            PRIMARY KEY (video_id, date)
        );
        """))

        # Helpful indices for trend queries
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_date ON video_stats_daily(date);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_niche ON video_stats_daily(niche);"))
        con.execute(text("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);"))


# -----------------------------
# Helpers
# -----------------------------

def existing_video_ids(engine: Engine, ids: Iterable[str]) -> Set[str]:
    """
    Return the subset of `ids` that already exist in videos table.
    Useful if you want to avoid refetching metadata (optional optimization).
    """
    ids = [i for i in ids if i]
    if not ids:
        return set()

    # SQLite has a limit on the number of bound parameters; chunk safely.
    out: Set[str] = set()
    chunk_size = 900

    with engine.begin() as con:
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i+chunk_size]
            res = con.execute(
                text(f"SELECT video_id FROM videos WHERE video_id IN ({','.join([':i'+str(k) for k in range(len(chunk))])})"),
                {f"i{k}": chunk[k] for k in range(len(chunk))}
            ).fetchall()
            out.update(r[0] for r in res)
    return out


# -----------------------------
# Upserts
# -----------------------------

def upsert_videos(engine: Engine, df: pd.DataFrame) -> None:
    """
    Upsert video metadata. Keeps latest known title/description/channel_title etc.
    """
    if df is None or df.empty:
        return

    required = ["video_id", "title", "description", "channel_id", "channel_title", "published_at", "duration_s"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"upsert_videos: missing required column '{c}'")

    data = df[required].drop_duplicates("video_id").copy()
    data["duration_s"] = pd.to_numeric(data["duration_s"], errors="coerce").fillna(0).astype(int)

    sql = text("""
        INSERT INTO videos(video_id, title, description, channel_id, channel_title, published_at, duration_s)
        VALUES(:video_id, :title, :description, :channel_id, :channel_title, :published_at, :duration_s)
        ON CONFLICT(video_id) DO UPDATE SET
            title=excluded.title,
            description=excluded.description,
            channel_id=excluded.channel_id,
            channel_title=excluded.channel_title,
            published_at=excluded.published_at,
            duration_s=excluded.duration_s
    """)

    with engine.begin() as con:
        con.execute(sql, data.to_dict(orient="records"))


def upsert_channels(engine: Engine, df: pd.DataFrame) -> None:
    """
    Upsert channel metadata.
    - If subscriberCount is hidden, subscribers may be NULL.
    """
    if df is None or df.empty:
        return

    required = ["channel_id", "channel_title", "subscribers", "updated_at"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"upsert_channels: missing required column '{c}'")

    data = df[required].drop_duplicates("channel_id").copy()
    # subscribers can be None; normalize numeric where present
    if "subscribers" in data.columns:
        data["subscribers"] = pd.to_numeric(data["subscribers"], errors="coerce").astype("Int64")

    sql = text("""
        INSERT INTO channels(channel_id, channel_title, subscribers, updated_at)
        VALUES(:channel_id, :channel_title, :subscribers, :updated_at)
        ON CONFLICT(channel_id) DO UPDATE SET
            channel_title=excluded.channel_title,
            subscribers=excluded.subscribers,
            updated_at=excluded.updated_at
    """)

    with engine.begin() as con:
        con.execute(sql, data.to_dict(orient="records"))


def insert_daily_stats(engine: Engine, df: pd.DataFrame) -> None:
    """
    Insert daily stats snapshot.
    Idempotent by design: PRIMARY KEY(video_id, date) and INSERT OR IGNORE.
    """
    if df is None or df.empty:
        return

    required = ["video_id", "date", "views", "views_per_day", "niche", "shorts_score", "source_query"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"insert_daily_stats: missing required column '{c}'")

    data = df[required].copy()
    data["views"] = pd.to_numeric(data["views"], errors="coerce").fillna(0).astype(int)
    data["views_per_day"] = pd.to_numeric(data["views_per_day"], errors="coerce").fillna(0.0).astype(float)
    data["shorts_score"] = pd.to_numeric(data["shorts_score"], errors="coerce").fillna(0.0).astype(float)

    sql = text("""
        INSERT OR IGNORE INTO video_stats_daily(video_id, date, views, views_per_day, niche, shorts_score, source_query)
        VALUES(:video_id, :date, :views, :views_per_day, :niche, :shorts_score, :source_query)
    """)

    with engine.begin() as con:
        con.execute(sql, data.to_dict(orient="records"))


# -----------------------------
# Reads for dashboard
# -----------------------------

def get_latest_date(engine: Engine) -> Optional[str]:
    with engine.begin() as con:
        d = con.execute(text("SELECT MAX(date) FROM video_stats_daily")).scalar()
    return d

def load_latest_snapshot(engine_or_db_url) -> pd.DataFrame:
    """
    Load the latest available date snapshot joined with videos table.
    Accepts either:
    - Engine
    - db_url string
    """
    engine = engine_or_db_url if hasattr(engine_or_db_url, "begin") else get_engine(engine_or_db_url)

    max_date = get_latest_date(engine)
    if not max_date:
        return pd.DataFrame()

    with engine.begin() as con:
        df = pd.read_sql(
            text("""
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
                v.duration_s
            FROM video_stats_daily s
            JOIN videos v ON v.video_id = s.video_id
            WHERE s.date = :d
            """),
            con,
            params={"d": max_date},
        )

    return df
