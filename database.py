import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "coretax.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT    NOT NULL,
            uploaded_at  TEXT    DEFAULT (datetime('now','localtime')),
            total_tweets INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tweets (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id         INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
            full_text         TEXT,
            clean_text        TEXT,
            periode           TEXT,
            created_at_tweet  TEXT,
            sentimen_prediksi TEXT,
            confidence        REAL
        );

        CREATE TABLE IF NOT EXISTS bertopic_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id        INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
            periode          TEXT,
            topic_id         INTEGER,
            jumlah_tweet     INTEGER,
            kata_kunci       TEXT,
            sentimen_dominan TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tweets_upload   ON tweets(upload_id);
        CREATE INDEX IF NOT EXISTS idx_tweets_periode  ON tweets(periode);
        CREATE INDEX IF NOT EXISTS idx_tweets_sentimen ON tweets(sentimen_prediksi);
        CREATE INDEX IF NOT EXISTS idx_bt_upload       ON bertopic_results(upload_id);
    """)
    conn.commit()
    conn.close()
