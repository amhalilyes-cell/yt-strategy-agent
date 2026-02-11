CREATE TABLE IF NOT EXISTS video_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    video_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,

    snapshot_at TEXT NOT NULL,
    published_at TEXT NOT NULL,

    hours_since_publish REAL NOT NULL,
    bucket_hours INTEGER NOT NULL,

    views INTEGER NOT NULL,
    likes INTEGER,
    comments INTEGER,

    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_vs_video_bucket_time
ON video_snapshots(video_id, bucket_hours, snapshot_at);

CREATE INDEX IF NOT EXISTS idx_vs_channel_bucket_time
ON video_snapshots(channel_id, bucket_hours, snapshot_at);