"""SQLite persistence for content, appeals, and immutable event records."""

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


@contextmanager
def connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                attribution TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_combined_score REAL,
                label TEXT NOT NULL,
                analysis_status TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'classified',
                analysis_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS appeals (
                appeal_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL UNIQUE REFERENCES content(content_id),
                creator_id TEXT NOT NULL,
                appeal_reasoning TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL REFERENCES content(content_id),
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp DESC);
        ''')


def save_decision(path, content_id, creator_id, original_text, decision):
    timestamp = utc_now()
    event = {'event_id': str(uuid4()), 'timestamp': timestamp,
             'content_id': content_id, 'creator_id': creator_id,
             'event_type': 'decision', **decision}
    with connect(path) as connection:
        connection.execute('''INSERT INTO content
            (content_id, creator_id, text, created_at, attribution, confidence,
             raw_combined_score, label, analysis_status, status, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (content_id, creator_id, original_text, timestamp,
             decision['attribution'], decision['confidence'], decision['raw_combined_score'],
             decision['label'], decision['analysis_status'], decision['status'],
             json.dumps(decision, ensure_ascii=False)))
        connection.execute('''INSERT INTO audit_events
            (event_id, content_id, event_type, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?)''',
            (event['event_id'], content_id, 'decision', timestamp,
             json.dumps(event, ensure_ascii=False)))
    return event


def get_log(path, limit):
    with connect(path) as connection:
        rows = connection.execute('''SELECT audit_events.payload_json,
                content.attribution, content.confidence
            FROM audit_events JOIN content USING (content_id)
            ORDER BY audit_events.timestamp DESC, audit_events.rowid DESC LIMIT ?''',
            (limit,)).fetchall()
    events = []
    for row in rows:
        event = json.loads(row['payload_json'])
        if event['event_type'] == 'appeal':
            event.setdefault('attribution', row['attribution'])
            event.setdefault('confidence', row['confidence'])
        events.append(event)
    return events


def save_appeal(path, content_id, creator_id, reasoning):
    """Return (status, payload); all three writes commit or roll back together."""
    with connect(path) as connection:
        connection.execute('BEGIN IMMEDIATE')
        content = connection.execute('SELECT * FROM content WHERE content_id=?',
                                     (content_id,)).fetchone()
        if content is None:
            return 'missing', None
        if content['creator_id'] != creator_id:
            return 'forbidden', None
        existing = connection.execute('SELECT appeal_id FROM appeals WHERE content_id=?',
                                      (content_id,)).fetchone()
        if existing:
            return 'duplicate', {'appeal_id': existing['appeal_id'],
                                 'content_id': content_id, 'status': content['status']}
        original = connection.execute('''SELECT event_id FROM audit_events
            WHERE content_id=? AND event_type='decision' ORDER BY rowid LIMIT 1''',
            (content_id,)).fetchone()
        timestamp = utc_now()
        appeal_id = str(uuid4())
        connection.execute('''INSERT INTO appeals
            (appeal_id, content_id, creator_id, appeal_reasoning, created_at)
            VALUES (?, ?, ?, ?, ?)''',
            (appeal_id, content_id, creator_id, reasoning, timestamp))
        connection.execute('UPDATE content SET status=? WHERE content_id=?',
                           ('under_review', content_id))
        event = {'event_id': str(uuid4()), 'timestamp': timestamp,
                 'event_type': 'appeal', 'content_id': content_id,
                 'creator_id': creator_id, 'appeal_id': appeal_id,
                 'attribution': content['attribution'],
                 'confidence': content['confidence'],
                 'original_decision_event_id': original['event_id'],
                 'appeal_reasoning': reasoning, 'appeal_timestamp': timestamp,
                 'status': 'under_review'}
        connection.execute('''INSERT INTO audit_events
            (event_id, content_id, event_type, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?)''',
            (event['event_id'], content_id, 'appeal', timestamp,
             json.dumps(event, ensure_ascii=False)))
    return 'created', {'appeal_id': appeal_id, 'content_id': content_id,
                       'status': 'under_review'}
