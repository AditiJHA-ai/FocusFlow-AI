import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "focusflow.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT,
            duration_mins REAL,
            focus_score   REAL,
            phone_count   INTEGER,
            posture_count INTEGER,
            grade         TEXT
        )
    """)
    con.commit()
    con.close()

def save_session(duration_mins, focus_score, phone_count, posture_count):
    grade = (
        "A+" if focus_score >= 90 else
        "A"  if focus_score >= 80 else
        "B"  if focus_score >= 70 else
        "C"  if focus_score >= 50 else "F"
    )
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions (date, duration_mins, focus_score, phone_count, posture_count, grade) VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), round(duration_mins, 1),
         round(focus_score, 1), phone_count, posture_count, grade)
    )
    con.commit()
    con.close()

def get_sessions(limit=30):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]