"""
Feedback Store — SQLite-backed persistence for the recursive learning system.

Six tables track every user interaction:
  sessions, feedback, ocr_corrections, match_selections,
  verification_logs, dimension_edits

WAL mode is enabled for concurrent read access from the Streamlit UI
while writes occur from pipeline callbacks.
"""

import os
import sqlite3
import uuid
import random
from datetime import datetime, timedelta
from typing import Optional


class FeedbackStore:
    """Persistent feedback store backed by SQLite with WAL mode."""

    def __init__(self, db_path: str = "sketch_to_cad/data/feedback.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        stmts = [
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'active'
            )""",
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                accuracy_rating INTEGER,
                match_rating INTEGER,
                comments TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS ocr_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                field_type TEXT DEFAULT 'dimension',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS match_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                reference_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL NOT NULL,
                action TEXT NOT NULL DEFAULT 'accept',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                total_checked INTEGER NOT NULL,
                max_error REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS dimension_edits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                original_value TEXT NOT NULL,
                edited_value TEXT NOT NULL,
                dimension_label TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )""",
        ]
        cur = self._conn.cursor()
        for stmt in stmts:
            cur.execute(stmt)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        """Create a new pipeline session and return its UUID."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO sessions (session_id, created_at, status) VALUES (?, ?, ?)",
            (session_id, now, "active"),
        )
        self._conn.commit()
        return session_id

    def complete_session(self, session_id: str):
        """Mark a session as completed."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE sessions SET completed_at = ?, status = 'completed' WHERE session_id = ?",
            (now, session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------
    def log_match_selection(
        self,
        session_id: str,
        reference_id: str,
        rank: int,
        score: float,
        action: str = "accept",
    ):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO match_selections (session_id, reference_id, rank, score, action, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, reference_id, rank, score, action, now),
        )
        self._conn.commit()

    def log_ocr_correction(
        self,
        session_id: str,
        raw_text: str,
        corrected_text: str,
        field_type: str = "dimension",
    ):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO ocr_corrections (session_id, raw_text, corrected_text, field_type, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, raw_text, corrected_text, field_type, now),
        )
        self._conn.commit()

    def log_dimension_edit(
        self,
        session_id: str,
        original_value: str,
        edited_value: str,
        dimension_label: str = "",
    ):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO dimension_edits (session_id, original_value, edited_value, dimension_label, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, original_value, edited_value, dimension_label, now),
        )
        self._conn.commit()

    def log_verification(
        self,
        session_id: str,
        iteration: int,
        passed: int,
        total_checked: int,
        max_error: float = 0.0,
    ):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO verification_logs (session_id, iteration, passed, total_checked, max_error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, iteration, passed, total_checked, max_error, now),
        )
        self._conn.commit()

    def log_feedback(
        self,
        session_id: str,
        accuracy_rating: int,
        match_rating: int,
        comments: str = "",
    ):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO feedback (session_id, accuracy_rating, match_rating, comments, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, accuracy_rating, match_rating, comments, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query / analytics helpers
    # ------------------------------------------------------------------
    def get_session_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def get_completed_session_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'completed'"
        ).fetchone()
        return row[0] if row else 0

    def get_correction_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM ocr_corrections").fetchone()
        return row[0] if row else 0

    def get_avg_accuracy(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(accuracy_rating) FROM feedback WHERE accuracy_rating IS NOT NULL"
        ).fetchone()
        return round(row[0], 2) if row and row[0] is not None else 0.0

    def get_avg_match_rating(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(match_rating) FROM feedback WHERE match_rating IS NOT NULL"
        ).fetchone()
        return round(row[0], 2) if row and row[0] is not None else 0.0

    def get_match_accept_rate(self) -> float:
        total = self._conn.execute("SELECT COUNT(*) FROM match_selections").fetchone()[0]
        if total == 0:
            return 0.0
        accepted = self._conn.execute(
            "SELECT COUNT(*) FROM match_selections WHERE action = 'accept'"
        ).fetchone()[0]
        return round(accepted / total * 100, 1)

    def get_recent_corrections(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT raw_text, corrected_text, COUNT(*) as cnt "
            "FROM ocr_corrections "
            "GROUP BY raw_text, corrected_text "
            "ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"raw": r["raw_text"], "corrected": r["corrected_text"], "count": r["cnt"]}
            for r in rows
        ]

    def get_correction_pairs(self, min_count: int = 2) -> dict[str, str]:
        """Return {raw_text: corrected_text} for corrections seen >= min_count times."""
        rows = self._conn.execute(
            "SELECT raw_text, corrected_text, COUNT(*) as cnt "
            "FROM ocr_corrections "
            "GROUP BY raw_text, corrected_text "
            "HAVING cnt >= ? "
            "ORDER BY cnt DESC",
            (min_count,),
        ).fetchall()
        return {r["raw_text"]: r["corrected_text"] for r in rows}

    def get_sessions_this_week(self) -> int:
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE created_at >= ?", (week_ago,)
        ).fetchone()
        return row[0] if row else 0

    def get_verification_pass_rate(self) -> float:
        row = self._conn.execute(
            "SELECT SUM(passed), SUM(total_checked) FROM verification_logs"
        ).fetchone()
        if row and row[1] and row[1] > 0:
            return round(row[0] / row[1] * 100, 1)
        return 0.0

    # ------------------------------------------------------------------
    # Demo seed data
    # ------------------------------------------------------------------
    def seed_demo_data(self):
        """Insert realistic sample data for the show-and-tell demo."""
        existing = self.get_session_count()
        if existing >= 5:
            return  # already seeded or has real data

        pool_types = [
            ("reference_001", "Rectangular Pool"),
            ("reference_002", "L-Shaped Pool"),
            ("reference_003", "Kidney Pool"),
            ("reference_004", "Freeform Pool"),
            ("reference_005", "Rectangular with Spa"),
        ]

        ocr_correction_pairs = [
            ("24", "2'4\"", "dimension"),
            ("126", "12'6\"", "dimension"),
            ("20`", "20'", "dimension"),
            ("10 ft", "10'", "dimension"),
            ("8''", "8\"", "dimension"),
            ("15,6\"", "15'6\"", "dimension"),
            ("3 6", "3'6\"", "dimension"),
        ]

        now = datetime.utcnow()

        for i in range(12):
            # Spread sessions over the past 14 days
            session_time = now - timedelta(days=random.randint(0, 13), hours=random.randint(0, 12))
            sid = str(uuid.uuid4())
            completed = random.random() > 0.15  # 85% completion rate

            self._conn.execute(
                "INSERT INTO sessions (session_id, created_at, completed_at, status) "
                "VALUES (?, ?, ?, ?)",
                (
                    sid,
                    session_time.isoformat(),
                    (session_time + timedelta(minutes=random.randint(3, 20))).isoformat()
                    if completed
                    else None,
                    "completed" if completed else "active",
                ),
            )

            # Match selection
            ref_id, ref_name = random.choice(pool_types)
            rank = random.randint(1, 3)
            score = round(random.uniform(0.72, 0.96), 4)
            self._conn.execute(
                "INSERT INTO match_selections (session_id, reference_id, rank, score, action, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, ref_id, rank, score, "accept", session_time.isoformat()),
            )

            # OCR corrections (1-3 per session, only some sessions)
            if random.random() > 0.3:
                n_corrections = random.randint(1, 3)
                for _ in range(n_corrections):
                    raw, corrected, ftype = random.choice(ocr_correction_pairs)
                    self._conn.execute(
                        "INSERT INTO ocr_corrections (session_id, raw_text, corrected_text, field_type, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (sid, raw, corrected, ftype, session_time.isoformat()),
                    )

            # Dimension edits (some sessions)
            if random.random() > 0.4:
                for label in random.sample(["length", "width", "depth"], k=random.randint(1, 2)):
                    orig = f"{random.randint(8, 25)}'"
                    edited = f"{random.randint(8, 25)}'{random.randint(0, 11)}\""
                    self._conn.execute(
                        "INSERT INTO dimension_edits (session_id, original_value, edited_value, dimension_label, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (sid, orig, edited, label, session_time.isoformat()),
                    )

            # Verification logs (1-3 iterations)
            if completed:
                n_iter = random.randint(1, 3)
                for it in range(1, n_iter + 1):
                    total = random.randint(4, 8)
                    passed = total if it == n_iter else random.randint(total - 2, total)
                    max_err = round(random.uniform(0.0, 1.5), 2) if passed < total else 0.0
                    self._conn.execute(
                        "INSERT INTO verification_logs (session_id, iteration, passed, total_checked, max_error, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (sid, it, passed, total, max_err, session_time.isoformat()),
                    )

            # Feedback (completed sessions)
            if completed and random.random() > 0.2:
                self._conn.execute(
                    "INSERT INTO feedback (session_id, accuracy_rating, match_rating, comments, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        sid,
                        random.randint(3, 5),
                        random.randint(3, 5),
                        random.choice([
                            "Works great!",
                            "Dimensions were slightly off on the deep end.",
                            "Good match, stair placement was accurate.",
                            "Had to correct a few OCR readings.",
                            "",
                            "Excellent result on first try.",
                            "L-shape corner needed manual adjustment.",
                        ]),
                        session_time.isoformat(),
                    ),
                )

        self._conn.commit()

    def close(self):
        self._conn.close()
