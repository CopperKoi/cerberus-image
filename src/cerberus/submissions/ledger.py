"""Durable answer ledger; remote side effects are never assumed exactly once."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Status = Literal["captured", "queued", "sending", "accepted", "rejected", "unknown"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Candidate:
    id: str
    run_id: str
    task_id: str
    instance_epoch: int
    question_id: str
    answer: str
    evidence_ref: str
    status: Status
    attempts: int
    remote_status: str | None
    remote_response: dict | None = None
    confirmed_question_id: str | None = None


class AnswerLedger:
    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.RLock()
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
            instance_epoch INTEGER NOT NULL, question_id TEXT NOT NULL,
            answer TEXT NOT NULL, evidence_ref TEXT NOT NULL,
            status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            remote_status TEXT, remote_response TEXT, confirmed_question_id TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS candidate_status ON candidates(run_id, status);
        CREATE TABLE IF NOT EXISTS submission_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL,
            event TEXT NOT NULL, remote_status TEXT, occurred_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS platform_confirmations (
            run_id TEXT NOT NULL, task_id TEXT NOT NULL, question_id TEXT NOT NULL,
            observed_at TEXT NOT NULL, source TEXT NOT NULL,
            PRIMARY KEY (run_id, task_id, question_id)
        );
        """)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(candidates)")}
        if "remote_response" not in columns:
            self._db.execute("ALTER TABLE candidates ADD COLUMN remote_response TEXT")
        if "confirmed_question_id" not in columns:
            self._db.execute("ALTER TABLE candidates ADD COLUMN confirmed_question_id TEXT")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def key(run_id: str, task_id: str, instance_epoch: int, question_id: str, answer: str) -> str:
        if not all((run_id, task_id, question_id)) or instance_epoch < 0 or not answer:
            raise ValueError("candidate identity and answer are required")
        parts = (run_id, task_id, str(instance_epoch), question_id,
                 hashlib.sha256(answer.encode("utf-8")).hexdigest())
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def _read(self, candidate_id: str) -> Candidate:
        row = self._db.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return Candidate(row["id"], row["run_id"], row["task_id"], row["instance_epoch"],
                         row["question_id"], row["answer"], row["evidence_ref"], row["status"],
                         row["attempts"], row["remote_status"],
                         json.loads(row["remote_response"]) if row["remote_response"] else None,
                         row["confirmed_question_id"])

    def capture(self, run_id: str, task_id: str, instance_epoch: int, question_id: str,
                answer: str, evidence_ref: str) -> Candidate:
        if not evidence_ref:
            raise ValueError("candidate requires evidence")
        candidate_id = self.key(run_id, task_id, instance_epoch, question_id, answer)
        with self._lock, self._db:
            now = _now()
            self._db.execute("""INSERT OR IGNORE INTO candidates
                (id,run_id,task_id,instance_epoch,question_id,answer,evidence_ref,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,'captured',?,?)""",
                (candidate_id, run_id, task_id, instance_epoch, question_id, answer, evidence_ref, now, now))
            self._db.execute("INSERT INTO submission_events(candidate_id,event,occurred_at) VALUES (?,?,?)",
                             (candidate_id, "capture", now))
            return self._read(candidate_id)

    def _transition(self, candidate_id: str, allowed: tuple[Status, ...], new: Status,
                    remote_status: str | None = None) -> Candidate:
        with self._lock, self._db:
            old = self._read(candidate_id)
            if old.status not in allowed:
                raise ValueError(f"invalid submission transition {old.status} -> {new}")
            now = _now()
            self._db.execute("""UPDATE candidates SET status=?, remote_status=?, updated_at=?,
                attempts=attempts+? WHERE id=?""",
                (new, remote_status, now, 1 if new == "sending" else 0, candidate_id))
            self._db.execute("""INSERT INTO submission_events(candidate_id,event,remote_status,occurred_at)
                VALUES (?,?,?,?)""", (candidate_id, new, remote_status, now))
            return self._read(candidate_id)

    def queue(self, candidate_id: str) -> Candidate:
        return self._transition(candidate_id, ("captured",), "queued")

    def mark_sending(self, candidate_id: str) -> Candidate:
        return self._transition(candidate_id, ("queued",), "sending")

    def record_result(self, candidate_id: str, status: Literal["accepted", "rejected", "unknown"],
                      *, remote_status: str | None = None,
                      remote_response: dict | None = None,
                      confirmed_question_id: str | None = None) -> Candidate:
        if status not in ("accepted", "rejected", "unknown"):
            raise ValueError("invalid platform result")
        if confirmed_question_id is not None and (status != "accepted" or not confirmed_question_id):
            raise ValueError("only an accepted result can confirm a question identity")
        with self._lock, self._db:
            result = self._transition(candidate_id, ("sending", "unknown"), status, remote_status)
            if remote_response is not None or confirmed_question_id is not None:
                self._db.execute("""UPDATE candidates SET remote_response=?, confirmed_question_id=?
                                    WHERE id=?""",
                                 (json.dumps(remote_response, ensure_ascii=False) if remote_response is not None
                                  else None, confirmed_question_id, candidate_id))
                result = self._read(candidate_id)
            return result

    def confirm_question(self, run_id: str, task_id: str, question_id: str, *, source: str) -> None:
        """Record a platform-visible solved state without attributing it to a candidate."""
        if not all((run_id, task_id, question_id, source)):
            raise ValueError("confirmation identity and source are required")
        with self._lock, self._db:
            self._db.execute("""INSERT OR IGNORE INTO platform_confirmations
                (run_id, task_id, question_id, observed_at, source) VALUES (?,?,?,?,?)""",
                (run_id, task_id, question_id, _now(), source))

    def recover_interrupted(self, run_id: str) -> int:
        """An in-flight request could have reached the platform; mark uncertain."""
        with self._lock, self._db:
            rows = self._db.execute("SELECT id FROM candidates WHERE run_id=? AND status='sending'",
                                    (run_id,)).fetchall()
            for row in rows:
                self._transition(row["id"], ("sending",), "unknown", "interrupted")
            return len(rows)

    def list(self, run_id: str, *, statuses: tuple[Status, ...] | None = None) -> list[Candidate]:
        with self._lock:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = self._db.execute(
                    f"SELECT id FROM candidates WHERE run_id=? AND status IN ({placeholders}) ORDER BY created_at,id",
                    (run_id, *statuses)).fetchall()
            else:
                rows = self._db.execute("SELECT id FROM candidates WHERE run_id=? ORDER BY created_at,id",
                                        (run_id,)).fetchall()
            return [self._read(row["id"]) for row in rows]

    def accepted_questions(self, run_id: str) -> dict[str, set[str]]:
        accepted: dict[str, set[str]] = {}
        for candidate in self.list(run_id, statuses=("accepted",)):
            accepted.setdefault(candidate.task_id, set()).add(
                candidate.confirmed_question_id or candidate.question_id)
        for row in self._db.execute("""SELECT task_id, question_id FROM platform_confirmations
                                      WHERE run_id=?""", (run_id,)):
            accepted.setdefault(row["task_id"], set()).add(row["question_id"])
        return accepted
