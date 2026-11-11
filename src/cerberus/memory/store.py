"""Run-scoped evidence and fact storage.

The event log is authoritative. ``facts`` is a transactional, rebuildable
projection. An evidence object is durable before an event may reference it.
This module deliberately does not decide whether a model's claim is true.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal

FactKind = Literal["observation", "hypothesis", "negative_result"]
FactScope = Literal["task", "instance"]
VerificationStatus = Literal["unverified", "observed", "verified", "rejected"]


def _validate_task_messages(messages: Any) -> list[str]:
    """Validate a provider transcript, allowing only a final incomplete tool group."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("invalid task history messages")
    pending: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid task history message")
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("orphan or duplicate task tool result")
            if not isinstance(message.get("content"), str):
                raise ValueError("invalid task tool result content")
            pending.remove(call_id)
            continue
        if pending:
            raise ValueError("incomplete task tool group before next message")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("invalid task tool calls")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("invalid task tool call")
                call_id = call.get("id")
                function = call.get("function")
                if (not isinstance(call_id, str) or not call_id or call_id in pending
                        or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("invalid task tool call")
                try:
                    arguments = json.loads(function["arguments"])
                except ValueError as exc:
                    raise ValueError("invalid task tool arguments") from exc
                if not isinstance(arguments, dict):
                    raise ValueError("invalid task tool arguments")
                pending.append(call_id)
        elif role not in ("system", "user"):
            raise ValueError("invalid task history role")
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("invalid task history content")
    return pending


@dataclass(frozen=True)
class EvidenceRef:
    sha256: str
    size: int


@dataclass(frozen=True)
class EvidenceMatch:
    byte_offset: int
    snippet: str


class EvidenceStore:
    """SHA-256 object store with atomic writes and bounded reads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / digest[:2] / digest[2:]

    def put_bytes(self, content: bytes) -> EvidenceRef:
        from io import BytesIO

        return self.put_stream(BytesIO(content))

    def put_stream(self, source: BinaryIO) -> EvidenceRef:
        temp_dir = self.root / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as output:
                temp_path = Path(output.name)
                while chunk := source.read(1024 * 1024):
                    if not isinstance(chunk, bytes):
                        raise TypeError("evidence stream must return bytes")
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            ref = EvidenceRef(digest.hexdigest(), size)
            destination = self._path(ref.sha256)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and self.verify(ref):
                temp_path.unlink()
                return ref
            os.replace(temp_path, destination)
            temp_path = None
            self._sync_directory(destination.parent)
            return ref
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def verify(self, ref: EvidenceRef) -> bool:
        path = self._path(ref.sha256)
        try:
            with path.open("rb") as source:
                digest = hashlib.sha256()
                size = 0
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            return size == ref.size and digest.hexdigest() == ref.sha256
        except FileNotFoundError:
            return False

    def read_page(self, ref: EvidenceRef, offset: int = 0, limit: int = 65536) -> tuple[bytes, int, int]:
        if offset < 0 or limit < 1 or limit > 1024 * 1024:
            raise ValueError("invalid evidence page")
        path = self._path(ref.sha256)
        with path.open("rb") as source:
            size = source.seek(0, os.SEEK_END)
            if size != ref.size:
                raise OSError("evidence object size mismatch")
            source.seek(min(offset, size))
            data = source.read(limit)
            return data, source.tell(), size

    def search_text(
        self, ref: EvidenceRef, query: str, *, max_matches: int = 20, context_bytes: int = 80
    ) -> list[EvidenceMatch]:
        if not query or max_matches < 1 or max_matches > 100 or context_bytes < 0 or context_bytes > 4096:
            raise ValueError("invalid evidence search")
        needle = query.encode("utf-8")
        path = self._path(ref.sha256)
        with path.open("rb") as source:
            size = source.seek(0, os.SEEK_END)
            if size != ref.size:
                raise OSError("evidence object size mismatch")
            if size == 0:
                return []
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as content:
                matches: list[EvidenceMatch] = []
                position = 0
                while len(matches) < max_matches:
                    index = content.find(needle, position)
                    if index < 0:
                        break
                    start = max(0, index - context_bytes)
                    end = min(size, index + len(needle) + context_bytes)
                    matches.append(EvidenceMatch(index, content[start:end].decode("utf-8", errors="replace")))
                    position = index + len(needle)
                return matches


@dataclass(frozen=True)
class FactInput:
    run_id: str
    task_id: str
    title: str
    summary: str
    source: str
    evidence: EvidenceRef
    kind: FactKind = "observation"
    scope: FactScope = "instance"
    instance_epoch: int | None = None
    verification_status: VerificationStatus = "observed"
    conditions: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    causation_event_id: str | None = None
    supersedes_fact_id: str | None = None
    contradicts_fact_id: str | None = None


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event_id: str
    run_id: str
    task_id: str
    instance_epoch: int | None
    occurred_at: str
    kind: str
    payload: dict[str, object]
    causation_event_id: str | None
    evidence_sha256: str | None


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    run_id: str
    task_id: str
    instance_epoch: int | None
    scope: FactScope
    kind: FactKind
    title: str
    summary: str
    source: str
    evidence: EvidenceRef
    verification_status: VerificationStatus
    conditions: dict[str, str]
    parameters: dict[str, str]
    active: bool
    supersedes_fact_id: str | None
    contradicts_fact_id: str | None


@dataclass(frozen=True)
class ResourceInput:
    run_id: str
    task_id: str
    kind: str
    owner: str
    evidence: EvidenceRef
    scope: FactScope = "instance"
    instance_epoch: int | None = None
    expires_at: str | None = None
    conditions: dict[str, str] = field(default_factory=dict)
    causation_event_id: str | None = None


@dataclass(frozen=True)
class ResourceRecord:
    resource_id: str
    run_id: str
    task_id: str
    kind: str
    owner: str
    evidence: EvidenceRef
    scope: FactScope
    instance_epoch: int | None
    expires_at: str | None
    conditions: dict[str, str]
    active: bool


@dataclass(frozen=True)
class IntentInput:
    run_id: str
    task_id: str
    instance_epoch: int
    hypothesis: str
    expected_observation: str
    stop_condition: str
    prerequisite_fact_ids: tuple[str, ...] = ()
    required_resource_ids: tuple[str, ...] = ()
    priority: float = 0.0
    budget_tokens: int = 0
    budget_tool_seconds: int = 0
    rationale: str = ""
    causation_event_id: str | None = None


@dataclass(frozen=True)
class IntentRecord:
    intent_id: str
    run_id: str
    task_id: str
    instance_epoch: int
    hypothesis: str
    expected_observation: str
    stop_condition: str
    prerequisite_fact_ids: tuple[str, ...]
    required_resource_ids: tuple[str, ...]
    priority: float
    budget_tokens: int
    budget_tool_seconds: int
    rationale: str
    status: str
    lease_owner: str | None
    lease_until: str | None
    attempts: int
    fencing_token: int
    completion_event_id: str | None


@dataclass(frozen=True)
class TaskHistory:
    """A durable, unverified provider transcript for one task in one run."""

    run_id: str
    task_id: str
    instance_epoch: int
    messages: list[dict[str, Any]]


class MemoryStore:
    """Single-writer event log and rebuildable fact projection.

    Each public write holds a SQLite immediate transaction. The root controller
    can serialize calls through this object; separate processes still contend
    through SQLite's WAL lock and busy timeout.
    """

    def __init__(self, database_path: str | Path, evidence: EvidenceStore) -> None:
        path = Path(database_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.RLock()
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                instance_epoch INTEGER,
                occurred_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                causation_event_id TEXT,
                evidence_sha256 TEXT,
                FOREIGN KEY (causation_event_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS events_by_task ON events(run_id, task_id, seq);
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                instance_epoch INTEGER,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                source TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                evidence_size INTEGER NOT NULL,
                verification_status TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                active INTEGER NOT NULL,
                supersedes_fact_id TEXT,
                contradicts_fact_id TEXT,
                FOREIGN KEY (fact_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS facts_current ON facts(run_id, task_id, active, scope, instance_epoch);
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                fact_id UNINDEXED, title, summary, source, tokenize='trigram'
            );
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                owner TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                evidence_size INTEGER NOT NULL,
                scope TEXT NOT NULL,
                instance_epoch INTEGER,
                expires_at TEXT,
                conditions_json TEXT NOT NULL,
                active INTEGER NOT NULL,
                FOREIGN KEY (resource_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS resources_current
                ON resources(run_id, task_id, active, scope, instance_epoch);
            CREATE TABLE IF NOT EXISTS intents (
                intent_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                instance_epoch INTEGER NOT NULL,
                hypothesis TEXT NOT NULL,
                expected_observation TEXT NOT NULL,
                stop_condition TEXT NOT NULL,
                prerequisite_fact_ids_json TEXT NOT NULL,
                required_resource_ids_json TEXT NOT NULL,
                priority REAL NOT NULL,
                budget_tokens INTEGER NOT NULL,
                budget_tool_seconds INTEGER NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_until TEXT,
                attempts INTEGER NOT NULL,
                fencing_token INTEGER NOT NULL DEFAULT 0,
                completion_event_id TEXT,
                FOREIGN KEY (intent_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS intents_frontier
                ON intents(run_id, task_id, instance_epoch, status, priority);
            CREATE TABLE IF NOT EXISTS task_histories (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                instance_epoch INTEGER NOT NULL,
                messages_json TEXT NOT NULL,
                messages_sha256 TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, task_id)
            );
            """
        )
        # Existing development runs predate fencing. Old claim events are
        # replayed as successive generations during rebuild_projection.
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(intents)")}
        if "fencing_token" not in columns:
            self._connection.execute(
                "ALTER TABLE intents ADD COLUMN fencing_token INTEGER NOT NULL DEFAULT 0"
            )
            self._connection.execute("UPDATE intents SET fencing_token = attempts")
        history_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(task_histories)")}
        if "messages_sha256" not in history_columns:
            self._connection.execute("ALTER TABLE task_histories ADD COLUMN messages_sha256 TEXT")

    def save_task_history(
        self, run_id: str, task_id: str, instance_epoch: int,
        messages: list[dict[str, Any]],
    ) -> None:
        """Atomically replace a task transcript, including an unfinished tool turn.

        The transcript is protocol state, not evidence or a verified fact. A
        caller recovering an unfinished tool turn must reconcile its outcome
        before making another model request or retrying the tool.
        """
        if not run_id or not task_id or instance_epoch < 0:
            raise ValueError("task history requires run, task and nonnegative epoch")
        _validate_task_messages(messages)
        encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._lock:
            self._connection.execute(
                """INSERT INTO task_histories
                   (run_id, task_id, instance_epoch, messages_json, messages_sha256, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, task_id) DO UPDATE SET
                     instance_epoch = excluded.instance_epoch,
                     messages_json = excluded.messages_json,
                     messages_sha256 = excluded.messages_sha256,
                     updated_at = excluded.updated_at""",
                (run_id, task_id, instance_epoch, encoded, digest, datetime.now(UTC).isoformat()),
            )

    def load_task_history(self, run_id: str, task_id: str) -> TaskHistory | None:
        if not run_id or not task_id:
            raise ValueError("task history requires run and task")
        with self._lock:
            row = self._connection.execute(
                """SELECT run_id, task_id, instance_epoch, messages_json, messages_sha256
                   FROM task_histories WHERE run_id = ? AND task_id = ?""",
                (run_id, task_id),
            ).fetchone()
        if row is None:
            return None
        encoded = row["messages_json"]
        digest = row["messages_sha256"]
        if digest is not None and hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest:
            raise ValueError("corrupt task history checksum")
        try:
            messages = json.loads(encoded)
            _validate_task_messages(messages)
        except (ValueError, TypeError) as exc:
            raise ValueError("corrupt task history transcript") from exc
        return TaskHistory(
            run_id=row["run_id"], task_id=row["task_id"],
            instance_epoch=row["instance_epoch"], messages=messages,
        )

    @staticmethod
    def _validate_fact(fact: FactInput) -> None:
        if not all((fact.run_id, fact.task_id, fact.title, fact.summary, fact.source)):
            raise ValueError("fact identity, title, summary and source are required")
        if fact.kind not in ("observation", "hypothesis", "negative_result"):
            raise ValueError("invalid fact kind")
        if fact.scope not in ("task", "instance"):
            raise ValueError("invalid fact scope")
        if fact.scope == "instance" and (fact.instance_epoch is None or fact.instance_epoch < 0):
            raise ValueError("instance facts require a nonnegative epoch")
        if fact.scope == "task" and fact.instance_epoch is not None:
            raise ValueError("task facts must not bind an instance epoch")
        if fact.verification_status not in ("unverified", "observed", "verified", "rejected"):
            raise ValueError("invalid verification status")
        if fact.kind == "hypothesis" and fact.verification_status == "verified":
            raise ValueError("record the verifying observation instead of a verified hypothesis")
        if fact.kind == "negative_result" and not fact.conditions:
            raise ValueError("negative results require bounded test conditions")
        if fact.supersedes_fact_id == fact.contradicts_fact_id and fact.supersedes_fact_id is not None:
            raise ValueError("a fact cannot both supersede and contradict the same fact")

    def record_fact(self, fact: FactInput) -> FactRecord:
        self._validate_fact(fact)
        if not self.evidence.verify(fact.evidence):
            raise ValueError("fact evidence is missing or corrupt")
        event_id = uuid.uuid4().hex
        occurred_at = datetime.now(UTC).isoformat()
        payload = {
            "schema_version": 1,
            "title": fact.title,
            "summary": fact.summary,
            "source": fact.source,
            "kind": fact.kind,
            "scope": fact.scope,
            "evidence_size": fact.evidence.size,
            "verification_status": fact.verification_status,
            "conditions": fact.conditions,
            "parameters": fact.parameters,
            "supersedes_fact_id": fact.supersedes_fact_id,
            "contradicts_fact_id": fact.contradicts_fact_id,
        }
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                if fact.causation_event_id is not None:
                    cause = connection.execute(
                        "SELECT run_id FROM events WHERE event_id = ?", (fact.causation_event_id,)
                    ).fetchone()
                    if cause is None or cause["run_id"] != fact.run_id:
                        raise ValueError("causation must reference an event in the same run")
                for reference in (fact.supersedes_fact_id, fact.contradicts_fact_id):
                    if reference is not None:
                        target = connection.execute(
                            "SELECT run_id, task_id, scope, instance_epoch FROM facts WHERE fact_id = ?",
                            (reference,),
                        ).fetchone()
                        if (
                            target is None
                            or target["run_id"] != fact.run_id
                            or target["task_id"] != fact.task_id
                            or target["scope"] != fact.scope
                            or target["instance_epoch"] != fact.instance_epoch
                        ):
                            raise ValueError("fact relation must stay within the same task and scope")
                connection.execute(
                    """INSERT INTO events
                    (event_id, run_id, task_id, instance_epoch, occurred_at, kind,
                     payload_json, causation_event_id, evidence_sha256)
                    VALUES (?, ?, ?, ?, ?, 'fact.recorded', ?, ?, ?)""",
                    (
                        event_id, fact.run_id, fact.task_id, fact.instance_epoch,
                        occurred_at, json.dumps(payload, sort_keys=True),
                        fact.causation_event_id, fact.evidence.sha256,
                    ),
                )
                self._apply_fact_event(connection, event_id, fact.run_id, fact.task_id, fact.instance_epoch, payload, fact.evidence.sha256)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return self.get_fact(event_id)

    @staticmethod
    def _apply_fact_event(
        connection: sqlite3.Connection,
        event_id: str,
        run_id: str,
        task_id: str,
        instance_epoch: int | None,
        payload: dict[str, object],
        evidence_sha256: str,
    ) -> None:
        superseded = payload["supersedes_fact_id"]
        if superseded is not None:
            result = connection.execute(
                "UPDATE facts SET active = 0 WHERE fact_id = ? AND active = 1", (superseded,)
            )
            if result.rowcount != 1:
                raise ValueError("superseded fact is already inactive")
        connection.execute(
            """INSERT INTO facts
            (fact_id, run_id, task_id, instance_epoch, scope, kind, title, summary,
             source, evidence_sha256, evidence_size, verification_status,
             conditions_json, parameters_json, active, supersedes_fact_id, contradicts_fact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                event_id, run_id, task_id, instance_epoch, payload["scope"],
                payload["kind"], payload["title"], payload["summary"], payload["source"],
                evidence_sha256, payload["evidence_size"], payload["verification_status"],
                json.dumps(payload["conditions"], sort_keys=True),
                json.dumps(payload["parameters"], sort_keys=True),
                superseded, payload["contradicts_fact_id"],
            ),
        )
        connection.execute(
            "INSERT INTO facts_fts (fact_id, title, summary, source) VALUES (?, ?, ?, ?)",
            (event_id, payload["title"], payload["summary"], payload["source"]),
        )

    @staticmethod
    def _fact_from_row(row: sqlite3.Row) -> FactRecord:
        return FactRecord(
            fact_id=row["fact_id"], run_id=row["run_id"], task_id=row["task_id"],
            instance_epoch=row["instance_epoch"], scope=row["scope"], kind=row["kind"],
            title=row["title"], summary=row["summary"], source=row["source"],
            evidence=EvidenceRef(row["evidence_sha256"], row["evidence_size"]),
            verification_status=row["verification_status"],
            conditions=json.loads(row["conditions_json"]),
            parameters=json.loads(row["parameters_json"]), active=bool(row["active"]),
            supersedes_fact_id=row["supersedes_fact_id"],
            contradicts_fact_id=row["contradicts_fact_id"],
        )

    def get_fact(self, fact_id: str) -> FactRecord:
        with self._lock:
            row = self._connection.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if row is None:
            raise KeyError(fact_id)
        return self._fact_from_row(row)

    def current_facts(self, run_id: str, task_id: str, instance_epoch: int) -> list[FactRecord]:
        """Return active task facts and facts from the current instance only."""
        if instance_epoch < 0:
            raise ValueError("instance epoch must be nonnegative")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM facts WHERE run_id = ? AND task_id = ? AND active = 1
                   AND (scope = 'task' OR (scope = 'instance' AND instance_epoch = ?))
                   ORDER BY rowid""",
                (run_id, task_id, instance_epoch),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def search_facts(
        self,
        run_id: str,
        task_id: str,
        instance_epoch: int,
        query: str,
        *,
        kind: FactKind | None = None,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> list[FactRecord]:
        """Search title, summary and source within a task's current scope.

        FTS5 trigram covers substrings of at least three Unicode characters.
        Shorter queries use a bounded SQL substring scan.
        """
        if not query or instance_epoch < 0 or limit < 1 or limit > 100:
            raise ValueError("invalid fact search")
        if kind is not None and kind not in ("observation", "hypothesis", "negative_result"):
            raise ValueError("invalid fact kind")
        sql = """SELECT facts.* FROM facts
                 WHERE facts.run_id = ? AND facts.task_id = ?
                   AND (facts.scope = 'task' OR facts.instance_epoch = ?)"""
        args: list[object] = [run_id, task_id, instance_epoch]
        if not include_inactive:
            sql += " AND facts.active = 1"
        if kind is not None:
            sql += " AND facts.kind = ?"
            args.append(kind)
        if len(query) >= 3:
            sql += " AND facts.fact_id IN (SELECT fact_id FROM facts_fts WHERE facts_fts MATCH ?)"
            args.append('"' + query.replace('"', '""') + '"')
        else:
            sql += " AND (instr(facts.title, ?) > 0 OR instr(facts.summary, ?) > 0 OR instr(facts.source, ?) > 0)"
            args.extend([query, query, query])
        sql += " ORDER BY facts.rowid DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, args).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def list_events(self, run_id: str, task_id: str | None = None, since_seq: int = 0) -> list[EventRecord]:
        if since_seq < 0:
            raise ValueError("since_seq must be nonnegative")
        sql = "SELECT * FROM events WHERE run_id = ? AND seq > ?"
        args: list[object] = [run_id, since_seq]
        if task_id is not None:
            sql += " AND task_id = ?"
            args.append(task_id)
        sql += " ORDER BY seq"
        with self._lock:
            rows = self._connection.execute(sql, args).fetchall()
        return [
            EventRecord(
                seq=row["seq"], event_id=row["event_id"], run_id=row["run_id"],
                task_id=row["task_id"], instance_epoch=row["instance_epoch"],
                occurred_at=row["occurred_at"], kind=row["kind"],
                payload=json.loads(row["payload_json"]),
                causation_event_id=row["causation_event_id"],
                evidence_sha256=row["evidence_sha256"],
            )
            for row in rows
        ]

    @staticmethod
    def _resource_from_row(row: sqlite3.Row) -> ResourceRecord:
        return ResourceRecord(
            resource_id=row["resource_id"], run_id=row["run_id"], task_id=row["task_id"],
            kind=row["kind"], owner=row["owner"],
            evidence=EvidenceRef(row["evidence_sha256"], row["evidence_size"]),
            scope=row["scope"], instance_epoch=row["instance_epoch"],
            expires_at=row["expires_at"], conditions=json.loads(row["conditions_json"]),
            active=bool(row["active"]),
        )

    def record_resource(self, resource: ResourceInput) -> ResourceRecord:
        if not all((resource.run_id, resource.task_id, resource.kind, resource.owner)):
            raise ValueError("resource identity, kind and owner are required")
        if resource.scope == "instance" and (resource.instance_epoch is None or resource.instance_epoch < 0):
            raise ValueError("instance resources require a nonnegative epoch")
        if resource.scope == "task" and resource.instance_epoch is not None:
            raise ValueError("task resources must not bind an instance epoch")
        if resource.scope not in ("task", "instance"):
            raise ValueError("invalid resource scope")
        if resource.expires_at is not None:
            self._parse_utc(resource.expires_at)
        if not self.evidence.verify(resource.evidence):
            raise ValueError("resource evidence is missing or corrupt")
        resource_id = uuid.uuid4().hex
        payload = {
            "schema_version": 1,
            "kind": resource.kind,
            "owner": resource.owner,
            "scope": resource.scope,
            "evidence_size": resource.evidence.size,
            "expires_at": resource.expires_at,
            "conditions": resource.conditions,
        }
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._check_causation(connection, resource.causation_event_id, resource.run_id)
                connection.execute(
                    """INSERT INTO events
                    (event_id, run_id, task_id, instance_epoch, occurred_at, kind,
                     payload_json, causation_event_id, evidence_sha256)
                    VALUES (?, ?, ?, ?, ?, 'resource.recorded', ?, ?, ?)""",
                    (
                        resource_id, resource.run_id, resource.task_id, resource.instance_epoch,
                        datetime.now(UTC).isoformat(), json.dumps(payload, sort_keys=True),
                        resource.causation_event_id, resource.evidence.sha256,
                    ),
                )
                self._apply_resource_event(
                    connection, resource_id, resource.run_id, resource.task_id,
                    resource.instance_epoch, payload, resource.evidence.sha256,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return self.get_resource(resource_id)

    @staticmethod
    def _check_causation(connection: sqlite3.Connection, event_id: str | None, run_id: str) -> None:
        if event_id is None:
            return
        row = connection.execute("SELECT run_id FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None or row["run_id"] != run_id:
            raise ValueError("causation must reference an event in the same run")

    @staticmethod
    def _apply_resource_event(
        connection: sqlite3.Connection,
        resource_id: str,
        run_id: str,
        task_id: str,
        instance_epoch: int | None,
        payload: dict[str, object],
        evidence_sha256: str,
    ) -> None:
        connection.execute(
            """INSERT INTO resources
            (resource_id, run_id, task_id, kind, owner, evidence_sha256, evidence_size,
             scope, instance_epoch, expires_at, conditions_json, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                resource_id, run_id, task_id, payload["kind"], payload["owner"],
                evidence_sha256, payload["evidence_size"], payload["scope"], instance_epoch,
                payload["expires_at"], json.dumps(payload["conditions"], sort_keys=True),
            ),
        )

    @staticmethod
    def _parse_utc(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must have a timezone")
        return parsed.astimezone(UTC)

    def get_resource(self, resource_id: str) -> ResourceRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM resources WHERE resource_id = ?", (resource_id,)
            ).fetchone()
        if row is None:
            raise KeyError(resource_id)
        return self._resource_from_row(row)

    def active_resources(
        self, run_id: str, task_id: str, instance_epoch: int, *, now: datetime | None = None
    ) -> list[ResourceRecord]:
        if instance_epoch < 0:
            raise ValueError("instance epoch must be nonnegative")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must have a timezone")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM resources WHERE run_id = ? AND task_id = ? AND active = 1
                   AND (scope = 'task' OR (scope = 'instance' AND instance_epoch = ?))
                   ORDER BY rowid""",
                (run_id, task_id, instance_epoch),
            ).fetchall()
        return [
            item for row in rows
            if (item := self._resource_from_row(row)).expires_at is None
            or self._parse_utc(item.expires_at) > current
        ]

    def revoke_resource(self, resource_id: str, reason: str) -> ResourceRecord:
        if not reason:
            raise ValueError("revocation reason is required")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM resources WHERE resource_id = ? AND active = 1", (resource_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(resource_id)
                connection.execute(
                    """INSERT INTO events
                    (event_id, run_id, task_id, instance_epoch, occurred_at, kind,
                     payload_json, causation_event_id, evidence_sha256)
                    VALUES (?, ?, ?, ?, ?, 'resource.revoked', ?, ?, NULL)""",
                    (
                        uuid.uuid4().hex, row["run_id"], row["task_id"], row["instance_epoch"],
                        datetime.now(UTC).isoformat(),
                        json.dumps({"schema_version": 1, "resource_id": resource_id, "reason": reason}),
                        resource_id,
                    ),
                )
                connection.execute("UPDATE resources SET active = 0 WHERE resource_id = ?", (resource_id,))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return self.get_resource(resource_id)

    @staticmethod
    def _intent_from_row(row: sqlite3.Row) -> IntentRecord:
        return IntentRecord(
            intent_id=row["intent_id"], run_id=row["run_id"], task_id=row["task_id"],
            instance_epoch=row["instance_epoch"], hypothesis=row["hypothesis"],
            expected_observation=row["expected_observation"],
            stop_condition=row["stop_condition"],
            prerequisite_fact_ids=tuple(json.loads(row["prerequisite_fact_ids_json"])),
            required_resource_ids=tuple(json.loads(row["required_resource_ids_json"])),
            priority=row["priority"], budget_tokens=row["budget_tokens"],
            budget_tool_seconds=row["budget_tool_seconds"], rationale=row["rationale"],
            status=row["status"], lease_owner=row["lease_owner"],
            lease_until=row["lease_until"], attempts=row["attempts"],
            fencing_token=row["fencing_token"],
            completion_event_id=row["completion_event_id"],
        )

    def get_intent(self, intent_id: str) -> IntentRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
        if row is None:
            raise KeyError(intent_id)
        return self._intent_from_row(row)

    @staticmethod
    def _validate_intent(intent: IntentInput) -> None:
        import math

        if not all((intent.run_id, intent.task_id, intent.hypothesis, intent.expected_observation, intent.stop_condition)):
            raise ValueError("intent identity, hypothesis, observation and stop condition are required")
        if intent.instance_epoch < 0 or intent.budget_tokens < 0 or intent.budget_tool_seconds < 0:
            raise ValueError("intent epoch and budgets must be nonnegative")
        if not math.isfinite(intent.priority):
            raise ValueError("intent priority must be finite")
        if len(set(intent.prerequisite_fact_ids)) != len(intent.prerequisite_fact_ids):
            raise ValueError("duplicate prerequisite fact")
        if len(set(intent.required_resource_ids)) != len(intent.required_resource_ids):
            raise ValueError("duplicate required resource")

    def record_intent(self, intent: IntentInput) -> IntentRecord:
        self._validate_intent(intent)
        payload: dict[str, object] = {
            "schema_version": 1,
            "hypothesis": intent.hypothesis,
            "expected_observation": intent.expected_observation,
            "stop_condition": intent.stop_condition,
            "prerequisite_fact_ids": intent.prerequisite_fact_ids,
            "required_resource_ids": intent.required_resource_ids,
            "priority": intent.priority,
            "budget_tokens": intent.budget_tokens,
            "budget_tool_seconds": intent.budget_tool_seconds,
            "rationale": intent.rationale,
        }
        event_id = uuid.uuid4().hex
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._check_causation(connection, intent.causation_event_id, intent.run_id)
                self._validate_intent_dependencies(connection, intent)
                self._insert_event(
                    connection, event_id, intent.run_id, intent.task_id,
                    intent.instance_epoch, "intent.recorded", payload, intent.causation_event_id,
                )
                self._apply_intent_event(
                    connection, event_id, intent.run_id, intent.task_id,
                    intent.instance_epoch, payload,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return self.get_intent(event_id)

    @staticmethod
    def _validate_intent_dependencies(connection: sqlite3.Connection, intent: IntentInput) -> None:
        for fact_id in intent.prerequisite_fact_ids:
            fact = connection.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
            if (
                fact is None or fact["run_id"] != intent.run_id or fact["task_id"] != intent.task_id
                or (fact["scope"] == "instance" and fact["instance_epoch"] != intent.instance_epoch)
            ):
                raise ValueError("prerequisite fact is outside the current task or instance")
        for resource_id in intent.required_resource_ids:
            resource = connection.execute(
                "SELECT * FROM resources WHERE resource_id = ?", (resource_id,)
            ).fetchone()
            if (
                resource is None or resource["run_id"] != intent.run_id
                or resource["task_id"] != intent.task_id
                or (resource["scope"] == "instance" and resource["instance_epoch"] != intent.instance_epoch)
            ):
                raise ValueError("required resource is outside the current task or instance")

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event_id: str,
        run_id: str,
        task_id: str,
        instance_epoch: int,
        kind: str,
        payload: dict[str, object],
        causation_event_id: str | None,
    ) -> None:
        connection.execute(
            """INSERT INTO events
            (event_id, run_id, task_id, instance_epoch, occurred_at, kind,
             payload_json, causation_event_id, evidence_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                event_id, run_id, task_id, instance_epoch,
                datetime.now(UTC).isoformat(), kind,
                json.dumps(payload, sort_keys=True), causation_event_id,
            ),
        )

    @staticmethod
    def _apply_intent_event(
        connection: sqlite3.Connection,
        event_id: str,
        run_id: str,
        task_id: str,
        instance_epoch: int,
        payload: dict[str, object],
    ) -> None:
        connection.execute(
            """INSERT INTO intents
            (intent_id, run_id, task_id, instance_epoch, hypothesis, expected_observation,
             stop_condition, prerequisite_fact_ids_json, required_resource_ids_json,
             priority, budget_tokens, budget_tool_seconds, rationale, status,
             lease_owner, lease_until, attempts, fencing_token, completion_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, NULL, 0, 0, NULL)""",
            (
                event_id, run_id, task_id, instance_epoch, payload["hypothesis"],
                payload["expected_observation"], payload["stop_condition"],
                json.dumps(payload["prerequisite_fact_ids"]),
                json.dumps(payload["required_resource_ids"]), payload["priority"],
                payload["budget_tokens"], payload["budget_tool_seconds"], payload["rationale"],
            ),
        )

    @staticmethod
    def _intent_ready(connection: sqlite3.Connection, intent: IntentRecord, now: datetime) -> bool:
        for fact_id in intent.prerequisite_fact_ids:
            fact = connection.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
            if (
                fact is None or not fact["active"] or fact["kind"] == "hypothesis"
                or fact["verification_status"] not in ("observed", "verified")
            ):
                return False
        for resource_id in intent.required_resource_ids:
            resource = connection.execute(
                "SELECT * FROM resources WHERE resource_id = ?", (resource_id,)
            ).fetchone()
            if (
                resource is None or not resource["active"]
                or (resource["expires_at"] is not None
                    and MemoryStore._parse_utc(resource["expires_at"]) <= now)
            ):
                return False
        return True

    def available_intents(
        self, run_id: str, task_id: str, instance_epoch: int, *, now: datetime | None = None
    ) -> list[IntentRecord]:
        if instance_epoch < 0:
            raise ValueError("instance epoch must be nonnegative")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must have a timezone")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM intents WHERE run_id = ? AND task_id = ? AND instance_epoch = ?
                   AND status IN ('queued', 'claimed')
                   ORDER BY priority DESC, rowid""",
                (run_id, task_id, instance_epoch),
            ).fetchall()
            return [
                intent for row in rows
                if (intent := self._intent_from_row(row)).status == "queued"
                or (intent.lease_until is not None and self._parse_utc(intent.lease_until) <= current)
                if self._intent_ready(self._connection, intent, current)
            ]

    def claim_intent(
        self,
        run_id: str,
        task_id: str,
        instance_epoch: int,
        owner: str,
        lease_seconds: int,
        *,
        now: datetime | None = None,
    ) -> IntentRecord | None:
        from datetime import timedelta

        if not owner or lease_seconds < 1 or lease_seconds > 86400:
            raise ValueError("invalid intent lease")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must have a timezone")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """SELECT * FROM intents WHERE run_id = ? AND task_id = ? AND instance_epoch = ?
                       AND status IN ('queued', 'claimed')
                       ORDER BY priority DESC, rowid""",
                    (run_id, task_id, instance_epoch),
                ).fetchall()
                selected = next(
                    (
                        item for row in rows
                        if (
                            (item := self._intent_from_row(row)).status == "queued"
                            or (item.lease_until is not None and self._parse_utc(item.lease_until) <= current)
                        ) and self._intent_ready(connection, item, current)
                    ), None,
                )
                if selected is None:
                    connection.commit()
                    return None
                lease_until = (current + timedelta(seconds=lease_seconds)).isoformat()
                fencing_token = selected.fencing_token + 1
                self._insert_event(
                    connection, uuid.uuid4().hex, run_id, task_id, instance_epoch,
                    "intent.claimed", {"intent_id": selected.intent_id, "owner": owner,
                                       "lease_until": lease_until, "fencing_token": fencing_token},
                    selected.intent_id,
                )
                connection.execute(
                    """UPDATE intents SET status = 'claimed', lease_owner = ?, lease_until = ?,
                       attempts = attempts + 1, fencing_token = ? WHERE intent_id = ?""",
                    (owner, lease_until, fencing_token, selected.intent_id),
                )
                connection.commit()
                return self.get_intent(selected.intent_id)
            except BaseException:
                connection.rollback()
                raise

    def heartbeat_intent(
        self, intent_id: str, owner: str, lease_seconds: int, *, fencing_token: int,
        now: datetime | None = None,
    ) -> IntentRecord:
        from datetime import timedelta

        if not owner or lease_seconds < 1 or lease_seconds > 86400 or fencing_token < 1:
            raise ValueError("invalid intent lease")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must have a timezone")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                intent = self._intent_from_row(row)
                if (
                    intent.status != "claimed" or intent.lease_owner != owner
                    or intent.fencing_token != fencing_token
                    or intent.lease_until is None or self._parse_utc(intent.lease_until) <= current
                ):
                    raise ValueError("intent lease is not owned or has expired")
                lease_until = (current + timedelta(seconds=lease_seconds)).isoformat()
                self._insert_event(
                    connection, uuid.uuid4().hex, intent.run_id, intent.task_id,
                    intent.instance_epoch, "intent.heartbeat",
                    {"intent_id": intent_id, "owner": owner, "lease_until": lease_until,
                     "fencing_token": fencing_token}, intent_id,
                )
                connection.execute(
                    "UPDATE intents SET lease_until = ? WHERE intent_id = ?", (lease_until, intent_id)
                )
                connection.commit()
                return self.get_intent(intent_id)
            except BaseException:
                connection.rollback()
                raise

    def release_intent(
        self, intent_id: str, owner: str, *, fencing_token: int,
        outcome: str = "queued", reason: str,
        now: datetime | None = None,
    ) -> IntentRecord:
        if (outcome not in ("queued", "completed", "cancelled") or not reason or not owner
                or fencing_token < 1):
            raise ValueError("invalid intent release")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must have a timezone")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                intent = self._intent_from_row(row)
                if (
                    intent.status != "claimed" or intent.lease_owner != owner
                    or intent.fencing_token != fencing_token
                    or intent.lease_until is None or self._parse_utc(intent.lease_until) <= current
                ):
                    raise ValueError("intent lease is not owned or has expired")
                event_id = uuid.uuid4().hex
                self._insert_event(
                    connection, event_id, intent.run_id, intent.task_id,
                    intent.instance_epoch, "intent.released",
                    {"intent_id": intent_id, "owner": owner, "fencing_token": fencing_token,
                     "outcome": outcome, "reason": reason},
                    intent_id,
                )
                connection.execute(
                    """UPDATE intents SET status = ?, lease_owner = NULL, lease_until = NULL,
                       completion_event_id = ? WHERE intent_id = ?""",
                    (outcome, event_id if outcome in ("completed", "cancelled") else None, intent_id),
                )
                connection.commit()
                return self.get_intent(intent_id)
            except BaseException:
                connection.rollback()
                raise

    def revisit_intent(self, intent_id: str, new_fact_id: str, reason: str) -> IntentRecord:
        """Requeue a finished direction when new evidence changes its premise."""
        if not reason:
            raise ValueError("revisit reason is required")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
                if row is None:
                    raise KeyError(intent_id)
                intent = self._intent_from_row(row)
                fact = connection.execute("SELECT * FROM facts WHERE fact_id = ?", (new_fact_id,)).fetchone()
                if (
                    intent.status not in ("completed", "cancelled") or fact is None
                    or fact["run_id"] != intent.run_id or fact["task_id"] != intent.task_id
                    or (fact["scope"] == "instance" and fact["instance_epoch"] != intent.instance_epoch)
                    or not fact["active"] or fact["kind"] == "hypothesis"
                    or fact["verification_status"] not in ("observed", "verified")
                ):
                    raise ValueError("revisit requires a completed intent and relevant new fact")
                completion = connection.execute(
                    "SELECT seq FROM events WHERE event_id = ?", (intent.completion_event_id,)
                ).fetchone()
                fact_event = connection.execute(
                    "SELECT seq FROM events WHERE event_id = ?", (new_fact_id,)
                ).fetchone()
                if completion is not None and fact_event["seq"] <= completion["seq"]:
                    raise ValueError("revisit requires new evidence after completion")
                self._insert_event(
                    connection, uuid.uuid4().hex, intent.run_id, intent.task_id,
                    intent.instance_epoch, "intent.revisited",
                    {"intent_id": intent_id, "new_fact_id": new_fact_id, "reason": reason},
                    new_fact_id,
                )
                connection.execute(
                    "UPDATE intents SET status = 'queued', completion_event_id = NULL WHERE intent_id = ?",
                    (intent_id,),
                )
                connection.commit()
                return self.get_intent(intent_id)
            except BaseException:
                connection.rollback()
                raise

    def rebuild_projection(self) -> int:
        """Rebuild fact, resource and intent views; return the fact count."""
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                events = connection.execute(
                    """SELECT event_id, run_id, task_id, instance_epoch, kind,
                              payload_json, evidence_sha256
                       FROM events ORDER BY seq"""
                ).fetchall()
                connection.execute("DELETE FROM facts")
                connection.execute("DELETE FROM facts_fts")
                connection.execute("DELETE FROM resources")
                connection.execute("DELETE FROM intents")
                count = 0
                for event in events:
                    payload = json.loads(event["payload_json"])
                    if event["kind"] in ("fact.recorded", "resource.recorded"):
                        ref = EvidenceRef(event["evidence_sha256"], payload["evidence_size"])
                        if not self.evidence.verify(ref):
                            raise ValueError(f"missing evidence for event {event['event_id']}")
                        if event["kind"] == "fact.recorded":
                            self._apply_fact_event(
                                connection, event["event_id"], event["run_id"], event["task_id"],
                                event["instance_epoch"], payload, event["evidence_sha256"],
                            )
                            count += 1
                        else:
                            self._apply_resource_event(
                                connection, event["event_id"], event["run_id"], event["task_id"],
                                event["instance_epoch"], payload, event["evidence_sha256"],
                            )
                    elif event["kind"] == "resource.revoked":
                        connection.execute(
                            "UPDATE resources SET active = 0 WHERE resource_id = ?",
                            (payload["resource_id"],),
                        )
                    elif event["kind"] == "intent.recorded":
                        self._apply_intent_event(
                            connection, event["event_id"], event["run_id"], event["task_id"],
                            event["instance_epoch"], payload,
                        )
                    elif event["kind"] == "intent.claimed":
                        connection.execute(
                            """UPDATE intents SET status = 'claimed', lease_owner = ?, lease_until = ?,
                               attempts = attempts + 1,
                               fencing_token = COALESCE(?, fencing_token + 1)
                               WHERE intent_id = ?""",
                            (payload["owner"], payload["lease_until"],
                             payload.get("fencing_token"), payload["intent_id"]),
                        )
                    elif event["kind"] == "intent.heartbeat":
                        connection.execute(
                            "UPDATE intents SET lease_until = ? WHERE intent_id = ?",
                            (payload["lease_until"], payload["intent_id"]),
                        )
                    elif event["kind"] == "intent.released":
                        connection.execute(
                            """UPDATE intents SET status = ?, lease_owner = NULL, lease_until = NULL,
                               completion_event_id = ? WHERE intent_id = ?""",
                            (
                                payload["outcome"],
                                event["event_id"] if payload["outcome"] in ("completed", "cancelled") else None,
                                payload["intent_id"],
                            ),
                        )
                    elif event["kind"] == "intent.revisited":
                        connection.execute(
                            "UPDATE intents SET status = 'queued', completion_event_id = NULL WHERE intent_id = ?",
                            (payload["intent_id"],),
                        )
                connection.commit()
                return count
            except BaseException:
                connection.rollback()
                raise
