"""CTFy competition player API adapter for multi-challenge runs.

The REST shapes below come from the public OpenAPI snapshot read on
2026-11-10. Competition image entrypoint, injected token scope, persistent
volume, and gateway limits remain ``contract_unverified`` until a real run.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from cerberus.contracts import Challenge, PlatformLimits, Question, SubmissionResult


class CompetitionContractError(ValueError):
    """The received player response cannot be safely interpreted."""


@dataclass(frozen=True)
class CompetitionSessionPaused(Exception):
    """The event has a published future session; the controller may resume then."""

    resume_at: str | None


@dataclass(frozen=True)
class CompetitionPlatformError(Exception):
    operation: str
    status_code: int
    code: str
    message: str
    retry_after_s: float | None = None


class CTFyCompetitionAdapter:
    contract_unverified = True

    def __init__(self, base_url: str, token: str, competition_id: str, *,
                 client: httpx.AsyncClient | None = None, page_size: int = 100,
                 instance_limit: int | None = None) -> None:
        if not base_url or not token or not competition_id:
            raise ValueError("base_url, token and competition_id are required")
        if not (1 <= page_size <= 10000):
            raise ValueError("page_size must be 1..10000")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.competition_id = competition_id
        self.page_size = page_size
        self.instance_limit = instance_limit
        self.client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._instances: dict[str, dict[str, Any]] = {}
        self._challenge_details: dict[str, dict[str, Any]] = {}
        self._competition: dict[str, Any] | None = None
        self._create_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _request(self, operation: str, method: str, path: str, *,
                       params: dict[str, Any] | None = None,
                       json_body: dict[str, Any] | None = None,
                       retry_reads: int = 2) -> Any:
        for attempt in range(retry_reads + 1):
            response = await self.client.request(
                method, self.base_url + "/api/v1" + path,
                params=params, json=json_body,
                headers={"Authorization": "Bearer " + self.token},
            )
            if method == "GET" and response.status_code in (429, 502, 503, 504) and attempt < retry_reads:
                raw_delay = response.headers.get("Retry-After", "")
                try:
                    delay = min(30.0, max(0.0, float(raw_delay)))
                except ValueError:
                    delay = min(8.0, 0.5 * 2**attempt)
                await asyncio.sleep(delay)
                continue
            break
        try:
            body = response.json()
        except ValueError as exc:
            raise CompetitionContractError(f"{operation} returned non-JSON response") from exc
        if response.is_error:
            detail = body.get("detail", body) if isinstance(body, dict) else {}
            if isinstance(detail, str):
                detail = {"message": detail}
            if not isinstance(detail, dict):
                detail = {}
            raw_delay = response.headers.get("Retry-After")
            try:
                retry_after = float(raw_delay) if raw_delay else None
            except ValueError:
                retry_after = None
            if response.status_code == 403 and detail.get("code") == "competition_paused":
                resume_at = detail.get("next_session_starts_at")
                if not isinstance(resume_at, str):
                    resume_at = body.get("next_session_starts_at") if isinstance(body, dict) else None
                raise CompetitionSessionPaused(resume_at if isinstance(resume_at, str) else None)
            raise CompetitionPlatformError(
                operation, response.status_code, str(detail.get("code", "unknown")),
                str(detail.get("message", "")), retry_after,
            )
        return body

    @property
    def _scoped_path(self) -> str:
        return "/competitions/" + quote(self.competition_id, safe="")

    async def get_run_limits(self) -> PlatformLimits:
        data = await self._request("competition", "GET", self._scoped_path)
        if not isinstance(data, dict) or data.get("id") != self.competition_id:
            raise CompetitionContractError("competition response does not match requested ID")
        self._competition = data
        cap = data.get("max_instances_per_team_effective")
        if not isinstance(cap, int) or cap < 1:
            cap = data.get("max_instances_per_team")
        if not isinstance(cap, int) or cap < 1:
            cap = self.instance_limit
        if self.instance_limit is not None and cap is not None:
            cap = min(cap, self.instance_limit)
        end = data.get("current_session_ends_at") or data.get("ends_at")
        deadline = None
        if isinstance(end, str) and end:
            try:
                deadline = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                deadline = None
        return PlatformLimits(max_instances=cap, deadline=deadline,
                              authoring_allowed=bool(data.get("authoring_enabled", False)))

    def _require_open_session(self) -> None:
        data = self._competition or {}
        sessions = data.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            return  # No session contract in this response; the server remains authoritative.
        now = datetime.now(UTC)
        upcoming: list[str] = []
        for session in sessions:
            if not isinstance(session, dict):
                raise CompetitionContractError("competition session is not an object")
            try:
                start = datetime.fromisoformat(str(session["starts_at"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(session["ends_at"]).replace("Z", "+00:00"))
            except (KeyError, ValueError) as exc:
                raise CompetitionContractError("competition session has invalid timestamps") from exc
            if start.tzinfo is None or end.tzinfo is None or start >= end:
                raise CompetitionContractError("competition session timestamps are invalid")
            if start <= now < end:
                return
            if now < start:
                upcoming.append(start.isoformat())
        raise CompetitionSessionPaused(min(upcoming) if upcoming else None)

    async def _paged(self, operation: str, path: str, *,
                     extra_params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, Any] = {"limit": self.page_size, "offset": offset}
            params.update(extra_params or {})
            page = await self._request(operation, "GET", path, params=params)
            if not isinstance(page, dict) or not isinstance(page.get("items"), list):
                raise CompetitionContractError(f"{operation} returned invalid page")
            batch = page["items"]
            if any(not isinstance(item, dict) for item in batch):
                raise CompetitionContractError(f"{operation} page has non-object item")
            items.extend(batch)
            total = page.get("total")
            if not isinstance(total, int) or total < 0:
                raise CompetitionContractError(f"{operation} page lacks total")
            if len(items) >= total:
                break
            if not batch:
                raise CompetitionContractError(f"{operation} pagination stopped before total")
            offset += len(batch)
            if offset > 100000:
                raise CompetitionContractError(f"{operation} pagination exceeded safety bound")
        return items

    async def _list_instances(self) -> list[dict[str, Any]]:
        return await self._paged("instances", "/instances", extra_params={"competition_id": self.competition_id})

    def _remember_instance(self, instance: dict[str, Any]) -> None:
        task_id = instance.get("challenge_id")
        instance_id = instance.get("id")
        if not isinstance(task_id, str) or not isinstance(instance_id, str):
            raise CompetitionContractError("instance lacks challenge and instance IDs")
        self._instances[task_id] = instance

    @staticmethod
    def _instance_epoch(instance: dict[str, Any]) -> int:
        """Use the remote generation identity, so a process restart cannot reset it.

        The platform does not expose an integer generation. Instance ID and
        started_at identify the lease in the public InstanceInfo contract.
        Zero is reserved for tasks without an active instance.
        """
        instance_id = instance.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            return 0
        identity = f"{instance_id}\x00{instance.get('started_at')}".encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1) or 1

    @staticmethod
    def _question(q: dict[str, Any], solved: set[str]) -> Question:
        question_id = q.get("id")
        if not isinstance(question_id, str):
            raise CompetitionContractError("question has no ID")
        if q.get("answered_correctly") or question_id in solved:
            status = "accepted"
        elif q.get("unlocked") is False or (
            isinstance(q.get("requires"), list)
            and any(required not in solved for required in q["requires"])
        ):
            status = "locked"
        else:
            status = "open"
        attempts_remaining = q.get("attempts_remaining")
        attempts_cap = q.get("attempts_cap")
        used = (attempts_cap - attempts_remaining
                if isinstance(attempts_cap, int) and isinstance(attempts_remaining, int) else None)
        return Question(id=question_id, prompt=str(q.get("prompt") or ""), status=status,
                        max_attempts=attempts_cap if isinstance(attempts_cap, int) else None,
                        attempts_used=used)

    def _challenge(self, row: dict[str, Any]) -> Challenge:
        task_id = row.get("id")
        if not isinstance(task_id, str):
            raise CompetitionContractError("challenge has no ID")
        instance = self._instances.get(task_id, {})
        questions = instance.get("questions") if instance.get("status") == "ready" else row.get("questions", [])
        if not isinstance(questions, list):
            raise CompetitionContractError("challenge questions are not an array")
        solved = set(row.get("my_team_solved_question_ids") or row.get("my_solved_question_ids") or [])
        attachments = (instance.get("attachments", row.get("attachments", []))
                       if instance.get("status") == "ready" else row.get("attachments", []))
        if not isinstance(attachments, list):
            raise CompetitionContractError("challenge attachments are not an array")
        names = [str(x["name"]) for x in attachments if isinstance(x, dict) and isinstance(x.get("name"), str)]
        target = {"instance_id": instance.get("id"), "instance_status": instance.get("status", "stopped"),
                  "entry_urls": instance.get("entry_urls", []), "services": instance.get("services", []),
                  "questions_scoped_to_instance": instance.get("status") == "ready",
                  "scoring": row.get("scoring", "scored"), "is_qa_only": bool(row.get("is_qa_only"))}
        return Challenge(task_id=task_id, title=str(row.get("name") or task_id),
                         description=str(instance.get("description") or row.get("description") or ""),
                         category=str(row.get("category") or "unknown"),
                         questions=[self._question(q, solved) for q in questions], attachments=names,
                         target=target, instance_epoch=self._instance_epoch(instance),
                         available=not bool(row.get("my_team_fully_solved") or row.get("my_fully_solved")
                                            or row.get("closed")))

    async def discover_tasks(self) -> list[Challenge]:
        rows = await self._paged("challenges", self._scoped_path + "/challenges")
        # Refresh all own instances so a reset/replacement changes instance_epoch.
        instances = await self._list_instances()
        visible_active: set[str] = set()
        for instance in instances:
            if instance.get("competition_id") not in (None, "", self.competition_id):
                raise CompetitionContractError("instance escaped competition scope")
            if instance.get("status") in ("ready", "starting"):
                self._remember_instance(instance)
                visible_active.add(str(instance["challenge_id"]))
        for task_id in set(self._instances) - visible_active:
            self._instances.pop(task_id)
        self._challenge_details = {str(row["id"]): row for row in rows if isinstance(row.get("id"), str)}
        return [self._challenge(row) for row in rows]

    async def refresh_progress(self) -> list[Challenge]:
        return await self.discover_tasks()

    async def ensure_instance(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._challenge_details:
            await self.discover_tasks()
        challenge = self._challenge_details.get(task_id)
        if challenge is None:
            raise CompetitionContractError("task is outside current competition")
        if challenge.get("is_qa_only"):
            return {"status": "not_required"}
        async with self._create_lock:
            current_instances = await self._list_instances()
            for existing in current_instances:
                if existing.get("competition_id") not in (None, "", self.competition_id):
                    continue
                if existing.get("challenge_id") == task_id and existing.get("status") in ("ready", "starting"):
                    self._remember_instance(existing)
                    return existing
            limits = await self.get_run_limits()
            self._require_open_session()
            if limits.max_instances is None:
                raise CompetitionContractError("instance capacity is unknown; explicit limit required")
            active = [x for x in current_instances if x.get("status") in ("ready", "starting")]
            if limits.max_instances is not None and len(active) >= limits.max_instances:
                raise CompetitionContractError("instance capacity reached")
            # A transport timeout after POST is uncertain. Caller reconciles with
            # the read-only list before any later create attempt.
            started = await self._request("start_instance", "POST", "/instances",
                                          json_body={"challenge_id": task_id,
                                                     "competition_id": self.competition_id})
            if not isinstance(started, dict) or not isinstance(started.get("id"), str):
                raise CompetitionContractError("start response has no instance ID")
            current = await self._request("get_instance", "GET",
                                          "/instances/" + quote(started["id"], safe=""),
                                          params={"competition_id": self.competition_id})
            if not isinstance(current, dict) or current.get("challenge_id") != task_id:
                raise CompetitionContractError("started instance does not match challenge")
            self._remember_instance(current)
            return current

    async def release_instance(self, task_id: str) -> dict[str, Any]:
        instance = self._instances.get(task_id)
        if instance is None or not isinstance(instance.get("id"), str):
            return {"status": "already_absent"}
        result = await self._request("stop_instance", "DELETE",
                                     self._scoped_path + "/instances/" + quote(instance["id"], safe=""))
        if not isinstance(result, dict):
            raise CompetitionContractError("stop response is not an object")
        self._instances.pop(task_id, None)
        return result

    async def submit_answer(self, task_id: str, question_id: str, answer: str) -> SubmissionResult:
        if not isinstance(answer, str):
            raise TypeError("answer must be exact text")
        if task_id not in self._challenge_details:
            await self.discover_tasks()
        challenge = self._challenge_details.get(task_id)
        if challenge is None:
            raise CompetitionContractError("task is outside current competition")
        solved = set(challenge.get("my_team_solved_question_ids")
                     or challenge.get("my_solved_question_ids") or [])
        visible = {q["id"] for q in challenge.get("questions", [])
                   if isinstance(q, dict) and self._question(q, solved).status == "open"}
        instance = self._instances.get(task_id)
        if instance and isinstance(instance.get("questions"), list):
            # Instance projection is authoritative for dynamic gating.
            visible = {q.get("id") for q in instance["questions"]
                       if isinstance(q, dict) and self._question(q, solved).status == "open"}
        if question_id not in visible:
            raise CompetitionContractError("question is not open in this task")
        await self.get_run_limits()
        self._require_open_session()
        if instance is not None:
            instance_questions = {q.get("id"): q for q in instance.get("questions", [])
                                  if isinstance(q, dict)}
            current_question = instance_questions.get(question_id)
            if current_question is not None:
                if current_question.get("answered_correctly"):
                    return SubmissionResult(task_id=task_id, question_id=question_id,
                                            status="duplicate", raw={"reason": "already_accepted"})
                if current_question.get("attempts_remaining") == 0:
                    raise CompetitionContractError("question has no attempts remaining")
        if challenge.get("is_qa_only"):
            path = "/qa/submissions"
            body = {"challenge_id": task_id, "competition_id": self.competition_id,
                    "question_id": question_id, "answer": answer}
        else:
            if instance is None or not isinstance(instance.get("id"), str):
                raise CompetitionContractError("no active instance for submission")
            path = "/submissions"
            body = {"instance_id": instance["id"], "question_id": question_id, "answer": answer}
        result = await self._request("submit_answer", "POST", path, json_body=body)
        if not isinstance(result, dict) or not isinstance(result.get("correct"), bool):
            raise CompetitionContractError("submission response lacks platform verdict")
        if result.get("question_id") not in (None, question_id):
            raise CompetitionContractError("submission response question mismatch")
        return SubmissionResult(task_id=task_id, question_id=question_id,
                                status="accepted" if result["correct"] else "rejected", raw=result)

    async def get_hint(self, task_id: str, *, authorized: bool = False) -> dict[str, Any]:
        # This call lists metadata; viewing a tier is the cost-bearing operation.
        if task_id not in self._challenge_details:
            raise CompetitionContractError("task is outside current competition")
        result = await self._request("list_hints", "GET", self._scoped_path + "/challenges/"
                                     + quote(task_id, safe="") + "/hints")
        if not isinstance(result, dict) or not isinstance(result.get("tiers"), list):
            raise CompetitionContractError("hint list has invalid shape")
        return result

    async def view_hint(self, task_id: str, hint_id: str, *, authorized: bool = False) -> dict[str, Any]:
        if not authorized:
            raise PermissionError("viewing hints may cost points; explicit policy required")
        listing = await self.get_hint(task_id)
        ids = {tier.get("id") for tier in listing["tiers"] if isinstance(tier, dict)}
        if hint_id not in ids:
            raise CompetitionContractError("hint is not listed for this task")
        result = await self._request("view_hint", "POST",
                                     self._scoped_path + "/challenges/" + quote(task_id, safe="")
                                     + "/hints/" + quote(hint_id, safe="") + "/view")
        if not isinstance(result, dict):
            raise CompetitionContractError("view hint result is not an object")
        return result
