"""TSecBench Challenges API adapter. API v1 is independent of corpus v1/v2."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cerberus.contracts import Challenge, PlatformLimits, Question, SubmissionResult


@dataclass(frozen=True)
class TSecError(Exception):
    operation: str
    status_code: int
    code: str
    message: str
    retry_after_s: float | None = None

    @property
    def task_ended(self) -> bool:
        if self.code != "invalid_state":
            return False
        words = self.message.lower()
        if self.instance_limit or self.hint_after_completion:
            return False
        return any(word in words for word in ("ended", "expired", "timeout", "stopped", "已结束", "已过期", "超时"))

    @property
    def hint_after_completion(self) -> bool:
        return self.operation == "hint" and self.code == "invalid_state" and any(
            word in self.message.lower() for word in ("completed", "solved", "通关", "已完成"))

    @property
    def instance_limit(self) -> bool:
        return self.operation == "start" and self.code == "invalid_state" and any(
            word in self.message.lower() for word in ("max active", "limit", "上限", "最大"))


class TSecBenchAdapter:
    """Authorized per-task API only; no corpus-specific answer logic."""

    def __init__(self, base_url: str, token: str, *, corpus_version: str,
                 client: httpx.AsyncClient | None = None,
                 instance_ready_timeout_s: float = 30.0) -> None:
        if not base_url or not token or not corpus_version:
            raise ValueError("base_url, token and corpus_version are required")
        if instance_ready_timeout_s <= 0:
            raise ValueError("instance_ready_timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.corpus_version = corpus_version
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(timeout=30.0)
        self.token = token
        self.instance_ready_timeout_s = instance_ready_timeout_s
        self._last_instances: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._instance_epochs: dict[str, int] = {}
        self._missing_instances: set[str] = set()
        self._state_path: Path | None = None
        self._new_task_epoch_floor: int | None = None
        self._confirmed_indices: dict[str, set[int]] = {}
        self._submitted_answers: set[tuple[str, str]] = set()

    def bind_run_state(self, path: str | Path, *, resume: bool) -> None:
        """Keep instance generations distinct across controller processes.

        The list API supplies no stable instance ID. A resumed process cannot
        prove that an instance stayed live while it was down, so every known
        generation advances once before the first remote read.
        """
        if self._state_path is not None:
            raise ValueError("TSecBench run state is already bound")
        state_path = Path(path)
        if resume and state_path.is_file():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("unreadable TSecBench instance checkpoint") from exc
            if not isinstance(saved, dict) or saved.get("schema_version") != 1:
                raise ValueError("invalid TSecBench instance checkpoint")
            instances = saved.get("instances")
            if not isinstance(instances, dict):
                raise ValueError("invalid TSecBench instance checkpoint")
            floor = saved.get("new_task_epoch_floor")
            if floor is not None and (type(floor) is not int or floor < 0):
                raise ValueError("invalid TSecBench instance checkpoint")
            self._new_task_epoch_floor = max(time.time_ns(), (floor or 0) + 1)
            for task_id, row in instances.items():
                if (not isinstance(task_id, str) or not task_id
                        or not isinstance(row, dict) or type(row.get("epoch")) is not int
                        or row["epoch"] < 0 or row.get("status") not in
                        ("stopped", "stop_pending", "pending", "available")
                        or not isinstance(row.get("addresses"), list)
                        or any(not isinstance(address, str) for address in row["addresses"])):
                    raise ValueError("invalid TSecBench instance checkpoint")
                self._instance_epochs[task_id] = row["epoch"] + 1
                self._last_instances[task_id] = (row["status"], tuple(row["addresses"]))
        elif resume:
            # Runs created before this checkpoint existed may still have old
            # answers or histories at a small local epoch. Use a fresh high
            # floor for first observations rather than silently reusing zero.
            self._new_task_epoch_floor = time.time_ns()
        self._state_path = state_path
        self._save_instance_state()

    def _save_instance_state(self) -> None:
        if self._state_path is None:
            return
        instances = {task_id: {"epoch": self._instance_epochs.get(task_id, 0),
                               "status": status, "addresses": list(addresses)}
                     for task_id, (status, addresses) in self._last_instances.items()}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._state_path.parent,
                                         prefix=".tsec-instance-", delete=False) as output:
            temporary = Path(output.name)
            json.dump({"schema_version": 1, "instances": instances,
                       "new_task_epoch_floor": self._new_task_epoch_floor}, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.replace(temporary, self._state_path)
        finally:
            temporary.unlink(missing_ok=True)

    async def close(self) -> None:
        if self._own_client:
            await self.client.aclose()

    async def _request(self, operation: str, method: str, path: str, *, params: dict[str, str] | None = None,
                       json_body: dict[str, Any] | None = None, retries: int = 3) -> Any:
        for attempt in range(retries + 1):
            try:
                response = await self.client.request(
                    method, self.base_url + path, params=params, json=json_body,
                    headers={"BENCHMARK_TOKEN": self.token},
                )
            except (httpx.TimeoutException, httpx.TransportError):
                # Read-only calls can be retried; a mutation may already have
                # succeeded remotely and must be reconciled by the controller.
                if method != "GET" or attempt >= retries:
                    raise
                await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
                continue
            if response.status_code in (429, 502, 503, 504) and method == "GET" and attempt < retries:
                raw_wait = response.headers.get("Retry-After", "")
                try:
                    wait_s = min(30.0, max(0.0, float(raw_wait)))
                except ValueError:
                    wait_s = min(8.0, 0.5 * 2**attempt)
                await asyncio.sleep(wait_s)
                continue
            break
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.is_error:
            detail = body.get("detail", body) if isinstance(body, dict) else {}
            if not isinstance(detail, dict):
                detail = {"message": str(detail)}
            retry = response.headers.get("Retry-After")
            try:
                retry_after = float(retry) if retry else None
            except ValueError:
                retry_after = None
            raise TSecError(operation, response.status_code,
                            str(detail.get("code", body.get("code", "unknown") if isinstance(body, dict) else "unknown")),
                            str(detail.get("message", body.get("message", "") if isinstance(body, dict) else "")),
                            retry_after)
        return body

    async def discover_tasks(self) -> list[Challenge]:
        rows = await self._request("list", "GET", "/openapi/v1/challenges")
        # The documented API returns an array. Sapper v2's arena client also
        # accepts a ``{"challenges": [...]}`` envelope observed in compatible
        # deployments; retain strict entry validation below.
        if isinstance(rows, dict) and isinstance(rows.get("challenges"), list):
            rows = rows["challenges"]
        if not isinstance(rows, list):
            raise ValueError("TSecBench list response must be an array")
        challenges: list[Challenge] = []
        seen_tasks: set[str] = set()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("unique_code"), str)
                    or not row["unique_code"]):
                raise ValueError("invalid TSecBench challenge entry")
            count = row.get("flag_count", 0)
            done = row.get("correct_flag_count", 0)
            if (type(count) is not int or type(done) is not int
                    or count < 0 or not 0 <= done <= count):
                raise ValueError("invalid TSecBench flag progress")
            completed = row.get("is_completed", False)
            if not isinstance(completed, bool) or (completed and done != count):
                raise ValueError("inconsistent TSecBench completion status")
            status = row.get("container_status", "stopped")
            if status not in ("stopped", "stop_pending", "pending", "available"):
                raise ValueError("unknown TSecBench container status")
            addresses = row.get("container_addr") or []
            if not isinstance(addresses, list) or any(not isinstance(x, str) for x in addresses):
                raise ValueError("invalid TSecBench container addresses")
            if status != "available" and addresses:
                raise ValueError("TSecBench addresses require an available container")
            task_id = row["unique_code"]
            seen_tasks.add(task_id)
            self._observe_instance(task_id, status, addresses, persist=False)
            known = self._confirmed_indices.setdefault(task_id, set())
            if completed:
                known.update(range(count))
            elif len(known) > done:
                # A task reset or an inconsistent view invalidates local
                # index knowledge. The list API supplies only an aggregate.
                known.clear()
            challenges.append(Challenge(
                task_id=task_id, title=task_id,
                description=str(row.get("description") or ""), category=str(row.get("category") or "unknown"),
                questions=[Question(id=f"flag:{i}", status="accepted" if i in known else "unknown")
                           for i in range(count)],
                target={"addresses": addresses,
                        "container_status": status,
                        "flag_count": count, "correct_flag_count": done,
                        "corpus_version": self.corpus_version,
                        "difficulty": row.get("difficulty"), "level": row.get("level")},
                instance_epoch=self._instance_epochs.get(task_id, 0),
                score=float(row["total_score"]) if row.get("total_score") is not None else None,
                available=not completed,
            ))
        self._missing_instances.update(self._last_instances.keys() - seen_tasks)
        self._save_instance_state()
        return challenges

    def _observe_instance(self, task_id: str, status: str, addresses: list[str], *, persist: bool = True) -> None:
        current = (status, tuple(addresses))
        prior = self._last_instances.get(task_id)
        if prior is None and self._new_task_epoch_floor is not None:
            self._instance_epochs[task_id] = self._new_task_epoch_floor
        if prior is not None:
            restarted = (task_id in self._missing_instances
                         or (prior[0] in ("stopped", "stop_pending") and status in ("pending", "available"))
                         or (prior[0] == "available" and status == "pending"))
            changed_address = (prior[0] == status == "available"
                               and prior[1] and current[1] and prior[1] != current[1])
            if restarted or changed_address:
                self._instance_epochs[task_id] = self._instance_epochs.get(task_id, 0) + 1
        self._missing_instances.discard(task_id)
        self._last_instances[task_id] = current
        if persist:
            self._save_instance_state()

    async def refresh_progress(self) -> list[Challenge]:
        return await self.discover_tasks()

    async def ensure_instance(self, task_id: str) -> dict[str, Any]:
        # First inspect state to avoid duplicate starts after a prior timeout.
        current = next((x for x in await self.discover_tasks() if x.task_id == task_id), None)
        if current is None:
            raise ValueError("task not found in current run")
        status = current.target.get("container_status")
        if status == "available":
            return current.target
        if status == "stop_pending":
            raise TSecError("start", 409, "resource_unavailable", "container is stopping")
        if status == "stopped":
            started = await self._request("start", "POST", "/openapi/v1/challenges/start",
                                          params={"unique_code": task_id})
            if not isinstance(started, dict) or started.get("unique_code") != task_id:
                raise ValueError("TSecBench start response does not match challenge")
            addresses = started.get("container_addr")
            if isinstance(addresses, list) and addresses and all(isinstance(x, str) for x in addresses):
                self._observe_instance(task_id, "available", addresses)
                return started
        deadline = time.monotonic() + self.instance_ready_timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            current = next((x for x in await self.discover_tasks() if x.task_id == task_id), None)
            if current is None:
                raise ValueError("task disappeared while instance was starting")
            if current.target["container_status"] == "available":
                return current.target
            if current.target["container_status"] in ("stopped", "stop_pending"):
                raise TSecError("start", 409, "resource_unavailable", "container did not become available")
        raise TSecError("start", 503, "resource_unavailable", "container readiness timed out")

    async def release_instance(self, task_id: str) -> dict[str, Any]:
        return await self._request("close", "POST", "/openapi/v1/challenges/close",
                                   params={"unique_code": task_id})

    async def submit_answer(self, task_id: str, question_id: str, answer: str) -> SubmissionResult:
        if not (1 <= len(answer) <= 4096):
            raise ValueError("flag length must be 1..4096")
        identity = (task_id, answer)
        if identity in self._submitted_answers:
            return SubmissionResult(task_id=task_id, question_id=question_id,
                                    status="duplicate", raw={"code": "already_submitted_locally"})
        try:
            body = await self._request("submit", "POST", "/openapi/v1/challenges/submit",
                                       json_body={"unique_code": task_id, "flag": answer})
        except TSecError as exc:
            if exc.code == "duplicate":
                self._submitted_answers.add(identity)
                return SubmissionResult(task_id=task_id, question_id=question_id, status="duplicate",
                                        raw={"code": "duplicate", "message": exc.message})
            raise
        self._submitted_answers.add(identity)
        if not isinstance(body, dict) or type(body.get("correct")) is not bool:
            raise ValueError("invalid TSecBench submission response")
        matched = body.get("matched_flag_index")
        total = body.get("total_flag_count")
        done = body.get("correct_flag_count")
        if (type(total) is not int or type(done) is not int or total < 0 or not 0 <= done <= total
                or (body["correct"] and (type(matched) is not int or not 0 <= matched < total))
                or (not body["correct"] and matched is not None)):
            raise ValueError("invalid TSecBench submission progress")
        if body["correct"]:
            self._confirmed_indices.setdefault(task_id, set()).add(matched)
        return SubmissionResult(task_id=task_id,
                                question_id=f"flag:{matched}" if body["correct"] else question_id,
                                status="accepted" if body["correct"] else "rejected", raw=body)

    async def get_hint(self, task_id: str, *, authorized: bool = False) -> str | None:
        if not authorized:
            raise PermissionError("hints may cost points; explicit policy required")
        body = await self._request("hint", "GET", "/openapi/v1/challenges/hint",
                                   params={"unique_code": task_id})
        return body.get("hint") if isinstance(body, dict) else None

    async def get_run_limits(self) -> PlatformLimits:
        # The current public guide says three active instances, but the task API
        # does not report a stable limit. Do not treat that snapshot as a contract.
        return PlatformLimits(max_instances=None)
