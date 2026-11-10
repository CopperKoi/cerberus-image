"""CTFY-0001 v3 hosted run adapter.

The hosted token is scoped to the one challenge in task.json. The platform
starts the target before this process runs; this adapter never starts one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from cerberus.contracts import Challenge, PlatformLimits, Question, SubmissionResult


class HostedContractError(ValueError):
    """The task or platform response cannot be safely interpreted."""


@dataclass(frozen=True)
class HostedPlatformError(Exception):
    operation: str
    status_code: int
    code: str
    message: str
    retry_after_s: float | None = None


@dataclass(frozen=True)
class HostedAttachment:
    name: str
    size: int | None
    path: Path | None


@dataclass(frozen=True)
class HostedTask:
    work_dir: Path
    challenge_id: str
    title: str
    description: str
    category: str
    run_mode: str
    competition_id: str
    questions: tuple[dict[str, Any], ...]
    gated_questions: int
    target_instance: str | None
    target_endpoints: tuple[dict[str, Any], ...]
    attachments: tuple[HostedAttachment, ...]
    readme: str
    platform_url: str
    max_turns: int | None
    dry_run: bool
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_work_dir(cls, work_dir: str | Path, *, environ: Mapping[str, str] | None = None) -> HostedTask:
        work = Path(work_dir).resolve()
        data = json.loads((work / "task.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 3:
            raise HostedContractError("task.json schema_version must be CTFY-0001 v3")
        result_contract = data.get("result")
        if isinstance(result_contract, dict) and result_contract.get("path") not in (None, str(work / "result.json")):
            raise HostedContractError("result.path conflicts with CTFY-0001 v3 work directory")
        challenge = data.get("challenge")
        if not isinstance(challenge, dict) or not isinstance(challenge.get("id"), str) or not challenge["id"]:
            raise HostedContractError("task.json.challenge.id is required")
        mode = data.get("run_mode")
        if mode not in {"instance", "qa", "defence", "awd", "engagement"}:
            raise HostedContractError("unsupported hosted run_mode")
        target_instance = data.get("target_instance")
        if target_instance is not None and (not isinstance(target_instance, str) or not target_instance):
            raise HostedContractError("target_instance must be a nonempty string or null")
        env = environ if environ is not None else os.environ
        dry_run = bool(data.get("dry_run")) or env.get("CTFY_EVAL_DRY_RUN") == "1"
        flavor = env.get("LLM_FLAVOR")
        if not dry_run and flavor is not None and flavor != "openai":
            raise HostedContractError("hosted LLM_FLAVOR must be openai for Chat Completions")
        if not dry_run and mode in {"instance", "defence", "awd", "engagement"} and not target_instance:
            raise HostedContractError("target_instance is required for a hosted target")
        if mode == "qa" and target_instance is not None:
            raise HostedContractError("qa task must not have target_instance")
        env_target = env.get("CTFY_EVAL_TARGET_INSTANCE")
        if env_target and env_target != target_instance:
            raise HostedContractError("target_instance conflicts with CTFY_EVAL_TARGET_INSTANCE")
        env_challenge = env.get("CTFY_EVAL_CHALLENGE_ID")
        if env_challenge and env_challenge != challenge["id"]:
            raise HostedContractError("challenge.id conflicts with CTFY_EVAL_CHALLENGE_ID")
        if env.get("CTFY_EVAL_COMPETITION_ID") and env["CTFY_EVAL_COMPETITION_ID"] != data.get("competition_id", ""):
            raise HostedContractError("competition_id conflicts with CTFY_EVAL_COMPETITION_ID")
        questions = data.get("questions", [])
        if not isinstance(questions, list) or any(not isinstance(q, dict) or not isinstance(q.get("id"), str)
                                                  or not q["id"] for q in questions):
            raise HostedContractError("questions must be objects with id")
        question_ids = [q["id"] for q in questions]
        if len(set(question_ids)) != len(question_ids):
            raise HostedContractError("questions contain duplicate IDs")
        gated = data.get("gated_questions", 0)
        if not isinstance(gated, int) or gated < 0:
            raise HostedContractError("gated_questions must be nonnegative")
        endpoints = data.get("target_endpoints", [])
        if not isinstance(endpoints, list) or any(not isinstance(x, dict) for x in endpoints):
            raise HostedContractError("target_endpoints must be an array")
        max_turns = data.get("max_turns")
        if max_turns is not None and (not isinstance(max_turns, int) or max_turns <= 0):
            raise HostedContractError("max_turns must be positive or null")
        env_turns = env.get("CTFY_EVAL_MAX_TURNS")
        if env_turns is not None:
            try:
                parsed_turns = int(env_turns)
            except ValueError as exc:
                raise HostedContractError("CTFY_EVAL_MAX_TURNS must be positive") from exc
            if parsed_turns <= 0 or (max_turns is not None and parsed_turns != max_turns):
                raise HostedContractError("CTFY_EVAL_MAX_TURNS conflicts with task.json")
            max_turns = parsed_turns
        listed = data.get("attachments", [])
        if not isinstance(listed, list):
            raise HostedContractError("attachments must be an array")
        attachments: list[HostedAttachment] = []
        root = (work / "attachments").resolve()
        for item in listed:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise HostedContractError("attachment name is required")
            name = item["name"]
            if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
                raise HostedContractError("attachment name must be a simple filename")
            declared_path = item.get("path")
            local: Path | None = None
            if declared_path is not None:
                if not isinstance(declared_path, str):
                    raise HostedContractError("attachment path must be a string or null")
                candidate = Path(declared_path)
                if not candidate.is_absolute():
                    candidate = work / candidate
                local = candidate.resolve()
                if local.parent != root or local.name != name or not local.is_file():
                    raise HostedContractError("local attachment is missing or outside work directory")
            size = item.get("size")
            if size is not None and (not isinstance(size, int) or size < 0):
                raise HostedContractError("attachment size must be nonnegative or null")
            attachments.append(HostedAttachment(name=name, size=size, path=local))
        platform_url = str(data.get("platform_url") or env.get("CTFY_PLATFORM_URL") or "")
        env_url = env.get("CTFY_PLATFORM_URL")
        if env_url and platform_url.rstrip("/") != env_url.rstrip("/"):
            raise HostedContractError("platform_url conflicts with CTFY_PLATFORM_URL")
        return cls(
            work_dir=work, challenge_id=challenge["id"],
            title=str(challenge.get("name") or challenge["id"]),
            description=str(challenge.get("description") or ""),
            category=str(challenge.get("category") or "unknown"),
            run_mode=mode, competition_id=str(data.get("competition_id") or ""),
            questions=tuple(questions), gated_questions=gated,
            target_instance=target_instance, target_endpoints=tuple(endpoints),
            attachments=tuple(attachments), readme=(work / "README.md").read_text(encoding="utf-8"),
            platform_url=platform_url.rstrip("/"), max_turns=max_turns,
            dry_run=dry_run,
            raw=data,
        )


class CTFyHostedAdapter:
    """One challenge adapter; durable answer deduplication belongs to the ledger."""

    def __init__(self, task: HostedTask, token: str | None, *, client: httpx.AsyncClient | None = None) -> None:
        if task.run_mode not in {"instance", "qa"} and not task.dry_run:
            raise HostedContractError(f"hosted run_mode {task.run_mode!r} is not implemented")
        self.task = task
        self.token = token or ""
        self.client = client or httpx.AsyncClient(timeout=30.0)
        self._own_client = client is None
        self.questions = list(task.questions)
        self.confirmed: set[str] = set()
        self.attempted: set[tuple[str, str]] = set()
        self.fully_solved = False

    async def close(self) -> None:
        if self._own_client:
            await self.client.aclose()

    async def _request(self, operation: str, method: str, path: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.task.dry_run or not self.task.platform_url or not self.token:
            raise HostedContractError("hosted platform access requires task URL and injected CTFY_TOKEN")
        response = await self.client.request(
            method, self.task.platform_url + "/api/v1" + path,
            json=body, headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            result = response.json()
        except ValueError as exc:
            raise HostedContractError(f"{operation} returned non-JSON response") from exc
        if response.is_error:
            detail = result.get("detail", result) if isinstance(result, dict) else {}
            if isinstance(detail, str):
                detail = {"message": detail}
            if not isinstance(detail, dict):
                detail = {}
            retry = response.headers.get("Retry-After")
            try:
                retry_after = float(retry) if retry else None
            except ValueError:
                retry_after = None
            raise HostedPlatformError(
                operation, response.status_code, str(detail.get("code", "unknown")),
                str(detail.get("message", "")), retry_after,
            )
        if not isinstance(result, dict):
            raise HostedContractError(f"{operation} returned non-object JSON")
        return result

    async def discover_tasks(self) -> list[Challenge]:
        epoch = 0
        if self.task.target_instance:
            epoch = int.from_bytes(hashlib.sha256(self.task.target_instance.encode()).digest()[:8], "big") & ((1 << 63) - 1) or 1
        return [Challenge(
            task_id=self.task.challenge_id, title=self.task.title,
            description=self.task.description, category=self.task.category,
            questions=[self._question(q) for q in self.questions],
            attachments=[str(x.path) if x.path else x.name for x in self.task.attachments],
            target={"instance_id": self.task.target_instance, "endpoints": list(self.task.target_endpoints),
                    "gated_questions": self.task.gated_questions, "run_mode": self.task.run_mode},
            instance_epoch=epoch,
            available=not self.fully_solved,
        )]

    def _question(self, raw: dict[str, Any]) -> Question:
        cap = raw.get("attempts_cap")
        remaining = raw.get("attempts_remaining")
        if type(cap) is not int or cap < 0:
            cap = None
        if type(remaining) is not int or remaining < 0:
            remaining = None
        used = cap - remaining if cap is not None and remaining is not None and remaining <= cap else None
        return Question(id=raw["id"], prompt=str(raw.get("prompt") or ""),
                        status="accepted" if raw["id"] in self.confirmed else "open",
                        max_attempts=cap, attempts_used=used)

    async def refresh_questions(self) -> Challenge:
        if self.task.dry_run:
            return (await self.discover_tasks())[0]
        result = await self._request("refresh_questions", "GET", "/challenges/" + quote(self.task.challenge_id, safe="")
                                     + "?competition_id=" + quote(self.task.competition_id, safe=""))
        if result.get("id") != self.task.challenge_id:
            raise HostedContractError("challenge refresh returned a different challenge")
        questions = result.get("questions")
        if not isinstance(questions, list) or any(not isinstance(q, dict) or not isinstance(q.get("id"), str)
                                                  or not q["id"] for q in questions):
            raise HostedContractError("challenge refresh has invalid questions")
        ids = [q["id"] for q in questions]
        if len(set(ids)) != len(ids):
            raise HostedContractError("challenge refresh has duplicate question IDs")
        self.questions = questions
        self.confirmed.update(str(x) for x in result.get("my_solved_question_ids", []) if isinstance(x, str))
        self.fully_solved = self.fully_solved or bool(result.get("my_fully_solved", False))
        return (await self.discover_tasks())[0]

    async def refresh_progress(self) -> list[Challenge]:
        return [await self.refresh_questions()]

    async def get_instance(self) -> dict[str, Any]:
        if not self.task.target_instance:
            raise HostedContractError("this hosted task has no target instance")
        result = await self._request("get_instance", "GET", "/instances/" + quote(self.task.target_instance, safe="")
                                     + "?competition_id=" + quote(self.task.competition_id, safe=""))
        if (result.get("id") != self.task.target_instance
                or result.get("challenge_id") not in (None, self.task.challenge_id)
                or result.get("competition_id") not in (None, self.task.competition_id)):
            raise HostedContractError("instance response does not match target_instance")
        return result

    async def ensure_instance(self, task_id: str) -> dict[str, Any]:
        if task_id != self.task.challenge_id:
            raise HostedContractError("hosted token is scoped to its assigned challenge")
        return await self.get_instance()

    async def submit_answer(self, task_id: str, question_id: str, answer: str) -> SubmissionResult:
        if task_id != self.task.challenge_id:
            raise HostedContractError("hosted token is scoped to its assigned challenge")
        if self.task.run_mode not in {"instance", "qa"}:
            raise HostedContractError("this run mode has a different deliverable than question answers")
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        if question_id not in {q["id"] for q in self.questions}:
            raise HostedContractError("question is not visible in this run")
        digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        key = (question_id, digest)
        if question_id in self.confirmed or key in self.attempted:
            return SubmissionResult(task_id=task_id, question_id=question_id, status="duplicate")
        if self.task.dry_run or not self.task.platform_url or not self.token:
            raise HostedContractError("cannot submit without injected platform credentials")
        # Mark before sending: a timeout may mean the server accepted the answer.
        self.attempted.add(key)
        if self.task.run_mode == "qa":
            path = "/qa/submissions"
            body = {"challenge_id": task_id, "competition_id": self.task.competition_id,
                    "question_id": question_id, "answer": answer}
        else:
            if not self.task.target_instance:
                raise HostedContractError("no target_instance for submission")
            path = "/submissions"
            body = {"instance_id": self.task.target_instance, "question_id": question_id, "answer": answer}
        try:
            result = await self._request("submit_answer", "POST", path, body=body)
        except HostedPlatformError as exc:
            # A rate limit or a closed play window rejects before grading.
            if exc.status_code == 429 or exc.code in {"competition_paused", "competition_not_in_session"}:
                self.attempted.discard(key)
            raise
        if result.get("question_id") not in (None, question_id) or not isinstance(result.get("correct"), bool):
            raise HostedContractError("submission response cannot be matched to question")
        if result["correct"]:
            self.confirmed.add(question_id)
            self.fully_solved = bool(result.get("challenge_fully_solved", False))
            if self.task.gated_questions:
                try:
                    await self.refresh_questions()
                except (httpx.HTTPError, HostedPlatformError, HostedContractError):
                    # The accepted receipt remains authoritative. The controller
                    # can refresh the unlocked questions on its next cycle.
                    pass
            return SubmissionResult.accepted(task_id, question_id, result)
        return SubmissionResult(task_id=task_id, question_id=question_id, status="rejected", raw=result)

    async def fetch_attachment(self, name: str, *, max_bytes: int = 100_000_000) -> Path:
        attachment = next((x for x in self.task.attachments if x.name == name), None)
        if attachment is None:
            raise HostedContractError("attachment is not listed in task.json")
        if attachment.path is not None:
            return attachment.path
        if not self.task.target_instance:
            raise HostedContractError("missing attachment has no target instance")
        if self.task.dry_run or not self.task.platform_url or not self.token:
            raise HostedContractError("attachment download requires injected platform credentials")
        url = (self.task.platform_url + "/api/v1/instances/" + quote(self.task.target_instance, safe="")
               + "/attachments/" + quote(name, safe="")
               + "?competition_id=" + quote(self.task.competition_id, safe=""))
        directory = self.task.work_dir / "attachments"
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            total = 0
            async with self.client.stream("GET", url, headers={"Authorization": f"Bearer {self.token}"}) as response:
                response.raise_for_status()
                with temp_path.open("wb") as out:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise HostedContractError("attachment exceeds configured size limit")
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            final = directory / name
            os.replace(temp_path, final)
            return final
        finally:
            temp_path.unlink(missing_ok=True)

    async def get_run_limits(self) -> PlatformLimits:
        return PlatformLimits(max_instances=1 if self.task.target_instance else 0)

    async def get_hint(self, task_id: str, *, authorized: bool = False) -> None:
        if task_id != self.task.challenge_id:
            raise HostedContractError("hosted token is scoped to its assigned challenge")
        if not authorized:
            raise PermissionError("hints may cost points; explicit policy required")
        raise HostedContractError("hosted hint purchase contract is not verified for this task")

    async def release_instance(self, task_id: str) -> None:
        if task_id != self.task.challenge_id:
            raise HostedContractError("hosted token is scoped to its assigned challenge")
        # The platform owns the prestarted target and tears it down after the run.
        return None

    def export_result(self, *, turns: int = 0, tool_calls_made: int = 0,
                      input_tokens: int = 0, output_tokens: int = 0,
                      stop_reason: str = "done", writeup: str | None = None,
                      unsubmitted_answers: Mapping[str, str] | None = None) -> Path:
        """Atomically write the v3 rollup; only platform-confirmed IDs enter solved."""
        if any(x < 0 for x in (turns, tool_calls_made, input_tokens, output_tokens)):
            raise ValueError("result counters must be nonnegative")
        data: dict[str, Any] = {
            "schema_version": 3, "solved": sorted(self.confirmed),
            "fully_solved": self.fully_solved, "turns": turns,
            "tool_calls_made": tool_calls_made, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "stop_reason": stop_reason,
        }
        pending: dict[str, str] = {}
        for question_id, answer in (unsubmitted_answers or {}).items():
            if question_id not in {q["id"] for q in self.questions} or question_id in self.confirmed:
                continue
            digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            if (question_id, digest) not in self.attempted:
                pending[question_id] = answer
        if pending:
            data["answers"] = pending
        if writeup is not None:
            data["writeup"] = writeup
        result_path = self.task.work_dir / "result.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.task.work_dir,
                                         prefix=".result-", delete=False) as tmp:
            temp_path = Path(tmp.name)
            json.dump(data, tmp, ensure_ascii=False)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            os.replace(temp_path, result_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return result_path
