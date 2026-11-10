"""Small platform-neutral contracts. IDs route requests; they never choose solve methods."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: int = 1


class Question(Contract):
    id: str
    prompt: str = ""
    status: Literal["open", "accepted", "rejected", "locked", "unknown"] = "unknown"
    max_attempts: int | None = None
    attempts_used: int | None = None


class Challenge(Contract):
    task_id: str
    title: str
    description: str = ""
    category: str = "unknown"
    questions: list[Question] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    target: dict[str, Any] = Field(default_factory=dict)
    instance_epoch: int = 0
    score: float | None = None
    available: bool = True


class PlatformLimits(Contract):
    max_instances: int | None = None
    max_submissions: int | None = None
    poll_after_s: float | None = None
    deadline: datetime | None = None
    authoring_allowed: bool = False


class SubmissionResult(Contract):
    task_id: str
    question_id: str
    status: Literal["accepted", "rejected", "unknown", "duplicate"]
    raw: dict[str, Any] = Field(default_factory=dict)
    accepted_at: datetime | None = None

    @classmethod
    def accepted(cls, task_id: str, question_id: str, raw: dict[str, Any] | None = None) -> SubmissionResult:
        return cls(task_id=task_id, question_id=question_id, status="accepted", raw=raw or {},
                   accepted_at=datetime.now(UTC))
