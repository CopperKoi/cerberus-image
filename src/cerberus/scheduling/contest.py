"""Bounded multi-challenge dispatch.

The controller owns limits. A model may suggest priority, but only verified
progress, declared cost and wait time affect this deterministic queue. A task
is revisited when its platform projection changes or a worker explicitly asks
for another bounded pass. This module never chooses an exploit method.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from cerberus.contracts import Challenge, PlatformLimits


@dataclass(frozen=True)
class SchedulePolicy:
    max_tasks: int = 2
    local_instance_limit: int = 1
    max_cycles: int = 20
    max_revisits_per_projection: int = 1
    idle_poll_s: float = 0.0
    max_idle_cycles: int = 0
    age_weight_per_s: float = 0.001
    progress_weight: float = 2.0
    point_weight: float = 1.0
    cost_weight: float = 0.5

    def __post_init__(self) -> None:
        if min(self.max_tasks, self.local_instance_limit, self.max_cycles) < 1:
            raise ValueError("task, instance and cycle limits must be positive")
        if self.max_revisits_per_projection < 0 or self.max_idle_cycles < 0:
            raise ValueError("revisit limit must be nonnegative")
        weights = (self.age_weight_per_s, self.progress_weight, self.point_weight,
                   self.cost_weight, self.idle_poll_s)
        if any(not math.isfinite(x) or x < 0 for x in weights):
            raise ValueError("scoring weights must be finite and nonnegative")


@dataclass(frozen=True)
class WorkOutcome:
    """Worker's scheduling observation, not a platform success verdict."""

    revisit: bool = False
    estimated_cost: float | None = None
    pause_s: float = 0.0

    def __post_init__(self) -> None:
        if self.estimated_cost is not None and (not math.isfinite(self.estimated_cost) or self.estimated_cost < 0):
            raise ValueError("estimated cost must be nonnegative and finite")
        if not math.isfinite(self.pause_s) or self.pause_s < 0:
            raise ValueError("pause must be nonnegative and finite")


@dataclass(frozen=True)
class DispatchRecord:
    task_id: str
    instance_epoch: int
    cycle: int
    priority: float
    result: str
    projection_changed: bool


class SlotPool:
    """Local permit pool that can tighten when the platform reports a lower cap."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("slot limit must be positive")
        self.limit = limit
        self.active = 0
        self.peak = 0
        self._condition = asyncio.Condition()

    async def set_limit(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("slot limit must be nonnegative")
        async with self._condition:
            self.limit = limit
            self._condition.notify_all()

    @asynccontextmanager
    async def reserve(self):
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.limit)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            async with self._condition:
                self.active -= 1
                self._condition.notify_all()


def _requires_instance(task: Challenge) -> bool:
    target = task.target
    if target.get("is_qa_only") is True or target.get("run_mode") == "qa":
        return False
    if target.get("requires_instance") is False:
        return False
    return True


def _progress(task: Challenge) -> tuple[int, int]:
    target = task.target
    if isinstance(target.get("flag_count"), int) and isinstance(target.get("correct_flag_count"), int):
        total = max(0, target["flag_count"])
        accepted = min(total, max(0, target["correct_flag_count"]))
        return accepted, total
    return sum(q.status == "accepted" for q in task.questions), len(task.questions)


def _projection(task: Challenge) -> tuple[Any, ...]:
    accepted, total = _progress(task)
    return (task.instance_epoch, task.available, accepted, total,
            tuple(sorted((q.id, q.status) for q in task.questions)),
            task.target.get("container_status"), task.target.get("instance_status"))


def _feasible(task: Challenge) -> bool:
    if not task.available or task.target.get("unservable") is True or task.target.get("blocked") is True:
        return False
    return not (task.questions and all(q.status in ("accepted", "locked") for q in task.questions))


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class ContestScheduler:
    """Dispatch across a full visible task pool with bounded rediscovery.

    ``work`` is responsible for task-local budgets and platform actions.
    ``closing`` is the global budget gate and is checked before every batch.
    The platform adapter remains authoritative for remote instance occupancy;
    this pool only prevents concurrent local requests from exceeding limits.
    """

    def __init__(self, policy: SchedulePolicy, *, now: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy
        self.now = now
        self.instances = SlotPool(policy.local_instance_limit)
        self.records: list[DispatchRecord] = []
        self._seen_at: dict[str, float] = {}
        self._costs: dict[str, float] = {}
        self._attempts: dict[tuple[str, tuple[Any, ...]], int] = {}
        self._paused_until: dict[str, float] = {}
        self._last_projection: dict[str, tuple[Any, ...]] = {}

    def priority(self, task: Challenge) -> float:
        if not _feasible(task):
            return float("-inf")
        accepted, total = _progress(task)
        progress = accepted / total if total else 0.0
        remaining = (task.score or 0.0) * (1.0 - progress)
        points = math.log1p(max(0.0, remaining))
        age = max(0.0, self.now() - self._seen_at.get(task.task_id, self.now()))
        declared_cost = task.target.get("estimated_cost", 0.0)
        if not isinstance(declared_cost, (int, float)) or not math.isfinite(declared_cost) or declared_cost < 0:
            declared_cost = 0.0
        cost = self._costs.get(task.task_id, float(declared_cost))
        return (self.policy.point_weight * points
                + self.policy.progress_weight * progress
                + self.policy.age_weight_per_s * age
                - self.policy.cost_weight * math.log1p(cost))

    async def run(
        self,
        discover: Callable[[], list[Challenge] | Awaitable[list[Challenge]]],
        work: Callable[[Challenge], WorkOutcome | Awaitable[WorkOutcome | None] | None],
        *,
        get_limits: Callable[[], PlatformLimits | Awaitable[PlatformLimits]] | None = None,
        closing: Callable[[], bool] = lambda: False,
    ) -> list[DispatchRecord]:
        idle_cycles = 0
        for cycle in range(self.policy.max_cycles):
            if closing():
                break
            tasks = await _await(discover())
            if len({task.task_id for task in tasks}) != len(tasks):
                raise ValueError("platform returned duplicate task IDs")
            current = self.now()
            for task in tasks:
                self._seen_at.setdefault(task.task_id, current)
            limits = None
            if get_limits is not None:
                limits = await _await(get_limits())
                if limits.max_instances is not None:
                    await self.instances.set_limit(min(self.policy.local_instance_limit, limits.max_instances))
            eligible = [task for task in tasks if self._eligible(task)]
            if not eligible:
                if (idle_cycles >= self.policy.max_idle_cycles or self.policy.idle_poll_s <= 0):
                    break
                idle_cycles += 1
                platform_wait = (limits.poll_after_s or 0.0) if limits is not None else 0.0
                await asyncio.sleep(max(self.policy.idle_poll_s, platform_wait))
                continue
            idle_cycles = 0
            eligible.sort(key=lambda task: (-self.priority(task), task.task_id))
            # Every batch is bounded by task slots. An instance permit is held
            # around the worker so its ensure/release pair cannot exceed cap.
            batch = eligible[: self.policy.max_tasks]
            await asyncio.gather(*(self._dispatch(task, cycle, work, closing) for task in batch))
        return list(self.records)

    def _eligible(self, task: Challenge) -> bool:
        if not _feasible(task) or self.now() < self._paused_until.get(task.task_id, 0.0):
            return False
        if _requires_instance(task) and self.instances.limit == 0:
            return False
        key = (task.task_id, _projection(task))
        return self._attempts.get(key, 0) <= self.policy.max_revisits_per_projection

    async def _dispatch(self, task: Challenge, cycle: int,
                        work: Callable[[Challenge], WorkOutcome | Awaitable[WorkOutcome | None] | None],
                        closing: Callable[[], bool]) -> None:
        if closing():
            return
        projection = _projection(task)
        key = (task.task_id, projection)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        priority = self.priority(task)
        try:
            if _requires_instance(task):
                async with self.instances.reserve():
                    outcome = await _await(work(task))
            else:
                outcome = await _await(work(task))
            if outcome is None:
                outcome = WorkOutcome()
            if not isinstance(outcome, WorkOutcome):
                raise TypeError("work must return WorkOutcome or None")
            if outcome.estimated_cost is not None:
                self._costs[task.task_id] = outcome.estimated_cost
            self._paused_until[task.task_id] = self.now() + outcome.pause_s
            # By default a projection is attempted once. An explicit revisit
            # allows a bounded second pass under the same platform state.
            if not outcome.revisit:
                self._attempts[key] = self.policy.max_revisits_per_projection + 1
            result = "worked"
        except Exception as exc:
            result = f"error:{type(exc).__name__}"
            self._attempts[key] = self.policy.max_revisits_per_projection + 1
        prior = self._last_projection.get(task.task_id)
        self._last_projection[task.task_id] = projection
        self.records.append(DispatchRecord(task.task_id, task.instance_epoch, cycle, priority,
                                           result, prior is not None and prior != projection))
