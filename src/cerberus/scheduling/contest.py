"""Bounded multi-challenge dispatch.

The controller owns limits. A model may suggest priority, but only verified
progress, declared cost and wait time affect this deterministic queue. A task
is revisited when its platform projection changes or a worker explicitly asks
for another bounded pass. This module never chooses an exploit method.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from cerberus.contracts import Challenge, PlatformLimits


@dataclass(frozen=True)
class SchedulePolicy:
    max_tasks: int = 2
    local_instance_limit: int = 1
    max_cycles: int = 20
    max_revisits_per_projection: int = 1
    max_total_dispatches_per_task: int = 0
    idle_poll_s: float = 0.0
    max_idle_cycles: int = 0
    wait_until_closing: bool = False
    age_weight_per_s: float = 0.001
    progress_weight: float = 2.0
    point_weight: float = 1.0
    cost_weight: float = 0.5

    def __post_init__(self) -> None:
        if min(self.max_tasks, self.local_instance_limit, self.max_cycles) < 1:
            raise ValueError("task, instance and cycle limits must be positive")
        if (self.max_revisits_per_projection < 0 or self.max_idle_cycles < 0
                or self.max_total_dispatches_per_task < 0):
            raise ValueError("revisit limit must be nonnegative")
        if self.wait_until_closing and self.idle_poll_s <= 0:
            raise ValueError("waiting for new tasks requires a positive idle poll interval")
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

    def __init__(self, policy: SchedulePolicy, *, now: Callable[[], float] = time.monotonic,
                 wall_now: Callable[[], float] = time.time,
                 checkpoint: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.policy = policy
        self.now = now
        self.wall_now = wall_now
        self.checkpoint = checkpoint
        self.instances = SlotPool(policy.local_instance_limit)
        self.records: list[DispatchRecord] = []
        self._seen_at: dict[str, float] = {}
        self._costs: dict[str, float] = {}
        self._attempts: dict[tuple[str, tuple[Any, ...]], int] = {}
        self._total_dispatches: dict[str, int] = {}
        self._paused_until: dict[str, float] = {}
        self._last_projection: dict[str, tuple[Any, ...]] = {}
        self._inflight: dict[str, tuple[Any, ...]] = {}
        self._cycle = 0
        self._checkpoint_lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        """Return durable scheduling decisions, excluding unfinished dispatches.

        Local slot permits and remote instance leases are deliberately absent:
        after a restart the platform adapter must reconcile live resources.
        Pauses and task age use wall-clock elapsed time across process restarts.
        """
        now = self.now()
        attempts = dict(self._attempts)
        dispatches = dict(self._total_dispatches)
        for task_id, projection in self._inflight.items():
            key = (task_id, projection)
            attempts[key] -= 1
            if attempts[key] == 0:
                del attempts[key]
            dispatches[task_id] -= 1
            if dispatches[task_id] == 0:
                del dispatches[task_id]
        return {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "saved_at_epoch": self.wall_now(),
            "cycle": self._cycle,
            "seen_age_s": {task_id: max(0.0, now - seen) for task_id, seen in self._seen_at.items()},
            "costs": dict(self._costs),
            "attempts": [[task_id, _json_projection(projection), count]
                         for (task_id, projection), count in attempts.items()],
            "total_dispatches": dispatches,
            "pause_remaining_s": {task_id: max(0.0, until - now)
                                  for task_id, until in self._paused_until.items()},
            "last_projection": {task_id: _json_projection(projection)
                                for task_id, projection in self._last_projection.items()},
            "records": [asdict(record) for record in self.records],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a completed checkpoint before dispatching any new work."""
        if self.records or self._attempts or self._inflight or self._cycle:
            raise ValueError("scheduler state can only be restored before work starts")
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ValueError("unsupported scheduler checkpoint")
        if snapshot.get("policy") != asdict(self.policy):
            raise ValueError("scheduler checkpoint policy does not match")
        try:
            saved_at = _finite_number(snapshot["saved_at_epoch"])
            elapsed = max(0.0, self.wall_now() - saved_at)
            cycle = _nonnegative_int(snapshot["cycle"])
            seen_age = _number_map(snapshot["seen_age_s"])
            costs = _number_map(snapshot["costs"])
            total = _int_map(snapshot["total_dispatches"])
            pause = _number_map(snapshot["pause_remaining_s"])
            attempts: dict[tuple[str, tuple[Any, ...]], int] = {}
            for item in snapshot["attempts"]:
                task_id, raw_projection, count = item
                if not isinstance(task_id, str):
                    raise ValueError("invalid task ID")
                key = (task_id, _restore_projection(raw_projection))
                if key in attempts:
                    raise ValueError("duplicate scheduler attempt")
                attempts[key] = _nonnegative_int(count)
            raw_last = snapshot["last_projection"]
            if not isinstance(raw_last, dict) or any(not isinstance(key, str) for key in raw_last):
                raise ValueError("invalid last projection")
            last = {task_id: _restore_projection(projection)
                    for task_id, projection in raw_last.items()}
            records = [DispatchRecord(**record) for record in snapshot["records"]]
            if any(not isinstance(record.task_id, str) or not math.isfinite(record.priority)
                   or record.cycle < 0 for record in records):
                raise ValueError("invalid dispatch record")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid scheduler checkpoint") from exc
        now = self.now()
        self._cycle = cycle
        self._seen_at = {task_id: now - age - elapsed for task_id, age in seen_age.items()}
        self._costs = costs
        self._attempts = attempts
        self._total_dispatches = total
        self._paused_until = {task_id: now + max(0.0, remaining - elapsed)
                              for task_id, remaining in pause.items()}
        self._last_projection = last
        self.records = records

    async def _persist_checkpoint(self) -> None:
        if self.checkpoint is None:
            return
        async with self._checkpoint_lock:
            await _await(self.checkpoint(self.snapshot()))

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
        cycle = self._cycle
        while cycle < self.policy.max_cycles:
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
                # A temporary worker failure may carry Retry-After longer than
                # the ordinary idle observation window. Keep its bounded retry
                # alive instead of exiting while that task is cooling down.
                retry_at = min((self._paused_until[task.task_id] for task in tasks
                                if _feasible(task)
                                and self._attempts.get((task.task_id, _projection(task)), 0)
                                <= self.policy.max_revisits_per_projection
                                and self._paused_until.get(task.task_id, 0.0) > self.now()),
                               default=None)
                waiting_for_retry = retry_at is not None
                if (not self.policy.wait_until_closing and not waiting_for_retry
                        and (idle_cycles >= self.policy.max_idle_cycles or self.policy.idle_poll_s <= 0)):
                    break
                platform_wait = (limits.poll_after_s or 0.0) if limits is not None else 0.0
                sleep_s = max(self.policy.idle_poll_s, platform_wait)
                if waiting_for_retry:
                    # Recheck the closing gate while honoring a platform poll floor.
                    sleep_s = max(sleep_s, min(max(0.0, retry_at - self.now()), 5.0))
                else:
                    idle_cycles += 1
                await asyncio.sleep(sleep_s)
                if not self.policy.wait_until_closing and not waiting_for_retry:
                    cycle += 1
                    self._cycle = cycle
                    await self._persist_checkpoint()
                continue
            idle_cycles = 0
            # Give each task one exploration pass before revisiting a task
            # whose score rose through partial progress. This stops a small
            # set of early tasks from consuming the entire shared budget.
            eligible.sort(key=lambda task: (
                self._total_dispatches.get(task.task_id, 0) > 0,
                -self.priority(task), task.task_id))
            # Every batch is bounded by task slots. An instance permit is held
            # around the worker so its ensure/release pair cannot exceed cap.
            batch = eligible[: self.policy.max_tasks]
            await asyncio.gather(*(self._dispatch(task, cycle, work, closing) for task in batch))
            cycle += 1
            self._cycle = cycle
            await self._persist_checkpoint()
        return list(self.records)

    def _eligible(self, task: Challenge) -> bool:
        if not _feasible(task) or self.now() < self._paused_until.get(task.task_id, 0.0):
            return False
        if (self.policy.max_total_dispatches_per_task
                and self._total_dispatches.get(task.task_id, 0)
                >= self.policy.max_total_dispatches_per_task):
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
        priority = self.priority(task)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        self._total_dispatches[task.task_id] = self._total_dispatches.get(task.task_id, 0) + 1
        self._inflight[task.task_id] = projection
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
        except asyncio.CancelledError:
            # A stopped quantum has no completed scheduling outcome. Platform
            # side effects are reconciled separately by the runtime and ledger.
            self._attempts[key] -= 1
            if self._attempts[key] == 0:
                del self._attempts[key]
            self._total_dispatches[task.task_id] -= 1
            if self._total_dispatches[task.task_id] == 0:
                del self._total_dispatches[task.task_id]
            del self._inflight[task.task_id]
            raise
        except Exception as exc:
            result = f"error:{type(exc).__name__}"
            if _retryable_worker_error(exc):
                retry_after = getattr(exc, "retry_after_s", None)
                if isinstance(retry_after, (int, float)) and math.isfinite(retry_after) and retry_after > 0:
                    self._paused_until[task.task_id] = self.now() + retry_after
            else:
                self._attempts[key] = self.policy.max_revisits_per_projection + 1
        prior = self._last_projection.get(task.task_id)
        self._last_projection[task.task_id] = projection
        self.records.append(DispatchRecord(task.task_id, task.instance_epoch, cycle, priority,
                                           result, prior is not None and prior != projection))
        del self._inflight[task.task_id]
        await self._persist_checkpoint()


def _json_projection(projection: tuple[Any, ...]) -> list[Any]:
    # Projection fields come from platform state; reject values a JSON
    # checkpoint could silently alter before they become retry keys.
    return json.loads(json.dumps(projection, allow_nan=False))


def _restore_projection(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid projection")

    def freeze(item: Any) -> Any:
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("invalid projection value")
            return item
        raise ValueError("invalid projection value")
    return tuple(freeze(item) for item in value)


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid number")
    if value < 0:
        raise ValueError("negative number")
    return float(value)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid count")
    return value


def _number_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("invalid map")
    return {key: _finite_number(number) for key, number in value.items()}


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("invalid map")
    return {key: _nonnegative_int(number) for key, number in value.items()}


def _retryable_worker_error(exc: Exception) -> bool:
    """Retry transport and temporary platform failures without replaying other errors."""
    if isinstance(exc, httpx.RequestError):
        return True
    status = getattr(exc, "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    return status == 409 and getattr(exc, "code", None) == "resource_unavailable"
