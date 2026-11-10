"""Deterministic contest-level dispatch and local resource admission."""

from .contest import ContestScheduler, DispatchRecord, SchedulePolicy, SlotPool, WorkOutcome

__all__ = ["ContestScheduler", "DispatchRecord", "SchedulePolicy", "SlotPool", "WorkOutcome"]
