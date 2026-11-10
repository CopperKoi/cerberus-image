"""Deterministic no-network mode for boot and packaging checks."""

from cerberus.contracts import Challenge, PlatformLimits, Question, SubmissionResult


class MockPlatform:
    def __init__(self) -> None:
        self._done: set[str] = set()

    def discover_tasks(self) -> list[Challenge]:
        return [Challenge(task_id="mock-task", title="Mock task", category="misc",
                          description="Offline boot diagnostic", questions=[Question(id="mock-question")],
                          available="mock-question" not in self._done)]

    def refresh_progress(self) -> list[Challenge]:
        return self.discover_tasks()

    def ensure_instance(self, task_id: str) -> dict:
        return {"status": "available", "task_id": task_id}

    def submit_answer(self, task_id: str, question_id: str, answer: str) -> SubmissionResult:
        if task_id == "mock-task" and question_id == "mock-question" and answer == "MOCK_OK":
            self._done.add(question_id)
            return SubmissionResult.accepted(task_id, question_id)
        return SubmissionResult(task_id=task_id, question_id=question_id, status="rejected")

    def get_run_limits(self) -> PlatformLimits:
        return PlatformLimits(max_instances=1)
