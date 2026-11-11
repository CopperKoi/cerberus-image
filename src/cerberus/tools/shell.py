"""Bounded, task-scoped shell execution with process-group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from cerberus.memory.store import EvidenceRef, EvidenceStore


@dataclass(frozen=True)
class ShellResult:
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_ref: EvidenceRef
    stderr_ref: EvidenceRef
    duration_s: float
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool


def _bounded_text(data: bytes, limit: int) -> tuple[str, bool]:
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


class ToolExecutor:
    def __init__(self, root: str | Path, evidence: EvidenceStore, *, output_limit: int = 65536,
                 max_processes: int = 2) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        if output_limit < 1 or max_processes < 1:
            raise ValueError("invalid tool limits")
        self.output_limit = output_limit
        self._semaphore = asyncio.Semaphore(max_processes)

    def task_dir(self, task_id: str) -> Path:
        # Task IDs come from a platform; do not interpret them as path segments.
        import hashlib
        safe = hashlib.sha256(task_id.encode()).hexdigest()[:16]
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _child_env() -> dict[str, str]:
        # Explicit allowlist prevents injected platform/model credentials from
        # reaching programs launched by an untrusted challenge or model output.
        allowed = ("PATH", "LANG", "LC_ALL", "TZ", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR")
        return {key: value for key in allowed if (value := os.environ.get(key)) is not None}

    async def run_shell(self, task_id: str, command: str, *, timeout_s: float = 30.0) -> ShellResult:
        if not command or timeout_s <= 0:
            raise ValueError("command and positive timeout are required")
        cwd = self.task_dir(task_id)
        started = time.monotonic()
        async with self._semaphore:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", command, cwd=cwd, env=self._child_env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
            except TimeoutError:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = await process.communicate()
            except asyncio.CancelledError:
                # A stopped worker must not leave its tool process running.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
                raise
        stdout_ref = self.evidence.put_bytes(stdout)
        stderr_ref = self.evidence.put_bytes(stderr)
        out_text, out_trunc = _bounded_text(stdout, self.output_limit)
        err_text, err_trunc = _bounded_text(stderr, self.output_limit)
        return ShellResult(process.returncode, out_text, err_text, stdout_ref, stderr_ref,
                           time.monotonic() - started, timed_out, out_trunc, err_trunc)

    def read_file(self, task_id: str, relative_path: str, *, offset: int = 0, limit: int = 65536) -> bytes:
        if offset < 0 or limit < 1 or limit > self.output_limit:
            raise ValueError("invalid read bounds")
        path = self._resolve(task_id, relative_path)
        with path.open("rb") as source:
            source.seek(offset)
            return source.read(limit)

    def write_file(self, task_id: str, relative_path: str, content: bytes) -> EvidenceRef:
        if len(content) > self.output_limit * 4:
            raise ValueError("file write exceeds per-call limit")
        path = self._resolve(task_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_NOFOLLOW prevents writing through a final symlink.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        return self.evidence.put_bytes(content)

    def _resolve(self, task_id: str, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("task file path must be relative")
        root = self.task_dir(task_id)
        path = (root / relative_path).resolve()
        if path == root or root not in path.parents:
            raise ValueError("task file escapes workspace")
        return path
