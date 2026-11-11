"""Bounded, task-scoped shell execution with process-group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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
    stdout_capture_truncated: bool = False
    stderr_capture_truncated: bool = False


def _bounded_text(data: bytes, limit: int) -> tuple[str, bool]:
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


class ToolExecutor:
    def __init__(self, root: str | Path, evidence: EvidenceStore, *, output_limit: int = 65536,
                 max_processes: int = 2, capture_limit: int | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        if capture_limit is None:
            capture_limit = max(output_limit, 16 * 1024 * 1024)
        if output_limit < 1 or max_processes < 1 or capture_limit < output_limit:
            raise ValueError("invalid tool limits")
        self.output_limit = output_limit
        self.capture_limit = capture_limit
        self._semaphore = asyncio.Semaphore(max_processes)

    async def _capture(self, stream: asyncio.StreamReader, output: BinaryIO) -> bool:
        """Drain a child pipe while retaining at most capture_limit bytes."""
        retained = 0
        truncated = False
        while chunk := await stream.read(65536):
            available = self.capture_limit - retained
            if available > 0:
                kept = chunk[:available]
                output.write(kept)
                retained += len(kept)
            if len(chunk) > available:
                truncated = True
        return truncated

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
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-c", command, cwd=cwd, env=self._child_env(),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                assert process.stdout is not None and process.stderr is not None
                stdout_task = asyncio.create_task(self._capture(process.stdout, stdout_file))
                stderr_task = asyncio.create_task(self._capture(process.stderr, stderr_file))
                finished = asyncio.gather(process.wait(), stdout_task, stderr_task)
                timed_out = False
                try:
                    _, out_capture_trunc, err_capture_trunc = await asyncio.wait_for(
                        asyncio.shield(finished), timeout=timeout_s
                    )
                except TimeoutError:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    _, out_capture_trunc, err_capture_trunc = await finished
                except asyncio.CancelledError:
                    # A stopped worker must not leave its tool process running.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await finished
                    raise
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_ref = self.evidence.put_stream(stdout_file)
                stderr_ref = self.evidence.put_stream(stderr_file)
                stdout_file.seek(0)
                stderr_file.seek(0)
                out_text, out_trunc = _bounded_text(stdout_file.read(self.output_limit + 1), self.output_limit)
                err_text, err_trunc = _bounded_text(stderr_file.read(self.output_limit + 1), self.output_limit)
        return ShellResult(process.returncode, out_text, err_text, stdout_ref, stderr_ref,
                           time.monotonic() - started, timed_out,
                           out_trunc or out_capture_trunc, err_trunc or err_capture_trunc,
                           out_capture_trunc, err_capture_trunc)

    def stage_attachment(self, task_id: str, name: str, source: str | Path,
                         *, max_bytes: int = 100_000_000) -> EvidenceRef:
        """Copy a platform attachment into one task without trusting its filename."""
        if (not name or Path(name).name != name or name in {".", ".."}
                or "\\" in name or max_bytes < 1):
            raise ValueError("attachment name must be a simple filename")
        destination = self.task_dir(task_id) / name
        source_path = Path(source)
        if source_path.resolve() == destination.resolve() and not destination.is_symlink():
            if destination.stat().st_size > max_bytes:
                raise ValueError("attachment exceeds size limit")
        else:
            temporary: Path | None = None
            try:
                with source_path.open("rb") as input_file, tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".attachment-", delete=False
                ) as output_file:
                    temporary = Path(output_file.name)
                    size = 0
                    while chunk := input_file.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("attachment exceeds size limit")
                        output_file.write(chunk)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.replace(temporary, destination)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        with destination.open("rb") as staged:
            return self.evidence.put_stream(staged)

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
