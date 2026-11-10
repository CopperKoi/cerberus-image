"""Command line entry point. Diagnostics never spend model credits by default."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import typer

from cerberus import __version__
from cerberus.config import RunConfig, load_config
from cerberus.platforms.tsecbench import TSecBenchAdapter
from cerberus.providers.deepseek import DeepSeekChatClient
from cerberus.runtime.agent import AgentRun

app = typer.Typer(no_args_is_help=True)


def _configuration(config: Path, mode: str | None, work_dir: Path | None) -> RunConfig:
    selected = load_config(config, mode=mode, work_dir=work_dir)
    hosted_work = os.environ.get("CTFY_WORK_DIR")
    if selected.mode == "ctfy-hosted" and hosted_work:
        injected = Path(hosted_work).resolve()
        if work_dir is not None and Path(work_dir).resolve() != injected:
            raise ValueError("--work-dir conflicts with injected CTFY_WORK_DIR")
        selected.work_dir = injected
    return selected


def _platform(config: RunConfig):
    if config.mode == "ctfy-hosted":
        from cerberus.platforms.ctfy_hosted import CTFyHostedAdapter, HostedTask

        task = HostedTask.from_work_dir(Path(os.environ.get("CTFY_WORK_DIR", config.work_dir)))
        return CTFyHostedAdapter(task, os.environ.get("CTFY_TOKEN"))
    if config.mode == "tsecbench":
        base = os.environ.get(config.platform_base_url_env)
        token = os.environ.get(config.platform_token_env)
        if not base or not token:
            raise ValueError("missing TSecBench platform URL or token")
        return TSecBenchAdapter(base, token, corpus_version=config.corpus_version or "unknown")
    if config.mode == "ctfy-competition":
        from cerberus.platforms.ctfy_competition import CTFyCompetitionAdapter

        base = os.environ.get("CTFY_PLATFORM_URL")
        token = os.environ.get("CTFY_TOKEN")
        if not base or not token:
            raise ValueError("ctfy-competition contract_unverified: CTFY_PLATFORM_URL and CTFY_TOKEN required")
        return CTFyCompetitionAdapter(base, token, config.competition_id or "", instance_limit=config.instance_limit)
    from cerberus.platforms.mock import MockPlatform

    return MockPlatform()


def _provider(config: RunConfig):
    if config.provider.kind == "mock":
        return None
    if config.mode == "ctfy-hosted" and os.environ.get("CTFY_EVAL_DRY_RUN") == "1":
        # CTFY-0001 conformance runs deliberately inject no model key or
        # gateway. The hosted task adapter still writes result.json.
        return None
    if config.mode == "ctfy-hosted":
        hosted_task = Path(os.environ.get("CTFY_WORK_DIR", config.work_dir)) / "task.json"
        if hosted_task.is_file():
            try:
                if json.loads(hosted_task.read_text(encoding="utf-8")).get("dry_run") is True:
                    return None
            except (OSError, ValueError):
                pass  # Task parsing reports the malformed contract separately.
    base, key, model = config.resolved_provider()
    return DeepSeekChatClient(base_url=base, api_key=key, model=model,
                              timeout_s=config.provider.request_timeout_s,
                              max_concurrency=config.concurrency.model_requests)


def _write_hosted_startup_failure(work_dir: Path, platform: object | None) -> None:
    """Record a failed launch even when the agent controller never started."""
    if platform is not None:
        platform.export_result(stop_reason="error")
        return
    result = {"schema_version": 3, "solved": [], "fully_solved": False,
              "turns": 0, "tool_calls_made": 0, "input_tokens": 0,
              "output_tokens": 0, "stop_reason": "error"}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=work_dir,
                                     prefix=".result-", delete=False) as output:
        temporary = Path(output.name)
        json.dump(result, output, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    try:
        os.replace(temporary, work_dir / "result.json")
    finally:
        temporary.unlink(missing_ok=True)


@app.command()
def doctor(config: Path = typer.Option(Path("configs/mock.yaml"), exists=True),
           mode: str | None = typer.Option(None), work_dir: Path | None = typer.Option(None),
           platform_readonly: bool = typer.Option(False, help="Make a platform read-only call."),
           model_check: bool = typer.Option(False, help="Spend model tokens on a gateway handshake.")) -> None:
    selected = _configuration(config, mode, work_dir)
    report = {"version": __version__, "mode": selected.mode,
              "config_hash": selected.config_hash(), "offline": "ok",
              "platform": "not_checked", "model": "not_checked"}
    if platform_readonly:
        platform = _platform(selected)
        async def check_platform():
            tasks = platform.discover_tasks()
            if asyncio.iscoroutine(tasks):
                tasks = await tasks
            close = getattr(platform, "close", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            return len(tasks)
        report["platform"] = {"readable_tasks": asyncio.run(check_platform())}
    if model_check:
        # An offline doctor never requires the hosted gateway variables. This
        # explicit check is the only doctor path allowed to spend model tokens.
        if selected.provider.kind != "mock":
            selected.resolved_provider()
        provider = _provider(selected)
        if provider is None:
            report["model"] = "mock"
        else:
            async def check_model():
                try:
                    turn = await provider.complete([{"role": "user", "content": "Reply OK."}], max_tokens=8)
                    return {"model": turn.model or provider.model, "usage_known": turn.usage is not None}
                finally:
                    await provider.aclose()
            report["model"] = asyncio.run(check_model())
    typer.echo(json.dumps(report, ensure_ascii=False))


@app.command()
def run(config: Path = typer.Option(Path("configs/mock.yaml"), exists=True),
        mode: str | None = typer.Option(None), work_dir: Path | None = typer.Option(None),
        dry_run: bool = typer.Option(False)) -> None:
    selected = _configuration(config, mode, work_dir)
    if dry_run:
        typer.echo(json.dumps({"config": selected.redacted(), "hash": selected.config_hash()}, ensure_ascii=False))
        return
    if selected.mode != "mock" and selected.provider.kind == "mock":
        raise ValueError("mock provider is only allowed in mock mode")
    platform = None
    run_started = False
    try:
        platform = _platform(selected)
        provider = _provider(selected)

        async def execute():
            nonlocal run_started
            agent = AgentRun(selected, platform, provider)
            try:
                run_started = True
                return await agent.run()
            finally:
                await agent.close()

        stats = asyncio.run(execute())
        typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))
    except Exception as exc:
        if selected.mode == "ctfy-hosted" and not run_started:
            hosted_work = Path(os.environ.get("CTFY_WORK_DIR", selected.work_dir)).resolve()
            try:
                _write_hosted_startup_failure(hosted_work, platform)
            except Exception as write_error:
                typer.echo(f"hosted result write failed: {write_error}", err=True)
        typer.echo(f"run failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def resume(config: Path = typer.Option(Path("configs/mock.yaml"), exists=True),
           run_id: str = typer.Option(...), work_dir: Path | None = typer.Option(None)) -> None:
    selected = _configuration(config, None, work_dir)
    selected.run_id = run_id
    if selected.mode != "mock" and selected.provider.kind == "mock":
        raise ValueError("mock provider is only allowed in mock mode")
    platform = _platform(selected)
    provider = _provider(selected)

    async def execute():
        agent = AgentRun(selected, platform, provider, resume=True)
        try:
            return await agent.run()
        finally:
            await agent.close()

    stats = asyncio.run(execute())
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("prepare")
def eval_prepare(release_dir: Path = typer.Argument(..., exists=True, file_okay=False),
                 mode: str = typer.Option(...), dataset: str = typer.Option(...),
                 scope: str = typer.Option("connectivity-smoke")) -> None:
    from cerberus.evaluation import prepare_request

    destination = prepare_request(release_dir, mode=mode, dataset=dataset, scope=scope)
    typer.echo(str(destination))


@eval_app.command("import")
def eval_import(source: Path = typer.Argument(..., exists=True),
                archive: Path = typer.Option(Path("evals/results"))) -> None:
    from cerberus.evaluation import import_feedback

    feedback, saved = import_feedback(source, archive)
    typer.echo(json.dumps({"saved": str(saved), "feedback_id": feedback.feedback_id,
                           "unknown_fields": feedback.unknown_fields()}, ensure_ascii=False))


@eval_app.command("compare")
def eval_compare(spec: Path = typer.Argument(..., exists=True),
                 baseline: Path = typer.Argument(..., exists=True),
                 candidate: Path = typer.Argument(..., exists=True)) -> None:
    import yaml

    from cerberus.evaluation import ExperimentSpec, Feedback, compare_feedback

    plan = ExperimentSpec.model_validate(yaml.safe_load(spec.read_text(encoding="utf-8")))
    left = Feedback.model_validate_json(baseline.read_text(encoding="utf-8"))
    right = Feedback.model_validate_json(candidate.read_text(encoding="utf-8"))
    typer.echo(compare_feedback(plan, left, right).model_dump_json(indent=2))


@eval_app.command("report")
def eval_report(feedback: Path = typer.Argument(..., exists=True)) -> None:
    from cerberus.evaluation import Feedback, failure_report

    result = Feedback.model_validate_json(feedback.read_text(encoding="utf-8"))
    typer.echo(json.dumps({"unknown_fields": result.unknown_fields(),
                           "failures": [x.model_dump() for x in failure_report(result)]}, ensure_ascii=False))


@eval_app.command("audit-context")
def eval_audit_context(context: Path = typer.Argument(..., exists=True)) -> None:
    from cerberus.evaluation import audit_image_context

    findings = audit_image_context(context)
    typer.echo(json.dumps([x.model_dump() for x in findings], ensure_ascii=False))
    if findings:
        raise typer.Exit(code=1)


@app.command("export-run")
def export_run(run_dir: Path = typer.Argument(..., exists=True),
               output: Path | None = typer.Option(None)) -> None:
    from cerberus.memory.store import EvidenceStore, MemoryStore
    from cerberus.submissions import AnswerLedger

    evidence = EvidenceStore(run_dir / "objects")
    memory = MemoryStore(run_dir / "run.db", evidence)
    ledger = AnswerLedger(run_dir / "answers.db")
    try:
        events = [e.__dict__ for e in memory.list_events(run_dir.name)]
        # Raw answers and private model reasoning never enter the report.
        candidates = [{"id": c.id, "task_id": c.task_id, "question_id": c.question_id,
                       "status": c.status, "attempts": c.attempts} for c in ledger.list(run_dir.name)]
        data = {"run_id": run_dir.name, "events": events, "submissions": candidates}
    finally:
        memory.close()
        ledger.close()
    rendered = json.dumps(data, ensure_ascii=False, indent=2)
    if output:
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    app()
