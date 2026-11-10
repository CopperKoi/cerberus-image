"""Explicit configuration, strict safety critical fields and redacted snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class Budget(BaseModel):
    model_config = ConfigDict(extra="allow")
    wall_seconds: PositiveInt = 1800
    total_tokens: PositiveInt = 120000
    max_model_calls: PositiveInt = 100
    max_tool_calls: PositiveInt = 200
    max_output_bytes: PositiveInt = 262144
    closing_seconds: PositiveInt = 60

    @model_validator(mode="after")
    def closing_fits(self) -> Budget:
        if self.closing_seconds >= self.wall_seconds:
            raise ValueError("budget.closing_seconds must be less than wall_seconds")
        return self


class Concurrency(BaseModel):
    model_config = ConfigDict(extra="allow")
    tasks: PositiveInt = 2
    workers_per_task: PositiveInt = 1
    model_requests: PositiveInt = 2
    tool_processes: PositiveInt = 2


class Provider(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: Literal["deepseek", "mock"] = "deepseek"
    base_url_env: str = "LLM_BASE_URL"
    api_key_env: str = "LLM_API_KEY"
    model_env: str = "LLM_MODEL"
    development_base_url: str | None = None
    development_model: str = "deepseek-flash"
    thinking: bool = False
    request_timeout_s: PositiveInt = 120


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1
    mode: Literal["mock", "ctfy-hosted", "tsecbench", "ctfy-competition"] = "mock"
    run_id: str | None = None
    work_dir: Path = Path("./work")
    contract_version: str | None = None
    budget: Budget = Field(default_factory=Budget)
    concurrency: Concurrency = Field(default_factory=Concurrency)
    provider: Provider = Field(default_factory=Provider)
    platform_base_url_env: str = "BENCHMARK_BASE_URL"
    platform_token_env: str = "BENCHMARK_TOKEN"
    corpus_version: str | None = None
    competition_id: str | None = None
    allow_hints: bool = False
    instance_limit: PositiveInt | None = None

    @model_validator(mode="after")
    def mode_fields(self) -> RunConfig:
        if self.contract_version is not None:
            expected = "CTFY-0001/3" if self.mode == "ctfy-hosted" else "TSecBench-Challenges/1" if self.mode == "tsecbench" else None
            if expected is not None and self.contract_version != expected:
                raise ValueError(f"unsupported contract_version for {self.mode}: {self.contract_version}")
        if self.mode == "tsecbench" and not self.corpus_version:
            raise ValueError("corpus_version is required for tsecbench; API version is not corpus version")
        if self.mode == "ctfy-competition" and not self.competition_id:
            raise ValueError("competition_id is required for ctfy-competition")
        return self

    def resolved_provider(self, *, environ: dict[str, str] | None = None) -> tuple[str, str, str]:
        env = os.environ if environ is None else environ
        if self.provider.kind == "mock":
            return "mock", "", "mock"
        key = env.get(self.provider.api_key_env, "")
        # Hosted endpoint and model are platform authority. Development defaults are forbidden there.
        hosted = self.mode in ("ctfy-hosted", "ctfy-competition", "tsecbench")
        base = env.get(self.provider.base_url_env) or (None if hosted else self.provider.development_base_url)
        model = env.get(self.provider.model_env) or (None if hosted else self.provider.development_model)
        missing = [name for name, value in [(self.provider.api_key_env, key),
                                            (self.provider.base_url_env, base),
                                            (self.provider.model_env, model)] if not value]
        if missing:
            raise ValueError("missing provider configuration: " + ", ".join(missing))
        return str(base), key, str(model)

    def redacted(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        # Extra forward-compatible fields can carry platform secrets.
        def scrub(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {k: scrub(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v, key) for v in value]
            key_lower = key.lower()
            sensitive = (key_lower in {"token", "api_key", "secret", "password", "credential"}
                         or any(key_lower.endswith("_" + word) for word in ("secret", "password", "token", "api_key", "credential")))
            if sensitive and not key_lower.endswith("_env"):
                return "<redacted>"
            return value
        data = scrub(data)
        data["work_dir"] = str(Path(data["work_dir"]).resolve())
        if data.get("provider", {}).get("development_base_url"):
            data["provider"]["development_base_url"] = "<configured>"
        return data

    def config_hash(self) -> str:
        data = json.dumps(self.redacted(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()


def load_config(path: Path, *, work_dir: Path | None = None, mode: str | None = None) -> RunConfig:
    with path.open("r", encoding="utf-8") as source:
        raw = yaml.safe_load(source) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    if mode is not None:
        if "mode" in raw and raw["mode"] != mode:
            raise ValueError(f"configuration mode {raw['mode']} conflicts with requested mode {mode}")
        raw["mode"] = mode
    if work_dir is not None:
        raw["work_dir"] = str(work_dir)
    elif "work_dir" in raw and not Path(raw["work_dir"]).is_absolute():
        raw["work_dir"] = str((path.parent / raw["work_dir"]).resolve())
    return RunConfig.model_validate(raw)
