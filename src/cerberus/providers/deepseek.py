"""DeepSeek Chat Completions transport with native tool call continuation.

The caller owns the transcript and tool execution. ``ChatTurn.message`` is the
exact assistant turn to append before the corresponding tool result messages.
Reasoning content is retained for protocol continuation but should not be
exported as a verified fact or included in ordinary reports.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx


class ProviderError(RuntimeError):
    """An upstream model request failed."""


class ProviderTransportError(ProviderError):
    """The model transport failed before a complete turn was returned."""


class ProviderProtocolError(ProviderError):
    """The model gateway returned an invalid or incomplete response."""


class ProviderHTTPError(ProviderError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"model gateway HTTP {status_code}: {message}")


class ProviderAuthError(ProviderHTTPError):
    """A credential or scope was rejected; retrying cannot repair it."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass(frozen=True)
class ChatTurn:
    message: dict[str, Any]
    tool_calls: tuple[ToolCall, ...]
    usage: Usage | None
    finish_reason: str | None
    model: str | None
    response_id: str | None

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


def tool_result_message(call_id: str, content: str) -> dict[str, str]:
    if not call_id:
        raise ValueError("tool call ID is required")
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    return Usage(
        prompt_tokens=_integer(raw.get("prompt_tokens")),
        completion_tokens=_integer(raw.get("completion_tokens")),
        total_tokens=_integer(raw.get("total_tokens")),
        prompt_cache_hit_tokens=_integer(raw.get("prompt_cache_hit_tokens")),
        prompt_cache_miss_tokens=_integer(raw.get("prompt_cache_miss_tokens")),
    )


def _parse_turn(raw: Mapping[str, Any]) -> ChatTurn:
    if not isinstance(raw, dict):
        raise ProviderProtocolError("Chat Completions response must be an object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ProviderProtocolError("expected exactly one Chat Completions choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ProviderProtocolError("assistant message is missing")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ProviderProtocolError("assistant content must be text or null")
    if "reasoning_content" in message and message["reasoning_content"] is not None and not isinstance(message["reasoning_content"], str):
        raise ProviderProtocolError("assistant reasoning_content must be text or null")
    normalized: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if "reasoning_content" in message:
        normalized["reasoning_content"] = message["reasoning_content"]
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ProviderProtocolError("tool_calls must be an array")
    if raw_calls:
        normalized["tool_calls"] = raw_calls
    seen_ids: set[str] = set()
    for call in raw_calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ProviderProtocolError("unsupported tool call type")
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            raise ProviderProtocolError("missing or duplicate tool call ID")
        seen_ids.add(call_id)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ProviderProtocolError("tool call name is missing")
        raw_args = function.get("arguments")
        if not isinstance(raw_args, str):
            raise ProviderProtocolError("tool arguments must be JSON text")
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("tool arguments are invalid JSON") from exc
        if not isinstance(args, dict):
            raise ProviderProtocolError("tool arguments must be a JSON object")
        calls.append(ToolCall(call_id, function["name"], args, raw_args))
    if choice.get("finish_reason") == "length":
        raise ProviderProtocolError("model output truncated at token limit")
    if choice.get("finish_reason") == "tool_calls" and not calls:
        raise ProviderProtocolError("tool_calls finish reason without calls")
    return ChatTurn(
        message=normalized,
        tool_calls=tuple(calls),
        usage=_parse_usage(raw.get("usage")),
        finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        model=raw.get("model") if isinstance(raw.get("model"), str) else None,
        response_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
    )


def _validate_history(messages: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a transcript cut through an assistant tool call group."""
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ProviderProtocolError("orphan or duplicate tool result")
            pending.remove(call_id)
            continue
        if pending:
            raise ProviderProtocolError("assistant tool calls lack matching tool results")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ProviderProtocolError("assistant tool_calls must be an array")
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                    raise ProviderProtocolError("assistant tool call lacks ID")
                if call["id"] in pending:
                    raise ProviderProtocolError("duplicate assistant tool call ID")
                pending.add(call["id"])
        elif role not in {"system", "user"}:
            raise ProviderProtocolError("unsupported message role")
    if pending:
        raise ProviderProtocolError("assistant tool calls lack matching tool results")


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            stamp = parsedate_to_datetime(value)
            return max(0.0, (stamp - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class DeepSeekChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        max_concurrency: int = 1,
        min_request_interval_s: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key and model are required")
        if timeout_s <= 0 or max_retries < 0 or max_concurrency <= 0 or min_request_interval_s < 0:
            raise ValueError("invalid timeout, retry count, concurrency or rate limit")
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_lock = asyncio.Lock()
        self._min_request_interval_s = min_request_interval_s
        self._next_request_at = 0.0

    async def __aenter__(self) -> DeepSeekChatClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        stream: bool = False,
        on_text: Callable[[str], Awaitable[None] | None] | None = None,
        thinking: str | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatTurn:
        _validate_history(messages)
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be enabled, disabled or omitted")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if on_text is not None and not stream:
            raise ValueError("on_text requires stream=True")
        payload: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages], "stream": stream}
        if tools is not None:
            payload["tools"] = list(tools)
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        async with self._semaphore:
            return await self._request(payload, stream=stream, on_text=on_text)

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        stream: bool,
        on_text: Callable[[str], Awaitable[None] | None] | None,
    ) -> ChatTurn:
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            try:
                if stream:
                    async with self._client.stream("POST", self._url, json=payload) as response:
                        if response.is_success:
                            return await _read_stream(response, on_text=on_text)
                        body = await response.aread()
                        status = response.status_code
                        retry_after = response.headers.get("Retry-After")
                else:
                    response = await self._client.post(self._url, json=payload)
                    if response.is_success:
                        try:
                            return _parse_turn(response.json())
                        except (ValueError, TypeError) as exc:
                            raise ProviderProtocolError("invalid JSON response") from exc
                    body = response.content
                    status = response.status_code
                    retry_after = response.headers.get("Retry-After")
            except httpx.RequestError as exc:
                # A transport exception might occur after the gateway accepted
                # work. The controller accounts for each attempted request
                # before deciding whether another turn can be requested.
                raise ProviderTransportError(f"model transport interrupted: {type(exc).__name__}") from exc
            if status in {401, 403}:
                raise ProviderAuthError(status, _safe_error(body))
            if status not in {429, 500, 502, 503, 504} or attempt >= self._max_retries:
                raise ProviderHTTPError(status, _safe_error(body))
            wait = _retry_after(retry_after)
            if wait is None:
                wait = min(8.0, 0.5 * 2**attempt) + random.uniform(0, 0.25)
            await asyncio.sleep(min(wait, 60.0))
        raise AssertionError("unreachable")

    async def _throttle(self) -> None:
        if self._min_request_interval_s <= 0:
            return
        async with self._rate_lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_request_at = time.monotonic() + self._min_request_interval_s


def _safe_error(body: bytes) -> str:
    try:
        raw = json.loads(body)
        error = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return error["code"]
    except (ValueError, TypeError):
        pass
    # Gateway responses may echo secrets or user prompts; no raw body in logs.
    return "request rejected"


async def _read_stream(
    response: httpx.Response,
    *,
    on_text: Callable[[str], Awaitable[None] | None] | None = None,
) -> ChatTurn:
    content: list[str] = []
    reasoning: list[str] = []
    reasoning_seen = False
    call_chunks: dict[int, dict[str, str]] = {}
    usage: Any = None
    finish_reason: str | None = None
    model: str | None = None
    response_id: str | None = None
    done = False
    event_lines: list[str] = []

    def consume_event(lines: list[str]) -> tuple[str, ...]:
        nonlocal usage, finish_reason, model, response_id, done, reasoning_seen
        data_lines = [line[5:].lstrip(" ") for line in lines if line.startswith("data:")]
        if not data_lines:
            return ()
        datum = "\n".join(data_lines)
        if datum == "[DONE]":
            done = True
            return ()
        try:
            chunk = json.loads(datum)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("invalid streaming JSON") from exc
        if not isinstance(chunk, dict):
            raise ProviderProtocolError("stream chunk is not an object")
        if isinstance(chunk.get("model"), str):
            model = chunk["model"]
        if isinstance(chunk.get("id"), str):
            response_id = chunk["id"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or len(choices) > 1:
            raise ProviderProtocolError("invalid streaming choices")
        text_fragments: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise ProviderProtocolError("invalid streaming choice")
            if choice.get("finish_reason") is not None:
                if not isinstance(choice["finish_reason"], str):
                    raise ProviderProtocolError("invalid streaming finish reason")
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ProviderProtocolError("invalid streaming delta")
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
                text_fragments.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_seen = True
                reasoning.append(delta["reasoning_content"])
            raw_calls = delta.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise ProviderProtocolError("invalid streamed tool_calls")
            for call in raw_calls:
                if not isinstance(call, dict) or type(call.get("index")) is not int:
                    raise ProviderProtocolError("streamed tool call lacks index")
                slot = call_chunks.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                if isinstance(call.get("id"), str):
                    slot["id"] += call["id"]
                fn = call.get("function") or {}
                if not isinstance(fn, dict):
                    raise ProviderProtocolError("invalid streamed function")
                if isinstance(fn.get("name"), str):
                    slot["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["arguments"] += fn["arguments"]
        return tuple(text_fragments)

    async def deliver(fragments: tuple[str, ...]) -> None:
        if on_text is None:
            return
        for fragment in fragments:
            callback_result = on_text(fragment)
            if inspect.isawaitable(callback_result):
                await callback_result

    try:
        async for line in response.aiter_lines():
            if line == "":
                await deliver(consume_event(event_lines))
                event_lines = []
                if done:
                    break
            else:
                event_lines.append(line)
    except httpx.RequestError as exc:
        raise ProviderProtocolError("model stream interrupted") from exc
    if event_lines and not done:
        await deliver(consume_event(event_lines))
    if not done:
        raise ProviderProtocolError("model stream ended without [DONE]")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if reasoning_seen:
        message["reasoning_content"] = "".join(reasoning)
    if call_chunks:
        message["tool_calls"] = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(call_chunks.items())
        ]
    return _parse_turn(
        {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
            "model": model,
            "id": response_id,
        }
    )
