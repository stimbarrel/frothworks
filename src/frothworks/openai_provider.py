import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

import httpx2
import openai
from openai.types.websocket_connection_options import WebSocketConnectionOptions

from .catalog import MAX_OUTPUT_TOKENS, OPENAI_COMPACT_FLOOR, ModelSpec
from .events import (
    Compaction,
    ContextUsage,
    Event,
    ReasoningSummary,
    Refusal,
    Text,
    ToolCall,
    Usage,
)
from .protocol import (
    ABORTED_TOOL_RESULT,
    COMPACTION_RESUME_TEXT,
    CONTINUATION_TEXT,
    FAILED_TOOL_MARKER,
    INTERRUPTION_MARKER,
    STEER_PREAMBLE,
    PendingCall,
)
from .retry import RequestFailed, classify_status, parse_retry_after, scrape_delay
from .tools import Tool

logger = logging.getLogger("frothworks")

# Wedge bound, not a liveness check: the WS keepalive detects a dead socket
# within ~40s, and a healthy pro-mode request can be silent for tens of
# minutes (live-verified 1,197s), which is exactly what trips Codex's 300s
# idle timer.
_WIRE_SILENCE_LIMIT = 3600.0
# Only `read` deviates from the SDK's Timeout(600, connect=5.0): 600s is too
# short for pro-mode silences.
_SSE_TIMEOUT = httpx2.Timeout(600.0, connect=5.0, read=_WIRE_SILENCE_LIMIT)
# The client-side ping is the only dead-socket detection (the server never
# pings first). The ping keys are absent from the SDK's TypedDict but are
# splatted into websockets.connect() at runtime; they restate the websockets
# defaults. max_size raises the websockets 1 MiB default to tungstenite's
# 64 MiB: one `response.output_item.done` frame carries a complete item, and
# an oversized frame fails with close code 1009 on every retry.
_WS_CONNECTION_OPTIONS = cast(
    WebSocketConnectionOptions,
    {"ping_interval": 20, "ping_timeout": 20, "max_size": 64 * 2**20},
)
# Prewarm holds the socket lock, so it must never make the first real
# request wait long.
_PREWARM_TIMEOUT = 10.0
_TERMINAL_EVENTS = ("response.completed", "response.failed", "response.incomplete")
# WS error codes that mean "server-side state is gone; redial and resend everything".
_WS_RETRYABLE_CODES = (
    "websocket_connection_limit_reached",
    "previous_response_not_found",
)
# response.failed codes that must fail fast, never retry (Codex's taxonomy).
_FATAL_RESPONSE_CODES = (
    "insufficient_quota",
    "invalid_prompt",
    "usage_not_included",
    "cyber_policy",
    "bio_policy",
    "misalignment_policy_violation",
)


def _strip_none(obj: Any) -> Any:
    """Recursively drop ``None``-valued dict keys (``status: None`` on resent
    reasoning items is a 400)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


class OpenAIProvider:
    """One session's OpenAI state: mirror, WS/SSE transport, stream parsing.

    WebSocket-first with a one-way SSE fallback (Codex policy); history is
    stateless full-resend except for the ``previous_response_id`` fast path.
    The mirror is the API's native Responses item list kept verbatim,
    sanitized only by dropping ``None``-valued keys.
    """

    def __init__(
        self,
        spec: ModelSpec,
        effort: str,
        pro: bool,
        auto_compact_limit: int,
        system_prompt: str,
        tools: Sequence[Tool],
        prompt_cache_key: str | None = None,
    ) -> None:
        self._spec = spec
        self._effort = effort
        self._pro = pro
        self._compact_threshold = max(auto_compact_limit, OPENAI_COMPACT_FLOOR)
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self.prompt_cache_key = prompt_cache_key or f"frothworks-{uuid.uuid4()}"
        self._client = openai.AsyncOpenAI(max_retries=0)
        self._items: list[dict] = []

        # WS transport state. `_ws_disabled` is the one-way session fallback
        # switch (Codex policy): once flipped, SSE for the rest of the session.
        self._ws_disabled = False
        self._ws_conn: Any = None
        self._ws_lock = asyncio.Lock()
        # previous_response_id fast path: valid only while the same socket
        # produced the previous response and history is a strict extension.
        self._baseline: list[dict] | None = None
        self._last_response_id: str | None = None
        self._last_config: dict | None = None

        # Per-attempt streaming state.
        self._turn_items: list[dict] = []
        self._pending_calls: list[PendingCall] = []
        self._committed = False
        self.last_stop_reason: str | None = None

    def append_user_text(self, text: str) -> None:
        self._items.append(
            {"role": "user", "content": [{"type": "input_text", "text": text}]}
        )

    def _append_developer(self, text: str) -> None:
        self._items.append(
            {"role": "developer", "content": [{"type": "input_text", "text": text}]}
        )

    def append_tool_results(self, results: Sequence[tuple[str, str, bool]]) -> None:
        for call_id, content, is_error in results:
            # No native error flag on function_call_output: mark errors
            # in-band. Synthesized aborted results are already SDK-authored
            # text and need no second marker.
            if not is_error or content == ABORTED_TOOL_RESULT:
                output = content
            else:
                output = f"{FAILED_TOOL_MARKER}\n{content}"
            self._items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )

    def append_steer(self, texts: Sequence[str]) -> None:
        self._append_developer(STEER_PREAMBLE + "\n\n" + "\n\n".join(texts))

    def append_continuation(self) -> None:
        self._append_developer(CONTINUATION_TEXT)

    def append_compaction_resume(self) -> None:
        self._append_developer(COMPACTION_RESUME_TEXT)

    def append_interruption_marker(self) -> None:
        self._append_developer(INTERRUPTION_MARKER)

    def serialize_items(self) -> list[dict]:
        return self._items

    def load_items(self, items: list[dict]) -> None:
        self._items = items

    def referenced_tool_names(self) -> set[str]:
        return {
            item["name"]
            for item in self._items
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and isinstance(item.get("name"), str)
        }

    async def aclose(self) -> None:
        self._drop_socket()
        await self._client.close()

    def _tool_params(self) -> list[dict]:
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
                "strict": False,
            }
            for t in self._tools
        ]

    def _config(self) -> dict:
        reasoning: dict = {
            "effort": self._effort,
            "mode": "pro" if self._pro else "standard",
            "summary": "auto",
        }
        cfg: dict = {
            "model": self._spec.id,
            "instructions": self._system_prompt,
            "reasoning": reasoning,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": "24h",
            "context_management": [
                {"type": "compaction", "compact_threshold": self._compact_threshold}
            ],
        }
        if self._tools:
            cfg["tools"] = self._tool_params()
            cfg["parallel_tool_calls"] = False
        return cfg

    def _build_body(self) -> dict:
        """Full request body; uses the previous_response_id fast path when the
        strict conditions hold, else a full stateless resend from the mirror."""
        cfg = self._config()
        body = dict(cfg)
        if (
            self._baseline is not None
            and self._last_response_id is not None
            and self._ws_conn is not None
            and self._last_config == cfg
            and len(self._items) >= len(self._baseline)
            and self._items[: len(self._baseline)] == self._baseline
        ):
            body["previous_response_id"] = self._last_response_id
            body["input"] = self._items[len(self._baseline) :]
        else:
            body["input"] = list(self._items)
        return body

    def can_fallback_transport(self) -> bool:
        return not self._ws_disabled

    def activate_sse_fallback(self) -> None:
        """One-way switch to SSE for the rest of the session."""
        if not self._ws_disabled:
            logger.warning("frothworks: falling back from WebSocket to SSE transport")
        self._ws_disabled = True
        self._drop_socket()

    def _drop_socket(self) -> None:
        conn = self._ws_conn
        self._ws_conn = None
        self._baseline = None
        self._last_response_id = None
        if conn is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(conn.close())
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())

    async def _dial(self) -> Any:
        """Dial a fresh WS connection. 426 flips the session to SSE silently."""
        from websockets.exceptions import InvalidStatus

        try:
            conn = await self._client.responses.connect(
                websocket_connection_options=_WS_CONNECTION_OPTIONS
            ).enter()
        except InvalidStatus as e:
            status = e.response.status_code
            if status == 426:
                self.activate_sse_fallback()
                return None
            raise RequestFailed(
                "connection",
                f"websocket handshake rejected: HTTP {status}",
                retryable=True,
            ) from e
        except RequestFailed:
            raise
        except Exception as e:
            raise RequestFailed(
                "connection", f"websocket connect failed: {e}", retryable=True
            ) from e
        return conn

    async def prewarm(self) -> None:
        """Upload the static prefix over a fresh socket (``generate: false``)
        so the first real request can ride the previous_response_id fast path.

        Best-effort and time-boxed: any failure (or a prewarm slower than
        ``_PREWARM_TIMEOUT``) is swallowed and the session proceeds normally,
        lazily dialing on first send.
        """
        if self._ws_disabled:
            return
        async with self._ws_lock:
            if self._ws_conn is not None:
                return
            try:
                await asyncio.wait_for(self._prewarm_once(), _PREWARM_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                logger.debug("frothworks: websocket prewarm failed: %s", e)
                self._drop_socket()

    async def _prewarm_once(self) -> None:
        conn = await self._dial()
        if conn is None:
            return
        # Stored immediately so every failure path below closes it
        # via _drop_socket.
        self._ws_conn = conn
        cfg = self._config()
        frame = {"type": "response.create", **cfg, "input": [], "generate": False}
        self._validate_frame(frame)
        await conn.send_raw(json.dumps(frame))
        while True:
            ev = json.loads(await conn.recv_bytes())
            etype = ev.get("type")
            if etype == "response.completed":
                response = ev.get("response") or {}
                self._baseline = []
                self._last_response_id = response.get("id")
                self._last_config = cfg
                return
            if etype in _TERMINAL_EVENTS or etype == "error":
                # A failed/incomplete prewarm must never seed the prev-id
                # fast path.
                raise RequestFailed("api", str(ev), retryable=False)

    @staticmethod
    def _validate_frame(frame: dict) -> None:
        # A malformed frame poisons the socket (server closes with code 1000,
        # outside the reconnectable set) — validate before sending.
        if not frame.get("model"):
            raise RequestFailed(
                "internal", "websocket frame missing required 'model'", retryable=False
            )
        json.dumps(frame)

    async def stream_once(self) -> AsyncGenerator[Event, None]:
        """Run one model request over WS (or SSE after fallback) and yield
        semantic events. Raises :class:`RequestFailed` on failure."""
        self._turn_items = []
        self._pending_calls = []
        self._committed = False
        self.last_stop_reason = None

        if self._pro:
            usage_preflight = await self._preflight_count()
            if usage_preflight is not None:
                yield ContextUsage(
                    used=usage_preflight, limit=self._spec.context_window
                )

        if not self._ws_disabled:
            async for ev in self._stream_ws():
                yield ev
            if not self._ws_disabled:
                return
            # A 426 during dial flipped us to SSE mid-call; fall through.
        async for ev in self._stream_sse():
            yield ev

    async def _preflight_count(self) -> int | None:
        """Free, exact contextual token count for the pro-mode gauge."""
        try:
            kwargs: dict = {
                "model": self._spec.id,
                "input": list(self._items),
                "instructions": self._system_prompt,
            }
            if self._tools:
                kwargs["tools"] = self._tool_params()
            result = await self._client.responses.input_tokens.count(**kwargs)
            return result.input_tokens
        except Exception as e:  # noqa: BLE001
            logger.debug("frothworks: input_tokens preflight failed: %s", e)
            return None

    async def _stream_ws(self) -> AsyncGenerator[Event, None]:
        async with self._ws_lock:
            terminal = False
            body: dict | None = None
            try:
                if self._ws_conn is None:
                    conn = await self._dial()
                    if conn is None:  # 426 -> SSE fallback
                        return
                    self._ws_conn = conn
                body = self._build_body()
                frame = {"type": "response.create", **body}
                self._validate_frame(frame)
                try:
                    await self._ws_conn.send_raw(json.dumps(frame))
                except Exception:  # noqa: BLE001
                    # Idle sockets die server-side; redial once inline.
                    self._drop_socket()
                    conn = await self._dial()
                    if conn is None:
                        return
                    self._ws_conn = conn
                    body = self._build_body()
                    frame = {"type": "response.create", **body}
                    try:
                        await self._ws_conn.send_raw(json.dumps(frame))
                    except Exception as e:
                        raise RequestFailed(
                            "connection",
                            f"websocket send failed after redial: {e}",
                            retryable=True,
                        ) from e

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            self._ws_conn.recv_bytes(), _WIRE_SILENCE_LIMIT
                        )
                    except TimeoutError as e:
                        raise RequestFailed(
                            "timeout",
                            "no data on the websocket for an hour; "
                            "the request is presumed wedged",
                            retryable=True,
                        ) from e
                    except Exception as e:
                        raise RequestFailed(
                            "connection",
                            f"websocket closed before response completed: {e}",
                            retryable=True,
                        ) from e
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug("skipping unparseable websocket frame")
                        continue
                    etype = ev.get("type", "")
                    if etype == "error":
                        raise self._classify_error_frame(ev)
                    for out in self._handle_event(ev):
                        yield out
                    if etype in _TERMINAL_EVENTS:
                        terminal = True
                        if etype == "response.completed":
                            response = ev.get("response") or {}
                            self._baseline = list(self._items)
                            self._last_response_id = response.get("id")
                            self._last_config = {
                                k: v
                                for k, v in (body or {}).items()
                                if k not in ("input", "previous_response_id")
                            }
                        else:
                            # incomplete/failed: the socket's response state
                            # is unknown — never chain prev-id through it.
                            self._baseline = None
                            self._last_response_id = None
                        return
            finally:
                if not terminal:
                    # Aborted or failed mid-response: the socket cannot be
                    # reused and the prev-id chain is dead.
                    self._drop_socket()

    async def _stream_sse(self) -> AsyncGenerator[Event, None]:
        body = self._build_body()
        try:
            stream = await self._client.responses.create(**body, timeout=_SSE_TIMEOUT)
        except Exception as e:
            raise self._classify(e) from e
        terminal = False
        try:
            # Drain to natural exhaustion (the server closes right after the
            # terminal event) — abandoning the body mid-stream leaves httpcore
            # asyncgen-finalization noise behind.
            async with stream:
                async for event in stream:
                    ev = event.to_dict()
                    if ev.get("type") == "error":
                        raise self._classify_error_frame(ev)
                    for out in self._handle_event(ev):
                        yield out
                    if ev.get("type") in _TERMINAL_EVENTS:
                        terminal = True
            if not terminal:
                raise RequestFailed(
                    "connection",
                    "stream closed before response completed",
                    retryable=True,
                )
        except RequestFailed:
            raise
        except Exception as e:
            raise self._classify(e) from e

    def _handle_event(self, ev: dict) -> list[Event]:
        etype = ev.get("type", "")
        out: list[Event] = []
        if etype == "response.output_item.done":
            item = _strip_none(ev.get("item") or {})
            out.extend(self._handle_item(item))
        elif etype == "response.completed":
            response = ev.get("response") or {}
            self.last_stop_reason = "end_turn"
            self._commit_turn_items()
            out.extend(self._usage_events(response))
        elif etype == "response.incomplete":
            response = ev.get("response") or {}
            details = response.get("incomplete_details") or {}
            self.last_stop_reason = (
                "max_tokens"
                if details.get("reason") == "max_output_tokens"
                else details.get("reason", "incomplete")
            )
            self._commit_turn_items()
            out.extend(self._usage_events(response))
        elif etype == "response.failed":
            response = ev.get("response") or {}
            error = response.get("error") or {}
            raise self._classify_response_error(error)
        return out

    def _handle_item(self, item: dict) -> list[Event]:
        if not item.get("type") and not item.get("role"):
            # A malformed/empty item must never enter the durable mirror.
            logger.debug("skipping malformed output item: %r", item)
            return []
        itype = item.get("type", "message")
        out: list[Event] = []
        if itype == "message":
            self._turn_items.append(item)
            text_parts = []
            for part in item.get("content", ()):
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    out.append(Refusal(category=None, explanation=part.get("refusal")))
            phase = "final" if item.get("phase") == "final_answer" else "commentary"
            if text_parts:
                out.append(Text(text="".join(text_parts), phase=phase))
        elif itype == "reasoning":
            self._turn_items.append(item)
            for part in item.get("summary", ()):
                text = part.get("text", "")
                if text:
                    out.append(ReasoningSummary(text=text))
        elif itype == "function_call":
            self._turn_items.append(item)
            raw_args = item.get("arguments", "")
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
            call = PendingCall(
                id=item.get("call_id", ""), name=item.get("name", ""), args=args
            )
            self._pending_calls.append(call)
            out.append(ToolCall(id=call.id, name=call.name, args=call.args))
        elif itype == "compaction":
            # Opaque encrypted compaction state; round-trips verbatim.
            self._turn_items.append(item)
            out.append(Compaction(summary=None))
        else:
            self._turn_items.append(item)
            logger.debug("keeping unknown output item type %r in mirror", itype)
        return out

    def _usage_events(self, response: dict) -> list[Event]:
        usage = response.get("usage") or {}
        if not usage:
            return []
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        events: list[Event] = [
            Usage(
                input_tokens=usage.get("input_tokens", 0) or 0,
                output_tokens=usage.get("output_tokens", 0) or 0,
                cache_read=input_details.get("cached_tokens", 0) or 0,
                cache_write=input_details.get("cache_write_tokens", 0) or 0,
                cache_write_5m=0,
                cache_write_1h=0,
                thinking_tokens=output_details.get("reasoning_tokens", 0) or 0,
            )
        ]
        if not self._pro:
            # Standard mode: billed input == contextual input (verified);
            # pro sessions already emitted the preflight gauge.
            events.append(
                ContextUsage(
                    used=usage.get("input_tokens", 0) or 0,
                    limit=self._spec.context_window,
                )
            )
        return events

    def _commit_turn_items(self) -> None:
        if self._committed:
            return
        self._committed = True
        self._items.extend(self._turn_items)

    def commit_aborted(self) -> tuple[list[Event], list[PendingCall]]:
        """Commit completed items of an aborted/failed stream to the mirror."""
        self._commit_turn_items()
        pending = list(self._pending_calls)
        self._pending_calls = []
        return [], pending

    def take_pending_calls(self) -> list[PendingCall]:
        """Hand the unanswered calls of the completed response to the caller,
        clearing them so a stale read can never settle them twice."""
        pending = list(self._pending_calls)
        self._pending_calls = []
        return pending

    @property
    def has_committed_blocks(self) -> bool:
        return bool(self._turn_items)

    def _classify_error_frame(self, ev: dict) -> RequestFailed:
        # WS error frames nest under "error"; SSE error events are flat.
        nested = ev.get("error")
        error: dict = nested if isinstance(nested, dict) else ev
        code = error.get("code") or ""
        message = error.get("message") or str(error)
        status = ev.get("status") or error.get("status")
        if code in _WS_RETRYABLE_CODES:
            # Server-side socket state is gone; redial and resend everything.
            self._drop_socket()
            return RequestFailed("connection", message, retryable=True)
        if code == "context_length_exceeded":
            return RequestFailed("context_overflow", message, retryable=False)
        if code in _FATAL_RESPONSE_CODES:
            return RequestFailed("api", message, retryable=False)
        # SSE error events carry no status: they ride an otherwise-200 stream,
        # so classify them the way in-stream errors are classified everywhere.
        return classify_status(
            status if isinstance(status, int) else 200, message, None
        )

    def _classify_response_error(self, error: dict) -> RequestFailed:
        code = error.get("code", "")
        message = error.get("message") or str(error)
        if code == "context_length_exceeded":
            return RequestFailed("context_overflow", message, retryable=False)
        if code in _FATAL_RESPONSE_CODES:
            return RequestFailed("api", message, retryable=False)
        return RequestFailed(
            "server", message, retryable=True, server_delay=scrape_delay(message)
        )

    def _classify(self, exc: Exception) -> RequestFailed:
        if isinstance(exc, RequestFailed):
            return exc
        if isinstance(exc, openai.APITimeoutError):
            return RequestFailed("timeout", str(exc), retryable=True)
        if isinstance(exc, openai.APIConnectionError):
            return RequestFailed("connection", str(exc), retryable=True)
        if isinstance(exc, openai.APIStatusError):
            status = exc.status_code
            message = str(exc)
            body = exc.body
            code = ""
            if isinstance(body, dict):
                code = str(body.get("code") or "")
                message = str(body.get("message") or message)
            if code == "context_length_exceeded" or "context window" in message:
                return RequestFailed("context_overflow", message, retryable=False)
            retry_after = parse_retry_after(
                exc.response.headers.get("retry-after")
                if exc.response is not None
                else None
            )
            return classify_status(status, message, retry_after)
        # The openai SDK rides httpx2 and does NOT wrap transport failures
        # that happen after the response headers arrive.
        if isinstance(exc, httpx2.TimeoutException):
            return RequestFailed("timeout", str(exc), retryable=True)
        if isinstance(exc, httpx2.HTTPError):
            return RequestFailed("connection", str(exc), retryable=True)
        logger.exception("unexpected error in openai stream")
        return RequestFailed("internal", str(exc), retryable=False)
