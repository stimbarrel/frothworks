import json
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import anthropic
import httpx

from .catalog import (
    ANTHROPIC_BETAS,
    ANTHROPIC_COMPACT_FLOOR,
    MAX_OUTPUT_TOKENS,
    ModelSpec,
)
from .events import (
    Compaction,
    CompactionFailed,
    ContextUsage,
    Event,
    ModelFallback,
    ReasoningSummary,
    Refusal,
    Text,
    ToolCall,
    Usage,
)
from .protocol import (
    COMPACTION_RESUME_TEXT,
    CONTINUATION_TEXT,
    INTERRUPTION_MARKER,
    STEER_PREAMBLE,
    PendingCall,
)
from .retry import RequestFailed, classify_status, parse_retry_after
from .tools import Tool

logger = logging.getLogger("frothworks")

# The cache hit lookback window is 20 content blocks; keep every breakpoint
# gap comfortably inside it.
_BURST_GUARD_GAP = 15
_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


class AnthropicProvider:
    """One session's Anthropic state: mirror, transport, and stream parsing.

    The mirror is the API's native ``messages`` array kept verbatim
    (thinking blocks with signatures, tool_use/tool_result pairs, compaction
    blocks). Requests send the mirror as-is, plus transient
    ``cache_control`` breakpoints added at build time and never stored.
    """

    def __init__(
        self,
        spec: ModelSpec,
        effort: str,
        auto_compact_limit: int,
        system_prompt: str,
        tools: Sequence[Tool],
    ) -> None:
        self._spec = spec
        self._effort = effort
        self._compact_trigger = max(auto_compact_limit, ANTHROPIC_COMPACT_FLOOR)
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self._client = anthropic.AsyncAnthropic(max_retries=0)
        self._messages: list[dict] = []
        self._cached_through = 0
        self._pending_cached_through = 0

        # Per-attempt streaming state, reset at the top of stream_once().
        self._turn_blocks: list[dict] = []
        self._pending_calls: list[PendingCall] = []
        self._held_text: str | None = None
        self._committed = False
        self.last_stop_reason: str | None = None

    def append_user_text(self, text: str) -> None:
        self._messages.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

    def append_tool_results(self, results: Sequence[tuple[str, str, bool]]) -> None:
        blocks: list[dict] = []
        for call_id, content, is_error in results:
            block: dict = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        self._messages.append({"role": "user", "content": blocks})

    def append_steer(self, texts: Sequence[str]) -> None:
        body = STEER_PREAMBLE + "\n\n" + "\n\n".join(texts)
        self.append_user_text(body)

    def append_continuation(self) -> None:
        self.append_user_text(CONTINUATION_TEXT)

    def append_compaction_resume(self) -> None:
        self.append_user_text(COMPACTION_RESUME_TEXT)

    def append_interruption_marker(self) -> None:
        self.append_user_text(INTERRUPTION_MARKER)

    def serialize_items(self) -> list[dict]:
        return self._messages

    def load_items(self, items: list[dict]) -> None:
        self._messages = items

    def referenced_tool_names(self) -> set[str]:
        names: set[str] = set()
        for message in self._messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name")
                    if isinstance(name, str):
                        names.add(name)
        return names

    async def aclose(self) -> None:
        await self._client.close()

    def can_fallback_transport(self) -> bool:
        return False

    def activate_sse_fallback(self) -> None:
        raise AssertionError("anthropic provider has no transport fallback")

    def _tool_params(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools
        ]

    def _build_request(self) -> dict:
        messages = [dict(m) for m in self._messages]
        # Rolling breakpoint on the final content block. The API rejects
        # assistant prefill, so the request always ends with a user message
        # and the breakpoint never lands on a thinking block.
        last = dict(messages[-1])
        blocks = [dict(b) for b in last["content"]]
        blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
        total_blocks = sum(len(m["content"]) for m in messages)
        # Burst guard: if this request appends more blocks than the lookback
        # window can bridge, place intermediate breakpoints so the previous
        # cached prefix stays findable. The API allows 4 breakpoints; the
        # system prompt and the rolling one use 2.
        guard_indices: list[int] = []
        gap_start = self._cached_through
        while (
            total_blocks - 1 - gap_start > _BURST_GUARD_GAP and len(guard_indices) < 2
        ):
            guard_indices.append(gap_start + _BURST_GUARD_GAP)
            gap_start += _BURST_GUARD_GAP
        if guard_indices:
            messages = self._apply_guard_breakpoints(messages, guard_indices)
            last = dict(messages[-1])
            blocks = [dict(b) for b in last["content"]]
            blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
        last["content"] = blocks
        messages[-1] = last
        # Committed only once the server accepts the request (message_start):
        # a retry of a never-processed attempt must re-place the same guards.
        self._pending_cached_through = total_blocks

        params: dict = {
            "model": self._spec.id,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": _CACHE_CONTROL,
                }
            ],
            "messages": messages,
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": self._compact_trigger,
                        },
                    }
                ]
            },
            "betas": list(ANTHROPIC_BETAS),
            "stream": True,
        }
        if self._spec.fallbacks:
            params["fallbacks"] = [{"model": m} for m in self._spec.fallbacks]
        if self._effort == "off":
            params["thinking"] = {"type": "disabled"}
        else:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            params["output_config"] = {"effort": self._effort}
        if self._tools:
            params["tools"] = self._tool_params()
            params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return params

    def _apply_guard_breakpoints(
        self, messages: list[dict], indices: list[int]
    ) -> list[dict]:
        """Place breakpoints at the given global block indices, sliding each
        off thinking blocks (thinking cannot carry ``cache_control``)."""
        flat: list[tuple[int, int, dict]] = []
        for mi, m in enumerate(messages):
            for bi, b in enumerate(m["content"]):
                flat.append((mi, bi, b))
        out = [dict(m) for m in messages]
        contents = [[dict(b) for b in m["content"]] for m in messages]
        for target in indices:
            idx = min(target, len(flat) - 1)
            while idx > 0 and flat[idx][2].get("type") in (
                "thinking",
                "redacted_thinking",
            ):
                idx -= 1
            mi, bi, _ = flat[idx]
            contents[mi][bi] = {**contents[mi][bi], "cache_control": _CACHE_CONTROL}
        for mi, m in enumerate(out):
            m["content"] = contents[mi]
        return out

    async def stream_once(self) -> AsyncGenerator[Event, None]:
        """Run one model request and yield semantic events.

        Raises :class:`RequestFailed` on transport/API failure. Completed
        blocks accumulate in provider state so an aborted or failed stream
        can be committed via :meth:`commit_aborted`.
        """
        self._turn_blocks = []
        self._pending_calls = []
        self._held_text = None
        self._committed = False
        self.last_stop_reason = None

        params = self._build_request()
        requested = self._spec.id
        served: str | None = None
        fallback_from_models: set[str] = set()
        first_fallback_from: str | None = None
        input_usage: Any = None
        final_usage: Any = None
        block: dict | None = None  # accumulator for the open content block

        try:
            stream = await self._client.beta.messages.create(**params)
        except Exception as e:
            raise self._classify(e) from e

        try:
            async with stream:
                async for event in stream:
                    etype = event.type
                    if etype == "message_start":
                        # The server accepted the request; the breakpoints we
                        # sent are now the cached position (see _build_request).
                        self._cached_through = self._pending_cached_through
                        input_usage = event.message.usage
                        served = event.message.model
                        used = (
                            (input_usage.input_tokens or 0)
                            + (input_usage.cache_read_input_tokens or 0)
                            + (input_usage.cache_creation_input_tokens or 0)
                        )
                        yield ContextUsage(used=used, limit=self._spec.context_window)
                    elif etype == "content_block_start":
                        if self._held_text is not None:
                            yield Text(text=self._held_text, phase="commentary")
                            self._held_text = None
                        block = self._open_block(event.content_block)
                    elif etype == "content_block_delta":
                        if block is not None:
                            self._accumulate(block, event.delta)
                    elif etype == "content_block_stop":
                        if block is not None:
                            for ev in self._close_block(block):
                                if isinstance(ev, ModelFallback):
                                    fallback_from_models.add(ev.from_model)
                                    if first_fallback_from is None:
                                        first_fallback_from = ev.from_model
                                yield ev
                            block = None
                    elif etype == "message_delta":
                        self.last_stop_reason = event.delta.stop_reason
                        final_usage = event.usage
                        if self._held_text is not None:
                            yield Text(text=self._held_text, phase="final")
                            self._held_text = None
                        if (
                            served is not None
                            and served != requested
                            and requested not in fallback_from_models
                        ):
                            # Pre-generation declines emit no boundary block;
                            # the requested model's hop is recoverable only
                            # from the served-model switch.
                            yield ModelFallback(
                                from_model=requested,
                                to_model=first_fallback_from or served,
                                category=None,
                            )
                        if self.last_stop_reason == "refusal":
                            details = getattr(event.delta, "stop_details", None)
                            yield Refusal(
                                category=getattr(details, "category", None),
                                explanation=getattr(details, "explanation", None),
                            )
                        yield self._merge_usage(input_usage, final_usage)
                    elif etype == "message_stop":
                        self._commit_turn_blocks()
            if not self._committed:
                if self.last_stop_reason is not None:
                    # The response was semantically complete at message_delta;
                    # only the trailing message_stop was lost. Committing here
                    # avoids re-asking a finished model to "continue".
                    self._commit_turn_blocks()
                else:
                    raise RequestFailed(
                        "connection",
                        "stream ended before message_stop",
                        retryable=True,
                    )
        except RequestFailed:
            raise
        except Exception as e:
            raise self._classify(e) from e

    def _open_block(self, content_block: Any) -> dict:
        state: dict = {"btype": content_block.type, "parts": [], "enc_parts": []}
        if content_block.type == "tool_use":
            state["id"] = content_block.id
            state["name"] = content_block.name
        elif content_block.type == "thinking":
            state["signature_parts"] = [content_block.signature or ""]
        elif content_block.type == "redacted_thinking":
            state["data"] = getattr(content_block, "data", "") or ""
        elif content_block.type == "fallback":
            state["fallback"] = {
                "from": content_block.from_.model,
                "to": content_block.to.model,
                "category": getattr(content_block.trigger, "category", None),
            }
        return state

    def _accumulate(self, state: dict, delta: Any) -> None:
        dtype = delta.type
        if dtype == "text_delta":
            state["parts"].append(delta.text)
        elif dtype == "thinking_delta":
            state["parts"].append(delta.thinking)
        elif dtype == "signature_delta":
            state.setdefault("signature_parts", []).append(delta.signature)
        elif dtype == "input_json_delta":
            state["parts"].append(delta.partial_json)
        elif dtype == "compaction_delta":
            if delta.content:
                state["parts"].append(delta.content)
            if getattr(delta, "encrypted_content", None):
                state["enc_parts"].append(delta.encrypted_content)

    def _close_block(self, state: dict) -> list[Event]:
        btype = state["btype"]
        joined = "".join(state["parts"])
        events: list[Event] = []
        if btype == "text":
            # Empty text blocks are rejected on resend ("text content blocks
            # must be non-empty") — never let one into the mirror.
            if joined:
                self._turn_blocks.append({"type": "text", "text": joined})
                self._held_text = joined
        elif btype == "thinking":
            signature = "".join(state.get("signature_parts", []))
            self._turn_blocks.append(
                {"type": "thinking", "thinking": joined, "signature": signature}
            )
            if joined:
                events.append(ReasoningSummary(text=joined))
        elif btype == "tool_use":
            try:
                args = json.loads(joined) if joined.strip() else {}
            except json.JSONDecodeError:
                args = {}
            self._turn_blocks.append(
                {
                    "type": "tool_use",
                    "id": state["id"],
                    "name": state["name"],
                    "input": args,
                }
            )
            call = PendingCall(id=state["id"], name=state["name"], args=args)
            self._pending_calls.append(call)
            events.append(ToolCall(id=call.id, name=call.name, args=call.args))
        elif btype == "compaction":
            encrypted = "".join(state["enc_parts"]) or None
            if joined:
                blk: dict = {"type": "compaction", "content": joined}
                if encrypted:
                    # Round-trip verbatim when present; omit entirely when null
                    # (sending the key with null is a 400).
                    blk["encrypted_content"] = encrypted
                self._turn_blocks.append(blk)
                events.append(Compaction(summary=joined))
            else:
                events.append(
                    CompactionFailed(
                        details="server compaction produced no content (likely a "
                        "classifier refusal); it will be re-attempted next request"
                    )
                )
        elif btype == "redacted_thinking":
            # Safety-redacted thinking must be passed back unchanged.
            self._turn_blocks.append(
                {"type": "redacted_thinking", "data": state.get("data", "")}
            )
        elif btype == "fallback":
            fb = state["fallback"]
            events.append(
                ModelFallback(
                    from_model=fb["from"], to_model=fb["to"], category=fb["category"]
                )
            )
        else:
            logger.debug("ignoring unknown content block type %r", btype)
        return events

    def _merge_usage(self, input_usage: Any, final_usage: Any) -> Usage:
        cache_creation = getattr(input_usage, "cache_creation", None)
        details = getattr(final_usage, "output_tokens_details", None)
        return Usage(
            input_tokens=getattr(final_usage, "input_tokens", None)
            or getattr(input_usage, "input_tokens", 0)
            or 0,
            output_tokens=getattr(final_usage, "output_tokens", 0) or 0,
            cache_read=getattr(input_usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(input_usage, "cache_creation_input_tokens", 0) or 0,
            cache_write_5m=getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0,
            cache_write_1h=getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0,
            thinking_tokens=getattr(details, "thinking_tokens", 0) or 0,
        )

    def _commit_turn_blocks(self) -> None:
        if self._committed:
            return
        self._committed = True
        if self._turn_blocks:
            self._messages.append({"role": "assistant", "content": self._turn_blocks})

    def commit_aborted(self) -> tuple[list[Event], list[PendingCall]]:
        """Commit completed blocks of an aborted/failed stream to the mirror.

        Returns events that were still held back (buffered commentary text)
        plus the committed-but-unanswered tool calls, which the caller must
        answer in the next message.
        """
        events: list[Event] = []
        if self._held_text is not None:
            events.append(Text(text=self._held_text, phase="commentary"))
            self._held_text = None
        self._commit_turn_blocks()
        pending = list(self._pending_calls)
        self._pending_calls = []
        return events, pending

    def take_pending_calls(self) -> list[PendingCall]:
        """Hand the unanswered calls of the completed response to the caller,
        clearing them so a stale read can never settle them twice."""
        pending = list(self._pending_calls)
        self._pending_calls = []
        return pending

    @property
    def has_committed_blocks(self) -> bool:
        return bool(self._turn_blocks)

    def _classify(self, exc: Exception) -> RequestFailed:
        if isinstance(exc, RequestFailed):
            return exc
        if isinstance(exc, anthropic.APITimeoutError):
            return RequestFailed("timeout", str(exc), retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return RequestFailed("connection", str(exc), retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            status = exc.status_code
            message = exc.message
            body = exc.body
            error_type = ""
            if isinstance(body, dict):
                inner = body.get("error")
                if isinstance(inner, dict):
                    error_type = inner.get("type") or ""
                    if isinstance(inner.get("message"), str):
                        message = inner["message"]
            retry_after = parse_retry_after(
                exc.response.headers.get("retry-after")
                if exc.response is not None
                else None
            )
            if error_type == "overloaded_error":
                return RequestFailed(
                    "overloaded", message, retryable=True, server_delay=retry_after
                )
            if status == 400 and "prompt is too long" in message:
                return RequestFailed("context_overflow", message, retryable=False)
            # In-stream error events surface with the stream's HTTP 200
            # status; classify_status treats those as retryable server errors.
            return classify_status(status, message, retry_after)
        if isinstance(exc, httpx.TimeoutException):
            return RequestFailed("timeout", str(exc), retryable=True)
        if isinstance(exc, httpx.HTTPError):
            return RequestFailed("connection", str(exc), retryable=True)
        logger.exception("unexpected error in anthropic stream")
        return RequestFailed("internal", str(exc), retryable=False)
