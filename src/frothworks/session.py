import asyncio
import contextlib
import copy
import inspect
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Literal, overload

from .anthropic_provider import AnthropicProvider
from .catalog import CATALOG, FableEffort, OpusEffort, SolEffort
from .errors import ConfigurationError, SerializationError, SessionStateError
from .events import Error, Event, ToolResult, TurnEnd
from .openai_provider import OpenAIProvider
from .prompt import build_system_prompt
from .protocol import ABORTED_TOOL_RESULT, PendingCall
from .retry import MAX_ATTEMPTS, RequestFailed, retry_delay_for
from .tools import Tool

logger = logging.getLogger("frothworks")

_STATE_VERSION = 1


class Session:
    """A model-locked, effort-locked conversation with tools.

    ``model``, ``effort``, and ``auto_compact_limit`` are required and
    immutable for the session's life. ``auto_compact_limit`` is the native
    compaction trigger in input tokens; it is clamped to each provider's
    floor (50,000 on Anthropic, 1,000 on OpenAI).
    """

    @overload
    def __init__(
        self,
        *,
        model: Literal["gpt-5.6-sol"],
        effort: SolEffort,
        auto_compact_limit: int,
        instructions: str,
        tools: Sequence[Tool] = (),
        guidelines: Sequence[str] = (),
        extra_sections: Sequence[tuple[str, str]] = (),
        pro: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: Literal["claude-fable-5"],
        effort: FableEffort,
        auto_compact_limit: int,
        instructions: str,
        tools: Sequence[Tool] = (),
        guidelines: Sequence[str] = (),
        extra_sections: Sequence[tuple[str, str]] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: Literal["claude-opus-5"],
        effort: OpusEffort,
        auto_compact_limit: int,
        instructions: str,
        tools: Sequence[Tool] = (),
        guidelines: Sequence[str] = (),
        extra_sections: Sequence[tuple[str, str]] = (),
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str,
        effort: str,
        auto_compact_limit: int,
        instructions: str,
        tools: Sequence[Tool] = (),
        guidelines: Sequence[str] = (),
        extra_sections: Sequence[tuple[str, str]] = (),
        pro: bool = False,
    ) -> None:
        tools = self._validate_config(model, effort, auto_compact_limit, tools, pro)
        if not isinstance(instructions, str) or not instructions.strip():
            raise ConfigurationError("instructions must be a non-empty string")
        if isinstance(guidelines, str):
            raise ConfigurationError("guidelines must be a sequence of strings")
        try:
            guidelines = list(guidelines)
            extra_sections = list(extra_sections)
        except TypeError as e:
            raise ConfigurationError(
                f"guidelines/extra_sections must be sequences: {e}"
            ) from e
        if not all(isinstance(g, str) for g in guidelines):
            raise ConfigurationError("guidelines must be strings")
        if not all(
            isinstance(s, tuple | list)
            and len(s) == 2
            and isinstance(s[0], str)
            and isinstance(s[1], str)
            for s in extra_sections
        ):
            raise ConfigurationError(
                "extra_sections must be (title, body) string pairs"
            )
        system_prompt = build_system_prompt(
            instructions, tools, guidelines, extra_sections
        )
        self._init_common(
            model=model,
            effort=effort,
            pro=pro,
            auto_compact_limit=auto_compact_limit,
            system_prompt=system_prompt,
            tools=tools,
            prompt_cache_key=None,
            items=None,
        )

    @staticmethod
    def _validate_config(
        model: str,
        effort: str,
        auto_compact_limit: int,
        tools: Sequence[Tool],
        pro: bool,
    ) -> list[Tool]:
        spec = CATALOG.get(model) if isinstance(model, str) else None
        if spec is None:
            raise ConfigurationError(
                f"unknown model {model!r}; choose one of {sorted(CATALOG)}"
            )
        if effort not in spec.efforts:
            raise ConfigurationError(
                f"model {model!r} does not support effort {effort!r}; "
                f"supported: {list(spec.efforts)}"
            )
        if not isinstance(pro, bool):
            raise ConfigurationError("pro must be a bool")
        if pro and spec.provider != "openai":
            raise ConfigurationError("pro=True is only supported on gpt-5.6-sol")
        if (
            not isinstance(auto_compact_limit, int)
            or isinstance(auto_compact_limit, bool)
            or auto_compact_limit <= 0
        ):
            raise ConfigurationError("auto_compact_limit must be a positive int")
        try:
            tool_list = list(tools)
        except TypeError as e:
            raise ConfigurationError(f"tools must be a sequence of Tool: {e}") from e
        names: set[str] = set()
        for t in tool_list:
            if not isinstance(t, Tool):
                raise ConfigurationError(
                    f"tools must be @tool-decorated handlers, got {t!r}"
                )
            if t.name in names:
                raise ConfigurationError(f"duplicate tool name {t.name!r}")
            names.add(t.name)
        if tool_list and effort == "off":
            # This combination has produced tool calls as garbled literal XML
            # text instead of tool_use blocks.
            raise ConfigurationError(
                'effort="off" cannot be combined with tools on claude-opus-5 '
                "(the API mishandles tool calls with thinking disabled); "
                'use effort="low" instead'
            )
        return tool_list

    def _init_common(
        self,
        *,
        model: str,
        effort: str,
        pro: bool,
        auto_compact_limit: int,
        system_prompt: str,
        tools: Sequence[Tool],
        prompt_cache_key: str | None,
        items: list[dict] | None,
    ) -> None:
        spec = CATALOG[model]
        self._spec = spec
        self._model = model
        self._effort = effort
        self._pro = pro
        self._auto_compact_limit = auto_compact_limit
        self._system_prompt = system_prompt
        self._tools_by_name = {t.name: t for t in tools}
        self._provider: AnthropicProvider | OpenAIProvider
        if spec.provider == "anthropic":
            self._provider = AnthropicProvider(
                spec, effort, auto_compact_limit, system_prompt, tools
            )
        else:
            self._provider = OpenAIProvider(
                spec,
                effort,
                pro,
                auto_compact_limit,
                system_prompt,
                tools,
                prompt_cache_key,
            )
        if items is not None:
            self._provider.load_items(items)

        self._running = False
        self._closed = False
        self._turn_gen: AsyncGenerator[Event, None] | None = None
        self._turn_token: object | None = None
        self._steers: list[str] = []
        self._abort: str | None = None
        self._active_task: asyncio.Task | None = None
        self._queue: asyncio.Queue | None = None
        self._phase: str | None = None
        # Tool calls in the mirror with no result yet; _turn's finally
        # guarantees each one gets a result even if the turn dies.
        self._unsettled: list[PendingCall] = []
        self._settled_records: list[tuple[str, str, bool]] = []
        self._prewarm_task: asyncio.Task | None = None
        if isinstance(self._provider, OpenAIProvider):
            with contextlib.suppress(RuntimeError):
                loop = asyncio.get_running_loop()
                self._prewarm_task = loop.create_task(self._provider.prewarm())

    @property
    def model(self) -> str:
        return self._model

    @property
    def effort(self) -> str:
        return self._effort

    @property
    def running(self) -> bool:
        """Whether a turn is in flight (including one returned by ``send()``
        but not yet iterated)."""
        self._release_closed_turn()
        return self._running

    def send(self, text: str) -> AsyncGenerator[Event, None]:
        """Start a turn. Returns an async generator of events; the turn runs
        as you iterate. The stream never raises — failures arrive as
        ``Error`` events and every turn ends with ``TurnEnd``.

        Must be driven from the session's event loop. Iterating through
        ``TurnEnd`` (or breaking on it) ends the turn cleanly. Abandoning the
        iterator earlier leaves the turn in flight: ``await it.aclose()`` on
        the returned generator to end it before starting another.
        """
        if self._closed:
            raise SessionStateError("send() called on a closed session")
        if not isinstance(text, str) or not text.strip():
            raise SessionStateError("send() requires a non-empty string")
        self._supersede_stale_turn()
        if self._running:
            raise SessionStateError("send() called while a turn is already running")
        self._running = True
        # A predecessor whose finally has not run yet (e.g. broken out of at
        # TurnEnd) must not leak steers or abort flags into this turn.
        self._steers = []
        self._abort = None
        token = object()
        self._turn_token = token
        gen = self._turn(text, token)
        self._turn_gen = gen
        return gen

    def _supersede_stale_turn(self) -> None:
        """Disown a previous turn whose iterator was never started (it did
        nothing) or is already closed. The disowned iterator becomes inert:
        iterating or finalizing it later cannot touch session state."""
        if not self._running or self._turn_gen is None:
            return
        if inspect.getasyncgenstate(self._turn_gen) in ("AGEN_CREATED", "AGEN_CLOSED"):
            self._release_turn()

    def _release_closed_turn(self) -> None:
        """Release a turn whose generator is closed for good — an unstarted
        turn that was ``aclose()``d never runs its own cleanup."""
        if not self._running or self._turn_gen is None:
            return
        if inspect.getasyncgenstate(self._turn_gen) == "AGEN_CLOSED":
            self._release_turn()

    def _release_turn(self) -> None:
        self._turn_token = None
        self._turn_gen = None
        self._running = False
        self._steers = []
        self._abort = None

    def steer(self, text: str) -> None:
        """Inject a new instruction into the running turn.

        Aborts the in-flight stream at the block boundary (a completed tool
        call is honored first), injects the instruction, and resends. Errors
        if no turn is running or the turn is already being interrupted. Must
        be called from the session's event loop.
        """
        if self._closed:
            raise SessionStateError("steer() called on a closed session")
        self._release_closed_turn()
        if not self._running:
            raise SessionStateError("steer() called while no turn is running")
        if self._abort == "interrupt":
            raise SessionStateError("steer() called on an interrupted turn")
        if not isinstance(text, str) or not text.strip():
            raise SessionStateError("steer() requires a non-empty string")
        self._steers.append(text)
        if self._phase == "stream" and self._active_task is not None:
            self._abort = "steer"
            self._active_task.cancel()
            if self._queue is not None:
                self._queue.put_nowait(("cancelled", None))

    def interrupt(self) -> None:
        """Kill the running turn. No-op when idle.

        Unanswered tool calls get synthesized aborted results so the
        transcript stays legal, a model-visible interruption marker is
        appended, and the turn ends with ``TurnEnd("interrupted")``. Must be
        called from the session's event loop.
        """
        self._release_closed_turn()
        if not self._running:
            return
        self._abort = "interrupt"
        if self._active_task is not None:
            self._active_task.cancel()
        if self._phase == "stream" and self._queue is not None:
            self._queue.put_nowait(("cancelled", None))

    async def aclose(self) -> None:
        """Release network resources. The session is unusable afterwards:
        ``send()`` and ``steer()`` raise once it has been closed."""
        self._closed = True
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
            with contextlib.suppress(BaseException):
                await self._prewarm_task
        if self._turn_gen is not None:
            with contextlib.suppress(Exception):
                await self._turn_gen.aclose()
        try:
            await self._provider.aclose()
        except Exception as e:  # noqa: BLE001
            # Pooled connections bound to an already-closed event loop
            # cannot be closed cleanly.
            logger.debug("frothworks: provider close failed: %s", e)

    def to_dict(self) -> dict:
        """Serialize the session to a JSON-safe dict. The consumer stores it
        anywhere; resume with :meth:`from_dict`. Errors mid-turn."""
        self._release_closed_turn()
        if self._running:
            raise SessionStateError("to_dict() called while a turn is running")
        return {
            "version": _STATE_VERSION,
            "provider": self._spec.provider,
            "model": self._model,
            "effort": self._effort,
            "pro": self._pro,
            "auto_compact_limit": self._auto_compact_limit,
            "system_prompt": self._system_prompt,
            "prompt_cache_key": (
                self._provider.prompt_cache_key
                if isinstance(self._provider, OpenAIProvider)
                else None
            ),
            "items": copy.deepcopy(self._provider.serialize_items()),
        }

    @classmethod
    def from_dict(cls, state: dict, *, tools: Sequence[Tool] = ()) -> "Session":
        """Resume a serialized session.

        Tools are re-supplied for handlers and API schemas only — the stored
        system prompt is reused byte-exact, never re-rendered. Raises
        :class:`SerializationError` if the dict is malformed or the supplied
        tools do not cover every tool name the history references.
        """
        if not isinstance(state, dict):
            raise SerializationError("state must be a dict produced by to_dict()")
        if state.get("version") != _STATE_VERSION:
            raise SerializationError(
                f"unsupported state version: {state.get('version')!r}"
            )
        model = state.get("model")
        if not isinstance(model, str) or model not in CATALOG:
            raise SerializationError(f"unknown model in state: {model!r}")
        spec = CATALOG[model]
        if state.get("provider") != spec.provider:
            raise SerializationError("state provider does not match its model")
        effort = state.get("effort")
        if not isinstance(effort, str) or effort not in spec.efforts:
            raise SerializationError(f"invalid effort in state: {effort!r}")
        system_prompt = state.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise SerializationError("state is missing its system_prompt")
        items = state.get("items")
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise SerializationError("state items must be a list of dicts")
        for i, item in enumerate(items):
            role = item.get("role")
            itype = item.get("type")
            if role is None and not isinstance(itype, str):
                raise SerializationError(f"state items[{i}] has no role or type")
            if role is not None and (
                not isinstance(role, str) or not isinstance(item.get("content"), list)
            ):
                raise SerializationError(
                    f"state items[{i}] must have a string role and list content"
                )
        limit = state.get("auto_compact_limit")
        if not isinstance(limit, int) or limit <= 0:
            raise SerializationError("state has an invalid auto_compact_limit")
        pro = state.get("pro", False)
        if not isinstance(pro, bool):
            raise SerializationError(f"state has an invalid pro flag: {pro!r}")
        cache_key = state.get("prompt_cache_key")
        if cache_key is not None and not isinstance(cache_key, str):
            raise SerializationError("state prompt_cache_key must be a string or null")
        try:
            tool_list = cls._validate_config(model, effort, limit, tools, pro)
        except ConfigurationError as e:
            raise SerializationError(str(e)) from e

        self = object.__new__(cls)
        self._init_common(
            model=model,
            effort=effort,
            pro=pro,
            auto_compact_limit=limit,
            system_prompt=system_prompt,
            tools=tool_list,
            prompt_cache_key=cache_key,
            items=copy.deepcopy(items),
        )
        referenced = self._provider.referenced_tool_names() - {""}
        missing = referenced - set(self._tools_by_name)
        if missing:
            raise SerializationError(
                f"history references tools not supplied to from_dict: {sorted(missing)}"
            )
        return self

    async def _turn(self, text: str, token: object) -> AsyncGenerator[Event, None]:
        try:
            if self._turn_token is not token:
                return  # superseded before it ever started; do nothing
            async for ev in self._turn_body(text, token):
                if isinstance(ev, TurnEnd) and self._turn_token is token:
                    # A consumer that breaks on TurnEnd must not find the
                    # session still "running", nor inherit this turn's steers
                    # or abort flag.
                    self._running = False
                    self._steers = []
                    self._abort = None
                yield ev
        finally:
            # A superseded generator's late finalization must not clobber a
            # newer turn's state.
            if self._turn_token is token:
                self._turn_token = None
                self._running = False
                self._turn_gen = None
                self._steers = []
                self._abort = None
                self._phase = None
                self._queue = None
                if self._active_task is not None:
                    self._active_task.cancel()
                    self._active_task = None
                # The turn may have died with unanswered tool calls, held by
                # the loop or still by the provider — settle both so the
                # mirror stays legal for the next request.
                _, stranded = self._provider.commit_aborted()
                known = {c.id for c in self._unsettled}
                self._unsettled.extend(c for c in stranded if c.id not in known)
                if self._unsettled:
                    self._flush_settlement(ABORTED_TOOL_RESULT)

    async def _turn_body(self, text: str, token: object) -> AsyncGenerator[Event, None]:
        provider = self._provider
        try:
            provider.append_user_text(text)
            attempt = 0
            while True:
                if self._abort == "interrupt":
                    async for ev in self._finish_interrupt():
                        yield ev
                    return
                if self._steers:
                    provider.append_steer(self._steers)
                    self._steers = []

                outcome, payload = "", None
                queue: asyncio.Queue = asyncio.Queue()
                gen = provider.stream_once()
                task = asyncio.ensure_future(self._pump(gen, queue))
                self._active_task = task
                self._queue = queue
                self._phase = "stream"
                try:
                    while True:
                        kind, payload = await queue.get()
                        if kind == "ev":
                            yield payload
                        else:
                            outcome = kind
                            break
                finally:
                    # Token-guarded: an abandoned turn's late finalization
                    # must not null a newer turn's stream bookkeeping.
                    if self._turn_token is token:
                        self._active_task = None
                        self._queue = None
                        self._phase = None
                    if outcome == "":
                        # The turn generator itself was cancelled/closed.
                        task.cancel()
                        await asyncio.wait({task}, timeout=0.1)

                if outcome == "cancelled":
                    # Let the pump finish aborting before reading provider state.
                    await asyncio.wait({task}, timeout=1.0)
                    reason = self._abort
                    events, pending = provider.commit_aborted()
                    self._begin_settlement(pending)
                    for ev in events:
                        yield ev
                    if reason == "interrupt":
                        async for ev in self._finish_interrupt():
                            yield ev
                        return
                    # Steer: honor a completed tool call, then resend with
                    # the queued steers injected at the top of the loop.
                    if self._abort == "steer":
                        self._abort = None
                    if pending:
                        async for ev in self._settle_calls(token):
                            yield ev
                        if self._abort == "interrupt":
                            async for ev in self._finish_interrupt():
                                yield ev
                            return
                    continue

                if outcome == "err":
                    failed: RequestFailed = payload
                    attempt += 1
                    retryable = failed.retryable
                    if retryable and attempt >= MAX_ATTEMPTS:
                        # The SSE fallback is for transport failures only.
                        if failed.kind in ("connection", "timeout") and (
                            provider.can_fallback_transport()
                        ):
                            provider.activate_sse_fallback()
                            attempt = 0
                        else:
                            retryable = False
                    delay = retry_delay_for(failed, max(attempt, 1))
                    if failed.retryable and delay is None:
                        retryable = False
                    if not retryable:
                        # Commit what streamed so partial output survives.
                        events, pending = provider.commit_aborted()
                        self._begin_settlement(pending)
                        for ev in events:
                            yield ev
                        for ev in self._flush_settlement(ABORTED_TOOL_RESULT):
                            yield ev
                        yield Error(
                            kind=failed.kind,
                            message=failed.message,
                            retryable=failed.retryable,
                        )
                        yield TurnEnd(stop_reason="error")
                        return
                    # Completed blocks survive a mid-stream failure; ask the
                    # model to continue rather than restarting the response.
                    if provider.has_committed_blocks:
                        events, pending = provider.commit_aborted()
                        self._begin_settlement(pending)
                        for ev in events:
                            yield ev
                        if pending:
                            async for ev in self._settle_calls(token):
                                yield ev
                            if self._abort == "interrupt":
                                async for ev in self._finish_interrupt():
                                    yield ev
                                return
                        else:
                            provider.append_continuation()
                    if await self._interruptible_sleep(delay or 0.0):
                        async for ev in self._finish_interrupt():
                            yield ev
                        return
                    continue

                # outcome == "done": the response completed normally.
                attempt = 0
                pending = provider.take_pending_calls()
                stop_reason = provider.last_stop_reason or "end_turn"
                self._begin_settlement(pending)
                if self._abort == "interrupt":
                    # The interrupt lost the race with response completion;
                    # honor it anyway.
                    async for ev in self._finish_interrupt():
                        yield ev
                    return
                if pending:
                    async for ev in self._settle_calls(token):
                        yield ev
                    if self._abort == "interrupt":
                        async for ev in self._finish_interrupt():
                            yield ev
                        return
                    continue
                if stop_reason == "refusal":
                    yield TurnEnd(stop_reason="refusal")
                    return
                if self._steers:
                    # A steer lost the race with response completion; honor it
                    # with one more request (injected at the top of the loop).
                    if self._abort == "steer":
                        self._abort = None
                    continue
                if stop_reason == "compaction":
                    # Defensive: we never request pause_after_compaction, but a
                    # paused compaction just means "resend on the compacted
                    # context" — unless it made no progress (refused).
                    if provider.has_committed_blocks:
                        provider.append_compaction_resume()
                        continue
                    yield Error(
                        kind="compaction_stalled",
                        message="compaction stop without a compaction block",
                        retryable=False,
                    )
                    yield TurnEnd(stop_reason="error")
                    return
                yield TurnEnd(stop_reason=stop_reason)
                return
        except GeneratorExit:
            raise
        except Exception as e:
            logger.exception("frothworks: unexpected error in turn loop")
            yield Error(kind="internal", message=str(e), retryable=False)
            yield TurnEnd(stop_reason="error")

    async def _pump(
        self, gen: AsyncGenerator[Event, None], queue: asyncio.Queue
    ) -> None:
        try:
            async for ev in gen:
                queue.put_nowait(("ev", ev))
            queue.put_nowait(("done", None))
        except RequestFailed as e:
            queue.put_nowait(("err", e))
        except asyncio.CancelledError:
            queue.put_nowait(("cancelled", None))
            raise
        except Exception as e:
            logger.exception("frothworks: unexpected provider stream error")
            queue.put_nowait(
                ("err", RequestFailed("internal", str(e), retryable=False))
            )
        finally:
            with contextlib.suppress(BaseException):
                await gen.aclose()

    def _begin_settlement(self, pending: Sequence[PendingCall]) -> None:
        self._unsettled = list(pending)
        self._settled_records = []

    def _flush_settlement(self, abort_content: str) -> list[Event]:
        """Append one result per unsettled call to the mirror — real results
        collected so far plus synthesized aborted results for the rest."""
        if not self._unsettled:
            return []
        events: list[Event] = []
        records = list(self._settled_records)
        done_ids = {r[0] for r in records}
        for call in self._unsettled:
            if call.id not in done_ids:
                records.append((call.id, abort_content, True))
                events.append(
                    ToolResult(
                        id=call.id,
                        content=abort_content,
                        is_error=True,
                        deliberate=False,
                    )
                )
        self._provider.append_tool_results(records)
        self._unsettled = []
        self._settled_records = []
        return events

    async def _settle_calls(self, token: object) -> AsyncGenerator[Event, None]:
        """Run the unsettled tool calls sequentially and append their results.

        On interrupt, the running call and any remaining ones get synthesized
        aborted results; ``self._abort`` stays ``"interrupt"``. The mirror is
        guaranteed to receive exactly one result per call.
        """
        for call in list(self._unsettled):
            if self._abort == "interrupt":
                break
            task = asyncio.ensure_future(self._execute_tool(call))
            self._active_task = task
            self._phase = "tool"
            try:
                content, is_error, deliberate = await task
            except asyncio.CancelledError:
                if self._abort != "interrupt":
                    # Foreign cancellation: _turn's finally settles the mirror.
                    raise
                await asyncio.wait({task}, timeout=0.1)
                break
            finally:
                # Token-guarded like _turn_body's stream finally.
                if self._turn_token is token:
                    self._active_task = None
                    self._phase = None
            self._settled_records.append((call.id, content, is_error))
            yield ToolResult(
                id=call.id, content=content, is_error=is_error, deliberate=deliberate
            )
        for ev in self._flush_settlement(ABORTED_TOOL_RESULT):
            yield ev

    async def _execute_tool(self, call: PendingCall) -> tuple[str, bool, bool]:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            return f"unknown tool: {call.name!r}", True, False
        return await tool.run(call.args)

    async def _finish_interrupt(self) -> AsyncGenerator[Event, None]:
        for ev in self._flush_settlement(ABORTED_TOOL_RESULT):
            yield ev
        self._provider.append_interruption_marker()
        yield TurnEnd(stop_reason="interrupted")

    async def _interruptible_sleep(self, delay: float) -> bool:
        """Sleep for ``delay`` seconds; returns True if interrupted."""
        task = asyncio.ensure_future(asyncio.sleep(delay))
        self._active_task = task
        self._phase = "sleep"
        try:
            await task
        except asyncio.CancelledError:
            if self._abort != "interrupt":
                raise
            return True
        finally:
            self._active_task = None
            self._phase = None
        return self._abort == "interrupt"

    def __repr__(self) -> str:
        return f"<frothworks.Session model={self._model!r} effort={self._effort!r}>"
