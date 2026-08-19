from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReasoningSummary:
    """A completed reasoning/thinking summary block."""

    text: str


@dataclass(frozen=True)
class Text:
    """A completed assistant text block.

    ``phase`` is ``"commentary"`` for intermediate narration (e.g. before a
    tool call) and ``"final"`` for the turn's answer. Native on OpenAI;
    derived on Anthropic (only a text block that ends the message is final).
    """

    text: str
    phase: Literal["commentary", "final"]


@dataclass(frozen=True)
class ToolCall:
    """The model requested a tool invocation."""

    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    """A tool invocation finished (or failed).

    ``deliberate`` is True when the tool raised ``ToolError`` on purpose,
    False for accidental failures (validation errors, unexpected exceptions).
    """

    id: str
    content: str
    is_error: bool
    deliberate: bool


@dataclass(frozen=True)
class Usage:
    """Token accounting for one completed model request, as the provider
    bills it — the fields mean what that provider means by them.

    In particular ``input_tokens`` is *uncached* input on Anthropic (the full
    prompt is ``input_tokens + cache_read + cache_write``) but the *whole*
    contextual input on OpenAI (cached tokens included). Use
    :class:`ContextUsage` for provider-independent context accounting.
    ``cache_write`` is the total cache write; the per-TTL split
    (``cache_write_5m`` / ``cache_write_1h``) is populated on Anthropic only.
    """

    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    cache_write_5m: int
    cache_write_1h: int
    thinking_tokens: int


@dataclass(frozen=True)
class ContextUsage:
    """Effective context consumption at a request boundary.

    ``used`` counts contextual tokens (never pro-mode billing inflation).
    """

    used: int
    limit: int


@dataclass(frozen=True)
class Compaction:
    """The server compacted the conversation history.

    ``summary`` carries the plaintext summary on Anthropic; OpenAI compaction
    state is opaque (encrypted), so ``summary`` is None there.
    """

    summary: str | None


@dataclass(frozen=True)
class CompactionFailed:
    """A server-side compaction attempt failed (e.g. classifier refusal).

    The compaction edit re-attempts automatically on every subsequent request.
    """

    details: str


@dataclass(frozen=True)
class ModelFallback:
    """The server answered with a fallback model instead of the requested one."""

    from_model: str
    to_model: str
    category: str | None


@dataclass(frozen=True)
class Refusal:
    """Terminal refusal: the requested model and its whole fallback chain declined."""

    category: str | None
    explanation: str | None


@dataclass(frozen=True)
class TurnEnd:
    """The turn is over. Always the last event of a turn."""

    stop_reason: str


@dataclass(frozen=True)
class Error:
    """A runtime failure the stream absorbed instead of raising."""

    kind: str
    message: str
    retryable: bool


Event = (
    ReasoningSummary
    | Text
    | ToolCall
    | ToolResult
    | Usage
    | ContextUsage
    | Compaction
    | CompactionFailed
    | ModelFallback
    | Refusal
    | TurnEnd
    | Error
)
"""Everything ``Session.send`` can yield. Each event is one complete semantic
unit; there are no raw deltas."""
