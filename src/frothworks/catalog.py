from dataclasses import dataclass
from typing import Literal

SolEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
FableEffort = Literal["low", "medium", "high", "xhigh", "max"]
OpusEffort = Literal["off", "low", "medium", "high", "xhigh", "max"]
Provider = Literal["anthropic", "openai"]

MAX_OUTPUT_TOKENS = 128_000

ANTHROPIC_COMPACT_FLOOR = 50_000
OPENAI_COMPACT_FLOOR = 1_000

ANTHROPIC_BETAS = ["server-side-fallback-2026-07-01", "compact-2026-01-12"]


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about one catalog model."""

    id: str
    provider: Provider
    efforts: tuple[str, ...]
    context_window: int
    fallbacks: tuple[str, ...]


CATALOG: dict[str, ModelSpec] = {
    "gpt-5.6-sol": ModelSpec(
        id="gpt-5.6-sol",
        provider="openai",
        efforts=("none", "low", "medium", "high", "xhigh", "max"),
        # Live-verified boundary: 921,603 input tokens accepted, 922,190 rejected.
        context_window=921_600,
        fallbacks=(),
    ),
    "claude-fable-5": ModelSpec(
        id="claude-fable-5",
        provider="anthropic",
        efforts=("low", "medium", "high", "xhigh", "max"),
        context_window=1_000_000,
        fallbacks=("claude-opus-5", "claude-opus-4-8"),
    ),
    "claude-opus-5": ModelSpec(
        id="claude-opus-5",
        provider="anthropic",
        efforts=("off", "low", "medium", "high", "xhigh", "max"),
        context_window=1_000_000,
        fallbacks=("claude-opus-4-8",),
    ),
}
"""The user-selectable models, live-verified against both APIs.
``claude-opus-4-8`` exists only as a fallback target."""
