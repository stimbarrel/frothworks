import email.utils
import random
import re
from datetime import UTC, datetime

MAX_ATTEMPTS = 5
BACKOFF_BASE = 0.2
BACKOFF_CAP = 30.0
MAX_SERVER_DELAY = 60.0

# Server-suggested delays sometimes arrive only in message prose; Codex
# scrapes the same text.
_TRY_AGAIN_RE = re.compile(
    r"(?i)try again in\s*"
    r"(?:(\d+)m(\d+(?:\.\d+)?)s"
    r"|(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?|m|minutes?)(?![a-z]))"
)


class RequestFailed(Exception):
    """A model request failed below the semantic layer.

    Providers classify and raise; the session's request loop owns the
    retrying.

    Attributes:
        kind: Coarse failure category (``"connection"``, ``"timeout"``,
            ``"rate_limit"``, ``"server"``, ``"overloaded"``, ``"api"``,
            ``"context_overflow"``, ``"internal"``). This is what surfaces
            as ``Error.kind`` when the turn gives up.
        retryable: Whether the unified retry loop may retry it.
        server_delay: Server-suggested delay in seconds (wins over backoff).
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool,
        server_delay: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.server_delay = server_delay


def backoff_delay(attempt: int) -> float:
    """Delay before retry number ``attempt`` (1-based): 200ms·2ⁿ capped at 30s, jittered."""
    base = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
    return base * random.uniform(0.9, 1.1)


def retry_delay_for(err: RequestFailed, attempt: int) -> float | None:
    """The delay to sleep before retrying, or None to give up.

    Server-suggested delays always win; a server asking for more than 60s
    means give up now.
    """
    if err.server_delay is not None:
        if err.server_delay > MAX_SERVER_DELAY:
            return None
        return err.server_delay
    return backoff_delay(attempt)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (seconds or HTTP-date form) into seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


_UNIT_SECONDS = {
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "s": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
}


def scrape_delay(message: str) -> float | None:
    """Extract a "try again in ..." delay (s/ms/m/minutes/1m30s forms)."""
    m = _TRY_AGAIN_RE.search(message)
    if not m:
        return None
    if m.group(1) is not None:  # the 1m30s form
        return float(m.group(1)) * 60 + float(m.group(2))
    return float(m.group(3)) * _UNIT_SECONDS[m.group(4).lower()]


def classify_status(
    status: int | None,
    message: str,
    retry_after: float | None,
) -> RequestFailed:
    """The shared HTTP-status half of both providers' error taxonomies.

    Callers handle their provider-specific cases first (overloaded events,
    context overflow, policy codes) and fall through to this.
    """
    delay = retry_after if retry_after is not None else scrape_delay(message)
    if status is None:
        return RequestFailed("api", message, retryable=False)
    if status == 200:
        # An in-stream error event on an otherwise-200 stream (e.g. Anthropic
        # overloaded variants): retry it.
        return RequestFailed("server", message, retryable=True, server_delay=delay)
    if status in (408, 529) or status >= 500:
        return RequestFailed("server", message, retryable=True, server_delay=delay)
    if status == 429 and delay is not None:
        return RequestFailed("rate_limit", message, retryable=True, server_delay=delay)
    return RequestFailed("api", message, retryable=False)
