from dataclasses import dataclass

# SDK-authored strings are full sentences in [square brackets]; consumer text
# (steer bodies, tool outputs) is never bracketed.
STEER_PREAMBLE = (
    "[You received the following new instruction(s) while working. "
    "Incorporate them and continue:]"
)
CONTINUATION_TEXT = (
    "[Your reply was cut off before it was finished. "
    "Continue exactly where you stopped.]"
)
INTERRUPTION_MARKER = (
    "[This turn was interrupted before it finished. Do not continue the "
    "interrupted work unless asked.]"
)
ABORTED_TOOL_RESULT = "[This tool call was aborted.]"
FAILED_TOOL_MARKER = "[This tool call failed.]"
COMPACTION_RESUME_TEXT = (
    "[The conversation history was just compacted. Continue with the task.]"
)


@dataclass(frozen=True)
class PendingCall:
    """A tool call the model has made that has not been answered yet."""

    id: str
    name: str
    args: dict
