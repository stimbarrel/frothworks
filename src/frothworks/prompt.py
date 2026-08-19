from collections.abc import Sequence

from .tools import Tool


def build_system_prompt(
    instructions: str,
    tools: Sequence[Tool],
    guidelines: Sequence[str],
    extra_sections: Sequence[tuple[str, str]],
) -> str:
    """Render the session's system prompt.

    Runs once, at session construction, and the result is frozen for the
    session's life — byte-stability keeps the provider prompt caches
    anchored, so ``Session.from_dict`` reuses the stored string verbatim.

    Layout: instructions verbatim (first bytes, no SDK preamble), then
    ``Available tools:`` mirroring the API tool descriptions, then
    ``Guidelines:`` (session-level first, then per-tool, exact-string
    dedup), then each extra section as ``Title:\\n{body}``. Empty sections
    are omitted.
    """
    parts = [instructions]

    if tools:
        lines = [f"- {t.name}: {t.description}" for t in tools]
        parts.append("Available tools:\n" + "\n".join(lines))

    seen: set[str] = set()
    guideline_lines: list[str] = []
    for g in list(guidelines) + [g for t in tools for g in t.guidelines]:
        if g not in seen:
            seen.add(g)
            guideline_lines.append(f"- {g}")
    if guideline_lines:
        parts.append("Guidelines:\n" + "\n".join(guideline_lines))

    for title, body in extra_sections:
        if title.strip() and body.strip():
            parts.append(f"{title}:\n{body}")

    return "\n\n".join(parts)
