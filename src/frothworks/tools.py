import inspect
import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .errors import ToolDefinitionError


class ToolError(Exception):
    """Raise inside a tool handler to fail deliberately.

    The message is sent to the model verbatim as an error tool result
    (``deliberate=True`` on the ``ToolResult`` event).
    """


@dataclass(frozen=True)
class Tool:
    """A registered tool: handler plus the schema derived from its signature."""

    name: str
    description: str
    guidelines: tuple[str, ...]
    handler: Callable[..., Awaitable[str]]
    input_schema: dict
    _model: type[BaseModel] = field(repr=False)

    async def __call__(self, *args: object, **kwargs: object) -> str:
        """Call the underlying handler directly (bypasses validation)."""
        return await self.handler(*args, **kwargs)

    async def run(self, args: dict) -> tuple[str, bool, bool]:
        """Validate ``args``, execute the handler, and normalize the outcome.

        Returns:
            ``(content, is_error, deliberate)``.
        """
        try:
            parsed = self._model.model_validate(args)
        except ValidationError as e:
            return str(e), True, False
        kwargs = {name: getattr(parsed, name) for name in type(parsed).model_fields}
        try:
            result = await self.handler(**kwargs)
        except ToolError as e:
            return str(e), True, True
        except Exception as e:  # noqa: BLE001 — accidental tool failures become results
            return f"{type(e).__name__}: {e}", True, False
        if not isinstance(result, str):
            message = (
                f"TypeError: tool '{self.name}' returned "
                f"{type(result).__name__}; tool handlers must return str"
            )
            return message, True, False
        return result, False, False


def _split_annotated(hint: object) -> tuple[object, str | None]:
    """Split ``Annotated[T, "desc"]`` into the base type and the description."""
    if typing.get_origin(hint) is typing.Annotated:
        args = typing.get_args(hint)
        desc = next((m for m in args[1:] if isinstance(m, str)), None)
        return args[0], desc
    return hint, None


def _build_tool(
    fn: Callable[..., Awaitable[str]],
    description: str,
    guidelines: Sequence[str],
    name: str | None,
) -> Tool:
    if name is not None and not isinstance(name, str):
        raise ToolDefinitionError("tool name must be a string")
    tool_name = name or getattr(fn, "__name__", "")
    if not tool_name:
        raise ToolDefinitionError("tool handlers without a __name__ need name=...")
    if not isinstance(description, str) or not description.strip():
        raise ToolDefinitionError(
            f"tool '{tool_name}' requires a non-empty string description"
        )
    if isinstance(guidelines, str):
        raise ToolDefinitionError(
            f"tool '{tool_name}' guidelines must be a sequence of strings"
        )
    try:
        guideline_list = list(guidelines)
    except TypeError as e:
        raise ToolDefinitionError(
            f"tool '{tool_name}' guidelines must be a sequence: {e}"
        ) from e
    if not all(isinstance(g, str) for g in guideline_list):
        raise ToolDefinitionError(f"tool '{tool_name}' guidelines must be strings")
    if not inspect.iscoroutinefunction(fn):
        raise ToolDefinitionError(
            f"tool '{tool_name}' must be an async function; wrap blocking work "
            "in `await asyncio.to_thread(...)` if you must"
        )
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as e:
        raise ToolDefinitionError(
            f"tool '{tool_name}' has uninspectable type hints: {e}"
        ) from e
    fields: dict[str, tuple[object, object]] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolDefinitionError(
                f"tool '{tool_name}' must not use *args/**kwargs (parameter '{pname}')"
            )
        if param.kind is param.POSITIONAL_ONLY:
            # Tool args always arrive by name.
            raise ToolDefinitionError(
                f"tool '{tool_name}' must not use positional-only parameters "
                f"(parameter '{pname}')"
            )
        if pname not in hints:
            raise ToolDefinitionError(
                f"tool '{tool_name}' parameter '{pname}' is missing a type hint"
            )
        if pname == "model_config" or (
            pname.startswith("model_") and hasattr(BaseModel, pname)
        ):
            # create_model consumes "model_config" silently as configuration;
            # other reserved names fail deep inside pydantic.
            raise ToolDefinitionError(
                f"tool '{tool_name}' parameter '{pname}' collides with "
                "pydantic's reserved model_* namespace; rename it"
            )
        base, desc = _split_annotated(hints[pname])
        field_kwargs: dict = {}
        if desc is not None:
            field_kwargs["description"] = desc
        if param.default is not param.empty:
            field_kwargs["default"] = param.default
        fields[pname] = (base, Field(**field_kwargs))
    try:
        model = create_model(
            f"{tool_name}_args",
            __config__=ConfigDict(extra="forbid"),
            **typing.cast(dict[str, typing.Any], fields),
        )
        schema = model.model_json_schema()
    except Exception as e:
        raise ToolDefinitionError(
            f"tool '{tool_name}' has parameters pydantic cannot build a schema for: {e}"
        ) from e
    return Tool(
        name=tool_name,
        description=description,
        guidelines=tuple(guideline_list),
        handler=fn,
        input_schema=schema,
        _model=model,
    )


def tool(
    *,
    description: str,
    guidelines: Sequence[str] = (),
    name: str | None = None,
) -> Callable[[Callable[..., Awaitable[str]]], Tool]:
    """Declare a tool.

    The input schema comes from the handler's type hints via pydantic
    (``Annotated[T, "desc"]`` adds a parameter description); docstrings are
    never parsed. Handlers must be ``async def`` and return ``str``. Raise
    :class:`ToolError` to fail deliberately; any other exception becomes an
    accidental error result, and the agent loop continues either way.

    Args:
        description: Sent to the model in the API tools array and mirrored in
            the system prompt's ``Available tools:`` section.
        guidelines: Bullet points merged into the system prompt's
            ``Guidelines:`` section, after session-level guidelines.
        name: Override for the tool name (defaults to the function name).

    Returns:
        A decorator producing a :class:`Tool`. Programmatic use is the same
        single API: ``tool(description=...)(handler)``.
    """

    def decorator(fn: Callable[..., Awaitable[str]]) -> Tool:
        return _build_tool(fn, description, guidelines, name)

    return decorator
