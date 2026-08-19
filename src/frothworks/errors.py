class FrothworksError(Exception):
    """Base class for every frothworks exception.

    Raised only at construction or misuse time. The event stream from
    ``Session.send`` never throws; runtime failures arrive as ``Error``
    events instead.
    """


class ConfigurationError(FrothworksError):
    """Invalid session configuration (unknown model, unsupported effort, ...)."""


class ToolDefinitionError(FrothworksError):
    """A tool was defined incorrectly (sync handler, missing type hints, ...)."""


class SessionStateError(FrothworksError):
    """A session method was called in a state that does not allow it."""


class SerializationError(FrothworksError):
    """``Session.from_dict`` received a dict it cannot resume from."""
