"""Input/format errors with actionable messages (CLI exit code 4)."""


class InputError(ValueError):
    """Unsupported or ambiguous input (codec profile, color metadata, path collision, ...)."""
