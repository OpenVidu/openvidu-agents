class _NotProvidedType:
    """Singleton sentinel indicating an argument was not provided.

    Use identity checks (is) against NOT_PROVIDED. Do not rely on equality.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "NOT_PROVIDED"


# Public singleton instance to import and use across the project
NOT_PROVIDED = _NotProvidedType()
