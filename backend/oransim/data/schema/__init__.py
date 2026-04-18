"""Canonical schemas — the cross-platform contract consumed by
``PlatformAdapter`` and produced by ``DataProvider``.

See :mod:`oransim.data.schema.canonical` for the full type definitions.
"""

from .canonical import (
    SCHEMA_VERSION,
    CanonicalFanProfile,
    CanonicalKOL,
    CanonicalNote,
    CanonicalNoteMetrics,
    CanonicalScenario,
)

__all__ = [
    "SCHEMA_VERSION",
    "CanonicalFanProfile",
    "CanonicalKOL",
    "CanonicalNote",
    "CanonicalNoteMetrics",
    "CanonicalScenario",
]
