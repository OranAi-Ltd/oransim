"""Instagram platform adapter.

Status: 🟡 stub — roadmap v0.5 (Q4 2026).

To track progress or volunteer: see ROADMAP.md and
https://github.com/oranai/oransim/issues?q=label%3Aplatforms+instagram

Attempting to use this adapter today will raise NotImplementedError.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    raise NotImplementedError(
        "Instagram adapter is on the roadmap for v0.5. "
        "See https://github.com/oranai/oransim/blob/main/ROADMAP.md"
    )
