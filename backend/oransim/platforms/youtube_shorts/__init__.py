"""YouTube Shorts platform adapter.

Status: 🟡 stub — roadmap v0.7 (Q1 2027).

To track progress or volunteer: see ROADMAP.md and
https://github.com/ORAN-cgsj/oransim/issues?q=label%3Aplatforms+youtube

Attempting to use this adapter today will raise NotImplementedError.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    raise NotImplementedError(
        "YouTube Shorts adapter is on the roadmap for v0.7. "
        "See https://github.com/ORAN-cgsj/oransim/blob/main/ROADMAP.md"
    )
