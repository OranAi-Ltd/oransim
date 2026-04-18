"""Abstract PlatformAdapter interface.

Phase 3 will flesh this out. For now it documents the shape so
adapter request issues (.github/ISSUE_TEMPLATE/adapter_request.yml)
can reference the expected surface.
"""

from abc import ABC, abstractmethod
from typing import Any


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific adapters.

    Implementations live in `oransim.platforms.<platform>.adapter`.

    The adapter separates platform semantics (what is a post, what is
    a conversion, what is a KOL) from data provenance (which vendor
    or data source produced the raw data). Data providers are pluggable
    via the `data_provider` attribute.
    """

    platform_id: str
    data_provider: Any  # oransim.platforms.<p>.providers.base.DataProvider

    @abstractmethod
    def simulate_impression(self, creative: Any, budget: float, **kwargs: Any) -> Any:
        """Simulate impression delivery for a creative under a budget."""
        raise NotImplementedError

    @abstractmethod
    def simulate_conversion(self, impression: Any, **kwargs: Any) -> Any:
        """Simulate conversion from an impression batch."""
        raise NotImplementedError

    @abstractmethod
    def get_kol(self, kol_id: str) -> Any:
        """Look up a KOL by ID and return CanonicalKOL."""
        raise NotImplementedError
