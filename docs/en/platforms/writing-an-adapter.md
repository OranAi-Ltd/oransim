# Writing a Platform Adapter

> **Status:** Coming soon (v0.2 — accompanying the TikTok adapter MVP as reference).

## TL;DR

Implement `backend/oransim/platforms/base.py::PlatformAdapter` for your platform. Place the implementation under `backend/oransim/platforms/<your_platform>/adapter.py`.

## Interface Contract

```python
from oransim.platforms.base import PlatformAdapter
from oransim.data.schema import CanonicalKOL, CanonicalNote

class MyPlatformAdapter(PlatformAdapter):
    platform_id = "my_platform"

    def __init__(self, data_provider):
        self.data_provider = data_provider

    def simulate_impression(self, creative, budget, **kwargs):
        # Your impression simulation logic
        ...

    def simulate_conversion(self, impression, **kwargs):
        ...

    def get_kol(self, kol_id: str) -> CanonicalKOL:
        return self.data_provider.fetch_kol(kol_id)
```

## Testing Your Adapter

```python
from oransim.platforms.<your_platform>.adapter import MyPlatformAdapter
from oransim.platforms.<your_platform>.providers.synthetic import SyntheticProvider

adapter = MyPlatformAdapter(data_provider=SyntheticProvider())
result = adapter.simulate_impression(creative=..., budget=10000)
assert result.impressions > 0
```

## Submitting Your Adapter

- Add tests under `tests/test_platforms_<your_platform>.py`
- Update the Platform Adapter Matrix in the README
- Update the roadmap entry (move from "🟡 stub" to "✅ v1")
- Sign off commits per [DCO](https://github.com/OranAi-Ltd/oransim/blob/main/CONTRIBUTING.md#developer-certificate-of-origin-dco)

Full walkthrough coming with v0.2. In the meantime, consult the `platforms/xhs/` reference implementation once it lands.
