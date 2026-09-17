# XHS outcome checkpoints

Two model versions used by the OranSim calibration study are distributed:

- `v3/`: five LightGBM outcome heads and a fitted PCA projection.
- `v3_1/`: the same five heads, three interaction/read ratios and a within-niche read percentile; the input adds normalized note age.

These models were trained on proprietary Huitun-derived observations. This release contains **fitted numeric parameters only**: native LightGBM tree files, PCA means/components and feature/target metadata. It excludes raw observations, training feature matrices, account identifiers, note text, learned topic strings, database configuration and credentials. Earlier experimental checkpoints are outside this study's release scope.

The parameters are exported from the original fitted models without refitting. Native LightGBM prediction equivalence and PCA projection equivalence were checked using synthetic numeric inputs. Source code and these model parameters are released under the repository's Apache-2.0 license; this grants no access to the underlying proprietary dataset.

## Prepared-feature inference

```python
import numpy as np
from oransim.world_model.xhs_outcome import XHSOutcomeModel

model = XHSOutcomeModel('models/xhs_outcome/v3_1')
# Replace this shape-only example with correctly encoded numeric features.
x = np.zeros((1, model.metadata['feature_dim']), dtype=np.float32)
prediction = model.predict(x)
print({name: value.tolist() for name, value in prediction.items()})
```

The first 128 columns are the fitted PCA projection of the original concatenated title/description embeddings (1,536 dimensions). Remaining columns follow the training schema: log follower count; capped duration/60; positive-duration indicator; 15 niche slots; 30 topic slots; hour sine/cosine; weekday sine/cosine; capped topic count/20; non-video indicator; and, for v3.1, clipped note age/180. `prepare_features(embeddings, hand_features)` applies the released PCA.

The original embedding service and learned topic vocabulary are not bundled. Callers need compatible prepared features; these weights alone do not provide raw-text inference or reproduce held-out accuracy. Five count heads (`exp`, `read`, `like`, `coll`, `comm`) and three ratio heads use `expm1` to invert training targets. `read_pct` uses the original identity scale. The inference interface does not apply an extra percentile clipping step.
