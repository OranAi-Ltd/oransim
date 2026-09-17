# Data and derived-artifact attribution

No raw platform, customer or transaction records are bundled. The included aggregate records, metrics and model snapshots were derived from the following sources or from synthetic generation.

| Artifact family | Source | Attribution / terms |
| --- | --- | --- |
| KuaiRand aggregates, Gaussian/GRU/DeepAR weights, evaluation results | Chongming Gao et al., KuaiRand (CIKM 2022), [upstream](https://github.com/chongminggao/KuaiRand), [dataset](https://zenodo.org/records/10439422) | CC BY-SA 4.0; retain attribution and share-alike terms for redistributed derived artifacts |
| Online Retail II aggregates, Gaussian/DeepAR weights, evaluation results | Daqing Chen, Online Retail II, [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) | CC BY 4.0; retain attribution and identify transformations |
| Synthetic recovery models and results | OranSim synthetic generators included in this repository | Apache-2.0 |
| X5 and Open Bandit summary metrics | Providers linked in [prelaunch/README.md](prelaunch/README.md) | Aggregate evaluation metrics only; acquire source datasets separately under upstream terms |

Transformations: historical cohort assignment, daily aggregation, binary behavior encoding, fitted parameters and aggregate evaluation. Aggregate benchmark inputs remove raw identifiers and event-level rows. Model snapshots preserve training-end state for causal replay. KuaiRand-derived artifacts are distributed under CC BY-SA 4.0 and Retail-derived artifacts under CC BY 4.0; these terms are separate from the repository's Apache-2.0 source-code license.

Weights under `models/gaussian_population/benchmarks/kuairand*` and `models/gaussian_population/deepar/kuairand/` use KuaiRand. Paths containing `retail/` use Online Retail II. `models/gaussian_population/controlled/choice/` uses synthetic data. Mixed-dataset result files retain both attributions and the applicable share-alike terms.

License texts: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).

## Proprietary-data-trained numeric models

`models/xhs_outcome/` contains fitted numeric parameters authorized for distribution under Apache-2.0. The proprietary Huitun-derived training dataset and its records are not distributed. The weight license does not grant rights to that dataset. See the [model card](../models/xhs_outcome/README.md).
