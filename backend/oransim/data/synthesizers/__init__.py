"""Population synthesizer abstraction.

A :class:`PopulationSynthesizer` draws a population of virtual consumers
satisfying target marginal / joint distributions. Four concrete
implementations are registered, trading different points along the
(simplicity, joint-fidelity, dependencies) frontier:

- :class:`IPFSynthesizer` (primary baseline) — Iterative Proportional Fitting
  (Deming & Stephan 1940). Matches marginals exactly, no joint-dependency
  modelling. Fast, reproducible, zero external dependencies.
- :class:`BayesianNetworkSynthesizer` (v0.2 roadmap) — learns a
  Bayesian-network structure over demographic variables and samples
  respecting conditional dependencies. Native fit with the Pearl SCM.
- :class:`TabDDPMSynthesizer` (v0.5 roadmap) — tabular diffusion (Kotelnikov
  et al. 2023) trained on real data when available; matches joint
  distributions beyond marginals. Enterprise-edition path.
- :class:`CausalDAGTabDDPMSynthesizer` (v1.0 research) — novel synthesis
  guided by the 64-node Pearl SCM so samples respect the causal DAG, not
  merely the aggregate marginals. Target for an academic submission.

Pick via registry::

    from oransim.data.synthesizers import get_synthesizer

    syn = get_synthesizer("ipf")               # primary baseline
    pop = syn.generate(N=1_000_000, seed=42)
"""

from .base import (
    PopulationSynthesizer,
    SynthesizerConfig,
    SynthesizedPopulation,
)
from .ipf import IPFSynthesizer, IPFConfig
from .registry import REGISTRY, get_synthesizer, list_synthesizers

__all__ = [
    "PopulationSynthesizer",
    "SynthesizerConfig",
    "SynthesizedPopulation",
    "IPFSynthesizer",
    "IPFConfig",
    "REGISTRY",
    "get_synthesizer",
    "list_synthesizers",
]
