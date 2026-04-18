"""Iterative Proportional Fitting synthesizer (primary baseline).

Wraps the existing :func:`oransim.data.population.generate_population` routine
behind the :class:`PopulationSynthesizer` interface, so callers can swap IPF
for Bayesian-network or TabDDPM variants without changing call sites.

References
----------

- W. E. Deming, F. F. Stephan. On a least squares adjustment of a sampled
  frequency table when the expected marginal totals are known. *Annals of
  Mathematical Statistics*, 1940. (The IPF algorithm.)
- P. Ye, K. Konduri, R. M. Pendyala, B. Sana, P. Waddell. A Methodology to
  Match Distributions of Both Household and Person Attributes in the
  Generation of Synthetic Populations. TRB 2009. (IPU extension.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import PopulationSynthesizer, SynthesizedPopulation, SynthesizerConfig


@dataclass
class IPFConfig(SynthesizerConfig):
    """Config for :class:`IPFSynthesizer`."""

    max_iter: int = 200
    tol: float = 1e-6


class IPFSynthesizer(PopulationSynthesizer):
    """Iterative Proportional Fitting over categorical demographic axes.

    Matches marginals exactly (up to ``tol``). Does not capture joint
    dependencies beyond what is already present in the age × gender × region
    × income × platform contingency table.
    """

    def __init__(self, config: IPFConfig | None = None):
        self.config = config or IPFConfig()

    def generate(self, N: int, *, seed: int | None = None, **kwargs: Any) -> SynthesizedPopulation:
        # Defer the heavy import so the module is cheap to load
        from ..population import generate_population

        s = seed if seed is not None else self.config.seed
        pop = generate_population(N=N, seed=s)
        return SynthesizedPopulation(
            N=pop.N,
            attributes={
                "age_idx": pop.age_idx,
                "gender_idx": pop.gender_idx,
                "city_idx": pop.city_idx,
                "income": pop.income,
                "edu_idx": pop.edu_idx,
                "occ_idx": pop.occ_idx,
            },
            latent={
                "backend": "ipf",
                "backing_object": pop,  # the original Population for API compat
            },
        )
