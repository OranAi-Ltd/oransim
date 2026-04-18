"""SLOC — Sparse LLM Oracle Calibration.

Core IP of Oransim. Uses stratified sampling theory (Kish 1965) combined
with PAC-Bayes bounds (McAllester 1999) to let O(√N) LLM oracles represent
a population of N statistical agents.

Theoretical guarantee:
   | ĈSLOC − C_population | ≤ 0.30 / √N_anchors    (PAC-Bayes)

Under the hood: KNN coreset partition → per-anchor territory → log-weighted
calibration factor. This module re-exports the implementation from
`calibration.py` with Oransim-official naming.
"""
from __future__ import annotations

# Re-export canonical implementation under Oransim naming
from .calibration import (
    voronoi_partition as build_sloc_coreset,
    calibrate_per_territory as sloc_calibrate,
    calibration_summary as sloc_summary,
    VoronoiPartition as SLOCCoreset,
    _build_features,
)

__all__ = [
    "build_sloc_coreset",
    "sloc_calibrate",
    "sloc_summary",
    "SLOCCoreset",
]
