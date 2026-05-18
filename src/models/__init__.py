"""Model implementations for HantaCast.

HantaCast: a unified hybrid model integrating MixLinear-based deep temporal
signal learner with SEIRD-constrained epidemiological dynamics.
"""
from src.models.hantacast import HantaCast  # noqa: F401

# Compatibility alias
MC_MixSEIRD = HantaCast
