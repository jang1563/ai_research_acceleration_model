"""
AI Research Acceleration Model v0.4
====================================

Model Refinement Based on Case Study Validation

Key improvements over v0.1:
1. Shift type classification (Type I, II, III)
2. Domain-specific calibration parameters
3. Revised M_max parameters based on empirical data
4. Backlog dynamics for Type I (scale) shifts

Based on v0.3 case study validation against AlphaFold, GNoME, ESM-3.
"""

from .refined_model import (
    RefinedAccelerationModel,
    ShiftType,
    DomainProfile,
    DOMAIN_PROFILES,
)

from .backlog_dynamics import (
    BacklogModel,
    BacklogMetrics,
)

__version__ = "0.4.0"
