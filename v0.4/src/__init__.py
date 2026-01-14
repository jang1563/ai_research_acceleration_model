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

Key Changes from v0.1:
----------------------
- M_max_cognitive: 25x → 10x-1000x (domain-specific)
- M_max_physical: 2.5x → 1.0-1.5x (case-study calibrated)
- Added shift types: TYPE_I_SCALE, TYPE_II_EFFICIENCY, TYPE_III_CAPABILITY
- Added 6 domain profiles: structural_biology, materials_science, etc.
- Added backlog dynamics for Type I shifts

Usage:
------
>>> from refined_model import RefinedAccelerationModel
>>> model = RefinedAccelerationModel(domain='structural_biology')
>>> forecast = model.forecast([2025, 2030, 2040])
>>> print(forecast[2030]['acceleration'])
2.3
"""

from .refined_model import (
    RefinedAccelerationModel,
    ShiftType,
    DomainProfile,
    DOMAIN_PROFILES,
    Scenario,
    StageAcceleration,
)

from .backlog_dynamics import (
    BacklogModel,
    BacklogMetrics,
    ValidationCapacity,
    VALIDATION_CAPACITIES,
)

__version__ = "0.4.0"
__all__ = [
    'RefinedAccelerationModel',
    'ShiftType',
    'DomainProfile',
    'DOMAIN_PROFILES',
    'Scenario',
    'StageAcceleration',
    'BacklogModel',
    'BacklogMetrics',
    'ValidationCapacity',
    'VALIDATION_CAPACITIES',
]
