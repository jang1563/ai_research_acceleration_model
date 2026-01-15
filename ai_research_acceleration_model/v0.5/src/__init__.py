"""
AI Research Acceleration Model v0.5
====================================

Autonomous Lab Integration Module

Key insight from v0.4: Physical stages (S4, S6) limit end-to-end acceleration
to 2-3x even when cognitive stages achieve 60-500x. This version models how
autonomous laboratories could unlock the physical bottleneck.

Key Features:
-------------
1. Autonomous Lab Scaling: Models A-Lab style automation growth
2. Physical Stage Acceleration: Updates M_max_physical based on automation level
3. Cost Dynamics: Models cost per experiment as automation scales
4. Integration Scenarios: Conservative/baseline/optimistic automation adoption

Case Study Basis:
-----------------
- A-Lab (Berkeley): 350 materials/year automated synthesis
- Emerald Cloud Lab: Remote robotic lab access
- Strateos: Automated drug discovery platform
- Transcriptic: Cloud lab for biology

Usage:
------
>>> from autonomous_lab import AutonomousLabModel, AutomationScenario
>>> model = AutonomousLabModel(domain='materials_science')
>>> forecast = model.forecast_with_automation([2025, 2030, 2040])
>>> print(forecast[2030]['acceleration'])
"""

from .autonomous_lab import (
    AutonomousLabModel,
    AutomationScenario,
    LabCapacity,
    LAB_CAPACITIES,
)

from .integrated_model import (
    IntegratedAccelerationModel,
    IntegratedForecast,
)

__version__ = "0.5.0"
__all__ = [
    'AutonomousLabModel',
    'AutomationScenario',
    'LabCapacity',
    'LAB_CAPACITIES',
    'IntegratedAccelerationModel',
    'IntegratedForecast',
]
