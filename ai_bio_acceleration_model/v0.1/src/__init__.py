"""
AI-Accelerated Biological Discovery Model

Version: 0.1 (Pilot Model)
"""

from .model import (
    AIBioAccelerationModel,
    ModelConfig,
    Stage,
    Scenario,
    run_default_model,
    compute_equivalent_years,
)

from .visualize import (
    ModelVisualizer,
    generate_all_visualizations,
)

__version__ = "0.1"
__all__ = [
    "AIBioAccelerationModel",
    "ModelConfig",
    "Stage",
    "Scenario",
    "run_default_model",
    "compute_equivalent_years",
    "ModelVisualizer",
    "generate_all_visualizations",
]
