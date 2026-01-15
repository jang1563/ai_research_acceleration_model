"""
AI-Accelerated Scientific Research Model
========================================

A quantitative model analyzing how AI accelerates scientific research,
from hypothesis generation to publication.

Version: 0.1
"""

__version__ = "0.1.0"
__author__ = "AI Research Acceleration Project"

from .pipeline import ResearchPipeline, Stage
from .paradigm_shift import ParadigmShiftModule, ShiftType
from .model import AIResearchAccelerationModel
from .simulation import MonteCarloSimulator

__all__ = [
    "ResearchPipeline",
    "Stage",
    "ParadigmShiftModule",
    "ShiftType",
    "AIResearchAccelerationModel",
    "MonteCarloSimulator",
]
