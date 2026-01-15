"""
AI Research Acceleration Model v0.3
====================================

Case Study Validation Module

Validates model predictions against real-world AI breakthroughs:
- AlphaFold 2/3 (Structural Biology, 2021-2024)
- GNoME (Materials Science, 2023)
- ESM-3 (Protein Design, 2024)

Key validation questions:
1. Does our acceleration model match observed speedups?
2. Are the bottlenecks where we predicted?
3. What can case studies tell us about future trajectory?
"""

from .case_study_framework import (
    CaseStudy,
    CaseStudyMetrics,
    CaseStudyValidator,
    ValidationResult,
)

from .alphafold_case_study import AlphaFoldCaseStudy
from .gnome_case_study import GNoMECaseStudy
from .esm3_case_study import ESM3CaseStudy

__version__ = "0.3.0"
