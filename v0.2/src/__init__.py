"""
AI Research Acceleration Model v0.2
====================================

Adds historical calibration module with Bayesian parameter estimation.

v0.2.1 Update: Extended calibration set with molecular biology and
computational biology technologies per expert reviewer feedback (H4, H5).
"""

from .historical_calibration import (
    HistoricalShift,
    ShiftCategory,
    HISTORICAL_SHIFTS,
    EXTENDED_HISTORICAL_SHIFTS,
    HistoricalCalibrator,
    CalibrationResult,
)

__version__ = "0.2.1"
