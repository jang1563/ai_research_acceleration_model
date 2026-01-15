#!/usr/bin/env python3
"""
Data Quality Module for AI-Accelerated Biological Discovery Model - v1.1

Updates from v1.0:
- P1-8: Global access factors integrated
- P2-14: Compute constraints on AI capability

This module implements the Data Quality Index D(t) that affects all stages
of the discovery pipeline.

Mathematical Framework:
-----------------------
D(t) = D_0 * (1 + gamma * log(A(t)))

Version: 1.1
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd


# P1-8: Global access factors by therapeutic area
GLOBAL_ACCESS_FACTORS = {
    'oncology': 0.4,      # Limited LMIC access to expensive cancer drugs
    'cns': 0.3,           # Poor access to neurological treatments
    'infectious': 0.7,    # Better access (GAVI, Global Fund)
    'rare': 0.2,          # Specialized centers only
    'general': 0.5,       # Default
    'pandemic': 0.8,      # COVAX-style distribution
}


@dataclass
class DataQualityConfig:
    """Configuration for data quality module."""

    # Data quality growth coefficient
    gamma: float = 0.08

    # Baseline data quality (normalized to 1.0 in 2024)
    D_0: float = 1.0

    # Stage-specific data quality elasticities
    elasticities: Dict[int, float] = field(default_factory=lambda: {
        1: 0.7,   # Hypothesis generation
        2: 0.5,   # Experiment design
        3: 0.3,   # Wet lab execution
        4: 0.9,   # Data analysis - HIGHEST
        5: 0.6,   # Validation
        6: 0.4,   # Phase I
        7: 0.6,   # Phase II
        8: 0.5,   # Phase III
        9: 0.2,   # Regulatory
        10: 0.3,  # Deployment
    })

    # Data generation rates by stage
    data_generation_rates: Dict[int, float] = field(default_factory=lambda: {
        1: 0.5, 2: 0.3, 3: 1.5, 4: 2.0, 5: 1.0,
        6: 0.8, 7: 1.2, 8: 1.5, 9: 0.2, 10: 0.5,
    })

    # Enable feedback loop
    enable_feedback: bool = True
    feedback_strength: float = 0.05

    # P1-8: Global access factor (therapeutic area specific)
    global_access_factor: float = 0.5

    # P2-14: Compute constraints
    compute_constraint_cognitive: float = 0.9
    compute_constraint_robotic: float = 1.0
    compute_constraint_scientific: float = 0.85


class DataQualityModule:
    """
    Models how data quality evolves over time and affects pipeline throughput.

    v1.1 Updates:
    - Global access factors for LMIC populations
    - Compute constraints on AI types
    """

    def __init__(self, config: Optional[DataQualityConfig] = None):
        self.config = config or DataQualityConfig()
        self._D_cache: Dict[float, float] = {}
        self._cumulative_data_production: float = 0.0

    def reset(self):
        """Reset cache and cumulative values."""
        self._D_cache.clear()
        self._cumulative_data_production = 0.0

    def compute_D(self, t: float, A_t: float,
                  cumulative_throughput: Optional[float] = None) -> float:
        """Compute data quality index at time t."""
        D_base = self.config.D_0 * (1 + self.config.gamma * np.log(max(A_t, 1.0)))

        if self.config.enable_feedback and cumulative_throughput is not None:
            feedback_effect = self.config.feedback_strength * np.log(1 + cumulative_throughput)
            D_base *= (1 + feedback_effect)

        return D_base

    def compute_DQM(self, stage_index: int, D_t: float) -> float:
        """Compute data quality multiplier for a specific stage."""
        epsilon = self.config.elasticities.get(stage_index, 0.5)
        return (D_t / self.config.D_0) ** epsilon

    def apply_global_access_factor(self, beneficiaries: float,
                                    therapeutic_area: str = 'general') -> float:
        """
        P1-8: Apply global access factor to beneficiary estimates.

        Reduces beneficiary count to account for limited LMIC access.
        """
        factor = GLOBAL_ACCESS_FACTORS.get(therapeutic_area, 0.5)
        return beneficiaries * factor

    def get_compute_constraint(self, ai_type: str) -> float:
        """
        P2-14: Get compute constraint factor for AI type.

        Reflects training compute limitations that slow growth.
        """
        if ai_type == 'cognitive':
            return self.config.compute_constraint_cognitive
        elif ai_type == 'robotic':
            return self.config.compute_constraint_robotic
        elif ai_type == 'scientific':
            return self.config.compute_constraint_scientific
        return 1.0

    def get_trajectory(self, years: np.ndarray,
                       A_trajectory: np.ndarray,
                       throughput_trajectory: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Compute full data quality trajectory."""
        results = []
        cumulative = 0.0

        for i, (year, A_t) in enumerate(zip(years, A_trajectory)):
            if throughput_trajectory is not None and i > 0:
                cumulative += throughput_trajectory[i-1]

            D_t = self.compute_D(year, A_t, cumulative if self.config.enable_feedback else None)

            row = {
                'year': year,
                'A': A_t,
                'D': D_t,
                'cumulative_throughput': cumulative,
            }

            for stage_idx in range(1, 11):
                row[f'DQM_S{stage_idx}'] = self.compute_DQM(stage_idx, D_t)

            results.append(row)

        return pd.DataFrame(results)


def create_default_module() -> DataQualityModule:
    """Create data quality module with default configuration."""
    return DataQualityModule(DataQualityConfig())


if __name__ == "__main__":
    print("=" * 60)
    print("Data Quality Module - v1.1")
    print("=" * 60)

    module = create_default_module()

    # Test global access factors
    print("\nGlobal Access Factors (P1-8):")
    for area, factor in GLOBAL_ACCESS_FACTORS.items():
        print(f"  {area}: {factor}")

    # Test compute constraints
    print("\nCompute Constraints (P2-14):")
    for ai_type in ['cognitive', 'robotic', 'scientific']:
        print(f"  {ai_type}: {module.get_compute_constraint(ai_type)}")

    print("\nModule loaded successfully.")
