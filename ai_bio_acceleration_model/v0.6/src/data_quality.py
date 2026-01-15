#!/usr/bin/env python3
"""
Data Quality Module for AI-Accelerated Biological Discovery Model

This module implements the Data Quality Index D(t) that affects all stages
of the discovery pipeline. Key concepts:

1. D(t) grows with AI capability (AI improves data generation/annotation)
2. Each stage has data quality elasticity (how much D affects throughput)
3. AI both consumes and produces data (virtuous feedback loop)

Mathematical Framework:
-----------------------
D(t) = D_0 * (1 + gamma * log(A(t)))

Where:
- D_0 = 1.0 (normalized baseline in 2024)
- gamma = data quality growth coefficient (default 0.15)
- A(t) = AI capability at time t

Effect on stage service rate:
DQM_i(t) = (D(t) / D_0)^epsilon_i

Where epsilon_i is the data quality elasticity for stage i.

The modified service rate becomes:
mu_i(t) = mu_i^0 * M_i(t) * DQM_i(t)

References:
-----------
- DeepMind (2024): "Scarcity of high-quality data remains single greatest barrier"
- Topol (2019): Data quality in clinical AI
- Rajkomar et al. (2018): ML for healthcare - data quality challenges

Version: 0.6
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


@dataclass
class DataQualityConfig:
    """Configuration for data quality module."""

    # Data quality growth coefficient
    # Higher gamma = faster data quality improvement with AI
    # 0.08 is conservative (yields ~30% improvement by 2050)
    # 0.15 is aggressive (yields ~80% improvement by 2050)
    gamma: float = 0.08

    # Baseline data quality (normalized to 1.0 in 2024)
    D_0: float = 1.0

    # Stage-specific data quality elasticities
    # Higher elasticity = stage more sensitive to data quality
    elasticities: Dict[int, float] = field(default_factory=lambda: {
        1: 0.7,   # Hypothesis generation - high dependence on literature quality
        2: 0.5,   # Experiment design - moderate dependence
        3: 0.3,   # Wet lab execution - lower dependence (physical)
        4: 0.9,   # Data analysis - HIGHEST dependence
        5: 0.6,   # Validation - depends on data reproducibility
        6: 0.4,   # Phase I - trial data quality matters
        7: 0.6,   # Phase II - endpoint data critical
        8: 0.5,   # Phase III - large-scale data management
        9: 0.2,   # Regulatory - mostly procedural
        10: 0.3,  # Deployment - operational data
    })

    # Data generation rates by stage (how much data each stage produces)
    # Normalized: 1.0 = average data production
    data_generation_rates: Dict[int, float] = field(default_factory=lambda: {
        1: 0.5,   # Hypothesis generation - produces new questions
        2: 0.3,   # Experiment design - produces protocols
        3: 1.5,   # Wet lab execution - MAJOR data producer
        4: 2.0,   # Data analysis - produces processed data
        5: 1.0,   # Validation - produces replication data
        6: 0.8,   # Phase I - produces safety data
        7: 1.2,   # Phase II - produces efficacy data
        8: 1.5,   # Phase III - produces large datasets
        9: 0.2,   # Regulatory - produces documentation
        10: 0.5,  # Deployment - produces real-world evidence
    })

    # Enable feedback loop (data production affects D(t))
    enable_feedback: bool = True

    # Feedback strength (how much stage outputs improve D)
    feedback_strength: float = 0.05


class DataQualityModule:
    """
    Models how data quality evolves over time and affects pipeline throughput.

    The module captures several key dynamics:
    1. AI capability improves data quality (better annotation, curation)
    2. Different stages have different sensitivity to data quality
    3. Stages that produce data contribute to overall data quality improvement
    """

    def __init__(self, config: Optional[DataQualityConfig] = None):
        """Initialize data quality module."""
        self.config = config or DataQualityConfig()

        # Cache for computed values
        self._D_cache: Dict[float, float] = {}
        self._cumulative_data_production: float = 0.0

    def reset(self):
        """Reset cache and cumulative values."""
        self._D_cache.clear()
        self._cumulative_data_production = 0.0

    def compute_D(self, t: float, A_t: float,
                  cumulative_throughput: Optional[float] = None) -> float:
        """
        Compute data quality index at time t.

        Parameters
        ----------
        t : float
            Time (year)
        A_t : float
            AI capability at time t
        cumulative_throughput : float, optional
            Cumulative pipeline throughput (for feedback effect)

        Returns
        -------
        float
            Data quality index D(t)
        """
        # Base data quality from AI capability
        # D(t) = D_0 * (1 + gamma * log(A(t)))
        D_base = self.config.D_0 * (1 + self.config.gamma * np.log(max(A_t, 1.0)))

        # Add feedback from cumulative data production
        if self.config.enable_feedback and cumulative_throughput is not None:
            # More data → better models → better data quality
            # Effect is logarithmic to prevent explosion
            feedback_effect = self.config.feedback_strength * np.log(1 + cumulative_throughput)
            D_base *= (1 + feedback_effect)

        return D_base

    def compute_DQM(self, stage_index: int, D_t: float) -> float:
        """
        Compute data quality multiplier for a specific stage.

        Parameters
        ----------
        stage_index : int
            Stage index (1-10)
        D_t : float
            Data quality index at time t

        Returns
        -------
        float
            Data quality multiplier DQM_i(t)
        """
        epsilon = self.config.elasticities.get(stage_index, 0.5)

        # DQM = (D(t) / D_0)^epsilon
        DQM = (D_t / self.config.D_0) ** epsilon

        return DQM

    def compute_data_production(self, stage_index: int,
                                throughput: float) -> float:
        """
        Compute data produced by a stage.

        Parameters
        ----------
        stage_index : int
            Stage index
        throughput : float
            Stage throughput (projects/year)

        Returns
        -------
        float
            Data production (arbitrary units)
        """
        rate = self.config.data_generation_rates.get(stage_index, 1.0)
        return throughput * rate

    def get_elasticity(self, stage_index: int) -> float:
        """Get data quality elasticity for a stage."""
        return self.config.elasticities.get(stage_index, 0.5)

    def get_trajectory(self, years: np.ndarray,
                       A_trajectory: np.ndarray,
                       throughput_trajectory: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Compute full data quality trajectory.

        Parameters
        ----------
        years : np.ndarray
            Array of years
        A_trajectory : np.ndarray
            AI capability trajectory
        throughput_trajectory : np.ndarray, optional
            Throughput trajectory for feedback

        Returns
        -------
        pd.DataFrame
            DataFrame with year, D, and DQM for each stage
        """
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

            # Add DQM for each stage
            for stage_idx in range(1, 11):
                row[f'DQM_S{stage_idx}'] = self.compute_DQM(stage_idx, D_t)

            results.append(row)

        return pd.DataFrame(results)

    def get_summary_statistics(self, trajectory: pd.DataFrame) -> Dict:
        """
        Get summary statistics from trajectory.

        Parameters
        ----------
        trajectory : pd.DataFrame
            Output from get_trajectory()

        Returns
        -------
        dict
            Summary statistics
        """
        final_row = trajectory.iloc[-1]
        initial_row = trajectory.iloc[0]

        return {
            'D_initial': initial_row['D'],
            'D_final': final_row['D'],
            'D_growth_factor': final_row['D'] / initial_row['D'],
            'highest_elasticity_stage': max(
                self.config.elasticities.items(),
                key=lambda x: x[1]
            )[0],
            'lowest_elasticity_stage': min(
                self.config.elasticities.items(),
                key=lambda x: x[1]
            )[0],
            'max_DQM': max(final_row[f'DQM_S{i}'] for i in range(1, 11)),
            'min_DQM': min(final_row[f'DQM_S{i}'] for i in range(1, 11)),
        }


@dataclass
class DataQualityParams:
    """Stage-specific data quality parameters."""
    stage_index: int
    elasticity: float
    data_generation_rate: float
    description: str


# Default parameters with justifications
DEFAULT_DQ_PARAMS = {
    1: DataQualityParams(
        stage_index=1,
        elasticity=0.7,
        data_generation_rate=0.5,
        description="Hypothesis Generation: High dependence on literature quality and knowledge synthesis"
    ),
    2: DataQualityParams(
        stage_index=2,
        elasticity=0.5,
        data_generation_rate=0.3,
        description="Experiment Design: Moderate dependence on prior experimental data"
    ),
    3: DataQualityParams(
        stage_index=3,
        elasticity=0.3,
        data_generation_rate=1.5,
        description="Wet Lab Execution: Physical process limits data dependence; major data producer"
    ),
    4: DataQualityParams(
        stage_index=4,
        elasticity=0.9,
        data_generation_rate=2.0,
        description="Data Analysis: HIGHEST dependence - garbage in, garbage out; major producer"
    ),
    5: DataQualityParams(
        stage_index=5,
        elasticity=0.6,
        data_generation_rate=1.0,
        description="Validation: Depends on reproducibility of data and methods"
    ),
    6: DataQualityParams(
        stage_index=6,
        elasticity=0.4,
        data_generation_rate=0.8,
        description="Phase I Trials: Safety data quality matters for dose selection"
    ),
    7: DataQualityParams(
        stage_index=7,
        elasticity=0.6,
        data_generation_rate=1.2,
        description="Phase II Trials: Endpoint data quality critical for go/no-go"
    ),
    8: DataQualityParams(
        stage_index=8,
        elasticity=0.5,
        data_generation_rate=1.5,
        description="Phase III Trials: Large-scale data management and quality control"
    ),
    9: DataQualityParams(
        stage_index=9,
        elasticity=0.2,
        data_generation_rate=0.2,
        description="Regulatory: Mostly procedural; data quality already locked in"
    ),
    10: DataQualityParams(
        stage_index=10,
        elasticity=0.3,
        data_generation_rate=0.5,
        description="Deployment: Operational data for real-world evidence"
    ),
}


def create_default_module() -> DataQualityModule:
    """Create data quality module with default configuration."""
    return DataQualityModule(DataQualityConfig())


def analyze_data_quality_impact(
    baseline_results: pd.DataFrame,
    dq_results: pd.DataFrame,
    year: int = 2050
) -> pd.DataFrame:
    """
    Compare results with and without data quality module.

    Parameters
    ----------
    baseline_results : pd.DataFrame
        Results without data quality
    dq_results : pd.DataFrame
        Results with data quality
    year : int
        Target year for comparison

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparisons = []

    scenarios = baseline_results['scenario'].unique()

    for scenario in scenarios:
        baseline_row = baseline_results[
            (baseline_results['scenario'] == scenario) &
            (baseline_results['year'] == year)
        ]
        dq_row = dq_results[
            (dq_results['scenario'] == scenario) &
            (dq_results['year'] == year)
        ]

        if len(baseline_row) == 0 or len(dq_row) == 0:
            continue

        baseline_progress = baseline_row['cumulative_progress'].iloc[0]
        dq_progress = dq_row['cumulative_progress'].iloc[0]

        comparisons.append({
            'scenario': scenario,
            'without_DQ': baseline_progress,
            'with_DQ': dq_progress,
            'delta': dq_progress - baseline_progress,
            'percent_change': (dq_progress / baseline_progress - 1) * 100
        })

    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Data Quality Module Test")
    print("=" * 60)

    module = create_default_module()

    # Simulate AI capability growth
    years = np.arange(2024, 2051)
    g = 0.50  # Baseline growth rate
    A = np.exp(g * (years - 2024))

    # Simulate throughput growth
    throughput = np.ones_like(years, dtype=float) * 10  # Base throughput
    for i in range(1, len(throughput)):
        throughput[i] = throughput[i-1] * 1.05  # 5% annual growth

    # Compute trajectory
    trajectory = module.get_trajectory(years, A, throughput)

    print("\nData Quality Trajectory (selected years):")
    print("-" * 60)
    for year in [2024, 2030, 2040, 2050]:
        row = trajectory[trajectory['year'] == year].iloc[0]
        print(f"  {year}: D={row['D']:.2f}, DQM_S4={row['DQM_S4']:.2f}, DQM_S7={row['DQM_S7']:.2f}")

    print("\nSummary Statistics:")
    print("-" * 60)
    stats = module.get_summary_statistics(trajectory)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\nStage Elasticities:")
    print("-" * 60)
    for stage_idx, params in DEFAULT_DQ_PARAMS.items():
        print(f"  S{stage_idx}: epsilon={params.elasticity:.1f} - {params.description[:50]}...")
