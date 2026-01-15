#!/usr/bin/env python3
"""
Pipeline Iteration Module for AI-Accelerated Biological Discovery Model

This module implements failure/rework dynamics where projects can fail at any stage
and return to earlier stages for iteration. This creates a more realistic model
of drug development where failures are common and rework is expected.

Mathematical Framework:
-----------------------
The pipeline is modeled as a semi-Markov process where:
1. Each stage has a success probability p_i(t) (can improve with AI)
2. Failed projects can return to earlier stages (not always stage 1)
3. The effective throughput accounts for rework cycles

Key equations:
- Failure probability: q_i(t) = 1 - p_i(t)
- Rework destination: stage j < i (configurable per stage)
- Expected cycles: E[cycles] = 1 / prod(p_i) for full pipeline
- Effective throughput: Theta_eff = Theta / E[cycles]

References:
-----------
- DiMasi et al. (2016): Drug development success rates and rework patterns
- Paul et al. (2010): "How to improve R&D productivity"
- Hay et al. (2014): Clinical development success rates

Version: 0.7
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ReworkConfig:
    """Configuration for a stage's rework behavior on failure."""
    stage_index: int
    return_stage: int  # Which stage to return to on failure (0 = exit pipeline)
    rework_fraction: float  # Fraction of failed projects that attempt rework (vs. abandon)
    max_attempts: int  # Maximum rework attempts before abandonment
    description: str


# Default rework configurations based on drug development patterns
DEFAULT_REWORK_CONFIG = {
    # Stage 1: Hypothesis Generation - failures often lead to new hypotheses
    1: ReworkConfig(
        stage_index=1,
        return_stage=0,  # Exit - generate new hypothesis
        rework_fraction=0.0,
        max_attempts=1,
        description="Hypothesis Generation: Failed hypotheses are abandoned, new ones generated"
    ),

    # Stage 2: Experiment Design - failures return to S1 for new approach
    2: ReworkConfig(
        stage_index=2,
        return_stage=1,
        rework_fraction=0.7,  # 70% try a different approach
        max_attempts=3,
        description="Experiment Design: Failures may require new hypothesis"
    ),

    # Stage 3: Wet Lab Execution - failures often repeat or return to S2
    3: ReworkConfig(
        stage_index=3,
        return_stage=2,  # Return to experiment design
        rework_fraction=0.8,  # 80% redesign and retry
        max_attempts=5,
        description="Wet Lab: Failures often require redesigned experiments"
    ),

    # Stage 4: Data Analysis - rarely fails, but may need more data (S3)
    4: ReworkConfig(
        stage_index=4,
        return_stage=3,
        rework_fraction=0.9,  # Almost always collect more data
        max_attempts=3,
        description="Data Analysis: Failures typically need more/better data"
    ),

    # Stage 5: Validation - failures return to S3 for new experiments
    5: ReworkConfig(
        stage_index=5,
        return_stage=3,
        rework_fraction=0.6,  # 60% try to validate again
        max_attempts=2,
        description="Validation: Failures may need new experimental approach"
    ),

    # Stage 6: Phase I - failures usually end the project or return to S1
    # CALIBRATED: Paul et al. (2010) - ~25% reformulation rate for safety failures
    6: ReworkConfig(
        stage_index=6,
        return_stage=1,  # Back to hypothesis for safety failures
        rework_fraction=0.25,  # Calibrated: 25% try reformulation (was 0.30)
        max_attempts=2,
        description="Phase I: Safety failures often require new compound"
    ),

    # Stage 7: Phase II - THE GRAVEYARD - most failures end here
    # CALIBRATED: Paul et al. (2010) - ~15% reformulation rate for efficacy failures
    7: ReworkConfig(
        stage_index=7,
        return_stage=2,  # Return to experiment design (reformulate)
        rework_fraction=0.15,  # Calibrated: 15% attempt reformulation (was 0.20)
        max_attempts=1,  # CALIBRATED: Most drugs get ONE shot (was 2)
        description="Phase II: Efficacy failures rarely recoverable"
    ),

    # Stage 8: Phase III - failures may return to Phase II with modified design
    # CALIBRATED: Paul et al. (2010) - ~30% attempt modified design
    8: ReworkConfig(
        stage_index=8,
        return_stage=7,
        rework_fraction=0.30,  # Calibrated: 30% try modified Phase III (was 0.40)
        max_attempts=1,  # CALIBRATED: Most drugs get ONE shot (was 2)
        description="Phase III: Failures may attempt modified trial design"
    ),

    # Stage 9: Regulatory - failures return to provide more data
    # CALIBRATED: FDA Complete Response Letters (CRLs) often addressed
    9: ReworkConfig(
        stage_index=9,
        return_stage=8,  # May need additional Phase III data
        rework_fraction=0.80,  # Calibrated: 80% address regulatory concerns (was 0.70)
        max_attempts=3,
        description="Regulatory: Failures often addressable with more data"
    ),

    # Stage 10: Deployment - failures are operational, usually fixable
    10: ReworkConfig(
        stage_index=10,
        return_stage=9,  # May need regulatory amendments
        rework_fraction=0.9,
        max_attempts=5,
        description="Deployment: Operational issues usually resolvable"
    ),
}


@dataclass
class PipelineIterationConfig:
    """Configuration for pipeline iteration dynamics."""

    # Rework configurations per stage
    rework_configs: Dict[int, ReworkConfig] = field(
        default_factory=lambda: DEFAULT_REWORK_CONFIG.copy()
    )

    # Whether AI can reduce rework (improve p_success over iterations)
    ai_improves_rework: bool = True

    # Learning rate: how much p_success improves per failed attempt
    # p_new = p_old + learning_rate * (p_max - p_old) per attempt
    learning_rate: float = 0.1

    # Maximum number of pipeline cycles to simulate
    max_simulation_cycles: int = 100

    # Track detailed rework statistics
    track_detailed_stats: bool = True


class PipelineIterationModule:
    """
    Models failure and rework dynamics in the discovery pipeline.

    Key features:
    1. Stage-specific failure handling (where to return on failure)
    2. Rework fractions (what % of failures attempt rework vs. abandon)
    3. Learning effects (AI improves success probability over attempts)
    4. Computes effective throughput accounting for rework cycles
    """

    def __init__(self, config: Optional[PipelineIterationConfig] = None):
        """Initialize pipeline iteration module."""
        self.config = config or PipelineIterationConfig()
        self._stats: Dict = {}

    def reset_stats(self):
        """Reset tracking statistics."""
        self._stats = {
            'total_attempts': {i: 0 for i in range(1, 11)},
            'successful_attempts': {i: 0 for i in range(1, 11)},
            'rework_cycles': [],
            'abandoned_projects': 0,
            'completed_projects': 0,
        }

    def compute_effective_throughput(
        self,
        base_throughput: float,
        p_success: Dict[int, float],
        n_stages: int = 10
    ) -> Tuple[float, Dict]:
        """
        Compute effective throughput accounting for rework cycles.

        Parameters
        ----------
        base_throughput : float
            Throughput without considering failures
        p_success : Dict[int, float]
            Success probability for each stage
        n_stages : int
            Number of stages in pipeline

        Returns
        -------
        Tuple[float, Dict]
            Effective throughput and detailed statistics
        """
        # Calculate expected number of attempts per stage
        expected_attempts = {}
        cumulative_success_prob = 1.0

        for i in range(1, n_stages + 1):
            p_i = p_success.get(i, 0.5)
            rework = self.config.rework_configs.get(i, DEFAULT_REWORK_CONFIG[i])

            # Expected attempts at this stage considering rework
            # If rework_fraction = 0, one attempt only
            # If rework_fraction > 0, geometric series
            if rework.rework_fraction > 0 and p_i < 1.0:
                # Geometric series: E[attempts] = 1 + rework_fraction * (1-p) / p
                # But capped at max_attempts
                q_i = 1 - p_i
                expected = 1.0
                for attempt in range(1, rework.max_attempts):
                    # Probability of reaching this attempt
                    prob_reach = (q_i * rework.rework_fraction) ** attempt
                    expected += prob_reach
                expected_attempts[i] = min(expected, rework.max_attempts)
            else:
                expected_attempts[i] = 1.0

            cumulative_success_prob *= p_i

        # Total expected attempts across pipeline
        total_expected_attempts = sum(expected_attempts.values())

        # Effective success rate (probability of completing pipeline)
        # Accounts for abandonment at each stage
        effective_success = cumulative_success_prob
        for i in range(1, n_stages + 1):
            rework = self.config.rework_configs.get(i, DEFAULT_REWORK_CONFIG[i])
            if rework.rework_fraction < 1.0:
                # Some projects are abandoned
                p_i = p_success.get(i, 0.5)
                abandon_prob = (1 - p_i) * (1 - rework.rework_fraction)
                effective_success *= (1 - abandon_prob)

        # Effective throughput
        # = base_throughput * effective_success / overhead_factor
        overhead_factor = total_expected_attempts / n_stages
        effective_throughput = base_throughput * effective_success / overhead_factor

        stats = {
            'expected_attempts_per_stage': expected_attempts,
            'total_expected_attempts': total_expected_attempts,
            'overhead_factor': overhead_factor,
            'cumulative_success_prob': cumulative_success_prob,
            'effective_success_rate': effective_success,
            'throughput_ratio': effective_throughput / base_throughput if base_throughput > 0 else 0,
        }

        return effective_throughput, stats

    def compute_stage_cycle_matrix(
        self,
        p_success: Dict[int, float],
        n_stages: int = 10
    ) -> np.ndarray:
        """
        Compute the transition matrix for the semi-Markov process.

        Returns matrix M where M[i,j] = probability of transitioning
        from stage i to stage j (including staying at i or going backward).

        Parameters
        ----------
        p_success : Dict[int, float]
            Success probability for each stage
        n_stages : int
            Number of stages

        Returns
        -------
        np.ndarray
            Transition matrix (n_stages+2 x n_stages+2)
            States: 0=start, 1-10=stages, 11=success, 12=abandon
        """
        n_states = n_stages + 3  # start, stages 1-10, success, abandon
        M = np.zeros((n_states, n_states))

        # Start state (0) -> Stage 1
        M[0, 1] = 1.0

        for i in range(1, n_stages + 1):
            p_i = p_success.get(i, 0.5)
            rework = self.config.rework_configs.get(i, DEFAULT_REWORK_CONFIG[i])

            # Success -> next stage (or completion)
            if i < n_stages:
                M[i, i + 1] = p_i
            else:
                M[i, n_stages + 1] = p_i  # To success state

            # Failure handling
            q_i = 1 - p_i

            if rework.return_stage == 0:
                # Failure -> abandon (exit pipeline)
                M[i, n_stages + 2] = q_i
            else:
                # Failure -> rework or abandon
                M[i, rework.return_stage] = q_i * rework.rework_fraction
                M[i, n_stages + 2] = q_i * (1 - rework.rework_fraction)

        # Absorbing states
        M[n_stages + 1, n_stages + 1] = 1.0  # Success
        M[n_stages + 2, n_stages + 2] = 1.0  # Abandon

        return M

    def simulate_projects(
        self,
        n_projects: int,
        p_success: Dict[int, float],
        n_stages: int = 10,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Monte Carlo simulation of projects through the pipeline.

        Parameters
        ----------
        n_projects : int
            Number of projects to simulate
        p_success : Dict[int, float]
            Success probability for each stage
        n_stages : int
            Number of stages
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Simulation results with project outcomes
        """
        if seed is not None:
            np.random.seed(seed)

        results = []

        for project_id in range(n_projects):
            current_stage = 1
            attempts = {i: 0 for i in range(1, n_stages + 1)}
            total_time = 0
            outcome = None
            path = [1]

            while outcome is None:
                p_i = p_success.get(current_stage, 0.5)
                rework = self.config.rework_configs.get(
                    current_stage, DEFAULT_REWORK_CONFIG[current_stage]
                )

                attempts[current_stage] += 1

                # Check for max attempts
                if attempts[current_stage] > rework.max_attempts:
                    outcome = 'abandoned'
                    break

                # Roll for success
                if np.random.random() < p_i:
                    # Success!
                    if current_stage == n_stages:
                        outcome = 'completed'
                    else:
                        current_stage += 1
                        path.append(current_stage)
                else:
                    # Failure
                    if rework.return_stage == 0:
                        outcome = 'abandoned'
                    elif np.random.random() < rework.rework_fraction:
                        # Rework
                        current_stage = rework.return_stage
                        path.append(current_stage)
                    else:
                        outcome = 'abandoned'

            results.append({
                'project_id': project_id,
                'outcome': outcome,
                'total_attempts': sum(attempts.values()),
                'path_length': len(path),
                'final_stage': current_stage,
                **{f'attempts_S{i}': attempts[i] for i in range(1, n_stages + 1)}
            })

        return pd.DataFrame(results)

    def get_rework_summary(self) -> pd.DataFrame:
        """Get summary of rework configurations."""
        rows = []
        for i, config in self.config.rework_configs.items():
            rows.append({
                'stage': i,
                'return_stage': config.return_stage,
                'rework_fraction': config.rework_fraction,
                'max_attempts': config.max_attempts,
                'description': config.description
            })
        return pd.DataFrame(rows)


def create_default_module() -> PipelineIterationModule:
    """Create pipeline iteration module with default configuration."""
    return PipelineIterationModule(PipelineIterationConfig())


def compute_pipeline_efficiency(
    p_success: Dict[int, float],
    rework_configs: Optional[Dict[int, ReworkConfig]] = None
) -> Dict:
    """
    Compute overall pipeline efficiency metrics.

    Parameters
    ----------
    p_success : Dict[int, float]
        Success probability for each stage
    rework_configs : Dict, optional
        Rework configurations (uses defaults if None)

    Returns
    -------
    Dict
        Efficiency metrics
    """
    if rework_configs is None:
        rework_configs = DEFAULT_REWORK_CONFIG

    # Naive success rate (no rework)
    naive_success = np.prod([p_success.get(i, 0.5) for i in range(1, 11)])

    # With rework
    module = PipelineIterationModule()
    _, stats = module.compute_effective_throughput(1.0, p_success)

    return {
        'naive_success_rate': naive_success,
        'effective_success_rate': stats['effective_success_rate'],
        'overhead_factor': stats['overhead_factor'],
        'throughput_ratio': stats['throughput_ratio'],
        'improvement_from_rework': stats['effective_success_rate'] / naive_success if naive_success > 0 else float('inf'),
    }


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Pipeline Iteration Module Test")
    print("=" * 60)

    module = create_default_module()

    # Test with typical success probabilities
    p_success = {
        1: 0.95,  # Hypothesis
        2: 0.90,  # Design
        3: 0.30,  # Wet Lab
        4: 0.95,  # Analysis
        5: 0.50,  # Validation
        6: 0.66,  # Phase I
        7: 0.33,  # Phase II (bottleneck)
        8: 0.58,  # Phase III
        9: 0.90,  # Regulatory
        10: 0.95, # Deployment
    }

    print("\nRework Configuration Summary:")
    print("-" * 60)
    print(module.get_rework_summary().to_string(index=False))

    print("\nEffective Throughput Calculation:")
    print("-" * 60)
    eff_throughput, stats = module.compute_effective_throughput(100.0, p_success)
    print(f"  Base throughput: 100.0")
    print(f"  Effective throughput: {eff_throughput:.2f}")
    print(f"  Throughput ratio: {stats['throughput_ratio']:.3f}")
    print(f"  Overhead factor: {stats['overhead_factor']:.2f}")
    print(f"  Cumulative success prob: {stats['cumulative_success_prob']:.4f}")
    print(f"  Effective success rate: {stats['effective_success_rate']:.4f}")

    print("\nExpected Attempts per Stage:")
    for i, attempts in stats['expected_attempts_per_stage'].items():
        print(f"  S{i}: {attempts:.2f}")

    print("\nMonte Carlo Simulation (1000 projects):")
    print("-" * 60)
    sim_results = module.simulate_projects(1000, p_success, seed=42)
    outcomes = sim_results['outcome'].value_counts()
    print(f"  Completed: {outcomes.get('completed', 0)} ({outcomes.get('completed', 0)/10:.1f}%)")
    print(f"  Abandoned: {outcomes.get('abandoned', 0)} ({outcomes.get('abandoned', 0)/10:.1f}%)")
    print(f"  Avg attempts: {sim_results['total_attempts'].mean():.1f}")
    print(f"  Avg path length: {sim_results['path_length'].mean():.1f}")
