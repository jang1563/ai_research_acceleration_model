#!/usr/bin/env python3
"""
Pipeline Iteration Module for AI-Accelerated Biological Discovery Model - v1.1

Updates from v1.0:
- P2-13: Manufacturing constraints for novel modalities

This module implements failure/rework dynamics where projects can fail at any stage
and return to earlier stages for iteration.

Version: 1.1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ReworkConfig:
    """Configuration for a stage's rework behavior on failure."""
    stage_index: int
    return_stage: int
    rework_fraction: float
    max_attempts: int
    description: str


# Default rework configurations (calibrated to Paul et al. 2010)
DEFAULT_REWORK_CONFIG = {
    1: ReworkConfig(1, 0, 0.0, 1, "Hypothesis Generation: Failed hypotheses abandoned"),
    2: ReworkConfig(2, 1, 0.7, 3, "Experiment Design: Failures may require new hypothesis"),
    3: ReworkConfig(3, 2, 0.8, 5, "Wet Lab: Failures often require redesigned experiments"),
    4: ReworkConfig(4, 3, 0.9, 3, "Data Analysis: Failures typically need more data"),
    5: ReworkConfig(5, 3, 0.6, 2, "Validation: Failures may need new approach"),
    6: ReworkConfig(6, 1, 0.25, 2, "Phase I: Safety failures require new compound"),
    7: ReworkConfig(7, 2, 0.15, 1, "Phase II: Efficacy failures rarely recoverable"),
    8: ReworkConfig(8, 7, 0.30, 1, "Phase III: Failures may attempt modified design"),
    9: ReworkConfig(9, 8, 0.80, 3, "Regulatory: Failures often addressable"),
    10: ReworkConfig(10, 9, 0.9, 5, "Deployment: Operational issues usually resolvable"),
}


@dataclass
class ManufacturingConstraint:
    """P2-13: Manufacturing constraints for novel modalities."""
    modality: str
    capacity_limit: float  # Maximum throughput multiplier
    scale_up_time_months: int
    description: str


# P2-13: Manufacturing constraints by modality
MANUFACTURING_CONSTRAINTS = {
    'small_molecule': ManufacturingConstraint(
        modality='small_molecule',
        capacity_limit=10.0,  # Well-established manufacturing
        scale_up_time_months=6,
        description="Traditional pharma, established supply chains"
    ),
    'biologic': ManufacturingConstraint(
        modality='biologic',
        capacity_limit=5.0,
        scale_up_time_months=12,
        description="Protein therapeutics, requires bioreactors"
    ),
    'cell_therapy': ManufacturingConstraint(
        modality='cell_therapy',
        capacity_limit=3.0,  # P2-13: Limited capacity
        scale_up_time_months=18,
        description="CAR-T and similar, patient-specific manufacturing"
    ),
    'gene_therapy': ManufacturingConstraint(
        modality='gene_therapy',
        capacity_limit=3.0,  # P2-13: Limited capacity
        scale_up_time_months=24,
        description="AAV vectors, complex production"
    ),
    'mrna': ManufacturingConstraint(
        modality='mrna',
        capacity_limit=8.0,  # COVID-19 expanded capacity
        scale_up_time_months=6,
        description="mRNA therapeutics, rapid scale-up proven"
    ),
}


@dataclass
class PipelineIterationConfig:
    """Configuration for pipeline iteration dynamics."""
    rework_configs: Dict[int, ReworkConfig] = field(
        default_factory=lambda: DEFAULT_REWORK_CONFIG.copy()
    )
    ai_improves_rework: bool = True
    learning_rate: float = 0.1
    max_simulation_cycles: int = 100
    track_detailed_stats: bool = True

    # P2-13: Manufacturing constraints
    manufacturing_constraints: Dict[str, ManufacturingConstraint] = field(
        default_factory=lambda: MANUFACTURING_CONSTRAINTS.copy()
    )
    default_modality: str = 'small_molecule'


class PipelineIterationModule:
    """Models failure and rework dynamics in the discovery pipeline."""

    def __init__(self, config: Optional[PipelineIterationConfig] = None):
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

    def get_manufacturing_constraint(self, modality: str) -> ManufacturingConstraint:
        """
        P2-13: Get manufacturing constraint for modality.

        Returns constraint object with capacity limit and scale-up time.
        """
        return self.config.manufacturing_constraints.get(
            modality,
            self.config.manufacturing_constraints[self.config.default_modality]
        )

    def apply_manufacturing_constraint(
        self,
        throughput: float,
        modality: str = 'small_molecule'
    ) -> float:
        """
        P2-13: Apply manufacturing capacity constraint to throughput.

        Novel modalities (cell/gene therapy) have supply constraints.
        """
        constraint = self.get_manufacturing_constraint(modality)
        return min(throughput, throughput * constraint.capacity_limit / 10.0)

    def compute_effective_throughput(
        self,
        base_throughput: float,
        p_success: Dict[int, float],
        n_stages: int = 10,
        modality: str = 'small_molecule'
    ) -> Tuple[float, Dict]:
        """
        Compute effective throughput accounting for rework cycles.

        P2-13: Now includes manufacturing constraints.
        """
        expected_attempts = {}
        cumulative_success_prob = 1.0

        for i in range(1, n_stages + 1):
            p_i = p_success.get(i, 0.5)
            rework = self.config.rework_configs.get(i, DEFAULT_REWORK_CONFIG[i])

            if rework.rework_fraction > 0 and p_i < 1.0:
                q_i = 1 - p_i
                expected = 1.0
                for attempt in range(1, rework.max_attempts):
                    prob_reach = (q_i * rework.rework_fraction) ** attempt
                    expected += prob_reach
                expected_attempts[i] = min(expected, rework.max_attempts)
            else:
                expected_attempts[i] = 1.0

            cumulative_success_prob *= p_i

        total_expected_attempts = sum(expected_attempts.values())

        effective_success = cumulative_success_prob
        for i in range(1, n_stages + 1):
            rework = self.config.rework_configs.get(i, DEFAULT_REWORK_CONFIG[i])
            if rework.rework_fraction < 1.0:
                p_i = p_success.get(i, 0.5)
                abandon_prob = (1 - p_i) * (1 - rework.rework_fraction)
                effective_success *= (1 - abandon_prob)

        overhead_factor = total_expected_attempts / n_stages
        effective_throughput = base_throughput * effective_success / overhead_factor

        # P2-13: Apply manufacturing constraint
        effective_throughput = self.apply_manufacturing_constraint(
            effective_throughput, modality
        )

        stats = {
            'expected_attempts_per_stage': expected_attempts,
            'total_expected_attempts': total_expected_attempts,
            'overhead_factor': overhead_factor,
            'cumulative_success_prob': cumulative_success_prob,
            'effective_success_rate': effective_success,
            'throughput_ratio': effective_throughput / base_throughput if base_throughput > 0 else 0,
            'modality': modality,
            'manufacturing_capacity_limit': self.get_manufacturing_constraint(modality).capacity_limit,
        }

        return effective_throughput, stats

    def simulate_projects(
        self,
        n_projects: int,
        p_success: Dict[int, float],
        n_stages: int = 10,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Monte Carlo simulation of projects through the pipeline."""
        if seed is not None:
            np.random.seed(seed)

        results = []

        for project_id in range(n_projects):
            current_stage = 1
            attempts = {i: 0 for i in range(1, n_stages + 1)}
            outcome = None
            path = [1]

            while outcome is None:
                p_i = p_success.get(current_stage, 0.5)
                rework = self.config.rework_configs.get(
                    current_stage, DEFAULT_REWORK_CONFIG[current_stage]
                )

                attempts[current_stage] += 1

                if attempts[current_stage] > rework.max_attempts:
                    outcome = 'abandoned'
                    break

                if np.random.random() < p_i:
                    if current_stage == n_stages:
                        outcome = 'completed'
                    else:
                        current_stage += 1
                        path.append(current_stage)
                else:
                    if rework.return_stage == 0:
                        outcome = 'abandoned'
                    elif np.random.random() < rework.rework_fraction:
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

    def get_manufacturing_summary(self) -> pd.DataFrame:
        """P2-13: Get summary of manufacturing constraints."""
        rows = []
        for modality, constraint in self.config.manufacturing_constraints.items():
            rows.append({
                'modality': modality,
                'capacity_limit': constraint.capacity_limit,
                'scale_up_months': constraint.scale_up_time_months,
                'description': constraint.description
            })
        return pd.DataFrame(rows)


def create_default_module() -> PipelineIterationModule:
    """Create pipeline iteration module with default configuration."""
    return PipelineIterationModule(PipelineIterationConfig())


if __name__ == "__main__":
    print("=" * 70)
    print("Pipeline Iteration Module - v1.1")
    print("=" * 70)

    module = create_default_module()

    print("\nP2-13 Manufacturing Constraints:")
    print("-" * 70)
    print(module.get_manufacturing_summary().to_string(index=False))

    print("\nRework Configuration Summary:")
    print("-" * 70)
    print(module.get_rework_summary().to_string(index=False))

    # Test with different modalities
    p_success = {
        1: 0.40, 2: 0.90, 3: 0.30, 4: 0.95, 5: 0.50,
        6: 0.66, 7: 0.33, 8: 0.58, 9: 0.90, 10: 0.95,
    }

    print("\n\nThroughput by Modality (P2-13):")
    print("-" * 70)
    for modality in ['small_molecule', 'biologic', 'cell_therapy', 'gene_therapy', 'mrna']:
        eff_throughput, stats = module.compute_effective_throughput(
            100.0, p_success, modality=modality
        )
        print(f"  {modality:15s}: throughput={eff_throughput:.2f}, "
              f"capacity_limit={stats['manufacturing_capacity_limit']}")

    print("\nModule loaded successfully.")
