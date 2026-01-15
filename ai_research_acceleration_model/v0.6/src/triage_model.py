#!/usr/bin/env python3
"""
Triage & Selection Model for v0.6
=================================

Models the critical bottleneck when AI-generated hypotheses exceed
physical validation capacity. This addresses the GNoME problem:
2.2M materials predicted, but only 350/year synthesized = 6,000 year backlog.

Key Concepts:
1. Generation Rate (G): Hypotheses/candidates generated per year by AI
2. Validation Capacity (V): Experiments that can be physically validated per year
3. Triage Efficiency (T): Fraction of validated hypotheses that succeed
4. Backlog (B): Accumulated unvalidated hypotheses

The Triage Problem:
- When G >> V, a backlog accumulates
- Simple FIFO processing wastes validation capacity on low-value hypotheses
- Intelligent triage (AI-guided selection) can dramatically improve T
- But triage itself has computational and review costs

Key Insight from Case Studies:
- GNoME: G=2.2M, V=350/yr, backlog=6,286 years
- AlphaMissense: G=71M variants, instant classification (no physical validation)
- ESM-3: G=millions of designs, V~hundreds tested

This module models:
1. Backlog dynamics over time
2. Triage efficiency improvements with AI
3. When triage becomes the binding constraint
4. Optimal allocation of validation capacity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Import from v0.5
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "v0.5" / "src"))
from autonomous_lab import AutomationScenario, LAB_CAPACITIES


class TriageStrategy(Enum):
    """Strategies for selecting which hypotheses to validate."""
    FIFO = "fifo"                    # First-in, first-out (no intelligence)
    RANDOM = "random"                # Random selection
    CONFIDENCE_RANKED = "confidence" # Rank by AI confidence score
    UNCERTAINTY_SAMPLING = "uncertainty"  # Prioritize uncertain predictions
    EXPECTED_VALUE = "expected_value"     # Maximize expected scientific value
    ACTIVE_LEARNING = "active_learning"   # Optimize for model improvement


@dataclass
class TriageParameters:
    """Parameters governing triage efficiency."""
    strategy: TriageStrategy = TriageStrategy.CONFIDENCE_RANKED

    # Base success rate without intelligent triage (FIFO)
    base_success_rate: float = 0.01  # 1% of random hypotheses validate

    # Success rate improvement from intelligent triage
    # Confidence ranking typically achieves 10-50x improvement
    triage_multiplier: float = 10.0

    # Computational cost per hypothesis for triage (relative to validation)
    triage_cost_ratio: float = 0.001  # Triage is 1000x cheaper than validation

    # Maximum triage throughput (hypotheses that can be scored per year)
    max_triage_throughput: float = 1e9  # 1 billion/year (computational)

    # Learning rate: how fast triage improves with more data
    learning_rate: float = 0.1  # 10% improvement per doubling of validated data


@dataclass
class DomainTriageProfile:
    """Domain-specific triage characteristics."""
    name: str
    base_generation_rate: float      # Hypotheses/year at t=0
    generation_growth_rate: float    # Annual growth in generation capacity
    max_generation_rate: float       # Ceiling on generation

    base_validation_capacity: float  # Validations/year at t=0
    validation_growth_rate: float    # Annual growth in validation capacity
    max_validation_capacity: float   # Ceiling on validation

    base_success_rate: float         # Fraction of validated that succeed (no triage)
    max_triage_improvement: float    # Maximum improvement from intelligent triage

    # Whether the domain can use simulation to bypass physical validation
    simulation_bypass_potential: float = 0.0  # 0-1, fraction replaceable


# Domain profiles based on case study data
DOMAIN_TRIAGE_PROFILES = {
    "materials_science": DomainTriageProfile(
        name="Materials Science",
        base_generation_rate=100_000,      # Pre-GNoME rate
        generation_growth_rate=0.50,       # 50% annual growth (AI improving)
        max_generation_rate=100_000_000,   # 100M/year ceiling
        base_validation_capacity=500,      # Manual synthesis rate
        validation_growth_rate=0.10,       # 10% growth (slow automation)
        max_validation_capacity=50_000,    # With full automation
        base_success_rate=0.01,            # 1% random success
        max_triage_improvement=20.0,       # 20x with good AI selection
        simulation_bypass_potential=0.3,   # 30% can be simulated
    ),
    "drug_discovery": DomainTriageProfile(
        name="Drug Discovery",
        base_generation_rate=10_000,       # Compound designs/year
        generation_growth_rate=0.30,       # 30% annual growth
        max_generation_rate=10_000_000,    # 10M/year ceiling
        base_validation_capacity=1_000,    # HTS capacity
        validation_growth_rate=0.15,       # 15% growth
        max_validation_capacity=100_000,   # High-throughput limit
        base_success_rate=0.001,           # 0.1% hit rate
        max_triage_improvement=50.0,       # 50x with good AI
        simulation_bypass_potential=0.2,   # ADMET prediction partial bypass
    ),
    "protein_design": DomainTriageProfile(
        name="Protein Design",
        base_generation_rate=1_000_000,    # Sequence designs/year
        generation_growth_rate=0.40,       # 40% annual growth
        max_generation_rate=1_000_000_000, # 1B/year (ESM-3 scale)
        base_validation_capacity=500,      # Expression/assay capacity
        validation_growth_rate=0.20,       # 20% growth
        max_validation_capacity=10_000,    # Automated expression
        base_success_rate=0.001,           # 0.1% functional rate
        max_triage_improvement=100.0,      # 100x with structural AI
        simulation_bypass_potential=0.4,   # AlphaFold structure validation
    ),
    "clinical_genomics": DomainTriageProfile(
        name="Clinical Genomics",
        base_generation_rate=1_000_000,    # Variant classifications/year
        generation_growth_rate=0.60,       # 60% annual growth
        max_generation_rate=1_000_000_000, # All human variants
        base_validation_capacity=10_000,   # Clinical validation capacity
        validation_growth_rate=0.10,       # 10% growth (regulatory limited)
        max_validation_capacity=100_000,   # Registry capacity
        base_success_rate=0.10,            # 10% true positive rate
        max_triage_improvement=10.0,       # 10x with AI confidence
        simulation_bypass_potential=0.5,   # Computational validation
    ),
    "structural_biology": DomainTriageProfile(
        name="Structural Biology",
        base_generation_rate=10_000,       # Structure predictions/year
        generation_growth_rate=0.80,       # 80% growth (AlphaFold effect)
        max_generation_rate=1_000_000_000, # All protein structures
        base_validation_capacity=5_000,    # Cryo-EM/X-ray capacity
        validation_growth_rate=0.15,       # 15% growth
        max_validation_capacity=50_000,    # Automated cryo-EM
        base_success_rate=0.50,            # 50% predictions validate
        max_triage_improvement=2.0,        # Limited improvement (already good)
        simulation_bypass_potential=0.8,   # High confidence predictions
    ),
}


@dataclass
class TriageState:
    """State of the triage system at a point in time."""
    year: int
    generation_rate: float          # Hypotheses generated this year
    validation_capacity: float      # Validation capacity this year
    backlog: float                  # Accumulated unvalidated hypotheses
    triage_efficiency: float        # Current success rate after triage
    validated_this_year: float      # Hypotheses validated
    successes_this_year: float      # Successful validations
    cumulative_successes: float     # Total successes to date
    backlog_years: float            # Years to clear backlog at current rate
    triage_bottleneck: bool         # Is triage the limiting factor?
    validation_bottleneck: bool     # Is validation the limiting factor?


class TriageModel:
    """
    Models the dynamics of hypothesis generation, triage, and validation.

    Key dynamics:
    1. AI generates hypotheses at rate G(t), growing over time
    2. Physical validation has capacity V(t), growing more slowly
    3. When G > V, backlog accumulates: B(t+1) = B(t) + G(t) - V(t)
    4. Intelligent triage improves success rate: T(t) = T_base * improvement(t)
    5. Effective acceleration = successes per year / baseline successes
    """

    def __init__(
        self,
        domain: str,
        triage_params: Optional[TriageParameters] = None,
        automation_scenario: AutomationScenario = AutomationScenario.BASELINE,
    ):
        self.domain = domain
        self.profile = DOMAIN_TRIAGE_PROFILES.get(domain, DOMAIN_TRIAGE_PROFILES["materials_science"])
        self.triage_params = triage_params or TriageParameters()
        self.automation_scenario = automation_scenario

        # Scenario multipliers for validation capacity growth
        self.scenario_multipliers = {
            AutomationScenario.CONSERVATIVE: 0.5,
            AutomationScenario.BASELINE: 1.0,
            AutomationScenario.OPTIMISTIC: 1.5,
            AutomationScenario.BREAKTHROUGH: 3.0,
        }

    def _generation_rate(self, year: int) -> float:
        """Calculate hypothesis generation rate at given year."""
        t = year - 2024  # Years since baseline
        rate = self.profile.base_generation_rate * np.exp(
            self.profile.generation_growth_rate * t
        )
        return min(rate, self.profile.max_generation_rate)

    def _validation_capacity(self, year: int) -> float:
        """Calculate validation capacity at given year."""
        t = year - 2024
        multiplier = self.scenario_multipliers[self.automation_scenario]

        # Logistic growth toward maximum
        growth_rate = self.profile.validation_growth_rate * multiplier
        capacity = self.profile.base_validation_capacity * np.exp(growth_rate * t)

        # Cap at maximum (which also scales with scenario)
        max_cap = self.profile.max_validation_capacity * (0.5 + 0.5 * multiplier)
        return min(capacity, max_cap)

    def _triage_efficiency(self, year: int, cumulative_validations: float) -> float:
        """
        Calculate triage efficiency (success rate after intelligent selection).

        Improves with:
        1. Time (better AI models)
        2. Cumulative validated data (learning)
        """
        t = year - 2024

        # Base improvement from AI progress (logistic growth)
        ai_improvement = 1 + (self.profile.max_triage_improvement - 1) * (
            1 - np.exp(-0.2 * t)
        )

        # Learning improvement from validated data
        learning_improvement = 1 + self.triage_params.learning_rate * np.log2(
            1 + cumulative_validations / 1000
        )

        # Combined efficiency
        efficiency = self.profile.base_success_rate * ai_improvement * learning_improvement

        # Cap at reasonable maximum (can't exceed 100% success)
        return min(efficiency, 0.9)

    def _simulation_bypass(self, year: int) -> float:
        """
        Fraction of validation that can be done computationally.

        Grows over time as simulation tools improve.
        """
        t = year - 2024
        max_bypass = self.profile.simulation_bypass_potential

        # Logistic growth toward maximum bypass
        bypass = max_bypass * (1 - np.exp(-0.1 * t))
        return bypass

    def simulate(self, years: List[int]) -> Dict[int, TriageState]:
        """
        Simulate triage dynamics over the specified years.

        Returns state at each year including backlog, throughput, and bottlenecks.
        """
        results = {}

        # Initial state
        backlog = 0.0
        cumulative_successes = 0.0
        cumulative_validations = 0.0

        for year in sorted(years):
            # Calculate rates
            gen_rate = self._generation_rate(year)
            val_cap = self._validation_capacity(year)
            sim_bypass = self._simulation_bypass(year)

            # Effective validation includes simulation bypass
            effective_val_cap = val_cap / (1 - sim_bypass) if sim_bypass < 1 else val_cap * 10

            # Triage efficiency
            triage_eff = self._triage_efficiency(year, cumulative_validations)

            # Backlog dynamics
            backlog += gen_rate
            validated = min(backlog, effective_val_cap)
            backlog -= validated

            # Successes
            successes = validated * triage_eff
            cumulative_successes += successes
            cumulative_validations += validated

            # Calculate backlog clearance time
            if effective_val_cap > gen_rate:
                # Can clear backlog
                net_clearance = effective_val_cap - gen_rate
                backlog_years = backlog / net_clearance if net_clearance > 0 else float('inf')
            else:
                # Backlog growing
                backlog_years = float('inf')

            # Identify bottleneck
            triage_bottleneck = (
                triage_eff < 0.1 and  # Low success rate
                self.profile.max_triage_improvement > 5  # Room for improvement
            )
            validation_bottleneck = (
                gen_rate > effective_val_cap * 2 and  # Generation >> validation
                backlog > gen_rate * 5  # Significant backlog
            )

            results[year] = TriageState(
                year=year,
                generation_rate=gen_rate,
                validation_capacity=effective_val_cap,
                backlog=backlog,
                triage_efficiency=triage_eff,
                validated_this_year=validated,
                successes_this_year=successes,
                cumulative_successes=cumulative_successes,
                backlog_years=backlog_years,
                triage_bottleneck=triage_bottleneck,
                validation_bottleneck=validation_bottleneck,
            )

        return results

    def effective_acceleration(self, year: int, baseline_year: int = 2024) -> float:
        """
        Calculate effective acceleration compared to baseline.

        Acceleration = (successes/year at target) / (successes/year at baseline)
        """
        results = self.simulate([baseline_year, year])

        baseline_successes = results[baseline_year].successes_this_year
        target_successes = results[year].successes_this_year

        if baseline_successes > 0:
            return target_successes / baseline_successes
        return 1.0

    def summary(self, year: int = 2030) -> str:
        """Generate summary of triage model state."""
        results = self.simulate([2024, 2030, 2040, 2050])

        lines = [
            "=" * 70,
            f"TRIAGE MODEL: {self.profile.name}",
            "=" * 70,
            "",
            f"Automation Scenario: {self.automation_scenario.value}",
            f"Triage Strategy: {self.triage_params.strategy.value}",
            "",
            "DYNAMICS OVER TIME:",
            "-" * 70,
            f"{'Year':<10} {'Generation':<15} {'Validation':<15} {'Backlog':<15} {'Success Rate':<12}",
            "-" * 70,
        ]

        for y in [2024, 2030, 2040, 2050]:
            state = results[y]
            lines.append(
                f"{y:<10} {state.generation_rate:>12,.0f}/yr "
                f"{state.validation_capacity:>12,.0f}/yr "
                f"{state.backlog:>12,.0f} "
                f"{state.triage_efficiency:>10.1%}"
            )

        lines.extend([
            "-" * 70,
            "",
            "BOTTLENECK ANALYSIS (2030):",
        ])

        state_2030 = results[2030]
        if state_2030.validation_bottleneck:
            lines.append(f"  ⚠ VALIDATION BOTTLENECK: Generation ({state_2030.generation_rate:,.0f}/yr) >> Validation ({state_2030.validation_capacity:,.0f}/yr)")
            lines.append(f"  ⚠ Backlog: {state_2030.backlog:,.0f} hypotheses ({state_2030.backlog_years:.1f} years to clear)")
        elif state_2030.triage_bottleneck:
            lines.append(f"  ⚠ TRIAGE BOTTLENECK: Low success rate ({state_2030.triage_efficiency:.1%}) limits throughput")
        else:
            lines.append(f"  ✓ No critical bottleneck detected")

        lines.extend([
            "",
            f"EFFECTIVE ACCELERATION:",
            f"  2030: {self.effective_acceleration(2030):.1f}x",
            f"  2040: {self.effective_acceleration(2040):.1f}x",
            f"  2050: {self.effective_acceleration(2050):.1f}x",
        ])

        return "\n".join(lines)


def run_triage_comparison():
    """Compare triage dynamics across domains and scenarios."""
    print("=" * 70)
    print("TRIAGE & SELECTION MODEL: CROSS-DOMAIN COMPARISON")
    print("=" * 70)
    print()

    years = [2024, 2030, 2040, 2050]
    domains = list(DOMAIN_TRIAGE_PROFILES.keys())

    print(f"{'Domain':<20} {'2024 Accel':<12} {'2030 Accel':<12} {'2040 Accel':<12} {'2050 Accel':<12}")
    print("-" * 70)

    for domain in domains:
        model = TriageModel(domain=domain)
        accels = [model.effective_acceleration(y) for y in years]
        print(f"{domain:<20} {accels[0]:>10.1f}x {accels[1]:>10.1f}x {accels[2]:>10.1f}x {accels[3]:>10.1f}x")

    print("-" * 70)
    print()
    print("KEY INSIGHT: Domains with high generation rates but low validation capacity")
    print("(materials_science, protein_design) show moderate acceleration due to backlog.")
    print()
    print("BOTTLENECK STATUS BY DOMAIN (2030):")
    print("-" * 50)

    for domain in domains:
        model = TriageModel(domain=domain)
        results = model.simulate([2030])
        state = results[2030]

        if state.validation_bottleneck:
            status = f"⚠ VALIDATION ({state.backlog_years:.0f}yr backlog)"
        elif state.triage_bottleneck:
            status = f"⚠ TRIAGE ({state.triage_efficiency:.1%} success)"
        else:
            status = "✓ Balanced"

        print(f"  {domain:<20}: {status}")


if __name__ == "__main__":
    # Run comparison
    run_triage_comparison()

    print()
    print()

    # Detailed view of materials science (GNoME problem)
    model = TriageModel(domain="materials_science")
    print(model.summary())
