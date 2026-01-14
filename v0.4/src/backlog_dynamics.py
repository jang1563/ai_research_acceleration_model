"""
Backlog Dynamics Model
======================

Models the accumulation and processing of hypotheses when AI
generates candidates faster than they can be validated.

Key insight from GNoME case study:
- 2.2M materials predicted
- 350 materials synthesized/year
- 6,000+ year backlog

This module captures the Type I (Scale) shift dynamics where
AI creates a triage problem rather than direct acceleration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BacklogMetrics:
    """Metrics for hypothesis backlog at a point in time."""
    year: int
    hypotheses_generated: int        # Cumulative AI-generated hypotheses
    hypotheses_validated: int        # Cumulative validated
    backlog_size: int               # Current backlog
    validation_rate: float          # Hypotheses validated per year
    backlog_years: float            # Years to clear backlog at current rate
    triage_overhead: float          # Additional time spent on selection


@dataclass
class ValidationCapacity:
    """
    Capacity for validating hypotheses in a domain.

    Validation is typically physical (wet lab) and doesn't
    scale with AI capabilities.
    """
    name: str
    base_rate: float                # Validations per year (baseline)
    automation_rate: float          # With automation (A-Lab, etc.)
    max_rate: float                 # Theoretical maximum
    current_rate: float = None
    bottleneck: str = "synthesis"   # What limits validation

    def __post_init__(self):
        if self.current_rate is None:
            self.current_rate = self.base_rate


# Validation capacities by domain (from case studies)
VALIDATION_CAPACITIES = {
    "materials_science": ValidationCapacity(
        name="Materials Synthesis",
        base_rate=100,               # Traditional synthesis per lab/year
        automation_rate=350,         # A-Lab rate
        max_rate=1000,              # With multiple automated labs
        bottleneck="synthesis",
    ),
    "structural_biology": ValidationCapacity(
        name="Structure Validation",
        base_rate=5000,              # Experimental structures per year globally
        automation_rate=10000,       # With improved pipelines
        max_rate=50000,
        bottleneck="crystallization",
    ),
    "protein_design": ValidationCapacity(
        name="Protein Expression & Testing",
        base_rate=1000,              # Proteins tested per year per lab
        automation_rate=5000,        # With automation
        max_rate=20000,
        bottleneck="expression",
    ),
    "drug_discovery": ValidationCapacity(
        name="Drug Candidates",
        base_rate=50,                # Compounds to clinical trials per year
        automation_rate=100,
        max_rate=200,
        bottleneck="clinical_trials",
    ),
}


class BacklogModel:
    """
    Models hypothesis generation and validation dynamics.

    Captures the Type I (Scale) shift pattern where AI generates
    hypotheses faster than they can be validated.
    """

    def __init__(
        self,
        domain: str,
        ai_generation_rate: float = 1000000,  # Hypotheses/year AI can generate
        initial_backlog: int = 0,
        triage_efficiency: float = 0.01,      # Fraction selected for validation
    ):
        """
        Initialize backlog model.

        Args:
            domain: Domain for validation capacity
            ai_generation_rate: How many hypotheses AI generates per year
            initial_backlog: Starting backlog
            triage_efficiency: What fraction of hypotheses are worth validating
        """
        if domain not in VALIDATION_CAPACITIES:
            raise ValueError(f"Unknown domain: {domain}")

        self.domain = domain
        self.capacity = VALIDATION_CAPACITIES[domain]
        self.ai_generation_rate = ai_generation_rate
        self.triage_efficiency = triage_efficiency

        # State
        self.cumulative_generated = initial_backlog
        self.cumulative_validated = 0
        self.backlog_history: List[BacklogMetrics] = []

    @property
    def current_backlog(self) -> int:
        """Current hypothesis backlog (worth validating but not yet validated)."""
        worth_validating = int(self.cumulative_generated * self.triage_efficiency)
        return max(0, worth_validating - self.cumulative_validated)

    @property
    def backlog_years(self) -> float:
        """Years to clear current backlog at current validation rate."""
        if self.capacity.current_rate <= 0:
            return float('inf')
        return self.current_backlog / self.capacity.current_rate

    def simulate_year(self, year: int, ai_generation_multiplier: float = 1.0) -> BacklogMetrics:
        """
        Simulate one year of hypothesis generation and validation.

        Args:
            year: The year being simulated
            ai_generation_multiplier: Multiplier on AI generation rate

        Returns:
            BacklogMetrics for this year
        """
        # AI generates hypotheses
        new_hypotheses = int(self.ai_generation_rate * ai_generation_multiplier)
        self.cumulative_generated += new_hypotheses

        # Validation proceeds at capacity rate
        validated_this_year = min(
            self.capacity.current_rate,
            self.current_backlog
        )
        self.cumulative_validated += validated_this_year

        # Calculate triage overhead
        # More backlog = more time spent selecting what to validate
        triage_overhead = np.log10(max(1, self.current_backlog)) / 10  # Rough estimate

        metrics = BacklogMetrics(
            year=year,
            hypotheses_generated=self.cumulative_generated,
            hypotheses_validated=self.cumulative_validated,
            backlog_size=self.current_backlog,
            validation_rate=self.capacity.current_rate,
            backlog_years=self.backlog_years,
            triage_overhead=triage_overhead,
        )

        self.backlog_history.append(metrics)
        return metrics

    def simulate_trajectory(
        self,
        start_year: int,
        end_year: int,
        ai_growth_rate: float = 0.4,
        automation_growth_rate: float = 0.1,
    ) -> List[BacklogMetrics]:
        """
        Simulate backlog trajectory over multiple years.

        Args:
            start_year: Starting year
            end_year: Ending year
            ai_growth_rate: Annual growth in AI generation capability
            automation_growth_rate: Annual growth in validation capacity

        Returns:
            List of BacklogMetrics for each year
        """
        self.backlog_history = []

        for year in range(start_year, end_year + 1):
            t = year - start_year

            # AI generation grows exponentially
            ai_multiplier = (1 + ai_growth_rate) ** t

            # Validation capacity grows more slowly
            self.capacity.current_rate = self.capacity.base_rate * (
                1 + automation_growth_rate
            ) ** t
            self.capacity.current_rate = min(
                self.capacity.current_rate,
                self.capacity.max_rate
            )

            self.simulate_year(year, ai_multiplier)

        return self.backlog_history

    def effective_acceleration(self, year: int) -> float:
        """
        Calculate effective acceleration considering backlog.

        For Type I shifts, the "acceleration" in hypothesis generation
        doesn't translate to research throughput because of the backlog.

        Returns:
            Effective acceleration (validated hypotheses per year / baseline)
        """
        # Find metrics for this year
        metrics = None
        for m in self.backlog_history:
            if m.year == year:
                metrics = m
                break

        if metrics is None:
            return 1.0

        # Effective acceleration is validation rate, not generation rate
        baseline_validation = self.capacity.base_rate
        return metrics.validation_rate / baseline_validation

    def summary(self) -> str:
        """Generate model summary."""
        lines = [
            "=" * 60,
            f"BACKLOG DYNAMICS - {self.domain.upper()}",
            "=" * 60,
            "",
            f"AI Generation Rate: {self.ai_generation_rate:,}/year",
            f"Triage Efficiency: {self.triage_efficiency:.1%}",
            f"Validation Capacity: {self.capacity.current_rate:,.0f}/year",
            f"Validation Bottleneck: {self.capacity.bottleneck}",
            "",
            "Current State:",
            f"  Total Generated: {self.cumulative_generated:,}",
            f"  Worth Validating: {int(self.cumulative_generated * self.triage_efficiency):,}",
            f"  Validated: {self.cumulative_validated:,}",
            f"  Backlog: {self.current_backlog:,}",
            f"  Years to Clear: {self.backlog_years:,.0f}",
            "",
        ]

        if self.backlog_history:
            lines.append("Trajectory:")
            lines.append("-" * 40)
            for m in self.backlog_history[-5:]:
                lines.append(
                    f"  {m.year}: Backlog {m.backlog_size:,} "
                    f"({m.backlog_years:,.0f} years to clear)"
                )

        return "\n".join(lines)


def gnome_backlog_simulation():
    """
    Simulate GNoME-style backlog dynamics.

    Demonstrates the Type I shift problem: massive hypothesis generation
    but physical validation unchanged.
    """
    print("=" * 60)
    print("GNoME BACKLOG SIMULATION")
    print("=" * 60)
    print()

    # GNoME generated 2.2M materials in 2023
    model = BacklogModel(
        domain="materials_science",
        ai_generation_rate=2200000,  # GNoME-scale generation
        initial_backlog=0,
        triage_efficiency=0.01,      # 1% worth synthesizing
    )

    # Simulate from 2023 to 2050
    trajectory = model.simulate_trajectory(
        start_year=2023,
        end_year=2050,
        ai_growth_rate=0.5,          # AI keeps improving
        automation_growth_rate=0.15, # A-Lab scaling
    )

    print("Year-by-Year Backlog:")
    print("-" * 50)
    for m in trajectory[::5]:  # Every 5 years
        print(f"  {m.year}:")
        print(f"    Generated: {m.hypotheses_generated:,}")
        print(f"    Validated: {m.hypotheses_validated:,}")
        print(f"    Backlog: {m.backlog_size:,}")
        print(f"    Years to clear: {m.backlog_years:,.0f}")
        print()

    print("Key Insight:")
    print("-" * 50)
    print("Even with 15% annual growth in automation,")
    print("backlog continues to grow because AI generation")
    print("outpaces physical validation.")
    print()
    print("Type I shifts create selection problems, not speed.")


def type_i_vs_type_iii_comparison():
    """
    Compare Type I (scale) vs Type III (capability) shift dynamics.
    """
    print("=" * 60)
    print("TYPE I vs TYPE III SHIFT COMPARISON")
    print("=" * 60)
    print()

    print("TYPE I (Scale) - GNoME Pattern:")
    print("-" * 40)
    print("  Hypothesis generation: 2,200,000x increase")
    print("  Validation rate: 1x (unchanged)")
    print("  Effective acceleration: ~1x per validated material")
    print("  Creates: Massive backlog, selection problem")
    print()

    print("TYPE III (Capability) - AlphaFold Pattern:")
    print("-" * 40)
    print("  Core task: 36,500x acceleration")
    print("  Validation rate: 1.5x (slightly faster)")
    print("  Effective acceleration: ~24x end-to-end")
    print("  Creates: New abilities, faster research")
    print()

    print("Implication for Model:")
    print("-" * 40)
    print("  Type I shifts should NOT use standard acceleration formula")
    print("  Instead model: backlog accumulation, triage overhead")
    print("  Effective acceleration = validation rate improvement, not generation")


if __name__ == "__main__":
    gnome_backlog_simulation()
    print()
    type_i_vs_type_iii_comparison()
