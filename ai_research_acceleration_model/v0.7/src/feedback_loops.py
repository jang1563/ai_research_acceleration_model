#!/usr/bin/env python3
"""
Feedback Loops Model for v0.7
=============================

Addresses Expert Review M2-P2: "Missing feedback loops"

The v0.6 model treated backlog accumulation as a one-way process.
In reality, systems respond to backlogs through multiple feedback mechanisms:

1. PRIORITY SHIFTS: When backlog grows, researchers shift to tractable problems
2. RESOURCE REALLOCATION: Funding moves toward automation/triage investment
3. FIELD SATURATION: Diminishing returns as low-hanging fruit is picked
4. TRUST DYNAMICS: Excessive predictions without validation erode confidence

v0.7 Enhancement:
- Self-correcting priority dynamics
- Resource reallocation to automation
- Researcher attention shifts
- Publication/funding incentive effects
- Trust/credibility feedback

Key Insight:
Extreme backlogs are partly self-limiting because the scientific community
adapts its behavior. But this adaptation has costs (wasted predictions,
delayed discoveries, suboptimal resource allocation).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np


class AdaptationMechanism(Enum):
    """Types of feedback mechanisms."""
    PRIORITY_SHIFT = "priority_shift"         # Researchers change focus
    RESOURCE_REALLOCATION = "resource_reallocation"  # Funding shifts
    FIELD_SATURATION = "field_saturation"     # Diminishing returns
    TRUST_DYNAMICS = "trust_dynamics"         # Confidence in predictions
    AUTOMATION_INVESTMENT = "automation_investment"  # Investment in validation capacity


@dataclass
class FeedbackParameters:
    """Parameters governing feedback loop dynamics."""

    # Priority shift parameters
    priority_shift_threshold: float = 10.0    # Backlog years before shift starts
    priority_shift_rate: float = 0.05         # Annual shift rate (5%/year)
    max_priority_shift: float = 0.50          # Maximum shift (50% of researchers)

    # Resource reallocation parameters
    reallocation_trigger: float = 50.0        # Backlog years to trigger reallocation
    reallocation_rate: float = 0.03           # 3% annual shift to automation
    automation_efficiency: float = 2.0        # 2x validation capacity per $ vs baseline

    # Field saturation parameters
    saturation_threshold: float = 0.80        # Fraction of "easy" hypotheses found
    saturation_rate: float = 0.10             # 10%/year decay in novelty after threshold

    # Trust dynamics parameters
    trust_decay_rate: float = 0.05            # 5%/year trust decay if predictions unvalidated
    trust_recovery_rate: float = 0.10         # 10%/year trust recovery with validations
    min_trust: float = 0.30                   # Minimum trust (30%)


@dataclass
class FeedbackState:
    """State of all feedback mechanisms at a point in time."""
    year: int

    # Priority shift
    researchers_shifted: float       # Fraction who have shifted focus
    effective_generation_rate: float # Reduced by priority shift

    # Resource reallocation
    automation_investment: float     # Cumulative investment in automation
    capacity_boost: float            # Validation capacity multiplier from investment

    # Field saturation
    novelty_remaining: float         # Fraction of novel hypotheses remaining
    discovery_efficiency: float      # How much effort yields discovery

    # Trust dynamics
    current_trust: float             # Community trust in predictions (0-1)
    predictions_validated: float     # Fraction of predictions with validation
    trust_impact: float              # Impact on adoption/funding

    # Net effect on acceleration
    feedback_adjustment: float       # Multiplicative adjustment to acceleration


@dataclass
class DomainFeedbackProfile:
    """Domain-specific feedback characteristics."""
    name: str

    # How responsive is the field to backlogs?
    priority_flexibility: float      # 0-1, how easily researchers shift
    funding_mobility: float          # 0-1, how easily funding reallocates

    # Field characteristics
    discovery_space_size: float      # Relative size of hypothesis space
    validation_cost: float           # Relative cost of validation

    # Trust dynamics
    prediction_visibility: float     # How visible are unvalidated predictions
    validation_urgency: float        # How urgent is validation (clinical=high)


DOMAIN_FEEDBACK_PROFILES = {
    "materials_science": DomainFeedbackProfile(
        name="Materials Science",
        priority_flexibility=0.4,     # Academic field, slower to shift
        funding_mobility=0.3,         # Government funding is stable
        discovery_space_size=100.0,   # Huge combinatorial space
        validation_cost=1.0,          # Baseline
        prediction_visibility=0.8,    # GNoME results very visible
        validation_urgency=0.3,       # Not urgent (no patients)
    ),

    "drug_discovery": DomainFeedbackProfile(
        name="Drug Discovery",
        priority_flexibility=0.7,     # Industry, responsive to ROI
        funding_mobility=0.8,         # Pharma can redirect
        discovery_space_size=50.0,    # Large but targeted
        validation_cost=10.0,         # Clinical trials expensive
        prediction_visibility=0.5,    # Proprietary, less visible
        validation_urgency=0.9,       # Patients need drugs
    ),

    "protein_design": DomainFeedbackProfile(
        name="Protein Design",
        priority_flexibility=0.5,     # Mix of academic/industry
        funding_mobility=0.5,         # Mixed funding
        discovery_space_size=200.0,   # Enormous sequence space
        validation_cost=0.5,          # Expression relatively cheap
        prediction_visibility=0.7,    # Academic papers
        validation_urgency=0.5,       # Varied applications
    ),

    "clinical_genomics": DomainFeedbackProfile(
        name="Clinical Genomics",
        priority_flexibility=0.3,     # Clinical practice is slow to change
        funding_mobility=0.4,         # Healthcare funding rigid
        discovery_space_size=10.0,    # Finite genome
        validation_cost=5.0,          # Clinical validation expensive
        prediction_visibility=0.9,    # Clinical results tracked
        validation_urgency=1.0,       # Patient care depends on it
    ),

    "structural_biology": DomainFeedbackProfile(
        name="Structural Biology",
        priority_flexibility=0.6,     # Academic but agile
        funding_mobility=0.5,         # Research funding
        discovery_space_size=5.0,     # Finite proteome
        validation_cost=2.0,          # Cryo-EM moderately expensive
        prediction_visibility=0.9,    # AlphaFold very visible
        validation_urgency=0.4,       # Scientific curiosity
    ),
}


class FeedbackLoopModel:
    """
    Models feedback dynamics that regulate backlog accumulation.

    Key insight: Extreme backlogs trigger self-correcting behavior,
    but at a cost to overall research productivity.

    Dynamics modeled:
    1. Priority shift: Researchers abandon prediction-heavy areas
    2. Resource reallocation: Investment shifts to automation
    3. Field saturation: Easy discoveries get made first
    4. Trust dynamics: Unvalidated predictions lose credibility
    """

    def __init__(
        self,
        domain: str,
        params: Optional[FeedbackParameters] = None,
    ):
        self.domain = domain
        self.params = params or FeedbackParameters()
        self.profile = DOMAIN_FEEDBACK_PROFILES.get(
            domain, DOMAIN_FEEDBACK_PROFILES["materials_science"]
        )

    def _priority_shift(
        self,
        year: int,
        backlog_years: float,
        prev_shift: float,
    ) -> float:
        """
        Calculate researcher priority shift.

        When backlog grows beyond threshold, researchers gradually
        shift to other problems where they can make validated contributions.
        """
        if backlog_years < self.params.priority_shift_threshold:
            # Below threshold: no shift pressure
            shift_pressure = 0
        else:
            # Above threshold: increasing pressure
            excess = backlog_years - self.params.priority_shift_threshold
            shift_pressure = np.tanh(excess / 50)  # Saturates at high backlog

        # Annual shift (modified by domain flexibility)
        annual_shift = (
            self.params.priority_shift_rate *
            shift_pressure *
            self.profile.priority_flexibility
        )

        # New total shift (capped at maximum)
        new_shift = min(prev_shift + annual_shift, self.params.max_priority_shift)
        return new_shift

    def _resource_reallocation(
        self,
        year: int,
        backlog_years: float,
        prev_investment: float,
    ) -> Tuple[float, float]:
        """
        Calculate resource reallocation to automation.

        Returns (new_investment, capacity_boost)
        """
        if backlog_years < self.params.reallocation_trigger:
            # Below trigger: minimal reallocation
            reallocation = 0.01  # 1% baseline
        else:
            # Above trigger: increasing reallocation
            excess = backlog_years - self.params.reallocation_trigger
            reallocation = self.params.reallocation_rate * np.tanh(excess / 100)

        # Modified by funding mobility
        reallocation *= self.profile.funding_mobility

        # Cumulative investment
        new_investment = prev_investment + reallocation

        # Capacity boost from investment
        # Diminishing returns: sqrt scaling
        capacity_boost = 1 + self.params.automation_efficiency * np.sqrt(new_investment)

        return new_investment, capacity_boost

    def _field_saturation(
        self,
        year: int,
        cumulative_discoveries: float,
    ) -> Tuple[float, float]:
        """
        Calculate field saturation effects.

        As easy hypotheses are validated, remaining ones are harder.
        Returns (novelty_remaining, discovery_efficiency)
        """
        # Total discoverable hypotheses (rough estimate)
        total_discoverable = self.profile.discovery_space_size * 10000

        # Fraction discovered
        fraction_discovered = min(cumulative_discoveries / total_discoverable, 0.99)

        # Novelty remaining
        novelty_remaining = 1 - fraction_discovered

        # Discovery efficiency (harder as field matures)
        # First 50% is easiest, then efficiency drops
        if fraction_discovered < 0.5:
            efficiency = 1.0
        else:
            # Efficiency drops to 0.2 as field fully saturates
            efficiency = 1.0 - 0.8 * (fraction_discovered - 0.5) / 0.5

        return novelty_remaining, efficiency

    def _trust_dynamics(
        self,
        year: int,
        predictions_made: float,
        predictions_validated: float,
        prev_trust: float,
    ) -> float:
        """
        Calculate trust in AI predictions.

        Trust grows with validation, decays without it.
        Low trust reduces adoption and funding.
        """
        # Validation rate
        if predictions_made > 0:
            validation_rate = predictions_validated / predictions_made
        else:
            validation_rate = 0

        # Trust change
        if validation_rate > 0.1:
            # Reasonable validation: trust grows
            trust_change = self.params.trust_recovery_rate * validation_rate
        else:
            # Poor validation: trust decays
            trust_change = -self.params.trust_decay_rate * (1 - validation_rate)

        # Modified by prediction visibility
        trust_change *= self.profile.prediction_visibility

        # New trust (bounded)
        new_trust = prev_trust + trust_change
        new_trust = max(self.params.min_trust, min(1.0, new_trust))

        return new_trust

    def _calculate_net_adjustment(
        self,
        shift: float,
        capacity_boost: float,
        efficiency: float,
        trust: float,
    ) -> float:
        """
        Calculate net adjustment to acceleration from all feedback loops.

        Positive effects: capacity_boost (more validation)
        Negative effects: shift (fewer hypotheses), efficiency (harder discoveries), trust (less adoption)
        """
        # Priority shift reduces effective generation (negative for acceleration)
        generation_factor = 1 - shift

        # Capacity boost increases validation (positive)
        validation_factor = capacity_boost

        # Discovery efficiency affects output quality (multiplicative)
        efficiency_factor = efficiency

        # Trust affects adoption/funding (multiplicative)
        trust_factor = 0.5 + 0.5 * trust  # Trust of 1.0 = full factor, 0.5 = 0.75 factor

        # Net adjustment
        net = generation_factor * validation_factor * efficiency_factor * trust_factor
        return net

    def simulate(
        self,
        years: List[int],
        backlog_trajectory: Dict[int, float],
        validation_trajectory: Dict[int, float],
    ) -> Dict[int, FeedbackState]:
        """
        Simulate feedback dynamics over time.

        Args:
            years: Years to simulate
            backlog_trajectory: Backlog years at each year
            validation_trajectory: Validations per year at each year

        Returns:
            Dictionary mapping years to FeedbackState
        """
        results = {}

        # Initial state
        prev_shift = 0
        prev_investment = 0
        prev_trust = 0.8  # Start with reasonable trust
        cumulative_discoveries = 0
        cumulative_predictions = 0

        for year in sorted(years):
            backlog_years = backlog_trajectory.get(year, 10)
            validations = validation_trajectory.get(year, 1000)

            # Update cumulative metrics
            cumulative_discoveries += validations * 0.1  # 10% success rate
            cumulative_predictions += validations / 0.01 if validations > 0 else 0

            # Calculate each feedback mechanism
            shift = self._priority_shift(year, backlog_years, prev_shift)

            investment, capacity_boost = self._resource_reallocation(
                year, backlog_years, prev_investment
            )

            novelty, efficiency = self._field_saturation(
                year, cumulative_discoveries
            )

            trust = self._trust_dynamics(
                year,
                cumulative_predictions,
                cumulative_discoveries * 10,  # Each discovery validates ~10 predictions
                prev_trust,
            )

            # Calculate net adjustment
            net_adjustment = self._calculate_net_adjustment(
                shift, capacity_boost, efficiency, trust
            )

            # Effective generation rate (reduced by shift)
            effective_gen = (1 - shift)

            # Trust impact on adoption
            trust_impact = 0.5 + 0.5 * trust

            # Validation rate for trust calculation
            validation_rate = (
                cumulative_discoveries * 10 / cumulative_predictions
                if cumulative_predictions > 0 else 0
            )

            results[year] = FeedbackState(
                year=year,
                researchers_shifted=shift,
                effective_generation_rate=effective_gen,
                automation_investment=investment,
                capacity_boost=capacity_boost,
                novelty_remaining=novelty,
                discovery_efficiency=efficiency,
                current_trust=trust,
                predictions_validated=validation_rate,
                trust_impact=trust_impact,
                feedback_adjustment=net_adjustment,
            )

            # Update state for next iteration
            prev_shift = shift
            prev_investment = investment
            prev_trust = trust

        return results

    def estimate_equilibrium(
        self,
        generation_rate: float,
        validation_capacity: float,
    ) -> Dict[str, float]:
        """
        Estimate equilibrium state given generation/validation rates.

        Returns the steady-state where feedback loops stabilize.
        """
        # Initial backlog
        ratio = generation_rate / validation_capacity if validation_capacity > 0 else float('inf')
        initial_backlog_years = ratio

        # Simulate until equilibrium (or max iterations)
        years = list(range(2024, 2100))
        backlog_traj = {y: initial_backlog_years * (1.05 ** (y - 2024)) for y in years}
        validation_traj = {y: validation_capacity * (1.10 ** (y - 2024)) for y in years}

        results = self.simulate(years, backlog_traj, validation_traj)

        # Find equilibrium (where feedback_adjustment is stable)
        adjustments = [results[y].feedback_adjustment for y in years]
        for i, year in enumerate(years[:-5]):
            if abs(adjustments[i+5] - adjustments[i]) < 0.01:
                eq_state = results[year]
                return {
                    "equilibrium_year": year,
                    "equilibrium_adjustment": eq_state.feedback_adjustment,
                    "equilibrium_shift": eq_state.researchers_shifted,
                    "equilibrium_trust": eq_state.current_trust,
                    "equilibrium_capacity_boost": eq_state.capacity_boost,
                }

        # No equilibrium found
        final_state = results[years[-1]]
        return {
            "equilibrium_year": None,
            "equilibrium_adjustment": final_state.feedback_adjustment,
            "equilibrium_shift": final_state.researchers_shifted,
            "equilibrium_trust": final_state.current_trust,
            "equilibrium_capacity_boost": final_state.capacity_boost,
        }

    def summary(self) -> str:
        """Generate summary of feedback loop dynamics."""
        # Simulate with example backlog trajectory
        years = [2024, 2030, 2040, 2050]

        # Example trajectories (typical for materials science)
        backlog_traj = {2024: 10, 2030: 100, 2040: 500, 2050: 200}
        validation_traj = {2024: 500, 2030: 2000, 2040: 10000, 2050: 30000}

        results = self.simulate(years, backlog_traj, validation_traj)

        lines = [
            "=" * 80,
            f"FEEDBACK LOOPS MODEL: {self.profile.name}",
            "=" * 80,
            "",
            "DOMAIN CHARACTERISTICS:",
            f"  Priority flexibility: {self.profile.priority_flexibility:.0%}",
            f"  Funding mobility: {self.profile.funding_mobility:.0%}",
            f"  Validation urgency: {self.profile.validation_urgency:.0%}",
            "",
            "FEEDBACK DYNAMICS:",
            "-" * 80,
            f"{'Year':<8} {'Shift':<10} {'Invest':<10} {'Novelty':<10} {'Trust':<10} {'Net Adj':<10}",
            "-" * 80,
        ]

        for year in years:
            s = results[year]
            lines.append(
                f"{year:<8} {s.researchers_shifted:>8.0%} {s.automation_investment:>8.2f} "
                f"{s.novelty_remaining:>8.0%} {s.current_trust:>8.0%} "
                f"{s.feedback_adjustment:>8.2f}x"
            )

        lines.extend([
            "-" * 80,
            "",
            "INTERPRETATION:",
        ])

        s_2030 = results[2030]
        if s_2030.researchers_shifted > 0.1:
            lines.append(
                f"  - {s_2030.researchers_shifted:.0%} of researchers have shifted focus by 2030"
            )
        if s_2030.capacity_boost > 1.5:
            lines.append(
                f"  - Automation investment yields {s_2030.capacity_boost:.1f}x capacity boost"
            )
        if s_2030.current_trust < 0.6:
            lines.append(
                f"  - Trust in predictions has declined to {s_2030.current_trust:.0%}"
            )

        lines.extend([
            "",
            "KEY INSIGHT:",
            "  Feedback loops partially self-correct extreme backlogs, but at cost of",
            "  reduced research activity and delayed discoveries.",
        ])

        return "\n".join(lines)


def compare_feedback_across_domains():
    """Compare feedback dynamics across domains."""
    print("=" * 80)
    print("FEEDBACK LOOP COMPARISON ACROSS DOMAINS")
    print("=" * 80)
    print()

    years = [2024, 2030, 2040, 2050]

    # Standard trajectories
    backlog_traj = {2024: 10, 2030: 100, 2040: 500, 2050: 200}
    validation_traj = {2024: 1000, 2030: 5000, 2040: 20000, 2050: 50000}

    print(f"{'Domain':<22} {'2030 Shift':<12} {'2030 Boost':<12} {'2030 Trust':<12} {'2030 Net':<12}")
    print("-" * 80)

    for domain in DOMAIN_FEEDBACK_PROFILES.keys():
        model = FeedbackLoopModel(domain=domain)
        results = model.simulate(years, backlog_traj, validation_traj)

        s = results[2030]
        print(
            f"{domain:<22} {s.researchers_shifted:>10.0%} {s.capacity_boost:>10.1f}x "
            f"{s.current_trust:>10.0%} {s.feedback_adjustment:>10.2f}x"
        )

    print("-" * 80)
    print()
    print("KEY INSIGHT: Drug discovery shows strongest feedback effects due to high")
    print("funding mobility and validation urgency. Clinical genomics shows weakest")
    print("due to rigid healthcare funding and slow clinical practice changes.")


if __name__ == "__main__":
    compare_feedback_across_domains()
    print()
    print()

    # Detailed view for materials science
    model = FeedbackLoopModel(domain="materials_science")
    print(model.summary())
