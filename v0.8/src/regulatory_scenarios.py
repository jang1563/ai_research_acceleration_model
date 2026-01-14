#!/usr/bin/env python3
"""
Regulatory Evolution Scenarios for v0.8
========================================

Addresses the recommendation for modeling regulatory changes,
particularly for drug discovery where clinical trials are the
dominant bottleneck.

Key Question:
What if regulators create AI-friendly approval pathways?

Scenarios Modeled:
1. STATUS_QUO: Current regulatory framework continues
2. INCREMENTAL: Modest reforms (adaptive trials, real-world evidence)
3. AI_ASSISTED: AI predictions accepted as supporting evidence
4. AI_PRIMARY: AI-based virtual trials partially replace physical trials
5. TRANSFORMED: Fundamental regulatory reform for AI-era therapeutics

Impact Analysis:
- Phase 1-3 clinical trial acceleration potential
- Time and cost savings
- Safety trade-offs
- Probability of each scenario
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class RegulatoryScenario(Enum):
    """Regulatory evolution scenarios."""
    STATUS_QUO = "status_quo"
    INCREMENTAL = "incremental"
    AI_ASSISTED = "ai_assisted"
    AI_PRIMARY = "ai_primary"
    TRANSFORMED = "transformed"


@dataclass
class ClinicalTrialRequirements:
    """Requirements for each clinical trial phase."""
    phase: str
    min_participants: int
    min_duration_months: int
    endpoints_required: List[str]
    ai_substitution_potential: float  # 0-1, how much can AI replace


@dataclass
class RegulatoryFramework:
    """A regulatory framework definition."""
    scenario: RegulatoryScenario
    name: str
    description: str

    # Timeline (years until implementation)
    earliest_implementation: int
    likely_implementation: int

    # Phase-specific acceleration
    phase1_acceleration: float
    phase2_acceleration: float
    phase3_acceleration: float
    approval_acceleration: float

    # Virtual trial potential
    virtual_trial_fraction: float  # Fraction of trials that can be virtual

    # Safety requirements
    safety_monitoring_level: str   # "standard", "enhanced", "continuous"
    post_market_requirements: str  # "standard", "enhanced", "real_time"

    # Probability
    probability: float

    # Prerequisites
    prerequisites: List[str]


# Standard clinical trial requirements (current)
CURRENT_REQUIREMENTS = {
    "phase1": ClinicalTrialRequirements(
        phase="Phase 1",
        min_participants=20,
        min_duration_months=6,
        endpoints_required=["safety", "pharmacokinetics", "dosing"],
        ai_substitution_potential=0.2,
    ),
    "phase2": ClinicalTrialRequirements(
        phase="Phase 2",
        min_participants=100,
        min_duration_months=12,
        endpoints_required=["efficacy_signal", "safety_profile", "dosing_optimization"],
        ai_substitution_potential=0.15,
    ),
    "phase3": ClinicalTrialRequirements(
        phase="Phase 3",
        min_participants=1000,
        min_duration_months=24,
        endpoints_required=["efficacy_vs_standard", "long_term_safety", "subgroup_analysis"],
        ai_substitution_potential=0.1,
    ),
}


# Define regulatory frameworks
REGULATORY_FRAMEWORKS = {
    RegulatoryScenario.STATUS_QUO: RegulatoryFramework(
        scenario=RegulatoryScenario.STATUS_QUO,
        name="Status Quo",
        description="Current FDA/EMA requirements remain unchanged",
        earliest_implementation=2024,
        likely_implementation=2024,
        phase1_acceleration=1.0,
        phase2_acceleration=1.0,
        phase3_acceleration=1.0,
        approval_acceleration=1.0,
        virtual_trial_fraction=0.0,
        safety_monitoring_level="standard",
        post_market_requirements="standard",
        probability=0.30,
        prerequisites=[],
    ),

    RegulatoryScenario.INCREMENTAL: RegulatoryFramework(
        scenario=RegulatoryScenario.INCREMENTAL,
        name="Incremental Reform",
        description="Adaptive trial designs, real-world evidence, faster approval for serious diseases",
        earliest_implementation=2025,
        likely_implementation=2027,
        phase1_acceleration=1.2,
        phase2_acceleration=1.3,
        phase3_acceleration=1.2,
        approval_acceleration=1.4,
        virtual_trial_fraction=0.05,
        safety_monitoring_level="enhanced",
        post_market_requirements="enhanced",
        probability=0.35,
        prerequisites=[
            "Successful pilot programs",
            "No major AI-related drug failures",
        ],
    ),

    RegulatoryScenario.AI_ASSISTED: RegulatoryFramework(
        scenario=RegulatoryScenario.AI_ASSISTED,
        name="AI-Assisted Evaluation",
        description="AI predictions accepted as supporting evidence, smaller trials for AI-confident cases",
        earliest_implementation=2027,
        likely_implementation=2030,
        phase1_acceleration=1.5,
        phase2_acceleration=1.8,
        phase3_acceleration=1.5,
        approval_acceleration=1.6,
        virtual_trial_fraction=0.15,
        safety_monitoring_level="enhanced",
        post_market_requirements="enhanced",
        probability=0.20,
        prerequisites=[
            "Validated AI prediction tools",
            "Regulatory AI expertise built",
            "Industry adoption of AI documentation",
        ],
    ),

    RegulatoryScenario.AI_PRIMARY: RegulatoryFramework(
        scenario=RegulatoryScenario.AI_PRIMARY,
        name="AI-Primary Pathways",
        description="Virtual trials with AI-predicted outcomes for select indications",
        earliest_implementation=2030,
        likely_implementation=2035,
        phase1_acceleration=2.0,
        phase2_acceleration=3.0,
        phase3_acceleration=2.5,
        approval_acceleration=2.0,
        virtual_trial_fraction=0.40,
        safety_monitoring_level="continuous",
        post_market_requirements="real_time",
        probability=0.10,
        prerequisites=[
            "Proven AI safety record (5+ years)",
            "Digital twin patient models validated",
            "Real-time safety monitoring infrastructure",
            "International regulatory harmonization",
        ],
    ),

    RegulatoryScenario.TRANSFORMED: RegulatoryFramework(
        scenario=RegulatoryScenario.TRANSFORMED,
        name="Transformed Paradigm",
        description="Fundamentally new approval framework designed for AI-era therapeutics",
        earliest_implementation=2035,
        likely_implementation=2045,
        phase1_acceleration=5.0,
        phase2_acceleration=10.0,
        phase3_acceleration=8.0,
        approval_acceleration=4.0,
        virtual_trial_fraction=0.80,
        safety_monitoring_level="continuous",
        post_market_requirements="real_time",
        probability=0.05,
        prerequisites=[
            "Paradigm shift in regulatory philosophy",
            "Perfect or near-perfect AI safety prediction",
            "Full digital twin capabilities",
            "Real-time patient monitoring ubiquitous",
            "Political will for regulatory reform",
        ],
    ),
}


@dataclass
class DrugDiscoveryForecast:
    """Forecast for drug discovery under a regulatory scenario."""
    regulatory_scenario: RegulatoryScenario
    year: int

    # Phase-by-phase acceleration
    phase1_accel: float
    phase2_accel: float
    phase3_accel: float
    approval_accel: float

    # End-to-end
    end_to_end_accel: float

    # Time savings
    time_saved_months: float
    cost_reduction_percent: float

    # Adoption status
    framework_active: bool
    adoption_fraction: float


class RegulatoryEvolutionModel:
    """
    Models how regulatory evolution affects drug discovery acceleration.

    Captures:
    - When different regulatory scenarios become active
    - How adoption ramps up over time
    - Trade-offs between speed and safety requirements
    """

    def __init__(self):
        self.frameworks = REGULATORY_FRAMEWORKS

    def _adoption_curve(
        self,
        year: int,
        earliest: int,
        likely: int,
    ) -> float:
        """Calculate adoption fraction using logistic curve."""
        if year < earliest:
            return 0.0

        # Midpoint is the "likely" year
        k = 0.3  # Steepness
        midpoint = likely

        adoption = 1 / (1 + np.exp(-k * (year - midpoint)))
        return min(adoption, 1.0)

    def _clinical_acceleration(
        self,
        framework: RegulatoryFramework,
        year: int,
    ) -> Tuple[float, float, float, float]:
        """Calculate phase-specific acceleration."""
        adoption = self._adoption_curve(
            year,
            framework.earliest_implementation,
            framework.likely_implementation,
        )

        # Blend with status quo based on adoption
        phase1 = 1.0 + (framework.phase1_acceleration - 1.0) * adoption
        phase2 = 1.0 + (framework.phase2_acceleration - 1.0) * adoption
        phase3 = 1.0 + (framework.phase3_acceleration - 1.0) * adoption
        approval = 1.0 + (framework.approval_acceleration - 1.0) * adoption

        return phase1, phase2, phase3, approval

    def forecast(
        self,
        scenario: RegulatoryScenario,
        year: int,
    ) -> DrugDiscoveryForecast:
        """Generate forecast for a specific regulatory scenario."""
        framework = self.frameworks[scenario]

        phase1, phase2, phase3, approval = self._clinical_acceleration(
            framework, year
        )

        # Calculate adoption
        adoption = self._adoption_curve(
            year,
            framework.earliest_implementation,
            framework.likely_implementation,
        )

        # End-to-end (weighted by phase duration)
        # Weights: Phase 1 = 10%, Phase 2 = 25%, Phase 3 = 45%, Approval = 20%
        end_to_end = (
            0.10 * phase1 +
            0.25 * phase2 +
            0.45 * phase3 +
            0.20 * approval
        )

        # Time saved (assuming 10-year baseline development)
        baseline_months = 120  # 10 years
        new_duration = baseline_months / end_to_end
        time_saved = baseline_months - new_duration

        # Cost reduction (roughly proportional to time)
        cost_reduction = (1 - 1/end_to_end) * 100

        return DrugDiscoveryForecast(
            regulatory_scenario=scenario,
            year=year,
            phase1_accel=phase1,
            phase2_accel=phase2,
            phase3_accel=phase3,
            approval_accel=approval,
            end_to_end_accel=end_to_end,
            time_saved_months=time_saved,
            cost_reduction_percent=cost_reduction,
            framework_active=adoption > 0.1,
            adoption_fraction=adoption,
        )

    def ensemble_forecast(
        self,
        year: int,
    ) -> Dict:
        """
        Probability-weighted ensemble across all regulatory scenarios.
        """
        forecasts = {}
        for scenario in RegulatoryScenario:
            forecasts[scenario] = self.forecast(scenario, year)

        # Calculate weighted mean
        accels = [forecasts[s].end_to_end_accel for s in RegulatoryScenario]
        probs = [self.frameworks[s].probability for s in RegulatoryScenario]

        weighted_mean = np.average(accels, weights=probs)

        # Weighted CI
        sorted_pairs = sorted(zip(accels, probs), key=lambda x: x[0])
        cumsum = 0
        ci_5 = accels[0]
        ci_95 = accels[-1]
        for accel, prob in sorted_pairs:
            cumsum += prob
            if cumsum >= 0.05 and ci_5 == accels[0]:
                ci_5 = accel
            if cumsum >= 0.95:
                ci_95 = accel
                break

        return {
            "year": year,
            "scenario_forecasts": forecasts,
            "weighted_mean": weighted_mean,
            "weighted_ci_90": (ci_5, ci_95),
        }

    def scenario_report(self, year: int = 2030) -> str:
        """Generate detailed regulatory scenario report."""
        lines = [
            "=" * 90,
            f"REGULATORY EVOLUTION SCENARIOS: DRUG DISCOVERY ({year})",
            "=" * 90,
            "",
            "FRAMEWORK DEFINITIONS:",
            "-" * 90,
        ]

        for scenario in RegulatoryScenario:
            fw = self.frameworks[scenario]
            lines.append(f"\n{fw.name} (P={fw.probability:.0%})")
            lines.append(f"  {fw.description}")
            lines.append(f"  Earliest: {fw.earliest_implementation}, Likely: {fw.likely_implementation}")
            lines.append(f"  Prerequisites:")
            for prereq in fw.prerequisites[:3]:
                lines.append(f"    - {prereq}")

        lines.extend([
            "",
            "-" * 90,
            "ACCELERATION BY SCENARIO:",
            "-" * 90,
            f"{'Scenario':<18} {'Phase 1':<10} {'Phase 2':<10} {'Phase 3':<10} {'Approval':<10} {'E2E':<10} {'Adopted':<10}",
            "-" * 90,
        ])

        for scenario in RegulatoryScenario:
            f = self.forecast(scenario, year)
            adopted = "Yes" if f.framework_active else "No"
            lines.append(
                f"{scenario.value:<18} {f.phase1_accel:>8.1f}x {f.phase2_accel:>8.1f}x "
                f"{f.phase3_accel:>8.1f}x {f.approval_accel:>8.1f}x {f.end_to_end_accel:>8.1f}x "
                f"{adopted:<10}"
            )

        # Ensemble
        ensemble = self.ensemble_forecast(year)
        lines.extend([
            "-" * 90,
            "",
            "PROBABILITY-WEIGHTED ENSEMBLE:",
            f"  Mean acceleration: {ensemble['weighted_mean']:.2f}x",
            f"  90% CI: [{ensemble['weighted_ci_90'][0]:.2f}x - {ensemble['weighted_ci_90'][1]:.2f}x]",
            "",
            "IMPLICATIONS:",
        ])

        # Key insights
        if year <= 2025:
            lines.append("  - Status quo likely dominates; limited regulatory change expected")
        elif year <= 2030:
            lines.append("  - Incremental reforms most likely; AI-assisted pathways emerging")
        else:
            lines.append("  - More transformative scenarios become plausible")
            lines.append("  - Significant uncertainty in regulatory evolution")

        return "\n".join(lines)

    def trajectory_analysis(self) -> str:
        """Analyze regulatory evolution trajectory over time."""
        years = [2025, 2030, 2035, 2040, 2050]

        lines = [
            "=" * 80,
            "REGULATORY EVOLUTION TRAJECTORY",
            "=" * 80,
            "",
            f"{'Year':<8} {'Status Quo':<12} {'Incremental':<12} {'AI-Assisted':<12} {'AI-Primary':<12} {'Ensemble':<12}",
            "-" * 80,
        ]

        for year in years:
            sq = self.forecast(RegulatoryScenario.STATUS_QUO, year)
            inc = self.forecast(RegulatoryScenario.INCREMENTAL, year)
            ai_a = self.forecast(RegulatoryScenario.AI_ASSISTED, year)
            ai_p = self.forecast(RegulatoryScenario.AI_PRIMARY, year)
            ens = self.ensemble_forecast(year)

            lines.append(
                f"{year:<8} {sq.end_to_end_accel:>10.1f}x {inc.end_to_end_accel:>10.1f}x "
                f"{ai_a.end_to_end_accel:>10.1f}x {ai_p.end_to_end_accel:>10.1f}x "
                f"{ens['weighted_mean']:>10.1f}x"
            )

        lines.extend([
            "-" * 80,
            "",
            "KEY INSIGHT: Phase 3 trials remain the dominant bottleneck",
            "under all but the most transformative regulatory scenarios.",
            "",
            "Even with AI-Primary pathways (10% probability by 2035),",
            "clinical trials still constrain drug discovery to ~2-3x acceleration.",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    model = RegulatoryEvolutionModel()

    # Current state
    print(model.scenario_report(2030))

    print()
    print()

    # Trajectory
    print(model.trajectory_analysis())
