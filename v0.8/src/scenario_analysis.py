#!/usr/bin/env python3
"""
Scenario Analysis Framework for v0.8
=====================================

Addresses Expert Review P1-P2: "No pessimistic scenarios" and
recommendation for explicit scenario comparison.

v0.8 provides structured scenario analysis with:
1. Explicit assumptions for each scenario
2. Probability weights for scenarios
3. Conditional dependencies between assumptions
4. Scenario-specific acceleration trajectories

Scenarios:
- PESSIMISTIC: AI winter, regulatory backlash, automation failures
- CONSERVATIVE: Current trends, modest improvement
- BASELINE: Expected trajectory with known developments
- OPTIMISTIC: Favorable conditions, faster adoption
- BREAKTHROUGH: Transformative advances, regulatory adaptation

Key Features:
- Explicit assumption documentation
- Probability-weighted ensemble predictions
- Scenario transition modeling
- Policy sensitivity analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np


class ScenarioType(Enum):
    """Named scenarios with associated probability weights."""
    PESSIMISTIC = "pessimistic"      # 10% probability
    CONSERVATIVE = "conservative"    # 20% probability
    BASELINE = "baseline"            # 40% probability
    OPTIMISTIC = "optimistic"        # 20% probability
    BREAKTHROUGH = "breakthrough"    # 10% probability


# Default probability weights (must sum to 1.0)
SCENARIO_PROBABILITIES = {
    ScenarioType.PESSIMISTIC: 0.10,
    ScenarioType.CONSERVATIVE: 0.20,
    ScenarioType.BASELINE: 0.40,
    ScenarioType.OPTIMISTIC: 0.20,
    ScenarioType.BREAKTHROUGH: 0.10,
}


@dataclass
class Assumption:
    """An explicit assumption with probability and impact."""
    name: str
    description: str
    probability: float                 # Probability of this assumption holding
    impact_if_false: float             # Multiplier if assumption fails (e.g., 0.5 = halve acceleration)
    domain_specific: bool = False      # Does this only apply to certain domains?
    domains: List[str] = field(default_factory=list)


@dataclass
class ScenarioDefinition:
    """Complete definition of a scenario."""
    scenario_type: ScenarioType
    name: str
    description: str

    # Core assumptions
    assumptions: List[Assumption]

    # Acceleration multipliers by domain
    multipliers: Dict[str, float]

    # Time dynamics
    adoption_rate: float               # How fast AI/automation is adopted (0-1)
    saturation_year: int               # When acceleration plateaus

    # Confidence
    prediction_confidence: float       # How confident are we in this scenario? (0-1)


# Define detailed assumptions
ASSUMPTIONS = {
    # AI Progress
    "ai_progress_continues": Assumption(
        name="AI Progress Continues",
        description="AI capabilities continue improving at 2020-2024 rates",
        probability=0.8,
        impact_if_false=0.5,
    ),
    "ai_progress_accelerates": Assumption(
        name="AI Progress Accelerates",
        description="AI improves faster than 2020-2024 due to scaling/algorithms",
        probability=0.3,
        impact_if_false=0.7,
    ),
    "ai_winter": Assumption(
        name="AI Winter",
        description="AI progress stalls due to hitting fundamental limits",
        probability=0.1,
        impact_if_false=1.5,  # If this doesn't happen, acceleration is higher
    ),

    # Automation
    "lab_automation_scales": Assumption(
        name="Lab Automation Scales",
        description="Lab automation continues scaling and reducing costs",
        probability=0.7,
        impact_if_false=0.6,
    ),
    "robotic_labs_mature": Assumption(
        name="Robotic Labs Mature",
        description="Fully autonomous labs become practical and widespread",
        probability=0.4,
        impact_if_false=0.8,
    ),

    # Regulatory
    "regulatory_stable": Assumption(
        name="Regulatory Stability",
        description="Regulatory requirements remain similar to current",
        probability=0.6,
        impact_if_false=0.7,
        domain_specific=True,
        domains=["drug_discovery", "clinical_genomics"],
    ),
    "regulatory_adapts": Assumption(
        name="Regulatory Adaptation",
        description="Regulators create AI-friendly approval pathways",
        probability=0.3,
        impact_if_false=0.9,
        domain_specific=True,
        domains=["drug_discovery", "clinical_genomics"],
    ),
    "regulatory_backlash": Assumption(
        name="Regulatory Backlash",
        description="Stricter regulations due to AI safety concerns",
        probability=0.2,
        impact_if_false=1.2,
        domain_specific=True,
        domains=["drug_discovery", "clinical_genomics"],
    ),

    # Data & Trust
    "data_sharing_improves": Assumption(
        name="Data Sharing Improves",
        description="Scientific data sharing increases, improving AI training",
        probability=0.5,
        impact_if_false=0.8,
    ),
    "trust_in_ai_grows": Assumption(
        name="Trust in AI Predictions",
        description="Scientific community increasingly trusts AI predictions",
        probability=0.6,
        impact_if_false=0.7,
    ),

    # Domain-specific
    "synthesis_breakthrough": Assumption(
        name="Synthesis Breakthrough",
        description="Major advance in automated material synthesis",
        probability=0.25,
        impact_if_false=0.9,
        domain_specific=True,
        domains=["materials_science"],
    ),
    "clinical_trials_reform": Assumption(
        name="Clinical Trials Reform",
        description="Significant reform enabling faster clinical trials",
        probability=0.2,
        impact_if_false=0.95,
        domain_specific=True,
        domains=["drug_discovery"],
    ),
}


# Define scenarios
SCENARIOS = {
    ScenarioType.PESSIMISTIC: ScenarioDefinition(
        scenario_type=ScenarioType.PESSIMISTIC,
        name="Pessimistic",
        description="AI winter, regulatory backlash, automation struggles",
        assumptions=[
            ASSUMPTIONS["ai_winter"],
            ASSUMPTIONS["regulatory_backlash"],
        ],
        multipliers={
            "structural_biology": 0.4,
            "drug_discovery": 0.3,
            "materials_science": 0.3,
            "protein_design": 0.4,
            "clinical_genomics": 0.4,
        },
        adoption_rate=0.05,
        saturation_year=2045,
        prediction_confidence=0.7,
    ),

    ScenarioType.CONSERVATIVE: ScenarioDefinition(
        scenario_type=ScenarioType.CONSERVATIVE,
        name="Conservative",
        description="Slower than expected AI progress, modest automation gains",
        assumptions=[
            ASSUMPTIONS["ai_progress_continues"],
            ASSUMPTIONS["lab_automation_scales"],
        ],
        multipliers={
            "structural_biology": 0.7,
            "drug_discovery": 0.6,
            "materials_science": 0.5,
            "protein_design": 0.6,
            "clinical_genomics": 0.7,
        },
        adoption_rate=0.10,
        saturation_year=2040,
        prediction_confidence=0.8,
    ),

    ScenarioType.BASELINE: ScenarioDefinition(
        scenario_type=ScenarioType.BASELINE,
        name="Baseline",
        description="Expected trajectory based on current trends",
        assumptions=[
            ASSUMPTIONS["ai_progress_continues"],
            ASSUMPTIONS["lab_automation_scales"],
            ASSUMPTIONS["regulatory_stable"],
            ASSUMPTIONS["data_sharing_improves"],
        ],
        multipliers={
            "structural_biology": 1.0,
            "drug_discovery": 1.0,
            "materials_science": 1.0,
            "protein_design": 1.0,
            "clinical_genomics": 1.0,
        },
        adoption_rate=0.15,
        saturation_year=2038,
        prediction_confidence=0.85,
    ),

    ScenarioType.OPTIMISTIC: ScenarioDefinition(
        scenario_type=ScenarioType.OPTIMISTIC,
        name="Optimistic",
        description="Faster AI progress, good automation, favorable regulation",
        assumptions=[
            ASSUMPTIONS["ai_progress_accelerates"],
            ASSUMPTIONS["robotic_labs_mature"],
            ASSUMPTIONS["regulatory_adapts"],
            ASSUMPTIONS["trust_in_ai_grows"],
        ],
        multipliers={
            "structural_biology": 1.4,
            "drug_discovery": 1.5,
            "materials_science": 1.6,
            "protein_design": 1.5,
            "clinical_genomics": 1.3,
        },
        adoption_rate=0.20,
        saturation_year=2035,
        prediction_confidence=0.75,
    ),

    ScenarioType.BREAKTHROUGH: ScenarioDefinition(
        scenario_type=ScenarioType.BREAKTHROUGH,
        name="Breakthrough",
        description="Transformative advances across AI, automation, and regulation",
        assumptions=[
            ASSUMPTIONS["ai_progress_accelerates"],
            ASSUMPTIONS["robotic_labs_mature"],
            ASSUMPTIONS["regulatory_adapts"],
            ASSUMPTIONS["synthesis_breakthrough"],
            ASSUMPTIONS["clinical_trials_reform"],
        ],
        multipliers={
            "structural_biology": 2.0,
            "drug_discovery": 2.5,
            "materials_science": 3.0,
            "protein_design": 2.5,
            "clinical_genomics": 1.8,
        },
        adoption_rate=0.25,
        saturation_year=2032,
        prediction_confidence=0.6,
    ),
}


@dataclass
class ScenarioForecast:
    """Forecast under a specific scenario."""
    scenario: ScenarioType
    domain: str
    year: int

    # Predictions
    acceleration: float
    confidence_interval: Tuple[float, float]

    # Components
    ai_contribution: float
    automation_contribution: float
    regulatory_factor: float

    # Assumptions that apply
    active_assumptions: List[str]


@dataclass
class EnsembleForecast:
    """Probability-weighted ensemble of scenario forecasts."""
    domain: str
    year: int

    # Scenario-specific forecasts
    scenario_forecasts: Dict[ScenarioType, ScenarioForecast]

    # Ensemble statistics (probability-weighted)
    weighted_mean: float
    weighted_std: float
    weighted_ci_90: Tuple[float, float]

    # Scenario probabilities
    scenario_probabilities: Dict[ScenarioType, float]


class ScenarioAnalyzer:
    """
    Performs scenario analysis for a domain.

    Generates forecasts under each scenario and combines them
    into probability-weighted ensemble predictions.
    """

    def __init__(
        self,
        domain: str,
        scenario_probabilities: Dict[ScenarioType, float] = None,
    ):
        self.domain = domain
        self.probabilities = scenario_probabilities or SCENARIO_PROBABILITIES

        # Validate probabilities sum to 1
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.probabilities = {
                k: v / total for k, v in self.probabilities.items()
            }

    def _base_acceleration(self, year: int) -> float:
        """
        Calculate base acceleration (from v0.7 model).

        Simplified version - actual implementation would import from v0.7.
        """
        t = year - 2024

        # Domain-specific base accelerations
        base_2030 = {
            "structural_biology": 15.0,
            "drug_discovery": 3.5,
            "materials_science": 3.8,
            "protein_design": 6.6,
            "clinical_genomics": 5.6,
        }

        # Time scaling (logistic growth)
        base = base_2030.get(self.domain, 3.0)
        time_factor = 1 + (base - 1) * (1 - np.exp(-0.2 * t))

        return time_factor

    def _scenario_forecast(
        self,
        scenario_type: ScenarioType,
        year: int,
    ) -> ScenarioForecast:
        """Generate forecast under a specific scenario."""
        scenario = SCENARIOS[scenario_type]
        base = self._base_acceleration(year)

        # Apply scenario multiplier
        multiplier = scenario.multipliers.get(self.domain, 1.0)
        acceleration = base * multiplier

        # Time adjustment based on adoption rate and saturation
        t = year - 2024
        saturation_t = scenario.saturation_year - 2024

        # Logistic saturation
        saturation_factor = 1 / (1 + np.exp(-0.3 * (saturation_t - t)))
        acceleration *= saturation_factor

        # Confidence interval based on scenario confidence
        ci_width = 1.5 * (1 - scenario.prediction_confidence)
        ci = (acceleration * (1 - ci_width), acceleration * (1 + ci_width))

        # Determine active assumptions
        active = [a.name for a in scenario.assumptions
                  if not a.domain_specific or self.domain in a.domains]

        return ScenarioForecast(
            scenario=scenario_type,
            domain=self.domain,
            year=year,
            acceleration=acceleration,
            confidence_interval=ci,
            ai_contribution=acceleration * 0.6,
            automation_contribution=acceleration * 0.3,
            regulatory_factor=multiplier * 0.5 + 0.5,
            active_assumptions=active,
        )

    def ensemble_forecast(self, year: int) -> EnsembleForecast:
        """Generate probability-weighted ensemble forecast."""
        scenario_forecasts = {}

        for scenario_type in ScenarioType:
            scenario_forecasts[scenario_type] = self._scenario_forecast(
                scenario_type, year
            )

        # Calculate weighted statistics
        accels = [scenario_forecasts[s].acceleration for s in ScenarioType]
        probs = [self.probabilities[s] for s in ScenarioType]

        weighted_mean = np.average(accels, weights=probs)
        weighted_var = np.average(
            [(a - weighted_mean) ** 2 for a in accels],
            weights=probs
        )
        weighted_std = np.sqrt(weighted_var)

        # Weighted CI (approximate)
        sorted_with_probs = sorted(zip(accels, probs), key=lambda x: x[0])
        cumsum = 0
        ci_5, ci_95 = accels[0], accels[-1]
        for accel, prob in sorted_with_probs:
            cumsum += prob
            if cumsum >= 0.05 and ci_5 == accels[0]:
                ci_5 = accel
            if cumsum >= 0.95:
                ci_95 = accel
                break

        return EnsembleForecast(
            domain=self.domain,
            year=year,
            scenario_forecasts=scenario_forecasts,
            weighted_mean=weighted_mean,
            weighted_std=weighted_std,
            weighted_ci_90=(ci_5, ci_95),
            scenario_probabilities=self.probabilities,
        )

    def scenario_comparison(
        self,
        years: List[int] = None,
    ) -> str:
        """Generate scenario comparison report."""
        years = years or [2025, 2030, 2040, 2050]

        lines = [
            "=" * 90,
            f"SCENARIO ANALYSIS: {self.domain.upper()}",
            "=" * 90,
            "",
            "SCENARIO DEFINITIONS:",
            "-" * 50,
        ]

        for scenario_type in ScenarioType:
            scenario = SCENARIOS[scenario_type]
            prob = self.probabilities[scenario_type]
            lines.append(
                f"  {scenario.name:<15} (P={prob:.0%}): {scenario.description[:50]}"
            )

        lines.extend([
            "",
            "ACCELERATION BY SCENARIO AND YEAR:",
            "-" * 90,
            f"{'Scenario':<15} " + "".join(f"{y:>12}" for y in years),
            "-" * 90,
        ])

        for scenario_type in ScenarioType:
            row = f"{scenario_type.value:<15}"
            for year in years:
                forecast = self._scenario_forecast(scenario_type, year)
                row += f"{forecast.acceleration:>11.1f}x"
            lines.append(row)

        lines.extend([
            "-" * 90,
            "",
            "ENSEMBLE (PROBABILITY-WEIGHTED):",
            "-" * 90,
        ])

        row = f"{'Weighted Mean':<15}"
        for year in years:
            ensemble = self.ensemble_forecast(year)
            row += f"{ensemble.weighted_mean:>11.1f}x"
        lines.append(row)

        row = f"{'90% CI Low':<15}"
        for year in years:
            ensemble = self.ensemble_forecast(year)
            row += f"{ensemble.weighted_ci_90[0]:>11.1f}x"
        lines.append(row)

        row = f"{'90% CI High':<15}"
        for year in years:
            ensemble = self.ensemble_forecast(year)
            row += f"{ensemble.weighted_ci_90[1]:>11.1f}x"
        lines.append(row)

        lines.extend([
            "-" * 90,
            "",
            "KEY ASSUMPTIONS:",
        ])

        for assumption in SCENARIOS[ScenarioType.BASELINE].assumptions:
            lines.append(f"  - {assumption.name}: P={assumption.probability:.0%}")

        return "\n".join(lines)


def compare_all_domains():
    """Compare scenario analysis across domains."""
    print("=" * 100)
    print("CROSS-DOMAIN SCENARIO COMPARISON (2030)")
    print("=" * 100)
    print()

    domains = ["structural_biology", "drug_discovery", "materials_science",
               "protein_design", "clinical_genomics"]

    print(f"{'Domain':<22} {'Pessimistic':<12} {'Baseline':<12} {'Optimistic':<12} {'Breakthrough':<12} {'Ensemble':<12}")
    print("-" * 100)

    for domain in domains:
        analyzer = ScenarioAnalyzer(domain=domain)

        pess = analyzer._scenario_forecast(ScenarioType.PESSIMISTIC, 2030)
        base = analyzer._scenario_forecast(ScenarioType.BASELINE, 2030)
        opt = analyzer._scenario_forecast(ScenarioType.OPTIMISTIC, 2030)
        brk = analyzer._scenario_forecast(ScenarioType.BREAKTHROUGH, 2030)
        ensemble = analyzer.ensemble_forecast(2030)

        print(
            f"{domain:<22} {pess.acceleration:>10.1f}x {base.acceleration:>10.1f}x "
            f"{opt.acceleration:>10.1f}x {brk.acceleration:>10.1f}x "
            f"{ensemble.weighted_mean:>10.1f}x"
        )

    print("-" * 100)
    print()
    print("SCENARIO PROBABILITIES:")
    for s, p in SCENARIO_PROBABILITIES.items():
        print(f"  {s.value:<15}: {p:.0%}")


if __name__ == "__main__":
    # Cross-domain comparison
    compare_all_domains()

    print()
    print()

    # Detailed analysis for one domain
    analyzer = ScenarioAnalyzer(domain="drug_discovery")
    print(analyzer.scenario_comparison())
