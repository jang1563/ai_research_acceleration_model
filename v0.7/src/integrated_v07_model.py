#!/usr/bin/env python3
"""
Integrated v0.7 Model
=====================

Combines v0.6.1 corrections with v0.7 enhancements:

1. Dynamic Simulation Bypass (M2-P3 fix)
   - Bypass potential grows with AI capability over time
   - Domain-specific simulation maturity curves

2. Feedback Loops (M2-P2 fix)
   - Priority shifts when backlog grows
   - Resource reallocation to automation
   - Trust dynamics for predictions

3. Sub-Domain Profiles (D1-P1, D1-P2 fixes)
   - Drug discovery: 7 sub-stages with different acceleration
   - Protein design: 4 sub-types with different bottlenecks

Key Formula:
  v0.7_Acceleration = v0.6.1_Calibrated * Dynamic_Bypass_Boost * Feedback_Adjustment * SubDomain_Factor

Where:
  - v0.6.1_Calibrated: Base acceleration with bias correction
  - Dynamic_Bypass_Boost: Throughput multiplier from simulation
  - Feedback_Adjustment: Self-correcting dynamics factor
  - SubDomain_Factor: Adjustment for specific sub-stages/sub-types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Set up paths
_v07_src = Path(__file__).parent
_v06_src = _v07_src.parent.parent / "v0.6" / "src"
_v05_src = _v07_src.parent.parent / "v0.5" / "src"
_v04_src = _v07_src.parent.parent / "v0.4" / "src"
_v03_src = _v07_src.parent.parent / "v0.3" / "src"

# Add all paths to sys.path for dependency resolution
sys.path.insert(0, str(_v03_src))
sys.path.insert(0, str(_v04_src))
sys.path.insert(0, str(_v05_src))
sys.path.insert(0, str(_v06_src))
sys.path.insert(0, str(_v07_src))

# Import using importlib to avoid conflicts
import importlib.util

def _import_module(name: str, path: Path):
    """Import a module from a specific path."""
    spec = importlib.util.spec_from_file_location(f"v07_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import v0.4 components using standard import (path already set)
from refined_model import Scenario

# Import v0.5 autonomous_lab first (dependency)
from autonomous_lab import AutomationScenario

# Import v0.6 components
_integrated_v06 = _import_module("integrated_v06_model", _v06_src / "integrated_v06_model.py")
IntegratedV06Model = _integrated_v06.IntegratedV06Model
V06Forecast = _integrated_v06.V06Forecast
UncertaintyBounds = _integrated_v06.UncertaintyBounds
ScenarioRange = _integrated_v06.ScenarioRange

# Import v0.7 components
_dynamic_bypass = _import_module("dynamic_bypass", _v07_src / "dynamic_bypass.py")
DynamicBypassModel = _dynamic_bypass.DynamicBypassModel
SimulationCapability = _dynamic_bypass.SimulationCapability

_feedback = _import_module("feedback_loops", _v07_src / "feedback_loops.py")
FeedbackLoopModel = _feedback.FeedbackLoopModel
FeedbackState = _feedback.FeedbackState

_subdomain = _import_module("subdomain_profiles", _v07_src / "subdomain_profiles.py")
SubDomainModel = _subdomain.SubDomainModel


@dataclass
class V07Forecast:
    """Extended forecast with v0.7 enhancements."""
    year: int
    domain: str

    # v0.6.1 base values (inherited)
    v06_calibrated: float
    v06_uncertainty: UncertaintyBounds
    v06_scenarios: ScenarioRange

    # v0.7 Dynamic Bypass
    bypass_potential: float
    bypass_accuracy: float
    bypass_throughput_multiplier: float
    simulation_maturity: str

    # v0.7 Feedback Loops
    researcher_shift: float
    capacity_boost: float
    trust_level: float
    feedback_adjustment: float

    # v0.7 Sub-Domain (if applicable)
    subdomain_factor: float
    bottleneck_stage: str
    bottleneck_type: str

    # Final v0.7 result
    v07_acceleration: float
    v07_uncertainty: UncertaintyBounds
    v07_scenarios: ScenarioRange

    # Comparison
    v06_to_v07_change: float  # Percentage change from v0.6.1


class IntegratedV07Model:
    """
    v0.7 Model integrating all enhancements:
    - v0.6.1: Calibrated predictions with uncertainty
    - Dynamic bypass: Time-varying simulation capability
    - Feedback loops: Self-correcting dynamics
    - Sub-domain profiles: Granular acceleration modeling
    """

    def __init__(
        self,
        domain: str,
        ai_scenario: Scenario = Scenario.BASELINE,
        automation_scenario: AutomationScenario = AutomationScenario.BASELINE,
        subtype: Optional[str] = None,  # For protein design sub-types
    ):
        self.domain = domain
        self.ai_scenario = ai_scenario
        self.automation_scenario = automation_scenario
        self.subtype = subtype

        # Initialize component models
        self.v06_model = IntegratedV06Model(
            domain=domain,
            ai_scenario=ai_scenario,
            automation_scenario=automation_scenario,
        )

        self.bypass_model = DynamicBypassModel(domain=domain)
        self.feedback_model = FeedbackLoopModel(domain=domain)
        self.subdomain_model = SubDomainModel(domain=domain)

    def _calculate_subdomain_factor(
        self,
        year: int,
    ) -> Tuple[float, str, str]:
        """
        Calculate sub-domain adjustment factor.

        Returns (factor, bottleneck_stage, bottleneck_type)
        """
        # For protein design with specific subtype
        if self.domain == "protein_design" and self.subtype:
            accel = self.subdomain_model.subtype_acceleration(self.subtype, year)
            subtype_info = self.subdomain_model.subtypes.get(self.subtype)
            if subtype_info:
                return accel / 3.0, subtype_info.primary_bottleneck, "subtype_specific"
            return 1.0, "unknown", "none"

        # For domains with sub-stage analysis
        if self.subdomain_model.stages:
            ai_scenario_str = self.ai_scenario.value if hasattr(self.ai_scenario, 'value') else "baseline"
            auto_scenario_str = self.automation_scenario.value if hasattr(self.automation_scenario, 'value') else "baseline"

            overall, _ = self.subdomain_model.end_to_end_acceleration(
                year,
                ai_scenario=ai_scenario_str,
                automation_scenario=auto_scenario_str,
            )

            bottlenecks = self.subdomain_model.identify_bottlenecks(
                year,
                ai_scenario=ai_scenario_str,
                automation_scenario=auto_scenario_str,
            )

            if bottlenecks:
                top_bottleneck = bottlenecks[0]
                return overall / 3.0, top_bottleneck[0], top_bottleneck[2]

            return overall / 3.0, "balanced", "none"

        return 1.0, "none", "none"

    def _apply_v07_uncertainty(
        self,
        v06_uncertainty: UncertaintyBounds,
        v07_acceleration: float,
        bypass_capability: SimulationCapability,
        feedback_state: FeedbackState,
    ) -> UncertaintyBounds:
        """
        Adjust uncertainty bounds for v0.7 factors.

        v0.7 adds uncertainty from:
        - Simulation accuracy variance
        - Feedback loop unpredictability
        - Sub-domain heterogeneity
        """
        # Base ratio from v0.6.1
        ratio = v07_acceleration / v06_uncertainty.median if v06_uncertainty.median > 0 else 1

        # Additional uncertainty from bypass (simulation accuracy)
        bypass_uncertainty = 1 + (1 - bypass_capability.confidence_calibration) * 0.3

        # Additional uncertainty from feedback (unpredictable dynamics)
        feedback_uncertainty = 1 + (1 - feedback_state.current_trust) * 0.2

        # Combined uncertainty multiplier
        total_uncertainty = bypass_uncertainty * feedback_uncertainty

        return UncertaintyBounds(
            lower_5=v07_acceleration / (total_uncertainty * 1.8),
            lower_25=v07_acceleration / (total_uncertainty * 1.3),
            median=v07_acceleration,
            upper_75=v07_acceleration * (total_uncertainty * 1.3),
            upper_95=v07_acceleration * (total_uncertainty * 1.8),
        )

    def _apply_v07_scenarios(
        self,
        v06_scenarios: ScenarioRange,
        v07_acceleration: float,
        bypass_capability: SimulationCapability,
    ) -> ScenarioRange:
        """
        Adjust scenario ranges for v0.7.
        """
        # Pessimistic: bypass fails, feedback negative
        pessimistic = v07_acceleration * 0.6

        # Optimistic: bypass succeeds better than expected
        optimistic = v07_acceleration * (1 + bypass_capability.bypass_potential * 0.5)

        return ScenarioRange(
            pessimistic=pessimistic,
            baseline=v07_acceleration,
            optimistic=optimistic,
        )

    def forecast(self, years: List[int]) -> Dict[int, V07Forecast]:
        """Generate v0.7 forecasts."""
        # Get v0.6.1 forecasts
        v06_forecasts = self.v06_model.forecast(years)

        # Get bypass trajectories
        bypass_results = self.bypass_model.simulate(years)

        # Build backlog/validation trajectories for feedback model
        backlog_traj = {}
        validation_traj = {}
        for year, v06f in v06_forecasts.items():
            backlog_traj[year] = min(v06f.backlog_years, 10000)
            validation_traj[year] = v06f.validation_capacity

        # Get feedback results
        feedback_results = self.feedback_model.simulate(
            years, backlog_traj, validation_traj
        )

        results = {}

        for year in years:
            v06f = v06_forecasts[year]
            bypass = bypass_results[year]
            feedback = feedback_results[year]

            # Get sub-domain factor
            subdomain_factor, bottleneck_stage, bottleneck_type = self._calculate_subdomain_factor(year)

            # Calculate v0.7 acceleration
            # IMPORTANT: v0.7 enhancements are INCREMENTAL, not multiplicative
            # v0.6.1 already captures most acceleration; v0.7 adds refinements
            #
            # Formula: v06_calibrated * (1 + incremental_boost)
            # Where incremental_boost = weighted combination of new factors
            #
            # Bypass: Adds effective capacity (but v0.6.1 partially accounts for this)
            # The incremental bypass benefit is (throughput_mult - 1) * 0.3
            bypass_increment = (bypass.effective_throughput_multiplier - 1) * 0.3

            # Feedback: Mostly neutral (self-correcting) with small net effect
            feedback_increment = (feedback.feedback_adjustment - 1)

            # Sub-domain: Provides more granular estimate
            # If subdomain_factor > 1, we were underestimating; < 1, overestimating
            # Apply as adjustment with dampening
            subdomain_increment = (subdomain_factor - 1) * 0.5

            # Total incremental boost (capped at reasonable levels)
            total_increment = bypass_increment + feedback_increment + subdomain_increment
            total_increment = max(-0.5, min(total_increment, 1.5))  # -50% to +150%

            v07_accel = v06f.calibrated_acceleration * (1 + total_increment)

            # Apply bounds (can't be < 1.0 or unreasonably high)
            v07_accel = max(1.0, min(v07_accel, 100.0))

            # Calculate change from v0.6.1
            change = ((v07_accel / v06f.calibrated_acceleration) - 1) * 100 if v06f.calibrated_acceleration > 0 else 0

            # Adjust uncertainty and scenarios
            v07_uncertainty = self._apply_v07_uncertainty(
                v06f.uncertainty, v07_accel, bypass, feedback
            )
            v07_scenarios = self._apply_v07_scenarios(
                v06f.scenarios, v07_accel, bypass
            )

            results[year] = V07Forecast(
                year=year,
                domain=self.domain,

                # v0.6.1 values
                v06_calibrated=v06f.calibrated_acceleration,
                v06_uncertainty=v06f.uncertainty,
                v06_scenarios=v06f.scenarios,

                # Dynamic bypass
                bypass_potential=bypass.bypass_potential,
                bypass_accuracy=bypass.accuracy,
                bypass_throughput_multiplier=bypass.effective_throughput_multiplier,
                simulation_maturity=bypass.maturity.value,

                # Feedback loops
                researcher_shift=feedback.researchers_shifted,
                capacity_boost=feedback.capacity_boost,
                trust_level=feedback.current_trust,
                feedback_adjustment=feedback.feedback_adjustment,

                # Sub-domain
                subdomain_factor=subdomain_factor,
                bottleneck_stage=bottleneck_stage,
                bottleneck_type=bottleneck_type,

                # Final v0.7
                v07_acceleration=v07_accel,
                v07_uncertainty=v07_uncertainty,
                v07_scenarios=v07_scenarios,

                # Comparison
                v06_to_v07_change=change,
            )

        return results

    def summary(self) -> str:
        """Generate comprehensive v0.7 model summary."""
        forecasts = self.forecast([2025, 2030, 2040, 2050])

        lines = [
            "=" * 100,
            f"INTEGRATED v0.7 MODEL: {self.domain.upper()}",
            "=" * 100,
            "",
            f"AI Scenario: {self.ai_scenario.value if hasattr(self.ai_scenario, 'value') else self.ai_scenario}",
            f"Automation Scenario: {self.automation_scenario.value if hasattr(self.automation_scenario, 'value') else self.automation_scenario}",
            f"Sub-type: {self.subtype or 'None'}",
            "",
            "ACCELERATION FORECAST:",
            "-" * 100,
            f"{'Year':<8} {'v0.6.1':<10} {'Bypass':<10} {'Feedback':<10} {'SubDom':<10} {'v0.7':<10} {'90% CI':<20} {'Change':<10}",
            "-" * 100,
        ]

        for year in [2025, 2030, 2040, 2050]:
            f = forecasts[year]
            ci_str = f"[{f.v07_uncertainty.lower_5:.1f}-{f.v07_uncertainty.upper_95:.1f}]"
            change_str = f"{f.v06_to_v07_change:+.0f}%"

            lines.append(
                f"{year:<8} {f.v06_calibrated:>8.1f}x {f.bypass_throughput_multiplier:>8.1f}x "
                f"{f.feedback_adjustment:>8.2f}x {f.subdomain_factor:>8.2f}x "
                f"{f.v07_acceleration:>8.1f}x {ci_str:<20} {change_str:<10}"
            )

        lines.extend([
            "-" * 100,
            "",
            "COMPONENT ANALYSIS (2030):",
        ])

        f_2030 = forecasts[2030]

        # Dynamic bypass
        lines.extend([
            "",
            "  DYNAMIC BYPASS:",
            f"    Bypass Potential: {f_2030.bypass_potential:.0%}",
            f"    Simulation Accuracy: {f_2030.bypass_accuracy:.0%}",
            f"    Throughput Multiplier: {f_2030.bypass_throughput_multiplier:.1f}x",
            f"    Maturity: {f_2030.simulation_maturity}",
        ])

        # Feedback loops
        lines.extend([
            "",
            "  FEEDBACK LOOPS:",
            f"    Researcher Shift: {f_2030.researcher_shift:.0%}",
            f"    Capacity Boost: {f_2030.capacity_boost:.1f}x",
            f"    Trust Level: {f_2030.trust_level:.0%}",
            f"    Net Adjustment: {f_2030.feedback_adjustment:.2f}x",
        ])

        # Sub-domain
        lines.extend([
            "",
            "  SUB-DOMAIN:",
            f"    Factor: {f_2030.subdomain_factor:.2f}x",
            f"    Bottleneck Stage: {f_2030.bottleneck_stage}",
            f"    Bottleneck Type: {f_2030.bottleneck_type}",
        ])

        # Scenarios
        lines.extend([
            "",
            "SCENARIO ANALYSIS (2030):",
            f"  Pessimistic: {f_2030.v07_scenarios.pessimistic:.1f}x",
            f"  Baseline:    {f_2030.v07_scenarios.baseline:.1f}x",
            f"  Optimistic:  {f_2030.v07_scenarios.optimistic:.1f}x",
        ])

        return "\n".join(lines)


def compare_v06_v07():
    """Compare v0.6.1 and v0.7 predictions across domains."""
    print("=" * 100)
    print("MODEL COMPARISON: v0.6.1 vs v0.7")
    print("=" * 100)
    print()
    print("v0.6.1: Calibrated AI + Automation + Triage")
    print("v0.7:   + Dynamic Bypass + Feedback Loops + Sub-Domain Profiles")
    print()

    domains = ["materials_science", "drug_discovery", "protein_design",
               "clinical_genomics", "structural_biology"]

    print(f"{'Domain':<22} {'v0.6.1 2030':<12} {'v0.7 2030':<12} {'Change':<10} {'Key Factor':<20}")
    print("-" * 100)

    for domain in domains:
        try:
            model = IntegratedV07Model(domain=domain)
            forecasts = model.forecast([2030])
            f = forecasts[2030]

            change = f"{f.v06_to_v07_change:+.0f}%"

            # Determine key factor
            factors = [
                ("bypass", f.bypass_throughput_multiplier),
                ("feedback", f.feedback_adjustment),
                ("subdomain", f.subdomain_factor),
            ]
            key_factor = max(factors, key=lambda x: abs(x[1] - 1.0))[0]

            print(
                f"{domain:<22} {f.v06_calibrated:>10.1f}x {f.v07_acceleration:>10.1f}x "
                f"{change:<10} {key_factor:<20}"
            )
        except Exception as e:
            print(f"{domain:<22} Error: {e}")

    print("-" * 100)
    print()
    print("KEY INSIGHT: v0.7 accounts for dynamic improvements in simulation bypass")
    print("and self-correcting feedback loops, generally increasing acceleration estimates.")


def run_subdomain_analysis():
    """Run detailed sub-domain analysis for drug discovery."""
    print("=" * 100)
    print("DRUG DISCOVERY SUB-DOMAIN ANALYSIS")
    print("=" * 100)
    print()

    model = IntegratedV07Model(domain="drug_discovery")

    # Get sub-domain model directly
    subdomain = model.subdomain_model
    print(subdomain.summary())


if __name__ == "__main__":
    # Compare versions
    compare_v06_v07()

    print()
    print()

    # Sub-domain analysis
    run_subdomain_analysis()

    print()
    print()

    # Detailed summary for one domain
    model = IntegratedV07Model(domain="materials_science")
    print(model.summary())
