#!/usr/bin/env python3
"""
Integrated v0.6 Model
=====================

Combines v0.5 (AI + Automation) with v0.6 Triage & Selection dynamics.

Key Innovation:
v0.5 assumed validation capacity grows smoothly with automation.
v0.6 models the SELECTION PROBLEM: when AI generates hypotheses faster
than they can be validated, backlog accumulates and triage efficiency
becomes the limiting factor.

New Equation:
  Effective_Acceleration = min(
      v0.5_Acceleration,               # Physical + cognitive gains
      Triage_Limited_Acceleration      # Constrained by selection capacity
  )

This addresses the GNoME problem where 365x stage acceleration
yielded only 1x end-to-end because of the validation bottleneck.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Set up paths carefully to avoid import conflicts
_v06_src = Path(__file__).parent
_v05_src = _v06_src.parent.parent / "v0.5" / "src"
_v04_src = _v06_src.parent.parent / "v0.4" / "src"

# Import v0.4 first (needed by v0.5)
sys.path.insert(0, str(_v04_src))
from refined_model import Scenario as AIScenario, DOMAIN_PROFILES

# Import v0.5 modules
sys.path.insert(0, str(_v05_src))
from integrated_model import IntegratedAccelerationModel, IntegratedForecast
from autonomous_lab import AutomationScenario

# Import v0.6 modules using importlib to avoid path conflicts
import importlib.util

def _import_v06_module(name):
    """Import a module from v0.6 src directory."""
    spec = importlib.util.spec_from_file_location(
        f"v06_{name}",
        _v06_src / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import triage_model
_triage_mod = _import_v06_module("triage_model")
TriageModel = _triage_mod.TriageModel
TriageParameters = _triage_mod.TriageParameters
TriageStrategy = _triage_mod.TriageStrategy
DOMAIN_TRIAGE_PROFILES = _triage_mod.DOMAIN_TRIAGE_PROFILES
TriageState = _triage_mod.TriageState

# Import backlog_dynamics
_backlog_mod = _import_v06_module("backlog_dynamics")
BacklogDynamicsModel = _backlog_mod.BacklogDynamicsModel
BacklogState = _backlog_mod.BacklogState
BacklogRiskLevel = _backlog_mod.BacklogRiskLevel
BACKLOG_SCENARIOS = _backlog_mod.BACKLOG_SCENARIOS

# Import model corrections (v0.6.1 fixes)
_corrections_mod = _import_v06_module("model_corrections")
apply_calibration = _corrections_mod.apply_calibration
calculate_uncertainty = _corrections_mod.calculate_uncertainty
get_scenario_range = _corrections_mod.get_scenario_range
get_empirical_triage_floor = _corrections_mod.get_empirical_triage_floor
get_historical_backlog_factor = _corrections_mod.get_historical_backlog_factor
apply_stage_dependencies = _corrections_mod.apply_stage_dependencies
CALIBRATION_FACTORS = _corrections_mod.CALIBRATION_FACTORS


@dataclass
class UncertaintyBounds:
    """Confidence intervals for predictions (P2-P1-P1 fix)."""
    lower_5: float    # 5th percentile
    lower_25: float   # 25th percentile
    median: float     # 50th percentile
    upper_75: float   # 75th percentile
    upper_95: float   # 95th percentile


@dataclass
class ScenarioRange:
    """Range of scenarios (P2-P1-P2 fix)."""
    pessimistic: float
    baseline: float
    optimistic: float


@dataclass
class V06Forecast:
    """Extended forecast including triage-adjusted acceleration."""
    year: int

    # v0.5 components
    cognitive_acceleration: float
    physical_acceleration: float
    v05_end_to_end: float

    # v0.6 triage components
    generation_rate: float
    validation_capacity: float
    backlog: float
    backlog_years: float
    triage_efficiency: float
    triage_limited_acceleration: float

    # Final v0.6 result
    effective_acceleration: float
    limiting_factor: str  # "physical", "triage", or "balanced"

    # Risk metrics
    backlog_risk: BacklogRiskLevel
    opportunity_cost: float

    # NEW: Corrected predictions (v0.6.1 fixes)
    calibrated_acceleration: float = None        # P1-E1-P1: Bias corrected
    uncertainty: UncertaintyBounds = None        # P2-P1-P1: Confidence intervals
    scenarios: ScenarioRange = None              # P2-P1-P2: Scenario range


# Domain mapping between triage model and v0.4/v0.5 models
DOMAIN_MAPPING = {
    "clinical_genomics": "genomics",  # Triage -> v0.4/v0.5
}

REVERSE_DOMAIN_MAPPING = {v: k for k, v in DOMAIN_MAPPING.items()}


class IntegratedV06Model:
    """
    v0.6 Model integrating AI acceleration, lab automation, and triage dynamics.

    Three constraints on acceleration:
    1. Cognitive: AI capability for prediction/analysis (10-1,000,000x)
    2. Physical: Lab automation for validation (1-50x)
    3. Triage: Selection efficiency when backlog accumulates (0.1-10x)

    Effective acceleration = min(cognitive, physical, triage-limited)
    """

    def __init__(
        self,
        domain: str,
        ai_scenario: AIScenario = AIScenario.BASELINE,
        automation_scenario: AutomationScenario = AutomationScenario.BASELINE,
        triage_params: Optional[TriageParameters] = None,
    ):
        self.domain = domain
        self.ai_scenario = ai_scenario
        self.automation_scenario = automation_scenario

        # Map domain names between models
        v05_domain = DOMAIN_MAPPING.get(domain, domain)
        triage_domain = REVERSE_DOMAIN_MAPPING.get(domain, domain)

        # Initialize component models
        self.v05_model = IntegratedAccelerationModel(
            domain=v05_domain,
            ai_scenario=ai_scenario,
            automation_scenario=automation_scenario,
        )

        self.triage_model = TriageModel(
            domain=domain,
            triage_params=triage_params,
            automation_scenario=automation_scenario,
        )

        self.backlog_model = BacklogDynamicsModel(
            domain=domain,
            automation_scenario=automation_scenario,
        )

        self.triage_profile = DOMAIN_TRIAGE_PROFILES.get(
            domain, DOMAIN_TRIAGE_PROFILES["materials_science"]
        )

    def _triage_limited_acceleration(
        self,
        triage_state: TriageState,
        baseline_throughput: float
    ) -> float:
        """
        Calculate acceleration limited by triage capacity.

        When backlog accumulates, effective throughput depends on:
        1. Validation capacity (experiments/year)
        2. Triage efficiency (success rate after selection)

        Acceleration = (current_successes / baseline_successes)
        """
        if baseline_throughput <= 0:
            return 1.0

        return triage_state.successes_this_year / baseline_throughput

    def forecast(self, years: List[int]) -> Dict[int, V06Forecast]:
        """Generate v0.6 forecasts including triage constraints."""
        # Get v0.5 forecasts
        v05_forecasts = self.v05_model.forecast(years)

        # Get triage simulations
        all_years = [2024] + years
        triage_results = self.triage_model.simulate(all_years)
        backlog_results = self.backlog_model.simulate(all_years)

        # Baseline throughput (2024)
        baseline_throughput = triage_results[2024].successes_this_year

        results = {}

        for year in years:
            v05 = v05_forecasts[year]
            triage = triage_results[year]
            backlog = backlog_results[year]

            # Calculate triage-limited acceleration
            triage_accel = self._triage_limited_acceleration(triage, baseline_throughput)

            # P1-M1-P1 FIX: Use empirically-derived triage floor instead of ad-hoc 0.5
            triage_floor = get_empirical_triage_floor(self.domain)

            # For historical/current years (<=2025), triage constraints are minimal
            # The backlog problem is forward-looking
            if year <= 2025:
                # Historical: v0.5 predictions are primary, triage has limited impact
                # P1-E1-P2 FIX: Apply historical backlog factor for known events (e.g., GNoME)
                historical_factor = get_historical_backlog_factor(self.domain, year)
                effective_accel = v05.end_to_end_acceleration * historical_factor
            else:
                # Future projections: triage becomes binding constraint
                # Triage factor shows how much backlog reduces effective acceleration
                triage_factor = min(1.0, triage_accel / max(v05.end_to_end_acceleration, 1))

                # P1-M1-P1 FIX: Use empirical floor instead of ad-hoc 0.5
                effective_accel = v05.end_to_end_acceleration * max(triage_factor, triage_floor)

            # P1-E1-P1 FIX: Apply calibration to reduce over-prediction bias
            calibrated_accel = apply_calibration(self.domain, effective_accel)

            # P2-P1-P1 FIX: Calculate uncertainty bounds
            uncertainty_bounds = calculate_uncertainty(calibrated_accel, self.domain, year)
            uncertainty = UncertaintyBounds(
                lower_5=uncertainty_bounds.lower_5,
                lower_25=uncertainty_bounds.lower_25,
                median=uncertainty_bounds.median,
                upper_75=uncertainty_bounds.upper_75,
                upper_95=uncertainty_bounds.upper_95,
            )

            # P2-P1-P2 FIX: Calculate scenario range
            scenario_range = get_scenario_range(calibrated_accel, self.domain)
            scenarios = ScenarioRange(
                pessimistic=scenario_range.pessimistic,
                baseline=scenario_range.baseline,
                optimistic=scenario_range.optimistic,
            )

            # Determine limiting factor
            if backlog.risk_level in [BacklogRiskLevel.HIGH, BacklogRiskLevel.CRITICAL]:
                if year > 2025 and triage_accel < v05.end_to_end_acceleration * 0.5:
                    limiting_factor = "triage"
                else:
                    limiting_factor = "physical"
            elif v05.end_to_end_acceleration < 3:
                limiting_factor = "physical"
            else:
                limiting_factor = "balanced"

            results[year] = V06Forecast(
                year=year,
                cognitive_acceleration=v05.cognitive_acceleration,
                physical_acceleration=v05.physical_acceleration,
                v05_end_to_end=v05.end_to_end_acceleration,
                generation_rate=triage.generation_rate,
                validation_capacity=triage.validation_capacity,
                backlog=triage.backlog,
                backlog_years=triage.backlog_years if triage.backlog_years < 10000 else float('inf'),
                triage_efficiency=triage.triage_efficiency,
                triage_limited_acceleration=triage_accel,
                effective_acceleration=effective_accel,
                limiting_factor=limiting_factor,
                backlog_risk=backlog.risk_level,
                opportunity_cost=backlog.opportunity_cost,
                # NEW v0.6.1 fields
                calibrated_acceleration=calibrated_accel,
                uncertainty=uncertainty,
                scenarios=scenarios,
            )

        return results

    def summary(self) -> str:
        """Generate comprehensive model summary."""
        forecasts = self.forecast([2025, 2030, 2040, 2050])

        lines = [
            "=" * 80,
            f"INTEGRATED v0.6.1 MODEL: {self.domain.upper()} (WITH CORRECTIONS)",
            "=" * 80,
            "",
            f"AI Scenario: {self.ai_scenario.value}",
            f"Automation Scenario: {self.automation_scenario.value}",
            "",
            "ACCELERATION FORECAST (v0.6.1 - Bias Corrected):",
            "-" * 80,
            f"{'Year':<8} {'v0.5':<10} {'Effective':<12} {'Calibrated':<12} {'90% CI':<20} {'Risk':<10}",
            "-" * 80,
        ]

        for year in [2025, 2030, 2040, 2050]:
            f = forecasts[year]
            ci_str = f"[{f.uncertainty.lower_5:.1f}-{f.uncertainty.upper_95:.1f}]"
            lines.append(
                f"{year:<8} {f.v05_end_to_end:>8.1f}x {f.effective_acceleration:>10.1f}x "
                f"{f.calibrated_acceleration:>10.1f}x {ci_str:<20} {f.backlog_risk.value:<10}"
            )

        lines.extend([
            "-" * 80,
            "",
            "BACKLOG STATUS (2030):",
            f"  Generation Rate: {forecasts[2030].generation_rate:,.0f}/year",
            f"  Validation Capacity: {forecasts[2030].validation_capacity:,.0f}/year",
            f"  Current Backlog: {forecasts[2030].backlog:,.0f}",
            f"  Years to Clear: {forecasts[2030].backlog_years:.1f}" if forecasts[2030].backlog_years < 10000 else "  Years to Clear: ∞",
            f"  Triage Efficiency: {forecasts[2030].triage_efficiency:.1%}",
            "",
            "KEY INSIGHT:",
        ])

        f_2030 = forecasts[2030]
        if f_2030.limiting_factor == "triage":
            lines.append(
                f"  ⚠ TRIAGE-LIMITED: Despite {f_2030.v05_end_to_end:.1f}x potential (v0.5), "
                f"effective acceleration only {f_2030.effective_acceleration:.1f}x due to backlog"
            )
        elif f_2030.limiting_factor == "physical":
            lines.append(
                f"  ⚠ PHYSICAL-LIMITED: Lab automation ({f_2030.physical_acceleration:.1f}x) "
                f"constrains end-to-end acceleration"
            )
        else:
            lines.append(
                f"  ✓ BALANCED: Cognitive and physical acceleration well-matched"
            )

        # Add scenario analysis (P2-P1-P2 fix)
        f_2030 = forecasts[2030]
        lines.extend([
            "",
            "SCENARIO ANALYSIS (2030):",
            f"  Pessimistic: {f_2030.scenarios.pessimistic:.1f}x",
            f"  Baseline:    {f_2030.scenarios.baseline:.1f}x",
            f"  Optimistic:  {f_2030.scenarios.optimistic:.1f}x",
        ])

        return "\n".join(lines)


def compare_v05_v06():
    """Compare v0.5 and v0.6.1 predictions across domains."""
    print("=" * 80)
    print("MODEL COMPARISON: v0.5 vs v0.6.1 (with Corrections)")
    print("=" * 80)
    print()
    print("v0.5:   AI + Automation (assumes validation scales smoothly)")
    print("v0.6:   AI + Automation + Triage (models selection bottleneck)")
    print("v0.6.1: + Calibration + Uncertainty + Scenarios")
    print()

    domains = list(DOMAIN_TRIAGE_PROFILES.keys())
    years = [2030, 2050]

    print(f"{'Domain':<22} {'v0.5':<10} {'v0.6':<10} {'Calibrated':<12} {'90% CI':<18}")
    print("-" * 80)

    for domain in domains:
        try:
            model = IntegratedV06Model(domain=domain)
            forecasts = model.forecast(years)

            v05_30 = forecasts[2030].v05_end_to_end
            v06_30 = forecasts[2030].effective_acceleration
            calib_30 = forecasts[2030].calibrated_acceleration
            ci_30 = f"[{forecasts[2030].uncertainty.lower_5:.1f}-{forecasts[2030].uncertainty.upper_95:.1f}]"

            print(
                f"{domain:<22} {v05_30:>8.1f}x {v06_30:>8.1f}x "
                f"{calib_30:>10.1f}x {ci_30:<18}"
            )
        except Exception as e:
            print(f"{domain:<22} Error: {e}")

    print("-" * 80)
    print()
    print("KEY INSIGHT: v0.6 shows lower effective acceleration in domains where")
    print("hypothesis generation far exceeds validation capacity (materials, protein).")


def run_scenario_matrix():
    """Run full scenario matrix for a domain."""
    print("=" * 80)
    print("v0.6 SCENARIO MATRIX: MATERIALS SCIENCE")
    print("=" * 80)
    print()

    domain = "materials_science"
    year = 2030

    header = "AI \\ Auto"
    print(f"{header:<15}", end="")
    for auto in AutomationScenario:
        print(f"{auto.value:>14}", end="")
    print()
    print("-" * 75)

    for ai in AIScenario:
        print(f"{ai.value:<15}", end="")
        for auto in AutomationScenario:
            try:
                model = IntegratedV06Model(
                    domain=domain,
                    ai_scenario=ai,
                    automation_scenario=auto,
                )
                forecast = model.forecast([year])[year]
                print(f"{forecast.effective_acceleration:>13.1f}x", end="")
            except Exception as e:
                print(f"{'ERR':>14}", end="")
        print()

    print("-" * 75)
    print()
    print("NOTE: Even with 'breakthrough' automation, triage constraints limit")
    print("effective acceleration due to hypothesis-validation imbalance.")


if __name__ == "__main__":
    # Compare v0.5 and v0.6
    compare_v05_v06()

    print()
    print()

    # Scenario matrix
    run_scenario_matrix()

    print()
    print()

    # Detailed summary for one domain
    model = IntegratedV06Model(domain="materials_science")
    print(model.summary())
