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

            # For historical/current years (<=2025), triage constraints are minimal
            # The backlog problem is forward-looking
            if year <= 2025:
                # Historical: v0.5 predictions are primary, triage has limited impact
                # Backlog is just starting to accumulate
                years_since_base = max(year - 2024, 0)
                triage_dampening = 1.0 - (0.1 * years_since_base)  # Minimal in 2024-2025
                effective_accel = v05.end_to_end_acceleration
            else:
                # Future projections: triage becomes binding constraint
                # Triage factor shows how much backlog reduces effective acceleration
                triage_factor = min(1.0, triage_accel / max(v05.end_to_end_acceleration, 1))

                # Apply triage constraint, but cap at 50% reduction
                # (triage can improve over time with better AI selection)
                effective_accel = v05.end_to_end_acceleration * max(triage_factor, 0.5)

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
            )

        return results

    def summary(self) -> str:
        """Generate comprehensive model summary."""
        forecasts = self.forecast([2025, 2030, 2040, 2050])

        lines = [
            "=" * 80,
            f"INTEGRATED v0.6 MODEL: {self.domain.upper()}",
            "=" * 80,
            "",
            f"AI Scenario: {self.ai_scenario.value}",
            f"Automation Scenario: {self.automation_scenario.value}",
            "",
            "ACCELERATION FORECAST:",
            "-" * 80,
            f"{'Year':<8} {'v0.5':<10} {'Triage':<10} {'Effective':<12} {'Limiting':<12} {'Backlog Risk':<14}",
            "-" * 80,
        ]

        for year in [2025, 2030, 2040, 2050]:
            f = forecasts[year]
            lines.append(
                f"{year:<8} {f.v05_end_to_end:>8.1f}x {f.triage_limited_acceleration:>8.1f}x "
                f"{f.effective_acceleration:>10.1f}x {f.limiting_factor:<12} {f.backlog_risk.value:<14}"
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

        return "\n".join(lines)


def compare_v05_v06():
    """Compare v0.5 and v0.6 predictions across domains."""
    print("=" * 80)
    print("MODEL COMPARISON: v0.5 vs v0.6 (with Triage Constraints)")
    print("=" * 80)
    print()
    print("v0.5: AI + Automation (assumes validation scales smoothly)")
    print("v0.6: AI + Automation + Triage (models selection bottleneck)")
    print()

    domains = list(DOMAIN_TRIAGE_PROFILES.keys())
    years = [2030, 2050]

    print(f"{'Domain':<22} {'v0.5 2030':<12} {'v0.6 2030':<12} {'v0.5 2050':<12} {'v0.6 2050':<12}")
    print("-" * 80)

    for domain in domains:
        try:
            model = IntegratedV06Model(domain=domain)
            forecasts = model.forecast(years)

            v05_30 = forecasts[2030].v05_end_to_end
            v06_30 = forecasts[2030].effective_acceleration
            v05_50 = forecasts[2050].v05_end_to_end
            v06_50 = forecasts[2050].effective_acceleration

            print(
                f"{domain:<22} {v05_30:>10.1f}x {v06_30:>10.1f}x "
                f"{v05_50:>10.1f}x {v06_50:>10.1f}x"
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
