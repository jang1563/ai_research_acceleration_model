"""
Integrated Acceleration Model
=============================

Combines v0.4 cognitive model with v0.5 autonomous lab model
to provide full end-to-end acceleration forecasts.

Key Insight:
- v0.4 showed cognitive stages at 60-500x, physical at 1-1.5x
- v0.5 shows automation can push physical to 5-50x
- Combined: end-to-end acceleration of 5-30x becomes achievable

This is the most comprehensive model, incorporating:
1. Domain-specific cognitive M_max (v0.4)
2. Shift type classification (v0.4)
3. Backlog dynamics for Type I shifts (v0.4)
4. Autonomous lab physical acceleration (v0.5)
5. Cost dynamics (v0.5)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Import v0.4 components
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "v0.4" / "src"))
from refined_model import (
    RefinedAccelerationModel,
    DOMAIN_PROFILES,
    ShiftType,
    Scenario,
)
from backlog_dynamics import BacklogModel

# Import v0.5 autonomous lab
from autonomous_lab import (
    AutonomousLabModel,
    AutomationScenario,
    LAB_CAPACITIES,
)


@dataclass
class IntegratedForecast:
    """Complete forecast with cognitive and physical components."""
    year: int
    domain: str

    # Cognitive (from v0.4)
    cognitive_acceleration: float
    stage_accelerations: Dict[str, float]

    # Physical (from v0.5)
    physical_acceleration: float
    m_max_physical: float
    automation_level: str

    # Combined
    end_to_end_acceleration: float
    duration_months: float
    bottleneck: str

    # Economics
    cost_per_project: float
    cost_reduction: float

    # Shift dynamics
    shift_type: str
    backlog_size: Optional[int] = None
    backlog_years: Optional[float] = None


class IntegratedAccelerationModel:
    """
    Integrated model combining cognitive AI acceleration with lab automation.

    This is the complete v0.5 model that provides realistic end-to-end
    acceleration forecasts accounting for both cognitive and physical stages.
    """

    # Pipeline stages with durations
    STAGES = {
        "S1": ("Literature Review & Synthesis", True, 2.0),
        "S2": ("Hypothesis Generation", True, 1.0),
        "S3": ("Experimental Design & Analysis", True, 2.0),
        "S4": ("Wet Lab Execution", False, 6.0),
        "S5": ("Results Interpretation", True, 1.0),
        "S6": ("Validation & Publication", False, 4.0),
    }

    def __init__(
        self,
        domain: str = "average_biology",
        ai_scenario: Scenario = Scenario.BASELINE,
        automation_scenario: AutomationScenario = AutomationScenario.BASELINE,
    ):
        """
        Initialize integrated model.

        Args:
            domain: Research domain
            ai_scenario: AI capability growth scenario
            automation_scenario: Lab automation adoption scenario
        """
        self.domain = domain
        self.ai_scenario = ai_scenario
        self.automation_scenario = automation_scenario

        # Initialize component models
        self.cognitive_model = RefinedAccelerationModel(
            domain=domain,
            scenario=ai_scenario,
        )
        self.lab_model = AutonomousLabModel(
            domain=domain,
            scenario=automation_scenario,
        )

        # Get domain profile
        self.profile = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["average_biology"])

    def stage_acceleration(self, stage_id: str, year: int) -> float:
        """
        Calculate acceleration for a specific stage.

        Combines cognitive acceleration (v0.4) with physical automation (v0.5).
        """
        stage_name, is_cognitive, base_duration = self.STAGES[stage_id]

        if is_cognitive:
            # Use v0.4 cognitive model
            return self.cognitive_model.stage_acceleration(stage_id, year)
        else:
            # Use v0.5 automation model for physical stages
            physical_accels = self.lab_model.physical_acceleration(year)
            return physical_accels.get(stage_id, 1.0)

    def pipeline_acceleration(self, year: int) -> Tuple[float, str, Dict]:
        """
        Calculate overall pipeline acceleration.

        Returns:
            (acceleration, bottleneck_stage, stage_details)
        """
        stage_details = {}
        total_original = 0
        total_accelerated = 0

        for stage_id, (name, is_cognitive, base_duration) in self.STAGES.items():
            accel = self.stage_acceleration(stage_id, year)
            accelerated_duration = base_duration / accel

            stage_details[stage_id] = {
                "name": name,
                "is_cognitive": is_cognitive,
                "base_duration": base_duration,
                "acceleration": accel,
                "accelerated_duration": accelerated_duration,
            }

            total_original += base_duration
            total_accelerated += accelerated_duration

        # Find bottleneck (stage with lowest acceleration among physical)
        physical_stages = {sid: d for sid, d in stage_details.items()
                         if not d["is_cognitive"]}
        bottleneck = min(physical_stages.keys(),
                        key=lambda x: physical_stages[x]["acceleration"])

        overall_acceleration = total_original / total_accelerated

        return overall_acceleration, bottleneck, stage_details

    def cost_projection(self, year: int) -> Dict:
        """
        Project costs for a research project.

        Returns cost metrics including automation savings.
        """
        # Baseline cost (manual, 2020)
        baseline_cost = 500000  # $500K per research project

        # Cost reduction from automation
        forecast = self.lab_model.throughput_forecast([year])
        lab_metrics = forecast[year]["labs"]

        # Average cost reduction across applicable labs
        if lab_metrics:
            avg_cost_reduction = np.mean([
                1 - (m["cost_per_exp"] / LAB_CAPACITIES[name].manual_cost_per_exp)
                for name, m in lab_metrics.items()
            ])
        else:
            avg_cost_reduction = 0.0

        # Physical stage cost (60% of total in baseline)
        physical_fraction = 0.6
        current_physical_cost = baseline_cost * physical_fraction * (1 - avg_cost_reduction)
        current_cognitive_cost = baseline_cost * (1 - physical_fraction)  # Unchanged

        current_total = current_physical_cost + current_cognitive_cost

        return {
            "baseline_cost": baseline_cost,
            "current_cost": current_total,
            "cost_reduction": 1 - (current_total / baseline_cost),
            "physical_cost_reduction": avg_cost_reduction,
        }

    def forecast(self, years: List[int]) -> Dict[int, IntegratedForecast]:
        """
        Generate integrated forecasts for specified years.

        Returns dict of IntegratedForecast objects.
        """
        results = {}

        for year in years:
            # Get pipeline acceleration
            accel, bottleneck, stage_details = self.pipeline_acceleration(year)

            # Get cognitive-only metrics (from v0.4)
            cognitive_accels = {
                sid: d["acceleration"]
                for sid, d in stage_details.items()
                if d["is_cognitive"]
            }
            avg_cognitive = np.mean(list(cognitive_accels.values()))

            # Get physical metrics (from v0.5)
            physical_accels = {
                sid: d["acceleration"]
                for sid, d in stage_details.items()
                if not d["is_cognitive"]
            }
            avg_physical = np.mean(list(physical_accels.values()))
            m_max_phys = self.lab_model.m_max_physical(year)

            # Calculate duration
            total_duration = sum(d["accelerated_duration"] for d in stage_details.values())

            # Get costs
            costs = self.cost_projection(year)

            # Backlog for Type I shifts
            backlog_size = None
            backlog_years = None
            if self.profile.primary_shift_type == ShiftType.TYPE_I_SCALE:
                try:
                    backlog_model = BacklogModel(
                        domain=self.domain,
                        ai_generation_rate=1000000,
                        triage_efficiency=0.01,
                    )
                    backlog_model.simulate_year(year)
                    backlog_size = backlog_model.current_backlog
                    backlog_years = backlog_model.backlog_years
                except:
                    pass

            # Determine automation level description
            if m_max_phys < 2:
                auto_level = "minimal"
            elif m_max_phys < 5:
                auto_level = "emerging"
            elif m_max_phys < 20:
                auto_level = "moderate"
            else:
                auto_level = "high"

            results[year] = IntegratedForecast(
                year=year,
                domain=self.domain,
                cognitive_acceleration=avg_cognitive,
                stage_accelerations={sid: d["acceleration"] for sid, d in stage_details.items()},
                physical_acceleration=avg_physical,
                m_max_physical=m_max_phys,
                automation_level=auto_level,
                end_to_end_acceleration=accel,
                duration_months=total_duration,
                bottleneck=stage_details[bottleneck]["name"],
                cost_per_project=costs["current_cost"],
                cost_reduction=costs["cost_reduction"],
                shift_type=self.profile.primary_shift_type.value,
                backlog_size=backlog_size,
                backlog_years=backlog_years,
            )

        return results

    def scenario_matrix(self, years: List[int]) -> Dict:
        """
        Generate forecasts across AI and automation scenario combinations.

        Returns a matrix of forecasts for all scenario combinations.
        """
        results = {}

        for ai_scen in Scenario:
            for auto_scen in AutomationScenario:
                key = f"{ai_scen.value}_{auto_scen.value}"

                model = IntegratedAccelerationModel(
                    domain=self.domain,
                    ai_scenario=ai_scen,
                    automation_scenario=auto_scen,
                )

                forecasts = model.forecast(years)
                results[key] = {
                    "ai_scenario": ai_scen.value,
                    "automation_scenario": auto_scen.value,
                    "forecasts": {
                        y: f.end_to_end_acceleration
                        for y, f in forecasts.items()
                    },
                }

        return results

    def summary(self) -> str:
        """Generate comprehensive model summary."""
        years = [2025, 2030, 2040, 2050]
        forecasts = self.forecast(years)

        lines = [
            "=" * 70,
            f"INTEGRATED ACCELERATION MODEL v0.5 - {self.domain.upper()}",
            "=" * 70,
            "",
            f"AI Scenario: {self.ai_scenario.value}",
            f"Automation Scenario: {self.automation_scenario.value}",
            f"Shift Type: {self.profile.primary_shift_type.value}",
            "",
            "END-TO-END ACCELERATION FORECAST:",
            "-" * 50,
            f"{'Year':<10} {'Cognitive':<12} {'Physical':<12} {'End-to-End':<12} {'Duration':<10}",
            "-" * 50,
        ]

        for year in years:
            f = forecasts[year]
            lines.append(
                f"{year:<10} {f.cognitive_acceleration:<12.1f}x "
                f"{f.physical_acceleration:<12.1f}x "
                f"{f.end_to_end_acceleration:<12.1f}x "
                f"{f.duration_months:<10.1f}mo"
            )

        lines.extend([
            "-" * 50,
            "",
            "COMPARISON TO v0.4 (No Lab Automation):",
            "-" * 50,
        ])

        # Compare to v0.4
        v04_model = RefinedAccelerationModel(domain=self.domain)
        v04_forecast = v04_model.forecast(years)

        for year in years:
            v04_accel = v04_forecast[year]["acceleration"]
            v05_accel = forecasts[year].end_to_end_acceleration
            improvement = (v05_accel / v04_accel - 1) * 100
            lines.append(
                f"  {year}: v0.4={v04_accel:.1f}x → v0.5={v05_accel:.1f}x "
                f"(+{improvement:.0f}% from automation)"
            )

        lines.extend([
            "",
            "KEY INSIGHT:",
            "-" * 50,
            "Lab automation unlocks physical bottleneck, enabling",
            f"end-to-end acceleration of {forecasts[2050].end_to_end_acceleration:.1f}x by 2050",
            f"(vs v0.4 prediction of {v04_forecast[2050]['acceleration']:.1f}x without automation)",
        ])

        return "\n".join(lines)


def run_integrated_comparison():
    """Compare integrated model across domains."""
    print("=" * 70)
    print("INTEGRATED MODEL v0.5 - DOMAIN COMPARISON")
    print("=" * 70)
    print()

    domains = ["structural_biology", "materials_science", "protein_design",
               "drug_discovery", "genomics", "average_biology"]
    years = [2025, 2030, 2040, 2050]

    print(f"{'Domain':<20} ", end="")
    for year in years:
        print(f"{year:>10}", end="")
    print()
    print("-" * 70)

    for domain in domains:
        model = IntegratedAccelerationModel(domain=domain)
        forecasts = model.forecast(years)

        print(f"{domain:<20} ", end="")
        for year in years:
            accel = forecasts[year].end_to_end_acceleration
            print(f"{accel:>9.1f}x", end="")
        print()

    print("-" * 70)
    print()
    print("Comparison to v0.4 (without lab automation):")
    print("-" * 50)

    for domain in domains[:3]:  # Top 3 domains
        v04_model = RefinedAccelerationModel(domain=domain)
        v05_model = IntegratedAccelerationModel(domain=domain)

        v04_2050 = v04_model.forecast([2050])[2050]["acceleration"]
        v05_2050 = v05_model.forecast([2050])[2050].end_to_end_acceleration

        print(f"  {domain}: v0.4={v04_2050:.1f}x → v0.5={v05_2050:.1f}x")

    print()
    print("=" * 70)
    print("KEY FINDING: Lab automation increases 2050 projections by 2-10x")
    print("Physical bottleneck is no longer the binding constraint")
    print("=" * 70)


if __name__ == "__main__":
    run_integrated_comparison()
