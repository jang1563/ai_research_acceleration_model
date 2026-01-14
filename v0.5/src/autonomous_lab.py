"""
Autonomous Laboratory Model
===========================

Models the scaling of automated/autonomous laboratories and their
impact on physical stage acceleration.

Key Insight: The physical bottleneck in v0.4 (M_max_physical = 1.0-1.5x)
assumes traditional manual labs. Autonomous labs can potentially
increase this by 10-100x, unlocking much higher end-to-end acceleration.

Case Studies:
-------------
- A-Lab (Berkeley, 2023): 350 materials/year, 71% success rate
  - Traditional manual: ~50-100 materials/year
  - Acceleration: ~3-7x for synthesis

- Emerald Cloud Lab: Remote robotic execution
  - 24/7 operation vs 8-hour human shifts
  - Potential 3x from continuous operation alone

- High-throughput screening: 100,000+ compounds/day
  - vs manual: ~100/day
  - Potential 1000x for specific assays
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class AutomationScenario(Enum):
    """Scenarios for lab automation adoption."""
    CONSERVATIVE = "conservative"   # Slow adoption, technical barriers
    BASELINE = "baseline"           # Current trajectory continues
    OPTIMISTIC = "optimistic"       # Rapid investment and adoption
    BREAKTHROUGH = "breakthrough"   # Major technical breakthrough


@dataclass
class LabCapacity:
    """
    Capacity model for an autonomous lab type.

    Models the growth in throughput as automation scales.
    """
    name: str
    description: str

    # Throughput parameters
    manual_rate: float              # Experiments per year (manual baseline)
    current_automated_rate: float   # Current automated rate (2024)
    max_automated_rate: float       # Theoretical maximum with full automation

    # Growth parameters
    adoption_year: int = 2023       # Year automation became available
    growth_rate: float = 0.20       # Annual growth in automated capacity

    # Cost parameters
    manual_cost_per_exp: float = 1000.0    # $ per experiment (manual)
    automated_cost_per_exp: float = 100.0  # $ per experiment (automated, at scale)

    # Quality parameters
    success_rate_manual: float = 0.3       # Success rate for manual
    success_rate_automated: float = 0.7    # Success rate for automated (A-Lab: 71%)

    # Domain
    applicable_stages: List[str] = field(default_factory=lambda: ["S4"])
    domain: str = "general"

    def throughput_at_year(self, year: int, scenario: AutomationScenario) -> float:
        """Calculate automated throughput at a given year."""
        if year < self.adoption_year:
            return self.manual_rate

        t = year - self.adoption_year

        # Scenario-dependent growth rates
        scenario_multipliers = {
            AutomationScenario.CONSERVATIVE: 0.5,
            AutomationScenario.BASELINE: 1.0,
            AutomationScenario.OPTIMISTIC: 1.5,
            AutomationScenario.BREAKTHROUGH: 2.5,
        }

        effective_growth = self.growth_rate * scenario_multipliers[scenario]

        # Logistic growth toward maximum
        midpoint = 15  # Years to reach half of max
        capacity_fraction = 1 / (1 + np.exp(-effective_growth * (t - midpoint)))

        current_capacity = self.manual_rate + (
            (self.max_automated_rate - self.manual_rate) * capacity_fraction
        )

        return current_capacity

    def acceleration_factor(self, year: int, scenario: AutomationScenario) -> float:
        """Calculate acceleration factor vs manual baseline."""
        throughput = self.throughput_at_year(year, scenario)

        # Account for improved success rate (capped to avoid extreme values)
        success_ratio = min(
            self.success_rate_automated / self.success_rate_manual,
            2.0  # Cap success rate improvement at 2x
        )

        effective_throughput = throughput * success_ratio

        # Cap maximum acceleration to realistic levels
        raw_accel = effective_throughput / self.manual_rate
        return min(raw_accel, 100.0)  # Cap at 100x max

    def cost_per_experiment(self, year: int, scenario: AutomationScenario) -> float:
        """Calculate cost per experiment at a given year."""
        if year < self.adoption_year:
            return self.manual_cost_per_exp

        t = year - self.adoption_year

        # Cost decreases as scale increases (learning curve)
        # Wright's Law: cost decreases ~20% for each doubling of cumulative production
        throughput = self.throughput_at_year(year, scenario)
        scale_factor = throughput / self.manual_rate

        # Cost reduction follows power law
        learning_rate = 0.8  # 80% learning curve (20% reduction per doubling)
        cost_factor = scale_factor ** (np.log2(learning_rate))

        current_cost = self.manual_cost_per_exp * cost_factor

        # Floor at automated cost
        return max(current_cost, self.automated_cost_per_exp)


# Lab capacities by domain (calibrated from real systems)
LAB_CAPACITIES = {
    "materials_synthesis": LabCapacity(
        name="Materials Synthesis Lab",
        description="A-Lab style automated materials synthesis",
        manual_rate=100,              # Materials per lab per year
        current_automated_rate=350,   # A-Lab 2023
        max_automated_rate=5000,      # With multiple A-Labs
        adoption_year=2023,
        growth_rate=0.25,             # 25% annual capacity growth
        manual_cost_per_exp=5000,
        automated_cost_per_exp=500,
        success_rate_manual=0.3,
        success_rate_automated=0.71,  # A-Lab achieved 71%
        applicable_stages=["S4"],
        domain="materials_science",
    ),

    "protein_expression": LabCapacity(
        name="Protein Expression Lab",
        description="Automated protein expression and purification",
        manual_rate=500,              # Proteins per lab per year
        current_automated_rate=2000,
        max_automated_rate=20000,
        adoption_year=2020,
        growth_rate=0.20,
        manual_cost_per_exp=2000,
        automated_cost_per_exp=200,
        success_rate_manual=0.4,
        success_rate_automated=0.6,
        applicable_stages=["S4"],
        domain="protein_design",
    ),

    "high_throughput_screening": LabCapacity(
        name="High-Throughput Screening",
        description="Automated compound screening",
        manual_rate=1000,             # Compounds per day manual
        current_automated_rate=100000, # Already highly automated
        max_automated_rate=1000000,
        adoption_year=2010,
        growth_rate=0.15,
        manual_cost_per_exp=100,
        automated_cost_per_exp=1,
        success_rate_manual=0.01,     # Hit rate
        success_rate_automated=0.01,  # Same hit rate, just faster
        applicable_stages=["S4"],
        domain="drug_discovery",
    ),

    "cell_culture": LabCapacity(
        name="Automated Cell Culture",
        description="Robotic cell culture and assays",
        manual_rate=200,              # Experiments per year
        current_automated_rate=1000,
        max_automated_rate=10000,
        adoption_year=2018,
        growth_rate=0.18,
        manual_cost_per_exp=500,
        automated_cost_per_exp=50,
        success_rate_manual=0.7,
        success_rate_automated=0.85,
        applicable_stages=["S4"],
        domain="average_biology",
    ),

    "dna_synthesis": LabCapacity(
        name="DNA Synthesis",
        description="Automated DNA/gene synthesis",
        manual_rate=100,              # Genes per year
        current_automated_rate=10000,
        max_automated_rate=100000,
        adoption_year=2015,
        growth_rate=0.30,             # Rapid improvement
        manual_cost_per_exp=10000,
        automated_cost_per_exp=100,
        success_rate_manual=0.8,
        success_rate_automated=0.95,
        applicable_stages=["S4"],
        domain="genomics",
    ),

    "crystallography": LabCapacity(
        name="Automated Crystallography",
        description="Robotic crystallization and X-ray (mostly replaced by AlphaFold)",
        manual_rate=50,               # Structures per year per lab
        current_automated_rate=200,   # More conservative (AlphaFold reduced need)
        max_automated_rate=1000,      # Limited by remaining validation needs
        adoption_year=2015,
        growth_rate=0.10,             # Slower growth (demand shifted to computational)
        manual_cost_per_exp=50000,
        automated_cost_per_exp=10000,
        success_rate_manual=0.3,
        success_rate_automated=0.4,   # Modest improvement
        applicable_stages=["S4", "S6"],
        domain="structural_biology",
    ),
}


class AutonomousLabModel:
    """
    Models autonomous laboratory scaling and its impact on research acceleration.

    Key insight: Autonomous labs can increase M_max_physical from 1.5x to 10-50x,
    dramatically changing the end-to-end acceleration picture.
    """

    def __init__(
        self,
        domain: str = "average_biology",
        scenario: AutomationScenario = AutomationScenario.BASELINE,
    ):
        """
        Initialize autonomous lab model.

        Args:
            domain: Research domain
            scenario: Automation adoption scenario
        """
        self.domain = domain
        self.scenario = scenario

        # Find applicable lab capacities for this domain
        self.applicable_labs = {
            name: cap for name, cap in LAB_CAPACITIES.items()
            if cap.domain == domain or cap.domain == "general"
        }

        if not self.applicable_labs:
            # Use cell_culture as default
            self.applicable_labs = {"cell_culture": LAB_CAPACITIES["cell_culture"]}

    def physical_acceleration(self, year: int) -> Dict[str, float]:
        """
        Calculate physical stage acceleration from automation.

        Returns dict with acceleration factor for each applicable stage.
        """
        stage_accels = {"S4": 1.0, "S6": 1.0}

        for lab_name, lab in self.applicable_labs.items():
            accel = lab.acceleration_factor(year, self.scenario)

            for stage in lab.applicable_stages:
                # Take maximum acceleration if multiple labs apply
                stage_accels[stage] = max(stage_accels[stage], accel)

        return stage_accels

    def m_max_physical(self, year: int) -> float:
        """
        Calculate revised M_max_physical based on automation level.

        This is the key output: automation increases the physical ceiling.
        """
        stage_accels = self.physical_acceleration(year)

        # Return the minimum (bottleneck) of physical stages
        return min(stage_accels.values())

    def throughput_forecast(self, years: List[int]) -> Dict[int, Dict]:
        """
        Forecast lab throughput over time.

        Returns dict with throughput metrics per year.
        """
        results = {}

        for year in years:
            lab_metrics = {}
            for lab_name, lab in self.applicable_labs.items():
                lab_metrics[lab_name] = {
                    "throughput": lab.throughput_at_year(year, self.scenario),
                    "acceleration": lab.acceleration_factor(year, self.scenario),
                    "cost_per_exp": lab.cost_per_experiment(year, self.scenario),
                }

            results[year] = {
                "year": year,
                "labs": lab_metrics,
                "m_max_physical": self.m_max_physical(year),
                "physical_acceleration": self.physical_acceleration(year),
            }

        return results

    def scenario_comparison(self, years: List[int]) -> Dict[str, Dict]:
        """
        Compare different automation scenarios.

        Returns dict with forecasts for each scenario.
        """
        results = {}

        for scenario in AutomationScenario:
            model = AutonomousLabModel(domain=self.domain, scenario=scenario)
            results[scenario.value] = model.throughput_forecast(years)

        return results

    def summary(self) -> str:
        """Generate model summary."""
        lines = [
            "=" * 60,
            f"AUTONOMOUS LAB MODEL - {self.domain.upper()}",
            "=" * 60,
            "",
            f"Scenario: {self.scenario.value}",
            f"Applicable Labs: {list(self.applicable_labs.keys())}",
            "",
            "Physical Stage Acceleration (M_max_physical):",
            "-" * 40,
        ]

        years = [2025, 2030, 2040, 2050]
        for year in years:
            m_max = self.m_max_physical(year)
            lines.append(f"  {year}: {m_max:.1f}x")

        lines.append("")
        lines.append("Lab-Level Details (2030):")
        lines.append("-" * 40)

        forecast = self.throughput_forecast([2030])
        for lab_name, metrics in forecast[2030]["labs"].items():
            lines.append(f"  {lab_name}:")
            lines.append(f"    Throughput: {metrics['throughput']:,.0f}/year")
            lines.append(f"    Acceleration: {metrics['acceleration']:.1f}x")
            lines.append(f"    Cost/exp: ${metrics['cost_per_exp']:,.0f}")

        return "\n".join(lines)


def run_automation_comparison():
    """Compare automation scenarios across domains."""
    print("=" * 70)
    print("AUTONOMOUS LAB SCALING - SCENARIO COMPARISON")
    print("=" * 70)
    print()

    domains = ["materials_science", "protein_design", "drug_discovery", "average_biology"]
    years = [2025, 2030, 2040, 2050]

    for domain in domains:
        print(f"\n{domain.upper()}")
        print("-" * 50)
        print(f"{'Scenario':<15} ", end="")
        for year in years:
            print(f"{year:>10}", end="")
        print()

        for scenario in AutomationScenario:
            model = AutonomousLabModel(domain=domain, scenario=scenario)
            print(f"{scenario.value:<15} ", end="")
            for year in years:
                m_max = model.m_max_physical(year)
                print(f"{m_max:>9.1f}x", end="")
            print()

    print()
    print("=" * 70)
    print("KEY INSIGHT: Automation can increase M_max_physical from 1.5x to 10-50x")
    print("This would dramatically change end-to-end acceleration projections")
    print("=" * 70)


if __name__ == "__main__":
    run_automation_comparison()
