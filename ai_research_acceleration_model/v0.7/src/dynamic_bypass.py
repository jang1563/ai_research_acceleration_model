#!/usr/bin/env python3
"""
Dynamic Simulation Bypass Model for v0.7
=========================================

Addresses Expert Review M2-P3: "Static simulation bypass potential"

The v0.6 model used a static `simulation_bypass_potential` (0-0.8) parameter.
In reality, simulation capability grows with AI improvement, compute availability,
and validated training data.

v0.7 Enhancement:
- Dynamic bypass potential that evolves over time
- AI capability-dependent bypass rates
- Domain-specific simulation maturity curves
- Compute scaling effects
- Validation feedback loops (better data -> better simulations)

Key Insight:
Simulation bypass is the key to breaking the validation bottleneck.
If AI can reliably predict experimental outcomes, physical validation
becomes confirmation rather than discovery.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class SimulationMaturity(Enum):
    """Maturity stages for simulation capabilities."""
    NASCENT = "nascent"           # Early stage, high error rates
    EMERGING = "emerging"         # Growing capability, moderate errors
    ESTABLISHED = "established"   # Reliable for standard cases
    MATURE = "mature"             # Near-experimental accuracy
    DOMINANT = "dominant"         # Simulation preferred over experiment


@dataclass
class SimulationCapability:
    """State of simulation capability at a given point."""
    year: int
    domain: str

    # Core metrics
    bypass_potential: float      # Fraction of validation replaceable (0-1)
    accuracy: float              # Simulation accuracy (0-1)
    confidence_calibration: float  # How well-calibrated confidence scores are (0-1)

    # Growth drivers
    compute_available: float     # Relative compute (1.0 = 2024 baseline)
    training_data_size: float    # Validated data points for training
    model_generation: int        # AI model generation (1=GPT-4 era, 2=next gen, etc.)

    # Derived metrics
    maturity: SimulationMaturity
    effective_throughput_multiplier: float  # How much simulation extends capacity


@dataclass
class DomainSimulationProfile:
    """Domain-specific simulation parameters."""
    name: str

    # Starting point (2024)
    base_bypass_potential: float    # Current bypass fraction
    base_accuracy: float            # Current simulation accuracy

    # Growth parameters
    max_bypass_potential: float     # Theoretical maximum bypass (asymptote)
    accuracy_growth_rate: float     # Annual improvement in accuracy
    compute_elasticity: float       # How much compute helps (0-1)
    data_elasticity: float          # How much more data helps (0-1)

    # Milestones (years when maturity levels are reached)
    emerging_year: int
    established_year: int
    mature_year: int
    dominant_year: int

    # Physics constraints
    inherently_physical_fraction: float  # Fraction that MUST be physical (0-1)


# Domain profiles based on literature and case studies
SIMULATION_PROFILES = {
    "structural_biology": DomainSimulationProfile(
        name="Structural Biology",
        base_bypass_potential=0.50,    # AlphaFold already high confidence
        base_accuracy=0.85,            # GDT-TS ~85 for AF2
        max_bypass_potential=0.95,     # Some structures need experiment
        accuracy_growth_rate=0.05,     # 5% annual improvement
        compute_elasticity=0.3,        # Diminishing returns on compute
        data_elasticity=0.2,           # Limited by structure diversity
        emerging_year=2020,            # AlphaFold2
        established_year=2024,         # Current state
        mature_year=2028,
        dominant_year=2032,
        inherently_physical_fraction=0.05,  # Novel folds, dynamics
    ),

    "materials_science": DomainSimulationProfile(
        name="Materials Science",
        base_bypass_potential=0.15,    # DFT limited, synthesis needed
        base_accuracy=0.70,            # DFT accuracy ~70%
        max_bypass_potential=0.80,     # Synthesis always needed for some
        accuracy_growth_rate=0.08,     # 8% annual (ML potentials improving)
        compute_elasticity=0.6,        # Compute helps significantly
        data_elasticity=0.5,           # More data helps
        emerging_year=2023,            # GNoME, M3GNet
        established_year=2028,
        mature_year=2035,
        dominant_year=2045,
        inherently_physical_fraction=0.20,  # Synthesis, defects
    ),

    "drug_discovery": DomainSimulationProfile(
        name="Drug Discovery",
        base_bypass_potential=0.10,    # ADMET prediction limited
        base_accuracy=0.60,            # ~60% ADMET accuracy
        max_bypass_potential=0.70,     # Biology too complex
        accuracy_growth_rate=0.06,     # 6% annual
        compute_elasticity=0.5,        # MD simulations help
        data_elasticity=0.7,           # Data is key bottleneck
        emerging_year=2024,            # Current AlphaFold + docking
        established_year=2030,
        mature_year=2038,
        dominant_year=2050,
        inherently_physical_fraction=0.30,  # Clinical trials mandatory
    ),

    "protein_design": DomainSimulationProfile(
        name="Protein Design",
        base_bypass_potential=0.20,    # Structure prediction good, function hard
        base_accuracy=0.65,            # ~65% fold prediction
        max_bypass_potential=0.85,     # Function prediction improving
        accuracy_growth_rate=0.10,     # 10% annual (rapid progress)
        compute_elasticity=0.4,        # Some compute benefit
        data_elasticity=0.6,           # Experimental data crucial
        emerging_year=2023,            # ESM-3, RFdiffusion
        established_year=2027,
        mature_year=2033,
        dominant_year=2040,
        inherently_physical_fraction=0.15,  # Expression, stability
    ),

    "clinical_genomics": DomainSimulationProfile(
        name="Clinical Genomics",
        base_bypass_potential=0.40,    # AlphaMissense high confidence
        base_accuracy=0.80,            # 80% variant classification
        max_bypass_potential=0.90,     # Clinical validation required
        accuracy_growth_rate=0.04,     # 4% annual (slower)
        compute_elasticity=0.2,        # Compute not limiting
        data_elasticity=0.8,           # Clinical data is key
        emerging_year=2022,            # CADD, REVEL era
        established_year=2023,         # AlphaMissense
        mature_year=2030,
        dominant_year=2040,
        inherently_physical_fraction=0.10,  # Clinical phenotyping
    ),
}


@dataclass
class ComputeScaling:
    """Models compute availability over time."""

    # Moore's law / scaling parameters
    base_year: int = 2024
    base_compute: float = 1.0

    # Annual compute growth rate (considering AI-specific hardware)
    annual_growth_rate: float = 0.35  # 35%/year (faster than Moore's law)

    # Cost reduction rate
    annual_cost_reduction: float = 0.20  # 20%/year cost decrease

    def available_compute(self, year: int, budget_growth: float = 0.10) -> float:
        """
        Calculate available compute at a given year.

        Args:
            year: Target year
            budget_growth: Annual research budget growth rate

        Returns:
            Relative compute (1.0 = 2024 baseline)
        """
        t = year - self.base_year

        # Hardware improvement
        hardware_factor = (1 + self.annual_growth_rate) ** t

        # Cost reduction allows more compute per dollar
        cost_factor = (1 + self.annual_cost_reduction) ** t

        # Budget growth
        budget_factor = (1 + budget_growth) ** t

        return hardware_factor * cost_factor * budget_factor


class DynamicBypassModel:
    """
    Models the evolution of simulation bypass capability over time.

    Key dynamics:
    1. AI model improvements increase accuracy
    2. More compute enables larger simulations
    3. Validated data improves training
    4. Confidence calibration improves trust

    The model captures the positive feedback loop:
    Better simulations -> More trust -> More use -> More validation data -> Better simulations
    """

    def __init__(
        self,
        domain: str,
        compute_scaling: Optional[ComputeScaling] = None,
    ):
        self.domain = domain
        self.profile = SIMULATION_PROFILES.get(
            domain, SIMULATION_PROFILES["materials_science"]
        )
        self.compute = compute_scaling or ComputeScaling()

    def _determine_maturity(self, year: int) -> SimulationMaturity:
        """Determine simulation maturity stage for a given year."""
        if year < self.profile.emerging_year:
            return SimulationMaturity.NASCENT
        elif year < self.profile.established_year:
            return SimulationMaturity.EMERGING
        elif year < self.profile.mature_year:
            return SimulationMaturity.ESTABLISHED
        elif year < self.profile.dominant_year:
            return SimulationMaturity.MATURE
        else:
            return SimulationMaturity.DOMINANT

    def _model_generation(self, year: int) -> int:
        """
        Estimate AI model generation.

        Generation 1: 2020-2024 (GPT-4, AlphaFold2)
        Generation 2: 2025-2028 (Next-gen multimodal)
        Generation 3: 2029-2033 (Advanced reasoning)
        Generation 4: 2034+ (Unknown advances)
        """
        if year < 2025:
            return 1
        elif year < 2029:
            return 2
        elif year < 2034:
            return 3
        else:
            return 4

    def _accuracy(self, year: int, validated_data: float) -> float:
        """
        Calculate simulation accuracy at given year.

        Accuracy improves with:
        1. Time (AI model improvements)
        2. Validated data (training data)
        3. Compute (can run larger models)
        """
        t = year - 2024

        # Base improvement from AI progress
        time_improvement = self.profile.base_accuracy + (
            (0.98 - self.profile.base_accuracy) *
            (1 - np.exp(-self.profile.accuracy_growth_rate * t))
        )

        # Data improvement (log scaling)
        data_factor = 1 + self.profile.data_elasticity * np.log10(
            1 + validated_data / 10000
        )

        # Compute improvement
        compute = self.compute.available_compute(year)
        compute_factor = 1 + self.profile.compute_elasticity * np.log10(
            max(1, compute)
        )

        # Combined accuracy (capped at 0.99)
        accuracy = time_improvement * data_factor * compute_factor
        return min(accuracy, 0.99)

    def _bypass_potential(self, year: int, accuracy: float) -> float:
        """
        Calculate simulation bypass potential.

        Bypass potential depends on:
        1. Simulation accuracy
        2. Domain-specific maximum
        3. Inherently physical fraction
        """
        # Accuracy-dependent bypass
        # Below 70% accuracy: minimal bypass (not trusted)
        # 70-90% accuracy: linear growth
        # Above 90%: approaches maximum

        if accuracy < 0.70:
            accuracy_factor = accuracy / 0.70 * 0.3  # Up to 30% of max
        elif accuracy < 0.90:
            accuracy_factor = 0.3 + (accuracy - 0.70) / 0.20 * 0.5  # 30-80% of max
        else:
            accuracy_factor = 0.8 + (accuracy - 0.90) / 0.10 * 0.2  # 80-100% of max

        # Domain maximum (limited by inherently physical work)
        domain_max = self.profile.max_bypass_potential

        return domain_max * accuracy_factor

    def _confidence_calibration(self, year: int, validated_data: float) -> float:
        """
        Calculate confidence calibration score.

        Well-calibrated confidence enables selective bypass:
        - High confidence predictions: skip physical validation
        - Low confidence predictions: require physical validation
        """
        t = year - 2024

        # Base improvement over time
        base_calibration = 0.5 + 0.4 * (1 - np.exp(-0.15 * t))

        # Data improvement (more validation data = better calibration)
        data_factor = 1 + 0.3 * np.log10(1 + validated_data / 1000)

        return min(base_calibration * data_factor, 0.95)

    def simulate(
        self,
        years: List[int],
        initial_validated_data: float = 10000,
    ) -> Dict[int, SimulationCapability]:
        """
        Simulate bypass capability evolution over time.

        Args:
            years: Years to simulate
            initial_validated_data: Starting validated dataset size

        Returns:
            Dictionary mapping years to SimulationCapability
        """
        results = {}
        validated_data = initial_validated_data

        for year in sorted(years):
            # Calculate compute available
            compute = self.compute.available_compute(year)

            # Calculate accuracy
            accuracy = self._accuracy(year, validated_data)

            # Calculate bypass potential
            bypass = self._bypass_potential(year, accuracy)

            # Calculate confidence calibration
            calibration = self._confidence_calibration(year, validated_data)

            # Determine maturity
            maturity = self._determine_maturity(year)

            # Calculate effective throughput multiplier
            # If bypass = 0.5 and accuracy = 0.9, then:
            # 50% of work is simulated (no physical validation needed)
            # Effective throughput = 1 / (1 - bypass * accuracy)
            effective_bypass = bypass * calibration
            throughput_multiplier = 1 / (1 - effective_bypass) if effective_bypass < 1 else 10.0

            results[year] = SimulationCapability(
                year=year,
                domain=self.domain,
                bypass_potential=bypass,
                accuracy=accuracy,
                confidence_calibration=calibration,
                compute_available=compute,
                training_data_size=validated_data,
                model_generation=self._model_generation(year),
                maturity=maturity,
                effective_throughput_multiplier=throughput_multiplier,
            )

            # Update validated data (grows with physical validations)
            # Assume 1000/year base + growth
            validated_data += 1000 * (1.1 ** (year - 2024))

        return results

    def get_bypass_trajectory(
        self,
        start_year: int = 2024,
        end_year: int = 2050,
    ) -> List[Tuple[int, float]]:
        """Get bypass potential trajectory over time."""
        years = list(range(start_year, end_year + 1))
        results = self.simulate(years)
        return [(year, results[year].bypass_potential) for year in years]

    def summary(self) -> str:
        """Generate summary of dynamic bypass model."""
        years = [2024, 2030, 2040, 2050]
        results = self.simulate(years)

        lines = [
            "=" * 80,
            f"DYNAMIC SIMULATION BYPASS MODEL: {self.profile.name}",
            "=" * 80,
            "",
            "KEY PARAMETERS:",
            f"  Base bypass (2024): {self.profile.base_bypass_potential:.0%}",
            f"  Maximum bypass: {self.profile.max_bypass_potential:.0%}",
            f"  Inherently physical: {self.profile.inherently_physical_fraction:.0%}",
            "",
            "BYPASS EVOLUTION:",
            "-" * 80,
            f"{'Year':<8} {'Bypass':<10} {'Accuracy':<10} {'Calibration':<12} {'Throughput':<12} {'Maturity':<12}",
            "-" * 80,
        ]

        for year in years:
            s = results[year]
            lines.append(
                f"{year:<8} {s.bypass_potential:>8.0%} {s.accuracy:>8.0%} "
                f"{s.confidence_calibration:>10.0%} {s.effective_throughput_multiplier:>10.1f}x "
                f"{s.maturity.value:<12}"
            )

        lines.extend([
            "-" * 80,
            "",
            "COMPUTE SCALING:",
            f"  2030: {self.compute.available_compute(2030):.1f}x vs 2024",
            f"  2040: {self.compute.available_compute(2040):.1f}x vs 2024",
            f"  2050: {self.compute.available_compute(2050):.1f}x vs 2024",
            "",
            "KEY INSIGHT:",
        ])

        state_2030 = results[2030]
        if state_2030.bypass_potential > 0.5:
            lines.append(
                f"  By 2030, {state_2030.bypass_potential:.0%} of validation can be simulated, "
                f"providing {state_2030.effective_throughput_multiplier:.1f}x effective throughput"
            )
        else:
            lines.append(
                f"  Simulation bypass growing but still limited ({state_2030.bypass_potential:.0%} by 2030)"
            )
            lines.append(f"  Physical validation remains the primary bottleneck")

        return "\n".join(lines)


def compare_bypass_across_domains():
    """Compare bypass evolution across all domains."""
    print("=" * 80)
    print("DYNAMIC BYPASS COMPARISON ACROSS DOMAINS")
    print("=" * 80)
    print()

    years = [2024, 2030, 2040, 2050]

    print(f"{'Domain':<22} {'2024':<10} {'2030':<10} {'2040':<10} {'2050':<10}")
    print("-" * 80)

    for domain in SIMULATION_PROFILES.keys():
        model = DynamicBypassModel(domain=domain)
        results = model.simulate(years)

        bypasses = [f"{results[y].bypass_potential:.0%}" for y in years]
        print(f"{domain:<22} {bypasses[0]:<10} {bypasses[1]:<10} {bypasses[2]:<10} {bypasses[3]:<10}")

    print("-" * 80)
    print()
    print("KEY INSIGHT: Structural biology leads in simulation bypass (AlphaFold effect),")
    print("while drug discovery lags due to inherent biological complexity.")


if __name__ == "__main__":
    compare_bypass_across_domains()
    print()
    print()

    # Detailed view for materials science
    model = DynamicBypassModel(domain="materials_science")
    print(model.summary())
