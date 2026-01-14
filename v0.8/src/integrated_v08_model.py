#!/usr/bin/env python3
"""
Integrated v0.8 Model
=====================

Combines v0.7 enhancements with v0.8 probabilistic capabilities:

1. Monte Carlo Uncertainty Propagation
2. Prospective Validation Framework
3. Scenario Analysis with Explicit Assumptions
4. Regulatory Evolution Modeling

Key Output:
- Full posterior distributions (not just point estimates)
- Probability-weighted scenario ensembles
- Registered predictions for prospective validation
- Regulatory-conditional forecasts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import sys
from pathlib import Path

# Set up paths
_v08_src = Path(__file__).parent
_v07_src = _v08_src.parent.parent / "v0.7" / "src"
_v06_src = _v08_src.parent.parent / "v0.6" / "src"
_v05_src = _v08_src.parent.parent / "v0.5" / "src"
_v04_src = _v08_src.parent.parent / "v0.4" / "src"
_v03_src = _v08_src.parent.parent / "v0.3" / "src"

# Add paths
sys.path.insert(0, str(_v03_src))
sys.path.insert(0, str(_v04_src))
sys.path.insert(0, str(_v05_src))
sys.path.insert(0, str(_v06_src))
sys.path.insert(0, str(_v07_src))
sys.path.insert(0, str(_v08_src))

# Import using importlib
import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"v08_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import v0.8 components
_mc = _import_module("monte_carlo", _v08_src / "monte_carlo.py")
MonteCarloEngine = _mc.MonteCarloEngine
MonteCarloResult = _mc.MonteCarloResult

_scenario = _import_module("scenario_analysis", _v08_src / "scenario_analysis.py")
ScenarioAnalyzer = _scenario.ScenarioAnalyzer
ScenarioType = _scenario.ScenarioType
EnsembleForecast = _scenario.EnsembleForecast

_reg = _import_module("regulatory_scenarios", _v08_src / "regulatory_scenarios.py")
RegulatoryEvolutionModel = _reg.RegulatoryEvolutionModel
RegulatoryScenario = _reg.RegulatoryScenario

_prospective = _import_module("prospective_validation", _v08_src / "prospective_validation.py")
PredictionRegistry = _prospective.PredictionRegistry
ProspectivePrediction = _prospective.ProspectivePrediction


@dataclass
class ProbabilisticForecast:
    """Full probabilistic forecast from v0.8 model."""
    domain: str
    year: int

    # Point estimates
    mean_acceleration: float
    median_acceleration: float

    # Uncertainty
    std: float
    ci_50: Tuple[float, float]
    ci_90: Tuple[float, float]
    ci_95: Tuple[float, float]

    # Scenario breakdown
    scenario_forecasts: Dict[ScenarioType, float]
    scenario_probabilities: Dict[ScenarioType, float]

    # Distribution properties
    skewness: float
    samples: np.ndarray

    # For drug discovery: regulatory impact
    regulatory_impact: Optional[Dict] = None


class IntegratedV08Model:
    """
    v0.8 Model with full probabilistic output.

    Combines:
    - Monte Carlo uncertainty propagation
    - Scenario-weighted ensemble
    - Regulatory evolution (for drug discovery)
    - Prospective validation registration
    """

    def __init__(
        self,
        domain: str,
        n_samples: int = 10000,
        seed: int = None,
    ):
        self.domain = domain
        self.n_samples = n_samples
        self.seed = seed

        # Initialize component models
        self.mc_engine = MonteCarloEngine(
            domain=domain,
            n_samples=n_samples,
            seed=seed,
        )
        self.scenario_analyzer = ScenarioAnalyzer(domain=domain)

        # Regulatory model (only for drug discovery)
        self.regulatory_model = (
            RegulatoryEvolutionModel()
            if domain == "drug_discovery"
            else None
        )

    def forecast(self, year: int) -> ProbabilisticForecast:
        """Generate full probabilistic forecast."""
        # Monte Carlo simulation
        mc_result = self.mc_engine.run(year)

        # Scenario analysis
        scenario_ensemble = self.scenario_analyzer.ensemble_forecast(year)

        # Regulatory impact (drug discovery only)
        reg_impact = None
        if self.regulatory_model:
            reg_ensemble = self.regulatory_model.ensemble_forecast(year)
            reg_impact = {
                "weighted_accel": reg_ensemble["weighted_mean"],
                "ci_90": reg_ensemble["weighted_ci_90"],
            }

        return ProbabilisticForecast(
            domain=self.domain,
            year=year,
            mean_acceleration=mc_result.mean,
            median_acceleration=mc_result.median,
            std=mc_result.std,
            ci_50=mc_result.ci_50,
            ci_90=mc_result.ci_90,
            ci_95=mc_result.ci_95,
            scenario_forecasts={
                s: scenario_ensemble.scenario_forecasts[s].acceleration
                for s in ScenarioType
            },
            scenario_probabilities={
                s: scenario_ensemble.scenario_probabilities[s]
                for s in ScenarioType
            },
            skewness=mc_result.skewness,
            samples=mc_result.samples,
            regulatory_impact=reg_impact,
        )

    def forecast_trajectory(
        self,
        years: List[int] = None,
    ) -> Dict[int, ProbabilisticForecast]:
        """Generate forecasts for multiple years."""
        years = years or [2025, 2030, 2040, 2050]
        return {year: self.forecast(year) for year in years}

    def register_predictions(
        self,
        registry: PredictionRegistry,
        years: List[int] = None,
    ) -> List[str]:
        """Register predictions for prospective validation."""
        years = years or [2025, 2030]
        prediction_ids = []

        for year in years:
            f = self.forecast(year)

            pred = ProspectivePrediction(
                domain=self.domain,
                target_year=year,
                metric="end_to_end_acceleration",
                point_prediction=f.mean_acceleration,
                ci_90_lower=f.ci_90[0],
                ci_90_upper=f.ci_90[1],
                ci_50_lower=f.ci_50[0],
                ci_50_upper=f.ci_50[1],
                assumptions=[
                    f"Baseline scenario probability: {f.scenario_probabilities[ScenarioType.BASELINE]:.0%}",
                    f"Monte Carlo samples: {self.n_samples}",
                ],
                caveats=[
                    "Uncertainty from parameter distributions",
                    "Scenario probabilities are subjective",
                ],
            )

            pred_id = registry.register(pred, "v0.8")
            prediction_ids.append(pred_id)

        return prediction_ids

    def summary(self) -> str:
        """Generate comprehensive model summary."""
        forecasts = self.forecast_trajectory([2025, 2030, 2040, 2050])

        lines = [
            "=" * 100,
            f"v0.8 PROBABILISTIC MODEL: {self.domain.upper()}",
            "=" * 100,
            "",
            f"Monte Carlo samples: {self.n_samples:,}",
            "",
            "PROBABILISTIC FORECAST:",
            "-" * 100,
            f"{'Year':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'50% CI':<18} {'90% CI':<18}",
            "-" * 100,
        ]

        for year in [2025, 2030, 2040, 2050]:
            f = forecasts[year]
            ci_50 = f"[{f.ci_50[0]:.1f}-{f.ci_50[1]:.1f}]"
            ci_90 = f"[{f.ci_90[0]:.1f}-{f.ci_90[1]:.1f}]"
            lines.append(
                f"{year:<8} {f.mean_acceleration:>8.1f}x {f.median_acceleration:>8.1f}x "
                f"{f.std:>8.2f} {ci_50:<18} {ci_90:<18}"
            )

        lines.extend([
            "-" * 100,
            "",
            "SCENARIO BREAKDOWN (2030):",
        ])

        f_2030 = forecasts[2030]
        for s in ScenarioType:
            accel = f_2030.scenario_forecasts[s]
            prob = f_2030.scenario_probabilities[s]
            lines.append(f"  {s.value:<15}: {accel:>6.1f}x (P={prob:.0%})")

        if self.regulatory_model and f_2030.regulatory_impact:
            lines.extend([
                "",
                "REGULATORY IMPACT (Drug Discovery):",
                f"  Ensemble acceleration: {f_2030.regulatory_impact['weighted_accel']:.2f}x",
                f"  90% CI: [{f_2030.regulatory_impact['ci_90'][0]:.2f}x - {f_2030.regulatory_impact['ci_90'][1]:.2f}x]",
            ])

        lines.extend([
            "",
            "DISTRIBUTION PROPERTIES (2030):",
            f"  Skewness: {f_2030.skewness:.2f} (>0 = right-tailed)",
            "",
            "KEY INSIGHT: v0.8 provides full posterior distributions,",
            "enabling rigorous uncertainty quantification for policy decisions.",
        ])

        return "\n".join(lines)


def compare_all_domains():
    """Compare v0.8 forecasts across domains."""
    print("=" * 100)
    print("v0.8 CROSS-DOMAIN COMPARISON (2030)")
    print("=" * 100)
    print()

    domains = ["structural_biology", "drug_discovery", "materials_science",
               "protein_design", "clinical_genomics"]

    print(f"{'Domain':<22} {'Mean':<10} {'Median':<10} {'90% CI':<20} {'Skewness':<10}")
    print("-" * 100)

    for domain in domains:
        model = IntegratedV08Model(domain=domain, n_samples=5000, seed=42)
        f = model.forecast(2030)

        ci = f"[{f.ci_90[0]:.1f} - {f.ci_90[1]:.1f}]"
        print(
            f"{domain:<22} {f.mean_acceleration:>8.1f}x {f.median_acceleration:>8.1f}x "
            f"{ci:<20} {f.skewness:>8.2f}"
        )

    print("-" * 100)
    print()
    print("KEY: All forecasts include full uncertainty propagation")
    print("     and probability-weighted scenario ensembles.")


if __name__ == "__main__":
    # Cross-domain comparison
    compare_all_domains()

    print()
    print()

    # Detailed summary for one domain
    model = IntegratedV08Model(domain="drug_discovery", n_samples=10000, seed=42)
    print(model.summary())
