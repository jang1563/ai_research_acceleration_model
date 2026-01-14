"""
Monte Carlo Simulation Module
=============================

Provides uncertainty quantification through Monte Carlo simulation
with Sobol sensitivity analysis support.

Based on PROJECT_BIBLE.md Section 6.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .pipeline import ResearchPipeline, Stage, StageParameters, DEFAULT_STAGES, StageType
from .paradigm_shift import ParadigmShiftModule, ShiftType
from .model import (
    AIResearchAccelerationModel,
    Scenario,
    ScenarioParams,
    SCENARIO_DEFAULTS,
    InfrastructureConstraints,
    AIFailureModes,
)


@dataclass
class SimulationResult:
    """Results from a single Monte Carlo run."""
    run_id: int
    scenario: str
    g_ai: float
    acceleration_by_year: Dict[int, float]
    throughput_by_year: Dict[int, float]
    discoveries_by_year: Dict[int, float]
    bottleneck_by_year: Dict[int, str]
    parameters: Dict[str, float]  # Sampled parameters for this run


@dataclass
class SimulationSummary:
    """Aggregated results from Monte Carlo simulation."""
    n_samples: int
    scenario: str
    years: List[int]

    # Acceleration statistics
    acceleration_mean: Dict[int, float] = field(default_factory=dict)
    acceleration_std: Dict[int, float] = field(default_factory=dict)
    acceleration_ci_80: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    acceleration_ci_95: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Throughput statistics
    throughput_mean: Dict[int, float] = field(default_factory=dict)
    throughput_ci_80: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Discovery statistics
    discoveries_mean: Dict[int, float] = field(default_factory=dict)
    discoveries_ci_80: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Bottleneck analysis
    bottleneck_frequency: Dict[int, Dict[str, float]] = field(default_factory=dict)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for uncertainty quantification.

    Samples from parameter distributions and runs multiple model instances
    to quantify forecast uncertainty.
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.BASELINE,
        n_samples: int = 1000,
        seed: Optional[int] = None,
        years: Optional[List[int]] = None,
    ):
        """
        Initialize the simulator.

        Args:
            scenario: AI development scenario
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            years: Years to forecast (default: 2025-2050)
        """
        self.scenario = scenario
        self.scenario_params = SCENARIO_DEFAULTS[scenario]
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.years = years or list(range(2025, 2051, 5))

        self.results: List[SimulationResult] = []
        self.summary: Optional[SimulationSummary] = None

    def sample_parameters(self) -> Dict[str, float]:
        """
        Sample model parameters from their distributions.

        Returns:
            Dictionary of sampled parameter values
        """
        params = {}

        # Sample g_ai from normal distribution
        params['g_ai'] = self.rng.normal(
            self.scenario_params.g_ai,
            self.scenario_params.g_ai_std
        )
        params['g_ai'] = max(0.05, params['g_ai'])  # Floor at 0.05

        # Sample stage parameters
        for i, stage in enumerate(DEFAULT_STAGES):
            prefix = f"stage_{stage.id}_"

            # Sample tau_0 (lognormal for positive values)
            if stage.params.tau_0_std > 0:
                params[prefix + 'tau_0'] = self.rng.lognormal(
                    np.log(stage.params.tau_0),
                    stage.params.tau_0_std / stage.params.tau_0
                )
            else:
                params[prefix + 'tau_0'] = stage.params.tau_0

            # Sample M_max_speed from range
            if stage.params.M_max_speed_range[1] > 0:
                params[prefix + 'M_max_speed'] = self.rng.uniform(
                    stage.params.M_max_speed_range[0],
                    stage.params.M_max_speed_range[1]
                )
            else:
                params[prefix + 'M_max_speed'] = stage.params.M_max_speed

            # Sample M_max_quality from range
            if stage.params.M_max_quality_range[1] > 0:
                params[prefix + 'M_max_quality'] = self.rng.uniform(
                    stage.params.M_max_quality_range[0],
                    stage.params.M_max_quality_range[1]
                )
            else:
                params[prefix + 'M_max_quality'] = stage.params.M_max_quality

        # Sample PSM weights
        for shift_type, shift_params in ParadigmShiftModule().shift_params.items():
            key = f"psm_{shift_type.value}_weight"
            params[key] = self.rng.uniform(
                shift_params.weight_ci[0],
                shift_params.weight_ci[1]
            )

        return params

    def create_model_from_params(self, params: Dict[str, float]) -> AIResearchAccelerationModel:
        """
        Create a model instance with sampled parameters.

        Args:
            params: Dictionary of sampled parameters

        Returns:
            Configured model instance
        """
        # Create modified stages
        stages = []
        for stage in DEFAULT_STAGES:
            prefix = f"stage_{stage.id}_"

            new_params = StageParameters(
                tau_0=params.get(prefix + 'tau_0', stage.params.tau_0),
                M_max_speed=params.get(prefix + 'M_max_speed', stage.params.M_max_speed),
                M_max_quality=params.get(prefix + 'M_max_quality', stage.params.M_max_quality),
                p_success=stage.params.p_success,
                k=stage.params.k,
                stage_type=stage.params.stage_type,
                reliability_2025=stage.params.reliability_2025,
                reliability_growth=stage.params.reliability_growth,
            )

            stages.append(Stage(
                id=stage.id,
                name=stage.name,
                description=stage.description,
                params=new_params,
            ))

        pipeline = ResearchPipeline(stages)

        # Create modified PSM
        from .paradigm_shift import DEFAULT_SHIFT_PARAMS, ShiftParameters
        psm_params = {}
        for shift_type, default_params in DEFAULT_SHIFT_PARAMS.items():
            key = f"psm_{shift_type.value}_weight"
            new_weight = params.get(key, default_params.weight)

            psm_params[shift_type] = ShiftParameters(
                weight=new_weight,
                weight_ci=default_params.weight_ci,
                threshold=default_params.threshold,
                growth_rate=default_params.growth_rate,
                saturation=default_params.saturation,
                stages_affected=default_params.stages_affected,
            )

        psm = ParadigmShiftModule(shift_params=psm_params)

        # Create modified scenario params
        scenario_params = ScenarioParams(
            g_ai=params['g_ai'],
            g_ai_std=self.scenario_params.g_ai_std,
            description=self.scenario_params.description,
            infrastructure_multiplier=self.scenario_params.infrastructure_multiplier,
        )

        # Create model
        model = AIResearchAccelerationModel(
            pipeline=pipeline,
            psm=psm,
            scenario=self.scenario,
        )
        model.scenario_params = scenario_params

        return model

    def run_single(self, run_id: int) -> SimulationResult:
        """
        Run a single Monte Carlo sample.

        Args:
            run_id: Identifier for this run

        Returns:
            SimulationResult for this run
        """
        # Sample parameters
        params = self.sample_parameters()

        # Create model
        model = self.create_model_from_params(params)

        # Run forecasts
        acceleration = {}
        throughput = {}
        discoveries = {}
        bottleneck = {}

        for year in self.years:
            t = year - 2025
            if t < 0:
                continue

            acceleration[year] = model.acceleration_factor(t)
            throughput[year] = model.throughput(t)
            discoveries[year] = model.validated_discoveries(t)
            bottleneck[year] = model.identify_bottleneck(t).name

        return SimulationResult(
            run_id=run_id,
            scenario=self.scenario.value,
            g_ai=params['g_ai'],
            acceleration_by_year=acceleration,
            throughput_by_year=throughput,
            discoveries_by_year=discoveries,
            bottleneck_by_year=bottleneck,
            parameters=params,
        )

    def run(self, parallel: bool = False) -> SimulationSummary:
        """
        Run the full Monte Carlo simulation.

        Args:
            parallel: Whether to use parallel processing

        Returns:
            SimulationSummary with aggregated results
        """
        self.results = []

        if parallel:
            # Note: Simplified serial for now due to pickling issues
            # Full parallel implementation would use multiprocessing
            pass

        # Serial execution
        for i in range(self.n_samples):
            result = self.run_single(i)
            self.results.append(result)

            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.n_samples} samples")

        # Compute summary statistics
        self.summary = self._compute_summary()
        return self.summary

    def _compute_summary(self) -> SimulationSummary:
        """Compute summary statistics from results."""
        summary = SimulationSummary(
            n_samples=len(self.results),
            scenario=self.scenario.value,
            years=self.years,
        )

        for year in self.years:
            # Collect values for this year
            accels = [r.acceleration_by_year.get(year, np.nan) for r in self.results]
            accels = [a for a in accels if not np.isnan(a)]

            if not accels:
                continue

            accels = np.array(accels)

            # Acceleration statistics
            summary.acceleration_mean[year] = float(np.mean(accels))
            summary.acceleration_std[year] = float(np.std(accels))
            summary.acceleration_ci_80[year] = (
                float(np.percentile(accels, 10)),
                float(np.percentile(accels, 90))
            )
            summary.acceleration_ci_95[year] = (
                float(np.percentile(accels, 2.5)),
                float(np.percentile(accels, 97.5))
            )

            # Throughput statistics
            throughputs = np.array([
                r.throughput_by_year.get(year, np.nan) for r in self.results
            ])
            throughputs = throughputs[~np.isnan(throughputs)]
            if len(throughputs) > 0:
                summary.throughput_mean[year] = float(np.mean(throughputs))
                summary.throughput_ci_80[year] = (
                    float(np.percentile(throughputs, 10)),
                    float(np.percentile(throughputs, 90))
                )

            # Discovery statistics
            discoveries = np.array([
                r.discoveries_by_year.get(year, np.nan) for r in self.results
            ])
            discoveries = discoveries[~np.isnan(discoveries)]
            if len(discoveries) > 0:
                summary.discoveries_mean[year] = float(np.mean(discoveries))
                summary.discoveries_ci_80[year] = (
                    float(np.percentile(discoveries, 10)),
                    float(np.percentile(discoveries, 90))
                )

            # Bottleneck frequency
            bottlenecks = [r.bottleneck_by_year.get(year, None) for r in self.results]
            bottlenecks = [b for b in bottlenecks if b is not None]
            if bottlenecks:
                freq = {}
                for b in bottlenecks:
                    freq[b] = freq.get(b, 0) + 1
                total = len(bottlenecks)
                summary.bottleneck_frequency[year] = {
                    k: v / total for k, v in freq.items()
                }

        return summary

    def sobol_sensitivity(self) -> Dict[str, float]:
        """
        Compute first-order Sobol sensitivity indices.

        Estimates which parameters contribute most to output variance.

        Returns:
            Dictionary mapping parameter names to sensitivity indices
        """
        if not self.results:
            raise ValueError("Must run simulation first")

        # Get parameter names
        param_names = list(self.results[0].parameters.keys())

        # Focus on 2050 acceleration as target output
        target_year = 2050
        outputs = np.array([
            r.acceleration_by_year.get(target_year, np.nan)
            for r in self.results
        ])
        outputs = outputs[~np.isnan(outputs)]

        if len(outputs) < 100:
            raise ValueError("Need at least 100 valid samples for sensitivity analysis")

        # Build parameter matrix
        param_matrix = np.array([
            [r.parameters[p] for p in param_names]
            for r in self.results
            if not np.isnan(r.acceleration_by_year.get(target_year, np.nan))
        ])

        # Compute variance-based sensitivity (simplified first-order)
        total_variance = np.var(outputs)
        if total_variance < 1e-10:
            return {p: 0.0 for p in param_names}

        sensitivities = {}
        for i, param in enumerate(param_names):
            # Correlation-based approximation of first-order index
            correlation = np.corrcoef(param_matrix[:, i], outputs)[0, 1]
            sensitivities[param] = correlation ** 2

        # Normalize to sum to ~1
        total = sum(sensitivities.values())
        if total > 0:
            sensitivities = {k: v / total for k, v in sensitivities.items()}

        return sensitivities

    def save_results(self, output_dir: str):
        """Save simulation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        if self.summary:
            summary_dict = {
                'n_samples': self.summary.n_samples,
                'scenario': self.summary.scenario,
                'years': self.summary.years,
                'acceleration_mean': self.summary.acceleration_mean,
                'acceleration_std': self.summary.acceleration_std,
                'acceleration_ci_80': {
                    str(k): list(v) for k, v in self.summary.acceleration_ci_80.items()
                },
                'acceleration_ci_95': {
                    str(k): list(v) for k, v in self.summary.acceleration_ci_95.items()
                },
                'throughput_mean': self.summary.throughput_mean,
                'discoveries_mean': self.summary.discoveries_mean,
                'bottleneck_frequency': self.summary.bottleneck_frequency,
            }

            with open(output_path / 'summary.json', 'w') as f:
                json.dump(summary_dict, f, indent=2)

        # Save raw results (subset for large simulations)
        if self.results:
            raw_results = [
                {
                    'run_id': r.run_id,
                    'g_ai': r.g_ai,
                    'acceleration_by_year': r.acceleration_by_year,
                    'bottleneck_by_year': r.bottleneck_by_year,
                }
                for r in self.results[:100]  # First 100 only
            ]

            with open(output_path / 'raw_results_sample.json', 'w') as f:
                json.dump(raw_results, f, indent=2)

    def print_summary(self):
        """Print a formatted summary of results."""
        if not self.summary:
            print("No results yet. Run simulation first.")
            return

        print(f"\n{'='*60}")
        print(f"Monte Carlo Simulation Summary")
        print(f"{'='*60}")
        print(f"Scenario: {self.summary.scenario}")
        print(f"Samples: {self.summary.n_samples}")
        print()

        print(f"{'Year':<8} {'Mean Accel':<12} {'80% CI':<20} {'Bottleneck'}")
        print(f"{'-'*60}")

        for year in self.summary.years:
            mean = self.summary.acceleration_mean.get(year, 0)
            ci = self.summary.acceleration_ci_80.get(year, (0, 0))

            # Find most common bottleneck
            freq = self.summary.bottleneck_frequency.get(year, {})
            if freq:
                bottleneck = max(freq.keys(), key=lambda k: freq[k])
                bn_pct = freq[bottleneck] * 100
            else:
                bottleneck = "N/A"
                bn_pct = 0

            print(f"{year:<8} {mean:<12.2f} [{ci[0]:.2f}, {ci[1]:.2f}]      {bottleneck} ({bn_pct:.0f}%)")


if __name__ == "__main__":
    # Quick test with small sample
    print("Running Monte Carlo simulation (n=100)...")

    sim = MonteCarloSimulator(
        scenario=Scenario.BASELINE,
        n_samples=100,
        seed=42,
        years=[2025, 2030, 2035, 2040, 2050]
    )

    summary = sim.run()
    sim.print_summary()

    print("\nTop 5 sensitive parameters:")
    sensitivities = sim.sobol_sensitivity()
    sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])
    for param, sens in sorted_sens[:5]:
        print(f"  {param}: {sens:.3f}")
