"""
Monte Carlo Uncertainty Quantification Module - v0.5

This module provides uncertainty quantification for the AI-Accelerated
Biological Discovery Model through Monte Carlo simulation.

Key Features:
- Parameter uncertainty specification (normal, uniform, triangular, lognormal)
- Monte Carlo simulation with configurable sample size
- Confidence interval computation (5th, 25th, 50th, 75th, 95th percentiles)
- Uncertainty propagation visualization
- Policy-relevant uncertainty metrics

v0.5 Changes:
- Support for multi-type AI growth rate uncertainties (g_cognitive, g_robotic, g_scientific)
- Therapeutic area-specific uncertainty propagation
- Updated Stage creation with v0.5 parameters (ai_type_weights)

References:
    - Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer"
    - Morgan & Henrion (1990) "Uncertainty: A Guide to Dealing with Uncertainty"
    - Helton & Davis (2003) "Latin hypercube sampling and the propagation of uncertainty"

Version: 0.5
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings

# Optional scipy import - for KDE in histograms
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - histograms will not show KDE curves")

# Import model components
from model import (
    AIBioAccelerationModel, ModelConfig, Stage, Scenario,
    AIType, TherapeuticArea
)


# =============================================================================
# UNCERTAINTY SPECIFICATION
# =============================================================================

@dataclass
class ParameterUncertainty:
    """
    Specification of uncertainty for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name (e.g., 'S7_M_max', 'g_cognitive')
    stage_index : int or None
        Stage index if stage-specific parameter, None for global
    parameter_type : str
        Type of parameter ('M_max', 'p_success', 'k_saturation', 'g_ai',
        'g_cognitive', 'g_robotic', 'g_scientific', 'tau_baseline')
    distribution : str
        Distribution type ('normal', 'uniform', 'triangular', 'lognormal')
    base_value : float
        Central/base value
    uncertainty_params : Dict
        Distribution-specific parameters
    """
    name: str
    stage_index: Optional[int]
    parameter_type: str
    distribution: str
    base_value: float
    uncertainty_params: Dict[str, float] = field(default_factory=dict)

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Generate random samples from the uncertainty distribution."""
        if self.distribution == 'normal':
            if 'std' in self.uncertainty_params:
                std = self.uncertainty_params['std']
            elif 'cv' in self.uncertainty_params:
                std = self.base_value * self.uncertainty_params['cv']
            else:
                std = self.base_value * 0.1

            samples = rng.normal(self.base_value, std, n_samples)

        elif self.distribution == 'uniform':
            if 'low' in self.uncertainty_params and 'high' in self.uncertainty_params:
                low = self.uncertainty_params['low']
                high = self.uncertainty_params['high']
            elif 'range_pct' in self.uncertainty_params:
                pct = self.uncertainty_params['range_pct']
                low = self.base_value * (1 - pct)
                high = self.base_value * (1 + pct)
            else:
                low = self.base_value * 0.8
                high = self.base_value * 1.2

            samples = rng.uniform(low, high, n_samples)

        elif self.distribution == 'triangular':
            if 'low' in self.uncertainty_params and 'high' in self.uncertainty_params:
                low = self.uncertainty_params['low']
                high = self.uncertainty_params['high']
            elif 'range_pct' in self.uncertainty_params:
                pct = self.uncertainty_params['range_pct']
                low = self.base_value * (1 - pct)
                high = self.base_value * (1 + pct)
            else:
                low = self.base_value * 0.8
                high = self.base_value * 1.2

            samples = rng.triangular(low, self.base_value, high, n_samples)

        elif self.distribution == 'lognormal':
            sigma = self.uncertainty_params.get('sigma', 0.2)
            mu = np.log(self.base_value) + sigma**2
            samples = rng.lognormal(mu, sigma, n_samples)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        samples = self._apply_bounds(samples)
        return samples

    def _apply_bounds(self, samples: np.ndarray) -> np.ndarray:
        """Apply physical bounds based on parameter type."""
        if self.parameter_type in ['p_success', 'p_success_max']:
            samples = np.clip(samples, 0.01, 0.99)
        elif self.parameter_type in ['M_max', 'tau_baseline']:
            samples = np.maximum(samples, 0.1)
        elif self.parameter_type == 'k_saturation':
            samples = np.clip(samples, 0.01, 2.0)
        elif self.parameter_type in ['g_ai', 'g_cognitive', 'g_robotic', 'g_scientific']:
            samples = np.maximum(samples, 0.01)

        return samples


# =============================================================================
# MONTE CARLO ENGINE
# =============================================================================

class MonteCarloUncertainty:
    """
    Monte Carlo uncertainty quantification engine.

    v0.5 Updates:
    - Support for multi-type AI growth rate uncertainties
    - Therapeutic area handling
    """

    def __init__(
        self,
        model: Optional[AIBioAccelerationModel] = None,
        n_samples: int = 1000,
        seed: int = 42
    ):
        """Initialize Monte Carlo engine."""
        self.base_model = model or AIBioAccelerationModel()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

        self.uncertainties: List[ParameterUncertainty] = []
        self.samples: Dict[str, np.ndarray] = {}
        self.mc_results: Dict[str, Dict[str, np.ndarray]] = {}

    def add_uncertainty(self, uncertainty: ParameterUncertainty) -> None:
        """Add a parameter uncertainty specification."""
        self.uncertainties.append(uncertainty)

    def add_default_uncertainties(
        self,
        scenario_name: str = 'Baseline',
        cv_M_max: float = 0.15,
        cv_p_success: float = 0.10,
        cv_p_success_max: float = 0.10,
        cv_g_cognitive: float = 0.20,
        cv_g_robotic: float = 0.25,
        cv_g_scientific: float = 0.20
    ) -> None:
        """
        Add default uncertainty specifications for key parameters.

        v0.5 Changes:
        - Added separate uncertainties for g_cognitive, g_robotic, g_scientific
        - Higher CV for robotic (0.25) due to hardware uncertainty
        """
        # Find base scenario
        scenario = None
        for s in self.base_model.config.scenarios:
            if s.name == scenario_name:
                scenario = s
                break

        if scenario is None:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        # v0.5 NEW: Multi-type AI growth rate uncertainties
        self.add_uncertainty(ParameterUncertainty(
            name='g_cognitive',
            stage_index=None,
            parameter_type='g_cognitive',
            distribution='normal',
            base_value=scenario.g_cognitive,
            uncertainty_params={'cv': cv_g_cognitive}
        ))

        self.add_uncertainty(ParameterUncertainty(
            name='g_robotic',
            stage_index=None,
            parameter_type='g_robotic',
            distribution='normal',
            base_value=scenario.g_robotic,
            uncertainty_params={'cv': cv_g_robotic}
        ))

        self.add_uncertainty(ParameterUncertainty(
            name='g_scientific',
            stage_index=None,
            parameter_type='g_scientific',
            distribution='normal',
            base_value=scenario.g_scientific,
            uncertainty_params={'cv': cv_g_scientific}
        ))

        # Add stage-specific uncertainties
        for stage in self.base_model.config.stages:
            # M_max uncertainty
            M_max = scenario.M_max_overrides.get(stage.index, stage.M_max)
            self.add_uncertainty(ParameterUncertainty(
                name=f'S{stage.index}_M_max',
                stage_index=stage.index,
                parameter_type='M_max',
                distribution='normal',
                base_value=M_max,
                uncertainty_params={'cv': cv_M_max}
            ))

            # p_success uncertainty
            self.add_uncertainty(ParameterUncertainty(
                name=f'S{stage.index}_p_success',
                stage_index=stage.index,
                parameter_type='p_success',
                distribution='normal',
                base_value=stage.p_success,
                uncertainty_params={'cv': cv_p_success}
            ))

            # p_success_max uncertainty
            if hasattr(stage, 'p_success_max') and stage.p_success_max is not None:
                self.add_uncertainty(ParameterUncertainty(
                    name=f'S{stage.index}_p_success_max',
                    stage_index=stage.index,
                    parameter_type='p_success_max',
                    distribution='normal',
                    base_value=stage.p_success_max,
                    uncertainty_params={'cv': cv_p_success_max}
                ))

    def _generate_samples(self) -> None:
        """Generate all parameter samples."""
        for unc in self.uncertainties:
            self.samples[unc.name] = unc.sample(self.n_samples, self.rng)

    def _create_modified_config(self, sample_idx: int) -> ModelConfig:
        """Create a model config with sampled parameter values."""
        config = ModelConfig(
            t0=self.base_model.config.t0,
            T=self.base_model.config.T,
            dt=self.base_model.config.dt,
            enable_dynamic_p_success=self.base_model.config.enable_dynamic_p_success,
            enable_stage_specific_g_ai=self.base_model.config.enable_stage_specific_g_ai,
            enable_ai_feedback=self.base_model.config.enable_ai_feedback,
            ai_feedback_alpha=self.base_model.config.ai_feedback_alpha,
            enable_multi_type_ai=self.base_model.config.enable_multi_type_ai,
            enable_therapeutic_areas=self.base_model.config.enable_therapeutic_areas,
        )

        # Modify stages
        modified_stages = []
        for stage in self.base_model.config.stages:
            new_stage = Stage(
                index=stage.index,
                name=stage.name,
                description=stage.description,
                tau_baseline=stage.tau_baseline,
                M_max=stage.M_max,
                p_success=stage.p_success,
                k_saturation=stage.k_saturation,
                p_success_max=stage.p_success_max,
                k_p_success=stage.k_p_success,
                g_ai_multiplier=stage.g_ai_multiplier,
                ai_type_weights=stage.ai_type_weights.copy(),
                therapeutic_sensitivity=stage.therapeutic_sensitivity,
            )

            # Apply sampled values
            M_max_key = f'S{stage.index}_M_max'
            if M_max_key in self.samples:
                new_stage.M_max = self.samples[M_max_key][sample_idx]

            p_success_key = f'S{stage.index}_p_success'
            if p_success_key in self.samples:
                new_stage.p_success = self.samples[p_success_key][sample_idx]

            p_max_key = f'S{stage.index}_p_success_max'
            if p_max_key in self.samples:
                new_stage.p_success_max = self.samples[p_max_key][sample_idx]
                new_stage.p_success_max = max(new_stage.p_success_max, new_stage.p_success + 0.01)

            modified_stages.append(new_stage)

        config.stages = modified_stages

        # Modify scenarios with sampled AI growth rates
        modified_scenarios = []
        for scenario in self.base_model.config.scenarios:
            # Get sampled growth rates
            g_c = self.samples.get('g_cognitive', [scenario.g_cognitive] * self.n_samples)[sample_idx]
            g_r = self.samples.get('g_robotic', [scenario.g_robotic] * self.n_samples)[sample_idx]
            g_s = self.samples.get('g_scientific', [scenario.g_scientific] * self.n_samples)[sample_idx]

            new_scenario = Scenario(
                name=scenario.name,
                g_ai=scenario.g_ai,
                description=scenario.description,
                M_max_overrides={},
                g_cognitive=g_c,
                g_robotic=g_r,
                g_scientific=g_s,
                therapeutic_area=scenario.therapeutic_area,
            )
            modified_scenarios.append(new_scenario)

        config.scenarios = modified_scenarios

        return config

    def run_simulation(
        self,
        scenario_name: str = 'Baseline',
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """Run Monte Carlo simulation."""
        if not self.uncertainties:
            raise ValueError("No uncertainties specified.")

        self._generate_samples()

        n_time = len(self.base_model.time_points)

        cumulative_progress = np.zeros((self.n_samples, n_time))
        progress_rate = np.zeros((self.n_samples, n_time))
        bottleneck = np.zeros((self.n_samples, n_time))

        if show_progress:
            print(f"Running Monte Carlo simulation ({self.n_samples} samples)...")

        for i in range(self.n_samples):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Sample {i + 1}/{self.n_samples}")

            config = self._create_modified_config(i)
            model = AIBioAccelerationModel(config)
            results = model.run_all_scenarios()

            df = results[results['scenario'] == scenario_name]

            cumulative_progress[i] = df['cumulative_progress'].values
            progress_rate[i] = df['progress_rate'].values
            bottleneck[i] = df['bottleneck_stage'].values

        # Store results
        self.mc_results[scenario_name] = {
            'cumulative_progress': cumulative_progress,
            'progress_rate': progress_rate,
            'bottleneck': bottleneck,
            'time_points': self.base_model.time_points,
        }

        # Extract key milestone values
        years = self.base_model.time_points
        idx_2030 = np.where(years == 2030)[0][0]
        idx_2040 = np.where(years == 2040)[0][0]
        idx_2050 = np.where(years == 2050)[0][0]

        self.mc_results[scenario_name]['progress_2030'] = cumulative_progress[:, idx_2030]
        self.mc_results[scenario_name]['progress_2040'] = cumulative_progress[:, idx_2040]
        self.mc_results[scenario_name]['progress_2050'] = cumulative_progress[:, idx_2050]

        if show_progress:
            print("  Simulation complete!")

        return self.mc_results[scenario_name]

    def compute_confidence_intervals(
        self,
        scenario_name: str = 'Baseline',
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> pd.DataFrame:
        """Compute confidence intervals for cumulative progress."""
        if scenario_name not in self.mc_results:
            raise ValueError(f"No results for scenario '{scenario_name}'.")

        results = self.mc_results[scenario_name]
        cp = results['cumulative_progress']

        ci_data = {'year': results['time_points']}

        for p in percentiles:
            ci_data[f'p{p}'] = np.percentile(cp, p, axis=0)

        return pd.DataFrame(ci_data)

    def get_summary_statistics(
        self,
        scenario_name: str = 'Baseline'
    ) -> pd.DataFrame:
        """Get summary statistics for key milestones."""
        if scenario_name not in self.mc_results:
            raise ValueError(f"No results for scenario '{scenario_name}'.")

        results = self.mc_results[scenario_name]

        milestones = {
            '2030': results['progress_2030'],
            '2040': results['progress_2040'],
            '2050': results['progress_2050'],
        }

        summaries = []
        for year, values in milestones.items():
            summaries.append({
                'year': int(year),
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values),
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                '90_ci_width': np.percentile(values, 95) - np.percentile(values, 5),
            })

        return pd.DataFrame(summaries)

    def plot_uncertainty_bands(
        self,
        scenario_name: str = 'Baseline',
        ax: Optional[plt.Axes] = None,
        show_samples: int = 0,
        figsize: Tuple[float, float] = (12, 8)
    ) -> plt.Figure:
        """Plot cumulative progress with uncertainty bands."""
        if scenario_name not in self.mc_results:
            raise ValueError(f"No results for scenario '{scenario_name}'.")

        results = self.mc_results[scenario_name]
        ci = self.compute_confidence_intervals(scenario_name)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        years = results['time_points']

        ax.fill_between(
            years, ci['p5'], ci['p95'],
            alpha=0.2, color='steelblue', label='90% CI'
        )
        ax.fill_between(
            years, ci['p25'], ci['p75'],
            alpha=0.3, color='steelblue', label='50% CI'
        )

        ax.plot(
            years, ci['p50'],
            color='steelblue', linewidth=2.5, label='Median'
        )

        if show_samples > 0:
            n_show = min(show_samples, self.n_samples)
            indices = self.rng.choice(self.n_samples, n_show, replace=False)
            for idx in indices:
                ax.plot(
                    years, results['cumulative_progress'][idx],
                    color='gray', alpha=0.1, linewidth=0.5
                )

        ax.plot(
            years, years - years[0],
            '--', color='gray', alpha=0.5, label='No acceleration'
        )

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Equivalent Years of Progress', fontsize=12)
        ax.set_title(
            f'AI-Accelerated Biological Discovery: {scenario_name} Scenario\n'
            f'Monte Carlo Uncertainty Analysis (n={self.n_samples})',
            fontsize=14
        )
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(years[0], years[-1])
        ax.set_ylim(0, None)

        summary = self.get_summary_statistics(scenario_name)
        for _, row in summary.iterrows():
            y_med = row['p50']
            y_lo = row['p5']
            y_hi = row['p95']
            ax.annotate(
                f"{row['year']}: {y_med:.0f}\n[{y_lo:.0f}, {y_hi:.0f}]",
                xy=(row['year'], y_med),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        plt.tight_layout()
        return fig

    def plot_histogram(
        self,
        scenario_name: str = 'Baseline',
        year: int = 2050,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10, 6)
    ) -> plt.Figure:
        """Plot histogram of cumulative progress for a specific year."""
        if scenario_name not in self.mc_results:
            raise ValueError(f"No results for scenario '{scenario_name}'.")

        results = self.mc_results[scenario_name]
        values = results[f'progress_{year}']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        n, bins, patches = ax.hist(
            values, bins=50, density=True,
            alpha=0.7, color='steelblue', edgecolor='white'
        )

        if HAS_SCIPY:
            kde = gaussian_kde(values)
            x_kde = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')

        percentiles = [5, 50, 95]
        colors = ['orange', 'red', 'orange']
        for p, c in zip(percentiles, colors):
            val = np.percentile(values, p)
            ax.axvline(val, color=c, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.annotate(
                f'P{p}: {val:.1f}',
                xy=(val, ax.get_ylim()[1] * 0.9),
                fontsize=10,
                ha='center'
            )

        ax.set_xlabel(f'Equivalent Years of Progress by {year}', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(
            f'Distribution of Progress by {year} ({scenario_name})\n'
            f'Mean = {np.mean(values):.1f}, Std = {np.std(values):.1f}',
            fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def export_results(self, filepath: str, scenario_name: str = 'Baseline') -> None:
        """Export Monte Carlo results to CSV."""
        if scenario_name not in self.mc_results:
            raise ValueError(f"No results for scenario '{scenario_name}'.")

        results = self.mc_results[scenario_name]

        data = {
            'sample': np.repeat(np.arange(self.n_samples), len(results['time_points'])),
            'year': np.tile(results['time_points'], self.n_samples),
            'cumulative_progress': results['cumulative_progress'].flatten(),
            'progress_rate': results['progress_rate'].flatten(),
            'bottleneck_stage': results['bottleneck'].flatten(),
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Monte Carlo results exported to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_monte_carlo_analysis(
    scenario_name: str = 'Baseline',
    n_samples: int = 1000,
    output_dir: str = 'outputs',
) -> Tuple[MonteCarloUncertainty, pd.DataFrame]:
    """Run complete Monte Carlo uncertainty analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    mc = MonteCarloUncertainty(n_samples=n_samples)

    mc.add_default_uncertainties(scenario_name=scenario_name)

    mc.run_simulation(scenario_name)

    summary = mc.get_summary_statistics(scenario_name)

    mc.export_results(
        os.path.join(output_dir, f'monte_carlo_{scenario_name.lower()}.csv'),
        scenario_name
    )

    ci = mc.compute_confidence_intervals(scenario_name)
    ci.to_csv(
        os.path.join(output_dir, f'confidence_intervals_{scenario_name.lower()}.csv'),
        index=False
    )

    fig1 = mc.plot_uncertainty_bands(scenario_name)
    fig1.savefig(
        os.path.join(output_dir, f'fig_uncertainty_bands_{scenario_name.lower()}.png'),
        dpi=300, bbox_inches='tight'
    )
    fig1.savefig(
        os.path.join(output_dir, f'fig_uncertainty_bands_{scenario_name.lower()}.pdf'),
        bbox_inches='tight'
    )
    plt.close(fig1)

    fig2 = mc.plot_histogram(scenario_name, year=2050)
    fig2.savefig(
        os.path.join(output_dir, f'fig_histogram_2050_{scenario_name.lower()}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig2)

    print("\n" + "=" * 60)
    print(f"Monte Carlo Summary: {scenario_name} Scenario")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("\n90% Confidence Intervals:")
    for _, row in summary.iterrows():
        print(f"  {row['year']}: {row['p50']:.1f} [{row['p5']:.1f}, {row['p95']:.1f}]")

    return mc, summary


if __name__ == "__main__":
    print("=" * 70)
    print("Monte Carlo Uncertainty Analysis - v0.5")
    print("=" * 70)

    mc, summary = run_monte_carlo_analysis(
        scenario_name='Baseline',
        n_samples=500,
        output_dir='outputs'
    )

    print("\nAnalysis complete!")
