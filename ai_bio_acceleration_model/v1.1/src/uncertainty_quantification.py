"""
Uncertainty Quantification Module - v1.1
========================================

Expert Review Fixes:
- P1-2: Calibrated g_ai distribution to Epoch AI empirical data
        Increased uncertainty from σ=0.25 to σ=0.50 (doubled)
- P2-11: Bootstrap confidence intervals on Sobol indices
- P2-16: Expanded QALY range to $50K-200K

IMPORTANT METHODOLOGY NOTE:
- These distributions were AI-simulated, not empirically validated
- Sobol indices labeled as "approximate" per Expert A1
- For rigorous use, calibrate against actual forecaster surveys

References:
- Epoch AI (2024) "Compute Trends Across Training Runs"
- AI Impacts Survey (2022) "Expert predictions on AI timelines"
- Saltelli et al. (2010) "Variance based sensitivity analysis"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42


class DistributionType(Enum):
    """Types of probability distributions for parameters."""
    LOGNORMAL = "lognormal"
    BETA = "beta"
    UNIFORM = "uniform"
    NORMAL = "normal"
    TRIANGULAR = "triangular"


@dataclass
class ParameterDistribution:
    """
    Defines the probability distribution for a model parameter.

    P1-2: Distributions calibrated to expert survey data where available.
    """
    name: str
    distribution: DistributionType
    params: Dict[str, float]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    description: str = ""

    # P1-2: Data source for calibration
    calibration_source: str = "expert judgment"
    calibration_year: int = 2024

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n samples from the distribution."""
        if self.distribution == DistributionType.LOGNORMAL:
            mu = self.params.get('mu', 0)
            sigma = self.params.get('sigma', 0.2)
            samples = rng.lognormal(mu, sigma, n)

        elif self.distribution == DistributionType.BETA:
            alpha = self.params.get('alpha', 2)
            beta = self.params.get('beta', 2)
            samples = rng.beta(alpha, beta, n)

        elif self.distribution == DistributionType.UNIFORM:
            low = self.params.get('low', 0)
            high = self.params.get('high', 1)
            samples = rng.uniform(low, high, n)

        elif self.distribution == DistributionType.NORMAL:
            mean = self.params.get('mean', 0)
            std = self.params.get('std', 1)
            samples = rng.normal(mean, std, n)

        elif self.distribution == DistributionType.TRIANGULAR:
            low = self.params.get('low', 0)
            mode = self.params.get('mode', 0.5)
            high = self.params.get('high', 1)
            samples = rng.triangular(low, mode, high, n)

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution}")

        # Apply bounds
        if self.lower_bound is not None:
            samples = np.maximum(samples, self.lower_bound)
        if self.upper_bound is not None:
            samples = np.minimum(samples, self.upper_bound)

        return samples


@dataclass
class ApproximateSobolIndices:
    """
    P1-1: Explicitly labeled as APPROXIMATE Sobol indices.

    These are computed using correlation-based approximation, NOT the
    full Saltelli method which requires N×(2k+2) evaluations.

    For rigorous analysis, use SALib or similar with full Saltelli sampling.
    """
    parameter_names: List[str]
    first_order: Dict[str, float]  # S_i: main effect (APPROXIMATE)
    total_order: Dict[str, float]  # S_Ti: total effect (APPROXIMATE)

    # P2-11: Bootstrap confidence intervals
    first_order_ci: Optional[Dict[str, Tuple[float, float]]] = None
    total_order_ci: Optional[Dict[str, Tuple[float, float]]] = None

    # Metadata
    n_samples: int = 0
    is_approximate: bool = True  # P1-1: Explicit flag
    approximation_method: str = "correlation-based"

    def get_ranking(self, use_total: bool = True) -> List[Tuple[str, float]]:
        """Return parameters ranked by sensitivity."""
        indices = self.total_order if use_total else self.first_order
        return sorted(indices.items(), key=lambda x: x[1], reverse=True)


@dataclass
class UncertaintyResults:
    """Complete results from uncertainty quantification."""
    n_samples: int
    output_samples: Dict[str, np.ndarray]

    # Summary statistics
    means: Dict[str, float]
    medians: Dict[str, float]
    std_devs: Dict[str, float]

    # Confidence intervals
    ci_80: Dict[str, Tuple[float, float]]
    ci_90: Dict[str, Tuple[float, float]]
    ci_95: Dict[str, Tuple[float, float]]

    # P1-1: Explicitly approximate Sobol
    sobol: Optional[ApproximateSobolIndices] = None

    # Convergence diagnostics
    convergence_cv: Optional[Dict[str, float]] = None

    # Parameter correlations with outputs
    correlations: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class UQConfig:
    """Configuration for uncertainty quantification."""
    n_samples: int = 10000
    n_bootstrap: int = 1000  # P2-11: Bootstrap samples for CIs
    seed: int = RANDOM_SEED
    compute_sobol: bool = True
    compute_correlations: bool = True
    ci_levels: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    convergence_check: bool = True


# =============================================================================
# MAIN UQ CLASS
# =============================================================================

class UncertaintyQuantification:
    """
    Comprehensive uncertainty quantification for the AI acceleration model.

    v1.1 Changes:
    - P1-2: Calibrated g_ai distribution (σ increased to 0.50)
    - P2-11: Bootstrap CIs on Sobol indices
    - P2-16: Expanded QALY range ($50K-200K)
    - P1-1: All Sobol indices labeled as APPROXIMATE
    """

    def __init__(self, config: UQConfig = None):
        self.config = config or UQConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.parameter_distributions: Dict[str, ParameterDistribution] = {}
        self._setup_calibrated_distributions()

    def _setup_calibrated_distributions(self):
        """
        Set up parameter distributions calibrated to empirical data.

        P1-2: Key changes:
        - g_ai σ increased from 0.25 to 0.50 (doubled uncertainty)
        - Based on AI Impacts survey showing 50%+ forecaster disagreement
        """

        # =================================================================
        # P1-2: AI GROWTH RATE - CALIBRATED TO EPOCH AI DATA
        # =================================================================
        # Epoch AI "Compute Trends" shows:
        # - Historical compute growth: ~4x/year (2012-2022)
        # - Recent scaling: showing diminishing returns at frontier
        # - Forecaster disagreement: 10-90% range spans decade+
        #
        # AI Impacts 2022 Survey:
        # - Median TAI: 2040-2060 (huge uncertainty)
        # - 10th percentile: ~2030
        # - 90th percentile: >2100
        #
        # We DOUBLE the uncertainty from v1.0:
        # σ = 0.25 → σ = 0.50
        # =================================================================
        self.add_parameter(ParameterDistribution(
            name="g_ai",
            distribution=DistributionType.LOGNORMAL,
            params={
                'mu': np.log(0.50),  # Median = 0.50
                'sigma': 0.50,       # P1-2: DOUBLED from 0.25
            },
            lower_bound=0.15,        # P1-2: Expanded lower bound
            upper_bound=1.0,         # P1-2: Expanded upper bound
            description="AI capability growth rate (per year)",
            calibration_source="Epoch AI + AI Impacts Survey",
            calibration_year=2024,
        ))

        # =================================================================
        # M_max PARAMETERS
        # =================================================================

        # Cognitive stages (S1, S2, S4)
        self.add_parameter(ParameterDistribution(
            name="M_max_cognitive",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(30), 'sigma': 0.4},
            lower_bound=10,
            upper_bound=100,
            description="Max AI multiplier for cognitive stages",
            calibration_source="AlphaFold analogy (adjusted)",
        ))

        # Physical stages (S3, S5) - P1-4: Reduced
        self.add_parameter(ParameterDistribution(
            name="M_max_physical",
            distribution=DistributionType.LOGNORMAL,
            params={
                'mu': np.log(2.5),   # P1-4: Reduced from 4
                'sigma': 0.25,
            },
            lower_bound=1.5,
            upper_bound=5.0,         # P1-4: Reduced from 10
            description="Max AI multiplier for wet lab (biological limits)",
            calibration_source="Expert C3 (robotics)",
        ))

        # Clinical stages (S6, S7, S8) - P2-12: Disease-specific range
        self.add_parameter(ParameterDistribution(
            name="M_max_clinical",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(3), 'sigma': 0.30},
            lower_bound=1.5,
            upper_bound=5.0,         # P2-12: Upper expanded for oncology
            description="Max AI multiplier for clinical stages (disease-specific)",
            calibration_source="FDA adaptive trial guidance + Expert B1",
        ))

        # Regulatory (S9) - constrained by PDUFA
        self.add_parameter(ParameterDistribution(
            name="M_max_regulatory",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(1.8), 'sigma': 0.15},
            lower_bound=1.2,
            upper_bound=2.5,         # P1-5: Capped by 6-month floor
            description="Max AI multiplier for regulatory (PDUFA floor)",
            calibration_source="Expert B3 (FDA)",
        ))

        # =================================================================
        # SUCCESS PROBABILITIES (Beta distributions)
        # =================================================================

        # Phase II baseline - Wong et al. (2019)
        self.add_parameter(ParameterDistribution(
            name="p_phase2_base",
            distribution=DistributionType.BETA,
            params={'alpha': 5, 'beta': 12},  # Mean ~0.29
            description="Phase II baseline success probability",
            calibration_source="Wong et al. (2019)",
            calibration_year=2019,
        ))

        # Phase II max with AI
        self.add_parameter(ParameterDistribution(
            name="p_phase2_max",
            distribution=DistributionType.BETA,
            params={'alpha': 8, 'beta': 8},  # Mean ~0.50
            lower_bound=0.35,
            upper_bound=0.65,
            description="Phase II max success probability with AI",
            calibration_source="Expert B2 estimate",
        ))

        # S1 Hypothesis success - P1-18: Reduced
        self.add_parameter(ParameterDistribution(
            name="p_hypothesis_base",
            distribution=DistributionType.BETA,
            params={'alpha': 4, 'beta': 6},  # Mean ~0.40
            lower_bound=0.20,
            upper_bound=0.60,
            description="Hypothesis success rate (most fail to translate)",
            calibration_source="Expert B2 - 90%+ hypotheses fail",
        ))

        # =================================================================
        # OTHER PARAMETERS
        # =================================================================

        # Saturation rate k
        self.add_parameter(ParameterDistribution(
            name="k_saturation",
            distribution=DistributionType.UNIFORM,
            params={'low': 0.2, 'high': 1.0},
            description="Saturation rate for AI acceleration",
        ))

        # Data quality growth
        self.add_parameter(ParameterDistribution(
            name="gamma_data",
            distribution=DistributionType.TRIANGULAR,
            params={'low': 0.04, 'mode': 0.08, 'high': 0.15},
            description="Data quality growth rate",
        ))

        # AI feedback strength
        self.add_parameter(ParameterDistribution(
            name="alpha_feedback",
            distribution=DistributionType.UNIFORM,
            params={'low': 0.05, 'high': 0.20},
            description="AI recursive improvement strength",
        ))

        # P1-8: Global access factor
        self.add_parameter(ParameterDistribution(
            name="global_access_factor",
            distribution=DistributionType.TRIANGULAR,
            params={'low': 0.2, 'mode': 0.5, 'high': 0.8},
            description="LMIC population access to new therapies",
            calibration_source="Expert D3 (global health)",
        ))

        # P1-7: AI winter probability
        self.add_parameter(ParameterDistribution(
            name="ai_winter_probability",
            distribution=DistributionType.BETA,
            params={'alpha': 3, 'beta': 17},  # Mean ~0.15
            lower_bound=0.05,
            upper_bound=0.30,
            description="Probability of AI progress plateau",
            calibration_source="Expert C2 (AI safety)",
        ))

        # =================================================================
        # P2-16: EXPANDED QALY RANGE ($50K - $200K)
        # =================================================================
        self.add_parameter(ParameterDistribution(
            name="qaly_per_cure",
            distribution=DistributionType.TRIANGULAR,
            params={'low': 2.0, 'mode': 5.0, 'high': 10.0},
            description="Average QALYs gained per cure",
        ))

        self.add_parameter(ParameterDistribution(
            name="value_per_qaly",
            distribution=DistributionType.LOGNORMAL,
            params={
                'mu': np.log(100000),
                'sigma': 0.35,       # P2-16: Wider range
            },
            lower_bound=50000,       # P2-16: NICE low estimate
            upper_bound=200000,      # P2-16: US willingness-to-pay high
            description="Economic value per QALY ($) - geographic range",
            calibration_source="NICE/ICER guidelines",
        ))

    def add_parameter(self, param_dist: ParameterDistribution):
        """Add a parameter distribution."""
        self.parameter_distributions[param_dist.name] = param_dist

    def sample_parameters(self, n: int = None) -> Dict[str, np.ndarray]:
        """Sample all parameters."""
        n = n or self.config.n_samples
        return {
            name: dist.sample(n, self.rng)
            for name, dist in self.parameter_distributions.items()
        }

    def compute_approximate_sobol(
        self,
        param_samples: Dict[str, np.ndarray],
        output_samples: np.ndarray,
        output_name: str = "progress_2050"
    ) -> ApproximateSobolIndices:
        """
        P1-1: Compute APPROXIMATE Sobol indices using correlation method.

        WARNING: This is NOT the full Saltelli method. For rigorous analysis,
        use SALib with N×(2k+2) model evaluations.

        Method: S_i ≈ r²(X_i, Y) where r is Pearson correlation
        """
        n = len(output_samples)
        param_names = list(param_samples.keys())

        # Compute correlations
        first_order = {}
        total_order = {}
        first_order_ci = {}
        total_order_ci = {}

        total_var = np.var(output_samples)
        if total_var < 1e-10:
            warnings.warn("Near-zero output variance")
            return ApproximateSobolIndices(
                parameter_names=param_names,
                first_order={p: 0.0 for p in param_names},
                total_order={p: 0.0 for p in param_names},
                n_samples=n,
                is_approximate=True,
            )

        for name in param_names:
            x = param_samples[name][:n]
            y = output_samples

            # Correlation-based approximation
            corr = np.corrcoef(x, y)[0, 1]
            r_squared = corr ** 2

            # Approximate S_i as r² (main effect)
            first_order[name] = r_squared

            # Approximate S_Ti (assume weak interactions)
            # S_Ti ≈ S_i + small interaction term
            total_order[name] = min(r_squared * 1.05, 1.0)

            # P2-11: Bootstrap confidence intervals
            if self.config.n_bootstrap > 0:
                s_i_boot = []
                s_ti_boot = []

                for _ in range(self.config.n_bootstrap):
                    idx = self.rng.choice(n, size=n, replace=True)
                    x_b = x[idx]
                    y_b = y[idx]
                    corr_b = np.corrcoef(x_b, y_b)[0, 1]
                    r2_b = corr_b ** 2
                    s_i_boot.append(r2_b)
                    s_ti_boot.append(min(r2_b * 1.05, 1.0))

                first_order_ci[name] = (
                    np.percentile(s_i_boot, 5),
                    np.percentile(s_i_boot, 95)
                )
                total_order_ci[name] = (
                    np.percentile(s_ti_boot, 5),
                    np.percentile(s_ti_boot, 95)
                )

        # Normalize if sum > 1
        total = sum(first_order.values())
        if total > 1.1:
            first_order = {k: v/total for k, v in first_order.items()}

        return ApproximateSobolIndices(
            parameter_names=param_names,
            first_order=first_order,
            total_order=total_order,
            first_order_ci=first_order_ci if first_order_ci else None,
            total_order_ci=total_order_ci if total_order_ci else None,
            n_samples=n,
            is_approximate=True,
            approximation_method="correlation-based (r² proxy)",
        )

    def run_monte_carlo(
        self,
        model_func: Callable[[Dict[str, float]], Dict[str, float]],
        output_names: List[str] = None,
    ) -> UncertaintyResults:
        """
        Run Monte Carlo simulation with uncertainty quantification.
        """
        n = self.config.n_samples
        param_samples = self.sample_parameters(n)

        # Run model for each sample
        all_outputs = []
        for i in range(n):
            params = {name: samples[i] for name, samples in param_samples.items()}
            try:
                output = model_func(params)
                all_outputs.append(output)
            except Exception as e:
                if i < 10:  # Only warn for first few
                    warnings.warn(f"Model eval failed sample {i}: {e}")
                continue

        if len(all_outputs) < 100:
            raise ValueError(f"Too few valid samples: {len(all_outputs)}")

        # Organize outputs
        if output_names is None:
            output_names = list(all_outputs[0].keys())

        output_samples = {
            name: np.array([o.get(name, np.nan) for o in all_outputs])
            for name in output_names
        }

        # Compute statistics
        means = {name: np.nanmean(s) for name, s in output_samples.items()}
        medians = {name: np.nanmedian(s) for name, s in output_samples.items()}
        std_devs = {name: np.nanstd(s) for name, s in output_samples.items()}

        ci_80 = {
            name: (np.nanpercentile(s, 10), np.nanpercentile(s, 90))
            for name, s in output_samples.items()
        }
        ci_90 = {
            name: (np.nanpercentile(s, 5), np.nanpercentile(s, 95))
            for name, s in output_samples.items()
        }
        ci_95 = {
            name: (np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5))
            for name, s in output_samples.items()
        }

        # Convergence check
        convergence_cv = {}
        for name, samples in output_samples.items():
            valid = samples[~np.isnan(samples)]
            if len(valid) > 100:
                running_mean = np.cumsum(valid) / np.arange(1, len(valid) + 1)
                last_10_pct = running_mean[int(0.9 * len(valid)):]
                cv = np.std(last_10_pct) / (np.mean(last_10_pct) + 1e-10)
                convergence_cv[name] = cv

        # Correlations
        correlations = {}
        for out_name, out_vals in output_samples.items():
            correlations[out_name] = {}
            valid_mask = ~np.isnan(out_vals)
            for param_name, param_vals in param_samples.items():
                min_len = min(len(out_vals), len(param_vals))
                mask = valid_mask[:min_len]
                if mask.sum() > 10:
                    corr = np.corrcoef(
                        param_vals[:min_len][mask],
                        out_vals[:min_len][mask]
                    )[0, 1]
                    correlations[out_name][param_name] = corr

        # Approximate Sobol (P1-1: labeled as approximate)
        sobol = None
        if self.config.compute_sobol and 'progress_2050' in output_samples:
            valid_output = output_samples['progress_2050']
            valid_mask = ~np.isnan(valid_output)
            valid_params = {
                k: v[:len(valid_output)][valid_mask]
                for k, v in param_samples.items()
            }
            sobol = self.compute_approximate_sobol(
                valid_params,
                valid_output[valid_mask],
            )

        return UncertaintyResults(
            n_samples=len(all_outputs),
            output_samples=output_samples,
            means=means,
            medians=medians,
            std_devs=std_devs,
            ci_80=ci_80,
            ci_90=ci_90,
            ci_95=ci_95,
            sobol=sobol,
            convergence_cv=convergence_cv,
            correlations=correlations,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_sobol_table(sobol: ApproximateSobolIndices, top_n: int = 10) -> str:
    """Format Sobol indices as markdown table with CIs."""
    lines = [
        "## APPROXIMATE Sobol Sensitivity Indices",
        "",
        "**WARNING**: These are correlation-based approximations, NOT full Saltelli.",
        f"Method: {sobol.approximation_method}",
        f"Samples: {sobol.n_samples:,}",
        "",
        "| Parameter | S_i (First) | 90% CI | S_Ti (Total) | 90% CI |",
        "|-----------|-------------|--------|--------------|--------|",
    ]

    ranking = sobol.get_ranking(use_total=True)
    for name, s_ti in ranking[:top_n]:
        s_i = sobol.first_order.get(name, 0)

        s_i_ci_str = ""
        s_ti_ci_str = ""
        if sobol.first_order_ci and name in sobol.first_order_ci:
            ci = sobol.first_order_ci[name]
            s_i_ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        if sobol.total_order_ci and name in sobol.total_order_ci:
            ci = sobol.total_order_ci[name]
            s_ti_ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"

        lines.append(f"| {name} | {s_i:.3f} | {s_i_ci_str} | {s_ti:.3f} | {s_ti_ci_str} |")

    return "\n".join(lines)


def create_default_uq() -> UncertaintyQuantification:
    """Create default UQ instance with v1.1 calibrations."""
    return UncertaintyQuantification(UQConfig(
        n_samples=10000,
        n_bootstrap=1000,
        seed=RANDOM_SEED,
    ))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Uncertainty Quantification Module - v1.1")
    print("=" * 60)
    print("\nKey calibrations:")

    uq = create_default_uq()

    for name, dist in uq.parameter_distributions.items():
        samples = dist.sample(1000, uq.rng)
        print(f"  {name}:")
        print(f"    Distribution: {dist.distribution.value}")
        print(f"    Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")
        print(f"    Source: {dist.calibration_source}")

    print("\nModule loaded successfully.")
