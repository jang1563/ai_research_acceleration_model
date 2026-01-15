"""
Uncertainty Quantification Module for v1.0
==========================================

Provides comprehensive uncertainty analysis including:
- Parameter distributions (LogNormal, Beta, Uniform)
- Monte Carlo simulation with N=10,000 samples
- Sobol sensitivity indices (first-order and total-order)
- 80% confidence intervals on all outputs
- Correlation analysis between parameters
- Convergence diagnostics

References:
- Saltelli et al. (2010). "Variance based sensitivity analysis"
- Sobol (2001). "Global sensitivity indices"
- Helton & Davis (2003). "Latin hypercube sampling"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings


class DistributionType(Enum):
    """Types of probability distributions for parameters."""
    LOGNORMAL = "lognormal"      # For M_max, g_ai (positive, right-skewed)
    BETA = "beta"                # For p_success (bounded [0,1])
    UNIFORM = "uniform"          # For k_saturation (bounded range)
    NORMAL = "normal"            # For general parameters
    TRIANGULAR = "triangular"    # For expert estimates with mode


@dataclass
class ParameterDistribution:
    """Defines the probability distribution for a model parameter."""
    name: str
    distribution: DistributionType

    # Distribution parameters (interpretation depends on distribution type)
    # LogNormal: mu, sigma (of log)
    # Beta: alpha, beta
    # Uniform: low, high
    # Normal: mean, std
    # Triangular: low, mode, high
    params: Dict[str, float]

    # Bounds for sampling (optional)
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Description for documentation
    description: str = ""

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
class SobolIndices:
    """Results from Sobol sensitivity analysis."""
    parameter_names: List[str]
    first_order: Dict[str, float]       # S_i: main effect
    total_order: Dict[str, float]       # S_Ti: total effect including interactions
    second_order: Optional[Dict[Tuple[str, str], float]] = None  # S_ij: pairwise
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

    def get_ranking(self, use_total: bool = True) -> List[Tuple[str, float]]:
        """Return parameters ranked by sensitivity."""
        indices = self.total_order if use_total else self.first_order
        return sorted(indices.items(), key=lambda x: x[1], reverse=True)


@dataclass
class UncertaintyResults:
    """Complete results from uncertainty quantification."""
    # Monte Carlo results
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

    # Sobol indices
    sobol: Optional[SobolIndices] = None

    # Convergence diagnostics
    convergence_metrics: Optional[Dict[str, float]] = None

    # Parameter correlations with outputs
    correlations: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class UQConfig:
    """Configuration for uncertainty quantification."""
    n_samples: int = 10000              # Number of Monte Carlo samples
    n_sobol_samples: int = 2048         # Must be power of 2 for Sobol
    seed: int = 42                      # Random seed for reproducibility
    compute_sobol: bool = True          # Whether to compute Sobol indices
    compute_correlations: bool = True   # Whether to compute correlations
    ci_levels: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    convergence_check: bool = True      # Check Monte Carlo convergence


class UncertaintyQuantification:
    """
    Comprehensive uncertainty quantification for the AI acceleration model.

    Implements:
    1. Monte Carlo simulation with Latin Hypercube Sampling
    2. Sobol sensitivity indices
    3. Confidence interval estimation
    4. Convergence diagnostics
    """

    def __init__(self, config: UQConfig = None):
        self.config = config or UQConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.parameter_distributions: Dict[str, ParameterDistribution] = {}
        self._setup_default_distributions()

    def _setup_default_distributions(self):
        """Set up default parameter distributions based on literature."""

        # AI growth rate g_ai
        # LogNormal with mean ~0.5, CV ~30%
        self.add_parameter(ParameterDistribution(
            name="g_ai",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(0.5), 'sigma': 0.25},
            lower_bound=0.2,
            upper_bound=0.9,
            description="AI capability growth rate (per year)"
        ))

        # M_max for different stage types
        # Cognitive stages (S1, S2, S4): High potential
        self.add_parameter(ParameterDistribution(
            name="M_max_cognitive",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(30), 'sigma': 0.4},
            lower_bound=10,
            upper_bound=100,
            description="Max AI multiplier for cognitive stages"
        ))

        # Physical stages (S3, S5): Limited potential
        self.add_parameter(ParameterDistribution(
            name="M_max_physical",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(4), 'sigma': 0.3},
            lower_bound=2,
            upper_bound=10,
            description="Max AI multiplier for physical stages"
        ))

        # Clinical stages (S6, S7, S8): Moderate potential
        self.add_parameter(ParameterDistribution(
            name="M_max_clinical",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(3), 'sigma': 0.25},
            lower_bound=1.5,
            upper_bound=6,
            description="Max AI multiplier for clinical stages"
        ))

        # Regulatory stage (S9): Low potential
        self.add_parameter(ParameterDistribution(
            name="M_max_regulatory",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(2), 'sigma': 0.2},
            lower_bound=1.2,
            upper_bound=4,
            description="Max AI multiplier for regulatory stage"
        ))

        # Success probabilities (Beta distributions)
        # Phase II baseline success
        self.add_parameter(ParameterDistribution(
            name="p_phase2_base",
            distribution=DistributionType.BETA,
            params={'alpha': 5, 'beta': 12},  # Mean ~0.29, matches Wong et al.
            description="Phase II baseline success probability"
        ))

        # Phase II max success with AI
        self.add_parameter(ParameterDistribution(
            name="p_phase2_max",
            distribution=DistributionType.BETA,
            params={'alpha': 8, 'beta': 8},  # Mean ~0.5, higher ceiling
            lower_bound=0.35,
            upper_bound=0.65,
            description="Phase II max success probability with AI"
        ))

        # Saturation rate k
        self.add_parameter(ParameterDistribution(
            name="k_saturation",
            distribution=DistributionType.UNIFORM,
            params={'low': 0.2, 'high': 1.0},
            description="Saturation rate for AI acceleration"
        ))

        # Data quality growth rate gamma
        self.add_parameter(ParameterDistribution(
            name="gamma_data",
            distribution=DistributionType.TRIANGULAR,
            params={'low': 0.04, 'mode': 0.08, 'high': 0.15},
            description="Data quality growth rate"
        ))

        # AI feedback strength alpha
        self.add_parameter(ParameterDistribution(
            name="alpha_feedback",
            distribution=DistributionType.UNIFORM,
            params={'low': 0.05, 'high': 0.20},
            description="AI recursive improvement strength"
        ))

        # QALY per cure (for policy analysis)
        self.add_parameter(ParameterDistribution(
            name="qaly_per_cure",
            distribution=DistributionType.TRIANGULAR,
            params={'low': 2.0, 'mode': 4.0, 'high': 8.0},
            description="Average QALYs gained per cure"
        ))

        # Value per QALY
        self.add_parameter(ParameterDistribution(
            name="value_per_qaly",
            distribution=DistributionType.LOGNORMAL,
            params={'mu': np.log(100000), 'sigma': 0.3},
            lower_bound=50000,
            upper_bound=200000,
            description="Economic value per QALY ($)"
        ))

    def add_parameter(self, param_dist: ParameterDistribution):
        """Add a parameter distribution."""
        self.parameter_distributions[param_dist.name] = param_dist

    def get_parameter(self, name: str) -> ParameterDistribution:
        """Get a parameter distribution by name."""
        return self.parameter_distributions.get(name)

    def sample_parameters(self, n: int = None) -> Dict[str, np.ndarray]:
        """
        Sample all parameters using Latin Hypercube Sampling.

        LHS ensures better coverage of parameter space than random sampling.
        """
        n = n or self.config.n_samples
        samples = {}

        for name, param_dist in self.parameter_distributions.items():
            # Use standard sampling (could enhance with LHS)
            samples[name] = param_dist.sample(n, self.rng)

        return samples

    def latin_hypercube_sample(self, n: int = None) -> Dict[str, np.ndarray]:
        """
        Generate Latin Hypercube samples for all parameters.

        Ensures even coverage across the parameter space.
        """
        n = n or self.config.n_samples
        n_params = len(self.parameter_distributions)

        # Generate base LHS in [0, 1]^d
        lhs_base = np.zeros((n, n_params))
        for j in range(n_params):
            perm = self.rng.permutation(n)
            for i in range(n):
                lhs_base[i, j] = (perm[i] + self.rng.random()) / n

        # Transform to actual distributions
        samples = {}
        for j, (name, param_dist) in enumerate(self.parameter_distributions.items()):
            u = lhs_base[:, j]

            # Transform uniform [0,1] to target distribution
            if param_dist.distribution == DistributionType.LOGNORMAL:
                mu = param_dist.params.get('mu', 0)
                sigma = param_dist.params.get('sigma', 0.2)
                from scipy import stats
                samples[name] = stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))

            elif param_dist.distribution == DistributionType.BETA:
                alpha = param_dist.params.get('alpha', 2)
                beta = param_dist.params.get('beta', 2)
                from scipy import stats
                samples[name] = stats.beta.ppf(u, alpha, beta)

            elif param_dist.distribution == DistributionType.UNIFORM:
                low = param_dist.params.get('low', 0)
                high = param_dist.params.get('high', 1)
                samples[name] = low + u * (high - low)

            elif param_dist.distribution == DistributionType.NORMAL:
                mean = param_dist.params.get('mean', 0)
                std = param_dist.params.get('std', 1)
                from scipy import stats
                samples[name] = stats.norm.ppf(u, loc=mean, scale=std)

            elif param_dist.distribution == DistributionType.TRIANGULAR:
                low = param_dist.params.get('low', 0)
                mode = param_dist.params.get('mode', 0.5)
                high = param_dist.params.get('high', 1)
                c = (mode - low) / (high - low)
                from scipy import stats
                samples[name] = stats.triang.ppf(u, c, loc=low, scale=high-low)

            # Apply bounds
            if param_dist.lower_bound is not None:
                samples[name] = np.maximum(samples[name], param_dist.lower_bound)
            if param_dist.upper_bound is not None:
                samples[name] = np.minimum(samples[name], param_dist.upper_bound)

        return samples

    def compute_sobol_indices(
        self,
        model_func: Callable[[Dict[str, float]], float],
        param_names: List[str] = None
    ) -> SobolIndices:
        """
        Compute Sobol sensitivity indices using Saltelli's method.

        This decomposes output variance into contributions from each parameter
        and their interactions.

        S_i = V[E[Y|X_i]] / V[Y]  (first-order)
        S_Ti = E[V[Y|X_~i]] / V[Y]  (total-order)

        References:
        - Saltelli et al. (2010). "Variance based sensitivity analysis"
        """
        param_names = param_names or list(self.parameter_distributions.keys())
        n_params = len(param_names)
        n = self.config.n_sobol_samples

        # Generate base matrices A and B
        A = np.zeros((n, n_params))
        B = np.zeros((n, n_params))

        for j, name in enumerate(param_names):
            param_dist = self.parameter_distributions[name]
            A[:, j] = param_dist.sample(n, self.rng)
            B[:, j] = param_dist.sample(n, self.rng)

        # Compute model outputs for A and B
        def evaluate_matrix(X):
            outputs = []
            for i in range(X.shape[0]):
                params = {name: X[i, j] for j, name in enumerate(param_names)}
                outputs.append(model_func(params))
            return np.array(outputs)

        f_A = evaluate_matrix(A)
        f_B = evaluate_matrix(B)

        # Total variance
        f_all = np.concatenate([f_A, f_B])
        total_variance = np.var(f_all)

        if total_variance < 1e-10:
            warnings.warn("Near-zero variance in model output")
            return SobolIndices(
                parameter_names=param_names,
                first_order={name: 0.0 for name in param_names},
                total_order={name: 0.0 for name in param_names}
            )

        first_order = {}
        total_order = {}

        # Compute indices for each parameter
        for j, name in enumerate(param_names):
            # Create AB_j matrix (A with j-th column from B)
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            f_AB_j = evaluate_matrix(AB_j)

            # First-order index (Jansen estimator)
            V_j = np.mean(f_B * (f_AB_j - f_A))
            first_order[name] = max(0, V_j / total_variance)

            # Total-order index
            V_Tj = 0.5 * np.mean((f_A - f_AB_j) ** 2)
            total_order[name] = max(0, V_Tj / total_variance)

        # Normalize if sum exceeds 1 (numerical issues)
        s1_sum = sum(first_order.values())
        if s1_sum > 1.1:
            first_order = {k: v/s1_sum for k, v in first_order.items()}

        return SobolIndices(
            parameter_names=param_names,
            first_order=first_order,
            total_order=total_order
        )

    def run_monte_carlo(
        self,
        model_func: Callable[[Dict[str, float]], Dict[str, float]],
        output_names: List[str] = None,
        use_lhs: bool = True
    ) -> UncertaintyResults:
        """
        Run Monte Carlo simulation with uncertainty quantification.

        Args:
            model_func: Function that takes parameter dict and returns output dict
            output_names: Names of outputs to track (if None, track all)
            use_lhs: Whether to use Latin Hypercube Sampling

        Returns:
            UncertaintyResults with full statistics
        """
        n = self.config.n_samples

        # Sample parameters
        if use_lhs:
            try:
                param_samples = self.latin_hypercube_sample(n)
            except ImportError:
                warnings.warn("scipy not available, using random sampling")
                param_samples = self.sample_parameters(n)
        else:
            param_samples = self.sample_parameters(n)

        # Run model for each sample
        all_outputs = []
        for i in range(n):
            params = {name: samples[i] for name, samples in param_samples.items()}
            try:
                output = model_func(params)
                all_outputs.append(output)
            except Exception as e:
                warnings.warn(f"Model evaluation failed for sample {i}: {e}")
                continue

        if len(all_outputs) == 0:
            raise ValueError("All model evaluations failed")

        # Organize outputs
        if output_names is None:
            output_names = list(all_outputs[0].keys())

        output_samples = {name: np.array([o[name] for o in all_outputs])
                         for name in output_names}

        # Compute statistics
        means = {name: np.mean(samples) for name, samples in output_samples.items()}
        medians = {name: np.median(samples) for name, samples in output_samples.items()}
        std_devs = {name: np.std(samples) for name, samples in output_samples.items()}

        # Confidence intervals
        ci_80 = {name: (np.percentile(samples, 10), np.percentile(samples, 90))
                 for name, samples in output_samples.items()}
        ci_90 = {name: (np.percentile(samples, 5), np.percentile(samples, 95))
                 for name, samples in output_samples.items()}
        ci_95 = {name: (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
                 for name, samples in output_samples.items()}

        # Convergence check
        convergence_metrics = None
        if self.config.convergence_check:
            convergence_metrics = self._check_convergence(output_samples)

        # Correlations
        correlations = None
        if self.config.compute_correlations:
            correlations = self._compute_correlations(param_samples, output_samples)

        return UncertaintyResults(
            n_samples=len(all_outputs),
            output_samples=output_samples,
            means=means,
            medians=medians,
            std_devs=std_devs,
            ci_80=ci_80,
            ci_90=ci_90,
            ci_95=ci_95,
            convergence_metrics=convergence_metrics,
            correlations=correlations
        )

    def _check_convergence(self, output_samples: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Check Monte Carlo convergence using running mean analysis.

        Returns coefficient of variation of running mean for last 10% of samples.
        """
        convergence = {}
        for name, samples in output_samples.items():
            n = len(samples)
            if n < 100:
                convergence[name] = float('inf')
                continue

            # Compute running mean
            running_mean = np.cumsum(samples) / np.arange(1, n + 1)

            # Check stability in last 10%
            last_10_pct = running_mean[int(0.9 * n):]
            cv = np.std(last_10_pct) / (np.mean(last_10_pct) + 1e-10)
            convergence[name] = cv

        return convergence

    def _compute_correlations(
        self,
        param_samples: Dict[str, np.ndarray],
        output_samples: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Pearson correlations between parameters and outputs.
        """
        correlations = {}
        for output_name, output_vals in output_samples.items():
            correlations[output_name] = {}
            for param_name, param_vals in param_samples.items():
                # Ensure same length
                min_len = min(len(output_vals), len(param_vals))
                corr = np.corrcoef(param_vals[:min_len], output_vals[:min_len])[0, 1]
                correlations[output_name][param_name] = corr

        return correlations

    def compute_roi_uncertainty(
        self,
        roi_func: Callable[[Dict[str, float]], Dict[str, float]],
        intervention_names: List[str]
    ) -> Dict[str, UncertaintyResults]:
        """
        Compute uncertainty in ROI estimates for policy interventions.

        This propagates QALY and other uncertainties through the ROI calculation.
        """
        n = self.config.n_samples

        # Sample QALY and value parameters
        qaly_samples = self.parameter_distributions['qaly_per_cure'].sample(n, self.rng)
        value_samples = self.parameter_distributions['value_per_qaly'].sample(n, self.rng)

        roi_results = {}
        for intervention in intervention_names:
            roi_samples = []
            for i in range(n):
                params = {
                    'qaly_per_cure': qaly_samples[i],
                    'value_per_qaly': value_samples[i]
                }
                try:
                    result = roi_func(params)
                    roi_samples.append(result.get(intervention, 0))
                except Exception:
                    continue

            roi_array = np.array(roi_samples)
            roi_results[intervention] = {
                'mean': np.mean(roi_array),
                'median': np.median(roi_array),
                'std': np.std(roi_array),
                'ci_80': (np.percentile(roi_array, 10), np.percentile(roi_array, 90)),
                'ci_95': (np.percentile(roi_array, 2.5), np.percentile(roi_array, 97.5))
            }

        return roi_results


def create_default_uq_config() -> UQConfig:
    """Create default UQ configuration for v1.0."""
    return UQConfig(
        n_samples=10000,
        n_sobol_samples=2048,
        seed=42,
        compute_sobol=True,
        compute_correlations=True,
        ci_levels=[0.80, 0.90, 0.95],
        convergence_check=True
    )


def format_ci(ci: Tuple[float, float], decimals: int = 1) -> str:
    """Format confidence interval for display."""
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def format_sobol_table(sobol: SobolIndices, top_n: int = 10) -> str:
    """Format Sobol indices as a markdown table."""
    lines = ["| Parameter | First-Order (S_i) | Total-Order (S_Ti) |",
             "|-----------|-------------------|-------------------|"]

    ranking = sobol.get_ranking(use_total=True)
    for name, s_ti in ranking[:top_n]:
        s_i = sobol.first_order.get(name, 0)
        lines.append(f"| {name} | {s_i:.3f} | {s_ti:.3f} |")

    return "\n".join(lines)
