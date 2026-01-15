"""
Full Sobol Sensitivity Analysis Module
======================================

Implements Saltelli's method for computing Sobol sensitivity indices
without external dependencies (SALib fallback).

This module provides:
- First-order indices (S_i): Main effect of parameter i
- Total-order indices (S_Ti): Total effect including interactions
- Second-order indices (S_ij): Pairwise interactions
- Parameter correlation structure with Cholesky decomposition
- Disease-specific QALY uncertainty propagation

References:
- Saltelli et al. (2010). "Variance based sensitivity analysis of model output"
- Sobol (2001). "Global sensitivity indices for nonlinear mathematical models"
- Iman & Conover (1982). "A distribution-free approach to inducing rank correlation"

EXPERT REVIEW IMPLEMENTATION:
- B1 (Dr. Rodriguez): Full Sobol with Saltelli estimator
- A2 (Dr. Mitchell): Parameter correlation structure
- E2 (Dr. Patel): Disease-specific QALY distributions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import warnings


@dataclass
class FullSobolResults:
    """Complete results from Saltelli-based Sobol analysis."""
    parameter_names: List[str]
    n_samples: int

    # First-order indices with confidence intervals
    first_order: Dict[str, float]
    first_order_ci: Dict[str, Tuple[float, float]]

    # Total-order indices with confidence intervals
    total_order: Dict[str, float]
    total_order_ci: Dict[str, Tuple[float, float]]

    # Second-order indices (pairwise)
    second_order: Optional[Dict[Tuple[str, str], float]] = None

    # Model evaluations count
    n_model_evals: int = 0

    def get_ranking(self, use_total: bool = True) -> List[Tuple[str, float]]:
        """Return parameters ranked by sensitivity."""
        indices = self.total_order if use_total else self.first_order
        return sorted(indices.items(), key=lambda x: x[1], reverse=True)

    def interaction_strength(self) -> Dict[str, float]:
        """Compute interaction strength: S_Ti - S_i for each parameter."""
        return {
            name: self.total_order[name] - self.first_order.get(name, 0)
            for name in self.parameter_names
        }


@dataclass
class CorrelationConfig:
    """Configuration for correlated parameter sampling."""
    # Correlation matrix entries (parameter pairs)
    # Positive correlation: g_ai and M_max tend to move together
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default correlations based on expert review A2."""
        if not self.correlations:
            # Expert A2 (Dr. Mitchell): g_ai and M_max likely have positive correlation
            self.correlations = {
                ('g_ai', 'M_max_cognitive'): 0.5,   # Higher AI growth → higher ceilings
                ('g_ai', 'M_max_clinical'): 0.3,   # Moderate correlation
                ('M_max_cognitive', 'M_max_clinical'): 0.4,  # Similar technology drivers
                ('p_phase2_base', 'p_phase2_max'): 0.6,  # Success rates correlated
                ('gamma_data', 'M_max_cognitive'): 0.3,  # Data quality enables AI
            }


@dataclass
class DiseaseQALYDistribution:
    """Disease-specific QALY distributions per Expert E2."""
    disease: str
    distribution_type: str  # 'triangular', 'lognormal', 'beta'
    params: Dict[str, float]
    description: str = ""

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from disease-specific QALY distribution."""
        if self.distribution_type == 'triangular':
            return rng.triangular(
                self.params['low'],
                self.params['mode'],
                self.params['high'],
                n
            )
        elif self.distribution_type == 'lognormal':
            return rng.lognormal(
                self.params['mu'],
                self.params['sigma'],
                n
            )
        elif self.distribution_type == 'beta':
            # Scale beta to range
            base = rng.beta(self.params['alpha'], self.params['beta'], n)
            return self.params['low'] + base * (self.params['high'] - self.params['low'])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution_type}")


# Disease-specific QALY distributions (Expert E2 - Dr. Patel)
DISEASE_QALY_DISTRIBUTIONS = {
    'cancer_early': DiseaseQALYDistribution(
        disease='cancer_early',
        distribution_type='triangular',
        params={'low': 10.0, 'mode': 15.0, 'high': 25.0},
        description="Early-stage cancer cure: young patients, good prognosis"
    ),
    'cancer_late': DiseaseQALYDistribution(
        disease='cancer_late',
        distribution_type='triangular',
        params={'low': 1.0, 'mode': 3.0, 'high': 6.0},
        description="Metastatic cancer: limited life extension"
    ),
    'alzheimers': DiseaseQALYDistribution(
        disease='alzheimers',
        distribution_type='triangular',
        params={'low': 0.5, 'mode': 2.0, 'high': 4.0},
        description="Alzheimer's: primarily quality improvement"
    ),
    'pandemic': DiseaseQALYDistribution(
        disease='pandemic',
        distribution_type='lognormal',
        params={'mu': np.log(0.3), 'sigma': 0.5},
        description="Pandemic vaccine: prevention, per dose (low variance)"
    ),
    'rare_genetic': DiseaseQALYDistribution(
        disease='rare_genetic',
        distribution_type='triangular',
        params={'low': 8.0, 'mode': 12.0, 'high': 20.0},
        description="Rare genetic cure: often pediatric, long life ahead"
    ),
    'infectious': DiseaseQALYDistribution(
        disease='infectious',
        distribution_type='triangular',
        params={'low': 5.0, 'mode': 8.0, 'high': 15.0},
        description="HIV/TB cure: significant life years gained"
    ),
}


class SaltelliSobolAnalysis:
    """
    Full Sobol sensitivity analysis using Saltelli's sampling scheme.

    This is a pure-numpy implementation that doesn't require SALib.

    The Saltelli method generates (2D + 2) * N model evaluations where:
    - D = number of parameters
    - N = base sample size

    For D=7 parameters and N=1024, this requires 16,384 evaluations.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.correlation_config = CorrelationConfig()

    def generate_saltelli_samples(
        self,
        n_samples: int,
        param_names: List[str],
        sample_func: Callable[[str, int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate Saltelli sampling matrices A, B, and AB_i.

        Returns:
            A: Base matrix (n_samples x n_params)
            B: Complementary matrix (n_samples x n_params)
            AB_list: List of matrices where AB_i has column i from B, rest from A
        """
        n_params = len(param_names)

        # Generate base matrices A and B
        A = np.zeros((n_samples, n_params))
        B = np.zeros((n_samples, n_params))

        for j, name in enumerate(param_names):
            A[:, j] = sample_func(name, n_samples)
            B[:, j] = sample_func(name, n_samples)

        # Generate AB_i matrices (A with i-th column from B)
        AB_list = []
        for i in range(n_params):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            AB_list.append(AB_i)

        return A, B, AB_list

    def compute_sobol_indices(
        self,
        model_func: Callable[[Dict[str, float]], float],
        param_names: List[str],
        sample_func: Callable[[str, int], np.ndarray],
        n_samples: int = 1024,
        compute_second_order: bool = False
    ) -> FullSobolResults:
        """
        Compute Sobol indices using Saltelli's method.

        Args:
            model_func: Function mapping parameter dict to scalar output
            param_names: List of parameter names
            sample_func: Function to sample each parameter
            n_samples: Base sample size (total evals = (2D+2)*N)
            compute_second_order: Whether to compute S_ij

        Returns:
            FullSobolResults with all indices and CIs
        """
        n_params = len(param_names)

        # Generate Saltelli samples
        A, B, AB_list = self.generate_saltelli_samples(
            n_samples, param_names, sample_func
        )

        # Evaluate model on all samples
        def eval_matrix(X: np.ndarray) -> np.ndarray:
            outputs = []
            for i in range(X.shape[0]):
                params = {name: X[i, j] for j, name in enumerate(param_names)}
                try:
                    outputs.append(model_func(params))
                except Exception:
                    outputs.append(np.nan)
            return np.array(outputs)

        print(f"  Evaluating model on {(2*n_params + 2) * n_samples:,} samples...")
        f_A = eval_matrix(A)
        f_B = eval_matrix(B)
        f_AB = [eval_matrix(AB_i) for AB_i in AB_list]

        # Remove NaN samples
        valid_mask = ~np.isnan(f_A) & ~np.isnan(f_B)
        for f_ab in f_AB:
            valid_mask &= ~np.isnan(f_ab)

        f_A = f_A[valid_mask]
        f_B = f_B[valid_mask]
        f_AB = [f_ab[valid_mask] for f_ab in f_AB]
        n_valid = len(f_A)

        if n_valid < 100:
            raise ValueError(f"Too few valid samples: {n_valid}")

        # Total variance
        f_all = np.concatenate([f_A, f_B])
        total_var = np.var(f_all)

        if total_var < 1e-10:
            warnings.warn("Near-zero output variance")
            return self._empty_results(param_names, n_samples)

        # Compute first-order and total-order indices
        first_order = {}
        total_order = {}
        first_order_ci = {}
        total_order_ci = {}

        for i, name in enumerate(param_names):
            # First-order: Jansen estimator
            # S_i = V[E[Y|X_i]] / V[Y]
            # Estimated as: mean(f_B * (f_AB_i - f_A)) / V[Y]
            V_i = np.mean(f_B * (f_AB[i] - f_A))
            S_i = max(0, V_i / total_var)
            first_order[name] = S_i

            # Total-order: Jansen estimator
            # S_Ti = E[V[Y|X_~i]] / V[Y]
            # Estimated as: 0.5 * mean((f_A - f_AB_i)^2) / V[Y]
            V_Ti = 0.5 * np.mean((f_A - f_AB[i]) ** 2)
            S_Ti = max(0, V_Ti / total_var)
            total_order[name] = S_Ti

            # Bootstrap confidence intervals
            S_i_boot, S_Ti_boot = self._bootstrap_indices(
                f_A, f_B, f_AB[i], total_var, n_bootstrap=100
            )
            first_order_ci[name] = (np.percentile(S_i_boot, 5), np.percentile(S_i_boot, 95))
            total_order_ci[name] = (np.percentile(S_Ti_boot, 5), np.percentile(S_Ti_boot, 95))

        # Normalize if sum exceeds 1 (numerical issues)
        s1_sum = sum(first_order.values())
        if s1_sum > 1.2:
            first_order = {k: v / s1_sum for k, v in first_order.items()}

        # Second-order indices (optional)
        second_order = None
        if compute_second_order and n_params <= 10:
            second_order = self._compute_second_order(
                f_A, f_B, f_AB, param_names, total_var
            )

        return FullSobolResults(
            parameter_names=param_names,
            n_samples=n_valid,
            first_order=first_order,
            first_order_ci=first_order_ci,
            total_order=total_order,
            total_order_ci=total_order_ci,
            second_order=second_order,
            n_model_evals=(2 * n_params + 2) * n_samples
        )

    def _bootstrap_indices(
        self,
        f_A: np.ndarray,
        f_B: np.ndarray,
        f_AB_i: np.ndarray,
        total_var: float,
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap confidence intervals for Sobol indices."""
        n = len(f_A)
        S_i_samples = []
        S_Ti_samples = []

        for _ in range(n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            f_A_b = f_A[idx]
            f_B_b = f_B[idx]
            f_AB_b = f_AB_i[idx]

            V_i = np.mean(f_B_b * (f_AB_b - f_A_b))
            S_i = max(0, V_i / total_var)
            S_i_samples.append(S_i)

            V_Ti = 0.5 * np.mean((f_A_b - f_AB_b) ** 2)
            S_Ti = max(0, V_Ti / total_var)
            S_Ti_samples.append(S_Ti)

        return np.array(S_i_samples), np.array(S_Ti_samples)

    def _compute_second_order(
        self,
        f_A: np.ndarray,
        f_B: np.ndarray,
        f_AB: List[np.ndarray],
        param_names: List[str],
        total_var: float
    ) -> Dict[Tuple[str, str], float]:
        """Compute second-order Sobol indices (pairwise interactions)."""
        second_order = {}
        n_params = len(param_names)

        for i in range(n_params):
            for j in range(i + 1, n_params):
                # S_ij = V_ij / V[Y] - S_i - S_j
                # This is an approximation
                V_ij = np.mean(f_AB[i] * f_AB[j]) - np.mean(f_A) ** 2
                V_i = np.mean(f_B * (f_AB[i] - f_A))
                V_j = np.mean(f_B * (f_AB[j] - f_A))

                S_ij = max(0, (V_ij - V_i - V_j) / total_var)
                second_order[(param_names[i], param_names[j])] = S_ij

        return second_order

    def _empty_results(self, param_names: List[str], n_samples: int) -> FullSobolResults:
        """Return empty results for zero-variance case."""
        return FullSobolResults(
            parameter_names=param_names,
            n_samples=n_samples,
            first_order={name: 0.0 for name in param_names},
            first_order_ci={name: (0.0, 0.0) for name in param_names},
            total_order={name: 0.0 for name in param_names},
            total_order_ci={name: (0.0, 0.0) for name in param_names},
        )


class CorrelatedSampler:
    """
    Generate correlated parameter samples using Iman-Conover method.

    This preserves marginal distributions while inducing rank correlation.
    Addresses Expert Review A2 (Dr. Mitchell).
    """

    def __init__(self, correlation_config: CorrelationConfig = None, seed: int = 42):
        self.config = correlation_config or CorrelationConfig()
        self.rng = np.random.default_rng(seed)

    def build_correlation_matrix(self, param_names: List[str]) -> np.ndarray:
        """Build full correlation matrix from pairwise correlations."""
        n = len(param_names)
        corr_matrix = np.eye(n)

        for (p1, p2), rho in self.config.correlations.items():
            if p1 in param_names and p2 in param_names:
                i = param_names.index(p1)
                j = param_names.index(p2)
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(corr_matrix)
        if np.min(eigvals) < 0:
            # Add small diagonal to make PD
            corr_matrix += np.eye(n) * (abs(np.min(eigvals)) + 0.01)
            # Renormalize diagonal
            d = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(d, d)

        return corr_matrix

    def sample_correlated(
        self,
        n_samples: int,
        param_names: List[str],
        marginal_samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated samples using Iman-Conover method.

        Args:
            n_samples: Number of samples
            param_names: Parameter names
            marginal_samples: Pre-generated marginal samples (uncorrelated)

        Returns:
            Correlated samples with same marginal distributions
        """
        n_params = len(param_names)

        # Build target correlation matrix
        target_corr = self.build_correlation_matrix(param_names)

        # Generate standard normal with target correlation
        L = np.linalg.cholesky(target_corr)
        Z = self.rng.standard_normal((n_samples, n_params))
        Z_corr = Z @ L.T

        # Get rank indices from correlated normal
        ranks_target = np.zeros((n_samples, n_params), dtype=int)
        for j in range(n_params):
            ranks_target[:, j] = np.argsort(np.argsort(Z_corr[:, j]))

        # Reorder marginal samples according to correlated ranks
        correlated = {}
        for j, name in enumerate(param_names):
            marginal = marginal_samples[name].copy()
            sorted_marginal = np.sort(marginal)
            correlated[name] = sorted_marginal[ranks_target[:, j]]

        return correlated


def compute_qaly_uncertainty_by_disease(
    n_samples: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute QALY uncertainty statistics for each disease category.

    Addresses Expert Review E2 (Dr. Patel): Disease-specific QALY distributions
    with different variances.

    Returns:
        Dict mapping disease to {mean, std, ci_80_low, ci_80_high, cv}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for disease, dist in DISEASE_QALY_DISTRIBUTIONS.items():
        samples = dist.sample(n_samples, rng)

        results[disease] = {
            'mean': float(np.mean(samples)),
            'median': float(np.median(samples)),
            'std': float(np.std(samples)),
            'cv': float(np.std(samples) / np.mean(samples)),  # Coefficient of variation
            'ci_80_low': float(np.percentile(samples, 10)),
            'ci_80_high': float(np.percentile(samples, 90)),
            'ci_95_low': float(np.percentile(samples, 2.5)),
            'ci_95_high': float(np.percentile(samples, 97.5)),
            'description': dist.description,
        }

    return results


def format_sobol_results(results: FullSobolResults) -> str:
    """Format Sobol results as a readable table."""
    lines = [
        "Sobol Sensitivity Indices (Saltelli Method)",
        "=" * 70,
        f"Total model evaluations: {results.n_model_evals:,}",
        f"Valid samples: {results.n_samples:,}",
        "",
        f"{'Parameter':<25} {'S_i (First)':<15} {'S_Ti (Total)':<15} {'Interaction':<15}",
        "-" * 70,
    ]

    interactions = results.interaction_strength()
    ranking = results.get_ranking(use_total=True)

    for name, s_ti in ranking:
        s_i = results.first_order.get(name, 0)
        s_i_ci = results.first_order_ci.get(name, (0, 0))
        s_ti_ci = results.total_order_ci.get(name, (0, 0))
        interaction = interactions.get(name, 0)

        lines.append(
            f"{name:<25} {s_i:>6.3f} [{s_i_ci[0]:.2f},{s_i_ci[1]:.2f}]  "
            f"{s_ti:>6.3f} [{s_ti_ci[0]:.2f},{s_ti_ci[1]:.2f}]  "
            f"{interaction:>6.3f}"
        )

    # Add second-order if available
    if results.second_order:
        lines.extend(["", "Top 5 Pairwise Interactions:", "-" * 40])
        sorted_pairs = sorted(
            results.second_order.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for (p1, p2), s_ij in sorted_pairs:
            if s_ij > 0.001:
                lines.append(f"  {p1} × {p2}: S_ij = {s_ij:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    print("Testing Sobol Analysis Module")
    print("=" * 50)

    # Test QALY uncertainty
    print("\nDisease-Specific QALY Uncertainty:")
    qaly_results = compute_qaly_uncertainty_by_disease(1000)
    for disease, stats in qaly_results.items():
        print(f"  {disease}: mean={stats['mean']:.1f}, CV={stats['cv']:.2f}, "
              f"80% CI=[{stats['ci_80_low']:.1f}, {stats['ci_80_high']:.1f}]")

    # Test correlation matrix
    print("\nCorrelation Matrix Test:")
    sampler = CorrelatedSampler()
    params = ['g_ai', 'M_max_cognitive', 'M_max_clinical', 'p_phase2_base']
    corr = sampler.build_correlation_matrix(params)
    print("  Parameters:", params)
    print("  Correlation matrix:")
    for i, p in enumerate(params):
        print(f"    {p}: {corr[i]}")

    print("\nModule loaded successfully.")
