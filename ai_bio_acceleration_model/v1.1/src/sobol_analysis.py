#!/usr/bin/env python3
"""
Full Sobol Sensitivity Analysis Module - v1.1

Updates from v1.0:
- P2-11: Bootstrap CIs on Sobol indices (1000 samples, 90% CI)
- P1-1: Explicit APPROXIMATE labeling

Implements Saltelli's method for computing Sobol sensitivity indices.

Version: 1.1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import warnings


@dataclass
class SobolIndices:
    """
    P1-1: Sobol indices with explicit APPROXIMATE flag.

    All Sobol sensitivity indices are explicitly marked as approximate
    because they use correlation-based proxy, not full Saltelli method.
    """
    parameter_name: str
    first_order: float
    total_order: float
    first_order_ci: Tuple[float, float]  # P2-11: Bootstrap CI
    total_order_ci: Tuple[float, float]   # P2-11: Bootstrap CI
    interaction_strength: float
    is_approximate: bool = True  # P1-1: Always True for this implementation
    method: str = "correlation_proxy"  # P1-1: Document method used


@dataclass
class FullSobolResults:
    """Complete results from Sobol analysis with P1-1 and P2-11 updates."""
    parameter_names: List[str]
    n_samples: int
    n_bootstrap: int  # P2-11

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

    # P1-1: Explicit approximation flag
    is_approximate: bool = True
    method_note: str = "Correlation-based proxy, not full Saltelli method"

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

    def get_sobol_indices(self) -> List[SobolIndices]:
        """P1-1: Get list of SobolIndices objects with explicit flags."""
        indices = []
        interactions = self.interaction_strength()

        for name in self.parameter_names:
            indices.append(SobolIndices(
                parameter_name=name,
                first_order=self.first_order[name],
                total_order=self.total_order[name],
                first_order_ci=self.first_order_ci[name],
                total_order_ci=self.total_order_ci[name],
                interaction_strength=interactions[name],
                is_approximate=True,  # P1-1
                method="correlation_proxy"
            ))

        return indices


class SaltelliSobolAnalysis:
    """
    Full Sobol sensitivity analysis using Saltelli's sampling scheme.

    v1.1 Updates:
    - P2-11: Bootstrap CIs with 1000 samples, 90% CI
    - P1-1: Results explicitly marked as APPROXIMATE
    """

    def __init__(self, seed: int = 42, n_bootstrap: int = 1000):
        self.rng = np.random.default_rng(seed)
        self.n_bootstrap = n_bootstrap  # P2-11

    def generate_saltelli_samples(
        self,
        n_samples: int,
        param_names: List[str],
        sample_func: Callable[[str, int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Generate Saltelli sampling matrices A, B, and AB_i."""
        n_params = len(param_names)

        A = np.zeros((n_samples, n_params))
        B = np.zeros((n_samples, n_params))

        for j, name in enumerate(param_names):
            A[:, j] = sample_func(name, n_samples)
            B[:, j] = sample_func(name, n_samples)

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

        P2-11: Now with bootstrap CIs (1000 samples, 90% CI)
        P1-1: Results marked as APPROXIMATE
        """
        n_params = len(param_names)

        A, B, AB_list = self.generate_saltelli_samples(
            n_samples, param_names, sample_func
        )

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

        valid_mask = ~np.isnan(f_A) & ~np.isnan(f_B)
        for f_ab in f_AB:
            valid_mask &= ~np.isnan(f_ab)

        f_A = f_A[valid_mask]
        f_B = f_B[valid_mask]
        f_AB = [f_ab[valid_mask] for f_ab in f_AB]
        n_valid = len(f_A)

        if n_valid < 100:
            raise ValueError(f"Too few valid samples: {n_valid}")

        f_all = np.concatenate([f_A, f_B])
        total_var = np.var(f_all)

        if total_var < 1e-10:
            warnings.warn("Near-zero output variance")
            return self._empty_results(param_names, n_samples)

        first_order = {}
        total_order = {}
        first_order_ci = {}
        total_order_ci = {}

        for i, name in enumerate(param_names):
            # First-order: Jansen estimator
            V_i = np.mean(f_B * (f_AB[i] - f_A))
            S_i = max(0, V_i / total_var)
            first_order[name] = S_i

            # Total-order: Jansen estimator
            V_Ti = 0.5 * np.mean((f_A - f_AB[i]) ** 2)
            S_Ti = max(0, V_Ti / total_var)
            total_order[name] = S_Ti

            # P2-11: Bootstrap confidence intervals (1000 samples, 90% CI)
            S_i_boot, S_Ti_boot = self._bootstrap_indices(
                f_A, f_B, f_AB[i], total_var, n_bootstrap=self.n_bootstrap
            )
            first_order_ci[name] = (
                float(np.percentile(S_i_boot, 5)),
                float(np.percentile(S_i_boot, 95))
            )
            total_order_ci[name] = (
                float(np.percentile(S_Ti_boot, 5)),
                float(np.percentile(S_Ti_boot, 95))
            )

        # Normalize if sum exceeds 1
        s1_sum = sum(first_order.values())
        if s1_sum > 1.2:
            first_order = {k: v / s1_sum for k, v in first_order.items()}

        second_order = None
        if compute_second_order and n_params <= 10:
            second_order = self._compute_second_order(
                f_A, f_B, f_AB, param_names, total_var
            )

        return FullSobolResults(
            parameter_names=param_names,
            n_samples=n_valid,
            n_bootstrap=self.n_bootstrap,  # P2-11
            first_order=first_order,
            first_order_ci=first_order_ci,
            total_order=total_order,
            total_order_ci=total_order_ci,
            second_order=second_order,
            n_model_evals=(2 * n_params + 2) * n_samples,
            is_approximate=True,  # P1-1
            method_note="APPROXIMATE: Correlation-based proxy, not full Saltelli method"
        )

    def _bootstrap_indices(
        self,
        f_A: np.ndarray,
        f_B: np.ndarray,
        f_AB_i: np.ndarray,
        total_var: float,
        n_bootstrap: int = 1000  # P2-11: Default 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """P2-11: Bootstrap confidence intervals for Sobol indices."""
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
            n_bootstrap=self.n_bootstrap,
            first_order={name: 0.0 for name in param_names},
            first_order_ci={name: (0.0, 0.0) for name in param_names},
            total_order={name: 0.0 for name in param_names},
            total_order_ci={name: (0.0, 0.0) for name in param_names},
            is_approximate=True,
        )


def format_sobol_results(results: FullSobolResults) -> str:
    """Format Sobol results as a readable table with P1-1 warning."""
    lines = [
        "=" * 70,
        "Sobol Sensitivity Indices",
        "=" * 70,
        "",
        "*** IMPORTANT: These indices are APPROXIMATE ***",  # P1-1
        f"Method: {results.method_note}",
        f"Bootstrap samples: {results.n_bootstrap} (90% CI)",  # P2-11
        "",
        f"Total model evaluations: {results.n_model_evals:,}",
        f"Valid samples: {results.n_samples:,}",
        "",
        f"{'Parameter':<25} {'S_i (First)':<20} {'S_Ti (Total)':<20}",
        "-" * 70,
    ]

    ranking = results.get_ranking(use_total=True)

    for name, s_ti in ranking:
        s_i = results.first_order.get(name, 0)
        s_i_ci = results.first_order_ci.get(name, (0, 0))
        s_ti_ci = results.total_order_ci.get(name, (0, 0))

        lines.append(
            f"{name:<25} {s_i:>5.3f} [{s_i_ci[0]:.3f},{s_i_ci[1]:.3f}]  "
            f"{s_ti:>5.3f} [{s_ti_ci[0]:.3f},{s_ti_ci[1]:.3f}]"
        )

    if results.second_order:
        lines.extend(["", "Top 5 Pairwise Interactions:", "-" * 40])
        sorted_pairs = sorted(
            results.second_order.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for (p1, p2), s_ij in sorted_pairs:
            if s_ij > 0.001:
                lines.append(f"  {p1} x {p2}: S_ij = {s_ij:.4f}")

    lines.extend([
        "",
        "NOTE: For full Saltelli method, use SALib library.",
        "These correlation-based indices provide relative rankings",
        "but may not sum to exactly 1.0.",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Sobol Analysis Module - v1.1")
    print("=" * 70)

    print("\nP1-1: All Sobol indices explicitly marked as APPROXIMATE")
    print("P2-11: Bootstrap CIs with 1000 samples, 90% CI")

    # Test bootstrap
    print("\nBootstrap CI Test:")
    analyzer = SaltelliSobolAnalysis(seed=42, n_bootstrap=1000)

    # Create dummy data for testing
    n = 500
    rng = np.random.default_rng(42)
    f_A = rng.normal(100, 20, n)
    f_B = rng.normal(100, 20, n)
    f_AB = f_A + rng.normal(0, 5, n)
    total_var = np.var(np.concatenate([f_A, f_B]))

    S_i_boot, S_Ti_boot = analyzer._bootstrap_indices(f_A, f_B, f_AB, total_var)

    print(f"  S_i: mean={np.mean(S_i_boot):.3f}, "
          f"90% CI=[{np.percentile(S_i_boot, 5):.3f}, {np.percentile(S_i_boot, 95):.3f}]")
    print(f"  S_Ti: mean={np.mean(S_Ti_boot):.3f}, "
          f"90% CI=[{np.percentile(S_Ti_boot, 5):.3f}, {np.percentile(S_Ti_boot, 95):.3f}]")

    print("\nModule loaded successfully.")
