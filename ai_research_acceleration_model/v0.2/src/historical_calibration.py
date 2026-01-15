"""
Historical Calibration Module
=============================

Calibrates the AI Research Acceleration Model against five historical
technology-enabled scientific paradigm shifts:

1. Microscope (1600s-1700s) - Capability extension → Conceptual shift
2. Telescope (1600s-1700s) - Capability extension
3. Human Genome Project (1990-2003) - Capability extension
4. DNA Sequencing (2005-2015) - Capability extension
5. CRISPR (2012-present) - Methodological shift

Uses Bayesian parameter estimation to find model parameters that best
match observed acceleration metrics from historical data.

Based on PROJECT_BIBLE.md Section 3 and Section 15 (v0.2 roadmap).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import json
import warnings

# Import v0.1 components
import sys
sys.path.insert(0, '/sessions/intelligent-beautiful-shannon/mnt/Accelerating_biology_with_AI/ai_research_acceleration_model/v0.1')
from src.model import AIResearchAccelerationModel, Scenario, SCENARIO_DEFAULTS


class ShiftCategory(Enum):
    """Categories of historical paradigm shifts."""
    CAPABILITY_EXTENSION = "capability_extension"    # Expand what can be done
    METHODOLOGICAL = "methodological"                # Change how research is done
    COMPUTATIONAL = "computational"                  # Computational/algorithmic acceleration
    CONCEPTUAL = "conceptual"                       # Change fundamental assumptions


@dataclass
class HistoricalShift:
    """
    Represents a historical technology-enabled scientific paradigm shift.

    Contains measured acceleration metrics that the model will be calibrated against.
    """
    name: str
    category: ShiftCategory
    start_year: int
    transformation_years: int           # Years to reach ~10x impact
    full_impact_years: int             # Years to field transformation

    # Observed acceleration metrics (for calibration targets)
    time_acceleration: float           # e.g., 4700x for HGP
    cost_reduction: Optional[float]    # e.g., 15,000,000x for sequencing
    publication_increase: float        # e.g., 160x for HGP

    # Uncertainty bounds (as factors, e.g., 0.5 = ±50%)
    time_accel_uncertainty: float = 0.3
    pub_increase_uncertainty: float = 0.4

    # Additional context
    description: str = ""
    key_insight: str = ""


# Historical data from PROJECT_BIBLE.md Section 3.2
HISTORICAL_SHIFTS = {
    "microscope": HistoricalShift(
        name="Microscope",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=1600,
        transformation_years=50,
        full_impact_years=150,
        time_acceleration=10.0,        # Enabled entirely new observations
        cost_reduction=None,
        publication_increase=100.0,    # Estimate: created entire new field
        time_accel_uncertainty=0.5,    # High uncertainty for historical data
        pub_increase_uncertainty=0.6,
        description="Revealed micro world, enabled cell theory and germ theory",
        key_insight="Capability extension enabled conceptual shift, but required additional human intellectual work",
    ),

    "telescope": HistoricalShift(
        name="Telescope",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=1609,
        transformation_years=30,
        full_impact_years=150,
        time_acceleration=100.0,       # 100-1000x data multiplication
        cost_reduction=None,
        publication_increase=50.0,     # Estimate
        time_accel_uncertainty=0.5,
        pub_increase_uncertainty=0.6,
        description="Provided overwhelming evidence for heliocentric model",
        key_insight="Provided evidence but conceptual frameworks still required human genius",
    ),

    "hgp": HistoricalShift(
        name="Human Genome Project / Sequencing",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=1990,
        transformation_years=5,        # Post-HGP to 10x impact
        full_impact_years=25,
        time_acceleration=4700.0,      # 13 years → 1 day
        cost_reduction=15_000_000.0,   # $3B → $200
        publication_increase=160.0,    # 500 → 80,000 pub/year
        time_accel_uncertainty=0.1,    # Well-documented
        pub_increase_uncertainty=0.2,
        description="Front-loaded infrastructure investment unlocked exponential returns",
        key_insight="Infrastructure investment enables sustained acceleration",
    ),

    "sequencing": HistoricalShift(
        name="DNA Sequencing Revolution",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=2005,
        transformation_years=5,
        full_impact_years=20,
        time_acceleration=1000.0,      # Week → day
        cost_reduction=15_000_000.0,   # $3B → $200 (similar trajectory)
        publication_increase=75.0,     # Field growth
        time_accel_uncertainty=0.15,
        pub_increase_uncertainty=0.25,
        description="Multiple technology generations drove continued acceleration",
        key_insight="Technology democratization enables field explosion",
    ),

    "crispr": HistoricalShift(
        name="CRISPR",
        category=ShiftCategory.METHODOLOGICAL,
        start_year=2012,
        transformation_years=3,
        full_impact_years=15,
        time_acceleration=15.0,        # 6-12 months → 2-4 weeks
        cost_reduction=1000.0,         # $50,000 → $50
        publication_increase=75.0,     # 200 → 15,000 pub/year
        time_accel_uncertainty=0.2,
        pub_increase_uncertainty=0.25,
        description="Made expert-only techniques accessible to all labs",
        key_insight="Democratization is key driver of methodological impact",
    ),
}

# Extended historical shifts including molecular biology and computational biology
# Added per expert reviewer feedback (H4, H5)
EXTENDED_HISTORICAL_SHIFTS = {
    **HISTORICAL_SHIFTS,

    # === Molecular Biology Technologies (H4 recommendations) ===

    "xray_crystallography": HistoricalShift(
        name="X-ray Crystallography",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=1953,
        transformation_years=15,
        full_impact_years=40,
        time_acceleration=100.0,       # Structure determination: months → days (with synchrotrons)
        cost_reduction=100.0,          # Equipment costs decreased significantly
        publication_increase=200.0,    # Created structural biology field
        time_accel_uncertainty=0.3,
        pub_increase_uncertainty=0.4,
        description="Enabled determination of molecular structures including DNA double helix",
        key_insight="Foundation for understanding molecular biology; AlphaFold's significance is replacing this",
    ),

    "recombinant_dna": HistoricalShift(
        name="Recombinant DNA",
        category=ShiftCategory.METHODOLOGICAL,
        start_year=1973,
        transformation_years=5,
        full_impact_years=20,
        time_acceleration=1000.0,      # Gene manipulation: impossible → routine
        cost_reduction=10000.0,        # Created entire industry
        publication_increase=500.0,    # Foundation for modern molecular biology
        time_accel_uncertainty=0.25,
        pub_increase_uncertainty=0.3,
        description="Cohen-Boyer cloning enabled genetic engineering",
        key_insight="More foundational than CRISPR for creating molecular biology field",
    ),

    "pcr": HistoricalShift(
        name="PCR",
        category=ShiftCategory.METHODOLOGICAL,
        start_year=1983,
        transformation_years=5,
        full_impact_years=15,
        time_acceleration=1000000.0,   # DNA amplification: impossible → trivial
        cost_reduction=100000.0,       # Enabled by thermostable polymerase
        publication_increase=300.0,    # Foundational technique
        time_accel_uncertainty=0.2,
        pub_increase_uncertainty=0.3,
        description="Enabled amplification of DNA from minimal samples",
        key_insight="Made molecular biology accessible; Nobel Prize 1993",
    ),

    "cryo_em": HistoricalShift(
        name="Cryo-EM",
        category=ShiftCategory.CAPABILITY_EXTENSION,
        start_year=2013,
        transformation_years=4,
        full_impact_years=12,
        time_acceleration=50.0,        # Structure without crystals: years → weeks
        cost_reduction=10.0,           # Still expensive but accessible
        publication_increase=100.0,    # Rapid field growth
        time_accel_uncertainty=0.2,
        pub_increase_uncertainty=0.25,
        description="Near-atomic resolution structures without crystallization",
        key_insight="Nobel Prize 2017; competes with AlphaFold for structural biology",
    ),

    # === Computational Biology Technologies (H5 recommendations) ===

    "blast": HistoricalShift(
        name="BLAST Algorithm",
        category=ShiftCategory.COMPUTATIONAL,
        start_year=1990,
        transformation_years=3,
        full_impact_years=10,
        time_acceleration=1000.0,      # Sequence search: days → seconds
        cost_reduction=1000.0,         # Made sequence comparison trivial
        publication_increase=50.0,     # Enabled new research paradigm
        time_accel_uncertainty=0.15,
        pub_increase_uncertainty=0.25,
        description="Basic Local Alignment Search Tool revolutionized sequence analysis",
        key_insight="Most cited paper in biology; foundation for bioinformatics",
    ),

    "genbank": HistoricalShift(
        name="GenBank/Databases",
        category=ShiftCategory.COMPUTATIONAL,
        start_year=1982,
        transformation_years=8,
        full_impact_years=20,
        time_acceleration=100.0,       # Data access: impossible → instant
        cost_reduction=10000.0,        # Free public access to sequences
        publication_increase=30.0,     # Enabled data sharing
        time_accel_uncertainty=0.3,
        pub_increase_uncertainty=0.35,
        description="NCBI databases created shared knowledge infrastructure",
        key_insight="Infrastructure investment that made HGP data useful",
    ),

    "rosetta": HistoricalShift(
        name="Rosetta Structure Prediction",
        category=ShiftCategory.COMPUTATIONAL,
        start_year=2005,
        transformation_years=5,
        full_impact_years=15,
        time_acceleration=10.0,        # Computational prediction: months → hours
        cost_reduction=100.0,          # No wet lab needed
        publication_increase=20.0,     # Specialized field
        time_accel_uncertainty=0.25,
        pub_increase_uncertainty=0.3,
        description="First effective computational protein structure prediction",
        key_insight="The correct baseline for measuring AlphaFold's impact (~10x over Rosetta)",
    ),

    "ml_biology": HistoricalShift(
        name="Machine Learning in Biology",
        category=ShiftCategory.COMPUTATIONAL,
        start_year=2010,
        transformation_years=5,
        full_impact_years=15,
        time_acceleration=10.0,        # Automated annotation: days → minutes
        cost_reduction=10.0,           # Reduced need for manual curation
        publication_increase=50.0,     # Rapid adoption
        time_accel_uncertainty=0.25,
        pub_increase_uncertainty=0.3,
        description="SVM, Random Forests, early neural networks for classification",
        key_insight="Foundation for deep learning; AI acceleration is cumulative, not sudden",
    ),
}


@dataclass
class CalibrationTarget:
    """A single metric to calibrate against."""
    shift_name: str
    metric_name: str                    # 'time_acceleration', 'publication_increase'
    observed_value: float
    uncertainty: float                  # As standard deviation
    weight: float = 1.0                 # Relative importance


@dataclass
class CalibrationResult:
    """Results from Bayesian parameter estimation."""
    # Estimated parameters
    parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float]

    # Calibration quality metrics
    log_likelihood: float
    aic: float                          # Akaike Information Criterion
    bic: float                          # Bayesian Information Criterion

    # Individual target fit
    target_residuals: Dict[str, float]
    target_mape: float                  # Mean Absolute Percentage Error

    # Posterior samples (if MCMC used)
    posterior_samples: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "HISTORICAL CALIBRATION RESULTS",
            "=" * 60,
            "",
            "Estimated Parameters:",
        ]

        for param, value in self.parameters.items():
            unc = self.parameter_uncertainties.get(param, 0)
            lines.append(f"  {param}: {value:.4f} ± {unc:.4f}")

        lines.extend([
            "",
            "Model Fit Quality:",
            f"  Log-Likelihood: {self.log_likelihood:.2f}",
            f"  AIC: {self.aic:.2f}",
            f"  BIC: {self.bic:.2f}",
            f"  Mean Abs % Error: {self.target_mape:.1%}",
            "",
            "Individual Target Residuals:",
        ])

        for target, residual in self.target_residuals.items():
            lines.append(f"  {target}: {residual:+.2f}σ")

        lines.append("=" * 60)
        return "\n".join(lines)


class HistoricalCalibrator:
    """
    Calibrates model parameters against historical paradigm shift data.

    Uses Bayesian inference to estimate:
    - Base acceleration rates for different shift categories
    - Time scaling factors
    - Uncertainty quantification on all parameters
    """

    # Parameters to calibrate with priors
    CALIBRATION_PARAMS = {
        # param_name: (prior_mean, prior_std, bounds)
        'capability_accel_scale': (100.0, 50.0, (10.0, 1000.0)),
        'methodological_accel_scale': (100.0, 50.0, (10.0, 10000.0)),  # Increased for PCR
        'computational_accel_scale': (100.0, 50.0, (10.0, 1000.0)),   # NEW: for BLAST, Rosetta, etc.
        'time_to_impact_scale': (1.0, 0.3, (0.3, 3.0)),
        'democratization_factor': (2.0, 0.5, (1.0, 5.0)),
        'infrastructure_boost': (1.5, 0.3, (1.0, 3.0)),
        'cumulative_factor': (1.2, 0.2, (1.0, 2.0)),  # NEW: models cumulative acceleration effect
    }

    def __init__(
        self,
        shifts: Optional[Dict[str, HistoricalShift]] = None,
        seed: int = 42,
    ):
        """
        Initialize the calibrator.

        Args:
            shifts: Historical shifts to calibrate against (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.shifts = shifts or HISTORICAL_SHIFTS
        self.rng = np.random.default_rng(seed)
        self.targets = self._build_calibration_targets()

    def _build_calibration_targets(self) -> List[CalibrationTarget]:
        """Build list of calibration targets from historical data."""
        targets = []

        for name, shift in self.shifts.items():
            # Time acceleration target
            targets.append(CalibrationTarget(
                shift_name=name,
                metric_name='time_acceleration',
                observed_value=np.log10(shift.time_acceleration),  # Use log scale
                uncertainty=shift.time_accel_uncertainty * np.log10(shift.time_acceleration),
                weight=1.0,
            ))

            # Publication increase target
            targets.append(CalibrationTarget(
                shift_name=name,
                metric_name='publication_increase',
                observed_value=np.log10(shift.publication_increase),
                uncertainty=shift.pub_increase_uncertainty * np.log10(shift.publication_increase),
                weight=0.8,  # Slightly lower weight (more noisy metric)
            ))

            # Transformation time target
            targets.append(CalibrationTarget(
                shift_name=name,
                metric_name='transformation_years',
                observed_value=shift.transformation_years,
                uncertainty=shift.transformation_years * 0.3,
                weight=0.6,  # Lower weight (qualitative)
            ))

        return targets

    def predict_metrics(
        self,
        shift: HistoricalShift,
        params: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Predict metrics for a historical shift given model parameters.

        This is the forward model that maps parameters to observables.

        Updated to handle:
        - Computational category (BLAST, Rosetta, etc.)
        - Cumulative acceleration effects (H5 insight)
        - Better handling of extreme accelerations (PCR)
        """
        predictions = {}

        # Determine base acceleration scale based on category
        if shift.category == ShiftCategory.CAPABILITY_EXTENSION:
            base_accel = params['capability_accel_scale']
        elif shift.category == ShiftCategory.METHODOLOGICAL:
            base_accel = params['methodological_accel_scale']
        elif shift.category == ShiftCategory.COMPUTATIONAL:
            base_accel = params.get('computational_accel_scale', params['capability_accel_scale'])
        else:
            base_accel = params['capability_accel_scale'] * 0.5  # Conceptual shifts

        # Apply modifiers based on shift characteristics

        # Infrastructure-heavy shifts get boost
        infrastructure_shifts = [
            "Human Genome Project / Sequencing",
            "DNA Sequencing Revolution",
            "GenBank/Databases",
            "X-ray Crystallography",
        ]
        if shift.name in infrastructure_shifts:
            base_accel *= params['infrastructure_boost']

        # Democratizing technologies get additional impact
        if shift.key_insight and "democratization" in shift.key_insight.lower():
            base_accel *= params['democratization_factor']

        # Foundational technologies get extra boost (they enabled everything after)
        foundational_shifts = ["PCR", "Recombinant DNA", "BLAST Algorithm"]
        if shift.name in foundational_shifts:
            base_accel *= params.get('cumulative_factor', 1.2) ** 2

        # Computational tools that build on each other (cumulative effect)
        # Later computational tools benefit from earlier infrastructure
        cumulative_order = {
            "GenBank/Databases": 1,
            "BLAST Algorithm": 2,
            "Rosetta Structure Prediction": 3,
            "Machine Learning in Biology": 4,
        }
        if shift.name in cumulative_order:
            order = cumulative_order[shift.name]
            base_accel *= params.get('cumulative_factor', 1.2) ** (order - 1)

        # Time acceleration (log scale)
        predictions['time_acceleration'] = np.log10(max(1.0, base_accel))

        # Publication increase (roughly proportional to time accel, with noise)
        # Foundational technologies have higher publication impact
        pub_factor = 0.7
        if shift.name in foundational_shifts:
            pub_factor = 0.85  # Higher publication impact for foundational tech
        predictions['publication_increase'] = np.log10(max(1.0, base_accel ** pub_factor))

        # Transformation years (inverse relationship with acceleration)
        base_transform_time = 30.0  # Base years for transformation

        # Modern technologies transform faster due to communication/globalization
        if shift.start_year > 1980:
            base_transform_time = 15.0
        if shift.start_year > 2000:
            base_transform_time = 10.0

        predictions['transformation_years'] = base_transform_time / (
            np.log10(base_accel + 1) * params['time_to_impact_scale']
        )

        return predictions

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Calculate log-likelihood of observed data given parameters.

        Uses Gaussian likelihood with heteroscedastic errors.
        """
        ll = 0.0

        for target in self.targets:
            shift = self.shifts[target.shift_name]
            predictions = self.predict_metrics(shift, params)

            predicted = predictions[target.metric_name]
            observed = target.observed_value
            sigma = target.uncertainty

            # Gaussian log-likelihood
            residual = (observed - predicted) / sigma
            ll += -0.5 * (residual ** 2 + np.log(2 * np.pi * sigma ** 2))
            ll *= target.weight

        return ll

    def log_prior(self, params: Dict[str, float]) -> float:
        """Calculate log-prior probability of parameters."""
        lp = 0.0

        for param_name, value in params.items():
            if param_name not in self.CALIBRATION_PARAMS:
                continue

            prior_mean, prior_std, bounds = self.CALIBRATION_PARAMS[param_name]

            # Check bounds
            if value < bounds[0] or value > bounds[1]:
                return -np.inf

            # Gaussian prior
            lp += -0.5 * ((value - prior_mean) / prior_std) ** 2

        return lp

    def log_posterior(self, params: Dict[str, float]) -> float:
        """Calculate log-posterior (unnormalized) probability."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array for optimization."""
        return np.array([params[k] for k in sorted(self.CALIBRATION_PARAMS.keys())])

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert array back to parameter dict."""
        keys = sorted(self.CALIBRATION_PARAMS.keys())
        return {k: v for k, v in zip(keys, arr)}

    def calibrate_mle(self) -> CalibrationResult:
        """
        Perform Maximum Likelihood Estimation of parameters.

        Uses gradient descent to find MLE, with uncertainty from numerical Hessian.
        """
        # Initial guess from priors
        x0 = np.array([
            self.CALIBRATION_PARAMS[k][0]
            for k in sorted(self.CALIBRATION_PARAMS.keys())
        ])

        # Bounds
        bounds = [
            self.CALIBRATION_PARAMS[k][2]
            for k in sorted(self.CALIBRATION_PARAMS.keys())
        ]

        # Objective function (negative log-posterior for minimization)
        def objective(x):
            params = self._array_to_params(x)
            return -self.log_posterior(params)

        # Simple gradient descent optimization
        x = x0.copy()
        lr = 0.01  # Learning rate
        best_x = x.copy()
        best_obj = objective(x)

        for iteration in range(2000):
            # Numerical gradient
            grad = self._numerical_gradient(x, objective)

            # Gradient descent step
            x_new = x - lr * grad

            # Apply bounds
            for i, (low, high) in enumerate(bounds):
                x_new[i] = np.clip(x_new[i], low, high)

            obj_new = objective(x_new)

            # Track best
            if obj_new < best_obj:
                best_obj = obj_new
                best_x = x_new.copy()

            # Adaptive learning rate
            if obj_new < objective(x):
                x = x_new
                lr *= 1.05
            else:
                lr *= 0.5

            lr = np.clip(lr, 1e-6, 0.1)

            # Convergence check
            if iteration > 100 and np.linalg.norm(grad) < 1e-6:
                break

        # Extract MLE parameters
        mle_params = self._array_to_params(best_x)

        # Estimate uncertainties from Hessian (inverse of Fisher information)
        try:
            # Numerical Hessian
            hess = self._numerical_hessian(result.x, objective)
            if np.all(np.linalg.eigvals(hess) > 0):  # Check positive definite
                cov = np.linalg.inv(hess)
                uncertainties = {
                    k: np.sqrt(max(0, cov[i, i]))
                    for i, k in enumerate(sorted(self.CALIBRATION_PARAMS.keys()))
                }
            else:
                # Fall back to prior uncertainties if Hessian is not positive definite
                uncertainties = {
                    k: self.CALIBRATION_PARAMS[k][1]
                    for k in self.CALIBRATION_PARAMS
                }
        except Exception:
            uncertainties = {
                k: self.CALIBRATION_PARAMS[k][1]
                for k in self.CALIBRATION_PARAMS
            }

        # Calculate fit quality metrics
        ll = self.log_likelihood(mle_params)
        n_params = len(self.CALIBRATION_PARAMS)
        n_targets = len(self.targets)

        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_targets) - 2 * ll

        # Calculate residuals for each target
        residuals = {}
        abs_pct_errors = []

        for target in self.targets:
            shift = self.shifts[target.shift_name]
            predictions = self.predict_metrics(shift, mle_params)

            predicted = predictions[target.metric_name]
            observed = target.observed_value
            sigma = target.uncertainty

            resid = (observed - predicted) / sigma
            residuals[f"{target.shift_name}_{target.metric_name}"] = resid

            # MAPE (in original units for interpretability)
            if abs(observed) > 0.01:
                pct_err = abs((predicted - observed) / observed)
                abs_pct_errors.append(pct_err)

        mape = np.mean(abs_pct_errors) if abs_pct_errors else 0.0

        return CalibrationResult(
            parameters=mle_params,
            parameter_uncertainties=uncertainties,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            target_residuals=residuals,
            target_mape=mape,
        )

    def _numerical_gradient(
        self,
        x: np.ndarray,
        func: Callable,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Calculate numerical gradient."""
        n = len(x)
        grad = np.zeros(n)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)

        return grad

    def _numerical_hessian(
        self,
        x: np.ndarray,
        func: Callable,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Calculate numerical Hessian matrix."""
        n = len(x)
        hess = np.zeros((n, n))
        f0 = func(x)

        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps

                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps

                hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps ** 2)
                hess[j, i] = hess[i, j]

        return hess

    def calibrate_mcmc(
        self,
        n_samples: int = 5000,
        n_burnin: int = 1000,
        thin: int = 2,
    ) -> CalibrationResult:
        """
        Perform Markov Chain Monte Carlo sampling of posterior.

        Uses adaptive Metropolis-Hastings algorithm.
        """
        # Start from MLE
        mle_result = self.calibrate_mle()
        current_params = mle_result.parameters.copy()
        current_lp = self.log_posterior(current_params)

        # Proposal covariance (start from prior uncertainties)
        prop_scales = {
            k: self.CALIBRATION_PARAMS[k][1] * 0.5
            for k in self.CALIBRATION_PARAMS
        }

        # Storage for samples
        samples = []
        accepted = 0

        # MCMC iterations
        total_iter = n_burnin + n_samples * thin

        for i in range(total_iter):
            # Propose new parameters
            proposed = {}
            for k in current_params:
                proposed[k] = current_params[k] + self.rng.normal(0, prop_scales[k])

            # Calculate acceptance probability
            proposed_lp = self.log_posterior(proposed)

            if np.isfinite(proposed_lp):
                log_alpha = proposed_lp - current_lp

                if np.log(self.rng.random()) < log_alpha:
                    current_params = proposed
                    current_lp = proposed_lp
                    accepted += 1

            # Store sample after burn-in, with thinning
            if i >= n_burnin and (i - n_burnin) % thin == 0:
                samples.append(list(current_params.values()))

            # Adaptive proposal scale (every 100 iterations during burn-in)
            if i < n_burnin and i > 0 and i % 100 == 0:
                acceptance_rate = accepted / i
                # Target 23% acceptance for optimal exploration
                if acceptance_rate > 0.3:
                    for k in prop_scales:
                        prop_scales[k] *= 1.2
                elif acceptance_rate < 0.15:
                    for k in prop_scales:
                        prop_scales[k] *= 0.8

        samples = np.array(samples)

        # Calculate posterior statistics
        param_names = list(current_params.keys())
        param_means = {k: np.mean(samples[:, i]) for i, k in enumerate(param_names)}
        param_stds = {k: np.std(samples[:, i]) for i, k in enumerate(param_names)}

        # Calculate fit quality using posterior mean
        ll = self.log_likelihood(param_means)
        n_params = len(self.CALIBRATION_PARAMS)
        n_targets = len(self.targets)

        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_targets) - 2 * ll

        # Residuals
        residuals = {}
        abs_pct_errors = []

        for target in self.targets:
            shift = self.shifts[target.shift_name]
            predictions = self.predict_metrics(shift, param_means)

            predicted = predictions[target.metric_name]
            observed = target.observed_value
            sigma = target.uncertainty

            resid = (observed - predicted) / sigma
            residuals[f"{target.shift_name}_{target.metric_name}"] = resid

            if abs(observed) > 0.01:
                pct_err = abs((predicted - observed) / observed)
                abs_pct_errors.append(pct_err)

        mape = np.mean(abs_pct_errors) if abs_pct_errors else 0.0

        return CalibrationResult(
            parameters=param_means,
            parameter_uncertainties=param_stds,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            target_residuals=residuals,
            target_mape=mape,
            posterior_samples=samples,
        )

    def validate_model(
        self,
        calibrated_params: Dict[str, float],
        holdout_shift: str = "crispr",
    ) -> Dict[str, float]:
        """
        Validate calibration by predicting held-out shift.

        Args:
            calibrated_params: Parameters calibrated on other shifts
            holdout_shift: Name of shift to predict

        Returns:
            Dictionary with prediction errors
        """
        shift = self.shifts[holdout_shift]
        predictions = self.predict_metrics(shift, calibrated_params)

        results = {
            'shift_name': shift.name,
            'time_accel_predicted': 10 ** predictions['time_acceleration'],
            'time_accel_observed': shift.time_acceleration,
            'time_accel_error': abs(predictions['time_acceleration'] - np.log10(shift.time_acceleration)),
            'pub_increase_predicted': 10 ** predictions['publication_increase'],
            'pub_increase_observed': shift.publication_increase,
            'pub_increase_error': abs(predictions['publication_increase'] - np.log10(shift.publication_increase)),
            'transform_years_predicted': predictions['transformation_years'],
            'transform_years_observed': shift.transformation_years,
            'transform_years_error': abs(predictions['transformation_years'] - shift.transformation_years),
        }

        return results

    def cross_validate(self) -> Dict[str, Dict]:
        """
        Perform leave-one-out cross-validation.

        Returns predictions for each shift when calibrated on the others.
        """
        results = {}
        shift_names = list(self.shifts.keys())

        for holdout in shift_names:
            # Create calibrator without holdout
            reduced_shifts = {k: v for k, v in self.shifts.items() if k != holdout}
            reduced_calibrator = HistoricalCalibrator(shifts=reduced_shifts)

            # Calibrate on reduced set
            calib_result = reduced_calibrator.calibrate_mle()

            # Validate on holdout
            results[holdout] = self.validate_model(calib_result.parameters, holdout)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive calibration report."""
        # Run full calibration
        mle_result = self.calibrate_mle()

        lines = [
            "=" * 70,
            "HISTORICAL CALIBRATION REPORT",
            "AI Research Acceleration Model v0.2",
            "=" * 70,
            "",
            "1. HISTORICAL SHIFTS USED FOR CALIBRATION",
            "-" * 40,
        ]

        for name, shift in self.shifts.items():
            lines.extend([
                f"\n{shift.name}",
                f"  Category: {shift.category.value}",
                f"  Period: {shift.start_year} - {shift.start_year + shift.full_impact_years}",
                f"  Time Acceleration: {shift.time_acceleration:,.0f}x",
                f"  Publication Increase: {shift.publication_increase:.0f}x",
                f"  Transformation Years: {shift.transformation_years}",
                f"  Key Insight: {shift.key_insight}",
            ])

        lines.extend([
            "",
            "2. CALIBRATED PARAMETERS (MLE)",
            "-" * 40,
        ])

        for param, value in mle_result.parameters.items():
            unc = mle_result.parameter_uncertainties.get(param, 0)
            lines.append(f"  {param}: {value:.3f} ± {unc:.3f}")

        lines.extend([
            "",
            "3. MODEL FIT QUALITY",
            "-" * 40,
            f"  Log-Likelihood: {mle_result.log_likelihood:.2f}",
            f"  AIC: {mle_result.aic:.2f}",
            f"  BIC: {mle_result.bic:.2f}",
            f"  Mean Absolute % Error: {mle_result.target_mape:.1%}",
            "",
            "4. INDIVIDUAL RESIDUALS (in σ units)",
            "-" * 40,
        ])

        for target, resid in sorted(mle_result.target_residuals.items()):
            status = "✓" if abs(resid) < 2 else "⚠" if abs(resid) < 3 else "✗"
            lines.append(f"  {status} {target}: {resid:+.2f}σ")

        # Cross-validation
        lines.extend([
            "",
            "5. LEAVE-ONE-OUT CROSS-VALIDATION",
            "-" * 40,
        ])

        cv_results = self.cross_validate()
        for holdout, results in cv_results.items():
            lines.extend([
                f"\n  Holdout: {results['shift_name']}",
                f"    Time Accel: {results['time_accel_observed']:.0f}x (obs) vs {results['time_accel_predicted']:.0f}x (pred)",
                f"    Pub Increase: {results['pub_increase_observed']:.0f}x (obs) vs {results['pub_increase_predicted']:.0f}x (pred)",
                f"    Transform Years: {results['transform_years_observed']:.0f}yr (obs) vs {results['transform_years_predicted']:.1f}yr (pred)",
            ])

        lines.extend([
            "",
            "6. IMPLICATIONS FOR AI PROJECTIONS",
            "-" * 40,
            "",
            "  Based on historical calibration, the model suggests:",
            "",
            f"  - Capability extensions can achieve {mle_result.parameters['capability_accel_scale']:.0f}x acceleration",
            f"  - Methodological shifts achieve {mle_result.parameters['methodological_accel_scale']:.0f}x acceleration",
            f"  - Democratization provides {mle_result.parameters['democratization_factor']:.1f}x additional boost",
            f"  - Infrastructure investment provides {mle_result.parameters['infrastructure_boost']:.1f}x boost",
            "",
            "  AI as capability extension (like HGP, AlphaFold) could achieve:",
            f"    - {mle_result.parameters['capability_accel_scale'] * mle_result.parameters['infrastructure_boost']:.0f}x with infrastructure investment",
            f"    - {mle_result.parameters['capability_accel_scale'] * mle_result.parameters['democratization_factor']:.0f}x with democratization",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


def run_calibration_demo():
    """Run a demonstration of the historical calibration."""
    print("Starting Historical Calibration...")
    print()

    calibrator = HistoricalCalibrator()

    # MLE calibration
    print("Running Maximum Likelihood Estimation...")
    mle_result = calibrator.calibrate_mle()
    print(mle_result.summary())
    print()

    # Generate full report
    report = calibrator.generate_report()
    print(report)

    return mle_result


if __name__ == "__main__":
    run_calibration_demo()
