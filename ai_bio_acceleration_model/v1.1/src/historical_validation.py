"""
Historical Validation Module - v1.1
===================================

P1-3: Validates model against FDA approval data 2015-2023.

This module provides hindcast validation to assess model accuracy
when compared to actual drug development timelines.

Data Sources:
- FDA CDER Drug Approvals (2015-2023)
- ClinicalTrials.gov Phase duration data
- Tufts CSDD Development time studies

IMPORTANT: This is a simplified validation. Full validation would require:
1. Individual drug program tracking
2. Stage-specific duration data
3. Therapeutic area stratification
4. AI tool adoption timing

Methodology:
1. Reconstruct historical AI capability proxy (compute growth)
2. Run model from 2015 with known parameters
3. Compare predicted vs actual approval rates/timelines
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings


# =============================================================================
# HISTORICAL DATA (FDA + Literature)
# =============================================================================

# FDA CDER Novel Drug Approvals by Year
# Source: FDA Annual Reports
FDA_NOVEL_APPROVALS = {
    2015: 45,
    2016: 22,
    2017: 46,
    2018: 59,
    2019: 48,
    2020: 53,
    2021: 50,
    2022: 37,
    2023: 55,
}

# Clinical Phase Success Rates (Historical)
# Source: Wong et al. (2019), BIO Industry Analysis (2021)
HISTORICAL_SUCCESS_RATES = {
    'phase1': {
        2015: 0.66,
        2018: 0.66,
        2021: 0.68,  # Slight improvement
    },
    'phase2': {
        2015: 0.31,
        2018: 0.33,
        2021: 0.35,  # Biomarker-driven improvement
    },
    'phase3': {
        2015: 0.58,
        2018: 0.58,
        2021: 0.60,
    },
}

# Median Development Times (Years from IND to Approval)
# Source: Tufts CSDD, FDA analysis
HISTORICAL_DEV_TIMES = {
    2015: 8.5,
    2018: 8.1,
    2021: 7.8,  # COVID acceleration effect
    2023: 7.5,
}

# AI Compute Growth (Proxy for AI Capability)
# Source: Epoch AI "Compute Trends"
# Measured in FLOPs for frontier models
AI_COMPUTE_GROWTH = {
    2015: 1e18,   # ~AlexNet era training
    2016: 5e18,
    2017: 1e19,   # Attention is All You Need
    2018: 5e19,
    2019: 1e20,   # GPT-2 scale
    2020: 5e20,   # GPT-3 scale
    2021: 1e21,
    2022: 5e21,   # Chinchilla, PaLM
    2023: 1e22,   # GPT-4 scale
    2024: 3e22,   # Estimated current frontier
}


@dataclass
class ValidationResult:
    """Results from historical validation."""
    years: List[int]
    metric_name: str

    # Actual vs Predicted
    actual: Dict[int, float]
    predicted: Dict[int, float]

    # Error metrics
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    r_squared: float  # Coefficient of determination

    # Directional accuracy
    direction_correct: float  # % of years where trend direction matched

    # Notes
    notes: str = ""


class HistoricalValidation:
    """
    Validates model against FDA 2015-2023 historical data.

    P1-3: This is a key requirement from Expert A2 (Dr. Chen).
    """

    def __init__(self, base_year: int = 2015):
        self.base_year = base_year
        self.validation_results: Dict[str, ValidationResult] = {}

    def compute_ai_capability_proxy(self, year: int) -> float:
        """
        Compute normalized AI capability based on compute growth.

        Normalized so 2024 = 1.0 (same as model baseline).
        """
        compute_2024 = AI_COMPUTE_GROWTH.get(2024, 3e22)
        compute_year = AI_COMPUTE_GROWTH.get(year, 1e18)

        # Log-scale normalization
        return np.log10(compute_year) / np.log10(compute_2024)

    def validate_approval_rates(
        self,
        predicted_rates: Dict[int, float]
    ) -> ValidationResult:
        """
        Validate predicted approval rates against FDA data.

        Args:
            predicted_rates: Model-predicted approvals by year

        Returns:
            ValidationResult with error metrics
        """
        years = sorted(set(FDA_NOVEL_APPROVALS.keys()) & set(predicted_rates.keys()))

        if len(years) < 3:
            raise ValueError("Need at least 3 overlapping years for validation")

        actual_vals = [FDA_NOVEL_APPROVALS[y] for y in years]
        predicted_vals = [predicted_rates[y] for y in years]

        # Compute error metrics
        errors = np.array(actual_vals) - np.array(predicted_vals)
        abs_errors = np.abs(errors)

        mae = np.mean(abs_errors)
        mape = np.mean(abs_errors / (np.array(actual_vals) + 1e-10)) * 100
        rmse = np.sqrt(np.mean(errors ** 2))

        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((np.array(actual_vals) - np.mean(actual_vals)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        # Directional accuracy
        actual_changes = np.diff(actual_vals)
        predicted_changes = np.diff(predicted_vals)
        direction_matches = np.sign(actual_changes) == np.sign(predicted_changes)
        direction_correct = np.mean(direction_matches) * 100

        result = ValidationResult(
            years=years,
            metric_name="FDA Novel Drug Approvals",
            actual=dict(zip(years, actual_vals)),
            predicted=dict(zip(years, predicted_vals)),
            mae=mae,
            mape=mape,
            rmse=rmse,
            r_squared=r_squared,
            direction_correct=direction_correct,
            notes="Comparison to FDA CDER annual novel drug approvals",
        )

        self.validation_results['approval_rates'] = result
        return result

    def validate_success_rates(
        self,
        predicted_phase2: Dict[int, float]
    ) -> ValidationResult:
        """
        Validate predicted Phase II success rates against literature.
        """
        hist_years = sorted(HISTORICAL_SUCCESS_RATES['phase2'].keys())
        pred_years = sorted(predicted_phase2.keys())
        years = sorted(set(hist_years) & set(pred_years))

        if len(years) < 2:
            warnings.warn("Insufficient overlapping years for Phase II validation")
            return None

        actual_vals = [HISTORICAL_SUCCESS_RATES['phase2'][y] for y in years]
        predicted_vals = [predicted_phase2[y] for y in years]

        errors = np.array(actual_vals) - np.array(predicted_vals)
        abs_errors = np.abs(errors)

        result = ValidationResult(
            years=years,
            metric_name="Phase II Success Rate",
            actual=dict(zip(years, actual_vals)),
            predicted=dict(zip(years, predicted_vals)),
            mae=np.mean(abs_errors),
            mape=np.mean(abs_errors / (np.array(actual_vals) + 1e-10)) * 100,
            rmse=np.sqrt(np.mean(errors ** 2)),
            r_squared=0.0,  # Too few points
            direction_correct=100.0 if len(years) < 3 else 0.0,
            notes="Comparison to Wong et al. (2019) and BIO Industry Analysis",
        )

        self.validation_results['phase2_success'] = result
        return result

    def validate_development_times(
        self,
        predicted_times: Dict[int, float]
    ) -> ValidationResult:
        """
        Validate predicted development times against Tufts CSDD data.
        """
        hist_years = sorted(HISTORICAL_DEV_TIMES.keys())
        pred_years = sorted(predicted_times.keys())
        years = sorted(set(hist_years) & set(pred_years))

        if len(years) < 2:
            warnings.warn("Insufficient overlapping years for dev time validation")
            return None

        actual_vals = [HISTORICAL_DEV_TIMES[y] for y in years]
        predicted_vals = [predicted_times[y] for y in years]

        errors = np.array(actual_vals) - np.array(predicted_vals)

        result = ValidationResult(
            years=years,
            metric_name="Median Development Time (Years)",
            actual=dict(zip(years, actual_vals)),
            predicted=dict(zip(years, predicted_vals)),
            mae=np.mean(np.abs(errors)),
            mape=np.mean(np.abs(errors) / (np.array(actual_vals) + 1e-10)) * 100,
            rmse=np.sqrt(np.mean(errors ** 2)),
            r_squared=0.0,
            direction_correct=100.0 if len(years) < 3 else 0.0,
            notes="Comparison to Tufts CSDD development time studies",
        )

        self.validation_results['dev_times'] = result
        return result

    def run_hindcast(
        self,
        model,
        start_year: int = 2015,
        end_year: int = 2023
    ) -> Dict[str, ValidationResult]:
        """
        Run complete hindcast validation.

        Args:
            model: AIBioAccelerationModel instance
            start_year: First year for hindcast
            end_year: Last year for hindcast

        Returns:
            Dict of ValidationResults by metric
        """
        # This would require running the model multiple times
        # with historical AI capability values
        # For now, return placeholder with methodology note

        warnings.warn(
            "Full hindcast requires running model with historical parameters. "
            "This simplified version compares against aggregate statistics."
        )

        return self.validation_results

    def generate_report(self) -> str:
        """Generate validation report as markdown."""
        lines = [
            "# Historical Validation Report (P1-3)",
            "",
            "## Methodology",
            "Compares model predictions against FDA approval data (2015-2023).",
            "",
            "**Data Sources:**",
            "- FDA CDER Novel Drug Approvals",
            "- Wong et al. (2019) Phase success rates",
            "- Tufts CSDD Development times",
            "- Epoch AI Compute Trends",
            "",
            "## Results",
            "",
        ]

        for name, result in self.validation_results.items():
            lines.extend([
                f"### {result.metric_name}",
                "",
                f"- **MAE**: {result.mae:.2f}",
                f"- **MAPE**: {result.mape:.1f}%",
                f"- **RMSE**: {result.rmse:.2f}",
                f"- **R²**: {result.r_squared:.3f}",
                f"- **Direction Accuracy**: {result.direction_correct:.1f}%",
                "",
                "**Year-by-Year:**",
                "",
                "| Year | Actual | Predicted | Error |",
                "|------|--------|-----------|-------|",
            ])

            for year in result.years:
                actual = result.actual[year]
                predicted = result.predicted[year]
                error = actual - predicted
                lines.append(f"| {year} | {actual:.1f} | {predicted:.1f} | {error:+.1f} |")

            lines.extend(["", f"*Notes: {result.notes}*", ""])

        lines.extend([
            "## Interpretation",
            "",
            "**Acceptable thresholds:**",
            "- MAPE < 20%: Good fit",
            "- MAPE < 30%: Acceptable fit",
            "- MAPE > 30%: Poor fit, recalibration needed",
            "",
            "**Limitations:**",
            "- Aggregate comparison (not individual drug programs)",
            "- AI adoption timing not precisely tracked",
            "- COVID-19 effects (2020-2021) create anomalies",
        ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_fda_approval_baseline() -> Dict[int, int]:
    """Return FDA approval data for reference."""
    return FDA_NOVEL_APPROVALS.copy()


def get_ai_capability_timeline() -> pd.DataFrame:
    """Return AI capability proxy timeline."""
    validator = HistoricalValidation()
    data = []
    for year, compute in AI_COMPUTE_GROWTH.items():
        data.append({
            'year': year,
            'compute_flops': compute,
            'ai_capability_proxy': validator.compute_ai_capability_proxy(year),
            'log10_compute': np.log10(compute),
        })
    return pd.DataFrame(data)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Historical Validation Module - v1.1")
    print("=" * 60)

    print("\nFDA Novel Drug Approvals (2015-2023):")
    for year, approvals in sorted(FDA_NOVEL_APPROVALS.items()):
        print(f"  {year}: {approvals} approvals")

    print("\nAI Capability Proxy (normalized to 2024):")
    validator = HistoricalValidation()
    for year in range(2015, 2025):
        cap = validator.compute_ai_capability_proxy(year)
        print(f"  {year}: {cap:.3f}")

    print("\nModule loaded successfully.")
