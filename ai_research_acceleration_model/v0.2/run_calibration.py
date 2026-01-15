#!/usr/bin/env python3
"""
Historical Calibration Runner
=============================

Runs the historical calibration analysis and generates reports/visualizations.

Usage:
    python run_calibration.py                    # Run full calibration
    python run_calibration.py --mle              # MLE only (fast)
    python run_calibration.py --mcmc             # MCMC with posterior samples
    python run_calibration.py --cv               # Cross-validation analysis
    python run_calibration.py --report           # Generate full report
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from historical_calibration import (
    HistoricalCalibrator,
    HISTORICAL_SHIFTS,
    EXTENDED_HISTORICAL_SHIFTS,
    run_calibration_demo,
)

# Also add v0.1 for model comparison
sys.path.insert(0, str(Path(__file__).parent.parent / "v0.1"))
from src.model import AIResearchAccelerationModel, Scenario


def run_mle_calibration():
    """Run MLE calibration and display results."""
    print("=" * 60)
    print("MAXIMUM LIKELIHOOD ESTIMATION")
    print("=" * 60)
    print()

    calibrator = HistoricalCalibrator()
    result = calibrator.calibrate_mle()

    print(result.summary())
    return result


def run_mcmc_calibration(n_samples: int = 5000):
    """Run MCMC calibration with posterior sampling."""
    print("=" * 60)
    print("MCMC POSTERIOR SAMPLING")
    print("=" * 60)
    print()

    calibrator = HistoricalCalibrator()

    print(f"Running MCMC with {n_samples} samples...")
    print("(This may take a few minutes)")
    print()

    result = calibrator.calibrate_mcmc(n_samples=n_samples)

    print(result.summary())
    print()

    if result.posterior_samples is not None:
        print("Posterior Sample Statistics:")
        param_names = list(result.parameters.keys())
        for i, name in enumerate(param_names):
            samples = result.posterior_samples[:, i]
            q05, q50, q95 = np.percentile(samples, [5, 50, 95])
            print(f"  {name}:")
            print(f"    Median: {q50:.3f}")
            print(f"    90% CI: [{q05:.3f}, {q95:.3f}]")
        print()

    return result


def run_cross_validation():
    """Run leave-one-out cross-validation."""
    print("=" * 60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 60)
    print()

    calibrator = HistoricalCalibrator()
    cv_results = calibrator.cross_validate()

    print("Prediction accuracy when each shift is held out:")
    print()

    for holdout, results in cv_results.items():
        print(f"{results['shift_name']}:")
        print(f"  Time Acceleration:")
        print(f"    Observed:  {results['time_accel_observed']:,.0f}x")
        print(f"    Predicted: {results['time_accel_predicted']:,.0f}x")
        print(f"    Log Error: {results['time_accel_error']:.2f}")
        print(f"  Publication Increase:")
        print(f"    Observed:  {results['pub_increase_observed']:.0f}x")
        print(f"    Predicted: {results['pub_increase_predicted']:.0f}x")
        print(f"    Log Error: {results['pub_increase_error']:.2f}")
        print()

    return cv_results


def compare_with_v01_model():
    """Compare historical predictions with v0.1 model outputs."""
    print("=" * 60)
    print("COMPARISON WITH v0.1 MODEL PREDICTIONS")
    print("=" * 60)
    print()

    # Calibrate
    calibrator = HistoricalCalibrator()
    calib_result = calibrator.calibrate_mle()

    # v0.1 model predictions
    model = AIResearchAccelerationModel(scenario=Scenario.BASELINE)

    print("Historical vs Model Comparison:")
    print()

    # Project what model predicts for AI
    forecasts = model.forecast([2025, 2030, 2035, 2040, 2050])

    print("v0.1 Model AI Acceleration Projections:")
    for year, data in forecasts.items():
        print(f"  {year}: {data['acceleration']:.1f}x")

    print()
    print("Historical Calibration Insights:")
    cap_scale = calib_result.parameters['capability_accel_scale']
    infra_boost = calib_result.parameters['infrastructure_boost']
    demo_factor = calib_result.parameters['democratization_factor']

    print(f"  - Historical capability extensions achieved ~{cap_scale:.0f}x")
    print(f"  - With infrastructure investment: ~{cap_scale * infra_boost:.0f}x")
    print(f"  - With democratization: ~{cap_scale * demo_factor:.0f}x")
    print()

    # Consistency check
    print("Model vs Historical Consistency:")
    model_2050 = forecasts[2050]['acceleration']
    hist_benchmark = cap_scale * infra_boost  # HGP-like scenario

    if model_2050 < hist_benchmark * 0.1:
        print(f"  ⚠ Model prediction ({model_2050:.0f}x) much lower than HGP benchmark ({hist_benchmark:.0f}x)")
        print("  → Physical constraints dominate in AI scenario")
    elif model_2050 > hist_benchmark * 10:
        print(f"  ⚠ Model prediction ({model_2050:.0f}x) much higher than HGP benchmark ({hist_benchmark:.0f}x)")
        print("  → May need to reconsider assumptions")
    else:
        print(f"  ✓ Model prediction ({model_2050:.0f}x) in range of historical benchmarks ({hist_benchmark:.0f}x)")


def generate_full_report():
    """Generate comprehensive calibration report."""
    print("Generating Full Calibration Report...")
    print()

    calibrator = HistoricalCalibrator()
    report = calibrator.generate_report()

    print(report)

    # Save to file
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print()
    print(f"Report saved to: {report_path}")

    return report


def save_calibration_results():
    """Save calibration results to JSON for future use."""
    calibrator = HistoricalCalibrator()
    result = calibrator.calibrate_mle()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'parameters': result.parameters,
        'parameter_uncertainties': result.parameter_uncertainties,
        'log_likelihood': result.log_likelihood,
        'aic': result.aic,
        'bic': result.bic,
        'target_mape': result.target_mape,
        'target_residuals': result.target_residuals,
    }

    json_path = output_dir / "calibration_results.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {json_path}")
    return output


def run_extended_calibration():
    """Run calibration with extended dataset (13 technologies)."""
    print("=" * 60)
    print("EXTENDED CALIBRATION (13 Technologies)")
    print("Includes: Molecular Biology + Computational Biology")
    print("=" * 60)
    print()

    print("Technologies in extended set:")
    for name, shift in EXTENDED_HISTORICAL_SHIFTS.items():
        cat = shift.category.value[:4].upper()
        print(f"  [{cat}] {shift.name}: {shift.time_acceleration:,.0f}x ({shift.start_year})")
    print()

    calibrator = HistoricalCalibrator(shifts=EXTENDED_HISTORICAL_SHIFTS)
    result = calibrator.calibrate_mle()

    print(result.summary())
    print()

    # Key insight from computational biology historian
    print("KEY INSIGHT (from H5 reviewer):")
    print("-" * 40)
    print("AI acceleration (~10x over ML) is consistent with")
    print("historical computational progression:")
    print()
    print("  Manual → Databases → BLAST → ML → Deep Learning")
    print("    1x      ~100x      ~1000x  ~10x    ~10x")
    print()
    print("Each step ~10x marginal acceleration.")
    print("AI is the latest in a 50-year trajectory, not a discontinuity.")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Historical Calibration for AI Research Acceleration Model"
    )
    parser.add_argument('--mle', action='store_true', help='Run MLE calibration')
    parser.add_argument('--mcmc', action='store_true', help='Run MCMC calibration')
    parser.add_argument('--mcmc-samples', type=int, default=5000, help='Number of MCMC samples')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--compare', action='store_true', help='Compare with v0.1 model')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    parser.add_argument('--extended', action='store_true', help='Use extended 13-technology dataset')

    args = parser.parse_args()

    # Default: run everything
    if not any([args.mle, args.mcmc, args.cv, args.compare, args.report, args.save, args.extended]):
        args.mle = True
        args.cv = True
        args.compare = True
        args.report = True
        args.save = True

    if args.extended:
        run_extended_calibration()
        print()

    if args.mle:
        run_mle_calibration()
        print()

    if args.mcmc:
        import numpy as np
        run_mcmc_calibration(args.mcmc_samples)
        print()

    if args.cv:
        run_cross_validation()
        print()

    if args.compare:
        compare_with_v01_model()
        print()

    if args.report:
        generate_full_report()
        print()

    if args.save:
        save_calibration_results()


if __name__ == "__main__":
    main()
