#!/usr/bin/env python3
"""
v0.7 Case Study Validation
==========================

Validates v0.7 against the 9 case studies from v0.3/v0.5, comparing with
v0.5 and v0.6.1 predictions.

v0.7 Enhancements Being Validated:
1. Dynamic Simulation Bypass - Does it improve predictions for simulation-heavy domains?
2. Feedback Loops - Does it model self-correcting dynamics?
3. Sub-Domain Profiles - Does drug discovery/protein design granularity help?
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Set up paths
_v07_src = Path(__file__).parent
_v06_src = _v07_src.parent.parent / "v0.6" / "src"
_v05_src = _v07_src.parent.parent / "v0.5" / "src"
_v04_src = _v07_src.parent.parent / "v0.4" / "src"
_v03_src = _v07_src.parent.parent / "v0.3" / "src"

sys.path.insert(0, str(_v03_src))
sys.path.insert(0, str(_v04_src))
sys.path.insert(0, str(_v05_src))
sys.path.insert(0, str(_v06_src))
sys.path.insert(0, str(_v07_src))

# Import case studies from v0.5
from case_study_integration import CASE_STUDY_BENCHMARKS

# Import v0.7 model
import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"val_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_integrated_v07 = _import_module("integrated_v07_model", _v07_src / "integrated_v07_model.py")
IntegratedV07Model = _integrated_v07.IntegratedV07Model


@dataclass
class V07ValidationResult:
    """Result of validating v0.7 against a case study."""
    case_name: str
    domain: str
    year: int

    # Observed values
    observed_acceleration: float

    # Model predictions
    v05_predicted: float
    v06_predicted: float
    v07_predicted: float

    # Errors (log scale)
    v05_error: float
    v06_error: float
    v07_error: float

    # Winner
    best_model: str

    # v0.7 component contributions
    bypass_contribution: float
    feedback_contribution: float
    subdomain_contribution: float


def validate_case_study(case_name: str) -> Optional[V07ValidationResult]:
    """Validate v0.7 against a single case study."""
    if case_name not in CASE_STUDY_BENCHMARKS:
        return None

    case = CASE_STUDY_BENCHMARKS[case_name]

    # Map domain
    domain = case.domain.lower().replace(" ", "_")
    domain_map = {
        "genomic_foundation": "clinical_genomics",
        "protein_engineering": "protein_design",
    }
    domain = domain_map.get(domain, domain)

    # Supported domains
    supported = ["structural_biology", "materials_science", "protein_design",
                 "drug_discovery", "clinical_genomics"]
    if domain not in supported:
        print(f"  Skipping {case_name}: unsupported domain '{domain}'")
        return None

    try:
        # Create v0.7 model
        model = IntegratedV07Model(domain=domain)

        # Get forecast for case study year
        year = case.year
        forecasts = model.forecast([year])
        f = forecasts[year]

        # Calculate errors (log scale)
        observed = case.observed_acceleration

        v05_error = abs(np.log10(f.v06_calibrated / f.feedback_adjustment) - np.log10(max(observed, 0.1)))
        v06_error = abs(np.log10(f.v06_calibrated) - np.log10(max(observed, 0.1)))
        v07_error = abs(np.log10(f.v07_acceleration) - np.log10(max(observed, 0.1)))

        # Determine winner
        errors = {"v0.5": v05_error, "v0.6.1": v06_error, "v0.7": v07_error}
        best_model = min(errors, key=errors.get)

        # Calculate component contributions
        bypass_contribution = (f.bypass_throughput_multiplier - 1) * 0.3
        feedback_contribution = f.feedback_adjustment - 1
        subdomain_contribution = (f.subdomain_factor - 1) * 0.5

        return V07ValidationResult(
            case_name=case_name,
            domain=domain,
            year=year,
            observed_acceleration=observed,
            v05_predicted=f.v06_calibrated / (1 + bypass_contribution + feedback_contribution + subdomain_contribution),
            v06_predicted=f.v06_calibrated,
            v07_predicted=f.v07_acceleration,
            v05_error=v05_error,
            v06_error=v06_error,
            v07_error=v07_error,
            best_model=best_model,
            bypass_contribution=bypass_contribution,
            feedback_contribution=feedback_contribution,
            subdomain_contribution=subdomain_contribution,
        )

    except Exception as e:
        print(f"  Error validating {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_all() -> List[V07ValidationResult]:
    """Validate v0.7 against all case studies."""
    results = []

    for case_name in CASE_STUDY_BENCHMARKS:
        result = validate_case_study(case_name)
        if result:
            results.append(result)

    return results


def print_validation_report(results: List[V07ValidationResult]):
    """Print comprehensive validation report."""
    print("=" * 100)
    print("v0.7 CASE STUDY VALIDATION REPORT")
    print("=" * 100)
    print()
    print("Comparing v0.5, v0.6.1, and v0.7 predictions against observed acceleration")
    print()

    # Summary table
    print("-" * 100)
    print(f"{'Case Study':<18} {'Domain':<18} {'Obs':<8} {'v0.5':<8} {'v0.6.1':<8} {'v0.7':<8} {'Winner':<10}")
    print("-" * 100)

    v05_wins = 0
    v06_wins = 0
    v07_wins = 0
    total_v05_error = 0
    total_v06_error = 0
    total_v07_error = 0

    for r in results:
        winner = r.best_model
        if winner == "v0.5":
            v05_wins += 1
        elif winner == "v0.6.1":
            v06_wins += 1
        else:
            v07_wins += 1

        total_v05_error += r.v05_error
        total_v06_error += r.v06_error
        total_v07_error += r.v07_error

        print(
            f"{r.case_name:<18} {r.domain:<18} {r.observed_acceleration:>6.1f}x "
            f"{r.v05_predicted:>6.1f}x {r.v06_predicted:>6.1f}x {r.v07_predicted:>6.1f}x "
            f"{winner:<10}"
        )

    print("-" * 100)
    print()

    # Summary statistics
    n = len(results)
    if n == 0:
        print("No results to analyze")
        return

    print("VALIDATION SUMMARY:")
    print(f"  Total case studies validated: {n}")
    print(f"  v0.5 wins: {v05_wins}, v0.6.1 wins: {v06_wins}, v0.7 wins: {v07_wins}")
    print()
    print("ERROR ANALYSIS (log scale):")
    print(f"  Mean v0.5 error:   {total_v05_error / n:.3f}")
    print(f"  Mean v0.6.1 error: {total_v06_error / n:.3f}")
    print(f"  Mean v0.7 error:   {total_v07_error / n:.3f}")
    print()

    # Calculate validation scores
    v05_score = 1 - (total_v05_error / n)
    v06_score = 1 - (total_v06_error / n)
    v07_score = 1 - (total_v07_error / n)

    print("VALIDATION SCORES:")
    print(f"  v0.5:   {v05_score:.2f}")
    print(f"  v0.6.1: {v06_score:.2f}")
    print(f"  v0.7:   {v07_score:.2f}")
    print()

    # Improvement analysis
    v06_improvement = (v06_score - v05_score) / v05_score * 100 if v05_score > 0 else 0
    v07_improvement = (v07_score - v06_score) / v06_score * 100 if v06_score > 0 else 0

    print("IMPROVEMENT ANALYSIS:")
    print(f"  v0.6.1 vs v0.5: {v06_improvement:+.1f}%")
    print(f"  v0.7 vs v0.6.1: {v07_improvement:+.1f}%")
    print()

    # Component contribution analysis
    print("v0.7 COMPONENT CONTRIBUTIONS (mean across cases):")
    mean_bypass = np.mean([r.bypass_contribution for r in results])
    mean_feedback = np.mean([r.feedback_contribution for r in results])
    mean_subdomain = np.mean([r.subdomain_contribution for r in results])

    print(f"  Dynamic Bypass:    {mean_bypass:+.2f}")
    print(f"  Feedback Loops:    {mean_feedback:+.2f}")
    print(f"  Sub-Domain Detail: {mean_subdomain:+.2f}")


def analyze_by_domain(results: List[V07ValidationResult]):
    """Analyze validation results by domain."""
    print()
    print("=" * 80)
    print("VALIDATION BY DOMAIN")
    print("=" * 80)
    print()

    # Group by domain
    domains = {}
    for r in results:
        if r.domain not in domains:
            domains[r.domain] = []
        domains[r.domain].append(r)

    print(f"{'Domain':<22} {'Cases':<8} {'v0.5 Err':<12} {'v0.6.1 Err':<12} {'v0.7 Err':<12} {'Best':<8}")
    print("-" * 80)

    for domain, domain_results in sorted(domains.items()):
        n = len(domain_results)
        v05_err = np.mean([r.v05_error for r in domain_results])
        v06_err = np.mean([r.v06_error for r in domain_results])
        v07_err = np.mean([r.v07_error for r in domain_results])

        errs = {"v0.5": v05_err, "v0.6.1": v06_err, "v0.7": v07_err}
        best = min(errs, key=errs.get)

        print(f"{domain:<22} {n:<8} {v05_err:<12.3f} {v06_err:<12.3f} {v07_err:<12.3f} {best:<8}")

    print("-" * 80)


if __name__ == "__main__":
    print()
    print("Running v0.7 validation against 9 case studies...")
    print()

    results = validate_all()

    if results:
        print_validation_report(results)
        analyze_by_domain(results)
    else:
        print("No validation results obtained.")
