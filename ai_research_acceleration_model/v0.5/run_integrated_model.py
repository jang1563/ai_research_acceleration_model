#!/usr/bin/env python3
"""
v0.5 Integrated Model Runner
============================

CLI interface for the integrated AI + automation acceleration model.

Usage:
    python run_integrated_model.py                     # All domains baseline
    python run_integrated_model.py --domain materials_science
    python run_integrated_model.py --automation        # Automation scenarios
    python run_integrated_model.py --matrix            # Full scenario matrix
    python run_integrated_model.py --compare           # v0.4 vs v0.5 comparison
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "v0.4" / "src"))

from autonomous_lab import (
    AutonomousLabModel,
    AutomationScenario,
    LAB_CAPACITIES,
    run_automation_comparison,
)
from integrated_model import (
    IntegratedAccelerationModel,
    run_integrated_comparison,
)
from refined_model import (
    RefinedAccelerationModel,
    DOMAIN_PROFILES,
    Scenario,
)


def run_all_domains(ai_scenario: str = "baseline", auto_scenario: str = "baseline"):
    """Run integrated model for all domains."""
    print("=" * 70)
    print("INTEGRATED ACCELERATION MODEL v0.5 - ALL DOMAINS")
    print("=" * 70)
    print()
    print(f"AI Scenario: {ai_scenario}")
    print(f"Automation Scenario: {auto_scenario}")
    print()

    ai_enum = Scenario[ai_scenario.upper()]
    auto_enum = AutomationScenario[auto_scenario.upper()]

    years = [2025, 2030, 2040, 2050]
    domains = list(DOMAIN_PROFILES.keys())

    # Header
    print(f"{'Domain':<22} ", end="")
    for year in years:
        print(f"{year:>10}", end="")
    print()
    print("-" * 70)

    results = {}
    for domain in domains:
        model = IntegratedAccelerationModel(
            domain=domain,
            ai_scenario=ai_enum,
            automation_scenario=auto_enum,
        )
        forecasts = model.forecast(years)

        print(f"{domain:<22} ", end="")
        for year in years:
            accel = forecasts[year].end_to_end_acceleration
            print(f"{accel:>9.1f}x", end="")
        print()

        results[domain] = {
            year: forecasts[year].end_to_end_acceleration
            for year in years
        }

    print("-" * 70)

    # Show v0.4 comparison
    print()
    print("Improvement over v0.4 (no lab automation):")
    print("-" * 50)

    for domain in domains[:3]:
        v04 = RefinedAccelerationModel(domain=domain).forecast([2050])[2050]["acceleration"]
        v05 = results[domain][2050]
        improvement = (v05 / v04 - 1) * 100
        print(f"  {domain}: {v04:.1f}x → {v05:.1f}x (+{improvement:.0f}%)")

    return results


def run_single_domain(
    domain: str,
    ai_scenario: str = "baseline",
    auto_scenario: str = "baseline"
):
    """Detailed analysis for a single domain."""
    ai_enum = Scenario[ai_scenario.upper()]
    auto_enum = AutomationScenario[auto_scenario.upper()]

    model = IntegratedAccelerationModel(
        domain=domain,
        ai_scenario=ai_enum,
        automation_scenario=auto_enum,
    )

    print(model.summary())
    print()

    # Stage breakdown
    years = [2025, 2030, 2040, 2050]
    forecasts = model.forecast(years)

    print("STAGE-LEVEL BREAKDOWN (2030):")
    print("-" * 60)
    f = forecasts[2030]
    for stage_id, accel in f.stage_accelerations.items():
        stage_name = model.STAGES[stage_id][0]
        is_cog = "cognitive" if model.STAGES[stage_id][1] else "physical"
        print(f"  {stage_id} ({is_cog}): {stage_name:<35} {accel:.1f}x")

    print()
    print("COST PROJECTION:")
    print("-" * 60)
    for year in years:
        f = forecasts[year]
        print(f"  {year}: ${f.cost_per_project:,.0f} ({f.cost_reduction*100:.0f}% reduction)")

    return forecasts


def run_scenario_matrix(domain: str = "average_biology"):
    """Generate full scenario matrix."""
    print("=" * 70)
    print(f"SCENARIO MATRIX - {domain.upper()}")
    print("=" * 70)
    print()

    years = [2030, 2050]

    for year in years:
        print(f"Year: {year}")
        print("-" * 60)

        # Header
        header = "AI \\ Auto"
        print(f"{header:<15}", end="")
        for auto_scen in AutomationScenario:
            print(f"{auto_scen.value:>12}", end="")
        print()

        # Rows
        for ai_scen in Scenario:
            print(f"{ai_scen.value:<15}", end="")
            for auto_scen in AutomationScenario:
                model = IntegratedAccelerationModel(
                    domain=domain,
                    ai_scenario=ai_scen,
                    automation_scenario=auto_scen,
                )
                accel = model.forecast([year])[year].end_to_end_acceleration
                print(f"{accel:>11.1f}x", end="")
            print()

        print()

    print("KEY: Rows = AI scenario, Columns = Automation scenario")
    print()
    print("INSIGHT: The 'breakthrough' automation scenario has the largest")
    print("impact because it removes the physical bottleneck entirely.")


def compare_v04_v05():
    """Compare v0.4 and v0.5 predictions."""
    print("=" * 70)
    print("MODEL COMPARISON: v0.4 vs v0.5")
    print("=" * 70)
    print()

    domains = list(DOMAIN_PROFILES.keys())
    years = [2025, 2030, 2040, 2050]

    print("v0.4: AI only (physical bottleneck at 1-1.5x)")
    print("v0.5: AI + Lab Automation (physical up to 5-50x)")
    print()

    print(f"{'Domain':<22} {'Version':<10} ", end="")
    for year in years:
        print(f"{year:>8}", end="")
    print()
    print("-" * 70)

    for domain in domains:
        # v0.4
        v04 = RefinedAccelerationModel(domain=domain)
        v04_forecast = v04.forecast(years)

        print(f"{domain:<22} {'v0.4':<10} ", end="")
        for year in years:
            print(f"{v04_forecast[year]['acceleration']:>7.1f}x", end="")
        print()

        # v0.5
        v05 = IntegratedAccelerationModel(domain=domain)
        v05_forecast = v05.forecast(years)

        print(f"{'':<22} {'v0.5':<10} ", end="")
        for year in years:
            print(f"{v05_forecast[year].end_to_end_acceleration:>7.1f}x", end="")
        print()

        # Improvement
        print(f"{'':<22} {'Δ':<10} ", end="")
        for year in years:
            v04_a = v04_forecast[year]['acceleration']
            v05_a = v05_forecast[year].end_to_end_acceleration
            delta = (v05_a / v04_a - 1) * 100
            print(f"{delta:>+6.0f}%", end="")
        print()
        print()

    print("-" * 70)
    print()
    print("SUMMARY:")
    print("  v0.4 2050 projection: ~2-3x (physical bottleneck)")
    print("  v0.5 2050 projection: ~5-20x (with automation)")
    print()
    print("  Lab automation is the key unlock for higher acceleration.")
    print("  Without it, cognitive AI gains are wasted on backlog.")


def save_results(results: dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"v05_results_{timestamp}.json"

    # Convert any non-serializable objects
    def serialize(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=serialize)

    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Acceleration Model v0.5 Runner"
    )
    parser.add_argument(
        "--domain", "-d",
        choices=list(DOMAIN_PROFILES.keys()),
        help="Specific domain to analyze"
    )
    parser.add_argument(
        "--ai-scenario", "-a",
        choices=["conservative", "baseline", "optimistic"],
        default="baseline",
        help="AI capability scenario"
    )
    parser.add_argument(
        "--auto-scenario", "-u",
        choices=["conservative", "baseline", "optimistic", "breakthrough"],
        default="baseline",
        help="Lab automation scenario"
    )
    parser.add_argument(
        "--automation",
        action="store_true",
        help="Run automation-only analysis"
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Generate full scenario matrix"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare v0.4 vs v0.5"
    )
    parser.add_argument(
        "--save", "-o",
        action="store_true",
        help="Save results to JSON"
    )

    args = parser.parse_args()

    results = {}

    if args.automation:
        run_automation_comparison()
    elif args.matrix:
        domain = args.domain or "average_biology"
        run_scenario_matrix(domain)
    elif args.compare:
        compare_v04_v05()
    elif args.domain:
        results = run_single_domain(args.domain, args.ai_scenario, args.auto_scenario)
    else:
        results = run_all_domains(args.ai_scenario, args.auto_scenario)

    if args.save and results:
        output_dir = Path(__file__).parent / "outputs"
        save_results(results, output_dir)


if __name__ == "__main__":
    main()
