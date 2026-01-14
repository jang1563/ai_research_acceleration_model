#!/usr/bin/env python3
"""
AI-Accelerated Scientific Research Model - Main Runner
=======================================================

Run forecasts and simulations for the AI research acceleration model.

Usage:
    python run_model.py                    # Quick forecast
    python run_model.py --monte-carlo 1000 # Full MC simulation
    python run_model.py --all-scenarios    # Compare all scenarios
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ResearchPipeline
from src.paradigm_shift import ParadigmShiftModule
from src.model import AIResearchAccelerationModel, Scenario
from src.simulation import MonteCarloSimulator


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AI-Accelerated Scientific Research Model v0.1            ║
║     Forecasting Research Acceleration 2025-2050              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_quick_forecast(scenario: Scenario = Scenario.BASELINE):
    """Run a quick deterministic forecast."""
    print(f"\n=== Quick Forecast: {scenario.value.upper()} Scenario ===\n")

    model = AIResearchAccelerationModel(scenario=scenario)

    # Print model config
    summary = model.summary()
    print(f"AI Growth Rate (g): {summary['scenario_params']['g_ai']}")
    print(f"Baseline Pipeline Duration: {summary['baseline_duration_months']:.1f} months")
    print()

    # Forecast table
    years = [2025, 2030, 2035, 2040, 2045, 2050]
    print(f"{'Year':<8} {'AI Cap':<10} {'Accel':<10} {'Duration':<12} {'Discoveries':<12} {'Bottleneck'}")
    print(f"{'-'*70}")

    for year in years:
        t = year - 2025
        forecasts = model.forecast([year])
        f = forecasts[year]

        print(f"{year:<8} {f['ai_capability']:<10.2f} {f['acceleration']:<10.2f}x "
              f"{f['duration_months']:<12.1f} {f['discoveries_per_year']:<12,.0f} {f['bottleneck']}")

    # Stage breakdown at 2050
    print(f"\n=== Stage Breakdown at 2050 ===\n")
    t = 25
    A = model.ai_capability(t)

    print(f"{'Stage':<30} {'Baseline':<10} {'Effective':<12} {'Multiplier':<12}")
    print(f"{'-'*64}")

    for stage in model.pipeline.stages:
        duration = model.stage_duration(stage, t)
        M = model.effective_multiplier_with_constraints(stage, t, A)
        print(f"{stage.name:<30} {stage.params.tau_0:<10.1f} {duration:<12.2f} {M:<12.2f}x")


def run_scenario_comparison():
    """Compare all scenarios."""
    print("\n=== Scenario Comparison (2050 Projections) ===\n")

    print(f"{'Scenario':<15} {'g_ai':<8} {'Acceleration':<14} {'Discoveries/yr':<15} {'Bottleneck'}")
    print(f"{'-'*70}")

    for scenario in Scenario:
        model = AIResearchAccelerationModel(scenario=scenario)
        forecasts = model.forecast([2050])
        f = forecasts[2050]

        print(f"{scenario.value:<15} {model.scenario_params.g_ai:<8.2f} "
              f"{f['acceleration']:<14.2f}x {f['discoveries_per_year']:<15,.0f} {f['bottleneck']}")


def run_unlock_comparison():
    """Compare scenarios with and without simulation unlock."""
    print("\n=== Simulation Unlock Impact Analysis ===\n")
    print("The Unlock: AI invents simulation tools to replace physical trials\n")

    years = [2025, 2030, 2035, 2040, 2050]

    model_unlock = AIResearchAccelerationModel(scenario=Scenario.BASELINE, enable_unlock=True)
    model_no_unlock = AIResearchAccelerationModel(scenario=Scenario.BASELINE, enable_unlock=False)

    print(f"{'Year':<8} {'No Unlock':<12} {'With Unlock':<14} {'Boost':<10} {'P(unlock)'}")
    print(f"{'-'*54}")

    for year in years:
        t = year - 2025
        accel_no = model_no_unlock.acceleration_factor(t)
        accel_yes = model_unlock.acceleration_factor(t)
        p_unlock = model_unlock.simulation_unlock.p_unlock(t)
        boost = accel_yes / accel_no

        print(f"{year:<8} {accel_no:<12.1f}x {accel_yes:<14.1f}x {boost:<10.2f}x {p_unlock:.0%}")

    print("\n=== Physical Stage Impact at 2050 ===\n")
    t = 25
    A = model_unlock.ai_capability(t)

    print(f"{'Stage':<30} {'Without':<12} {'With':<12} {'Unlock Boost'}")
    print(f"{'-'*66}")

    for stage in model_unlock.pipeline.stages:
        M_no = model_no_unlock.effective_multiplier_with_constraints(stage, t, A)
        M_yes = model_unlock.effective_multiplier_with_constraints(stage, t, A)
        is_physical = stage.id in model_unlock.simulation_unlock.affected_stages
        label = " (physical)" if is_physical else ""

        print(f"{stage.name + label:<30} {M_no:<12.1f}x {M_yes:<12.1f}x {M_yes/M_no:.2f}x")


def run_monte_carlo(n_samples: int, scenario: Scenario = Scenario.BASELINE):
    """Run Monte Carlo simulation."""
    print(f"\n=== Monte Carlo Simulation ===")
    print(f"Scenario: {scenario.value}")
    print(f"Samples: {n_samples}")
    print()

    sim = MonteCarloSimulator(
        scenario=scenario,
        n_samples=n_samples,
        seed=42,
        years=[2025, 2030, 2035, 2040, 2045, 2050]
    )

    summary = sim.run()
    sim.print_summary()

    # Sensitivity analysis
    if n_samples >= 100:
        print("\n=== Parameter Sensitivity (Sobol Indices) ===\n")
        sensitivities = sim.sobol_sensitivity()

        # Sort and show top 10
        sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])
        print(f"{'Parameter':<40} {'Sensitivity':<12}")
        print(f"{'-'*52}")
        for param, sens in sorted_sens[:10]:
            print(f"{param:<40} {sens:<12.4f}")

    # Save results
    output_dir = Path(__file__).parent / "outputs" / f"mc_{scenario.value}_{n_samples}"
    sim.save_results(str(output_dir))
    print(f"\nResults saved to: {output_dir}")


def run_pipeline_analysis():
    """Analyze the research pipeline in detail."""
    print("\n=== Pipeline Analysis ===\n")

    pipeline = ResearchPipeline()

    print("8-Stage Research Pipeline:")
    print(f"{'ID':<5} {'Stage':<30} {'Duration':<10} {'Type':<12} {'M_max (spd/qual)':<18}")
    print(f"{'-'*80}")

    for stage in pipeline.stages:
        print(f"{stage.id:<5} {stage.name:<30} {stage.params.tau_0:<10.1f} "
              f"{stage.params.stage_type.value:<12} "
              f"{stage.params.M_max_speed:.0f} / {stage.params.M_max_quality:.0f}")

    print(f"\nTotal Baseline Duration: {pipeline.total_baseline_duration():.1f} months")
    print(f"                       = {pipeline.total_baseline_duration()/12:.1f} years")


def run_psm_analysis():
    """Analyze the Paradigm Shift Module."""
    print("\n=== Paradigm Shift Module Analysis ===\n")

    psm = ParadigmShiftModule()

    print("Shift Types and Parameters:")
    print(f"{'Type':<15} {'Weight':<10} {'95% CI':<15} {'Threshold':<12} {'Saturation'}")
    print(f"{'-'*65}")

    for shift_type, params in psm.shift_params.items():
        print(f"{shift_type.value:<15} {params.weight:<10.2f} "
              f"[{params.weight_ci[0]:.2f}, {params.weight_ci[1]:.2f}]    "
              f"{params.threshold:<12.1f} {params.saturation:.1f}")

    print("\nPSM Values at Different AI Capability Levels:")
    print(f"{'A(t)':<10} {'PSM':<10} {'Type I':<12} {'Type II':<12} {'Type III'}")
    print(f"{'-'*56}")

    for A in [1.0, 2.0, 3.0, 5.0, 10.0, 15.0]:
        total = psm.calculate_psm(A)
        type_i = psm.impact_function(psm.shift_params[list(psm.shift_params.keys())[0]].__class__.__bases__[0], A) if hasattr(psm, 'impact_function') else 0

        summary = psm.summary(A)
        contributions = [summary['by_type'][st.value]['contribution'] for st in psm.shift_params.keys()]

        print(f"{A:<10.1f} {total:<10.2f} {contributions[0]:<12.2f} {contributions[1]:<12.2f} {contributions[2]:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Accelerated Scientific Research Model"
    )
    parser.add_argument(
        '--monte-carlo', '-mc',
        type=int,
        metavar='N',
        help='Run Monte Carlo simulation with N samples'
    )
    parser.add_argument(
        '--scenario', '-s',
        choices=['ai_winter', 'conservative', 'baseline', 'ambitious'],
        default='baseline',
        help='Scenario to use (default: baseline)'
    )
    parser.add_argument(
        '--all-scenarios',
        action='store_true',
        help='Compare all scenarios'
    )
    parser.add_argument(
        '--pipeline',
        action='store_true',
        help='Show pipeline analysis'
    )
    parser.add_argument(
        '--psm',
        action='store_true',
        help='Show Paradigm Shift Module analysis'
    )
    parser.add_argument(
        '--unlock',
        action='store_true',
        help='Compare simulation unlock scenarios (AI replacing physical trials)'
    )

    args = parser.parse_args()

    # Map scenario string to enum
    scenario_map = {
        'ai_winter': Scenario.AI_WINTER,
        'conservative': Scenario.CONSERVATIVE,
        'baseline': Scenario.BASELINE,
        'ambitious': Scenario.AMBITIOUS,
    }
    scenario = scenario_map[args.scenario]

    print_banner()

    if args.pipeline:
        run_pipeline_analysis()
    elif args.psm:
        run_psm_analysis()
    elif args.all_scenarios:
        run_scenario_comparison()
    elif args.unlock:
        run_unlock_comparison()
    elif args.monte_carlo:
        run_monte_carlo(args.monte_carlo, scenario)
    else:
        run_quick_forecast(scenario)


if __name__ == "__main__":
    main()
