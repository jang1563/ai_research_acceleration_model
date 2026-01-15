#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model

This script runs the complete v0.3 model pipeline:
1. Initializes model with default parameters (10-stage pipeline)
2. Runs all scenarios (Pessimistic, Baseline, Optimistic)
3. Generates all outputs (data, figures, summary)
4. Runs sensitivity analysis on key parameters
5. Identifies highest-leverage parameters for policy
6. Monte Carlo uncertainty quantification (NEW in v0.3.1)

Key Changes in v0.3.1:
- Monte Carlo simulation for uncertainty propagation
- Confidence interval computation (5th, 25th, 50th, 75th, 95th percentiles)
- Uncertainty band visualizations
- Distribution histograms for key milestones

Usage:
    python run_model.py [--skip-sensitivity] [--skip-monte-carlo] [--mc-samples N]

Outputs:
    - outputs/results.csv: Complete model results
    - outputs/parameters.json: Model parameters
    - outputs/summary.txt: Summary statistics
    - outputs/fig*.png/pdf: Publication-quality figures
    - outputs/sensitivity_summary.csv: Sensitivity analysis results
    - outputs/fig_tornado.png: Tornado diagram
    - outputs/monte_carlo_*.csv: Monte Carlo simulation results
    - outputs/confidence_intervals_*.csv: Confidence intervals
    - outputs/fig_uncertainty_bands_*.png: Uncertainty band plots
    - outputs/fig_histogram_2050_*.png: Distribution histograms

Version: 0.3.1

References:
- Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer"
- Morgan & Henrion (1990) "Uncertainty: A Guide to Dealing with Uncertainty"
- Helton & Davis (2003) "Latin hypercube sampling and uncertainty propagation"
"""

import os
import sys
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, ModelConfig, run_default_model
from visualize import ModelVisualizer, generate_all_visualizations


def main(run_sensitivity: bool = True, run_monte_carlo: bool = True, mc_samples: int = 1000):
    """Run the complete model pipeline.

    Parameters
    ----------
    run_sensitivity : bool
        Whether to run sensitivity analysis
    run_monte_carlo : bool
        Whether to run Monte Carlo uncertainty analysis
    mc_samples : int
        Number of Monte Carlo samples
    """

    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.3.1 (Monte Carlo Uncertainty)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize and Run Model
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Initializing model...")
    
    model = AIBioAccelerationModel()
    
    print(f"  - Time horizon: {model.config.t0} to {model.config.T}")
    print(f"  - Number of stages: {model.n_stages}")
    print(f"  - Number of scenarios: {len(model.config.scenarios)}")
    
    print("\n[Step 2/4] Running scenarios...")
    
    results = model.run_all_scenarios()
    
    for scenario in model.config.scenarios:
        print(f"  - {scenario.name} (g = {scenario.g_ai}): Complete")
    
    # -------------------------------------------------------------------------
    # Step 2: Export Data
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Exporting data...")
    
    # Export results
    results_path = os.path.join(output_dir, 'results.csv')
    model.export_results(results_path)
    print(f"  - Results: {results_path}")
    
    # Export parameters
    params_path = os.path.join(output_dir, 'parameters.json')
    model.export_parameters(params_path)
    print(f"  - Parameters: {params_path}")
    
    # Export summary
    summary = model.get_summary_statistics()
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("AI-Accelerated Biological Discovery Model - Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Version: 0.2\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Configuration\n")
        f.write("-" * 60 + "\n")
        f.write(f"Time horizon: {model.config.t0} - {model.config.T}\n")
        f.write(f"Time step: {model.config.dt} year(s)\n\n")
        
        f.write("Pipeline Stages\n")
        f.write("-" * 60 + "\n")
        for stage in model.config.stages:
            f.write(f"S{stage.index}: {stage.name}\n")
            f.write(f"    Baseline duration: {stage.tau_baseline} months\n")
            f.write(f"    Max AI multiplier: {stage.M_max}x\n")
            f.write(f"    Success probability: {stage.p_success:.0%}\n")
            f.write(f"    Saturation rate: {stage.k_saturation}\n\n")
        
        f.write("Scenarios\n")
        f.write("-" * 60 + "\n")
        for scenario in model.config.scenarios:
            f.write(f"{scenario.name}: g = {scenario.g_ai}\n")
            f.write(f"    {scenario.description}\n\n")
        
        f.write("Summary Statistics\n")
        f.write("-" * 60 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("Bottleneck Transitions (Baseline)\n")
        f.write("-" * 60 + "\n")
        transitions = model.get_bottleneck_transitions('Baseline')
        if len(transitions) > 0:
            f.write(transitions.to_string(index=False))
        else:
            f.write("No bottleneck transitions detected\n")
        f.write("\n\n")
        
        f.write("Key Findings\n")
        f.write("-" * 60 + "\n")
        
        baseline_summary = summary[summary['scenario'] == 'Baseline'].iloc[0]
        
        f.write(f"By 2030 (Baseline): {baseline_summary['progress_by_2030']:.1f} equivalent years\n")
        f.write(f"By 2040 (Baseline): {baseline_summary['progress_by_2040']:.1f} equivalent years\n")
        f.write(f"By 2050 (Baseline): {baseline_summary['progress_by_2050']:.1f} equivalent years\n\n")
        
        f.write(f"Maximum progress rate achieved: {baseline_summary['max_progress_rate']:.1f}x\n")
        f.write(f"Final bottleneck stage: S{baseline_summary['final_bottleneck']}\n")
    
    print(f"  - Summary: {summary_path}")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Generating visualizations...")
    
    figures = generate_all_visualizations(model, results, output_dir)
    
    for fig_name in figures.keys():
        print(f"  - {fig_name}.png/.pdf")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Model Run Complete")
    print("=" * 70)
    
    print("\nKey Results (Baseline Scenario):")
    print("-" * 40)
    
    baseline_df = results[results['scenario'] == 'Baseline']
    
    # Progress by decade
    for year in [2030, 2040, 2050]:
        progress = baseline_df[baseline_df['year'] == year]['cumulative_progress'].iloc[0]
        calendar_years = year - 2024
        ratio = progress / calendar_years
        print(f"  By {year}: {progress:.1f} equiv. years ({calendar_years} calendar → {ratio:.1f}x)")
    
    # Bottleneck info
    print("\nBottleneck Transitions:")
    print("-" * 40)
    transitions = model.get_bottleneck_transitions('Baseline')
    if len(transitions) > 0:
        for _, row in transitions.iterrows():
            print(f"  {int(row['year'])}: S{int(row['from_stage'])} ({row['from_name']}) → S{int(row['to_stage'])} ({row['to_name']})")
    else:
        initial_bottleneck = baseline_df['bottleneck_stage'].iloc[0]
        print(f"  No transitions: S{int(initial_bottleneck)} remains bottleneck throughout")
    
    print("\nScenario Comparison (Equiv. Years by 2050):")
    print("-" * 40)
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]
        progress = df[df['year'] == 2050]['cumulative_progress'].iloc[0]
        print(f"  {scenario:12}: {progress:6.1f} years")

    print(f"\nAll outputs saved to: {output_dir}/")

    # -------------------------------------------------------------------------
    # Step 5: Sensitivity Analysis (v0.3)
    # -------------------------------------------------------------------------
    if run_sensitivity:
        print("\n" + "=" * 70)
        print("[Step 5/5] Running Sensitivity Analysis...")
        print("=" * 70)

        from sensitivity import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer()

        # Run OAT analysis on key parameters (M_max and p_success for clinical phases)
        print("\nAnalyzing key parameters (this may take a few minutes)...")

        sensitivity_summary = analyzer.run_full_oat_analysis(
            parameters=['M_max', 'p_success'],
            stages=[6, 7, 8, 9],  # Focus on clinical phases and regulatory
            scenario_name='Baseline',
            variation_range=(0.5, 1.5),
            n_points=7
        )

        # Save sensitivity summary
        sensitivity_summary.to_csv(
            os.path.join(output_dir, 'sensitivity_summary.csv'),
            index=False
        )
        print(f"\n  Sensitivity summary saved to: {output_dir}/sensitivity_summary.csv")

        # Generate tornado diagram
        print("  Generating tornado diagram...")
        analyzer.plot_tornado_diagram(
            top_n=10,
            save_path=os.path.join(output_dir, 'fig_tornado.png')
        )
        print(f"  Tornado diagram saved to: {output_dir}/fig_tornado.png")

        # Show top recommendations
        print("\n" + "-" * 70)
        print("Highest Leverage Parameters for Policy Intervention:")
        print("-" * 70)
        recommendations = analyzer.identify_highest_leverage_parameters(top_n=5)
        for i, (param, sens, rec) in enumerate(recommendations, 1):
            print(f"\n  {i}. {param} (sensitivity = {sens:.3f})")
            print(f"     → {rec}")

        import matplotlib.pyplot as plt
        plt.close('all')
    else:
        print("\n[Skipping sensitivity analysis]")

    # -------------------------------------------------------------------------
    # Step 6: Monte Carlo Uncertainty Analysis (v0.3.1)
    # -------------------------------------------------------------------------
    if run_monte_carlo:
        print("\n" + "=" * 70)
        print("[Step 6/6] Running Monte Carlo Uncertainty Analysis...")
        print("=" * 70)

        from uncertainty import MonteCarloUncertainty

        # Run for all scenarios
        for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic']:
            print(f"\n--- {scenario_name} Scenario ---")

            mc = MonteCarloUncertainty(model=model, n_samples=mc_samples)

            # Add default uncertainties with expert-elicited CVs
            # CV values based on Morgan & Henrion (1990) elicitation guidelines
            mc.add_default_uncertainties(
                scenario_name=scenario_name,
                cv_M_max=0.15,      # ±15% for AI acceleration potential
                cv_p_success=0.10,  # ±10% for clinical trial success rates
                cv_g_ai=0.20        # ±20% for AI progress rate
            )

            # Run simulation
            mc.run_simulation(scenario_name)

            # Get summary
            summary = mc.get_summary_statistics(scenario_name)
            print("\nKey Milestones (90% CI):")
            for _, row in summary.iterrows():
                print(f"  {row['year']}: {row['p50']:.1f} [{row['p5']:.1f}, {row['p95']:.1f}]")

            # Export results
            mc.export_results(
                os.path.join(output_dir, f'monte_carlo_{scenario_name.lower()}.csv'),
                scenario_name
            )

            # Export confidence intervals
            ci = mc.compute_confidence_intervals(scenario_name)
            ci.to_csv(
                os.path.join(output_dir, f'confidence_intervals_{scenario_name.lower()}.csv'),
                index=False
            )

            # Generate plots
            import matplotlib.pyplot as plt

            fig1 = mc.plot_uncertainty_bands(scenario_name)
            fig1.savefig(
                os.path.join(output_dir, f'fig_uncertainty_bands_{scenario_name.lower()}.png'),
                dpi=300, bbox_inches='tight'
            )
            fig1.savefig(
                os.path.join(output_dir, f'fig_uncertainty_bands_{scenario_name.lower()}.pdf'),
                bbox_inches='tight'
            )
            plt.close(fig1)

            fig2 = mc.plot_histogram(scenario_name, year=2050)
            fig2.savefig(
                os.path.join(output_dir, f'fig_histogram_2050_{scenario_name.lower()}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig2)

            print(f"  Outputs saved for {scenario_name}")

        print("\nMonte Carlo analysis complete!")
    else:
        print("\n[Skipping Monte Carlo analysis]")

    print("\n" + "=" * 70)
    print("Model Run Complete")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip sensitivity analysis (faster)')
    parser.add_argument('--skip-monte-carlo', action='store_true',
                        help='Skip Monte Carlo uncertainty analysis')
    parser.add_argument('--mc-samples', type=int, default=1000,
                        help='Number of Monte Carlo samples (default: 1000)')
    args = parser.parse_args()

    model, results = main(
        run_sensitivity=not args.skip_sensitivity,
        run_monte_carlo=not args.skip_monte_carlo,
        mc_samples=args.mc_samples
    )
