#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model v0.7

Key Changes in v0.7:
- Pipeline Iteration Module: Models failure/rework dynamics
- Amodei Scenario: Optimistic upper-bound (10x acceleration target)
- Semi-Markov process for stage transitions
- Comparison with Amodei's "Machines of Loving Grace" predictions

Usage:
    python run_model.py [--skip-comparison] [--skip-pipeline-iteration]

Version: 0.7
"""

import os
import sys
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, ModelConfig, Scenario, TherapeuticArea
from data_quality import DataQualityModule, DataQualityConfig
from pipeline_iteration import PipelineIterationModule, PipelineIterationConfig


def main(run_comparison: bool = True, run_pipeline_iteration: bool = True):
    """Run the complete model pipeline."""

    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.7 (Pipeline Iteration + Amodei Scenario)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run Full Model with All Features
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Running full model (all features enabled)...")

    model = AIBioAccelerationModel()

    print(f"  - Time horizon: {model.config.t0} to {model.config.T}")
    print(f"  - Pipeline iteration: {model.config.enable_pipeline_iteration}")
    print(f"  - Data quality: {model.config.enable_data_quality}")
    print(f"  - Scenarios: {len(model.config.scenarios)}")

    results = model.run_all_scenarios()

    # -------------------------------------------------------------------------
    # Step 2: Key Results Summary
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Computing summary statistics...")

    summary = model.get_summary_statistics()

    print("\nScenario Summary (2050 Progress):")
    print("-" * 70)
    for _, row in summary.iterrows():
        amodei_flag = " [AMODEI]" if row['is_amodei'] else ""
        print(f"  {row['scenario']:25s}: {row['progress_by_2050']:6.1f} equiv years "
              f"({row['progress_by_2050']/26:.1f}x acceleration){amodei_flag}")

    # -------------------------------------------------------------------------
    # Step 3: Amodei Comparison
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Comparing with Amodei's predictions...")

    amodei_comparison = model.compare_with_amodei()

    print("\nAmodei Comparison (Target: 10x = 50-100 years in 10 years):")
    print("-" * 70)
    print(f"{'Scenario':<25} {'10yr Progress':>12} {'10yr Accel':>10} {'Meets Target':>12}")
    print("-" * 70)
    for _, row in amodei_comparison.iterrows():
        meets = "YES" if row['meets_amodei_low'] else "No"
        print(f"{row['scenario']:<25} {row['progress_10yr']:>12.1f} {row['acceleration_10yr']:>10.1f}x {meets:>12}")

    # -------------------------------------------------------------------------
    # Step 4: Pipeline Iteration Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Analyzing pipeline iteration dynamics...")

    if run_pipeline_iteration:
        pi_module = PipelineIterationModule()

        # Get baseline p_success values
        baseline_results = results[results['scenario'] == 'Baseline']
        p_success_2024 = {i+1: baseline_results[baseline_results['year'] == 2024][f'p_{i+1}'].iloc[0]
                         for i in range(10)}
        p_success_2050 = {i+1: baseline_results[baseline_results['year'] == 2050][f'p_{i+1}'].iloc[0]
                         for i in range(10)}

        _, stats_2024 = pi_module.compute_effective_throughput(100.0, p_success_2024)
        _, stats_2050 = pi_module.compute_effective_throughput(100.0, p_success_2050)

        print("\nRework Overhead (Baseline Scenario):")
        print("-" * 70)
        print(f"  2024: {stats_2024['overhead_factor']:.2f}x (cumulative p = {stats_2024['cumulative_success_prob']:.4f})")
        print(f"  2050: {stats_2050['overhead_factor']:.2f}x (cumulative p = {stats_2050['cumulative_success_prob']:.4f})")
        print(f"  Improvement: {stats_2024['overhead_factor']/stats_2050['overhead_factor']:.2f}x reduction")

    # -------------------------------------------------------------------------
    # Step 5: Export Results
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Exporting results...")

    # Export main results
    results_path = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_path, index=False)
    print(f"  - Results: {results_path}")

    # Export summary
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  - Summary: {summary_path}")

    # Export Amodei comparison
    amodei_path = os.path.join(output_dir, 'amodei_comparison.csv')
    amodei_comparison.to_csv(amodei_path, index=False)
    print(f"  - Amodei comparison: {amodei_path}")

    # Export parameters
    params_path = os.path.join(output_dir, 'parameters.json')
    model.export_parameters(params_path)
    print(f"  - Parameters: {params_path}")

    # -------------------------------------------------------------------------
    # Step 6: Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Generating visualizations...")

    import matplotlib.pyplot as plt

    # Figure 1: Scenario Comparison with Amodei
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # Colorblind-safe palette
    colors = {
        'Pessimistic': '#4575b4',  # Blue
        'Baseline': '#fdae61',     # Orange
        'Optimistic': '#f46d43',   # Red-orange
        'Amodei': '#d73027',       # Red
    }

    for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic', 'Amodei']:
        if scenario_name in model.results:
            data = model.results[scenario_name]
            color = colors.get(scenario_name, 'gray')
            linestyle = '--' if scenario_name == 'Amodei' else '-'
            linewidth = 3 if scenario_name == 'Amodei' else 2
            ax1.plot(data['year'], data['cumulative_progress'],
                    label=scenario_name, color=color, linestyle=linestyle, linewidth=linewidth)

    # Add Amodei target zone
    ax1.axhspan(130, 260, alpha=0.1, color='green', label='Amodei target zone (10x)')
    ax1.axhline(y=260, color='green', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cumulative Progress (equiv years)', fontsize=12)
    ax1.set_title('Model Predictions vs. Amodei Target (10x Acceleration)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2024, 2050)
    fig1.savefig(os.path.join(output_dir, 'fig_amodei_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  - fig_amodei_comparison.png")

    # Figure 2: 10-Year Progress Comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    scenarios = ['Pessimistic', 'Baseline', 'Optimistic', 'Amodei']
    progress_10yr = [amodei_comparison[amodei_comparison['scenario'] == s]['progress_10yr'].iloc[0]
                    for s in scenarios if s in amodei_comparison['scenario'].values]
    scenario_labels = [s for s in scenarios if s in amodei_comparison['scenario'].values]

    bar_colors = [colors.get(s, 'gray') for s in scenario_labels]
    bars = ax2.bar(scenario_labels, progress_10yr, color=bar_colors, alpha=0.8)

    # Add Amodei targets
    ax2.axhline(y=50, color='green', linestyle='--', linewidth=2, label='Amodei low (50yr = 5x)')
    ax2.axhline(y=100, color='green', linestyle='-', linewidth=2, label='Amodei high (100yr = 10x)')

    ax2.set_ylabel('Progress in First 10 Years (equiv years)', fontsize=12)
    ax2.set_title('10-Year Progress: Model vs. Amodei Targets', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, progress_10yr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

    fig2.savefig(os.path.join(output_dir, 'fig_10yr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  - fig_10yr_comparison.png")

    # Figure 3: Rework Overhead Over Time
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic', 'Amodei']:
        if scenario_name in model.results:
            data = model.results[scenario_name]
            color = colors.get(scenario_name, 'gray')
            ax3.plot(data['year'], data['rework_overhead'],
                    label=scenario_name, color=color, linewidth=2)

    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Rework Overhead Factor', fontsize=12)
    ax3.set_title('Pipeline Rework Overhead Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.savefig(os.path.join(output_dir, 'fig_rework_overhead.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  - fig_rework_overhead.png")

    # Figure 4: Acceleration Factor Timeline
    fig4, ax4 = plt.subplots(figsize=(12, 7))

    for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic', 'Amodei']:
        if scenario_name in model.results:
            data = model.results[scenario_name]
            # Compute running acceleration (cumulative progress / calendar years elapsed)
            calendar_years = data['year'] - 2024
            calendar_years = calendar_years.replace(0, 0.5)  # Avoid division by zero
            acceleration = data['cumulative_progress'] / calendar_years
            color = colors.get(scenario_name, 'gray')
            linestyle = '--' if scenario_name == 'Amodei' else '-'
            ax4.plot(data['year'], acceleration,
                    label=scenario_name, color=color, linestyle=linestyle, linewidth=2)

    # Amodei targets
    ax4.axhline(y=5, color='green', linestyle=':', alpha=0.7, label='5x (Amodei low)')
    ax4.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10x (Amodei high)')

    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Acceleration Factor (equiv years / calendar years)', fontsize=12)
    ax4.set_title('Acceleration Factor Over Time vs. Amodei Targets', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2025, 2050)
    ax4.set_ylim(0, 15)
    fig4.savefig(os.path.join(output_dir, 'fig_acceleration_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("  - fig_acceleration_timeline.png")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("v0.7 Model Run Complete")
    print("=" * 70)

    # Key findings
    amodei_result = results[results['scenario'] == 'Upper_Bound_Amodei']
    baseline_result = results[results['scenario'] == 'Baseline']

    amodei_2050 = amodei_result[amodei_result['year'] == 2050]['cumulative_progress'].iloc[0]
    baseline_2050 = baseline_result[baseline_result['year'] == 2050]['cumulative_progress'].iloc[0]
    amodei_10yr = amodei_comparison[amodei_comparison['scenario'] == 'Upper_Bound_Amodei']['progress_10yr'].iloc[0]

    print("\nKey Results:")
    print("-" * 40)
    print(f"Baseline 2050: {baseline_2050:.1f} equiv years ({baseline_2050/26:.1f}x acceleration)")
    print(f"Amodei 2050:   {amodei_2050:.1f} equiv years ({amodei_2050/26:.1f}x acceleration)")
    print(f"Amodei 10yr:   {amodei_10yr:.1f} equiv years ({amodei_10yr/10:.1f}x acceleration)")

    meets_target = "YES" if amodei_10yr >= 50 else "NO"
    print(f"\nAmodei scenario meets 5-10x target in 10 years: {meets_target}")

    print(f"\nAll outputs saved to: {output_dir}/")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model v0.7')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip Amodei comparison analysis')
    parser.add_argument('--skip-pipeline-iteration', action='store_true',
                        help='Skip pipeline iteration analysis')
    args = parser.parse_args()

    model, results = main(
        run_comparison=not args.skip_comparison,
        run_pipeline_iteration=not args.skip_pipeline_iteration
    )
