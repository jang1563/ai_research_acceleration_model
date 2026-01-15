#!/usr/bin/env python3
"""
Generate Improved Visualizations for AI-Accelerated Biological Discovery Model

This script creates publication-quality figures with enhanced design:
1. Improved Tornado Diagram (bidirectional bars)
2. Combined Fan Chart (all scenarios with uncertainty)
3. Bottleneck Heatmap (time × stage constraint matrix)
4. Improved Summary Dashboard (4 informative panels)

Usage:
    python generate_improved_figures.py

Version: 0.4.2
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, run_default_model
from visualize_improved import (
    plot_tornado_improved,
    plot_combined_fan_chart,
    plot_bottleneck_heatmap,
    plot_summary_dashboard_improved,
    generate_improved_visualizations
)


def load_or_compute_results(output_dir='outputs'):
    """Load existing results or compute fresh ones."""
    results_path = os.path.join(output_dir, 'results.csv')

    if os.path.exists(results_path):
        print("Loading existing results...")
        results = pd.read_csv(results_path)
        model, _ = run_default_model()
        return model, results
    else:
        print("Computing fresh results...")
        model, results = run_default_model()
        return model, results


def load_monte_carlo_results(output_dir='outputs'):
    """Load Monte Carlo results if available."""
    mc_results = {}

    for scenario in ['pessimistic', 'baseline', 'optimistic']:
        path = os.path.join(output_dir, f'confidence_intervals_{scenario}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            mc_results[scenario.capitalize()] = df

    if mc_results:
        print(f"Loaded Monte Carlo results for {len(mc_results)} scenarios")
    else:
        print("No Monte Carlo results found")

    return mc_results if mc_results else None


def create_sensitivity_dataframe(output_dir='outputs'):
    """Create sensitivity results from existing analysis."""
    # Read sensitivity summary if available
    sens_path = os.path.join(output_dir, 'sensitivity_summary.csv')

    if os.path.exists(sens_path):
        df = pd.read_csv(sens_path)

        # Parse output_range column to get low/high values
        # Format: "76.0 - 153.0"
        if 'output_range' in df.columns:
            low_values = []
            high_values = []

            for range_str in df['output_range']:
                parts = str(range_str).split(' - ')
                if len(parts) == 2:
                    low_values.append(float(parts[0]))
                    high_values.append(float(parts[1]))
                else:
                    low_values.append(85.0)  # Default
                    high_values.append(85.0)

            df['low_value'] = low_values
            df['high_value'] = high_values

        # Rename sensitivity_index to sensitivity for compatibility
        if 'sensitivity_index' in df.columns:
            df['sensitivity'] = df['sensitivity_index']

        return df
    else:
        # Create sample sensitivity data for demonstration
        print("Creating sample sensitivity data...")

        parameters = [
            'S7_p_success', 'S8_p_success', 'g_ai', 'S7_M_max',
            'S8_M_max', 'S6_M_max', 'S6_p_success', 'S9_M_max', 'S9_p_success'
        ]

        baseline = 83.0

        # Based on actual sensitivity analysis results
        sensitivities = [0.900, 0.681, 0.636, 0.624, 0.372, 0.15, 0.10, 0.05, 0.03]

        df = pd.DataFrame({
            'parameter': parameters,
            'sensitivity': sensitivities,
            'low_value': [baseline * (1 - 0.15 * s) for s in sensitivities],
            'high_value': [baseline * (1 + 0.20 * s) for s in sensitivities],
        })

        return df


def main():
    print("=" * 70)
    print("Generating Improved Visualizations")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    model, results = load_or_compute_results(output_dir)

    # Load Monte Carlo results
    mc_results = load_monte_carlo_results(output_dir)

    # Create sensitivity dataframe
    sensitivity_df = create_sensitivity_dataframe(output_dir)

    # Get baseline value for tornado
    baseline_df = results[results['scenario'] == 'Baseline']
    baseline_value = baseline_df['cumulative_progress'].iloc[-1]

    print(f"\nBaseline progress by 2050: {baseline_value:.1f} equivalent years")

    # Generate all improved figures
    figures = generate_improved_visualizations(
        results=results,
        stages=model.config.stages,
        sensitivity_df=sensitivity_df,
        mc_results=mc_results,
        baseline_value=baseline_value,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("Improved Figures Generated Successfully!")
    print("=" * 70)

    print("\nNew figures:")
    print(f"  - {output_dir}/fig_tornado_improved.png")
    print(f"  - {output_dir}/fig_combined_fan_chart.png")
    print(f"  - {output_dir}/fig_bottleneck_heatmap.png")
    print(f"  - {output_dir}/summary_dashboard_improved.png")

    return figures


if __name__ == "__main__":
    main()
