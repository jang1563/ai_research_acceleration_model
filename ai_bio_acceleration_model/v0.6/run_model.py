#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model v0.6

Key Changes in v0.6:
- Data Quality Module D(t): Models how data quality affects all stages
- D(t) grows with AI capability (gamma=0.15 default)
- Stage-specific data quality elasticities (S4 Data Analysis highest at 0.9)
- Comparison of results with/without data quality

Usage:
    python run_model.py [--skip-sensitivity] [--skip-monte-carlo] [--mc-samples N]

Version: 0.6
"""

import os
import sys
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, ModelConfig
from data_quality import DataQualityModule, DataQualityConfig, DEFAULT_DQ_PARAMS


def main(run_sensitivity: bool = True, run_monte_carlo: bool = True, mc_samples: int = 500):
    """Run the complete model pipeline."""

    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.6 (Data Quality Module)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run Model WITH Data Quality (v0.6)
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Running model WITH data quality...")

    model_dq = AIBioAccelerationModel()

    print(f"  - Time horizon: {model_dq.config.t0} to {model_dq.config.T}")
    print(f"  - Data quality enabled: {model_dq.config.enable_data_quality}")
    print(f"  - Data quality gamma: {model_dq.config.data_quality_config.gamma}")

    results_dq = model_dq.run_all_scenarios()

    # -------------------------------------------------------------------------
    # Step 2: Run Model WITHOUT Data Quality (for comparison)
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Running model WITHOUT data quality (comparison)...")

    config_no_dq = ModelConfig(enable_data_quality=False)
    model_no_dq = AIBioAccelerationModel(config=config_no_dq)
    results_no_dq = model_no_dq.run_all_scenarios()

    # -------------------------------------------------------------------------
    # Step 3: Compare Results
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Comparing results...")

    print("\nBaseline Scenario Comparison (2050):")
    print("-" * 60)

    baseline_dq = results_dq[(results_dq['scenario'] == 'Baseline') & (results_dq['year'] == 2050)]
    baseline_no_dq = results_no_dq[(results_no_dq['scenario'] == 'Baseline') & (results_no_dq['year'] == 2050)]

    progress_dq = baseline_dq['cumulative_progress'].iloc[0]
    progress_no_dq = baseline_no_dq['cumulative_progress'].iloc[0]
    delta = progress_dq - progress_no_dq
    pct_change = (progress_dq / progress_no_dq - 1) * 100

    print(f"  Without Data Quality: {progress_no_dq:.1f} equiv years")
    print(f"  With Data Quality:    {progress_dq:.1f} equiv years")
    print(f"  Improvement:          +{delta:.1f} years ({pct_change:+.1f}%)")

    # Data quality trajectory
    print("\nData Quality Trajectory (Baseline):")
    print("-" * 60)
    baseline_full = results_dq[results_dq['scenario'] == 'Baseline']
    for year in [2024, 2030, 2040, 2050]:
        row = baseline_full[baseline_full['year'] == year].iloc[0]
        print(f"  {year}: D(t)={row['data_quality']:.2f}")

    # DQM by stage at 2050
    print("\nData Quality Multipliers by Stage (2050):")
    print("-" * 60)
    row_2050 = baseline_full[baseline_full['year'] == 2050].iloc[0]
    for i in range(1, 11):
        dqm = row_2050[f'DQM_{i}']
        elasticity = model_dq.data_quality.get_elasticity(i)
        stage_name = model_dq.config.stages[i-1].name
        print(f"  S{i:2d} ({stage_name[:15]:15s}): DQM={dqm:.3f} (elasticity={elasticity:.1f})")

    # -------------------------------------------------------------------------
    # Step 4: Export Results
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Exporting results...")

    # Export results with data quality
    results_path = os.path.join(output_dir, 'results.csv')
    results_dq.to_csv(results_path, index=False)
    print(f"  - Results: {results_path}")

    # Export comparison
    comparison_path = os.path.join(output_dir, 'data_quality_comparison.csv')
    comparison_data = []
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        for year in [2030, 2040, 2050]:
            row_dq = results_dq[(results_dq['scenario'] == scenario) & (results_dq['year'] == year)]
            row_no_dq = results_no_dq[(results_no_dq['scenario'] == scenario) & (results_no_dq['year'] == year)]

            if len(row_dq) > 0 and len(row_no_dq) > 0:
                p_dq = row_dq['cumulative_progress'].iloc[0]
                p_no_dq = row_no_dq['cumulative_progress'].iloc[0]
                comparison_data.append({
                    'scenario': scenario,
                    'year': year,
                    'without_dq': p_no_dq,
                    'with_dq': p_dq,
                    'delta': p_dq - p_no_dq,
                    'pct_change': (p_dq / p_no_dq - 1) * 100
                })

    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  - Comparison: {comparison_path}")

    # Export parameters
    params_path = os.path.join(output_dir, 'parameters.json')
    import json
    params = {
        'version': '0.6',
        'data_quality': {
            'enabled': True,
            'gamma': model_dq.config.data_quality_config.gamma,
            'elasticities': {f'S{k}': v for k, v in model_dq.config.data_quality_config.elasticities.items()},
        },
        'model': {
            't0': model_dq.config.t0,
            'T': model_dq.config.T,
            'n_stages': len(model_dq.config.stages),
            'n_scenarios': len(model_dq.config.scenarios),
        }
    }
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  - Parameters: {params_path}")

    # -------------------------------------------------------------------------
    # Step 5: Generate DQM-Specific Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 5/7] Generating DQM-specific visualizations...")

    import matplotlib.pyplot as plt

    # Figure DQ1: Data Quality Trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        data = results_dq[results_dq['scenario'] == scenario]
        ax1.plot(data['year'], data['data_quality'], label=scenario, linewidth=2)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Data Quality Index D(t)', fontsize=12)
    ax1.set_title('Data Quality Evolution Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.savefig(os.path.join(output_dir, 'fig_data_quality_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  - fig_data_quality_trajectory.png")

    # Figure DQ2: DQM by Stage (2050)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    stages = [f'S{i}' for i in range(1, 11)]
    dqm_values = [row_2050[f'DQM_{i}'] for i in range(1, 11)]
    elasticities = [model_dq.data_quality.get_elasticity(i) for i in range(1, 11)]

    x = range(len(stages))
    bars = ax2.bar(x, dqm_values, color='steelblue', alpha=0.8, label='DQM (2050)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s}\n(e={e:.1f})' for s, e in zip(stages, elasticities)])
    ax2.set_xlabel('Stage (elasticity)', fontsize=12)
    ax2.set_ylabel('Data Quality Multiplier', fontsize=12)
    ax2.set_title('Data Quality Impact by Stage (Baseline, 2050)', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (no DQ effect)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.savefig(os.path.join(output_dir, 'fig_dqm_by_stage.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  - fig_dqm_by_stage.png")

    # Figure DQ3: Comparison with/without Data Quality
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for scenario, color, ls in [('Pessimistic', 'blue', '--'), ('Baseline', 'green', '-'), ('Optimistic', 'red', '-.')]:
        data_dq = results_dq[results_dq['scenario'] == scenario]
        data_no_dq = results_no_dq[results_no_dq['scenario'] == scenario]
        ax3.plot(data_dq['year'], data_dq['cumulative_progress'], color=color, linestyle=ls, linewidth=2, label=f'{scenario} (with DQ)')
        ax3.plot(data_no_dq['year'], data_no_dq['cumulative_progress'], color=color, linestyle=ls, linewidth=1, alpha=0.5, label=f'{scenario} (w/o DQ)')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Cumulative Progress (equiv years)', fontsize=12)
    ax3.set_title('Impact of Data Quality Module on Progress', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.savefig(os.path.join(output_dir, 'fig_dq_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  - fig_dq_comparison.png")

    # -------------------------------------------------------------------------
    # Step 6: Generate Standard Model Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Generating standard model visualizations...")

    from visualize import ModelVisualizer
    visualizer = ModelVisualizer(results_dq, model_dq.config.stages, output_dir)
    visualizer.generate_all_figures()
    print("  - Standard figures (fig1-fig7, dashboard) generated")

    # -------------------------------------------------------------------------
    # Step 7: Generate Improved Visualizations (optional)
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Generating improved visualizations...")

    try:
        from visualize_improved import generate_improved_visualizations

        # Run sensitivity analysis for tornado diagram
        if run_sensitivity:
            from sensitivity import SensitivityAnalyzer
            print("  - Running sensitivity analysis...")
            sensitivity_analyzer = SensitivityAnalyzer(model_dq)
            sensitivity_results = sensitivity_analyzer.run_oat_analysis()
            sensitivity_df = sensitivity_analyzer.to_dataframe()
            sensitivity_df.to_csv(os.path.join(output_dir, 'sensitivity_summary.csv'), index=False)
            print("  - sensitivity_summary.csv saved")
        else:
            sensitivity_df = None

        # Run Monte Carlo for uncertainty bands
        if run_monte_carlo:
            from uncertainty import MonteCarloAnalyzer
            print(f"  - Running Monte Carlo ({mc_samples} samples)...")
            mc_analyzer = MonteCarloAnalyzer(model_dq, n_samples=mc_samples)
            mc_results = {}
            ci_results = {}
            for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic']:
                samples, confidence_intervals = mc_analyzer.run_scenario(scenario_name)
                mc_results[scenario_name] = samples
                ci_results[scenario_name] = confidence_intervals
                # Save MC results
                samples.to_csv(os.path.join(output_dir, f'monte_carlo_{scenario_name.lower()}.csv'), index=False)
                confidence_intervals.to_csv(os.path.join(output_dir, f'confidence_intervals_{scenario_name.lower()}.csv'), index=False)
            print("  - Monte Carlo results saved")
        else:
            mc_results = None

        # Get baseline value for tornado diagram
        baseline_2050 = results_dq[(results_dq['scenario'] == 'Baseline') & (results_dq['year'] == 2050)]
        baseline_value = baseline_2050['cumulative_progress'].iloc[0] if len(baseline_2050) > 0 else 83.0

        # Generate all improved visualizations
        generate_improved_visualizations(
            results=results_dq,
            stages=model_dq.config.stages,
            sensitivity_df=sensitivity_df,
            mc_results=mc_results,
            baseline_value=baseline_value,
            output_dir=output_dir
        )

    except Exception as e:
        import traceback
        print(f"  - Warning: Could not generate improved visualizations: {e}")
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("v0.6 Model Run Complete")
    print("=" * 70)

    print("\nKey Results:")
    print("-" * 40)
    print(f"Data Quality Impact: +{pct_change:.1f}% progress by 2050")
    print(f"D(2050): {row_2050['data_quality']:.2f} (vs D(2024)=1.0)")
    print(f"Highest DQM: S4 Data Analysis ({row_2050['DQM_4']:.3f})")
    print(f"Lowest DQM: S9 Regulatory ({row_2050['DQM_9']:.3f})")

    print(f"\nAll outputs saved to: {output_dir}/")

    return model_dq, results_dq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model v0.6')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip sensitivity analysis')
    parser.add_argument('--skip-monte-carlo', action='store_true',
                        help='Skip Monte Carlo uncertainty analysis')
    parser.add_argument('--mc-samples', type=int, default=500,
                        help='Number of Monte Carlo samples (default: 500)')
    args = parser.parse_args()

    model, results = main(
        run_sensitivity=not args.skip_sensitivity,
        run_monte_carlo=not args.skip_monte_carlo,
        mc_samples=args.mc_samples
    )
