#!/usr/bin/env python3
"""
Run Model Script for AI-Accelerated Biological Discovery Model - v1.1

This script runs the complete v1.1 model with all P1/P2 fixes and generates
outputs including summary statistics, figures, and validation results.

Usage:
    python run_model.py [--full] [--quick]

Options:
    --full: Run full analysis including Monte Carlo and Sobol
    --quick: Quick run with minimal samples for testing

Version: 1.1
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import (
    AIBioAccelerationModel, ModelConfig, Scenario, ScenarioType,
    AIGrowthModel, DEFAULT_SCENARIOS, MODEL_VERSION, RANDOM_SEED
)
from uncertainty_quantification import UncertaintyQuantification, UQConfig
from historical_validation import HistoricalValidation, get_fda_approval_baseline
from data_quality import DataQualityModule, DataQualityConfig, GLOBAL_ACCESS_FACTORS
from disease_models import DiseaseModelModule, DiseaseModelConfig, DiseaseCategory
from pipeline_iteration import PipelineIterationModule, PipelineIterationConfig
from policy_analysis import PolicyAnalysisModule, PolicyAnalysisConfig


def run_model(full_analysis: bool = False, quick: bool = False):
    """Run the complete v1.1 model."""

    print("=" * 70)
    print(f"AI-Accelerated Biological Discovery Model - v{MODEL_VERSION}")
    print("=" * 70)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Initialize model
    config = ModelConfig()
    model = AIBioAccelerationModel(config)

    # Run all default scenarios
    print("\n" + "-" * 70)
    print("Running Scenarios...")
    print("-" * 70)

    all_results = []
    scenario_summaries = []

    for scenario in DEFAULT_SCENARIOS:
        print(f"\n  Running: {scenario.name}")
        results = model.run_scenario(scenario)
        all_results.append(results)

        # Get final values
        final = results.iloc[-1]
        scenario_summaries.append({
            'scenario': scenario.name,
            'scenario_type': scenario.scenario_type.value,
            'ai_growth_model': scenario.ai_growth_model.value,
            'g_ai': scenario.g_ai,
            'progress_by_2050': final['cumulative_progress'],
            'max_progress_rate': results['progress_rate'].max(),
            'bottleneck_2050': final.get('bottleneck_stage', 'N/A'),
        })

    # Summary DataFrame
    summary_df = pd.DataFrame(scenario_summaries)
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(summary_df.to_string(index=False))

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    # Run additional analyses if requested
    if full_analysis or not quick:
        print("\n" + "-" * 70)
        print("Running Data Quality Analysis...")
        print("-" * 70)

        dq_module = DataQualityModule()
        years = np.arange(2024, 2051)
        A_trajectory = np.exp(0.50 * (years - 2024))  # Baseline scenario

        dq_trajectory = dq_module.get_trajectory(years, A_trajectory)
        print(f"  D(2024): {dq_trajectory.iloc[0]['D']:.2f}")
        print(f"  D(2050): {dq_trajectory.iloc[-1]['D']:.2f}")
        print(f"  Growth factor: {dq_trajectory.iloc[-1]['D'] / dq_trajectory.iloc[0]['D']:.2f}x")

        print("\n  Global Access Factors (P1-8):")
        for area, factor in GLOBAL_ACCESS_FACTORS.items():
            print(f"    {area}: {factor}")

        print("\n" + "-" * 70)
        print("Running Disease Model Analysis...")
        print("-" * 70)

        disease_module = DiseaseModelModule()
        disease_summary = disease_module.get_all_profiles_summary()
        print(disease_summary[['disease', 'ai_potential', 'phase2_M_max', 'global_access']].to_string(index=False))

        disease_summary.to_csv(os.path.join(output_dir, 'disease_profiles.csv'), index=False)

        print("\n" + "-" * 70)
        print("Running Pipeline Iteration Analysis...")
        print("-" * 70)

        pipeline_module = PipelineIterationModule()
        mfg_summary = pipeline_module.get_manufacturing_summary()
        print("\n  Manufacturing Constraints (P2-13):")
        print(mfg_summary.to_string(index=False))

        print("\n" + "-" * 70)
        print("Running Policy Analysis...")
        print("-" * 70)

        policy_module = PolicyAnalysisModule()
        rankings = policy_module.rank_interventions()
        print("\n  Top 5 Interventions by ROI:")
        print(rankings[['rank', 'intervention_name', 'annual_cost_usd', 'roi']].head(5).to_string(index=False))

        rankings.to_csv(os.path.join(output_dir, 'policy_rankings.csv'), index=False)

    if full_analysis:
        print("\n" + "-" * 70)
        print("Running Historical Validation (P1-3)...")
        print("-" * 70)

        validator = HistoricalValidation()
        fda_data = get_fda_approval_baseline()
        print("\n  FDA Novel Drug Approvals (2015-2023):")
        for year, approvals in sorted(fda_data.items()):
            print(f"    {year}: {approvals}")

        print("\n" + "-" * 70)
        print("Running Uncertainty Quantification...")
        print("-" * 70)

        n_samples = 100 if quick else 1000
        uq = UncertaintyQuantification(UQConfig())

        print(f"\n  Running Monte Carlo with {n_samples} samples...")
        print("  (Use --full for complete analysis with 10,000 samples)")

        # Just show parameter distributions
        print("\n  Key Parameter Distributions:")
        for name, param in list(uq.parameters.items())[:5]:
            print(f"    {name}: {param.distribution.value}, "
                  f"bounds=[{param.lower_bound:.2f}, {param.upper_bound:.2f}]")

    print("\n" + "=" * 70)
    print("Model run complete.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model v1.1')
    parser.add_argument('--full', action='store_true', help='Run full analysis')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    args = parser.parse_args()

    run_model(full_analysis=args.full, quick=args.quick)


if __name__ == "__main__":
    main()
