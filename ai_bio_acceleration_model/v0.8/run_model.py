#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model v0.8

Key Changes in v0.8:
- Disease Models Module: Disease-specific time-to-cure calculations
- Case Studies: Cancer, Alzheimer's, Pandemic Preparedness
- Patient Impact Projections: Expected beneficiaries by scenario
- Cure Probability Distributions: Monte Carlo uncertainty

Usage:
    python run_model.py [--skip-disease-models]

Version: 0.8
"""

import os
import sys
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, ModelConfig, Scenario, TherapeuticArea
from data_quality import DataQualityModule, DataQualityConfig
from pipeline_iteration import PipelineIterationModule, PipelineIterationConfig
from disease_models import (
    DiseaseModelModule, DiseaseModelConfig, DiseaseCategory, DISEASE_PROFILES
)


def main(run_disease_models: bool = True):
    """Run the complete model pipeline."""

    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.8 (Disease Models + Case Studies)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run Full Model with All Features
    # -------------------------------------------------------------------------
    print("\n[Step 1/7] Running full model (all features enabled)...")

    model = AIBioAccelerationModel()

    print(f"  - Time horizon: {model.config.t0} to {model.config.T}")
    print(f"  - Pipeline iteration: {model.config.enable_pipeline_iteration}")
    print(f"  - Data quality: {model.config.enable_data_quality}")
    print(f"  - Scenarios: {len(model.config.scenarios)}")

    results = model.run_all_scenarios()

    # -------------------------------------------------------------------------
    # Step 2: Key Results Summary
    # -------------------------------------------------------------------------
    print("\n[Step 2/7] Computing summary statistics...")

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
    print("\n[Step 3/7] Comparing with Amodei's predictions...")

    amodei_comparison = model.compare_with_amodei()

    print("\nAmodei Comparison (Target: 10x = 50-100 years in 10 years):")
    print("-" * 70)
    print(f"{'Scenario':<25} {'10yr Progress':>12} {'10yr Accel':>10} {'Meets Target':>12}")
    print("-" * 70)
    for _, row in amodei_comparison.iterrows():
        meets = "YES" if row['meets_amodei_low'] else "No"
        print(f"{row['scenario']:<25} {row['progress_10yr']:>12.1f} {row['acceleration_10yr']:>10.1f}x {meets:>12}")

    # -------------------------------------------------------------------------
    # Step 4: Disease Models Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 4/7] Running disease-specific models...")

    if run_disease_models:
        disease_module = DiseaseModelModule()

        # Base durations (months)
        base_durations = {
            1: 6, 2: 3, 3: 12, 4: 2, 5: 8,
            6: 12, 7: 24, 8: 36, 9: 12, 10: 12
        }

        # Extract scenario data from model results
        scenario_data = {}
        for scenario_name in ['Baseline', 'Optimistic', 'Upper_Bound_Amodei']:
            if scenario_name in model.results:
                data = model.results[scenario_name]
                # Get 2035 values (midpoint) as representative
                year_data = data[data['year'] == 2035].iloc[0]

                multipliers = {i+1: year_data[f'M_{i+1}'] for i in range(10)}
                p_success = {i+1: year_data[f'p_{i+1}'] for i in range(10)}

                scenario_data[scenario_name] = {
                    'multipliers': multipliers,
                    'p_success': p_success,
                }

        # Case Study Diseases
        case_study_diseases = [
            DiseaseCategory.BREAST_CANCER,
            DiseaseCategory.ALZHEIMERS,
            DiseaseCategory.PANDEMIC_NOVEL,
            DiseaseCategory.PANCREATIC_CANCER,
            DiseaseCategory.RARE_GENETIC,
        ]

        print("\nDisease Case Studies:")
        print("-" * 70)

        all_case_studies = []
        for disease in case_study_diseases:
            case_study = disease_module.generate_case_study(
                disease, scenario_data, base_durations
            )
            all_case_studies.append(case_study)

            profile = disease_module.get_disease_profile(disease)
            print(f"\n  {profile.name}:")
            print(f"    Starting Stage: {profile.starting_stage}, Advances Needed: {profile.advances_needed}")
            print(f"    Unmet Need Score: {profile.unmet_need_score}/10")

            for _, row in case_study.iterrows():
                print(f"      {row['scenario']:20s}: {row['expected_time_years']:.1f} yr to cure, "
                      f"P(cure by 2050)={row['cure_probability_26yr']:.1%}")

        # Combine all case studies
        case_studies_df = pd.concat(all_case_studies, ignore_index=True)

        # -------------------------------------------------------------------------
        # Step 5: Compute Patient Impact
        # -------------------------------------------------------------------------
        print("\n[Step 5/7] Computing patient impact projections...")

        print("\nExpected Beneficiaries by Disease (Upper Bound Scenario):")
        print("-" * 70)

        impact_results = []
        for disease in case_study_diseases:
            profile = disease_module.get_disease_profile(disease)

            # Get cure probability from case study
            cure_prob = case_studies_df[
                (case_studies_df['disease'] == profile.name) &
                (case_studies_df['scenario'] == 'Upper_Bound_Amodei')
            ]['cure_probability_26yr'].values[0]

            impact = disease_module.compute_patients_impacted(disease, cure_prob)
            impact_results.append(impact)

            print(f"  {profile.name:30s}: {impact['expected_beneficiaries']:>15,} patients "
                  f"(P={cure_prob:.1%})")

        impact_df = pd.DataFrame(impact_results)

    # -------------------------------------------------------------------------
    # Step 6: Export Results
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Exporting results...")

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

    if run_disease_models:
        # Export disease case studies
        case_studies_path = os.path.join(output_dir, 'disease_case_studies.csv')
        case_studies_df.to_csv(case_studies_path, index=False)
        print(f"  - Disease case studies: {case_studies_path}")

        # Export patient impact
        impact_path = os.path.join(output_dir, 'patient_impact.csv')
        impact_df.to_csv(impact_path, index=False)
        print(f"  - Patient impact: {impact_path}")

        # Export disease profiles
        profiles_path = os.path.join(output_dir, 'disease_profiles.csv')
        disease_module.get_all_profiles_summary().to_csv(profiles_path, index=False)
        print(f"  - Disease profiles: {profiles_path}")

    # -------------------------------------------------------------------------
    # Step 7: Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Generating visualizations...")

    import matplotlib.pyplot as plt

    # Colorblind-safe palette
    colors = {
        'Pessimistic': '#4575b4',
        'Baseline': '#fdae61',
        'Optimistic': '#f46d43',
        'Upper_Bound_Amodei': '#d73027',
    }

    # Figure 1: Scenario Comparison with Amodei
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    for scenario_name in ['Pessimistic', 'Baseline', 'Optimistic', 'Upper_Bound_Amodei']:
        if scenario_name in model.results:
            data = model.results[scenario_name]
            color = colors.get(scenario_name, 'gray')
            linestyle = '--' if scenario_name == 'Upper_Bound_Amodei' else '-'
            linewidth = 3 if scenario_name == 'Upper_Bound_Amodei' else 2
            label = 'Upper Bound (Amodei)' if scenario_name == 'Upper_Bound_Amodei' else scenario_name
            ax1.plot(data['year'], data['cumulative_progress'],
                    label=label, color=color, linestyle=linestyle, linewidth=linewidth)

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

    if run_disease_models:
        # Figure 2: Disease Time-to-Cure Comparison WITH CONFIDENCE INTERVALS
        # UPDATED per Expert F1 (Dr. Nakamura): Added error bars
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        diseases = case_studies_df['disease'].unique()
        scenarios = ['Baseline', 'Optimistic', 'Upper_Bound_Amodei']
        x = np.arange(len(diseases))
        width = 0.25

        # Compute confidence intervals for Upper_Bound scenario
        ci_data = {}
        for disease in case_study_diseases:
            ci = disease_module.compute_time_to_cure_distribution(
                disease, scenario_data['Upper_Bound_Amodei']['multipliers'],
                scenario_data['Upper_Bound_Amodei']['p_success'], base_durations
            )
            ci_data[DISEASE_PROFILES[disease].name] = ci

        for i, scenario in enumerate(scenarios):
            times = [case_studies_df[(case_studies_df['disease'] == d) &
                                     (case_studies_df['scenario'] == scenario)]['expected_time_years'].values[0]
                     for d in diseases]
            color = colors.get(scenario, 'gray')
            label = 'Upper Bound (Amodei)' if scenario == 'Upper_Bound_Amodei' else scenario

            # Add error bars for Upper Bound scenario (90% CI)
            if scenario == 'Upper_Bound_Amodei':
                # Use median from distribution, ensure non-negative error bars
                yerr_lower = [max(0, ci_data[d]['median'] - ci_data[d]['ci_5']) for j, d in enumerate(diseases)]
                yerr_upper = [max(0, ci_data[d]['ci_95'] - ci_data[d]['median']) for j, d in enumerate(diseases)]
                # Use median for consistency with CI
                medians = [ci_data[d]['median'] for d in diseases]
                ax2.bar(x + i*width, medians, width, label=label, color=color, alpha=0.8,
                       yerr=[yerr_lower, yerr_upper], capsize=3, error_kw={'elinewidth': 1.5})
            else:
                ax2.bar(x + i*width, times, width, label=label, color=color, alpha=0.8)

        ax2.set_xlabel('Disease', fontsize=12)
        ax2.set_ylabel('Expected Time to Cure (years)', fontsize=12)
        ax2.set_title('Time to Cure by Disease and Scenario\n(Error bars show 90% CI for Upper Bound)',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(diseases, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=26, color='red', linestyle='--', alpha=0.5, label='2050 horizon')

        fig2.savefig(os.path.join(output_dir, 'fig_disease_time_to_cure.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("  - fig_disease_time_to_cure.png")

        # Export confidence interval data
        ci_df = pd.DataFrame([ci_data[d] for d in diseases])
        ci_path = os.path.join(output_dir, 'time_to_cure_confidence_intervals.csv')
        ci_df.to_csv(ci_path, index=False)
        print(f"  - time_to_cure_confidence_intervals.csv")

        # Figure 3: Cure Probability by Disease
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        for i, scenario in enumerate(scenarios):
            probs = [case_studies_df[(case_studies_df['disease'] == d) &
                                     (case_studies_df['scenario'] == scenario)]['cure_probability_26yr'].values[0]
                     for d in diseases]
            color = colors.get(scenario, 'gray')
            label = 'Upper Bound (Amodei)' if scenario == 'Upper_Bound_Amodei' else scenario
            ax3.bar(x + i*width, [p*100 for p in probs], width, label=label, color=color, alpha=0.8)

        ax3.set_xlabel('Disease', fontsize=12)
        ax3.set_ylabel('Probability of Cure by 2050 (%)', fontsize=12)
        ax3.set_title('Cure Probability by Disease (2024-2050)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(diseases, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 100)

        fig3.savefig(os.path.join(output_dir, 'fig_cure_probability.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  - fig_cure_probability.png")

        # Figure 4: Patient Impact (LOG SCALE per Expert F2)
        # UPDATED per Expert F2 (Dr. Nakamura): Log scale to show variation across diseases
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        impact_data = impact_df.sort_values('expected_beneficiaries', ascending=True)
        colors_list = ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']

        bars = ax4.barh(impact_data['disease_name'], impact_data['expected_beneficiaries'],
                       color=colors_list[:len(impact_data)])

        ax4.set_xscale('log')  # Log scale per Expert F2
        ax4.set_xlabel('Expected Beneficiaries (Log Scale)', fontsize=12)
        ax4.set_title('Patient Impact: Expected Beneficiaries by 2050\n(Upper Bound Scenario, Log Scale)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x', which='both')

        # Add value labels with appropriate formatting
        for bar, val in zip(bars, impact_data['expected_beneficiaries']):
            if val >= 1e9:
                label = f'{val/1e9:.2f}B'
            elif val >= 1e6:
                label = f'{val/1e6:.1f}M'
            else:
                label = f'{val/1e3:.0f}K'
            ax4.text(val * 1.5, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=10)

        fig4.savefig(os.path.join(output_dir, 'fig_patient_impact.png'), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  - fig_patient_impact.png")

        # Figure 5: Cure Probability Trajectories (NEW per Expert F3)
        # ADDED per Expert F3 (Dr. Nakamura): P(cure) over time curves
        fig5, ax5 = plt.subplots(figsize=(12, 7))

        # Sample years for trajectory (every 2 years to reduce computation)
        sample_years = list(range(2024, 2051, 2))

        trajectory_colors = {
            'Breast Cancer': '#d73027',
            'Alzheimer\'s Disease': '#4575b4',
            'Novel Pandemic Pathogen': '#1a9850',
            'Pancreatic Cancer': '#762a83',
            'Rare Genetic Disease (Generic)': '#fdae61',
        }

        trajectory_data = []
        for disease in case_study_diseases:
            traj = disease_module.compute_cure_probability_trajectory(
                disease, scenario_data['Upper_Bound_Amodei']['multipliers'],
                scenario_data['Upper_Bound_Amodei']['p_success'], base_durations,
                years=sample_years
            )
            trajectory_data.append(traj)

            years_list = list(traj['trajectory'].keys())
            probs = [traj['trajectory'][y] * 100 for y in years_list]
            color = trajectory_colors.get(traj['disease_name'], 'gray')
            ax5.plot(years_list, probs, label=traj['disease_name'],
                    color=color, linewidth=2.5, marker='o', markersize=4)

        ax5.set_xlabel('Year', fontsize=12)
        ax5.set_ylabel('Cumulative P(Cure) (%)', fontsize=12)
        ax5.set_title('Cure Probability Trajectories (Upper Bound Scenario)\nHow P(cure) evolves from 2024 to 2050',
                     fontsize=14, fontweight='bold')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(2024, 2050)
        ax5.set_ylim(0, 105)

        fig5.savefig(os.path.join(output_dir, 'fig_cure_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("  - fig_cure_trajectories.png")

        # Export trajectory data
        traj_rows = []
        for traj in trajectory_data:
            for year, prob in traj['trajectory'].items():
                traj_rows.append({
                    'disease': traj['disease_name'],
                    'year': year,
                    'cure_probability': prob
                })
        traj_df = pd.DataFrame(traj_rows)
        traj_path = os.path.join(output_dir, 'cure_probability_trajectories.csv')
        traj_df.to_csv(traj_path, index=False)
        print(f"  - cure_probability_trajectories.csv")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("v0.8 Model Run Complete")
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
    print(f"Upper Bound 2050: {amodei_2050:.1f} equiv years ({amodei_2050/26:.1f}x acceleration)")
    print(f"Upper Bound 10yr: {amodei_10yr:.1f} equiv years ({amodei_10yr/10:.1f}x acceleration)")

    meets_target = "YES" if amodei_10yr >= 50 else "NO"
    print(f"\nUpper Bound scenario meets 5-10x target in 10 years: {meets_target}")

    if run_disease_models:
        print("\nDisease Model Highlights:")
        print("-" * 40)
        for disease in case_study_diseases:
            profile = DISEASE_PROFILES[disease]
            row = case_studies_df[
                (case_studies_df['disease'] == profile.name) &
                (case_studies_df['scenario'] == 'Upper_Bound_Amodei')
            ].iloc[0]
            print(f"  {profile.name:30s}: {row['cure_probability_26yr']:.0%} P(cure), "
                  f"{row['expected_time_years']:.1f} yr expected")

    print(f"\nAll outputs saved to: {output_dir}/")

    return model, results


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model v0.8')
    parser.add_argument('--skip-disease-models', action='store_true',
                        help='Skip disease model analysis')
    args = parser.parse_args()

    model, results = main(
        run_disease_models=not args.skip_disease_models
    )
