#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model v0.9

Key Changes in v0.9:
- Policy Analysis Module: Intervention effects on model parameters
- ROI Calculations: Value per QALY, cost-effectiveness rankings
- Portfolio Optimization: Budget-constrained intervention selection
- Policy Timing: Implementation lag and duration effects

Usage:
    python run_model.py [--skip-policy-analysis]

Version: 0.9
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
from policy_analysis import (
    PolicyAnalysisModule, PolicyAnalysisConfig, POLICY_INTERVENTIONS,
    InterventionCategory
)


def main(run_policy_analysis: bool = True):
    """Run the complete model pipeline."""

    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.9 (Policy Analysis + Intervention ROI)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run Full Model with All Features
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Running full model (all features enabled)...")

    model = AIBioAccelerationModel()

    print(f"  - Time horizon: {model.config.t0} to {model.config.T}")
    print(f"  - Pipeline iteration: {model.config.enable_pipeline_iteration}")
    print(f"  - Data quality: {model.config.enable_data_quality}")
    print(f"  - Scenarios: {len(model.config.scenarios)}")

    results = model.run_all_scenarios()

    # -------------------------------------------------------------------------
    # Step 2: Key Results Summary
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Computing summary statistics...")

    summary = model.get_summary_statistics()

    print("\nScenario Summary (2050 Progress):")
    print("-" * 70)
    for _, row in summary.iterrows():
        amodei_flag = " [AMODEI]" if row['is_amodei'] else ""
        print(f"  {row['scenario']:25s}: {row['progress_by_2050']:6.1f} equiv years "
              f"({row['progress_by_2050']/26:.1f}x acceleration){amodei_flag}")

    # -------------------------------------------------------------------------
    # Step 3: Policy Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Running policy analysis...")

    if run_policy_analysis:
        policy_module = PolicyAnalysisModule()

        # Get baseline metrics for policy analysis
        baseline_result = results[results['scenario'] == 'Baseline']
        baseline_2050 = baseline_result[baseline_result['year'] == 2050]['cumulative_progress'].iloc[0]
        baseline_acceleration = baseline_2050 / 26

        # Use disease model to estimate total beneficiaries
        disease_module = DiseaseModelModule()
        total_beneficiaries = 0
        case_diseases = [
            DiseaseCategory.BREAST_CANCER,
            DiseaseCategory.ALZHEIMERS,
            DiseaseCategory.PANDEMIC_NOVEL,
            DiseaseCategory.PANCREATIC_CANCER,
            DiseaseCategory.RARE_GENETIC,
        ]
        for disease in case_diseases:
            # Assume 50% average cure probability across diseases
            impact = disease_module.compute_patients_impacted(disease, 0.50)
            total_beneficiaries += impact['expected_beneficiaries']

        print(f"\n  Baseline acceleration: {baseline_acceleration:.1f}x")
        print(f"  Estimated total beneficiaries: {total_beneficiaries/1e9:.2f}B")

        # -------------------------------------------------------------------------
        # Step 4: Rank Interventions by ROI
        # -------------------------------------------------------------------------
        print("\n[Step 4/8] Ranking policy interventions...")

        rankings = policy_module.rank_interventions(
            baseline_acceleration=baseline_acceleration,
            baseline_beneficiaries=total_beneficiaries,
            rank_by='roi'
        )

        print("\nPolicy Intervention Rankings (by ROI):")
        print("-" * 90)
        print(f"{'Rank':<5} {'Intervention':<35} {'Cost/yr':>12} {'ROI':>10} {'ΔAccel':>8} {'Evidence':>8}")
        print("-" * 90)
        for _, row in rankings.head(12).iterrows():
            print(f"{row['rank']:<5} {row['intervention_name'][:35]:<35} "
                  f"${row['annual_cost_usd']/1e6:>9.0f}M {row['roi']:>10.1f} "
                  f"{row['delta_acceleration']:>+7.2f}x {row['evidence_quality']:>8}")

        # -------------------------------------------------------------------------
        # Step 5: Analyze Intervention Categories
        # -------------------------------------------------------------------------
        print("\n[Step 5/8] Analyzing intervention categories...")

        category_summary = rankings.groupby('category').agg({
            'annual_cost_usd': 'sum',
            'delta_beneficiaries': 'sum',
            'roi': 'mean',
            'intervention_key': 'count'
        }).rename(columns={'intervention_key': 'n_interventions'})

        print("\nCategory Summary:")
        print("-" * 70)
        for category, row in category_summary.iterrows():
            print(f"  {category:30s}: {row['n_interventions']:.0f} interventions, "
                  f"avg ROI {row['roi']:.1f}, ${row['annual_cost_usd']/1e9:.1f}B total")

        # -------------------------------------------------------------------------
        # Step 6: Portfolio Recommendations
        # -------------------------------------------------------------------------
        print("\n[Step 6/8] Generating portfolio recommendations...")

        budgets = [2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000]

        print("\nBudget-Constrained Portfolios:")
        print("-" * 90)

        portfolio_results = []
        for budget in budgets:
            rec = policy_module.recommend_portfolio(
                budget_usd=budget,
                baseline_acceleration=baseline_acceleration,
                baseline_beneficiaries=total_beneficiaries,
                min_evidence_quality=2
            )
            portfolio = rec['portfolio_analysis']

            if portfolio.get('roi'):
                print(f"\n  Budget: ${budget/1e9:.0f}B/year")
                print(f"    Selected ({len(rec['selected_interventions'])}): {', '.join(rec['selected_interventions'][:3])}...")
                print(f"    Acceleration boost: {portfolio['combined_boost_factor']:.2f}x → "
                      f"{portfolio['modified_acceleration']:.1f}x total")
                print(f"    Additional beneficiaries: {portfolio['delta_beneficiaries']/1e6:.0f}M")
                print(f"    Portfolio ROI: {portfolio['roi']:.1f}")

                portfolio_results.append({
                    'budget_usd': budget,
                    'n_interventions': len(rec['selected_interventions']),
                    'selected': ', '.join(rec['selected_interventions']),
                    'boost_factor': portfolio['combined_boost_factor'],
                    'modified_acceleration': portfolio['modified_acceleration'],
                    'delta_beneficiaries': portfolio['delta_beneficiaries'],
                    'roi': portfolio['roi'],
                })

        portfolio_df = pd.DataFrame(portfolio_results)

        # -------------------------------------------------------------------------
        # Step 7: Top Intervention Deep Dive
        # -------------------------------------------------------------------------
        print("\n[Step 7/8] Deep dive on top interventions...")

        top_3_keys = rankings.head(3)['intervention_key'].tolist()

        print("\nTop 3 Interventions - Detailed Analysis:")
        print("-" * 70)

        for key in top_3_keys:
            intervention = POLICY_INTERVENTIONS[key]
            effect = policy_module.estimate_intervention_effect_simple(
                key, baseline_acceleration, total_beneficiaries
            )

            print(f"\n  {intervention.name}")
            print(f"    Category: {intervention.category.value}")
            print(f"    Annual Cost: ${intervention.cost_usd/1e6:.0f}M")
            print(f"    Implementation Lag: {intervention.implementation_lag_years:.1f} years")
            print(f"    Evidence Quality: {intervention.evidence_quality}/5")
            print(f"    Acceleration Boost: {effect['acceleration_boost_factor']:.3f}x")
            print(f"    Additional Beneficiaries: {effect['delta_beneficiaries']/1e6:.1f}M")
            print(f"    Cost per QALY: ${effect['cost_per_qaly']:,.0f}")
            print(f"    ROI: {effect['roi']:.1f}")
            print(f"    Description: {intervention.description[:100]}...")

    # -------------------------------------------------------------------------
    # Step 8: Export Results
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Exporting results...")

    # Export main results
    results_path = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_path, index=False)
    print(f"  - Results: {results_path}")

    # Export summary
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  - Summary: {summary_path}")

    # Export parameters
    params_path = os.path.join(output_dir, 'parameters.json')
    model.export_parameters(params_path)
    print(f"  - Parameters: {params_path}")

    if run_policy_analysis:
        # Export policy rankings
        rankings_path = os.path.join(output_dir, 'policy_rankings.csv')
        rankings.to_csv(rankings_path, index=False)
        print(f"  - Policy rankings: {rankings_path}")

        # Export portfolio recommendations
        portfolio_path = os.path.join(output_dir, 'portfolio_recommendations.csv')
        portfolio_df.to_csv(portfolio_path, index=False)
        print(f"  - Portfolio recommendations: {portfolio_path}")

        # Export intervention catalog
        catalog_path = os.path.join(output_dir, 'intervention_catalog.csv')
        policy_module.list_interventions().to_csv(catalog_path, index=False)
        print(f"  - Intervention catalog: {catalog_path}")

    # -------------------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[Visualizations] Generating figures...")

    import matplotlib.pyplot as plt

    # Colorblind-safe palette
    colors = {
        'ai_investment': '#4575b4',
        'regulatory_reform': '#d73027',
        'data_infrastructure': '#1a9850',
        'talent_development': '#fdae61',
        'research_funding': '#762a83',
        'international_coordination': '#91bfdb',
    }

    if run_policy_analysis:
        # Figure 1: ROI by Intervention
        fig1, ax1 = plt.subplots(figsize=(14, 8))

        top_10 = rankings.head(10)
        y_pos = np.arange(len(top_10))

        bar_colors = [colors.get(cat, 'gray') for cat in top_10['category']]
        bars = ax1.barh(y_pos, top_10['roi'], color=bar_colors, alpha=0.8)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_10['intervention_name'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Return on Investment (ROI)', fontsize=12)
        ax1.set_title('Policy Intervention ROI Rankings\n(Value Generated / Cost)',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, roi in zip(bars, top_10['roi']):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{roi:.1f}', va='center', fontsize=10)

        # Legend
        legend_handles = [plt.Rectangle((0,0),1,1, color=colors[cat], alpha=0.8)
                         for cat in colors.keys()]
        legend_labels = [cat.replace('_', ' ').title() for cat in colors.keys()]
        ax1.legend(legend_handles, legend_labels, loc='lower right', fontsize=9)

        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'fig_policy_roi.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  - fig_policy_roi.png")

        # Figure 2: Cost vs Impact
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        for cat in colors.keys():
            cat_data = rankings[rankings['category'] == cat]
            ax2.scatter(cat_data['annual_cost_usd'] / 1e9,
                       cat_data['delta_beneficiaries'] / 1e6,
                       s=cat_data['evidence_quality'] * 50,
                       c=colors[cat], alpha=0.7,
                       label=cat.replace('_', ' ').title())

        ax2.set_xlabel('Annual Cost ($B)', fontsize=12)
        ax2.set_ylabel('Additional Beneficiaries (Millions)', fontsize=12)
        ax2.set_title('Policy Interventions: Cost vs Impact\n(Bubble size = evidence quality)',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'fig_cost_impact.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("  - fig_cost_impact.png")

        # Figure 3: Portfolio Frontier
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        ax3.plot(portfolio_df['budget_usd'] / 1e9,
                portfolio_df['modified_acceleration'],
                marker='o', markersize=10, linewidth=2.5, color='#d73027')

        ax3.axhline(y=baseline_acceleration, color='gray', linestyle='--',
                   label=f'Baseline ({baseline_acceleration:.1f}x)')

        for _, row in portfolio_df.iterrows():
            ax3.annotate(f"{row['n_interventions']} interventions\nROI={row['roi']:.0f}",
                        (row['budget_usd']/1e9, row['modified_acceleration']),
                        textcoords='offset points', xytext=(10, 5), fontsize=9)

        ax3.set_xlabel('Annual Budget ($B)', fontsize=12)
        ax3.set_ylabel('Acceleration Factor', fontsize=12)
        ax3.set_title('Portfolio Frontier: Acceleration vs Budget\n(Optimal intervention combinations)',
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)

        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'fig_portfolio_frontier.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  - fig_portfolio_frontier.png")

        # Figure 4: Category Comparison
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        categories = category_summary.index.tolist()
        x = np.arange(len(categories))
        width = 0.35

        ax4.bar(x - width/2, category_summary['roi'], width,
               label='Avg ROI', color='#4575b4', alpha=0.8)
        ax4.bar(x + width/2, category_summary['delta_beneficiaries'] / 1e8, width,
               label='Beneficiaries (100M)', color='#d73027', alpha=0.8)

        ax4.set_xlabel('Intervention Category', fontsize=12)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title('Policy Impact by Category', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        fig4.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'fig_category_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  - fig_category_comparison.png")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("v0.9 Model Run Complete")
    print("=" * 70)

    # Key findings
    baseline_result = results[results['scenario'] == 'Baseline']
    amodei_result = results[results['scenario'] == 'Upper_Bound_Amodei']

    baseline_2050 = baseline_result[baseline_result['year'] == 2050]['cumulative_progress'].iloc[0]
    amodei_2050 = amodei_result[amodei_result['year'] == 2050]['cumulative_progress'].iloc[0]

    print("\nKey Results:")
    print("-" * 40)
    print(f"Baseline 2050: {baseline_2050:.1f} equiv years ({baseline_2050/26:.1f}x acceleration)")
    print(f"Upper Bound 2050: {amodei_2050:.1f} equiv years ({amodei_2050/26:.1f}x acceleration)")

    if run_policy_analysis:
        print("\nPolicy Analysis Highlights:")
        print("-" * 40)
        top_intervention = rankings.iloc[0]
        print(f"  Top intervention: {top_intervention['intervention_name']}")
        print(f"    ROI: {top_intervention['roi']:.1f}")
        print(f"    Cost: ${top_intervention['annual_cost_usd']/1e6:.0f}M/year")
        print(f"    Acceleration boost: +{top_intervention['delta_acceleration']:.2f}x")

        print(f"\n  $10B Budget Portfolio:")
        rec = policy_module.recommend_portfolio(10_000_000_000, baseline_acceleration, total_beneficiaries)
        if rec['portfolio_analysis'].get('roi'):
            print(f"    Interventions: {len(rec['selected_interventions'])}")
            print(f"    Combined acceleration: {rec['portfolio_analysis']['modified_acceleration']:.1f}x")
            print(f"    Portfolio ROI: {rec['portfolio_analysis']['roi']:.1f}")

    print(f"\nAll outputs saved to: {output_dir}/")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Bio Acceleration Model v0.9')
    parser.add_argument('--skip-policy-analysis', action='store_true',
                        help='Skip policy analysis')
    args = parser.parse_args()

    model, results = main(
        run_policy_analysis=not args.skip_policy_analysis
    )
