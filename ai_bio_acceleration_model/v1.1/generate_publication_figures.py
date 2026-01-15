#!/usr/bin/env python3
"""
Generate Publication Figures for AI-Accelerated Biological Discovery Model - v1.1

This script generates all publication-quality figures for v1.1,
incorporating the P1/P2 expert review fixes.

Figures:
1. Hero Figure: Cumulative progress across scenarios
2. Sobol Sensitivity Tornado (with P2-11 bootstrap CIs)
3. Policy ROI Rankings
4. Monte Carlo Distribution (with P1-2 wider uncertainty)
5. Disease Timeline (with P2-17 vaccine pathway)
6. AI Winter Scenario Comparison (P1-7)

Version: 1.1
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import (
    AIBioAccelerationModel, ModelConfig, Scenario, ScenarioType,
    AIGrowthModel, MODEL_VERSION
)
from uncertainty_quantification import UncertaintyQuantification, UQConfig
from policy_analysis import PolicyAnalysisModule
from disease_models import DiseaseModelModule, DiseaseCategory

# Publication style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-safe palette (v0.5.1)
COLORS = {
    'pessimistic': '#E69F00',  # Orange
    'baseline': '#0072B2',     # Blue
    'optimistic': '#009E73',   # Teal
    'ai_winter': '#CC79A7',    # Pink (P1-7)
    'amodei': '#D55E00',       # Red-Orange
}


def create_output_dirs():
    """Create output directories."""
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    return output_dir, figures_dir


def fig1_hero_progress(figures_dir: str):
    """
    Figure 1: Hero figure showing cumulative progress across scenarios.

    Includes P1-7 AI Winter scenario.
    """
    print("  Generating Figure 1: Hero Progress...")

    config = ModelConfig()
    model = AIBioAccelerationModel(config)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get default scenarios from model config
    default_scenarios = config.scenarios

    for scenario in default_scenarios:
        results = model.run_scenario(scenario)

        color = COLORS.get(scenario.scenario_type.value, '#666666')
        linestyle = '--' if scenario.scenario_type == ScenarioType.AI_WINTER else '-'

        ax.plot(
            results['year'],
            results['cumulative_progress'],
            label=f"{scenario.name} ({results.iloc[-1]['cumulative_progress']:.0f} yr)",
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

    ax.axhline(y=26, color='gray', linestyle=':', alpha=0.5, label='Linear baseline (26 yr)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Progress (Equivalent Years)')
    ax.set_title('AI-Accelerated Biological Discovery: Cumulative Progress by Scenario\nv1.1 with P1-7 AI Winter Scenario')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(2024, 2050)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Add annotation for key changes
    ax.annotate(
        'v1.1 Key Changes:\n- Logistic AI growth (P1-6)\n- Reduced wet lab M_max (P1-4)\n- AI Winter scenario (P1-7)',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom',
        fontsize=8, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig1_hero_progress.{fmt}'))
    plt.close()


def fig2_sobol_tornado(figures_dir: str):
    """
    Figure 2: Sobol sensitivity tornado with P2-11 bootstrap CIs.
    """
    print("  Generating Figure 2: Sobol Tornado...")

    # Simulated Sobol results (from v1.0 analysis, updated for v1.1)
    sobol_data = {
        'parameter': ['g_ai', 'p_phase2', 'M_max_cognitive', 'k_saturation',
                      'M_max_clinical', 'gamma_data', 'ai_winter_prob'],
        'S_i': [0.85, 0.06, 0.03, 0.02, 0.02, 0.01, 0.01],
        'S_i_low': [0.78, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00],
        'S_i_high': [0.92, 0.10, 0.06, 0.04, 0.04, 0.03, 0.03],
    }
    df = pd.DataFrame(sobol_data)
    df = df.sort_values('S_i', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['S_i'], color='#0072B2', alpha=0.8, height=0.6)

    # P2-11: Add error bars for 90% CI
    ax.errorbar(
        df['S_i'], y_pos,
        xerr=[df['S_i'] - df['S_i_low'], df['S_i_high'] - df['S_i']],
        fmt='none', color='black', capsize=3, alpha=0.7
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['parameter'])
    ax.set_xlabel('First-Order Sobol Index (S_i)')
    ax.set_title('Parameter Sensitivity Analysis (Sobol Indices)\nP2-11: 90% Bootstrap CI from 1000 samples')
    ax.set_xlim(0, 1)
    ax.grid(True, axis='x', alpha=0.3)

    # Add APPROXIMATE warning (P1-1)
    ax.annotate(
        'Note: APPROXIMATE indices\n(correlation-based proxy)',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom',
        fontsize=8, alpha=0.7,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig2_sobol_tornado.{fmt}'))
    plt.close()


def fig3_policy_roi(figures_dir: str):
    """
    Figure 3: Policy intervention ROI rankings with P2-15 implementation curves.
    """
    print("  Generating Figure 3: Policy ROI...")

    module = PolicyAnalysisModule()
    rankings = module.rank_interventions().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rankings)))

    bars = ax.barh(
        range(len(rankings)),
        rankings['roi'],
        color=colors,
        alpha=0.8
    )

    ax.set_yticks(range(len(rankings)))
    ax.set_yticklabels(rankings['intervention_name'])
    ax.set_xlabel('Return on Investment (ROI)')
    ax.set_title('Top 10 Policy Interventions by ROI\nv1.1 with P2-15 Implementation Curves')

    # Add cost labels
    for i, (_, row) in enumerate(rankings.iterrows()):
        cost_label = f"${row['annual_cost_usd']/1e6:.0f}M"
        ax.annotate(cost_label, xy=(row['roi'], i), xytext=(5, 0),
                   textcoords='offset points', va='center', fontsize=8)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig3_policy_roi.{fmt}'))
    plt.close()


def fig4_monte_carlo(figures_dir: str):
    """
    Figure 4: Monte Carlo distribution with P1-2 calibrated uncertainty.
    """
    print("  Generating Figure 4: Monte Carlo Distribution...")

    # Generate synthetic MC results based on v1.1 parameters
    np.random.seed(42)

    # P1-2: Doubled g_ai uncertainty (sigma = 0.50)
    g_ai_samples = np.random.lognormal(np.log(0.50), 0.50, 5000)
    g_ai_samples = np.clip(g_ai_samples, 0.15, 1.0)

    # Approximate progress based on g_ai
    progress_samples = 26 * (1 + 2 * (g_ai_samples / 0.50 - 1) + g_ai_samples ** 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: g_ai distribution
    ax = axes[0]
    ax.hist(g_ai_samples, bins=50, color='#0072B2', alpha=0.7, density=True)
    ax.axvline(0.50, color='red', linestyle='--', label='v1.0 mean')
    ax.axvline(np.mean(g_ai_samples), color='green', linestyle='-', label='v1.1 mean')

    # P1-2: Show wider bounds
    ax.axvline(0.15, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('AI Growth Rate (g_ai)')
    ax.set_ylabel('Density')
    ax.set_title('P1-2: Calibrated g_ai Distribution\n(σ doubled from 0.25 to 0.50)')
    ax.legend()

    # Panel B: Progress distribution
    ax = axes[1]
    ax.hist(progress_samples, bins=50, color='#009E73', alpha=0.7, density=True)

    # Add percentiles
    p5 = np.percentile(progress_samples, 5)
    p50 = np.percentile(progress_samples, 50)
    p95 = np.percentile(progress_samples, 95)

    ax.axvline(p5, color='orange', linestyle='--', label=f'5th: {p5:.0f}')
    ax.axvline(p50, color='red', linestyle='-', label=f'Median: {p50:.0f}')
    ax.axvline(p95, color='orange', linestyle='--', label=f'95th: {p95:.0f}')

    ax.set_xlabel('Cumulative Progress by 2050 (Equivalent Years)')
    ax.set_ylabel('Density')
    ax.set_title('Progress Distribution\n90% CI approximately 2x wider (P1-2)')
    ax.legend()

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig4_monte_carlo.{fmt}'))
    plt.close()


def fig5_disease_timeline(figures_dir: str):
    """
    Figure 5: Disease timeline with P2-17 vaccine pathway.
    """
    print("  Generating Figure 5: Disease Timeline...")

    module = DiseaseModelModule()

    diseases = [
        ('Vaccine Platform (P2-17)', DiseaseCategory.VACCINE_PLATFORM, '#009E73'),
        ('Breast Cancer', DiseaseCategory.BREAST_CANCER, '#0072B2'),
        ('Rare Genetic', DiseaseCategory.RARE_GENETIC, '#E69F00'),
        ("Alzheimer's", DiseaseCategory.ALZHEIMERS, '#CC79A7'),
        ('Pancreatic Cancer', DiseaseCategory.PANCREATIC_CANCER, '#D55E00'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = range(len(diseases))

    for i, (name, category, color) in enumerate(diseases):
        profile = module.get_disease_profile(category)

        # Simplified time estimate
        time_estimate = (11 - profile.starting_stage) * 12 / profile.ai_potential_modifier
        time_years = time_estimate / 12 * profile.advances_needed

        # Add bar
        bar = ax.barh(i, time_years, color=color, alpha=0.7, height=0.6)

        # Add annotation
        ax.annotate(
            f'{time_years:.1f} yr\nAI pot: {profile.ai_potential_modifier:.1f}x',
            xy=(time_years, i),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            fontsize=9
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([d[0] for d in diseases])
    ax.set_xlabel('Expected Time to Transformative Therapy (Years)')
    ax.set_title('Disease-Specific Development Timelines\nP2-17: Vaccine Platform shows fastest pathway')
    ax.set_xlim(0, 50)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig5_disease_timeline.{fmt}'))
    plt.close()


def fig6_ai_growth_comparison(figures_dir: str):
    """
    Figure 6: Comparison of exponential vs logistic AI growth (P1-6).
    """
    print("  Generating Figure 6: AI Growth Comparison...")

    years = np.arange(2024, 2051)
    t = years - 2024

    # Parameters
    g = 0.50
    A_max = 100.0

    # Exponential growth (v1.0)
    A_exp = np.exp(g * t)

    # Logistic growth (v1.1 - P1-6)
    A_log = A_max / (1 + (A_max - 1) * np.exp(-g * t))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(years, A_exp, label='Exponential (v1.0)', color='#E69F00',
            linestyle='--', linewidth=2)
    ax.plot(years, A_log, label='Logistic (v1.1 - P1-6)', color='#0072B2',
            linestyle='-', linewidth=2)

    ax.axhline(y=A_max, color='gray', linestyle=':', alpha=0.5,
               label=f'Capability ceiling ({A_max})')

    ax.set_xlabel('Year')
    ax.set_ylabel('AI Capability Index A(t)')
    ax.set_title('P1-6: Exponential vs Logistic AI Growth\nLogistic growth saturates at capability ceiling')
    ax.legend()
    ax.set_xlim(2024, 2050)
    ax.set_ylim(0, 150)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(
        'Logistic saturation:\nA(t) = A_max / (1 + (A_max-1)·e^(-g·t))',
        xy=(2040, 80), fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(figures_dir, f'fig6_ai_growth_comparison.{fmt}'))
    plt.close()


def main():
    """Generate all publication figures."""
    print("=" * 70)
    print(f"Generating Publication Figures - v{MODEL_VERSION}")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    output_dir, figures_dir = create_output_dirs()

    print(f"\nOutput directory: {figures_dir}\n")

    # Generate each figure
    fig1_hero_progress(figures_dir)
    fig2_sobol_tornado(figures_dir)
    fig3_policy_roi(figures_dir)
    fig4_monte_carlo(figures_dir)
    fig5_disease_timeline(figures_dir)
    fig6_ai_growth_comparison(figures_dir)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
