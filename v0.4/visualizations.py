#!/usr/bin/env python3
"""
v0.4 Visualizations
===================

Publication-quality figures for the refined model.

Figures:
1. Domain comparison - acceleration by domain over time
2. Shift type patterns - Type I vs II vs III
3. Backlog dynamics - hypothesis accumulation
4. Stage breakdown - cognitive vs physical stages
5. Model comparison - v0.1 vs v0.4 predictions
6. Validation scatter - predicted vs observed
"""

import sys
from pathlib import Path
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refined_model import (
    RefinedAccelerationModel,
    DOMAIN_PROFILES,
    ShiftType,
    Scenario,
)
from backlog_dynamics import BacklogModel, VALIDATION_CAPACITIES

# Check matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Generating text-based summaries.")


def setup_style():
    """Setup publication-quality plotting style."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def fig1_domain_comparison(output_dir: Path):
    """Figure 1: Acceleration by domain over time."""
    print("Generating Figure 1: Domain Comparison...")

    years = list(range(2025, 2051, 5))
    domains = list(DOMAIN_PROFILES.keys())

    # Gather data
    data = {}
    for domain in domains:
        model = RefinedAccelerationModel(domain=domain)
        forecast = model.forecast(years)
        data[domain] = [forecast[y]["acceleration"] for y in years]

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary:")
        for domain, accels in data.items():
            print(f"    {domain}: {accels[0]:.1f}x -> {accels[-1]:.1f}x")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    for i, domain in enumerate(domains):
        label = DOMAIN_PROFILES[domain].name
        ax.plot(years, data[domain], 'o-', color=colors[i], label=label, linewidth=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('End-to-End Acceleration (x)')
    ax.set_title('AI Research Acceleration by Domain\nv0.4 Refined Model with Domain-Specific Parameters')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1, 10)

    # Add annotation
    ax.annotate(
        'Physical bottleneck limits\nall domains to <10x end-to-end',
        xy=(2045, 3), fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_domain_comparison.png')
    plt.close()
    print("  Saved: fig1_domain_comparison.png")


def fig2_shift_type_patterns(output_dir: Path):
    """Figure 2: Type I vs Type II vs Type III shift patterns."""
    print("Generating Figure 2: Shift Type Patterns...")

    # Representative domains for each shift type
    type_examples = {
        'Type I (Scale)': 'materials_science',
        'Type II (Efficiency)': 'genomics',
        'Type III (Capability)': 'structural_biology',
    }

    years = list(range(2020, 2051))

    data = {}
    for shift_name, domain in type_examples.items():
        model = RefinedAccelerationModel(domain=domain)
        forecast = model.forecast(years)
        data[shift_name] = {
            'acceleration': [forecast[y]["acceleration"] for y in years],
            'cognitive': [max(forecast[y]["stage_accelerations"][s]
                            for s in ["S1", "S2", "S3", "S5"]) for y in years],
        }

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary:")
        for name, vals in data.items():
            print(f"    {name}: End-to-end {vals['acceleration'][-1]:.1f}x, "
                  f"Cognitive {vals['cognitive'][-1]:.0f}x")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (shift_name, shift_data) in enumerate(data.items()):
        ax = axes[i]

        ax.plot(years, shift_data['acceleration'], 'b-', linewidth=2,
                label='End-to-End')
        ax.plot(years, shift_data['cognitive'], 'r--', linewidth=2,
                label='Cognitive Stage')

        ax.set_xlabel('Year')
        ax.set_ylabel('Acceleration (x)')
        ax.set_title(shift_name)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(1, 10000)

    fig.suptitle('Shift Type Comparison: Cognitive vs End-to-End Acceleration',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_shift_type_patterns.png')
    plt.close()
    print("  Saved: fig2_shift_type_patterns.png")


def fig3_backlog_dynamics(output_dir: Path):
    """Figure 3: Backlog accumulation over time."""
    print("Generating Figure 3: Backlog Dynamics...")

    # GNoME-style simulation
    model = BacklogModel(
        domain="materials_science",
        ai_generation_rate=2200000,
        triage_efficiency=0.01,
    )

    trajectory = model.simulate_trajectory(
        start_year=2023,
        end_year=2050,
        ai_growth_rate=0.5,
        automation_growth_rate=0.15,
    )

    years = [m.year for m in trajectory]
    backlogs = [m.backlog_size for m in trajectory]
    backlog_years = [m.backlog_years for m in trajectory]

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary:")
        print(f"    2023: Backlog {backlogs[0]:,.0f} ({backlog_years[0]:,.0f} years)")
        print(f"    2050: Backlog {backlogs[-1]:,.0f} ({backlog_years[-1]:,.0f} years)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Backlog size
    ax1.semilogy(years, backlogs, 'b-', linewidth=2)
    ax1.fill_between(years, 1, backlogs, alpha=0.3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Backlog Size (hypotheses)')
    ax1.set_title('Hypothesis Backlog Accumulation\n(GNoME Pattern)')
    ax1.grid(True, alpha=0.3)

    # Years to clear
    ax2.semilogy(years, backlog_years, 'r-', linewidth=2)
    ax2.axhline(y=100, color='gray', linestyle='--', label='100 years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Years to Clear Backlog')
    ax2.set_title('Time to Validate All Hypotheses\n(at current synthesis rate)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Type I Shift Creates Selection Problem, Not Speed',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_backlog_dynamics.png')
    plt.close()
    print("  Saved: fig3_backlog_dynamics.png")


def fig4_stage_breakdown(output_dir: Path):
    """Figure 4: Stage-level acceleration breakdown."""
    print("Generating Figure 4: Stage Breakdown...")

    domain = "structural_biology"
    model = RefinedAccelerationModel(domain=domain)
    forecast = model.forecast([2025, 2030, 2040, 2050])

    stages = list(model.STAGES.keys())
    stage_names = [model.STAGES[s][0][:15] for s in stages]

    data = {year: [forecast[year]["stage_accelerations"][s] for s in stages]
            for year in forecast}

    if not MATPLOTLIB_AVAILABLE:
        print(f"  Text summary for {domain}:")
        for year, accels in data.items():
            print(f"    {year}: {dict(zip(stages, [f'{a:.1f}x' for a in accels]))}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(stages))
    width = 0.2
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(data)))

    for i, (year, accels) in enumerate(data.items()):
        ax.bar(x + i * width, accels, width, label=str(year), color=colors[i])

    ax.set_ylabel('Acceleration (x)')
    ax.set_title(f'Stage-Level Acceleration: {DOMAIN_PROFILES[domain].name}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stage_names, rotation=45, ha='right')
    ax.legend(title='Year')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Mark physical stages
    for i, s in enumerate(stages):
        if not model.STAGES[s][1]:  # Not cognitive
            ax.axvspan(i - 0.3, i + 0.7, alpha=0.2, color='red')

    # Legend for physical stages
    ax.annotate('Shaded = Physical stages (bottleneck)',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_stage_breakdown.png')
    plt.close()
    print("  Saved: fig4_stage_breakdown.png")


def fig5_model_comparison(output_dir: Path):
    """Figure 5: v0.1 vs v0.4 model comparison."""
    print("Generating Figure 5: Model Comparison...")

    years = list(range(2025, 2051, 5))

    # v0.4 projections for different domains
    v04_data = {}
    for domain in ['structural_biology', 'materials_science', 'average_biology']:
        model = RefinedAccelerationModel(domain=domain)
        forecast = model.forecast(years)
        v04_data[domain] = [forecast[y]["acceleration"] for y in years]

    # v0.1 approximation (single domain-agnostic curve)
    # Based on v0.1 model behavior
    v01_data = [1.5, 2.5, 4.0, 6.0, 15.0, 38.0]  # Approximate from v0.1

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary:")
        print(f"    v0.1: {v01_data}")
        for domain, accels in v04_data.items():
            print(f"    v0.4 {domain}: {accels}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # v0.1 line
    ax.plot(years, v01_data, 'k--', linewidth=3, label='v0.1 (Domain-agnostic)',
            marker='s', markersize=8)

    # v0.4 lines
    colors = {'structural_biology': 'blue', 'materials_science': 'red',
              'average_biology': 'green'}
    for domain, accels in v04_data.items():
        ax.plot(years, accels, 'o-', linewidth=2, color=colors[domain],
                label=f'v0.4 - {DOMAIN_PROFILES[domain].name}', markersize=6)

    ax.set_xlabel('Year')
    ax.set_ylabel('End-to-End Acceleration (x)')
    ax.set_title('Model Version Comparison: v0.1 vs v0.4\nDomain-Agnostic vs Domain-Specific Projections')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1, 100)

    # Annotation
    ax.annotate(
        'v0.1 overestimates: assumes\nM_max_physical = 2.5x\n(Case studies show 1.0-1.5x)',
        xy=(2040, 10), fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_model_comparison.png')
    plt.close()
    print("  Saved: fig5_model_comparison.png")


def fig6_validation_scatter(output_dir: Path):
    """Figure 6: Predicted vs observed acceleration."""
    print("Generating Figure 6: Validation Scatter...")

    # Case study data
    case_studies = {
        'AlphaFold 2/3': {'domain': 'structural_biology', 'year': 2021, 'observed': 24.0},
        'GNoME': {'domain': 'materials_science', 'year': 2023, 'observed': 1.0},
        'ESM-3': {'domain': 'protein_design', 'year': 2024, 'observed': 4.0},
    }

    predictions = {}
    for name, cs in case_studies.items():
        model = RefinedAccelerationModel(domain=cs['domain'])
        forecast = model.forecast([cs['year']])
        predictions[name] = forecast[cs['year']]['acceleration']

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary:")
        for name, cs in case_studies.items():
            print(f"    {name}: observed={cs['observed']:.1f}x, predicted={predictions[name]:.1f}x")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    observed = [case_studies[n]['observed'] for n in case_studies]
    predicted = [predictions[n] for n in case_studies]

    # Perfect prediction line
    ax.plot([0.1, 100], [0.1, 100], 'k--', label='Perfect prediction', alpha=0.5)

    # 3x bounds
    ax.fill_between([0.1, 100], [0.1/3, 100/3], [0.3, 300],
                    alpha=0.2, color='green', label='Within 3x')

    # Data points
    for i, name in enumerate(case_studies.keys()):
        ax.scatter(observed[i], predicted[i], s=200, zorder=5)
        ax.annotate(name, (observed[i], predicted[i]),
                    textcoords='offset points', xytext=(10, 10), fontsize=10)

    ax.set_xlabel('Observed End-to-End Acceleration (x)')
    ax.set_ylabel('v0.4 Predicted Acceleration (x)')
    ax.set_title('Model Validation: Predicted vs Observed\n(Full Pipeline with Physical Stages)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.5, 50)
    ax.set_ylim(0.5, 50)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Key insight annotation
    ax.annotate(
        'AlphaFold 24x: achievable when\nstructure IS the deliverable\n(no physical S4/S6 needed)',
        xy=(24, 1.6), xytext=(5, 5),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_validation_scatter.png')
    plt.close()
    print("  Saved: fig6_validation_scatter.png")


def generate_all_figures():
    """Generate all figures for v0.4."""
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING v0.4 VISUALIZATIONS")
    print("=" * 60)
    print()

    if MATPLOTLIB_AVAILABLE:
        setup_style()

    fig1_domain_comparison(output_dir)
    fig2_shift_type_patterns(output_dir)
    fig3_backlog_dynamics(output_dir)
    fig4_stage_breakdown(output_dir)
    fig5_model_comparison(output_dir)
    fig6_validation_scatter(output_dir)

    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
