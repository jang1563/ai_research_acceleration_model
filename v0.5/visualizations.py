#!/usr/bin/env python3
"""
v0.5 Visualizations
===================

Publication-quality figures for the integrated model with lab automation.

Figures:
1. v0.4 vs v0.5 comparison - Impact of lab automation
2. Automation scenarios - Conservative to breakthrough
3. Physical bottleneck unlock - Before/after automation
4. Cost dynamics - Lab automation reduces costs
5. Domain-specific trajectories - With automation
6. Scenario matrix heatmap - AI x Automation combinations
"""

import sys
from pathlib import Path
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "v0.4" / "src"))

from autonomous_lab import (
    AutonomousLabModel,
    AutomationScenario,
    LAB_CAPACITIES,
)
from integrated_model import (
    IntegratedAccelerationModel,
)
from refined_model import (
    RefinedAccelerationModel,
    DOMAIN_PROFILES,
    Scenario,
)

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


def fig1_v04_vs_v05(output_dir: Path):
    """Figure 1: v0.4 vs v0.5 comparison across domains."""
    print("Generating Figure 1: v0.4 vs v0.5 Comparison...")

    domains = list(DOMAIN_PROFILES.keys())
    years = [2025, 2030, 2040, 2050]

    v04_data = {}
    v05_data = {}

    for domain in domains:
        v04 = RefinedAccelerationModel(domain=domain)
        v05 = IntegratedAccelerationModel(domain=domain)

        v04_data[domain] = [v04.forecast([y])[y]["acceleration"] for y in years]
        v05_data[domain] = [v05.forecast([y])[y].end_to_end_acceleration for y in years]

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary (2050):")
        for domain in domains:
            print(f"    {domain}: v0.4={v04_data[domain][-1]:.1f}x, "
                  f"v0.5={v05_data[domain][-1]:.1f}x")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, domain in enumerate(domains):
        ax = axes[i]

        ax.plot(years, v04_data[domain], 'b-o', linewidth=2,
                label='v0.4 (No automation)', markersize=8)
        ax.plot(years, v05_data[domain], 'r-s', linewidth=2,
                label='v0.5 (With automation)', markersize=8)

        ax.fill_between(years, v04_data[domain], v05_data[domain],
                       alpha=0.3, color='green', label='Automation gain')

        ax.set_xlabel('Year')
        ax.set_ylabel('End-to-End Acceleration (x)')
        ax.set_title(DOMAIN_PROFILES[domain].name)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        if max(v05_data[domain]) > 10:
            ax.set_yscale('log')

    fig.suptitle('Impact of Lab Automation on Research Acceleration\nv0.4 (AI only) vs v0.5 (AI + Automation)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_v04_vs_v05_comparison.png')
    plt.close()
    print("  Saved: fig1_v04_vs_v05_comparison.png")


def fig2_automation_scenarios(output_dir: Path):
    """Figure 2: Different automation adoption scenarios."""
    print("Generating Figure 2: Automation Scenarios...")

    domain = "materials_science"
    years = list(range(2025, 2051, 5))

    scenario_data = {}
    for scenario in AutomationScenario:
        model = IntegratedAccelerationModel(
            domain=domain,
            automation_scenario=scenario,
        )
        forecasts = model.forecast(years)
        scenario_data[scenario.value] = [
            forecasts[y].end_to_end_acceleration for y in years
        ]

    if not MATPLOTLIB_AVAILABLE:
        print(f"  Text summary ({domain}, 2050):")
        for name, accels in scenario_data.items():
            print(f"    {name}: {accels[-1]:.1f}x")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'conservative': 'blue', 'baseline': 'green',
              'optimistic': 'orange', 'breakthrough': 'red'}
    linestyles = {'conservative': '--', 'baseline': '-',
                  'optimistic': '-', 'breakthrough': '-'}

    for name, accels in scenario_data.items():
        ax.plot(years, accels, color=colors[name], linestyle=linestyles[name],
                linewidth=2, marker='o', label=name.capitalize())

    ax.set_xlabel('Year')
    ax.set_ylabel('End-to-End Acceleration (x)')
    ax.set_title(f'Automation Scenario Comparison: {DOMAIN_PROFILES[domain].name}')
    ax.legend(title='Scenario')
    ax.grid(True, alpha=0.3)

    ax.annotate(
        '"Breakthrough" scenario assumes\nmajor advances in automated synthesis',
        xy=(2040, scenario_data['breakthrough'][3]), xytext=(2030, 8),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_automation_scenarios.png')
    plt.close()
    print("  Saved: fig2_automation_scenarios.png")


def fig3_bottleneck_unlock(output_dir: Path):
    """Figure 3: Physical bottleneck before/after automation."""
    print("Generating Figure 3: Bottleneck Unlock...")

    domain = "average_biology"
    year = 2040

    # v0.4 stage breakdown
    v04 = RefinedAccelerationModel(domain=domain)
    v04_forecast = v04.forecast([year])[year]

    # v0.5 stage breakdown
    v05 = IntegratedAccelerationModel(domain=domain)
    v05_forecast = v05.forecast([year])[year]

    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stage_names = ['Literature', 'Hypothesis', 'Design', 'Wet Lab', 'Interpret', 'Validate']

    v04_accels = [v04_forecast['stage_accelerations'][s] for s in stages]
    v05_accels = [v05_forecast.stage_accelerations[s] for s in stages]

    if not MATPLOTLIB_AVAILABLE:
        print(f"  Text summary ({domain}, {year}):")
        for s, n, a4, a5 in zip(stages, stage_names, v04_accels, v05_accels):
            print(f"    {s} ({n}): v0.4={a4:.1f}x, v0.5={a5:.1f}x")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(stages))
    width = 0.35

    # v0.4 plot
    colors_v04 = ['blue' if v04.STAGES[s][1] else 'red' for s in stages]
    ax1.bar(x, v04_accels, width, color=colors_v04, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_names, rotation=45, ha='right')
    ax1.set_ylabel('Stage Acceleration (x)')
    ax1.set_title('v0.4: Physical Bottleneck Limits Gains')
    ax1.set_yscale('log')
    ax1.axhline(y=1.5, color='red', linestyle='--', label='Physical ceiling')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # v0.5 plot
    colors_v05 = ['blue' if v05.STAGES[s][1] else 'green' for s in stages]
    ax2.bar(x, v05_accels, width, color=colors_v05, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_names, rotation=45, ha='right')
    ax2.set_ylabel('Stage Acceleration (x)')
    ax2.set_title('v0.5: Automation Unlocks Physical Stages')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    # Legend
    blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Cognitive (AI)')
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Physical (Manual)')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Physical (Automated)')
    ax2.legend(handles=[blue_patch, green_patch])

    fig.suptitle(f'Bottleneck Unlock: {year}\nBlue = Cognitive stages, Red/Green = Physical stages',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_bottleneck_unlock.png')
    plt.close()
    print("  Saved: fig3_bottleneck_unlock.png")


def fig4_cost_dynamics(output_dir: Path):
    """Figure 4: Cost reduction from automation."""
    print("Generating Figure 4: Cost Dynamics...")

    domain = "materials_science"
    years = list(range(2025, 2051, 5))

    # Get cost projections
    model = IntegratedAccelerationModel(domain=domain)
    forecasts = model.forecast(years)

    costs = [forecasts[y].cost_per_project / 1000 for y in years]  # In thousands
    reductions = [forecasts[y].cost_reduction * 100 for y in years]

    if not MATPLOTLIB_AVAILABLE:
        print(f"  Text summary ({domain}):")
        for y, c, r in zip(years, costs, reductions):
            print(f"    {y}: ${c:.0f}K ({r:.0f}% reduction)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cost over time
    ax1.plot(years, costs, 'g-o', linewidth=2, markersize=8)
    ax1.fill_between(years, costs, alpha=0.3, color='green')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cost per Project ($K)')
    ax1.set_title('Project Cost Reduction from Lab Automation')
    ax1.grid(True, alpha=0.3)

    # Cumulative reduction
    ax2.bar(years, reductions, color='green', alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cost Reduction (%)')
    ax2.set_title('Cumulative Cost Savings vs 2020 Baseline')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Cost Dynamics: {DOMAIN_PROFILES[domain].name}',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_cost_dynamics.png')
    plt.close()
    print("  Saved: fig4_cost_dynamics.png")


def fig5_domain_trajectories(output_dir: Path):
    """Figure 5: Domain-specific acceleration trajectories with automation."""
    print("Generating Figure 5: Domain Trajectories...")

    domains = list(DOMAIN_PROFILES.keys())
    years = list(range(2025, 2051, 5))

    data = {}
    for domain in domains:
        model = IntegratedAccelerationModel(domain=domain)
        forecasts = model.forecast(years)
        data[domain] = [forecasts[y].end_to_end_acceleration for y in years]

    if not MATPLOTLIB_AVAILABLE:
        print("  Text summary (2050):")
        for domain, accels in data.items():
            print(f"    {domain}: {accels[-1]:.1f}x")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    for i, domain in enumerate(domains):
        label = DOMAIN_PROFILES[domain].name
        ax.plot(years, data[domain], 'o-', color=colors[i],
                label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Year')
    ax.set_ylabel('End-to-End Acceleration (x)')
    ax.set_title('Acceleration Trajectories by Domain\nv0.5 Integrated Model (AI + Lab Automation)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Use log scale if range is large
    max_val = max(max(d) for d in data.values())
    if max_val > 20:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_domain_trajectories.png')
    plt.close()
    print("  Saved: fig5_domain_trajectories.png")


def fig6_scenario_heatmap(output_dir: Path):
    """Figure 6: AI x Automation scenario matrix heatmap."""
    print("Generating Figure 6: Scenario Heatmap...")

    domain = "average_biology"
    year = 2050

    ai_scenarios = list(Scenario)
    auto_scenarios = list(AutomationScenario)

    # Build matrix
    matrix = np.zeros((len(ai_scenarios), len(auto_scenarios)))
    for i, ai_scen in enumerate(ai_scenarios):
        for j, auto_scen in enumerate(auto_scenarios):
            model = IntegratedAccelerationModel(
                domain=domain,
                ai_scenario=ai_scen,
                automation_scenario=auto_scen,
            )
            matrix[i, j] = model.forecast([year])[year].end_to_end_acceleration

    if not MATPLOTLIB_AVAILABLE:
        print(f"  Text summary ({domain}, {year}):")
        for i, ai in enumerate(ai_scenarios):
            for j, auto in enumerate(auto_scenarios):
                print(f"    {ai.value}/{auto.value}: {matrix[i,j]:.1f}x")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(len(auto_scenarios)))
    ax.set_xticklabels([s.value.capitalize() for s in auto_scenarios], rotation=45, ha='right')
    ax.set_yticks(range(len(ai_scenarios)))
    ax.set_yticklabels([s.value.capitalize() for s in ai_scenarios])

    ax.set_xlabel('Lab Automation Scenario')
    ax.set_ylabel('AI Capability Scenario')
    ax.set_title(f'Scenario Matrix: End-to-End Acceleration ({year})\n{DOMAIN_PROFILES[domain].name}')

    # Add values to cells
    for i in range(len(ai_scenarios)):
        for j in range(len(auto_scenarios)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}x',
                          ha='center', va='center', color='black', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Acceleration (x)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_scenario_heatmap.png')
    plt.close()
    print("  Saved: fig6_scenario_heatmap.png")


def generate_all_figures():
    """Generate all figures for v0.5."""
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING v0.5 VISUALIZATIONS")
    print("=" * 60)
    print()

    if MATPLOTLIB_AVAILABLE:
        setup_style()

    fig1_v04_vs_v05(output_dir)
    fig2_automation_scenarios(output_dir)
    fig3_bottleneck_unlock(output_dir)
    fig4_cost_dynamics(output_dir)
    fig5_domain_trajectories(output_dir)
    fig6_scenario_heatmap(output_dir)

    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
