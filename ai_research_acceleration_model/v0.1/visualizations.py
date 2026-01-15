#!/usr/bin/env python3
"""
Scientific Visualizations for AI Research Acceleration Model
=============================================================

Publication-quality figures following best practices in scientific visualization.
Design philosophy: Clean, informative, accessible.

Author: Scientific Visual Expert Agency
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import AIResearchAccelerationModel, Scenario, SimulationUnlock
from src.pipeline import ResearchPipeline
from src.paradigm_shift import ParadigmShiftModule


# Scientific visualization style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'light': '#E8E8E8',        # Light gray
    'unlock': '#28A745',       # Green for unlock
    'no_unlock': '#6C757D',    # Gray for no unlock
}

SCENARIO_COLORS = {
    'ai_winter': '#DC3545',
    'conservative': '#FD7E14',
    'baseline': '#2E86AB',
    'ambitious': '#28A745',
}


def setup_figure(figsize=(10, 6), dpi=150):
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def fig1_acceleration_trajectory():
    """
    Figure 1: Acceleration trajectory over time (2025-2050)
    Shows all four scenarios with confidence bands.
    """
    fig, ax = setup_figure(figsize=(12, 7))

    years = np.arange(2025, 2051)

    for scenario in Scenario:
        model = AIResearchAccelerationModel(scenario=scenario, enable_unlock=True)
        accelerations = [model.acceleration_factor(y - 2025) for y in years]

        ax.plot(years, accelerations,
                color=SCENARIO_COLORS[scenario.value],
                linewidth=2.5,
                label=f"{scenario.value.replace('_', ' ').title()}")

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acceleration Factor (×)', fontsize=12, fontweight='bold')
    ax.set_title('AI-Accelerated Scientific Research: Trajectory Projections\n(with Simulation Unlock potential)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(2025, 2050)
    ax.set_ylim(0, 50)

    # Add annotation for key insight
    ax.annotate('Physical constraints\nlimit acceleration',
                xy=(2035, 22), xytext=(2038, 35),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=1.5))

    plt.tight_layout()
    return fig


def fig2_pipeline_bottleneck():
    """
    Figure 2: Pipeline stage analysis showing bottleneck
    Horizontal bar chart of stage durations and multipliers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

    pipeline = ResearchPipeline()
    model = AIResearchAccelerationModel(enable_unlock=True)

    stages = [s.name for s in pipeline.stages]
    baseline_durations = [s.params.tau_0 for s in pipeline.stages]

    # Year 2050 projections
    t = 25
    A = model.ai_capability(t)
    effective_durations = [model.stage_duration(s, t) for s in pipeline.stages]
    multipliers = [model.effective_multiplier_with_constraints(s, t, A)
                   for s in pipeline.stages]

    # Color by stage type
    stage_colors = []
    for s in pipeline.stages:
        if s.params.stage_type.value == 'cognitive':
            stage_colors.append(COLORS['primary'])
        elif s.params.stage_type.value == 'physical':
            stage_colors.append(COLORS['success'])
        elif s.params.stage_type.value == 'hybrid':
            stage_colors.append(COLORS['accent'])
        else:
            stage_colors.append(COLORS['secondary'])

    # Left plot: Duration comparison
    y_pos = np.arange(len(stages))
    width = 0.35

    bars1 = axes[0].barh(y_pos - width/2, baseline_durations, width,
                         label='Baseline (2025)', color=COLORS['light'], edgecolor=COLORS['neutral'])
    bars2 = axes[0].barh(y_pos + width/2, effective_durations, width,
                         label='With AI (2050)', color=stage_colors, edgecolor=COLORS['neutral'])

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(stages)
    axes[0].set_xlabel('Duration (months)', fontsize=11, fontweight='bold')
    axes[0].set_title('Stage Duration Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].invert_yaxis()

    # Right plot: Multipliers
    bars3 = axes[1].barh(y_pos, multipliers, color=stage_colors, edgecolor=COLORS['neutral'])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(stages)
    axes[1].set_xlabel('Acceleration Multiplier (×)', fontsize=11, fontweight='bold')
    axes[1].set_title('AI Acceleration by Stage (2050)', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].set_xscale('log')

    # Highlight bottleneck
    bottleneck_idx = multipliers.index(min(multipliers))
    bars3[bottleneck_idx].set_edgecolor(COLORS['success'])
    bars3[bottleneck_idx].set_linewidth(3)

    # Legend for stage types
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['primary'], label='Cognitive'),
        mpatches.Patch(facecolor=COLORS['success'], label='Physical (bottleneck)'),
        mpatches.Patch(facecolor=COLORS['accent'], label='Hybrid'),
        mpatches.Patch(facecolor=COLORS['secondary'], label='Social'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('Research Pipeline: Bottleneck Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def fig3_simulation_unlock():
    """
    Figure 3: The Unlock - Impact of AI simulation tools
    Comparison of with/without simulation unlock scenarios.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    years = np.arange(2025, 2051)

    model_unlock = AIResearchAccelerationModel(scenario=Scenario.BASELINE, enable_unlock=True)
    model_no_unlock = AIResearchAccelerationModel(scenario=Scenario.BASELINE, enable_unlock=False)

    accel_unlock = [model_unlock.acceleration_factor(y - 2025) for y in years]
    accel_no_unlock = [model_no_unlock.acceleration_factor(y - 2025) for y in years]
    p_unlock = [model_unlock.simulation_unlock.p_unlock(y - 2025) for y in years]

    # Left plot: Acceleration comparison
    axes[0].fill_between(years, accel_no_unlock, accel_unlock,
                         alpha=0.3, color=COLORS['unlock'], label='Unlock potential')
    axes[0].plot(years, accel_no_unlock, color=COLORS['no_unlock'],
                 linewidth=2.5, linestyle='--', label='Without Unlock')
    axes[0].plot(years, accel_unlock, color=COLORS['unlock'],
                 linewidth=2.5, label='With Unlock')

    axes[0].set_xlabel('Year', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Acceleration Factor (×)', fontsize=11, fontweight='bold')
    axes[0].set_title('Impact of Simulation Unlock on Acceleration', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].set_xlim(2025, 2050)

    # Annotate the gap
    gap_2050 = accel_unlock[-1] - accel_no_unlock[-1]
    axes[0].annotate(f'+{gap_2050:.0f}× potential\nfrom simulation',
                     xy=(2050, (accel_unlock[-1] + accel_no_unlock[-1])/2),
                     xytext=(2042, 30),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', color=COLORS['neutral']))

    # Right plot: P(unlock) trajectory
    ax2 = axes[1]
    ax2.fill_between(years, 0, [p*100 for p in p_unlock],
                     alpha=0.3, color=COLORS['unlock'])
    ax2.plot(years, [p*100 for p in p_unlock], color=COLORS['unlock'], linewidth=2.5)

    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('P(Simulation Unlock) %', fontsize=11, fontweight='bold')
    ax2.set_title('Probability of Simulation Achieving Physical-Trial Equivalence',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(2025, 2050)
    ax2.set_ylim(0, 100)

    # Add milestone markers
    milestones = [(2030, 18, 'Early\nvalidation'), (2040, 32, 'Partial\nreplacement'), (2050, 47, 'Substantial\nreplacement')]
    for yr, p, label in milestones:
        ax2.plot(yr, p, 'o', color=COLORS['unlock'], markersize=10)
        ax2.annotate(label, xy=(yr, p), xytext=(yr, p+12),
                     fontsize=8, ha='center')

    plt.suptitle('The Unlock: AI Invents Simulation Tools to Replace Physical Trials',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def fig4_marginal_returns_framework():
    """
    Figure 4: Marginal Returns to Intelligence Framework
    Visualizing Amodei's bottleneck concept with our model parameters.
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Create funnel visualization
    stages = ['Raw\nIntelligence', 'Speed of\nPhysical World', 'Data\nScarcity',
              'Intrinsic\nComplexity', 'Human\nConstraints', 'Output']

    # Bottleneck widths (normalized flow)
    widths = [1.0, 0.3, 0.5, 0.7, 0.6, 0.25]

    x_positions = np.arange(len(stages)) * 2.5
    max_height = 3

    for i, (stage, width) in enumerate(zip(stages, widths)):
        # Draw bottleneck shape
        height = max_height * width
        bottom = (max_height - height) / 2

        if i == 0:
            # Input sphere
            circle = plt.Circle((x_positions[i], max_height/2), 1.2,
                                 color=COLORS['primary'], alpha=0.7)
            ax.add_patch(circle)
            ax.text(x_positions[i], max_height/2, stage, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        elif i == len(stages) - 1:
            # Output arrow
            arrow = mpatches.FancyArrow(x_positions[i]-0.5, max_height/2, 1.5, 0,
                                        width=height*0.8, head_width=height,
                                        head_length=0.5, color=COLORS['success'])
            ax.add_patch(arrow)
            ax.text(x_positions[i]+1.5, max_height/2, stage, ha='left', va='center',
                    fontsize=10, fontweight='bold')
        else:
            # Funnel shape
            funnel = mpatches.FancyBboxPatch(
                (x_positions[i]-0.4, bottom), 0.8, height,
                boxstyle="round,pad=0.05",
                facecolor=COLORS['light'], edgecolor=COLORS['neutral'], linewidth=2
            )
            ax.add_patch(funnel)
            ax.text(x_positions[i], max_height + 0.5, stage, ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # Bottleneck description
            descriptions = ['', 'Cell growth,\nlab time', 'Training data\nlimits',
                           'Chaos,\nunpredictability', 'Regulations,\nhabits']
            if descriptions[i]:
                ax.text(x_positions[i], -0.5, descriptions[i], ha='center', va='top',
                        fontsize=8, style='italic', color=COLORS['neutral'])

        # Draw connecting arrows
        if i > 0 and i < len(stages) - 1:
            ax.annotate('', xy=(x_positions[i]-0.5, max_height/2),
                        xytext=(x_positions[i-1]+0.5 if i > 1 else x_positions[i-1]+1.2, max_height/2),
                        arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=2))

    # The Unlock bypass arrow
    ax.annotate('', xy=(x_positions[-1]-0.5, max_height/2+1),
                xytext=(x_positions[0]+1.2, max_height/2+1),
                arrowprops=dict(arrowstyle='->', color=COLORS['unlock'], lw=3,
                               connectionstyle='arc3,rad=0.3'))
    ax.text((x_positions[0] + x_positions[-1])/2, max_height + 1.8,
            'The Unlock: AI simulation bypasses physical bottlenecks',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['unlock'])

    ax.set_xlim(-2, x_positions[-1] + 3)
    ax.set_ylim(-1.5, max_height + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('The Framework: Marginal Returns to Intelligence\n(Adapted from Amodei, 2024)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def fig5_scenario_comparison_heatmap():
    """
    Figure 5: Scenario × Year heatmap of acceleration factors.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    years = [2025, 2030, 2035, 2040, 2045, 2050]
    scenarios = list(Scenario)

    # Build data matrix
    data = np.zeros((len(scenarios), len(years)))
    for i, scenario in enumerate(scenarios):
        model = AIResearchAccelerationModel(scenario=scenario, enable_unlock=True)
        for j, year in enumerate(years):
            data[i, j] = model.acceleration_factor(year - 2025)

    # Create heatmap
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(years)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(years)
    ax.set_yticklabels([s.value.replace('_', ' ').title() for s in scenarios])

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(years)):
            text = ax.text(j, i, f'{data[i, j]:.0f}×',
                          ha='center', va='center', color='white' if data[i,j] > 20 else 'black',
                          fontsize=10, fontweight='bold')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_title('Acceleration Factor by Scenario and Year\n(with Simulation Unlock)',
                 fontsize=14, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Acceleration (×)', fontsize=11)

    plt.tight_layout()
    return fig


def generate_all_figures(output_dir: str = 'figures'):
    """Generate all figures and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = [
        ('fig1_acceleration_trajectory.png', fig1_acceleration_trajectory),
        ('fig2_pipeline_bottleneck.png', fig2_pipeline_bottleneck),
        ('fig3_simulation_unlock.png', fig3_simulation_unlock),
        ('fig4_marginal_returns_framework.png', fig4_marginal_returns_framework),
        ('fig5_scenario_heatmap.png', fig5_scenario_comparison_heatmap),
    ]

    for filename, fig_func in figures:
        print(f"Generating {filename}...")
        fig = fig_func()
        fig.savefig(output_path / filename, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved to {output_path / filename}")

    print(f"\nAll figures saved to {output_path}/")


if __name__ == "__main__":
    generate_all_figures()
