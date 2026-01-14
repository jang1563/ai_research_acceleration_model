#!/usr/bin/env python3
"""
Case Study Validation Visualizations
====================================

Publication-quality figures for v0.3 case study validation.

Figures:
1. Validation comparison (predicted vs observed)
2. Acceleration by pipeline stage
3. Bottleneck analysis
4. Type I vs Type III shift comparison
5. Historical trajectory with case studies
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from case_study_framework import CaseStudyValidator, ShiftType
from alphafold_case_study import AlphaFoldCaseStudy, compare_to_baseline
from gnome_case_study import GNoMECaseStudy, gnome_synthesis_bottleneck_analysis
from esm3_case_study import ESM3CaseStudy, esm3_bottleneck_analysis

# Add v0.1 for model
sys.path.insert(0, str(Path(__file__).parent.parent / "v0.1"))
from src.model import AIResearchAccelerationModel, Scenario

# Check matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be text-based.")


def setup_style():
    """Set up publication-quality style."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def fig1_validation_comparison():
    """
    Figure 1: Predicted vs Observed Acceleration

    Scatter plot comparing model predictions to case study observations.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 1: Validation Comparison (text mode)")
        print("=" * 50)
        print("AlphaFold: Predicted 0.5x, Observed 24.3x")
        print("GNoME:     Predicted 1.0x, Observed 365.0x")
        print("ESM-3:     Predicted 1.4x, Observed 4.0x")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Data
    case_studies = {
        'AlphaFold 2/3': {'predicted': 0.5, 'observed': 24.3, 'type': 'capability', 'year': 2021},
        'GNoME': {'predicted': 1.0, 'observed': 365.0, 'type': 'scale', 'year': 2023},
        'ESM-3': {'predicted': 1.4, 'observed': 4.0, 'type': 'capability', 'year': 2024},
    }

    colors = {'capability': '#2E86AB', 'scale': '#A23B72'}
    markers = {'capability': 'o', 'scale': 's'}

    # Plot each case study
    for name, data in case_studies.items():
        ax.scatter(
            data['predicted'], data['observed'],
            c=colors[data['type']],
            marker=markers[data['type']],
            s=200,
            label=f"{name} ({data['year']})",
            edgecolors='black',
            linewidths=1,
            zorder=5,
        )

    # Perfect prediction line
    ax.plot([0.1, 1000], [0.1, 1000], 'k--', alpha=0.3, label='Perfect Prediction')

    # 10x error bands
    ax.fill_between(
        [0.1, 1000], [0.01, 100], [1, 10000],
        alpha=0.1, color='green',
        label='Within 10x'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(1, 1000)

    ax.set_xlabel('Model Predicted Acceleration (x)')
    ax.set_ylabel('Observed Acceleration (x)')
    ax.set_title('Case Study Validation: Model Predictions vs. Observations')

    # Add annotations
    for name, data in case_studies.items():
        offset = (10, 10) if name != 'ESM-3' else (10, -15)
        ax.annotate(
            name,
            (data['predicted'], data['observed']),
            xytext=offset,
            textcoords='offset points',
            fontsize=9,
            ha='left',
        )

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Save
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig1_validation_comparison.png")
    plt.close()
    print("Saved: fig1_validation_comparison.png")


def fig2_stage_acceleration():
    """
    Figure 2: Acceleration by Pipeline Stage

    Bar chart showing observed acceleration at each pipeline stage
    for each case study.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 2: Stage Acceleration (text mode)")
        print("=" * 50)
        print("AlphaFold S3: 36,500x | S4: 1.5x | S6: 1.5x")
        print("GNoME S2/S3: 100,000x | S4: 1.0x")
        print("ESM-3 S2/S3: 30,000x | S4: 1.0x | S5: 1.5x")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stage_labels = ['Lit Review', 'Hypothesis', 'Analysis', 'Wet Lab', 'Interpret', 'Validation']

    # AlphaFold data
    alphafold_accel = [15, 14, 36500, 1.5, 4.3, 1.5]
    axes[0].bar(stages, alphafold_accel, color='#2E86AB', edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_ylim(1, 100000)
    axes[0].set_title('AlphaFold 2/3')
    axes[0].set_ylabel('Acceleration (x)')
    axes[0].axhline(y=25, color='red', linestyle='--', alpha=0.5, label='M_max cognitive')
    axes[0].axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, label='M_max physical')

    # GNoME data
    gnome_accel = [8.6, 30000, 70000, 1.0, 4.3, 2.0]
    axes[1].bar(stages, gnome_accel, color='#A23B72', edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1, 100000)
    axes[1].set_title('GNoME')
    axes[1].axhline(y=25, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=2.5, color='orange', linestyle='--', alpha=0.5)

    # ESM-3 data
    esm3_accel = [10, 6000, 30000, 1.0, 1.5, 1.5]
    axes[2].bar(stages, esm3_accel, color='#2E86AB', edgecolor='black')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1, 100000)
    axes[2].set_title('ESM-3')
    axes[2].axhline(y=25, color='red', linestyle='--', alpha=0.5)
    axes[2].axhline(y=2.5, color='orange', linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right', fontsize=8)

    for ax in axes:
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Stage-Level Acceleration by Case Study', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig2_stage_acceleration.png")
    plt.close()
    print("Saved: fig2_stage_acceleration.png")


def fig3_bottleneck_analysis():
    """
    Figure 3: Bottleneck Analysis

    Visual comparison of where bottlenecks occur across case studies.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 3: Bottleneck Analysis (text mode)")
        print("=" * 50)
        print("All case studies bottleneck at S4 (Wet Lab) or S6 (Validation)")
        print("Model correctly predicts physical stages as limiting factors")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    case_studies = ['AlphaFold', 'GNoME', 'ESM-3']
    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

    # Create heatmap-like visualization
    # 0 = minimal, 1 = moderate, 2 = bottleneck
    bottleneck_matrix = np.array([
        [0, 0, 0, 2, 0, 1],  # AlphaFold: S4 bottleneck, S6 secondary
        [0, 0, 0, 2, 1, 0],  # GNoME: S4 primary, S5 secondary
        [0, 0, 0, 2, 1, 0],  # ESM-3: S4 primary, S5 secondary
    ])

    # Colors: green (no bottleneck) to red (bottleneck)
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(bottleneck_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(['Lit Review', 'Hypothesis', 'Analysis', 'Wet Lab', 'Interpret', 'Validation'])
    ax.set_yticks(range(len(case_studies)))
    ax.set_yticklabels(case_studies)

    # Add text annotations
    for i in range(len(case_studies)):
        for j in range(len(stages)):
            text = ['', 'Secondary', 'PRIMARY'][bottleneck_matrix[i, j]]
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   color='white' if bottleneck_matrix[i, j] == 2 else 'black')

    ax.set_title('Bottleneck Identification: Model Prediction Validated')
    ax.set_xlabel('Pipeline Stage')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['No Constraint', 'Secondary', 'Primary Bottleneck'])

    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig3_bottleneck_analysis.png")
    plt.close()
    print("Saved: fig3_bottleneck_analysis.png")


def fig4_shift_type_comparison():
    """
    Figure 4: Type I vs Type III Shifts

    Compare acceleration patterns between scale (Type I) and
    capability (Type III) shifts.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 4: Shift Type Comparison (text mode)")
        print("=" * 50)
        print("Type I (Scale): Massive hypothesis generation, synthesis bottleneck")
        print("Type III (Capability): New abilities, validation bottleneck")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Type III (Capability) - AlphaFold, ESM-3
    ax1 = axes[0]
    categories = ['Stage\nAcceleration', 'End-to-End\nAcceleration', 'Time\nSavings']
    type_iii_data = [
        [36500, 30000],  # Stage acceleration
        [24, 4],         # End-to-end
        [730/30, 180/45], # Time savings ratio
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, [d[0] for d in type_iii_data], width, label='AlphaFold', color='#2E86AB')
    ax1.bar(x + width/2, [d[1] for d in type_iii_data], width, label='ESM-3', color='#5BC0EB')

    ax1.set_yscale('log')
    ax1.set_ylabel('Factor (x)')
    ax1.set_title('Type III: Capability Extension')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Type I (Scale) - GNoME
    ax2 = axes[1]
    categories2 = ['Materials\nPredicted', 'Materials\nSynthesized/yr', 'Backlog\n(years)']
    type_i_data = [2200000, 350, 6286]

    colors = ['#A23B72', '#F18F01', '#C73E1D']
    ax2.bar(categories2, type_i_data, color=colors)

    ax2.set_yscale('log')
    ax2.set_ylabel('Count / Time')
    ax2.set_title('Type I: Scale Increase')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotation
    ax2.annotate(
        'AI creates massive backlog\nPhysical synthesis unchanged',
        xy=(1, 350),
        xytext=(1.5, 10000),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black'),
        ha='center',
    )

    plt.suptitle('Shift Type Determines Impact Pattern', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig4_shift_type_comparison.png")
    plt.close()
    print("Saved: fig4_shift_type_comparison.png")


def fig5_historical_trajectory():
    """
    Figure 5: Historical Trajectory with Case Studies

    Plot model projection with case study data points overlaid.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 5: Historical Trajectory (text mode)")
        print("=" * 50)
        print("Model projects 38x by 2050")
        print("Case studies: 4-24x already in 2021-2024")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Model projections
    model = AIResearchAccelerationModel(scenario=Scenario.BASELINE)
    years = list(range(2025, 2051))
    forecasts = model.forecast(years)

    model_accel = [forecasts[y]['acceleration'] for y in years]
    ax.plot(years, model_accel, 'b-', linewidth=2, label='Model Baseline')

    # Add uncertainty band (rough estimate)
    upper = [a * 1.5 for a in model_accel]
    lower = [a / 1.5 for a in model_accel]
    ax.fill_between(years, lower, upper, alpha=0.2, color='blue')

    # Case studies
    case_data = {
        'AlphaFold 2': (2021, 24.3, '#2E86AB'),
        'GNoME': (2023, 365, '#A23B72'),  # Note: this is for hypothesis gen
        'ESM-3': (2024, 4.0, '#5BC0EB'),
    }

    for name, (year, accel, color) in case_data.items():
        ax.scatter(year, accel, s=150, c=color, marker='o', edgecolors='black',
                  linewidths=1, zorder=5, label=name)

    ax.set_yscale('log')
    ax.set_xlim(2020, 2055)
    ax.set_ylim(1, 1000)

    ax.set_xlabel('Year')
    ax.set_ylabel('Acceleration Factor (x)')
    ax.set_title('AI Acceleration Trajectory: Model vs. Case Studies')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(
        'Case studies ahead\nof model projection',
        xy=(2022, 30),
        xytext=(2028, 100),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    ax.annotate(
        'GNoME: Stage acceleration\nnot end-to-end',
        xy=(2023, 365),
        xytext=(2030, 500),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig5_historical_trajectory.png")
    plt.close()
    print("Saved: fig5_historical_trajectory.png")


def fig6_model_improvements():
    """
    Figure 6: Suggested Model Improvements

    Visual summary of how case studies suggest model refinements.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 6: Model Improvements (text mode)")
        print("=" * 50)
        print("1. Distinguish shift types (I, II, III)")
        print("2. Reduce M_max_physical from 2.5x to 1.5x")
        print("3. Add domain-specific parameters")
        print("4. Model backlog effects for Type I shifts")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Create text-based diagram
    improvements = [
        ("Current Model", "Single M_max for physical stages\nNo shift type distinction\n38x by 2050 projection"),
        ("Case Study Insights", "Physical stages: 1.0-1.5x (not 2.5x)\nType I creates backlog, not speed\nDomain-specific acceleration"),
        ("Proposed v0.4", "M_max_physical: 2.5x → 1.5x\nAdd shift_type parameter\nDomain-specific calibration"),
    ]

    y_positions = [0.75, 0.45, 0.15]
    colors = ['#FFCDD2', '#FFF9C4', '#C8E6C9']

    for (title, content), y, color in zip(improvements, y_positions, colors):
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='gray')
        ax.text(0.5, y, f"{title}\n\n{content}",
               transform=ax.transAxes,
               fontsize=11,
               ha='center', va='center',
               bbox=bbox)

    # Add arrows
    for i in range(len(y_positions) - 1):
        ax.annotate('', xy=(0.5, y_positions[i+1] + 0.12),
                   xytext=(0.5, y_positions[i] - 0.08),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title('Model Improvement Pathway Based on Case Study Validation',
                fontsize=14, pad=20)

    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig6_model_improvements.png")
    plt.close()
    print("Saved: fig6_model_improvements.png")


def generate_all_figures():
    """Generate all figures."""
    print("Generating v0.3 Case Study Validation Figures...")
    print("=" * 50)

    setup_style()

    fig1_validation_comparison()
    fig2_stage_acceleration()
    fig3_bottleneck_analysis()
    fig4_shift_type_comparison()
    fig5_historical_trajectory()
    fig6_model_improvements()

    print()
    print("All figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
