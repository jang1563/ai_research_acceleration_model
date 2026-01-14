#!/usr/bin/env python3
"""
Case Study Validation Visualizations
====================================

Publication-quality figures for v0.3 case study validation.
Updated for 6 case studies (v0.3.1):
- AlphaFold 2/3 (Structural Biology, Type III)
- GNoME (Materials Science, Type I)
- ESM-3 (Protein Design, Type III)
- Recursion (Drug Discovery, Type II)
- Isomorphic Labs (Drug Discovery, Type III)
- Cradle Bio (Protein Design, Type II)

Figures:
1. Validation comparison (predicted vs observed) - all 6 case studies
2. Acceleration by pipeline stage - all 6 case studies
3. Bottleneck analysis - all 6 case studies
4. Shift type comparison (Type I vs II vs III)
5. Historical trajectory with case studies
6. Drug discovery focus (Recursion vs Isomorphic)
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
from recursion_case_study import RecursionCaseStudy, recursion_pipeline_analysis
from isomorphic_case_study import IsomorphicCaseStudy, isomorphic_metrics_analysis
from cradle_case_study import CradleCaseStudy, cradle_metrics_analysis

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
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def fig1_validation_comparison():
    """
    Figure 1: Predicted vs Observed Acceleration

    Scatter plot comparing model predictions to case study observations.
    Updated for all 6 case studies.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 1: Validation Comparison (text mode)")
        print("=" * 50)
        print("AlphaFold:  Predicted 0.5x, Observed 24.3x (Type III)")
        print("GNoME:      Predicted 1.0x, Observed 365.0x (Type I)")
        print("ESM-3:      Predicted 1.4x, Observed 4.0x (Type III)")
        print("Recursion:  Predicted 1.4x, Observed 1.5x (Type II)")
        print("Isomorphic: Predicted 1.4x, Observed 1.6x (Type III)")
        print("Cradle:     Predicted 1.4x, Observed 2.1x (Type II)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Data for all 6 case studies
    case_studies = {
        'AlphaFold 2/3': {'predicted': 0.5, 'observed': 24.3, 'type': 'capability', 'year': 2021},
        'GNoME': {'predicted': 1.0, 'observed': 365.0, 'type': 'scale', 'year': 2023},
        'ESM-3': {'predicted': 1.4, 'observed': 4.0, 'type': 'capability', 'year': 2024},
        'Recursion': {'predicted': 1.4, 'observed': 1.5, 'type': 'efficiency', 'year': 2024},
        'Isomorphic': {'predicted': 1.4, 'observed': 1.6, 'type': 'capability', 'year': 2024},
        'Cradle Bio': {'predicted': 1.4, 'observed': 2.1, 'type': 'efficiency', 'year': 2024},
    }

    colors = {'capability': '#2E86AB', 'scale': '#A23B72', 'efficiency': '#28A745'}
    markers = {'capability': 'o', 'scale': 's', 'efficiency': '^'}

    # Plot each case study
    for name, data in case_studies.items():
        ax.scatter(
            data['predicted'], data['observed'],
            c=colors[data['type']],
            marker=markers[data['type']],
            s=200,
            label=f"{name} ({data['type']})",
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
    ax.set_ylim(0.5, 1000)

    ax.set_xlabel('Model Predicted Acceleration (x)')
    ax.set_ylabel('Observed Acceleration (x)')
    ax.set_title('Case Study Validation: Model Predictions vs. Observations (6 Studies)')

    # Add annotations
    offsets = {
        'AlphaFold 2/3': (10, 10),
        'GNoME': (10, 5),
        'ESM-3': (10, -15),
        'Recursion': (-60, 10),
        'Isomorphic': (10, -15),
        'Cradle Bio': (10, 10),
    }
    for name, data in case_studies.items():
        ax.annotate(
            name,
            (data['predicted'], data['observed']),
            xytext=offsets.get(name, (10, 10)),
            textcoords='offset points',
            fontsize=9,
            ha='left',
        )

    # Legend with shift types
    handles = [
        mpatches.Patch(color='#2E86AB', label='Type III (Capability)'),
        mpatches.Patch(color='#A23B72', label='Type I (Scale)'),
        mpatches.Patch(color='#28A745', label='Type II (Efficiency)'),
    ]
    ax.legend(handles=handles, loc='upper left')
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
    for all 6 case studies.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 2: Stage Acceleration (text mode)")
        print("=" * 50)
        print("AlphaFold S3: 36,500x | S4: 1.5x | S6: 1.5x")
        print("GNoME S2/S3: 100,000x | S4: 1.0x")
        print("ESM-3 S2/S3: 30,000x | S4: 1.0x | S5: 1.5x")
        print("Recursion S2: 12x | S6: 1.2x (clinical)")
        print("Isomorphic S1: 36,500x | S6: 1.2x (clinical)")
        print("Cradle S2: 24x | S4: 1.5x (wet lab)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stage_labels = ['Lit Review', 'Hypothesis', 'Analysis', 'Wet Lab', 'Interpret', 'Validation']

    # Data for all 6 case studies
    case_data = [
        ('AlphaFold 2/3', [15, 14, 36500, 1.5, 4.3, 1.5], '#2E86AB'),
        ('GNoME', [8.6, 30000, 70000, 1.0, 4.3, 2.0], '#A23B72'),
        ('ESM-3', [10, 6000, 30000, 1.0, 1.5, 1.5], '#5BC0EB'),
        ('Recursion', [3.0, 12.0, 3.0, 1.2, 1.2, 1.2], '#28A745'),
        ('Isomorphic', [36500, 1800, 12, 1.2, 3.0, 1.17], '#9B59B6'),
        ('Cradle Bio', [8.0, 24.0, 15.0, 1.5, 10.0, 1.2], '#E67E22'),
    ]

    for idx, (name, accel, color) in enumerate(case_data):
        ax = axes[idx]
        ax.bar(stages, accel, color=color, edgecolor='black')
        ax.set_yscale('log')
        ax.set_ylim(0.5, 100000)
        ax.set_title(name)
        if idx % 3 == 0:
            ax.set_ylabel('Acceleration (x)')
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='M_max cognitive')
        ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5, label='M_max physical')
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    # Add legend to last subplot
    axes[-1].legend(loc='upper right', fontsize=8)

    plt.suptitle('Stage-Level Acceleration by Case Study (6 Studies)', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig2_stage_acceleration.png")
    plt.close()
    print("Saved: fig2_stage_acceleration.png")


def fig3_bottleneck_analysis():
    """
    Figure 3: Bottleneck Analysis

    Visual comparison of where bottlenecks occur across all 6 case studies.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 3: Bottleneck Analysis (text mode)")
        print("=" * 50)
        print("All case studies bottleneck at S4 (Wet Lab) or S6 (Clinical/Validation)")
        print("Model correctly predicts physical stages as limiting factors")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    case_studies = ['AlphaFold', 'GNoME', 'ESM-3', 'Recursion', 'Isomorphic', 'Cradle']
    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

    # Create heatmap-like visualization
    # 0 = minimal, 1 = moderate, 2 = bottleneck
    bottleneck_matrix = np.array([
        [0, 0, 0, 2, 0, 1],  # AlphaFold: S4 bottleneck, S6 secondary
        [0, 0, 0, 2, 1, 0],  # GNoME: S4 primary, S5 secondary
        [0, 0, 0, 2, 1, 0],  # ESM-3: S4 primary, S5 secondary
        [0, 0, 0, 1, 1, 2],  # Recursion: S6 (clinical) primary
        [0, 0, 0, 1, 0, 2],  # Isomorphic: S6 (clinical) primary
        [0, 0, 0, 2, 0, 1],  # Cradle: S4 (wet lab) primary
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
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color='white' if bottleneck_matrix[i, j] == 2 else 'black')

    ax.set_title('Bottleneck Identification: Model Prediction Validated (6 Case Studies)')
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
    Figure 4: Type I vs Type II vs Type III Shifts

    Compare acceleration patterns across all three shift types.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 4: Shift Type Comparison (text mode)")
        print("=" * 50)
        print("Type I (Scale): Massive generation, synthesis bottleneck (GNoME)")
        print("Type II (Efficiency): Faster iterations (Recursion, Cradle)")
        print("Type III (Capability): New abilities (AlphaFold, ESM-3, Isomorphic)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Type III (Capability) - AlphaFold, ESM-3, Isomorphic
    ax1 = axes[0]
    categories = ['Stage\nAccel', 'End-to-End', 'Cost\nSavings']
    names = ['AlphaFold', 'ESM-3', 'Isomorphic']
    data = [
        [36500, 24, 10],     # AlphaFold
        [30000, 4, 5],       # ESM-3
        [36500, 1.6, 2],     # Isomorphic
    ]
    colors = ['#2E86AB', '#5BC0EB', '#9B59B6']

    x = np.arange(len(categories))
    width = 0.25
    for i, (name, d, c) in enumerate(zip(names, data, colors)):
        ax1.bar(x + (i - 1) * width, d, width, label=name, color=c)

    ax1.set_yscale('log')
    ax1.set_ylabel('Factor (x)')
    ax1.set_title('Type III: Capability Extension')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Type II (Efficiency) - Recursion, Cradle
    ax2 = axes[1]
    categories2 = ['Stage\nAccel', 'End-to-End', 'Iteration\nReduction']
    names2 = ['Recursion', 'Cradle']
    data2 = [
        [12, 2.3, 2],   # Recursion
        [24, 2.1, 5],   # Cradle
    ]
    colors2 = ['#28A745', '#E67E22']

    x2 = np.arange(len(categories2))
    width2 = 0.35
    for i, (name, d, c) in enumerate(zip(names2, data2, colors2)):
        ax2.bar(x2 + (i - 0.5) * width2, d, width2, label=name, color=c)

    ax2.set_yscale('log')
    ax2.set_ylabel('Factor (x)')
    ax2.set_title('Type II: Efficiency')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Type I (Scale) - GNoME
    ax3 = axes[2]
    categories3 = ['Materials\nPredicted', 'Synthesized\n/year', 'Backlog\n(years)']
    type_i_data = [2200000, 350, 6286]
    colors3 = ['#A23B72', '#F18F01', '#C73E1D']

    ax3.bar(categories3, type_i_data, color=colors3)
    ax3.set_yscale('log')
    ax3.set_ylabel('Count / Time')
    ax3.set_title('Type I: Scale (GNoME)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Annotation
    ax3.annotate(
        'AI creates backlog\nPhysical unchanged',
        xy=(1, 350),
        xytext=(1.5, 10000),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black'),
        ha='center',
    )

    plt.suptitle('Shift Type Determines Impact Pattern (6 Case Studies)', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig4_shift_type_comparison.png")
    plt.close()
    print("Saved: fig4_shift_type_comparison.png")


def fig5_historical_trajectory():
    """
    Figure 5: Historical Trajectory with Case Studies

    Plot model projection with all 6 case study data points overlaid.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 5: Historical Trajectory (text mode)")
        print("=" * 50)
        print("Model projects 38x by 2050")
        print("Case studies: 1.5-24x in 2021-2024")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Model projections
    model = AIResearchAccelerationModel(scenario=Scenario.BASELINE)
    years = list(range(2025, 2051))
    forecasts = model.forecast(years)

    model_accel = [forecasts[y]['acceleration'] for y in years]
    ax.plot(years, model_accel, 'b-', linewidth=2, label='Model Baseline')

    # Add uncertainty band
    upper = [a * 1.5 for a in model_accel]
    lower = [a / 1.5 for a in model_accel]
    ax.fill_between(years, lower, upper, alpha=0.2, color='blue')

    # All 6 case studies
    case_data = {
        'AlphaFold': (2021, 24.3, '#2E86AB', 'o'),
        'GNoME': (2023, 365, '#A23B72', 's'),
        'ESM-3': (2024, 4.0, '#5BC0EB', 'o'),
        'Recursion': (2024, 1.5, '#28A745', '^'),
        'Isomorphic': (2024, 1.6, '#9B59B6', 'o'),
        'Cradle': (2024, 2.1, '#E67E22', '^'),
    }

    for name, (year, accel, color, marker) in case_data.items():
        ax.scatter(year, accel, s=150, c=color, marker=marker, edgecolors='black',
                  linewidths=1, zorder=5, label=name)

    ax.set_yscale('log')
    ax.set_xlim(2020, 2055)
    ax.set_ylim(0.5, 1000)

    ax.set_xlabel('Year')
    ax.set_ylabel('Acceleration Factor (x)')
    ax.set_title('AI Acceleration Trajectory: Model vs. 6 Case Studies')

    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate(
        'GNoME: Stage acceleration\nnot end-to-end',
        xy=(2023, 365),
        xytext=(2028, 600),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    ax.annotate(
        'Drug discovery cases:\n1.5-1.6x (clinical bottleneck)',
        xy=(2024, 1.55),
        xytext=(2030, 0.8),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig5_historical_trajectory.png")
    plt.close()
    print("Saved: fig5_historical_trajectory.png")


def fig6_drug_discovery_comparison():
    """
    Figure 6: Drug Discovery Case Study Comparison

    Compare Recursion (Type II) vs Isomorphic (Type III) approaches.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 6: Drug Discovery Comparison (text mode)")
        print("=" * 50)
        print("Recursion: Target to IND 18mo vs 42mo (2.3x)")
        print("Isomorphic: Full pipeline 144mo vs 89mo (1.6x)")
        print("Both limited by clinical trials (5-7 years)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Recursion timeline comparison
    ax1 = axes[0]
    stages = ['Target ID', 'Hit ID', 'Hit-to-Lead', 'Preclinical', 'IND Studies', 'Clinical']
    traditional = [6, 12, 18, 12, 12, 72]  # months
    recursion = [2, 1, 6, 10, 10, 60]  # months

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax1.barh(x - width/2, traditional, width, label='Traditional', color='#CCCCCC')
    bars2 = ax1.barh(x + width/2, recursion, width, label='Recursion AI', color='#28A745')

    ax1.set_yticks(x)
    ax1.set_yticklabels(stages)
    ax1.set_xlabel('Duration (months)')
    ax1.set_title('Recursion: Type II Efficiency Shift')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Add totals
    ax1.axvline(x=sum(traditional), color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=sum(recursion), color='#28A745', linestyle='--', alpha=0.5)
    ax1.text(sum(traditional) + 2, 5.5, f'{sum(traditional)}mo', fontsize=9, color='gray')
    ax1.text(sum(recursion) + 2, 5.5, f'{sum(recursion)}mo', fontsize=9, color='#28A745')

    # Isomorphic/AlphaFold 3 timeline
    ax2 = axes[1]
    stages2 = ['Structure', 'Binding Site', 'Lead Design', 'Synthesis', 'Optimization', 'Clinical']
    traditional2 = [12, 6, 12, 12, 18, 84]  # months
    isomorphic = [0.01, 0.1, 1, 10, 6, 72]  # months

    bars3 = ax2.barh(x - width/2, traditional2, width, label='Traditional', color='#CCCCCC')
    bars4 = ax2.barh(x + width/2, isomorphic, width, label='Isomorphic/AF3', color='#9B59B6')

    ax2.set_yticks(x)
    ax2.set_yticklabels(stages2)
    ax2.set_xlabel('Duration (months)')
    ax2.set_title('Isomorphic: Type III Capability Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # Add totals
    ax2.axvline(x=sum(traditional2), color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=sum(isomorphic), color='#9B59B6', linestyle='--', alpha=0.5)
    ax2.text(sum(traditional2) + 2, 5.5, f'{sum(traditional2)}mo', fontsize=9, color='gray')
    ax2.text(sum(isomorphic) + 2, 5.5, f'{sum(isomorphic):.0f}mo', fontsize=9, color='#9B59B6')

    plt.suptitle('Drug Discovery: Both Approaches Limited by Clinical Trials', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig6_drug_discovery_comparison.png")
    plt.close()
    print("Saved: fig6_drug_discovery_comparison.png")


def fig7_model_validation_summary():
    """
    Figure 7: Model Validation Summary

    Visual summary of validation results across all 6 case studies.
    """
    if not HAS_MATPLOTLIB:
        print("Figure 7: Model Validation Summary (text mode)")
        print("=" * 50)
        print("Validated: Recursion (0.97), Isomorphic (0.94), Cradle (0.82)")
        print("Partial: ESM-3 (0.54)")
        print("Rejected: AlphaFold (0.00), GNoME (0.00)")
        print("Physical bottleneck hypothesis: CONFIRMED")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Validation scores
    ax1 = axes[0]
    case_studies = ['Recursion', 'Isomorphic', 'Cradle', 'ESM-3', 'AlphaFold', 'GNoME']
    scores = [0.97, 0.94, 0.82, 0.54, 0.00, 0.00]
    colors = ['#28A745', '#28A745', '#28A745', '#FFC107', '#DC3545', '#DC3545']

    bars = ax1.barh(case_studies, scores, color=colors, edgecolor='black')
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Validation Score')
    ax1.set_title('Model Validation Scores by Case Study')
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Validated threshold')
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Partial threshold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add score labels
    for bar, score in zip(bars, scores):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10)

    # Right: Key findings
    ax2 = axes[1]
    ax2.axis('off')

    findings = """
    KEY VALIDATION FINDINGS (6 Case Studies)
    ========================================

    ✓ VALIDATED: Physical bottleneck hypothesis
      - All 6 cases show S4/S6 as limiting factor
      - Drug discovery: Clinical trials = 5-7 years
      - Protein design: Wet lab = 1-2 months/cycle

    ✓ VALIDATED: Cognitive stages achieve 10-36,500x

    ✓ REFINED: Shift type determines outcome
      - Type III (capability): 1.6-24x end-to-end
      - Type II (efficiency): 1.5-2.3x end-to-end
      - Type I (scale): Creates backlog, not speed

    ⚠ MODEL LIMITATION: Under-predicts for
      domain-specific breakthroughs (AlphaFold)
      and Type I scale shifts (GNoME)
    """

    ax2.text(0.05, 0.95, findings, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))

    plt.suptitle('Model Validation Summary: 6 Case Studies', fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "fig7_model_validation_summary.png")
    plt.close()
    print("Saved: fig7_model_validation_summary.png")


def generate_all_figures():
    """Generate all figures."""
    print("Generating v0.3 Case Study Validation Figures (6 Case Studies)...")
    print("=" * 60)

    setup_style()

    fig1_validation_comparison()
    fig2_stage_acceleration()
    fig3_bottleneck_analysis()
    fig4_shift_type_comparison()
    fig5_historical_trajectory()
    fig6_drug_discovery_comparison()
    fig7_model_validation_summary()

    print()
    print("All 7 figures generated successfully.")
    print("Figures include all 6 case studies:")
    print("  - AlphaFold 2/3 (Type III)")
    print("  - GNoME (Type I)")
    print("  - ESM-3 (Type III)")
    print("  - Recursion (Type II)")
    print("  - Isomorphic Labs (Type III)")
    print("  - Cradle Bio (Type II)")


if __name__ == "__main__":
    generate_all_figures()
