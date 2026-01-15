#!/usr/bin/env python3
"""
Historical Calibration Visualizations
=====================================

Generates publication-quality figures for v0.2 historical calibration analysis.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from historical_calibration import (
    HistoricalCalibrator,
    HISTORICAL_SHIFTS,
    ShiftCategory,
)

# Also add v0.1 for model comparison
sys.path.insert(0, str(Path(__file__).parent.parent / "v0.1"))
from src.model import AIResearchAccelerationModel, Scenario

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
COLORS = {
    'capability': '#0072B2',      # Blue
    'methodological': '#D55E00',   # Orange/Red
    'ai_projected': '#009E73',     # Green
    'uncertainty': '#999999',      # Gray
    'highlight': '#CC79A7',        # Pink
}


def fig1_historical_acceleration_timeline():
    """
    Figure 1: Timeline of historical paradigm shifts and their acceleration.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    shifts = list(HISTORICAL_SHIFTS.values())

    # Sort by start year
    shifts.sort(key=lambda s: s.start_year)

    y_positions = list(range(len(shifts)))

    for i, shift in enumerate(shifts):
        # Bar showing duration
        bar_start = shift.start_year
        bar_width = shift.full_impact_years

        color = COLORS['capability'] if shift.category == ShiftCategory.CAPABILITY_EXTENSION else COLORS['methodological']

        ax.barh(i, bar_width, left=bar_start, height=0.6, color=color, alpha=0.7,
                edgecolor='black', linewidth=0.5)

        # Acceleration annotation
        ax.annotate(
            f'{shift.time_acceleration:,.0f}x',
            xy=(bar_start + bar_width, i),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            va='center',
        )

        # Mark transformation point
        transform_x = bar_start + shift.transformation_years
        ax.plot(transform_x, i, 'o', color='white', markersize=8, markeredgecolor='black', zorder=5)

    # AI projection
    ax.barh(len(shifts), 25, left=2025, height=0.6, color=COLORS['ai_projected'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.annotate('~38x\n(v0.1)', xy=(2050, len(shifts)), xytext=(10, 0),
                textcoords='offset points', fontsize=9, fontweight='bold', va='center')
    shifts.append(type('obj', (object,), {'name': 'AI (Projected)'})())

    ax.set_yticks(range(len(shifts)))
    ax.set_yticklabels([s.name for s in shifts])
    ax.set_xlabel('Year')
    ax.set_title('Historical Paradigm Shifts: Timeline and Acceleration', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.7, label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.7, label='Methodological Shift'),
        mpatches.Patch(facecolor=COLORS['ai_projected'], alpha=0.7, label='AI (Projected)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='10x Impact Point'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlim(1550, 2080)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def fig2_acceleration_comparison():
    """
    Figure 2: Comparison of acceleration factors across paradigm shifts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    shifts = list(HISTORICAL_SHIFTS.values())
    names = [s.name for s in shifts]

    # Time Acceleration (log scale)
    ax = axes[0]
    time_accels = [s.time_acceleration for s in shifts]
    colors = [COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
              else COLORS['methodological'] for s in shifts]

    bars = ax.bar(range(len(shifts)), time_accels, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels([s.name.replace(' ', '\n').replace('/', '\n') for s in shifts], fontsize=8)
    ax.set_ylabel('Time Acceleration Factor (log scale)')
    ax.set_yscale('log')
    ax.set_title('Time Acceleration by Paradigm Shift', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, time_accels):
        ax.annotate(f'{val:,.0f}x',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold')

    # Publication Increase
    ax = axes[1]
    pub_increases = [s.publication_increase for s in shifts]

    bars = ax.bar(range(len(shifts)), pub_increases, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels([s.name.replace(' ', '\n').replace('/', '\n') for s in shifts], fontsize=8)
    ax.set_ylabel('Publication Increase Factor (log scale)')
    ax.set_yscale('log')
    ax.set_title('Publication Growth by Paradigm Shift', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, pub_increases):
        ax.annotate(f'{val:.0f}x',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.7, label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.7, label='Methodological Shift'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    return fig


def fig3_calibration_fit():
    """
    Figure 3: Calibration fit - observed vs predicted values.
    """
    calibrator = HistoricalCalibrator()
    result = calibrator.calibrate_mle()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    shifts = list(HISTORICAL_SHIFTS.values())

    # Time Acceleration
    ax = axes[0]
    observed = [s.time_acceleration for s in shifts]
    predicted = [10 ** calibrator.predict_metrics(s, result.parameters)['time_acceleration'] for s in shifts]

    ax.scatter(observed, predicted, c=[COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
                                       else COLORS['methodological'] for s in shifts],
               s=100, alpha=0.7, edgecolors='black')

    # Perfect fit line
    lims = [0.5, max(max(observed), max(predicted)) * 2]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Time Acceleration', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Label points
    for s, obs, pred in zip(shifts, observed, predicted):
        ax.annotate(s.name[:10], xy=(obs, pred), xytext=(5, 5), textcoords='offset points', fontsize=7)

    # Publication Increase
    ax = axes[1]
    observed = [s.publication_increase for s in shifts]
    predicted = [10 ** calibrator.predict_metrics(s, result.parameters)['publication_increase'] for s in shifts]

    ax.scatter(observed, predicted, c=[COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
                                       else COLORS['methodological'] for s in shifts],
               s=100, alpha=0.7, edgecolors='black')

    lims = [1, max(max(observed), max(predicted)) * 2]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Publication Increase', fontweight='bold')
    ax.grid(True, alpha=0.3)

    for s, obs, pred in zip(shifts, observed, predicted):
        ax.annotate(s.name[:10], xy=(obs, pred), xytext=(5, 5), textcoords='offset points', fontsize=7)

    # Transformation Years
    ax = axes[2]
    observed = [s.transformation_years for s in shifts]
    predicted = [calibrator.predict_metrics(s, result.parameters)['transformation_years'] for s in shifts]

    ax.scatter(observed, predicted, c=[COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
                                       else COLORS['methodological'] for s in shifts],
               s=100, alpha=0.7, edgecolors='black')

    lims = [0, max(max(observed), max(predicted)) * 1.2]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Fit')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed (years)')
    ax.set_ylabel('Predicted (years)')
    ax.set_title('Transformation Time', fontweight='bold')
    ax.grid(True, alpha=0.3)

    for s, obs, pred in zip(shifts, observed, predicted):
        ax.annotate(s.name[:10], xy=(obs, pred), xytext=(5, 5), textcoords='offset points', fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.7, label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.7, label='Methodological Shift'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Perfect Fit'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    return fig


def fig4_ai_vs_historical():
    """
    Figure 4: AI projections compared to historical paradigm shifts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Historical shifts
    shifts = list(HISTORICAL_SHIFTS.values())

    # Model projections
    model = AIResearchAccelerationModel(scenario=Scenario.BASELINE)
    forecasts = model.forecast([2025, 2030, 2035, 2040, 2050])

    # Plot historical points
    for i, shift in enumerate(shifts):
        color = COLORS['capability'] if shift.category == ShiftCategory.CAPABILITY_EXTENSION else COLORS['methodological']
        ax.scatter(shift.transformation_years, shift.time_acceleration,
                   s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(shift.name, xy=(shift.transformation_years, shift.time_acceleration),
                    xytext=(8, 0), textcoords='offset points', fontsize=9, va='center')

    # Plot AI trajectory
    years = list(forecasts.keys())
    accels = [forecasts[y]['acceleration'] for y in years]
    transform_years = [y - 2025 for y in years]  # Years from start

    ax.plot(transform_years, accels, 'o-', color=COLORS['ai_projected'],
            markersize=10, linewidth=2, label='AI Projection (v0.1)', zorder=10)

    # Add uncertainty band for AI
    # Simple ±30% uncertainty
    lower = [a * 0.7 for a in accels]
    upper = [a * 1.3 for a in accels]
    ax.fill_between(transform_years, lower, upper, color=COLORS['ai_projected'], alpha=0.2)

    ax.set_xlabel('Years to Transformation')
    ax.set_ylabel('Acceleration Factor (log scale)')
    ax.set_yscale('log')
    ax.set_title('AI Projections vs Historical Paradigm Shifts', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.7, label='Capability Extension (Historical)'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.7, label='Methodological Shift (Historical)'),
        Line2D([0], [0], marker='o', color=COLORS['ai_projected'], markersize=10, label='AI Projection'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 55)

    plt.tight_layout()
    return fig


def fig5_parameter_sensitivity():
    """
    Figure 5: Parameter sensitivity for calibrated model.
    """
    calibrator = HistoricalCalibrator()

    # Get baseline result
    baseline_result = calibrator.calibrate_mle()
    baseline_ll = baseline_result.log_likelihood

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    params = list(calibrator.CALIBRATION_PARAMS.keys())

    for ax, param in zip(axes, params):
        # Get bounds
        prior_mean, prior_std, bounds = calibrator.CALIBRATION_PARAMS[param]

        # Create range of values
        values = np.linspace(bounds[0], bounds[1], 50)
        likelihoods = []

        for val in values:
            test_params = baseline_result.parameters.copy()
            test_params[param] = val
            ll = calibrator.log_likelihood(test_params)
            likelihoods.append(ll)

        ax.plot(values, likelihoods, 'b-', linewidth=2)
        ax.axvline(baseline_result.parameters[param], color='r', linestyle='--',
                   label=f'MLE = {baseline_result.parameters[param]:.2f}')
        ax.axhline(baseline_ll - 2, color='gray', linestyle=':', alpha=0.7,
                   label='95% CI threshold')

        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Log-Likelihood')
        ax.set_title(f'Sensitivity: {param}', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    axes[-1].axis('off')

    plt.suptitle('Parameter Sensitivity Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def generate_all_figures():
    """Generate all figures and save to figures/ directory."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    figures = [
        ('fig1_historical_timeline.png', fig1_historical_acceleration_timeline),
        ('fig2_acceleration_comparison.png', fig2_acceleration_comparison),
        ('fig3_calibration_fit.png', fig3_calibration_fit),
        ('fig4_ai_vs_historical.png', fig4_ai_vs_historical),
        ('fig5_parameter_sensitivity.png', fig5_parameter_sensitivity),
    ]

    for filename, fig_func in figures:
        print(f"Generating {filename}...")
        try:
            fig = fig_func()
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"  ✓ Saved to {output_dir / filename}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_figures()
