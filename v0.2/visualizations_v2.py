#!/usr/bin/env python3
"""
Historical Calibration Visualizations v2
========================================

Improved publication-quality figures addressing expert reviewer feedback.

Key changes from v1:
- V1-P1: Split timeline into eras for better visibility
- V1-P8: Smart label placement to avoid overlaps
- V1-P11: Fixed label positioning in AI vs Historical
- V1-P14: Fixed time_to_impact_scale numerical issues
- V2-R2: Era-based visualization for timeline
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
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

# Color palette (colorblind-friendly)
COLORS = {
    'capability': '#0072B2',       # Blue
    'methodological': '#D55E00',   # Orange/Red
    'computational': '#009E73',    # Teal/Green
    'ai_projected': '#CC79A7',     # Pink/Magenta
    'uncertainty': '#999999',      # Gray
    'highlight': '#F0E442',        # Yellow
}


def fig1_historical_timeline_v2():
    """
    Figure 1 v2: Timeline with era-based panels for better visibility.

    Addresses: V1-P1 (misleading scale), V2-R2 (era breaks)
    """
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.15)

    # Era 1: Pre-modern (1600-1800)
    ax1 = fig.add_subplot(gs[0])

    # Era 2: Modern (1970-2050)
    ax2 = fig.add_subplot(gs[1])

    shifts = list(HISTORICAL_SHIFTS.values())

    # Separate by era
    premodern = [s for s in shifts if s.start_year < 1900]
    modern = [s for s in shifts if s.start_year >= 1900]

    # Plot pre-modern era
    for i, shift in enumerate(premodern):
        bar_start = shift.start_year
        bar_width = shift.full_impact_years
        color = COLORS['capability'] if shift.category == ShiftCategory.CAPABILITY_EXTENSION else COLORS['methodological']

        ax1.barh(i, bar_width, left=bar_start, height=0.6, color=color, alpha=0.8,
                edgecolor='black', linewidth=1)

        # Acceleration annotation
        ax1.annotate(
            f'{shift.time_acceleration:,.0f}×',
            xy=(bar_start + bar_width + 5, i),
            fontsize=11, fontweight='bold', va='center', ha='left',
        )

        # 10x Impact point
        transform_x = bar_start + shift.transformation_years
        ax1.plot(transform_x, i, 'D', color='white', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax1.set_yticks(range(len(premodern)))
    ax1.set_yticklabels([s.name for s in premodern], fontsize=11)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_title('Pre-Modern Era\n(1600-1800)', fontweight='bold', fontsize=12)
    ax1.set_xlim(1550, 1850)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Plot modern era
    y_pos = 0
    for shift in modern:
        bar_start = shift.start_year
        bar_width = shift.full_impact_years
        color = COLORS['capability'] if shift.category == ShiftCategory.CAPABILITY_EXTENSION else COLORS['methodological']

        ax2.barh(y_pos, bar_width, left=bar_start, height=0.6, color=color, alpha=0.8,
                edgecolor='black', linewidth=1)

        # Acceleration annotation - positioned to avoid overlap
        ax2.annotate(
            f'{shift.time_acceleration:,.0f}×',
            xy=(bar_start + bar_width + 1, y_pos),
            fontsize=11, fontweight='bold', va='center', ha='left',
        )

        # 10x Impact point
        transform_x = bar_start + shift.transformation_years
        ax2.plot(transform_x, y_pos, 'D', color='white', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

        y_pos += 1

    # Add AI projection
    ax2.barh(y_pos, 25, left=2025, height=0.6, color=COLORS['ai_projected'], alpha=0.8,
            edgecolor='black', linewidth=1)
    ax2.annotate('~38×\n(projected)', xy=(2051, y_pos), fontsize=10, fontweight='bold',
                va='center', ha='left')
    modern_names = [s.name for s in modern] + ['AI (2025-2050)']

    ax2.set_yticks(range(len(modern_names)))
    ax2.set_yticklabels(modern_names, fontsize=11)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_title('Modern Era\n(1970-2050)', fontweight='bold', fontsize=12)
    ax2.set_xlim(1980, 2065)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.8, edgecolor='black',
                      label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.8, edgecolor='black',
                      label='Methodological Shift'),
        mpatches.Patch(facecolor=COLORS['ai_projected'], alpha=0.8, edgecolor='black',
                      label='AI (Projected)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=10, markeredgewidth=1.5,
               label='10× Impact Achieved'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True)

    fig.suptitle('Historical Technology-Enabled Scientific Shifts: Timeline and Acceleration',
                 fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


def fig2_acceleration_comparison_v2():
    """
    Figure 2 v2: Acceleration comparison with fixed labels.

    Addresses: V1-P5 (label overlap), V1-P6 (spurious label), V1-P7 (edge labels)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    shifts = list(HISTORICAL_SHIFTS.values())
    n = len(shifts)

    # Prepare data
    names = []
    for s in shifts:
        # Wrap long names
        name = s.name.replace(' / ', '\n').replace(' ', '\n', 1)
        if len(name) > 15:
            name = s.name[:12] + '...'
        names.append(name)

    time_accels = [s.time_acceleration for s in shifts]
    pub_increases = [s.publication_increase for s in shifts]
    colors = [COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
              else COLORS['methodological'] for s in shifts]

    # Time Acceleration
    ax = axes[0]
    x_pos = np.arange(n)
    bars = ax.bar(x_pos, time_accels, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Micro-\nscope', 'Tele-\nscope', 'HGP/\nSeq', 'DNA\nSeq', 'CRISPR'],
                       fontsize=10)
    ax.set_ylabel('Time Acceleration Factor', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(5, 10000)
    ax.set_title('Time Acceleration by Technology', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels above bars (not overlapping)
    for bar, val in zip(bars, time_accels):
        height = bar.get_height()
        ax.annotate(f'{val:,.0f}×',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Publication Increase
    ax = axes[1]
    bars = ax.bar(x_pos, pub_increases, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Micro-\nscope', 'Tele-\nscope', 'HGP/\nSeq', 'DNA\nSeq', 'CRISPR'],
                       fontsize=10)
    ax.set_ylabel('Publication Increase Factor', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(20, 300)
    ax.set_title('Publication Growth by Technology', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val in zip(bars, pub_increases):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}×',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Shared legend at top
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.8, edgecolor='black',
                      label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.8, edgecolor='black',
                      label='Methodological Shift'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True)

    plt.tight_layout()
    return fig


def fig3_calibration_fit_v2():
    """
    Figure 3 v2: Calibration fit with smart label placement.

    Addresses: V1-P8 (label overlap), V3-P4 (prediction intervals)
    """
    calibrator = HistoricalCalibrator()
    result = calibrator.calibrate_mle()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    shifts = list(HISTORICAL_SHIFTS.values())

    # Helper function for smart label placement
    def place_labels(ax, points, labels, log_scale=False):
        """Place labels avoiding overlaps using simple offset strategy."""
        placed = []
        for (x, y), label in zip(points, labels):
            # Determine offset direction based on position
            if log_scale:
                x_off, y_off = 8, 0
            else:
                x_off, y_off = 5, 5

            # Check for nearby points and adjust
            for px, py in placed:
                if log_scale:
                    dist = abs(np.log10(y) - np.log10(py)) if y > 0 and py > 0 else 1
                else:
                    dist = abs(y - py)
                if dist < 0.3:
                    y_off = -15 if y > py else 15

            ax.annotate(label, xy=(x, y), xytext=(x_off, y_off),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='none', alpha=0.7))
            placed.append((x, y))

    # Time Acceleration
    ax = axes[0]
    observed = [s.time_acceleration for s in shifts]
    predicted = [10 ** calibrator.predict_metrics(s, result.parameters)['time_acceleration']
                 for s in shifts]
    colors = [COLORS['capability'] if s.category == ShiftCategory.CAPABILITY_EXTENSION
              else COLORS['methodological'] for s in shifts]
    short_names = ['Micro', 'Tele', 'HGP', 'Seq', 'CRISPR']

    ax.scatter(observed, predicted, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

    # Perfect fit line
    lims = [5, max(max(observed), max(predicted)) * 2]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5, label='Perfect Fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title('Time Acceleration', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    place_labels(ax, list(zip(observed, predicted)), short_names, log_scale=True)

    # Publication Increase
    ax = axes[1]
    observed = [s.publication_increase for s in shifts]
    predicted = [10 ** calibrator.predict_metrics(s, result.parameters)['publication_increase']
                 for s in shifts]

    ax.scatter(observed, predicted, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

    lims = [10, max(max(observed), max(predicted)) * 2]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title('Publication Increase', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    place_labels(ax, list(zip(observed, predicted)), short_names, log_scale=True)

    # Transformation Years
    ax = axes[2]
    observed = [s.transformation_years for s in shifts]
    predicted = [calibrator.predict_metrics(s, result.parameters)['transformation_years']
                 for s in shifts]

    ax.scatter(observed, predicted, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

    lims = [0, max(max(observed), max(predicted)) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed (years)', fontsize=11)
    ax.set_ylabel('Predicted (years)', fontsize=11)
    ax.set_title('Transformation Time', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    place_labels(ax, list(zip(observed, predicted)), short_names, log_scale=False)

    # Add note about systematic bias
    ax.annotate('Note: Model shows\nsystematic bias',
               xy=(0.95, 0.05), xycoords='axes fraction',
               fontsize=9, ha='right', va='bottom', style='italic',
               color='gray')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.8, edgecolor='black',
                      label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.8, edgecolor='black',
                      label='Methodological Shift'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.6, linewidth=1.5,
               label='Perfect Fit'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.05), frameon=True, fancybox=True)

    plt.tight_layout()
    return fig


def fig4_ai_vs_historical_v2():
    """
    Figure 4 v2: AI vs Historical with fixed label placement.

    Addresses: V1-P11 (CRISPR overlap), V1-P12 (inconsistent uncertainty)
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    shifts = list(HISTORICAL_SHIFTS.values())

    # Model projections
    model = AIResearchAccelerationModel(scenario=Scenario.BASELINE)
    forecasts = model.forecast([2025, 2030, 2035, 2040, 2050])

    # Plot historical points with uncertainty bars
    for shift in shifts:
        color = COLORS['capability'] if shift.category == ShiftCategory.CAPABILITY_EXTENSION else COLORS['methodological']

        # Add uncertainty (±30% shown as error bars)
        yerr_low = shift.time_acceleration * 0.3
        yerr_high = shift.time_acceleration * 0.3

        ax.errorbar(shift.transformation_years, shift.time_acceleration,
                   yerr=[[yerr_low], [yerr_high]], fmt='o',
                   color=color, markersize=14, markeredgecolor='black', markeredgewidth=1.5,
                   ecolor=color, elinewidth=2, capsize=5, capthick=2, alpha=0.8, zorder=5)

    # Smart label placement for historical points
    label_offsets = {
        'Microscope': (10, -5),
        'Telescope': (10, 5),
        'Human Genome Project / Sequencing': (-80, 15),
        'DNA Sequencing Revolution': (-100, -20),
        'CRISPR': (-50, -15),  # Moved away from AI line
    }

    for shift in shifts:
        offset = label_offsets.get(shift.name, (10, 0))
        ax.annotate(shift.name.replace(' / ', '/\n'),
                   xy=(shift.transformation_years, shift.time_acceleration),
                   xytext=offset, textcoords='offset points',
                   fontsize=10, va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.9))

    # Plot AI trajectory
    years = list(forecasts.keys())
    accels = [forecasts[y]['acceleration'] for y in years]
    transform_years = [y - 2025 for y in years]

    ax.plot(transform_years, accels, 'o-', color=COLORS['ai_projected'],
            markersize=12, markeredgecolor='black', markeredgewidth=1.5,
            linewidth=3, label='AI Projection (v0.1)', zorder=10)

    # Uncertainty band for AI (±30%)
    lower = [a * 0.7 for a in accels]
    upper = [a * 1.3 for a in accels]
    ax.fill_between(transform_years, lower, upper, color=COLORS['ai_projected'],
                   alpha=0.2, zorder=1)

    # Add year labels to AI trajectory
    for t, y, accel in zip(transform_years, years, accels):
        ax.annotate(f'{y}', xy=(t, accel), xytext=(0, -20),
                   textcoords='offset points', ha='center', fontsize=9,
                   color=COLORS['ai_projected'])

    ax.set_xlabel('Years to Transformation', fontsize=12)
    ax.set_ylabel('Acceleration Factor (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('AI Projections vs Historical Technology Shifts', fontweight='bold', fontsize=14)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['capability'], alpha=0.8, edgecolor='black',
                      label='Capability Extension'),
        mpatches.Patch(facecolor=COLORS['methodological'], alpha=0.8, edgecolor='black',
                      label='Methodological Shift'),
        Line2D([0], [0], marker='o', color=COLORS['ai_projected'], markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, linewidth=3,
               label='AI Projection'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-3, 55)
    ax.set_ylim(1, 10000)

    plt.tight_layout()
    return fig


def fig5_parameter_sensitivity_v2():
    """
    Figure 5 v2: Parameter sensitivity with normalized scales.

    Addresses: V1-P14 (numerical issues), V1-P15 (inconsistent scales)
    """
    calibrator = HistoricalCalibrator()
    baseline_result = calibrator.calibrate_mle()
    baseline_ll = baseline_result.log_likelihood

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    params = list(calibrator.CALIBRATION_PARAMS.keys())

    for ax, param in zip(axes[:5], params):
        prior_mean, prior_std, bounds = calibrator.CALIBRATION_PARAMS[param]

        # Create range of values within reasonable bounds
        # Limit range to avoid extreme likelihood values
        center = baseline_result.parameters[param]
        range_width = min(bounds[1] - bounds[0], 4 * prior_std)
        low = max(bounds[0], center - range_width/2)
        high = min(bounds[1], center + range_width/2)

        values = np.linspace(low, high, 100)

        # Calculate relative log-likelihood (normalized to baseline)
        rel_likelihoods = []
        for val in values:
            test_params = baseline_result.parameters.copy()
            test_params[param] = val
            ll = calibrator.log_likelihood(test_params)
            rel_ll = ll - baseline_ll  # Relative to MLE
            # Clip extreme values for visualization
            rel_ll = max(-20, rel_ll)
            rel_likelihoods.append(rel_ll)

        ax.plot(values, rel_likelihoods, 'b-', linewidth=2.5)
        ax.axvline(baseline_result.parameters[param], color='r', linestyle='--',
                   linewidth=2, label=f'MLE = {baseline_result.parameters[param]:.2f}')
        ax.axhline(-2, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
                   label='95% CI threshold')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        # Clean parameter name for title
        clean_name = param.replace('_', ' ').title()
        ax.set_xlabel(clean_name, fontsize=10)
        ax.set_ylabel('Relative Log-Likelihood', fontsize=10)
        ax.set_title(f'{clean_name}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-20, 2)  # Standardized y-axis

    # Hide unused subplot
    axes[5].axis('off')

    # Add note about interpretation
    axes[5].text(0.5, 0.5,
                'Interpretation:\n\n'
                '• Higher = better fit\n'
                '• 0 = MLE (best fit)\n'
                '• -2 = 95% CI boundary\n\n'
                'Flat curves indicate\nlow sensitivity to parameter',
                ha='center', va='center', fontsize=11,
                transform=axes[5].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Parameter Sensitivity Analysis (Relative Log-Likelihood)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def fig6_cumulative_acceleration():
    """
    Figure 6 (NEW): Cumulative computational acceleration in biology.

    Addresses: H5-P5 (ML history), H5 insight on cumulative acceleration
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Cumulative computational milestones
    milestones = [
        ('Manual\nMethods', 1950, 1, 'baseline'),
        ('Sequence\nAlignment', 1970, 10, 'computational'),
        ('Databases\n(GenBank)', 1982, 100, 'computational'),
        ('BLAST', 1990, 1000, 'computational'),
        ('Genome\nAssembly', 2001, 10000, 'computational'),
        ('Rosetta', 2005, 100000, 'computational'),
        ('ML\nClassifiers', 2010, 200000, 'computational'),
        ('AlphaFold', 2020, 1000000, 'ai'),
        ('AI (2025)', 2025, 2000000, 'ai'),
    ]

    years = [m[1] for m in milestones]
    accels = [m[2] for m in milestones]
    colors_list = []
    for m in milestones:
        if m[3] == 'baseline':
            colors_list.append('gray')
        elif m[3] == 'computational':
            colors_list.append(COLORS['computational'])
        else:
            colors_list.append(COLORS['ai_projected'])

    # Plot cumulative trajectory
    ax.plot(years, accels, 'o-', color='gray', alpha=0.5, linewidth=2, markersize=0, zorder=1)
    ax.scatter(years, accels, c=colors_list, s=150, edgecolors='black', linewidth=1.5, zorder=5)

    # Add labels
    for name, year, accel, _ in milestones:
        offset = (0, 15) if accel < 10000 else (0, -25)
        ax.annotate(name, xy=(year, accel), xytext=offset,
                   textcoords='offset points', ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Add marginal acceleration annotations
    ax.annotate('~10×', xy=(1970, 10), xytext=(20, 0), textcoords='offset points',
               fontsize=9, color='gray', style='italic')
    ax.annotate('~10×', xy=(1982, 100), xytext=(20, 0), textcoords='offset points',
               fontsize=9, color='gray', style='italic')
    ax.annotate('~10×', xy=(1990, 1000), xytext=(20, 0), textcoords='offset points',
               fontsize=9, color='gray', style='italic')
    ax.annotate('~10×', xy=(2020, 1000000), xytext=(20, 0), textcoords='offset points',
               fontsize=9, color='gray', style='italic')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Acceleration (vs. manual baseline)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Cumulative Computational Acceleration in Biology\n'
                 '(Each step ~10× over previous)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='black', label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['computational'],
               markersize=12, markeredgecolor='black', label='Computational Tools'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ai_projected'],
               markersize=12, markeredgecolor='black', label='AI/Deep Learning'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Key insight box
    insight_text = ('Key Insight: AI acceleration (~10×) is consistent with\n'
                   'historical computational progression, not a discontinuity.')
    ax.annotate(insight_text, xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', va='bottom', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlim(1945, 2030)
    plt.tight_layout()
    return fig


def generate_all_figures_v2():
    """Generate all improved figures."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    figures = [
        ('fig1_historical_timeline_v2.png', fig1_historical_timeline_v2),
        ('fig2_acceleration_comparison_v2.png', fig2_acceleration_comparison_v2),
        ('fig3_calibration_fit_v2.png', fig3_calibration_fit_v2),
        ('fig4_ai_vs_historical_v2.png', fig4_ai_vs_historical_v2),
        ('fig5_parameter_sensitivity_v2.png', fig5_parameter_sensitivity_v2),
        ('fig6_cumulative_acceleration.png', fig6_cumulative_acceleration),
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
            import traceback
            traceback.print_exc()

    print()
    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_figures_v2()
