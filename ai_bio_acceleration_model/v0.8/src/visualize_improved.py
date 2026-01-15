"""
Improved Visualization Module for AI-Accelerated Biological Discovery Model

Publication-quality figures with enhanced clarity and design.

Version: 0.4.2 (Visualization improvements)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Dict, Tuple
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
})

# Enhanced color scheme
COLORS = {
    'Pessimistic': '#C0392B',   # Dark red
    'Baseline': '#2980B9',       # Strong blue
    'Optimistic': '#27AE60',     # Green
}

COLORS_LIGHT = {
    'Pessimistic': '#F5B7B1',   # Light red
    'Baseline': '#AED6F1',       # Light blue
    'Optimistic': '#ABEBC6',     # Light green
}

# Stage colors - colorblind-friendly palette
STAGE_COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
    '#9C755F', '#BAB0AC'
]


def plot_tornado_improved(sensitivity_results: pd.DataFrame,
                          baseline_value: float,
                          output_dir: str = 'outputs',
                          save: bool = True) -> plt.Figure:
    """
    Improved Tornado Diagram: Parameter Sensitivity Analysis

    Shows the impact of varying each parameter from low to high,
    with clear baseline reference and proper bidirectional bars.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sort by total range (impact)
    df = sensitivity_results.copy()
    df['range'] = df['high_value'] - df['low_value']
    df = df.sort_values('range', ascending=True)

    # Take top 10 parameters
    df = df.tail(10)

    y_pos = np.arange(len(df))

    # Plot bidirectional bars
    for i, (idx, row) in enumerate(df.iterrows()):
        param = row['parameter']
        low_val = row['low_value']
        high_val = row['high_value']

        # Left bar (low parameter value)
        left_width = baseline_value - low_val
        ax.barh(y_pos[i], -left_width, left=baseline_value,
                height=0.7, color='#5DADE2', edgecolor='#2E86AB', linewidth=1,
                label='Low parameter' if i == 0 else '')

        # Right bar (high parameter value)
        right_width = high_val - baseline_value
        ax.barh(y_pos[i], right_width, left=baseline_value,
                height=0.7, color='#58D68D', edgecolor='#28B463', linewidth=1,
                label='High parameter' if i == 0 else '')

        # Add value labels
        ax.text(low_val - 2, y_pos[i], f'{low_val:.0f}',
                va='center', ha='right', fontsize=9, color='#2E86AB')
        ax.text(high_val + 2, y_pos[i], f'{high_val:.0f}',
                va='center', ha='left', fontsize=9, color='#28B463')

    # Baseline reference line
    ax.axvline(x=baseline_value, color='#2C3E50', linewidth=2, linestyle='-', zorder=10)
    ax.text(baseline_value, len(df) + 0.3, f'Baseline\n{baseline_value:.0f} years',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C3E50')

    # Labels
    param_labels = []
    for idx, row in df.iterrows():
        param = row['parameter']
        # Make labels more readable
        if param == 'g_ai':
            label = 'AI Growth Rate (g)'
        elif '_p_success' in param:
            stage = param.split('_')[0]
            label = f'{stage} Success Rate'
        elif '_M_max' in param:
            stage = param.split('_')[0]
            label = f'{stage} Max AI Multiplier'
        else:
            label = param.replace('_', ' ').title()
        param_labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_labels)

    ax.set_xlabel('Equivalent Years of Progress by 2050', fontsize=12)
    ax.set_title('Parameter Sensitivity Analysis\n(Impact of ±20% parameter variation)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='lower right', framealpha=0.9)

    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust x-axis limits
    x_min = df['low_value'].min() - 10
    x_max = df['high_value'].max() + 10
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(output_dir, 'fig_tornado_improved.png'))
        fig.savefig(os.path.join(output_dir, 'fig_tornado_improved.pdf'))

    return fig


def plot_combined_fan_chart(mc_results: Dict[str, pd.DataFrame],
                            output_dir: str = 'outputs',
                            save: bool = True) -> plt.Figure:
    """
    Combined Fan Chart: All scenarios with uncertainty bands in one figure.

    Shows 50% and 90% confidence intervals for all three scenarios.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    scenarios = ['Pessimistic', 'Baseline', 'Optimistic']

    for scenario in scenarios:
        if scenario not in mc_results:
            continue

        df = mc_results[scenario]
        years = df['year'].values

        # Get percentiles (handle both 'median' and 'p50' column names)
        median_col = 'median' if 'median' in df.columns else 'p50'
        median = df[median_col].values
        p5 = df['p5'].values
        p95 = df['p95'].values
        p25 = df['p25'].values
        p75 = df['p75'].values

        color = COLORS[scenario]
        color_light = COLORS_LIGHT[scenario]

        # 90% CI band
        ax.fill_between(years, p5, p95, alpha=0.15, color=color, linewidth=0)

        # 50% CI band
        ax.fill_between(years, p25, p75, alpha=0.3, color=color, linewidth=0)

        # Median line
        ax.plot(years, median, color=color, linewidth=2.5, label=f'{scenario}')

    # Reference line (no acceleration)
    years_ref = np.arange(2024, 2051)
    ax.plot(years_ref, years_ref - 2024, color='#7F8C8D', linewidth=1.5,
            linestyle='--', label='No AI acceleration')

    # Annotations for 2050 values
    for scenario in scenarios:
        if scenario in mc_results:
            df = mc_results[scenario]
            final_row = df[df['year'] == 2050].iloc[0]
            median_col = 'median' if 'median' in df.columns else 'p50'
            median = final_row[median_col]
            p5 = final_row['p5']
            p95 = final_row['p95']

            y_offset = {'Pessimistic': -8, 'Baseline': 0, 'Optimistic': 8}
            ax.annotate(f'{median:.0f}\n[{p5:.0f}-{p95:.0f}]',
                       xy=(2050, median),
                       xytext=(2052, median + y_offset[scenario]),
                       fontsize=10, fontweight='bold', color=COLORS[scenario],
                       ha='left', va='center')

    # Labels and title
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Equivalent Years of Scientific Progress', fontsize=12)
    ax.set_title('AI-Accelerated Biological Discovery: Scenario Projections\nwith 50% and 90% Confidence Intervals',
                 fontsize=14, fontweight='bold', pad=15)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['Pessimistic'], linewidth=2.5, label='Pessimistic'),
        Line2D([0], [0], color=COLORS['Baseline'], linewidth=2.5, label='Baseline'),
        Line2D([0], [0], color=COLORS['Optimistic'], linewidth=2.5, label='Optimistic'),
        Line2D([0], [0], color='#7F8C8D', linewidth=1.5, linestyle='--', label='No AI'),
        mpatches.Patch(facecolor='gray', alpha=0.3, label='50% CI'),
        mpatches.Patch(facecolor='gray', alpha=0.15, label='90% CI'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=10)

    # Grid and limits
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(2024, 2055)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(output_dir, 'fig_combined_fan_chart.png'))
        fig.savefig(os.path.join(output_dir, 'fig_combined_fan_chart.pdf'))

    return fig


def plot_bottleneck_heatmap(results: pd.DataFrame,
                            stages: List,
                            output_dir: str = 'outputs',
                            save: bool = True) -> plt.Figure:
    """
    Bottleneck Heatmap: Time × Stage matrix showing constraint intensity.

    Shows effective service rates as a heatmap to visualize which stages
    are closest to being bottlenecks over time.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    scenarios = ['Pessimistic', 'Baseline', 'Optimistic']

    for ax, scenario in zip(axes, scenarios):
        df = results[results['scenario'] == scenario]

        # Build matrix: rows = stages, columns = years
        years = df['year'].values
        n_stages = len(stages)

        # Get effective service rates
        matrix = np.zeros((n_stages, len(years)))
        for i in range(n_stages):
            col = f'mu_eff_{i+1}'
            if col in df.columns:
                matrix[i, :] = df[col].values

        # Normalize by row (each stage's max)
        # Lower values = closer to bottleneck
        # We want to show "constraint intensity" = 1 / normalized_rate
        min_rates = matrix.min(axis=0, keepdims=True)
        constraint_intensity = min_rates / (matrix + 1e-10)  # Higher = more constrained

        # Create heatmap
        cmap = LinearSegmentedColormap.from_list('bottleneck',
                                                  ['#FFFFFF', '#FFF3CD', '#F8D7DA', '#C0392B'])

        im = ax.imshow(constraint_intensity, aspect='auto', cmap=cmap,
                       extent=[years[0], years[-1], n_stages + 0.5, 0.5],
                       vmin=0, vmax=1)

        # Mark actual bottleneck with black outline
        bottlenecks = df['bottleneck_stage'].values.astype(int)
        for j, (year, bn) in enumerate(zip(years, bottlenecks)):
            if j < len(years) - 1:
                ax.add_patch(plt.Rectangle((year, bn - 0.5), 1, 1,
                                          fill=False, edgecolor='black', linewidth=1.5))

        ax.set_xlabel('Year', fontsize=11)
        ax.set_title(f'{scenario} Scenario', fontsize=12, fontweight='bold')

        # Y-axis labels
        if ax == axes[0]:
            ax.set_ylabel('Pipeline Stage', fontsize=11)
            ax.set_yticks(range(1, n_stages + 1))
            ax.set_yticklabels([f'S{i+1}: {stages[i].name[:12]}' for i in range(n_stages)],
                              fontsize=9)

        ax.set_xlim(2024, 2050)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Constraint Intensity\n(1.0 = bottleneck)', fontsize=10)

    fig.suptitle('Bottleneck Analysis: Which Stages Constrain Progress?',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(output_dir, 'fig_bottleneck_heatmap.png'))
        fig.savefig(os.path.join(output_dir, 'fig_bottleneck_heatmap.pdf'))

    return fig


def plot_summary_dashboard_improved(results: pd.DataFrame,
                                    stages: List,
                                    mc_results: Dict[str, pd.DataFrame] = None,
                                    output_dir: str = 'outputs',
                                    save: bool = True) -> plt.Figure:
    """
    Improved Summary Dashboard with four informative panels.
    """
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ========== Panel A: Cumulative Progress with Uncertainty ==========
    ax1 = fig.add_subplot(gs[0, 0])

    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]

        # If MC results available, show uncertainty
        if mc_results and scenario in mc_results:
            mc_df = mc_results[scenario]
            median_col = 'median' if 'median' in mc_df.columns else 'p50'
            ax1.fill_between(mc_df['year'], mc_df['p5'], mc_df['p95'],
                           alpha=0.2, color=COLORS[scenario])
            ax1.plot(mc_df['year'], mc_df[median_col], color=COLORS[scenario],
                    linewidth=2.5, label=scenario)
        else:
            ax1.plot(df['year'], df['cumulative_progress'], color=COLORS[scenario],
                    linewidth=2.5, label=scenario)

    # Reference line
    years = np.arange(2024, 2051)
    ax1.plot(years, years - 2024, 'k--', linewidth=1, alpha=0.5, label='No AI')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Equivalent Years')
    ax1.set_title('A) Cumulative Scientific Progress', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2024, 2050)

    # ========== Panel B: Progress Rate Comparison ==========
    ax2 = fig.add_subplot(gs[0, 1])

    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]
        ax2.plot(df['year'], df['progress_rate'], color=COLORS[scenario],
                linewidth=2.5, label=scenario)

    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Progress Rate (× baseline)')
    ax2.set_title('B) Scientific Progress Rate', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2024, 2050)

    # Add annotation for final values
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]
        final_rate = df['progress_rate'].iloc[-1]
        ax2.annotate(f'{final_rate:.1f}×', xy=(2050, final_rate),
                    xytext=(2051, final_rate), fontsize=9, color=COLORS[scenario],
                    fontweight='bold', va='center')

    # ========== Panel C: Milestone Bar Chart ==========
    ax3 = fig.add_subplot(gs[1, 0])

    milestones = [2030, 2040, 2050]
    x = np.arange(len(milestones))
    width = 0.25

    for i, scenario in enumerate(['Pessimistic', 'Baseline', 'Optimistic']):
        df = results[results['scenario'] == scenario]
        values = [df[df['year'] == y]['cumulative_progress'].values[0] for y in milestones]
        bars = ax3.bar(x + (i - 1) * width, values, width,
                      label=scenario, color=COLORS[scenario], edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Equivalent Years')
    ax3.set_title('C) Progress at Key Milestones', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(milestones)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)

    # ========== Panel D: p_success Evolution (if available) ==========
    ax4 = fig.add_subplot(gs[1, 1])

    # Check if p_success columns exist
    p_success_cols = [col for col in results.columns if col.startswith('p_success_')]

    if p_success_cols:
        df_baseline = results[results['scenario'] == 'Baseline']

        # Plot Phase I, II, III success rates
        clinical_stages = [(6, 'Phase I'), (7, 'Phase II'), (8, 'Phase III')]
        colors_clinical = ['#3498DB', '#E74C3C', '#27AE60']

        for (stage_idx, stage_name), color in zip(clinical_stages, colors_clinical):
            col = f'p_success_{stage_idx}'
            if col in df_baseline.columns:
                ax4.plot(df_baseline['year'], df_baseline[col] * 100,
                        color=color, linewidth=2.5, label=stage_name)

        ax4.set_xlabel('Year')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('D) Clinical Trial Success Rate Evolution', fontweight='bold')
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(2024, 2050)
        ax4.set_ylim(0, 100)
    else:
        # Alternative: Acceleration factor comparison
        scenarios = ['Pessimistic', 'Baseline', 'Optimistic']
        final_progress = []
        calendar_years = 26  # 2050 - 2024

        for scenario in scenarios:
            df = results[results['scenario'] == scenario]
            final_progress.append(df['cumulative_progress'].iloc[-1])

        acceleration = [p / calendar_years for p in final_progress]

        bars = ax4.bar(scenarios, acceleration, color=[COLORS[s] for s in scenarios],
                      edgecolor='white', linewidth=2)

        for bar, acc in zip(bars, acceleration):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{acc:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax4.set_ylabel('Acceleration Factor')
        ax4.set_title('D) Overall Acceleration by 2050', fontweight='bold')
        ax4.set_ylim(0, max(acceleration) * 1.2)
        ax4.grid(True, axis='y', alpha=0.3)

    fig.suptitle('AI-Accelerated Biological Discovery Model\nSummary Dashboard (v0.4.1)',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(output_dir, 'summary_dashboard_improved.png'))
        fig.savefig(os.path.join(output_dir, 'summary_dashboard_improved.pdf'))

    return fig


def generate_improved_visualizations(results: pd.DataFrame,
                                     stages: List,
                                     sensitivity_df: pd.DataFrame = None,
                                     mc_results: Dict[str, pd.DataFrame] = None,
                                     baseline_value: float = 83.0,
                                     output_dir: str = 'outputs'):
    """
    Generate all improved visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}

    print("Generating improved visualizations...")

    # 1. Improved Tornado Diagram
    if sensitivity_df is not None:
        print("  - Tornado diagram...")
        figures['tornado'] = plot_tornado_improved(sensitivity_df, baseline_value, output_dir)

    # 2. Combined Fan Chart
    if mc_results is not None:
        print("  - Combined fan chart...")
        figures['fan_chart'] = plot_combined_fan_chart(mc_results, output_dir)

    # 3. Bottleneck Heatmap
    print("  - Bottleneck heatmap...")
    figures['heatmap'] = plot_bottleneck_heatmap(results, stages, output_dir)

    # 4. Improved Summary Dashboard
    print("  - Summary dashboard...")
    figures['dashboard'] = plot_summary_dashboard_improved(results, stages, mc_results, output_dir)

    plt.close('all')

    print(f"\nGenerated {len(figures)} improved figures in '{output_dir}/'")

    return figures


if __name__ == "__main__":
    print("Run this module via generate_improved_figures.py")
