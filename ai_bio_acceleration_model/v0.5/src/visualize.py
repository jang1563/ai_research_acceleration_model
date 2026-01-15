"""
Visualization Module for AI-Accelerated Biological Discovery Model

Generates publication-quality figures for the model outputs.

Version: 0.5 (Multi-Type AI + Therapeutic Areas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional, List, Dict, Tuple
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme
COLORS = {
    'Pessimistic': '#E74C3C',   # Red
    'Baseline': '#3498DB',       # Blue
    'Optimistic': '#27AE60',     # Green
}

# Therapeutic area colors
AREA_COLORS = {
    'Oncology': '#E74C3C',
    'CNS': '#9B59B6',
    'Infectious Disease': '#27AE60',
    'Rare Disease': '#F39C12',
    'Cardiovascular': '#3498DB',
    'General': '#7F8C8D',
}

# AI type colors
AI_COLORS = {
    'Cognitive': '#3498DB',
    'Robotic': '#E74C3C',
    'Scientific': '#27AE60',
}

# Updated for 10 stages
STAGE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
]


class ModelVisualizer:
    """
    Generate publication-quality visualizations for model results.

    v0.5 Additions:
    - plot_ai_types(): Compare multi-type AI capabilities
    - plot_therapeutic_comparison(): Compare therapeutic areas
    """

    def __init__(
        self,
        results: pd.DataFrame,
        stages: List,
        output_dir: str = 'outputs'
    ):
        """Initialize visualizer."""
        self.results = results
        self.stages = stages
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def plot_ai_capability(self, save: bool = True) -> plt.Figure:
        """Figure 1: AI Capability Growth Over Time (Global)."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax.plot(
                df['year'],
                df.get('ai_capability_global', df.get('ai_capability', df['A_eff_1'])),
                color=COLORS[scenario],
                linewidth=2,
                label=scenario
            )

        ax.set_xlabel('Year')
        ax.set_ylabel('AI Capability (relative to 2024)')
        ax.set_title('Figure 1: AI Capability Growth Over Time')
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig1_ai_capability.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig1_ai_capability.pdf'))

        return fig

    def plot_ai_types(self, scenario: str = 'Baseline', save: bool = True) -> plt.Figure:
        """
        Figure 1b: Multi-Type AI Capability Comparison (v0.5 NEW).

        Shows Cognitive, Robotic, and Scientific AI growth.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        df = self.results[self.results['scenario'] == scenario]
        if len(df) == 0:
            return fig

        # Plot each AI type
        ai_cols = [
            ('ai_capability_cognitive', 'Cognitive', AI_COLORS['Cognitive']),
            ('ai_capability_robotic', 'Robotic', AI_COLORS['Robotic']),
            ('ai_capability_scientific', 'Scientific', AI_COLORS['Scientific']),
        ]

        for col, label, color in ai_cols:
            if col in df.columns:
                ax.plot(
                    df['year'],
                    df[col],
                    color=color,
                    linewidth=2.5,
                    label=f'{label} AI'
                )

        ax.set_xlabel('Year')
        ax.set_ylabel('AI Capability (relative to 2024)')
        ax.set_title(f'Figure 1b: Multi-Type AI Capability ({scenario} Scenario)')
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)

        # Add annotations for 2050 values
        for col, label, color in ai_cols:
            if col in df.columns:
                y_2050 = df[df['year'] == 2050][col].iloc[0]
                ax.annotate(
                    f'{y_2050:.0f}x',
                    xy=(2050, y_2050),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig1b_ai_types.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig1b_ai_types.pdf'))

        return fig

    def plot_therapeutic_comparison(self, save: bool = True) -> plt.Figure:
        """
        Figure 7: Therapeutic Area Comparison (v0.5 NEW).

        Compares progress across different therapeutic areas.
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get baseline scenarios for each therapeutic area
        area_scenarios = [s for s in self.results['scenario'].unique()
                        if s.startswith('Baseline')]

        for scenario in area_scenarios:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue

            area = df['therapeutic_area'].iloc[0]
            color = AREA_COLORS.get(area, '#7F8C8D')
            linestyle = '-' if scenario == 'Baseline' else '--'
            linewidth = 2.5 if scenario == 'Baseline' else 1.5

            label = area if scenario == 'Baseline' else f'{area}'

            ax.plot(
                df['year'],
                df['cumulative_progress'],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label
            )

        # Reference line
        years = np.arange(2024, 2051)
        ax.plot(years, years - 2024, 'k:', alpha=0.3, label='No acceleration')

        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Progress (equivalent years)')
        ax.set_title('Figure 7: Therapeutic Area Comparison (Baseline Scenarios)')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)
        ax.set_ylim(0, None)

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig7_therapeutic_comparison.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig7_therapeutic_comparison.pdf'))

        return fig

    def plot_therapeutic_bar_chart(self, save: bool = True) -> plt.Figure:
        """
        Figure 7b: Therapeutic Area Progress Bar Chart (v0.5 NEW).

        Shows 2050 progress by therapeutic area.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get 2050 progress for each area
        area_progress = []
        for scenario in self.results['scenario'].unique():
            if not scenario.startswith('Baseline'):
                continue

            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue

            area = df['therapeutic_area'].iloc[0]
            progress_2050 = df[df['year'] == 2050]['cumulative_progress'].iloc[0]

            area_progress.append({
                'area': area,
                'progress': progress_2050,
                'color': AREA_COLORS.get(area, '#7F8C8D'),
            })

        # Sort by progress
        area_progress = sorted(area_progress, key=lambda x: x['progress'], reverse=True)

        # Create bar chart
        areas = [a['area'] for a in area_progress]
        progress = [a['progress'] for a in area_progress]
        colors = [a['color'] for a in area_progress]

        bars = ax.barh(areas, progress, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, progress):
            ax.annotate(
                f'{val:.0f} yr',
                xy=(val + 1, bar.get_y() + bar.get_height()/2),
                fontsize=10,
                va='center'
            )

        ax.set_xlabel('Equivalent Years of Progress by 2050')
        ax.set_title('Figure 7b: Progress by Therapeutic Area (Baseline)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig7b_therapeutic_bars.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig7b_therapeutic_bars.pdf'))

        return fig

    def plot_ai_multipliers(self, scenario: str = 'Baseline', save: bool = True) -> plt.Figure:
        """Figure 2: Stage-Specific AI Acceleration Multipliers."""
        fig, ax = plt.subplots(figsize=(10, 6))

        df = self.results[self.results['scenario'] == scenario]
        if len(df) == 0:
            return fig

        for i, stage in enumerate(self.stages):
            col = f'M_{i+1}'
            if col not in df.columns:
                continue
            ax.plot(
                df['year'],
                df[col],
                color=STAGE_COLORS[i],
                linewidth=2,
                label=f'S{i+1}: {stage.name}'
            )

            ax.axhline(
                y=stage.M_max,
                color=STAGE_COLORS[i],
                linestyle='--',
                alpha=0.3,
                linewidth=1
            )

        ax.set_xlabel('Year')
        ax.set_ylabel('AI Acceleration Multiplier')
        ax.set_title(f'Figure 2: Stage-Specific AI Multipliers ({scenario} Scenario)')
        ax.set_yscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig2_ai_multipliers.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig2_ai_multipliers.pdf'))

        return fig

    def plot_effective_service_rates(self, scenario: str = 'Baseline', save: bool = True) -> plt.Figure:
        """Figure 3: Effective Service Rates by Stage."""
        fig, ax = plt.subplots(figsize=(10, 6))

        df = self.results[self.results['scenario'] == scenario]
        if len(df) == 0:
            return fig

        for i, stage in enumerate(self.stages):
            col = f'mu_eff_{i+1}'
            if col not in df.columns:
                continue
            ax.plot(
                df['year'],
                df[col],
                color=STAGE_COLORS[i],
                linewidth=2,
                label=f'S{i+1}: {stage.name}'
            )

        ax.plot(
            df['year'],
            df['throughput'],
            color='black',
            linewidth=3,
            linestyle='-',
            label='System Throughput (min)'
        )

        ax.set_xlabel('Year')
        ax.set_ylabel('Effective Service Rate (projects/year)')
        ax.set_title(f'Figure 3: Effective Service Rates by Stage ({scenario} Scenario)')
        ax.set_yscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)

        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig3_service_rates.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig3_service_rates.pdf'))

        return fig

    def plot_bottleneck_timeline(self, save: bool = True) -> plt.Figure:
        """Figure 4: Bottleneck Stage Over Time."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        for ax, scenario in zip(axes, ['Pessimistic', 'Baseline', 'Optimistic']):
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue

            years = df['year'].values
            bottlenecks = df['bottleneck_stage'].values.astype(int)

            for i in range(len(years) - 1):
                ax.axvspan(
                    years[i],
                    years[i+1],
                    facecolor=STAGE_COLORS[bottlenecks[i] - 1],
                    alpha=0.7
                )

            ax.set_ylabel('Stage')
            ax.set_title(f'{scenario} Scenario')
            ax.set_yticks(range(1, 11))
            ax.set_yticklabels([f'S{i}' for i in range(1, 11)], fontsize=7)
            ax.set_ylim(0.5, 10.5)
            ax.grid(True, alpha=0.3, axis='x')

        axes[-1].set_xlabel('Year')
        axes[-1].set_xlim(2024, 2050)

        legend_elements = [
            mpatches.Patch(facecolor=STAGE_COLORS[i], label=f'S{i+1}: {self.stages[i].name[:15]}...')
            for i in range(min(len(self.stages), 10))
        ]
        fig.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(1.28, 0.5),
            fontsize=7
        )

        fig.suptitle('Figure 4: Bottleneck Stage Over Time', fontsize=12, y=1.02)
        plt.tight_layout()

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4_bottleneck_timeline.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig4_bottleneck_timeline.pdf'))

        return fig

    def plot_progress_rate(self, save: bool = True) -> plt.Figure:
        """Figure 5: Progress Rate Over Time."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax.plot(
                df['year'],
                df['progress_rate'],
                color=COLORS[scenario],
                linewidth=2,
                label=scenario
            )

        ax.set_xlabel('Year')
        ax.set_ylabel('Progress Rate (relative to 2024)')
        ax.set_title('Figure 5: Scientific Progress Rate Over Time')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Rate')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig5_progress_rate.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig5_progress_rate.pdf'))

        return fig

    def plot_cumulative_progress(self, save: bool = True) -> plt.Figure:
        """Figure 6: Cumulative Equivalent Years of Progress."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax.fill_between(
                df['year'],
                0,
                df['cumulative_progress'],
                alpha=0.3,
                color=COLORS[scenario]
            )
            ax.plot(
                df['year'],
                df['cumulative_progress'],
                color=COLORS[scenario],
                linewidth=2,
                label=scenario
            )

        years = np.arange(2024, 2051)
        ax.plot(
            years,
            years - 2024,
            color='gray',
            linestyle='--',
            linewidth=1,
            label='Calendar Years (no AI)'
        )

        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Progress (equivalent years)')
        ax.set_title('Figure 6: Cumulative Equivalent Years of Scientific Progress')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, 2050)
        ax.set_ylim(0, None)

        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig6_cumulative_progress.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig6_cumulative_progress.pdf'))

        return fig

    def plot_summary_dashboard(self, save: bool = True) -> plt.Figure:
        """Summary Dashboard: Combined view of key results."""
        fig = plt.figure(figsize=(16, 12))

        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Panel A: AI Capability
        ax1 = fig.add_subplot(gs[0, 0])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax1.plot(df['year'], df.get('ai_capability_global', df['A_eff_1']),
                    color=COLORS[scenario], linewidth=2, label=scenario)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('AI Capability')
        ax1.set_title('A) AI Capability Growth')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel B: Multi-Type AI
        ax2 = fig.add_subplot(gs[0, 1])
        df = self.results[self.results['scenario'] == 'Baseline']
        if len(df) > 0:
            for col, label, color in [
                ('ai_capability_cognitive', 'Cognitive', AI_COLORS['Cognitive']),
                ('ai_capability_robotic', 'Robotic', AI_COLORS['Robotic']),
                ('ai_capability_scientific', 'Scientific', AI_COLORS['Scientific']),
            ]:
                if col in df.columns:
                    ax2.plot(df['year'], df[col], color=color, linewidth=2, label=label)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('AI Capability')
        ax2.set_title('B) Multi-Type AI (Baseline)')
        ax2.set_yscale('log')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel C: Progress Rate
        ax3 = fig.add_subplot(gs[0, 2])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax3.plot(df['year'], df['progress_rate'], color=COLORS[scenario],
                    linewidth=2, label=scenario)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Progress Rate')
        ax3.set_title('C) Scientific Progress Rate')
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        # Panel D: Cumulative Progress
        ax4 = fig.add_subplot(gs[1, 0])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            ax4.fill_between(df['year'], 0, df['cumulative_progress'],
                           alpha=0.3, color=COLORS[scenario])
            ax4.plot(df['year'], df['cumulative_progress'], color=COLORS[scenario],
                    linewidth=2, label=scenario)
        years = np.arange(2024, 2051)
        ax4.plot(years, years - 2024, 'k--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Equivalent Years')
        ax4.set_title('D) Cumulative Progress')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Panel E: Therapeutic Area Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        for scenario in self.results['scenario'].unique():
            if not scenario.startswith('Baseline'):
                continue
            df = self.results[self.results['scenario'] == scenario]
            if len(df) == 0:
                continue
            area = df['therapeutic_area'].iloc[0]
            color = AREA_COLORS.get(area, '#7F8C8D')
            ax5.plot(df['year'], df['cumulative_progress'],
                    color=color, linewidth=2, label=area[:15])
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Equivalent Years')
        ax5.set_title('E) Therapeutic Areas')
        ax5.legend(loc='upper left', fontsize=7)
        ax5.grid(True, alpha=0.3)

        # Panel F: Bottleneck Summary
        ax6 = fig.add_subplot(gs[1, 2])
        df = self.results[self.results['scenario'] == 'Baseline']
        if len(df) > 0:
            bottleneck_counts = df['bottleneck_stage'].value_counts().sort_index()

            bars = ax6.bar(
                [f'S{int(i)}' for i in bottleneck_counts.index],
                bottleneck_counts.values,
                color=[STAGE_COLORS[int(i)-1] for i in bottleneck_counts.index]
            )
            ax6.set_xlabel('Stage')
            ax6.set_ylabel('Years as Bottleneck')
            ax6.set_title('F) Bottleneck Duration (Baseline)')

        fig.suptitle('AI-Accelerated Biological Discovery Model - Summary Dashboard (v0.5)',
                    fontsize=14, y=1.02)

        if save:
            fig.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'))
            fig.savefig(os.path.join(self.output_dir, 'summary_dashboard.pdf'))

        return fig

    def generate_all_figures(self) -> Dict[str, plt.Figure]:
        """Generate all figures and return them."""
        figures = {
            'fig1_ai_capability': self.plot_ai_capability(),
            'fig1b_ai_types': self.plot_ai_types(),
            'fig2_ai_multipliers': self.plot_ai_multipliers(),
            'fig3_service_rates': self.plot_effective_service_rates(),
            'fig4_bottleneck_timeline': self.plot_bottleneck_timeline(),
            'fig5_progress_rate': self.plot_progress_rate(),
            'fig6_cumulative_progress': self.plot_cumulative_progress(),
            'fig7_therapeutic_comparison': self.plot_therapeutic_comparison(),
            'fig7b_therapeutic_bars': self.plot_therapeutic_bar_chart(),
            'summary_dashboard': self.plot_summary_dashboard(),
        }

        plt.close('all')

        return figures


def generate_all_visualizations(model, results, output_dir='outputs'):
    """Convenience function to generate all visualizations."""
    viz = ModelVisualizer(results, model.config.stages, output_dir)
    figures = viz.generate_all_figures()

    print(f"Generated {len(figures)} figures in '{output_dir}/'")

    return figures


if __name__ == "__main__":
    from model import run_default_model

    model, results = run_default_model()
    generate_all_visualizations(model, results)
