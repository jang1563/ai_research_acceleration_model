"""
Visualization Module for AI-Accelerated Biological Discovery Model

Generates publication-quality figures for the model outputs.

Version: 0.2 (Updated for 10-stage pipeline)
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

# Updated for 10 stages (v0.2)
STAGE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'  # Added 2 more colors for 10 stages
]


class ModelVisualizer:
    """
    Generate publication-quality visualizations for model results.
    """
    
    def __init__(
        self, 
        results: pd.DataFrame, 
        stages: List, 
        output_dir: str = 'outputs'
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        results : pd.DataFrame
            Model results from run_all_scenarios()
        stages : List[Stage]
            List of Stage objects from model config
        output_dir : str
            Directory for saving figures
        """
        self.results = results
        self.stages = stages
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_ai_capability(self, save: bool = True) -> plt.Figure:
        """
        Figure 1: AI Capability Growth Over Time.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            ax.plot(
                df['year'], 
                df['ai_capability'],
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
    
    def plot_ai_multipliers(self, scenario: str = 'Baseline', save: bool = True) -> plt.Figure:
        """
        Figure 2: Stage-Specific AI Acceleration Multipliers.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = self.results[self.results['scenario'] == scenario]
        
        for i, stage in enumerate(self.stages):
            col = f'M_{i+1}'
            ax.plot(
                df['year'],
                df[col],
                color=STAGE_COLORS[i],
                linewidth=2,
                label=f'S{i+1}: {stage.name}'
            )
            
            # Add horizontal line for M_max
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
        """
        Figure 3: Effective Service Rates by Stage (identifies bottleneck).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = self.results[self.results['scenario'] == scenario]
        
        for i, stage in enumerate(self.stages):
            col = f'mu_eff_{i+1}'
            ax.plot(
                df['year'],
                df[col],
                color=STAGE_COLORS[i],
                linewidth=2,
                label=f'S{i+1}: {stage.name}'
            )
        
        # Highlight system throughput (minimum)
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
        """
        Figure 4: Bottleneck Stage Over Time (All Scenarios).
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        for ax, scenario in zip(axes, ['Pessimistic', 'Baseline', 'Optimistic']):
            df = self.results[self.results['scenario'] == scenario]
            
            # Create color-coded bar for each year
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
            ax.set_yticks(range(1, 9))
            ax.set_yticklabels([f'S{i}' for i in range(1, 9)], fontsize=8)
            ax.set_ylim(0.5, 8.5)
            ax.grid(True, alpha=0.3, axis='x')
        
        axes[-1].set_xlabel('Year')
        axes[-1].set_xlim(2024, 2050)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=STAGE_COLORS[i], label=f'S{i+1}: {self.stages[i].name}')
            for i in range(len(self.stages))
        ]
        fig.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(1.25, 0.5),
            fontsize=8
        )
        
        fig.suptitle('Figure 4: Bottleneck Stage Over Time', fontsize=12, y=1.02)
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'fig4_bottleneck_timeline.png'))
            fig.savefig(os.path.join(self.output_dir, 'fig4_bottleneck_timeline.pdf'))
        
        return fig
    
    def plot_progress_rate(self, save: bool = True) -> plt.Figure:
        """
        Figure 5: Progress Rate Over Time.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
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
        """
        Figure 6: Cumulative Equivalent Years of Progress.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
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
        
        # Add reference line for calendar years (no acceleration)
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
        """
        Summary Dashboard: Combined view of key results.
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: AI Capability
        ax1 = fig.add_subplot(gs[0, 0])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            ax1.plot(df['year'], df['ai_capability'], color=COLORS[scenario], 
                    linewidth=2, label=scenario)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('AI Capability')
        ax1.set_title('A) AI Capability Growth')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Progress Rate
        ax2 = fig.add_subplot(gs[0, 1])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            ax2.plot(df['year'], df['progress_rate'], color=COLORS[scenario],
                    linewidth=2, label=scenario)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Progress Rate')
        ax2.set_title('B) Scientific Progress Rate')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Cumulative Progress
        ax3 = fig.add_subplot(gs[1, 0])
        for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
            df = self.results[self.results['scenario'] == scenario]
            ax3.fill_between(df['year'], 0, df['cumulative_progress'],
                           alpha=0.3, color=COLORS[scenario])
            ax3.plot(df['year'], df['cumulative_progress'], color=COLORS[scenario],
                    linewidth=2, label=scenario)
        years = np.arange(2024, 2051)
        ax3.plot(years, years - 2024, 'k--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Equivalent Years')
        ax3.set_title('C) Cumulative Progress')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Bottleneck Summary (Baseline)
        ax4 = fig.add_subplot(gs[1, 1])
        df = self.results[self.results['scenario'] == 'Baseline']
        bottleneck_counts = df['bottleneck_stage'].value_counts().sort_index()
        
        bars = ax4.bar(
            [f'S{i}' for i in bottleneck_counts.index],
            bottleneck_counts.values,
            color=[STAGE_COLORS[i-1] for i in bottleneck_counts.index]
        )
        ax4.set_xlabel('Stage')
        ax4.set_ylabel('Years as Bottleneck')
        ax4.set_title('D) Bottleneck Duration (Baseline)')
        
        # Add stage names
        stage_labels = [self.stages[i-1].name for i in bottleneck_counts.index]
        ax4.set_xticklabels([f'S{i}\n{self.stages[i-1].name[:8]}...' 
                           for i in bottleneck_counts.index], fontsize=7)
        
        fig.suptitle('AI-Accelerated Biological Discovery Model - Summary Dashboard (v0.2)',
                    fontsize=14, y=1.02)
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'))
            fig.savefig(os.path.join(self.output_dir, 'summary_dashboard.pdf'))
        
        return fig
    
    def generate_all_figures(self) -> Dict[str, plt.Figure]:
        """
        Generate all figures and return them.
        """
        figures = {
            'fig1_ai_capability': self.plot_ai_capability(),
            'fig2_ai_multipliers': self.plot_ai_multipliers(),
            'fig3_service_rates': self.plot_effective_service_rates(),
            'fig4_bottleneck_timeline': self.plot_bottleneck_timeline(),
            'fig5_progress_rate': self.plot_progress_rate(),
            'fig6_cumulative_progress': self.plot_cumulative_progress(),
            'summary_dashboard': self.plot_summary_dashboard(),
        }
        
        plt.close('all')  # Clean up
        
        return figures


def generate_all_visualizations(model, results, output_dir='outputs'):
    """
    Convenience function to generate all visualizations.
    
    Parameters
    ----------
    model : AIBioAccelerationModel
        The model instance
    results : pd.DataFrame
        Combined results from run_all_scenarios()
    output_dir : str
        Directory for saving figures
    """
    viz = ModelVisualizer(results, model.config.stages, output_dir)
    figures = viz.generate_all_figures()
    
    print(f"Generated {len(figures)} figures in '{output_dir}/'")
    
    return figures


if __name__ == "__main__":
    # Import model and run
    from model import run_default_model
    
    model, results = run_default_model()
    generate_all_visualizations(model, results)
