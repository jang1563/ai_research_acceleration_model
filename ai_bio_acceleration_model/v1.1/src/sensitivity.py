"""
Sensitivity Analysis Module for AI-Accelerated Biological Discovery Model

Provides tools for:
1. One-at-a-time (OAT) sensitivity analysis
2. Parameter sweep analysis
3. Tornado diagrams for parameter importance
4. Identification of highest-leverage parameters

Version: 0.3
Date: January 2026

References:
- Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer"
- Pianosi et al. (2016) "Sensitivity analysis of environmental models"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import warnings

from model import AIBioAccelerationModel, ModelConfig, Stage, Scenario


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results."""
    parameter_name: str
    stage_index: Optional[int]
    base_value: float
    test_values: np.ndarray
    output_values: np.ndarray
    sensitivity_index: float  # Normalized sensitivity
    elasticity: float  # % change in output / % change in input


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on the AI Bio Acceleration Model.

    Key analyses:
    1. OAT (One-at-a-time): Vary one parameter while holding others constant
    2. Parameter sweeps: Systematic variation across parameter space
    3. Tornado diagrams: Rank parameters by impact
    """

    def __init__(self, base_config: Optional[ModelConfig] = None):
        """
        Initialize with a base configuration.

        Parameters
        ----------
        base_config : ModelConfig, optional
            Base configuration to perturb. If None, uses defaults.
        """
        self.base_config = base_config or ModelConfig()
        self.base_model = AIBioAccelerationModel(self.base_config)
        self.results: Dict[str, SensitivityResult] = {}

    def _run_model_for_output(
        self,
        config: ModelConfig,
        scenario_name: str = 'Baseline',
        output_metric: str = 'progress_by_2050'
    ) -> float:
        """
        Run model and extract a single output metric.

        Parameters
        ----------
        config : ModelConfig
            Model configuration to use
        scenario_name : str
            Which scenario to analyze
        output_metric : str
            Which output to extract ('progress_by_2050', 'progress_by_2040',
            'max_progress_rate', etc.)

        Returns
        -------
        float
            The requested output metric value
        """
        model = AIBioAccelerationModel(config)
        model.run_all_scenarios()
        summary = model.get_summary_statistics()

        row = summary[summary['scenario'] == scenario_name].iloc[0]
        return row[output_metric]

    def oat_analysis_stage_parameter(
        self,
        stage_index: int,
        parameter_name: str,
        variation_range: Tuple[float, float] = (0.5, 1.5),
        n_points: int = 11,
        scenario_name: str = 'Baseline',
        output_metric: str = 'progress_by_2050'
    ) -> SensitivityResult:
        """
        One-at-a-time sensitivity analysis for a stage parameter.

        Parameters
        ----------
        stage_index : int
            Stage index (1-10)
        parameter_name : str
            Parameter to vary ('M_max', 'tau_baseline', 'p_success', 'k_saturation')
        variation_range : tuple
            (min_multiplier, max_multiplier) relative to base value
        n_points : int
            Number of test points
        scenario_name : str
            Scenario to analyze
        output_metric : str
            Output metric to track

        Returns
        -------
        SensitivityResult
            Results of the sensitivity analysis
        """
        # Get base value
        base_stage = self.base_config.stages[stage_index - 1]
        base_value = getattr(base_stage, parameter_name)

        # Create test values
        multipliers = np.linspace(variation_range[0], variation_range[1], n_points)
        test_values = base_value * multipliers

        # Run model for each test value
        output_values = []
        for test_val in test_values:
            # Create modified config
            config = deepcopy(self.base_config)
            setattr(config.stages[stage_index - 1], parameter_name, test_val)

            try:
                output = self._run_model_for_output(config, scenario_name, output_metric)
                output_values.append(output)
            except Exception as e:
                warnings.warn(f"Error at {parameter_name}={test_val}: {e}")
                output_values.append(np.nan)

        output_values = np.array(output_values)

        # Calculate sensitivity metrics
        # Get base output
        base_output = self._run_model_for_output(self.base_config, scenario_name, output_metric)

        # Sensitivity index: (max - min) / base
        valid_outputs = output_values[~np.isnan(output_values)]
        sensitivity_index = (valid_outputs.max() - valid_outputs.min()) / base_output

        # Elasticity at base point (% change output / % change input)
        # Use central difference approximation
        center_idx = n_points // 2
        if center_idx > 0 and center_idx < n_points - 1:
            delta_output = output_values[center_idx + 1] - output_values[center_idx - 1]
            delta_input = test_values[center_idx + 1] - test_values[center_idx - 1]
            elasticity = (delta_output / base_output) / (delta_input / base_value)
        else:
            elasticity = np.nan

        result = SensitivityResult(
            parameter_name=f"S{stage_index}_{parameter_name}",
            stage_index=stage_index,
            base_value=base_value,
            test_values=test_values,
            output_values=output_values,
            sensitivity_index=sensitivity_index,
            elasticity=elasticity
        )

        self.results[result.parameter_name] = result
        return result

    def oat_analysis_g_ai(
        self,
        variation_range: Tuple[float, float] = (0.1, 1.0),
        n_points: int = 19,
        scenario_name: str = 'Baseline',
        output_metric: str = 'progress_by_2050'
    ) -> SensitivityResult:
        """
        One-at-a-time sensitivity analysis for AI growth rate (g_ai).

        This modifies the scenario's g_ai value directly.
        """
        # Get base g_ai
        base_scenario = next(s for s in self.base_config.scenarios if s.name == scenario_name)
        base_value = base_scenario.g_ai

        # Create test values (absolute, not multiplier)
        test_values = np.linspace(variation_range[0], variation_range[1], n_points)

        # Run model for each test value
        output_values = []
        for test_val in test_values:
            config = deepcopy(self.base_config)
            for scenario in config.scenarios:
                if scenario.name == scenario_name:
                    scenario.g_ai = test_val

            try:
                output = self._run_model_for_output(config, scenario_name, output_metric)
                output_values.append(output)
            except Exception as e:
                warnings.warn(f"Error at g_ai={test_val}: {e}")
                output_values.append(np.nan)

        output_values = np.array(output_values)

        # Calculate sensitivity metrics
        base_output = self._run_model_for_output(self.base_config, scenario_name, output_metric)
        valid_outputs = output_values[~np.isnan(output_values)]
        sensitivity_index = (valid_outputs.max() - valid_outputs.min()) / base_output

        # Elasticity
        center_idx = np.argmin(np.abs(test_values - base_value))
        if center_idx > 0 and center_idx < n_points - 1:
            delta_output = output_values[center_idx + 1] - output_values[center_idx - 1]
            delta_input = test_values[center_idx + 1] - test_values[center_idx - 1]
            elasticity = (delta_output / base_output) / (delta_input / base_value)
        else:
            elasticity = np.nan

        result = SensitivityResult(
            parameter_name="g_ai",
            stage_index=None,
            base_value=base_value,
            test_values=test_values,
            output_values=output_values,
            sensitivity_index=sensitivity_index,
            elasticity=elasticity
        )

        self.results["g_ai"] = result
        return result

    def run_full_oat_analysis(
        self,
        parameters: List[str] = ['M_max', 'p_success'],
        stages: Optional[List[int]] = None,
        scenario_name: str = 'Baseline',
        output_metric: str = 'progress_by_2050',
        variation_range: Tuple[float, float] = (0.5, 1.5),
        n_points: int = 11
    ) -> pd.DataFrame:
        """
        Run OAT analysis on all specified parameters for all stages.

        Parameters
        ----------
        parameters : list
            Stage parameters to analyze
        stages : list, optional
            Stage indices to analyze. If None, analyzes all stages.
        scenario_name : str
            Scenario to analyze
        output_metric : str
            Output metric to track
        variation_range : tuple
            Multiplier range for parameter variation
        n_points : int
            Number of test points per parameter

        Returns
        -------
        pd.DataFrame
            Summary of sensitivity results sorted by impact
        """
        if stages is None:
            stages = list(range(1, len(self.base_config.stages) + 1))

        # Run g_ai analysis first
        print("Analyzing g_ai...")
        self.oat_analysis_g_ai(
            scenario_name=scenario_name,
            output_metric=output_metric,
            n_points=n_points
        )

        # Run stage parameter analyses
        total = len(stages) * len(parameters)
        current = 0

        for stage_idx in stages:
            stage_name = self.base_config.stages[stage_idx - 1].name
            for param in parameters:
                current += 1
                print(f"[{current}/{total}] Analyzing S{stage_idx} {param} ({stage_name[:20]}...)")

                self.oat_analysis_stage_parameter(
                    stage_index=stage_idx,
                    parameter_name=param,
                    variation_range=variation_range,
                    n_points=n_points,
                    scenario_name=scenario_name,
                    output_metric=output_metric
                )

        # Create summary dataframe
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'parameter': name,
                'stage_index': result.stage_index,
                'base_value': result.base_value,
                'sensitivity_index': result.sensitivity_index,
                'elasticity': result.elasticity,
                'output_range': f"{result.output_values.min():.1f} - {result.output_values.max():.1f}"
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('sensitivity_index', ascending=False)

        return df

    def plot_tornado_diagram(
        self,
        top_n: int = 15,
        output_metric: str = 'progress_by_2050',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create tornado diagram showing parameter importance.

        Parameters
        ----------
        top_n : int
            Number of top parameters to show
        output_metric : str
            Output metric name for label
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            The tornado diagram figure
        """
        if not self.results:
            raise ValueError("No sensitivity results. Run analysis first.")

        # Sort by sensitivity index
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.sensitivity_index,
            reverse=True
        )[:top_n]

        # Reverse for plotting (highest at top)
        sorted_results = sorted_results[::-1]

        fig, ax = plt.subplots(figsize=figsize)

        y_positions = np.arange(len(sorted_results))

        # Get base output
        base_output = self._run_model_for_output(self.base_config, 'Baseline', 'progress_by_2050')

        for i, result in enumerate(sorted_results):
            low = result.output_values.min()
            high = result.output_values.max()

            # Color based on elasticity sign
            color = '#E74C3C' if result.elasticity < 0 else '#27AE60'

            ax.barh(
                y_positions[i],
                high - low,
                left=low,
                height=0.6,
                color=color,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

            # Add elasticity annotation
            ax.annotate(
                f'ε={result.elasticity:.2f}',
                xy=(high + 1, y_positions[i]),
                fontsize=8,
                va='center'
            )

        # Add base output line
        ax.axvline(base_output, color='black', linestyle='--', linewidth=1.5, label='Base value')

        ax.set_yticks(y_positions)
        ax.set_yticklabels([r.parameter_name for r in sorted_results])
        ax.set_xlabel(f'{output_metric} (equivalent years)')
        ax.set_title(f'Tornado Diagram: Parameter Sensitivity\n(Top {top_n} parameters by impact)')

        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', alpha=0.7, label='Positive elasticity'),
            Patch(facecolor='#E74C3C', alpha=0.7, label='Negative elasticity'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

        return fig

    def plot_parameter_sweep(
        self,
        parameter_name: str,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the sensitivity curve for a single parameter.

        Parameters
        ----------
        parameter_name : str
            Name of parameter to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            The parameter sweep figure
        """
        if parameter_name not in self.results:
            raise ValueError(f"No results for parameter '{parameter_name}'")

        result = self.results[parameter_name]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(result.test_values, result.output_values, 'o-',
                color='#3498DB', linewidth=2, markersize=6)

        # Mark base value
        ax.axvline(result.base_value, color='gray', linestyle='--',
                   linewidth=1, label=f'Base value = {result.base_value:.2f}')

        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Equivalent Years by 2050')
        ax.set_title(f'Sensitivity Analysis: {parameter_name}\n'
                     f'Sensitivity Index = {result.sensitivity_index:.3f}, '
                     f'Elasticity = {result.elasticity:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def identify_highest_leverage_parameters(
        self,
        top_n: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Identify the parameters with highest leverage for policy intervention.

        Returns list of (parameter_name, sensitivity_index, recommendation)
        """
        if not self.results:
            raise ValueError("No sensitivity results. Run analysis first.")

        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.sensitivity_index,
            reverse=True
        )[:top_n]

        recommendations = []
        for result in sorted_results:
            if 'g_ai' in result.parameter_name:
                rec = "Invest in AI R&D to increase capability growth rate"
            elif 'M_max' in result.parameter_name:
                stage_idx = result.stage_index
                stage_name = self.base_config.stages[stage_idx - 1].name
                rec = f"Develop AI tools to increase automation ceiling for {stage_name}"
            elif 'p_success' in result.parameter_name:
                stage_idx = result.stage_index
                stage_name = self.base_config.stages[stage_idx - 1].name
                rec = f"Improve success rates in {stage_name} through better methods"
            else:
                rec = "Further analysis needed"

            recommendations.append((
                result.parameter_name,
                result.sensitivity_index,
                rec
            ))

        return recommendations


def run_sensitivity_analysis(
    output_dir: str = 'outputs',
    scenario_name: str = 'Baseline'
) -> Tuple[SensitivityAnalyzer, pd.DataFrame]:
    """
    Convenience function to run complete sensitivity analysis.

    Parameters
    ----------
    output_dir : str
        Directory for output files
    scenario_name : str
        Scenario to analyze

    Returns
    -------
    analyzer : SensitivityAnalyzer
        The analyzer with results
    summary : pd.DataFrame
        Summary dataframe sorted by sensitivity
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    analyzer = SensitivityAnalyzer()

    print("=" * 60)
    print("Running Full Sensitivity Analysis")
    print("=" * 60)

    # Run OAT analysis on key parameters
    summary = analyzer.run_full_oat_analysis(
        parameters=['M_max', 'p_success'],
        scenario_name=scenario_name,
        variation_range=(0.5, 1.5),
        n_points=9
    )

    print("\n" + "=" * 60)
    print("Sensitivity Analysis Complete")
    print("=" * 60)

    # Save summary
    summary.to_csv(os.path.join(output_dir, 'sensitivity_summary.csv'), index=False)

    # Generate tornado diagram
    print("\nGenerating tornado diagram...")
    analyzer.plot_tornado_diagram(
        top_n=15,
        save_path=os.path.join(output_dir, 'fig_tornado.png')
    )

    # Plot top 3 parameter sweeps
    print("Generating parameter sweep plots...")
    top_params = summary.head(3)['parameter'].tolist()
    for param in top_params:
        analyzer.plot_parameter_sweep(
            param,
            save_path=os.path.join(output_dir, f'fig_sweep_{param}.png')
        )

    # Identify highest leverage parameters
    print("\n" + "=" * 60)
    print("Highest Leverage Parameters for Policy Intervention:")
    print("=" * 60)
    recommendations = analyzer.identify_highest_leverage_parameters(top_n=5)
    for param, sens, rec in recommendations:
        print(f"\n{param} (sensitivity = {sens:.3f}):")
        print(f"  → {rec}")

    plt.close('all')

    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = run_sensitivity_analysis()
    print("\n\nTop 10 Parameters by Sensitivity:")
    print(summary.head(10).to_string(index=False))
