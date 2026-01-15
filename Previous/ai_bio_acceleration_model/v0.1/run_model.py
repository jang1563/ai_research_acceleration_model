#!/usr/bin/env python3
"""
Main Execution Script for AI-Accelerated Biological Discovery Model

This script runs the complete v0.1 model pipeline:
1. Initializes model with default parameters
2. Runs all scenarios (Pessimistic, Baseline, Optimistic)
3. Generates all outputs (data, figures, summary)
4. Exports results for publication

Usage:
    python run_model.py

Outputs:
    - outputs/results.csv: Complete model results
    - outputs/parameters.json: Model parameters
    - outputs/summary.txt: Summary statistics
    - outputs/fig*.png/pdf: Publication-quality figures

Version: 0.1
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AIBioAccelerationModel, ModelConfig, run_default_model
from visualize import ModelVisualizer, generate_all_visualizations


def main():
    """Run the complete model pipeline."""
    
    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model")
    print("Version: 0.1 (Pilot Model)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize and Run Model
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Initializing model...")
    
    model = AIBioAccelerationModel()
    
    print(f"  - Time horizon: {model.config.t0} to {model.config.T}")
    print(f"  - Number of stages: {model.n_stages}")
    print(f"  - Number of scenarios: {len(model.config.scenarios)}")
    
    print("\n[Step 2/4] Running scenarios...")
    
    results = model.run_all_scenarios()
    
    for scenario in model.config.scenarios:
        print(f"  - {scenario.name} (g = {scenario.g_ai}): Complete")
    
    # -------------------------------------------------------------------------
    # Step 2: Export Data
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Exporting data...")
    
    # Export results
    results_path = os.path.join(output_dir, 'results.csv')
    model.export_results(results_path)
    print(f"  - Results: {results_path}")
    
    # Export parameters
    params_path = os.path.join(output_dir, 'parameters.json')
    model.export_parameters(params_path)
    print(f"  - Parameters: {params_path}")
    
    # Export summary
    summary = model.get_summary_statistics()
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("AI-Accelerated Biological Discovery Model - Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Version: 0.1\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Configuration\n")
        f.write("-" * 60 + "\n")
        f.write(f"Time horizon: {model.config.t0} - {model.config.T}\n")
        f.write(f"Time step: {model.config.dt} year(s)\n\n")
        
        f.write("Pipeline Stages\n")
        f.write("-" * 60 + "\n")
        for stage in model.config.stages:
            f.write(f"S{stage.index}: {stage.name}\n")
            f.write(f"    Baseline duration: {stage.tau_baseline} months\n")
            f.write(f"    Max AI multiplier: {stage.M_max}x\n")
            f.write(f"    Success probability: {stage.p_success:.0%}\n")
            f.write(f"    Saturation rate: {stage.k_saturation}\n\n")
        
        f.write("Scenarios\n")
        f.write("-" * 60 + "\n")
        for scenario in model.config.scenarios:
            f.write(f"{scenario.name}: g = {scenario.g_ai}\n")
            f.write(f"    {scenario.description}\n\n")
        
        f.write("Summary Statistics\n")
        f.write("-" * 60 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("Bottleneck Transitions (Baseline)\n")
        f.write("-" * 60 + "\n")
        transitions = model.get_bottleneck_transitions('Baseline')
        if len(transitions) > 0:
            f.write(transitions.to_string(index=False))
        else:
            f.write("No bottleneck transitions detected\n")
        f.write("\n\n")
        
        f.write("Key Findings\n")
        f.write("-" * 60 + "\n")
        
        baseline_summary = summary[summary['scenario'] == 'Baseline'].iloc[0]
        
        f.write(f"By 2030 (Baseline): {baseline_summary['progress_by_2030']:.1f} equivalent years\n")
        f.write(f"By 2040 (Baseline): {baseline_summary['progress_by_2040']:.1f} equivalent years\n")
        f.write(f"By 2050 (Baseline): {baseline_summary['progress_by_2050']:.1f} equivalent years\n\n")
        
        f.write(f"Maximum progress rate achieved: {baseline_summary['max_progress_rate']:.1f}x\n")
        f.write(f"Final bottleneck stage: S{baseline_summary['final_bottleneck']}\n")
    
    print(f"  - Summary: {summary_path}")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Generating visualizations...")
    
    figures = generate_all_visualizations(model, results, output_dir)
    
    for fig_name in figures.keys():
        print(f"  - {fig_name}.png/.pdf")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Model Run Complete")
    print("=" * 70)
    
    print("\nKey Results (Baseline Scenario):")
    print("-" * 40)
    
    baseline_df = results[results['scenario'] == 'Baseline']
    
    # Progress by decade
    for year in [2030, 2040, 2050]:
        progress = baseline_df[baseline_df['year'] == year]['cumulative_progress'].iloc[0]
        calendar_years = year - 2024
        ratio = progress / calendar_years
        print(f"  By {year}: {progress:.1f} equiv. years ({calendar_years} calendar → {ratio:.1f}x)")
    
    # Bottleneck info
    print("\nBottleneck Transitions:")
    print("-" * 40)
    transitions = model.get_bottleneck_transitions('Baseline')
    if len(transitions) > 0:
        for _, row in transitions.iterrows():
            print(f"  {int(row['year'])}: S{int(row['from_stage'])} ({row['from_name']}) → S{int(row['to_stage'])} ({row['to_name']})")
    else:
        initial_bottleneck = baseline_df['bottleneck_stage'].iloc[0]
        print(f"  No transitions: S{int(initial_bottleneck)} remains bottleneck throughout")
    
    print("\nScenario Comparison (Equiv. Years by 2050):")
    print("-" * 40)
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]
        progress = df[df['year'] == 2050]['cumulative_progress'].iloc[0]
        print(f"  {scenario:12}: {progress:6.1f} years")
    
    print(f"\nAll outputs saved to: {output_dir}/")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
