#!/usr/bin/env python3
"""
Generate Version Comparison Figure: v0.3 → v0.4 → v0.4.1

This script creates a visualization showing the evolution of model predictions
across versions, highlighting the impact of:
- v0.4: Dynamic p_success + stage-specific g_ai
- v0.4.1: AI-AI feedback loop

Run this from the v0.4 directory:
    python generate_comparison.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import AIBioAccelerationModel, ModelConfig

print('Generating Version Comparison Figure...')

# Define model configurations for each version
configs = {
    'v0.3 (Static)': ModelConfig(
        enable_dynamic_p_success=False,
        enable_stage_specific_g_ai=False,
        enable_ai_feedback=False,
    ),
    'v0.4 (Dynamic p)': ModelConfig(
        enable_dynamic_p_success=True,
        enable_stage_specific_g_ai=True,
        enable_ai_feedback=False,
    ),
    'v0.4.1 (AI Feedback)': ModelConfig(
        enable_dynamic_p_success=True,
        enable_stage_specific_g_ai=True,
        enable_ai_feedback=True,
        ai_feedback_alpha=0.1,
    ),
}

# Run each version
results_all = {}
for name, config in configs.items():
    model = AIBioAccelerationModel(config)
    results_all[name] = model.run_all_scenarios()
    print(f"  {name}: Complete")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

scenarios = ['Pessimistic', 'Baseline', 'Optimistic']
colors = {'v0.3 (Static)': '#e74c3c', 'v0.4 (Dynamic p)': '#3498db', 'v0.4.1 (AI Feedback)': '#2ecc71'}
linestyles = {'v0.3 (Static)': '--', 'v0.4 (Dynamic p)': '-.', 'v0.4.1 (AI Feedback)': '-'}

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]

    for version, results in results_all.items():
        df = results[results['scenario'] == scenario]
        ax.plot(
            df['year'], df['cumulative_progress'],
            color=colors[version],
            linestyle=linestyles[version],
            linewidth=2.5 if version == 'v0.4.1 (AI Feedback)' else 1.5,
            label=version
        )

    # Reference line
    years = np.arange(2024, 2051)
    ax.plot(years, years - 2024, 'k:', alpha=0.3, label='No acceleration')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Equivalent Years of Progress', fontsize=11)
    ax.set_title(f'{scenario} Scenario', fontsize=12, fontweight='bold')
    ax.set_xlim(2024, 2050)
    ax.set_ylim(0, 160)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Add annotations for 2050 values
    y_offset = 5
    for version, results in results_all.items():
        df = results[results['scenario'] == scenario]
        y_2050 = df[df['year'] == 2050]['cumulative_progress'].iloc[0]
        ax.annotate(f'{y_2050:.0f}', xy=(2050, y_2050), xytext=(-5, y_offset),
                   textcoords='offset points', fontsize=9, color=colors[version])
        y_offset += 12

plt.suptitle('Model Evolution: v0.3 → v0.4 → v0.4.1\n(AI-Accelerated Biological Discovery)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
os.makedirs('outputs', exist_ok=True)
fig.savefig('outputs/fig_version_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig('outputs/fig_version_comparison.pdf', bbox_inches='tight')
plt.close()

# Print summary table
print('\n' + '='*70)
print('VERSION COMPARISON SUMMARY (Equivalent Years by 2050)')
print('='*70)
header = f"{'Version':<25} {'Pessimistic':>12} {'Baseline':>12} {'Optimistic':>12}"
print(header)
print('-'*70)
for version, results in results_all.items():
    values = []
    for scenario in scenarios:
        df = results[results['scenario'] == scenario]
        values.append(df[df['year'] == 2050]['cumulative_progress'].iloc[0])
    print(f"{version:<25} {values[0]:>12.1f} {values[1]:>12.1f} {values[2]:>12.1f}")

# Calculate improvements
print('\n' + '-'*70)
print('IMPROVEMENT FROM v0.3:')
print('-'*70)
v03 = results_all['v0.3 (Static)']
for version in ['v0.4 (Dynamic p)', 'v0.4.1 (AI Feedback)']:
    results = results_all[version]
    improvements = []
    for scenario in scenarios:
        df_new = results[results['scenario'] == scenario]
        df_old = v03[v03['scenario'] == scenario]
        v_new = df_new[df_new['year'] == 2050]['cumulative_progress'].iloc[0]
        v_old = df_old[df_old['year'] == 2050]['cumulative_progress'].iloc[0]
        improvements.append((v_new - v_old) / v_old * 100)
    print(f"{version:<25} {improvements[0]:>+11.1f}% {improvements[1]:>+11.1f}% {improvements[2]:>+11.1f}%")

print('\nFigure saved to outputs/fig_version_comparison.png')
print('Done!')
