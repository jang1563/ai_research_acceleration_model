#!/usr/bin/env python3
"""
v0.6 Visualizations
===================

Generates figures showing:
1. v0.5 vs v0.6 model comparison
2. Triage constraint impact by domain
3. Backlog dynamics over time
4. Validation results against case studies
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up paths
v06_src = Path(__file__).parent / "src"
v05_src = Path(__file__).parent.parent / "v0.5" / "src"
v04_src = Path(__file__).parent.parent / "v0.4" / "src"

sys.path.insert(0, str(v04_src))
sys.path.insert(0, str(v05_src))

# Import v0.6 modules using importlib
import importlib.util

def _import_v06_module(name):
    spec = importlib.util.spec_from_file_location(f"v06_{name}", v06_src / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_integrated = _import_v06_module("integrated_v06_model")
IntegratedV06Model = _integrated.IntegratedV06Model
DOMAIN_MAPPING = _integrated.DOMAIN_MAPPING

_triage = _import_v06_module("triage_model")
DOMAIN_TRIAGE_PROFILES = _triage.DOMAIN_TRIAGE_PROFILES

_backlog = _import_v06_module("backlog_dynamics")
BacklogDynamicsModel = _backlog.BacklogDynamicsModel
BACKLOG_BENCHMARKS = _backlog.BACKLOG_BENCHMARKS

# Import case studies
from case_study_integration import CASE_STUDY_BENCHMARKS

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'v05': '#2ecc71',    # Green
    'v06': '#3498db',    # Blue
    'triage': '#e74c3c',  # Red
    'physical': '#f39c12', # Orange
    'cognitive': '#9b59b6', # Purple
}


def fig1_v05_v06_comparison():
    """Compare v0.5 and v0.6 predictions across domains."""
    print("Generating Figure 1: v0.5 vs v0.6 Comparison...")

    domains = ["structural_biology", "materials_science", "protein_design",
               "drug_discovery", "clinical_genomics"]
    domain_labels = ["Structural\nBiology", "Materials\nScience", "Protein\nDesign",
                     "Drug\nDiscovery", "Clinical\nGenomics"]

    years = [2030, 2050]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, year in enumerate(years):
        ax = axes[idx]
        v05_vals = []
        v06_vals = []

        for domain in domains:
            try:
                model = IntegratedV06Model(domain=domain)
                f = model.forecast([year])[year]
                v05_vals.append(f.v05_end_to_end)
                v06_vals.append(f.effective_acceleration)
            except Exception as e:
                print(f"  Warning: {domain} - {e}")
                v05_vals.append(0)
                v06_vals.append(0)

        x = np.arange(len(domains))
        width = 0.35

        bars1 = ax.bar(x - width/2, v05_vals, width, label='v0.5 (AI + Automation)',
                       color=COLORS['v05'], alpha=0.8)
        bars2 = ax.bar(x + width/2, v06_vals, width, label='v0.6 (+ Triage)',
                       color=COLORS['v06'], alpha=0.8)

        ax.set_ylabel('Acceleration Factor (×)', fontsize=12)
        ax.set_title(f'{year} Projections', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(domain_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(v05_vals), max(v06_vals)) * 1.2)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}×',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}×',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('v0.5 vs v0.6 Model Comparison: Impact of Triage Constraints',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "fig1_v05_v06_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_v05_v06_comparison.png")


def fig2_triage_impact():
    """Show the impact of triage constraints by domain."""
    print("Generating Figure 2: Triage Impact Analysis...")

    domains = ["structural_biology", "materials_science", "protein_design",
               "drug_discovery", "clinical_genomics"]

    fig, ax = plt.subplots(figsize=(12, 6))

    results = []
    for domain in domains:
        try:
            model = IntegratedV06Model(domain=domain)
            f = model.forecast([2030])[2030]
            impact = ((f.v05_end_to_end - f.effective_acceleration) /
                      f.v05_end_to_end * 100)
            results.append({
                'domain': domain,
                'v05': f.v05_end_to_end,
                'v06': f.effective_acceleration,
                'triage_limited': f.triage_limited_acceleration,
                'impact': impact
            })
        except Exception as e:
            print(f"  Warning: {domain} - {e}")

    domains_sorted = sorted(results, key=lambda x: x['impact'], reverse=True)

    x = np.arange(len(domains_sorted))
    width = 0.25

    ax.bar(x - width, [r['v05'] for r in domains_sorted], width,
           label='v0.5 Potential', color=COLORS['v05'], alpha=0.8)
    ax.bar(x, [r['v06'] for r in domains_sorted], width,
           label='v0.6 Effective', color=COLORS['v06'], alpha=0.8)
    ax.bar(x + width, [min(r['triage_limited'], 50) for r in domains_sorted], width,
           label='Triage Limited', color=COLORS['triage'], alpha=0.8)

    ax.set_ylabel('Acceleration Factor (×)', fontsize=12)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_title('Triage Constraint Impact by Domain (2030)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['domain'].replace('_', '\n') for r in domains_sorted], fontsize=10)
    ax.legend(fontsize=10, loc='upper right')

    # Add impact annotations
    for i, r in enumerate(domains_sorted):
        impact_text = f"-{r['impact']:.0f}%" if r['impact'] > 0 else "No impact"
        ax.annotate(impact_text, xy=(i, r['v05'] + 1),
                    ha='center', fontsize=9, color='red' if r['impact'] > 10 else 'gray')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "fig2_triage_impact.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_triage_impact.png")


def fig3_backlog_evolution():
    """Show backlog evolution over time for key domains."""
    print("Generating Figure 3: Backlog Evolution...")

    domains = ["materials_science", "protein_design", "structural_biology"]
    domain_labels = ["Materials Science", "Protein Design", "Structural Biology"]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    years = list(range(2024, 2051))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Backlog accumulation
    ax = axes[0]
    for idx, domain in enumerate(domains):
        try:
            model = BacklogDynamicsModel(domain=domain)
            results = model.simulate(years)
            backlogs = [results[y].current_backlog for y in years]
            ax.plot(years, backlogs, label=domain_labels[idx],
                    color=colors[idx], linewidth=2)
        except Exception as e:
            print(f"  Warning: {domain} - {e}")

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Backlog Size (Hypotheses)', fontsize=12)
    ax.set_title('Hypothesis Backlog Accumulation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')

    # Right: Years to clear
    ax = axes[1]
    for idx, domain in enumerate(domains):
        try:
            model = BacklogDynamicsModel(domain=domain)
            results = model.simulate(years)
            years_to_clear = []
            for y in years:
                ytc = results[y].backlog_years
                years_to_clear.append(min(ytc, 1000))  # Cap at 1000 for visualization
            ax.plot(years, years_to_clear, label=domain_labels[idx],
                    color=colors[idx], linewidth=2)
        except Exception as e:
            print(f"  Warning: {domain} - {e}")

    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Years to Clear Backlog', fontsize=12)
    ax.set_title('Backlog Clearance Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')

    plt.suptitle('v0.6 Backlog Dynamics: The Triage Bottleneck',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "fig3_backlog_evolution.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_backlog_evolution.png")


def fig4_validation_summary():
    """Show validation results against 9 case studies."""
    print("Generating Figure 4: Validation Summary...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Observed vs Predicted
    ax = axes[0]
    case_names = list(CASE_STUDY_BENCHMARKS.keys())
    observed = [CASE_STUDY_BENCHMARKS[c].observed_acceleration for c in case_names]

    v06_predicted = []
    for case_name in case_names:
        case = CASE_STUDY_BENCHMARKS[case_name]
        domain = case.domain.lower().replace(" ", "_")
        domain_map = {"protein_engineering": "protein_design"}
        domain = domain_map.get(domain, domain)

        try:
            model = IntegratedV06Model(domain=domain)
            f = model.forecast([case.year])[case.year]
            v06_predicted.append(f.effective_acceleration)
        except:
            v06_predicted.append(0)

    # Scatter plot
    ax.scatter(observed, v06_predicted, s=100, alpha=0.7, color=COLORS['v06'])

    # Add labels
    for i, name in enumerate(case_names):
        ax.annotate(name[:15], (observed[i], v06_predicted[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Perfect prediction line
    max_val = max(max(observed), max(v06_predicted)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect prediction')

    ax.set_xlabel('Observed Acceleration (×)', fontsize=12)
    ax.set_ylabel('v0.6 Predicted (×)', fontsize=12)
    ax.set_title('v0.6 Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Right: Error comparison
    ax = axes[1]
    errors = []
    for i in range(len(case_names)):
        if observed[i] > 0 and v06_predicted[i] > 0:
            error = abs(np.log10(v06_predicted[i]) - np.log10(observed[i]))
            errors.append(error)
        else:
            errors.append(0)

    colors_bar = ['green' if e < 0.3 else 'orange' if e < 0.5 else 'red' for e in errors]
    bars = ax.barh(case_names, errors, color=colors_bar, alpha=0.7)

    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good (<0.3)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (<0.5)')

    ax.set_xlabel('Log Error', fontsize=12)
    ax.set_title('Prediction Error by Case Study', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.suptitle('v0.6 Model Validation Against 9 Case Studies',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "fig4_validation_summary.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_validation_summary.png")


def fig5_gnome_deep_dive():
    """Deep dive into the GNoME case study showing triage bottleneck."""
    print("Generating Figure 5: GNoME Deep Dive...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get GNoME benchmark data
    gnome = BACKLOG_BENCHMARKS["GNoME"]

    # Top-left: Generation vs Validation capacity
    ax = axes[0, 0]
    years = list(range(2020, 2051))
    gen_rates = [100_000 * (1.5 ** (y - 2020)) for y in years]  # 50% growth
    val_rates = [350 * (1.1 ** (y - 2020)) for y in years]  # 10% growth

    ax.fill_between(years, gen_rates, alpha=0.3, color=COLORS['cognitive'], label='Generation (AI)')
    ax.fill_between(years, val_rates, alpha=0.3, color=COLORS['physical'], label='Validation (Lab)')
    ax.plot(years, gen_rates, color=COLORS['cognitive'], linewidth=2)
    ax.plot(years, val_rates, color=COLORS['physical'], linewidth=2)
    ax.axvline(x=2023, color='red', linestyle='--', alpha=0.5, label='GNoME Release')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Rate (Hypotheses/Year)', fontsize=12)
    ax.set_title('The Growing Gap: Generation vs Validation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')

    # Top-right: Backlog accumulation
    ax = axes[0, 1]
    backlogs = []
    cumulative = 0
    for i, y in enumerate(years):
        new_hypotheses = gen_rates[i]
        validated = val_rates[i]
        cumulative = max(0, cumulative + new_hypotheses - validated)
        backlogs.append(cumulative)

    ax.fill_between(years, backlogs, alpha=0.3, color=COLORS['triage'])
    ax.plot(years, backlogs, color=COLORS['triage'], linewidth=2)
    ax.axhline(y=2_200_000, color='red', linestyle='--', alpha=0.5, label='GNoME output (2.2M)')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Backlog Size', fontsize=12)
    ax.set_title('Backlog Accumulation Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')

    # Bottom-left: Stage acceleration breakdown
    ax = axes[1, 0]
    stages = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stage_names = ['Lit Review', 'Hypothesis', 'Analysis', 'Wet Lab', 'Interpret', 'Validate']
    gnome_stages = [8, 100000, 50000, 1, 2, 1.5]

    colors_stage = [COLORS['cognitive']] * 3 + [COLORS['physical']] * 3
    ax.bar(stages, gnome_stages, color=colors_stage, alpha=0.8)

    for i, (stage, val) in enumerate(zip(stages, gnome_stages)):
        ax.annotate(f'{val}×', (i, val * 1.1), ha='center', fontsize=10)

    ax.set_ylabel('Acceleration Factor (×)', fontsize=12)
    ax.set_xlabel('Research Stage', fontsize=12)
    ax.set_title('GNoME Stage Acceleration (365× vs 1× end-to-end)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')

    # Add legend for cognitive vs physical
    cognitive_patch = mpatches.Patch(color=COLORS['cognitive'], alpha=0.8, label='Cognitive (AI)')
    physical_patch = mpatches.Patch(color=COLORS['physical'], alpha=0.8, label='Physical (Lab)')
    ax.legend(handles=[cognitive_patch, physical_patch], fontsize=10)

    # Bottom-right: Key metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
    GNoME: The Triage Bottleneck Case Study
    ═══════════════════════════════════════

    Key Metrics:
    • Materials Generated: 2,200,000
    • Synthesis Rate: ~350/year
    • Backlog Years: 6,286 years
    • Stage Acceleration: 365×
    • End-to-End: 1.0× (unchanged)

    The Problem:
    AI generated hypotheses 365× faster, but
    physical validation remained unchanged.
    Result: 6,000+ year backlog.

    v0.6 Insight:
    Without triage (intelligent selection),
    massive AI acceleration creates backlogs
    that eliminate practical speedup.

    Solution Pathways:
    1. Lab automation (v0.5): 10-50× validation
    2. AI triage: 10-100× better selection
    3. Simulation bypass: Skip physical validation
    """

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('GNoME Case Study: Why Stage Acceleration ≠ End-to-End Acceleration',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "fig5_gnome_deep_dive.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_gnome_deep_dive.png")


def generate_all_figures():
    """Generate all v0.6 figures."""
    # Create figures directory
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating v0.6 Visualizations")
    print("=" * 60)
    print()

    fig1_v05_v06_comparison()
    fig2_triage_impact()
    fig3_backlog_evolution()
    fig4_validation_summary()
    fig5_gnome_deep_dive()

    print()
    print("=" * 60)
    print("All v0.6 figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
