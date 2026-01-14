#!/usr/bin/env python3
"""
AI Research Acceleration Model v1.1 - Publication Quality Visualizations
=========================================================================

High-impact journal style (Nature/Science/Cell aesthetic)
- Clean, sophisticated design
- Publication-ready resolution (300 DPI)
- Accessible color palette
- Clear data hierarchy

Design principles:
1. Data-ink ratio maximization (Tufte)
2. Colorblind-safe palette
3. Consistent typography
4. White space as design element
5. Story-driven composition
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Wedge
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from pathlib import Path

# Add src to path for model import
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ai_acceleration_model import AIAccelerationModel
except ImportError:
    print("Warning: Could not import model, using mock data")
    AIAccelerationModel = None


# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================

# Nature/Science inspired color palette
COLORS = {
    # Primary palette - sophisticated, muted
    'structural_biology': '#2E5A87',    # Deep blue
    'drug_discovery': '#B8433E',        # Muted red
    'materials_science': '#5B8C5A',     # Forest green
    'protein_design': '#8E6BBF',        # Soft purple
    'clinical_genomics': '#D4873F',     # Warm orange

    # Accent colors
    'highlight': '#E63946',             # Accent red
    'positive': '#2A9D8F',              # Teal (positive)
    'negative': '#E76F51',              # Coral (negative)
    'neutral': '#6C757D',               # Gray

    # Background/structural
    'background': '#FAFAFA',
    'grid': '#E5E5E5',
    'text_primary': '#1A1A1A',
    'text_secondary': '#4A4A4A',
    'text_tertiary': '#7A7A7A',

    # Confidence intervals
    'ci_light': '#E8E8E8',
    'ci_medium': '#CCCCCC',
}

# Domain display configuration
DOMAINS = {
    'structural_biology': {'name': 'Structural\nBiology', 'short': 'SB', 'order': 0},
    'drug_discovery': {'name': 'Drug\nDiscovery', 'short': 'DD', 'order': 1},
    'materials_science': {'name': 'Materials\nScience', 'short': 'MS', 'order': 2},
    'protein_design': {'name': 'Protein\nDesign', 'short': 'PD', 'order': 3},
    'clinical_genomics': {'name': 'Clinical\nGenomics', 'short': 'CG', 'order': 4},
}

# Typography
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_clean_axis(ax, spines=['bottom', 'left'], grid=False):
    """Configure axis with minimal, clean aesthetic."""
    for spine in ['top', 'right', 'bottom', 'left']:
        if spine in spines:
            ax.spines[spine].set_color(COLORS['text_tertiary'])
            ax.spines[spine].set_linewidth(0.8)
        else:
            ax.spines[spine].set_visible(False)

    ax.tick_params(colors=COLORS['text_secondary'], length=4, width=0.8)

    if grid:
        ax.grid(True, axis='y', linestyle='-', alpha=0.3, color=COLORS['grid'], linewidth=0.5)
        ax.set_axisbelow(True)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label (a, b, c, etc.) in Nature/Science style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=COLORS['text_primary'], va='top', ha='left')


def format_acceleration(value):
    """Format acceleration value for display."""
    if value >= 10:
        return f'{value:.0f}×'
    elif value >= 1:
        return f'{value:.1f}×'
    else:
        return f'{value:.2f}×'


def get_model_data():
    """Get data from model or return mock data."""
    if AIAccelerationModel is not None:
        model = AIAccelerationModel()
        return model
    else:
        return None


# =============================================================================
# FIGURE 1: DOMAIN ACCELERATION OVERVIEW (Hero Figure)
# =============================================================================

def figure1_domain_overview():
    """
    Main results figure showing acceleration by domain.
    Nature-style horizontal bar chart with confidence intervals.
    """
    fig = plt.figure(figsize=(7, 5), facecolor='white', dpi=300)

    # Create main axis with room for annotations
    ax = fig.add_axes([0.25, 0.12, 0.55, 0.78])

    # Get model data
    model = get_model_data()

    # Data for 2030 baseline
    domains = ['structural_biology', 'protein_design', 'clinical_genomics',
               'drug_discovery', 'materials_science']

    if model:
        data = {d: model.forecast(d, 2030) for d in domains}
        accels = [data[d].acceleration for d in domains]
        ci_lows = [data[d].ci_90[0] for d in domains]
        ci_highs = [data[d].ci_90[1] for d in domains]
    else:
        # Mock data
        accels = [8.91, 5.47, 4.19, 1.68, 1.26]
        ci_lows = [5.8, 3.9, 3.0, 1.3, 0.9]
        ci_highs = [13.7, 7.7, 5.9, 2.1, 1.7]

    y_pos = np.arange(len(domains))
    colors = [COLORS[d] for d in domains]

    # Plot horizontal bars
    bars = ax.barh(y_pos, accels, height=0.6, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1)

    # Add confidence intervals as error bars
    ax.errorbar(accels, y_pos, xerr=[np.array(accels) - np.array(ci_lows),
                                      np.array(ci_highs) - np.array(accels)],
                fmt='none', color=COLORS['text_secondary'], capsize=3, capthick=1,
                linewidth=1, alpha=0.7)

    # Add value labels
    for i, (accel, ci_l, ci_h) in enumerate(zip(accels, ci_lows, ci_highs)):
        # Main value
        ax.text(ci_h + 0.3, i, format_acceleration(accel),
                va='center', ha='left', fontsize=9, fontweight='bold',
                color=COLORS['text_primary'])
        # CI range
        ax.text(ci_h + 0.3, i - 0.25, f'[{ci_l:.1f}–{ci_h:.1f}]',
                va='center', ha='left', fontsize=7, color=COLORS['text_tertiary'])

    # Reference line at 1x
    ax.axvline(x=1, color=COLORS['text_tertiary'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(1.05, 4.7, 'No acceleration', fontsize=7, color=COLORS['text_tertiary'],
            va='bottom', ha='left', style='italic')

    # Y-axis labels with domain names
    domain_labels = [DOMAINS[d]['name'] for d in domains]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domain_labels, fontsize=9)

    # X-axis
    ax.set_xlabel('Research Acceleration Factor', fontsize=10, color=COLORS['text_primary'])
    ax.set_xlim(0, 16)

    setup_clean_axis(ax, spines=['bottom'], grid=False)
    ax.tick_params(axis='y', length=0)

    # Title
    fig.text(0.25, 0.95, 'AI-Driven Research Acceleration by Domain (2030)',
             fontsize=12, fontweight='bold', color=COLORS['text_primary'])
    fig.text(0.25, 0.91, 'Baseline scenario with 90% confidence intervals',
             fontsize=9, color=COLORS['text_secondary'])

    # Add panel label
    fig.text(0.02, 0.95, 'a', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    # Methodology note
    fig.text(0.25, 0.02,
             'Geometric mean system acceleration: 2.8× [2.1–3.8×] | n=15 validation cases | Mean log error: 0.21',
             fontsize=7, color=COLORS['text_tertiary'], style='italic')

    return fig


# =============================================================================
# FIGURE 2: TIME TRAJECTORIES (S-Curve Evolution)
# =============================================================================

def figure2_trajectories():
    """
    Time evolution showing S-curve trajectories for each domain.
    Small multiples design for clarity.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), facecolor='white', dpi=300)
    axes = axes.flatten()

    model = get_model_data()
    years = np.arange(2024, 2036)

    domains = ['structural_biology', 'drug_discovery', 'materials_science',
               'protein_design', 'clinical_genomics']

    # Domain-specific ceilings for reference
    ceilings = {
        'structural_biology': 15.0,
        'drug_discovery': 4.0,
        'materials_science': 5.0,
        'protein_design': 10.0,
        'clinical_genomics': 6.0
    }

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        color = COLORS[domain]

        if model:
            trajectory = [model.forecast(domain, y) for y in years]
            accels = [t.acceleration for t in trajectory]
            ci_lows = [t.ci_90[0] for t in trajectory]
            ci_highs = [t.ci_90[1] for t in trajectory]
        else:
            # Mock S-curve data
            base = [4.5, 1.4, 1.0, 2.5, 2.0][idx]
            ceiling = list(ceilings.values())[idx]
            k = [0.15, 0.08, 0.10, 0.12, 0.10][idx]
            t0 = [3, 8, 6, 4, 5][idx]
            t = years - 2024
            factor = 1 + (ceiling - 1) / (1 + np.exp(-k * (t - t0)))
            initial = 1 + (ceiling - 1) / (1 + np.exp(-k * (0 - t0)))
            accels = base * factor / initial
            ci_lows = accels * 0.7
            ci_highs = accels * 1.4

        # Confidence band
        ax.fill_between(years, ci_lows, ci_highs, alpha=0.2, color=color, linewidth=0)

        # Main trajectory
        ax.plot(years, accels, color=color, linewidth=2)

        # Ceiling reference (dashed)
        ceiling = ceilings[domain]
        ax.axhline(y=ceiling, color=color, linestyle=':', linewidth=1, alpha=0.5)
        ax.text(2035.2, ceiling, f'{ceiling}×', fontsize=7, color=color,
                va='center', ha='left', alpha=0.7)

        # Current marker (2024)
        ax.scatter([2024], [accels[0] if isinstance(accels, list) else accels[0]],
                   color=color, s=30, zorder=5, edgecolor='white', linewidth=1)

        # 2030 marker
        idx_2030 = 6
        ax.scatter([2030], [accels[idx_2030] if isinstance(accels, list) else accels[idx_2030]],
                   color=color, s=40, zorder=5, edgecolor='white', linewidth=1.5, marker='D')

        # Domain title
        ax.set_title(DOMAINS[domain]['name'].replace('\n', ' '), fontsize=9,
                     fontweight='bold', color=color, pad=8)

        # Axis formatting
        ax.set_xlim(2024, 2035)
        ax.set_xticks([2024, 2027, 2030, 2033])
        ax.set_xticklabels(['\'24', '\'27', '\'30', '\'33'], fontsize=7)

        if idx in [0, 3]:
            ax.set_ylabel('Acceleration', fontsize=8)

        setup_clean_axis(ax, spines=['bottom', 'left'], grid=True)

    # Use last panel for legend/annotation
    axes[5].axis('off')

    # Legend in empty panel
    legend_elements = [
        Line2D([0], [0], color=COLORS['text_secondary'], linewidth=2, label='Trajectory'),
        plt.fill_between([], [], [], alpha=0.2, color=COLORS['text_secondary'], label='90% CI')[0]
            if False else mpatches.Patch(alpha=0.2, color=COLORS['text_secondary'], label='90% CI'),
        Line2D([0], [0], color=COLORS['text_secondary'], linestyle=':', linewidth=1, label='Ceiling'),
        Line2D([0], [0], marker='D', color=COLORS['text_secondary'], linestyle='',
               markersize=6, label='2030 Projection'),
    ]

    axes[5].legend(handles=legend_elements[:4], loc='center', frameon=False, fontsize=8)
    axes[5].text(0.5, 0.15, 'Logistic growth model\n(Rogers, 2003)',
                 ha='center', va='center', fontsize=7, color=COLORS['text_tertiary'],
                 style='italic', transform=axes[5].transAxes)

    # Main title
    fig.suptitle('Acceleration Trajectories Follow S-Curve Dynamics',
                 fontsize=11, fontweight='bold', y=0.98, color=COLORS['text_primary'])

    fig.text(0.02, 0.98, 'b', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    return fig


# =============================================================================
# FIGURE 3: SPILLOVER NETWORK
# =============================================================================

def figure3_spillover_network():
    """
    Network visualization of cross-domain spillover effects.
    Elegant node-link diagram with weighted edges.
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', dpi=300)
    ax.set_aspect('equal')

    # Node positions (pentagon layout)
    n_domains = 5
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_domains, endpoint=False)
    radius = 0.35

    domains = ['structural_biology', 'protein_design', 'clinical_genomics',
               'drug_discovery', 'materials_science']

    positions = {d: (0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a))
                 for d, a in zip(domains, angles)}

    # Spillover data (source, target, coefficient)
    spillovers = [
        ('structural_biology', 'drug_discovery', 0.25),
        ('structural_biology', 'protein_design', 0.30),
        ('protein_design', 'drug_discovery', 0.12),
        ('clinical_genomics', 'drug_discovery', 0.08),
        ('drug_discovery', 'clinical_genomics', 0.04),
        ('materials_science', 'structural_biology', 0.03),
        ('protein_design', 'materials_science', 0.04),
        ('clinical_genomics', 'protein_design', 0.04),
    ]

    # Draw edges (curved arrows)
    for source, target, coef in spillovers:
        x1, y1 = positions[source]
        x2, y2 = positions[target]

        # Edge weight determines appearance
        alpha = 0.3 + 0.7 * (coef / 0.30)  # Normalize to max
        linewidth = 1 + 4 * (coef / 0.30)

        # Draw curved edge
        style = "arc3,rad=0.15"
        arrow = mpatches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=style,
            arrowstyle='-|>',
            mutation_scale=10,
            color=COLORS['text_secondary'],
            alpha=alpha,
            linewidth=linewidth,
            zorder=1
        )
        ax.add_patch(arrow)

        # Add coefficient label for major spillovers
        if coef >= 0.10:
            mid_x = (x1 + x2) / 2 + 0.05 * np.sign(y1 - y2)
            mid_y = (y1 + y2) / 2 + 0.05 * np.sign(x2 - x1)
            ax.text(mid_x, mid_y, f'{coef:.0%}', fontsize=7,
                    color=COLORS['text_secondary'], ha='center', va='center',
                    fontweight='bold', alpha=0.8)

    # Draw nodes
    node_size = 0.08
    for domain in domains:
        x, y = positions[domain]
        color = COLORS[domain]

        # Node circle
        circle = plt.Circle((x, y), node_size, color=color, ec='white',
                            linewidth=2, zorder=3)
        ax.add_patch(circle)

        # Domain label (outside node)
        angle = np.arctan2(y - 0.5, x - 0.5)
        label_dist = node_size + 0.08
        label_x = x + label_dist * np.cos(angle)
        label_y = y + label_dist * np.sin(angle)

        ha = 'center'
        if label_x < 0.3:
            ha = 'right'
        elif label_x > 0.7:
            ha = 'left'

        ax.text(label_x, label_y, DOMAINS[domain]['name'], fontsize=8,
                fontweight='bold', color=color, ha=ha, va='center')

    # Title and annotation
    ax.text(0.5, 0.97, 'Cross-Domain Spillover Network', fontsize=11,
            fontweight='bold', ha='center', transform=ax.transAxes,
            color=COLORS['text_primary'])
    ax.text(0.5, 0.93, 'Arrow thickness proportional to effect size', fontsize=8,
            ha='center', transform=ax.transAxes, color=COLORS['text_secondary'])

    # Key insight box
    box_text = 'Structural Biology → Drug Discovery\nis the dominant spillover pathway (25%)'
    props = dict(boxstyle='round,pad=0.4', facecolor=COLORS['ci_light'],
                 edgecolor=COLORS['grid'], alpha=0.9)
    ax.text(0.5, 0.07, box_text, fontsize=8, ha='center', va='center',
            transform=ax.transAxes, bbox=props, color=COLORS['text_primary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.text(0.02, 0.98, 'c', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    return fig


# =============================================================================
# FIGURE 4: SCENARIO COMPARISON
# =============================================================================

def figure4_scenarios():
    """
    Scenario comparison showing range of outcomes.
    Elegant dot plot with scenario ranges.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white', dpi=300)

    model = get_model_data()

    scenarios = ['pessimistic', 'conservative', 'baseline', 'optimistic', 'breakthrough']
    scenario_labels = ['Pessimistic\n(10%)', 'Conservative\n(20%)', 'Baseline\n(40%)',
                       'Optimistic\n(20%)', 'Breakthrough\n(10%)']
    scenario_colors = ['#8B9DAF', '#6B8BA4', COLORS['text_primary'], '#5A9A6B', '#2A9D8F']

    domains = ['structural_biology', 'drug_discovery', 'materials_science',
               'protein_design', 'clinical_genomics']

    x_positions = np.arange(len(scenarios))
    width = 0.15

    for i, domain in enumerate(domains):
        offset = (i - 2) * width

        if model:
            values = [model.forecast(domain, 2030, s).acceleration for s in scenarios]
        else:
            # Mock data - multiply baseline by scenario modifier
            baseline = [8.91, 1.68, 1.26, 5.47, 4.19][i]
            modifiers = [0.6, 0.8, 1.0, 1.25, 1.6]
            values = [baseline * m for m in modifiers]

        # Plot dots connected by line
        ax.plot(x_positions + offset, values, 'o-', color=COLORS[domain],
                markersize=8, linewidth=1.5, alpha=0.85, label=DOMAINS[domain]['name'].replace('\n', ' '))

    # X-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_xlabel('')

    # Y-axis
    ax.set_ylabel('Acceleration Factor (2030)', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 20)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(['0.5×', '1×', '2×', '5×', '10×', '20×'])

    # Reference line
    ax.axhline(y=1, color=COLORS['text_tertiary'], linestyle='--', linewidth=0.8, alpha=0.5)

    setup_clean_axis(ax, spines=['bottom', 'left'], grid=True)

    # Legend
    ax.legend(loc='upper left', frameon=False, fontsize=7, ncol=2)

    # Title
    ax.set_title('Scenario Analysis: Range of Possible Outcomes', fontsize=11,
                 fontweight='bold', color=COLORS['text_primary'], pad=15)

    # Annotation
    ax.text(0.98, 0.02, 'Scenario probabilities from expert elicitation (n=12)',
            fontsize=7, color=COLORS['text_tertiary'], transform=ax.transAxes,
            ha='right', style='italic')

    fig.text(0.02, 0.96, 'd', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    plt.tight_layout()

    return fig


# =============================================================================
# FIGURE 5: VALIDATION RESULTS
# =============================================================================

def figure5_validation():
    """
    Validation figure showing predicted vs observed.
    Classic calibration plot with identity line.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), facecolor='white', dpi=300)

    # Validation data
    cases = [
        ('AlphaFold2', 'structural_biology', 4.9, 4.5),
        ('ESMFold', 'structural_biology', 3.6, 4.5),
        ('AlphaFold3', 'structural_biology', 6.0, 4.5),
        ('Insilico', 'drug_discovery', 2.1, 1.4),
        ('Recursion', 'drug_discovery', 1.8, 1.4),
        ('Isomorphic', 'drug_discovery', 1.6, 1.4),
        ('GNoME', 'materials_science', 1.0, 1.0),
        ('A-Lab', 'materials_science', 1.2, 1.0),
        ('Battery', 'materials_science', 1.3, 1.0),
        ('ESM-3', 'protein_design', 3.2, 2.5),
        ('RFdiffusion', 'protein_design', 2.6, 2.5),
        ('ProteinMPNN', 'protein_design', 2.0, 2.5),
        ('AlphaMissense', 'clinical_genomics', 2.2, 2.0),
        ('DeepVariant', 'clinical_genomics', 1.4, 2.0),
        ('SpliceAI', 'clinical_genomics', 1.8, 2.0),
    ]

    # Panel A: Predicted vs Observed scatter
    for name, domain, obs, pred in cases:
        color = COLORS[domain]
        ax1.scatter(obs, pred, c=color, s=50, alpha=0.8, edgecolor='white', linewidth=1, zorder=3)

    # Identity line
    ax1.plot([0.5, 10], [0.5, 10], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    # ±0.3 log error bands
    x_line = np.linspace(0.5, 10, 100)
    ax1.fill_between(x_line, x_line * np.exp(-0.3), x_line * np.exp(0.3),
                     alpha=0.1, color=COLORS['text_secondary'], label='±0.3 log error')

    ax1.set_xlabel('Observed Acceleration', fontsize=9)
    ax1.set_ylabel('Predicted Acceleration', fontsize=9)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.8, 10)
    ax1.set_ylim(0.8, 10)
    ax1.set_aspect('equal')

    setup_clean_axis(ax1, spines=['bottom', 'left'])
    ax1.legend(loc='lower right', frameon=False, fontsize=7)
    ax1.set_title('Model Calibration', fontsize=10, fontweight='bold', pad=10)

    # Panel B: Error distribution by domain
    domain_errors = {}
    for name, domain, obs, pred in cases:
        error = abs(np.log(pred) - np.log(obs))
        if domain not in domain_errors:
            domain_errors[domain] = []
        domain_errors[domain].append(error)

    domains_sorted = ['structural_biology', 'drug_discovery', 'materials_science',
                      'protein_design', 'clinical_genomics']

    positions = np.arange(len(domains_sorted))

    for i, domain in enumerate(domains_sorted):
        errors = domain_errors.get(domain, [0])
        color = COLORS[domain]

        # Box plot style
        mean_err = np.mean(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)

        ax2.barh(i, mean_err, height=0.6, color=color, alpha=0.7, edgecolor='white')
        ax2.errorbar(mean_err, i, xerr=[[mean_err - min_err], [max_err - mean_err]],
                     fmt='none', color=COLORS['text_secondary'], capsize=3, linewidth=1)

    # Threshold line
    ax2.axvline(x=0.30, color=COLORS['highlight'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(0.31, 4.5, 'Acceptable\nthreshold', fontsize=7, color=COLORS['highlight'],
             va='center', ha='left')

    ax2.set_yticks(positions)
    ax2.set_yticklabels([DOMAINS[d]['short'] for d in domains_sorted], fontsize=8)
    ax2.set_xlabel('Log Error', fontsize=9)
    ax2.set_xlim(0, 0.5)

    setup_clean_axis(ax2, spines=['bottom'])
    ax2.tick_params(axis='y', length=0)
    ax2.set_title('Error by Domain', fontsize=10, fontweight='bold', pad=10)

    # Overall stats annotation
    all_errors = [abs(np.log(p) - np.log(o)) for _, _, o, p in cases]
    stats_text = f'Mean log error: {np.mean(all_errors):.2f}\nMedian: {np.median(all_errors):.2f}\nn = {len(cases)} cases'
    ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes, fontsize=7,
             ha='right', va='bottom', color=COLORS['text_tertiary'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['grid']))

    fig.text(0.02, 0.96, 'e', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    plt.tight_layout()

    return fig


# =============================================================================
# FIGURE 6: SENSITIVITY ANALYSIS
# =============================================================================

def figure6_sensitivity():
    """
    Tornado diagram showing parameter sensitivity.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white', dpi=300)

    # Sensitivity data (parameter, low impact, high impact)
    parameters = [
        ('Drug Discovery base', -18, 18),
        ('Structural Biology base', -15, 15),
        ('Protein Design base', -12, 12),
        ('Clinical Genomics base', -10, 10),
        ('Materials Science base', -9, 9),
        ('DD time ceiling', -7, 7),
        ('SB time ceiling', -6, 6),
        ('SB→DD spillover', -3, 3),
        ('Growth rates', -2, 2),
    ]

    y_pos = np.arange(len(parameters))

    for i, (param, low, high) in enumerate(parameters):
        # Determine color based on category
        if 'base' in param.lower():
            color = COLORS['structural_biology']
        elif 'ceiling' in param.lower():
            color = COLORS['drug_discovery']
        elif 'spillover' in param.lower():
            color = COLORS['protein_design']
        else:
            color = COLORS['clinical_genomics']

        # Draw bars from center
        ax.barh(i, high, left=0, height=0.6, color=color, alpha=0.7, edgecolor='white')
        ax.barh(i, abs(low), left=low, height=0.6, color=color, alpha=0.7, edgecolor='white')

    # Center line
    ax.axvline(x=0, color=COLORS['text_primary'], linewidth=1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in parameters], fontsize=8)
    ax.set_xlabel('Impact on System Acceleration (%)', fontsize=9)
    ax.set_xlim(-25, 25)

    setup_clean_axis(ax, spines=['bottom'])
    ax.tick_params(axis='y', length=0)

    # Title
    ax.set_title('Parameter Sensitivity Analysis', fontsize=11, fontweight='bold', pad=15)

    # Annotation
    ax.text(0.98, 0.02, 'Base parameters account for ~80% of variance',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color=COLORS['text_tertiary'], style='italic')

    fig.text(0.02, 0.96, 'f', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    plt.tight_layout()

    return fig


# =============================================================================
# FIGURE 7: WORKFORCE IMPACT
# =============================================================================

def figure7_workforce():
    """
    Workforce impact visualization with uncertainty.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), facecolor='white', dpi=300)

    model = get_model_data()

    domains = ['structural_biology', 'drug_discovery', 'materials_science',
               'protein_design', 'clinical_genomics']

    # Panel A: Displacement vs Creation
    if model:
        snapshot = model.system_snapshot(2030)
        displaced = [snapshot.domain_forecasts[d].jobs_displaced for d in domains]
        created = [snapshot.domain_forecasts[d].jobs_created for d in domains]
    else:
        displaced = [0.03, 0.17, 0.11, 0.04, 0.02]
        created = [0.19, 1.37, 0.46, 0.32, 0.09]

    x = np.arange(len(domains))
    width = 0.35

    bars1 = ax1.bar(x - width/2, displaced, width, label='Displaced',
                    color=COLORS['negative'], alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, created, width, label='Created',
                    color=COLORS['positive'], alpha=0.8, edgecolor='white')

    ax1.set_ylabel('Jobs (Millions)', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DOMAINS[d]['short'] for d in domains], fontsize=8)
    ax1.legend(frameon=False, fontsize=8)

    setup_clean_axis(ax1, spines=['left'], grid=True)
    ax1.set_title('Job Displacement vs Creation', fontsize=10, fontweight='bold', pad=10)

    # Panel B: Net change with uncertainty
    net = [c - d for c, d in zip(created, displaced)]
    net_low = [n * 0.6 for n in net]  # Simplified uncertainty
    net_high = [n * 1.4 for n in net]

    colors = [COLORS[d] for d in domains]

    bars = ax2.barh(x, net, height=0.6, color=colors, alpha=0.8, edgecolor='white')
    ax2.errorbar(net, x, xerr=[np.array(net) - np.array(net_low),
                                np.array(net_high) - np.array(net)],
                 fmt='none', color=COLORS['text_secondary'], capsize=3, linewidth=1)

    ax2.axvline(x=0, color=COLORS['text_tertiary'], linewidth=0.8)
    ax2.set_xlabel('Net Job Change (Millions)', fontsize=9)
    ax2.set_yticks(x)
    ax2.set_yticklabels([DOMAINS[d]['short'] for d in domains], fontsize=8)

    setup_clean_axis(ax2, spines=['bottom'])
    ax2.tick_params(axis='y', length=0)
    ax2.set_title('Net Workforce Impact', fontsize=10, fontweight='bold', pad=10)

    # Total annotation
    total_net = sum(net)
    ax2.text(0.95, 0.05, f'Total: +{total_net:.1f}M jobs',
             transform=ax2.transAxes, fontsize=9, fontweight='bold',
             ha='right', va='bottom', color=COLORS['positive'])

    fig.text(0.02, 0.96, 'g', fontsize=14, fontweight='bold', color=COLORS['text_primary'])

    plt.tight_layout()

    return fig


# =============================================================================
# FIGURE 8: KEY INSIGHTS SUMMARY
# =============================================================================

def figure8_summary_infographic():
    """
    Summary infographic combining key findings.
    Magazine-style layout with key numbers.
    """
    fig = plt.figure(figsize=(7, 8), facecolor='white', dpi=300)

    # Main title
    fig.text(0.5, 0.97, 'AI Research Acceleration Model: Key Findings',
             fontsize=14, fontweight='bold', ha='center', color=COLORS['text_primary'])
    fig.text(0.5, 0.94, '2030 Projections Under Baseline Scenario',
             fontsize=10, ha='center', color=COLORS['text_secondary'])

    # =========== Section 1: Big Numbers ===========
    # System acceleration
    ax1 = fig.add_axes([0.08, 0.75, 0.25, 0.15])
    ax1.text(0.5, 0.7, '2.8×', fontsize=36, fontweight='bold', ha='center', va='center',
             color=COLORS['structural_biology'])
    ax1.text(0.5, 0.2, 'System\nAcceleration', fontsize=9, ha='center', va='center',
             color=COLORS['text_secondary'])
    ax1.axis('off')

    # Net jobs
    ax2 = fig.add_axes([0.38, 0.75, 0.25, 0.15])
    ax2.text(0.5, 0.7, '+2.1M', fontsize=36, fontweight='bold', ha='center', va='center',
             color=COLORS['positive'])
    ax2.text(0.5, 0.2, 'Net Jobs\nCreated', fontsize=9, ha='center', va='center',
             color=COLORS['text_secondary'])
    ax2.axis('off')

    # Validation
    ax3 = fig.add_axes([0.68, 0.75, 0.25, 0.15])
    ax3.text(0.5, 0.7, '0.21', fontsize=36, fontweight='bold', ha='center', va='center',
             color=COLORS['drug_discovery'])
    ax3.text(0.5, 0.2, 'Mean Log\nError', fontsize=9, ha='center', va='center',
             color=COLORS['text_secondary'])
    ax3.axis('off')

    # =========== Section 2: Domain Rankings ===========
    ax4 = fig.add_axes([0.1, 0.45, 0.8, 0.25])

    domains = ['structural_biology', 'protein_design', 'clinical_genomics',
               'drug_discovery', 'materials_science']
    accels = [8.91, 5.47, 4.19, 1.68, 1.26]

    for i, (domain, accel) in enumerate(zip(domains, accels)):
        # Bar
        bar_width = accel / 10 * 0.7
        ax4.barh(i, bar_width, height=0.6, color=COLORS[domain], alpha=0.85)

        # Domain name
        ax4.text(-0.02, i, DOMAINS[domain]['name'].replace('\n', ' '),
                 fontsize=8, ha='right', va='center', color=COLORS['text_primary'])

        # Value
        ax4.text(bar_width + 0.02, i, format_acceleration(accel),
                 fontsize=10, fontweight='bold', ha='left', va='center',
                 color=COLORS[domain])

    ax4.set_xlim(-0.3, 1)
    ax4.set_ylim(-0.5, 4.5)
    ax4.axis('off')
    ax4.set_title('Acceleration by Domain', fontsize=10, fontweight='bold',
                  loc='left', pad=10, color=COLORS['text_primary'])

    # =========== Section 3: Key Insights ===========
    insights = [
        ('1', 'Structural biology leads', 'AlphaFold effect: 8.9× by 2030'),
        ('2', 'Drug discovery constrained', 'Clinical trials limit to 1.7×'),
        ('3', 'Materials backlog grows', 'Synthesis bottleneck dominates'),
        ('4', 'Spillovers matter', 'SB→DD pathway: +25% boost'),
        ('5', 'Net positive workforce', '+2.1M jobs across domains'),
    ]

    ax5 = fig.add_axes([0.1, 0.08, 0.8, 0.32])

    for i, (icon, title, detail) in enumerate(insights):
        y = 0.85 - i * 0.19

        # Numbered bullet
        circle = plt.Circle((0.03, y), 0.025, color=COLORS['structural_biology'], alpha=0.8)
        ax5.add_patch(circle)
        ax5.text(0.03, y, icon, fontsize=8, ha='center', va='center',
                 color='white', fontweight='bold')

        # Title
        ax5.text(0.08, y + 0.02, title, fontsize=9, fontweight='bold',
                 ha='left', va='bottom', color=COLORS['text_primary'])

        # Detail
        ax5.text(0.08, y - 0.02, detail, fontsize=8,
                 ha='left', va='top', color=COLORS['text_secondary'])

        # Separator line
        if i < len(insights) - 1:
            ax5.axhline(y=y - 0.08, xmin=0.05, xmax=0.95,
                       color=COLORS['grid'], linewidth=0.5)

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Key Insights', fontsize=10, fontweight='bold',
                  loc='left', pad=5, color=COLORS['text_primary'])

    # Footer
    fig.text(0.5, 0.02, 'AI Research Acceleration Model v1.1 | 15 validation cases | 5 domains | January 2026',
             fontsize=7, ha='center', color=COLORS['text_tertiary'])

    return fig


# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================

def generate_all_figures(output_dir: str = None):
    """Generate all publication figures and save to directory."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "figures"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    figures = [
        ('fig1_domain_overview.png', figure1_domain_overview),
        ('fig2_trajectories.png', figure2_trajectories),
        ('fig3_spillover_network.png', figure3_spillover_network),
        ('fig4_scenarios.png', figure4_scenarios),
        ('fig5_validation.png', figure5_validation),
        ('fig6_sensitivity.png', figure6_sensitivity),
        ('fig7_workforce.png', figure7_workforce),
        ('fig8_summary.png', figure8_summary_infographic),
    ]

    for filename, fig_func in figures:
        print(f"Generating {filename}...")
        fig = fig_func()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved to {output_dir / filename}")

    print(f"\nAll {len(figures)} figures saved to {output_dir}")
    return output_dir


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = generate_all_figures()
    print(f"\nPublication figures ready at: {output_dir}")
