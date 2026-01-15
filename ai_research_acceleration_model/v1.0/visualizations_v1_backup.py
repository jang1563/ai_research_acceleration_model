#!/usr/bin/env python3
"""
v1.0 Visualizations v2: Expert-Level Visual Storytelling
=========================================================

Improved visualizations with:
1. Clear narrative arc across figures
2. Visual hierarchy emphasizing key insights
3. Consistent color palette with meaning
4. Strategic use of white space
5. Data-ink ratio optimization
6. Actionable takeaways prominently featured

Design Principles Applied:
- "Less is more" - Remove chart junk
- Color encodes meaning consistently
- Headlines tell the story, charts prove it
- Progressive disclosure of complexity
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_acceleration_model import AIAccelerationModel

# =============================================================================
# DESIGN SYSTEM - Consistent across all figures
# =============================================================================

# Color palette with semantic meaning
COLORS = {
    # Domain colors - ordered by typical acceleration (high to low)
    'structural_biology': '#1a9850',   # Green - fastest, most mature
    'protein_design': '#91cf60',       # Light green - fast, emerging
    'clinical_genomics': '#fee08b',    # Yellow - moderate
    'drug_discovery': '#fc8d59',       # Orange - bounded by trials
    'materials_science': '#d73027',    # Red - slowest, synthesis limited

    # Semantic colors
    'positive': '#2e7d32',             # Dark green for positive outcomes
    'negative': '#c62828',             # Dark red for negative/displaced
    'neutral': '#757575',              # Gray for reference lines
    'highlight': '#1565c0',            # Blue for emphasis
    'background': '#fafafa',           # Off-white background

    # Scenario colors
    'pessimistic': '#b71c1c',
    'conservative': '#e65100',
    'baseline': '#f9a825',
    'optimistic': '#558b2f',
    'breakthrough': '#1565c0',
}

# Typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = COLORS['background']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def add_headline(fig, headline, subheadline=None, y_pos=0.95):
    """Add newspaper-style headline that tells the story."""
    fig.text(0.5, y_pos, headline, ha='center', fontsize=18, fontweight='bold',
             color='#212121')
    if subheadline:
        fig.text(0.5, y_pos - 0.04, subheadline, ha='center', fontsize=12,
                 color='#616161', style='italic')


def add_source_note(fig, note="AI Research Acceleration Model v1.0 | Baseline Scenario | 2030"):
    """Add source attribution."""
    fig.text(0.99, 0.01, note, ha='right', fontsize=8, color='#9e9e9e')


# =============================================================================
# FIGURE 1: THE HEADLINE CHART - Domain Comparison
# Story: "Structural Biology leads, but Drug Discovery is bounded"
# =============================================================================

def fig1_headline_chart():
    """
    Figure 1: Clear hierarchy showing domain acceleration ranking.

    Key improvements:
    - Sorted by acceleration (tells story of leaders vs laggards)
    - Color encodes speed (green=fast, red=slow)
    - Single key insight called out
    - Clean, minimal design
    """
    model = AIAccelerationModel()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS['background'])

    # Get forecasts and sort by acceleration
    forecasts = {d: model.forecast(d, 2030) for d in model.domains}
    sorted_domains = sorted(forecasts.keys(), key=lambda d: forecasts[d].acceleration)

    # Prepare data
    names = [model.DOMAIN_NAMES[d] for d in sorted_domains]
    accels = [forecasts[d].acceleration for d in sorted_domains]
    ci_lows = [forecasts[d].ci_90[0] for d in sorted_domains]
    ci_highs = [forecasts[d].ci_90[1] for d in sorted_domains]

    # Color gradient from red (slow) to green (fast)
    colors = [COLORS[d] for d in sorted_domains]

    # Create horizontal bars
    y_pos = np.arange(len(sorted_domains))
    bars = ax.barh(y_pos, accels, color=colors, alpha=0.85, height=0.7,
                   edgecolor='white', linewidth=1.5)

    # Add error bars (subtle)
    errors = [[a - l for a, l in zip(accels, ci_lows)],
              [h - a for a, h in zip(accels, ci_highs)]]
    ax.errorbar(accels, y_pos, xerr=errors, fmt='none', color='#424242',
                capsize=4, capthick=1.5, alpha=0.7)

    # Value labels with context
    for i, (accel, ci_l, ci_h) in enumerate(zip(accels, ci_lows, ci_highs)):
        ax.text(accel + 0.3, i, f'{accel:.1f}x', va='center', fontsize=13,
                fontweight='bold', color='#212121')
        ax.text(ci_h + 0.5, i, f'[{ci_l:.1f}-{ci_h:.1f}]', va='center',
                fontsize=9, color='#757575')

    # Reference line at 1x (no acceleration)
    ax.axvline(x=1, color=COLORS['neutral'], linestyle='-', linewidth=2,
               alpha=0.5, zorder=0)
    ax.text(1.05, len(sorted_domains) - 0.3, 'No acceleration', fontsize=9,
            color=COLORS['neutral'], va='bottom')

    # Key insight callout - Drug Discovery bounded
    dd_idx = sorted_domains.index('drug_discovery')
    ax.annotate(
        'Clinical trials\nremain the\nbottleneck',
        xy=(accels[dd_idx], dd_idx),
        xytext=(6, dd_idx - 0.5),
        fontsize=10,
        ha='left',
        va='top',
        color=COLORS['drug_discovery'],
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['drug_discovery'],
                       connectionstyle='arc3,rad=-0.2', lw=2)
    )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('Research Acceleration Factor', fontsize=13, fontweight='bold',
                  labelpad=10)
    ax.set_xlim(0, max(ci_highs) + 2.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)

    # Headline
    add_headline(fig,
                 'Structural Biology Leads AI-Driven Acceleration',
                 'By 2030, AI will speed research 1.5x to 6.7x depending on domain')

    add_source_note(fig)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.savefig(OUTPUT_DIR / "fig1_domain_comparison.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig1_domain_comparison.png")


# =============================================================================
# FIGURE 2: THE TRAJECTORY STORY
# Story: "Acceleration compounds over time - act now"
# =============================================================================

def fig2_trajectory_story():
    """
    Figure 2: Simplified trajectory focusing on the key message.

    Key improvements:
    - Single main chart (not 4 competing panels)
    - Shaded uncertainty band shows risk
    - Key milestones annotated
    - Clear call-to-action
    """
    model = AIAccelerationModel()

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(COLORS['background'])

    # Main trajectory chart
    ax = fig.add_axes([0.08, 0.15, 0.6, 0.68])

    years = list(range(2025, 2036))

    # Plot each domain with appropriate styling
    domain_order = ['structural_biology', 'protein_design', 'clinical_genomics',
                    'drug_discovery', 'materials_science']

    for domain in domain_order:
        traj = model.trajectory(domain, 2025, 2035)
        accels = [f.acceleration for f in traj]
        ci_lows = [f.ci_90[0] for f in traj]
        ci_highs = [f.ci_90[1] for f in traj]

        color = COLORS[domain]

        # Shaded confidence band (subtle)
        ax.fill_between(years, ci_lows, ci_highs, alpha=0.15, color=color)

        # Main line
        ax.plot(years, accels, color=color, linewidth=2.5, marker='o',
                markersize=5, label=model.DOMAIN_NAMES[domain])

    # Reference line
    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Milestone annotations
    ax.annotate('AlphaFold3\nimpact begins', xy=(2025, 3.5), fontsize=9,
               color='#616161', ha='center')
    ax.annotate('Regulatory\nreforms expected', xy=(2028, 1.5), fontsize=9,
               color='#616161', ha='center',
               arrowprops=dict(arrowstyle='->', color='#bdbdbd', lw=1))
    ax.annotate('Autonomous\nlabs scale', xy=(2032, 7), fontsize=9,
               color='#616161', ha='center')

    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Acceleration Factor (x)', fontsize=13, fontweight='bold')
    ax.set_xlim(2024.5, 2035.5)
    ax.set_ylim(0, 12)
    ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9,
              fontsize=10)

    # Side panel: Key numbers
    ax_side = fig.add_axes([0.72, 0.15, 0.25, 0.68])
    ax_side.axis('off')

    # Get 2030 and 2035 snapshots
    snap_2030 = model.system_snapshot(2030)
    snap_2035 = model.system_snapshot(2035)

    # Key metrics
    metrics = [
        ('2030', f'{snap_2030.total_acceleration:.1f}x', 'System Acceleration'),
        ('2035', f'{snap_2035.total_acceleration:.1f}x', 'System Acceleration'),
        ('Delta', f'+{(snap_2035.total_acceleration - snap_2030.total_acceleration):.1f}x', '5-Year Gain'),
    ]

    for i, (year, value, label) in enumerate(metrics):
        y = 0.85 - i * 0.28
        ax_side.text(0.1, y, year, fontsize=14, fontweight='bold', color='#424242')
        ax_side.text(0.1, y - 0.08, value, fontsize=28, fontweight='bold',
                    color=COLORS['highlight'])
        ax_side.text(0.1, y - 0.16, label, fontsize=11, color='#757575')

    # Call to action box
    ax_side.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.2,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#e3f2fd', edgecolor=COLORS['highlight'],
                                      linewidth=2))
    ax_side.text(0.5, 0.12, 'Early investment\ncompounds gains', ha='center',
                va='center', fontsize=11, fontweight='bold', color=COLORS['highlight'])

    add_headline(fig,
                 'Acceleration Compounds: Every Year of Delay Costs More',
                 'Domains diverge significantly by 2035 - early movers gain exponential advantage')

    add_source_note(fig)

    plt.savefig(OUTPUT_DIR / "fig2_trajectory_dashboard.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig2_trajectory_dashboard.png")


# =============================================================================
# FIGURE 3: SPILLOVER NETWORK - Simplified
# Story: "Structural Biology is the keystone - invest here first"
# =============================================================================

def fig3_spillover_simplified():
    """
    Figure 3: Clean network showing Structural Biology as the hub.

    Key improvements:
    - Structural Biology clearly positioned as central hub
    - Only major spillovers shown (>10%)
    - Flow direction emphasized
    - Clear hierarchy of importance
    """
    model = AIAccelerationModel()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    forecasts_2030 = {d: model.forecast(d, 2030) for d in model.domains}

    # Position Structural Biology at center-top as the hub
    positions = {
        'structural_biology': (0, 0.8),      # Top center - THE HUB
        'protein_design': (-1.1, 0),         # Left
        'drug_discovery': (1.1, 0),          # Right
        'clinical_genomics': (0.6, -0.9),    # Bottom right
        'materials_science': (-0.6, -0.9),   # Bottom left
    }

    # Draw major spillover arrows FIRST (underneath nodes)
    major_spillovers = [
        ('structural_biology', 'drug_discovery', 0.33),
        ('structural_biology', 'protein_design', 0.37),
        ('protein_design', 'drug_discovery', 0.16),
        ('clinical_genomics', 'drug_discovery', 0.12),
    ]

    for source, target, strength in major_spillovers:
        sx, sy = positions[source]
        tx, ty = positions[target]

        # Arrow styling based on strength
        lw = strength * 25
        alpha = 0.4 + strength * 0.8

        # Calculate offset for curved arrow
        dx, dy = tx - sx, ty - sy
        dist = np.sqrt(dx**2 + dy**2)

        # Draw curved arrow
        style = f"Simple,tail_width={lw},head_width={lw*2},head_length={lw}"
        arrow = FancyArrowPatch(
            (sx, sy), (tx, ty),
            connectionstyle="arc3,rad=0.15",
            arrowstyle=style,
            color='#90a4ae',
            alpha=alpha,
            zorder=1
        )
        ax.add_patch(arrow)

        # Spillover label
        mid_x = (sx + tx) / 2 + (ty - sy) * 0.15
        mid_y = (sy + ty) / 2 - (tx - sx) * 0.15
        ax.text(mid_x, mid_y, f'+{strength:.0%}', fontsize=11, fontweight='bold',
               ha='center', va='center', color='#546e7a',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='#90a4ae', alpha=0.9))

    # Draw domain nodes
    for domain, (x, y) in positions.items():
        accel = forecasts_2030[domain].acceleration

        # Size based on acceleration
        size = 0.18 + accel * 0.02

        # Structural biology gets special treatment as the hub
        if domain == 'structural_biology':
            size *= 1.3
            edge_width = 4
            edge_color = '#1a9850'
        else:
            edge_width = 2
            edge_color = 'white'

        circle = Circle((x, y), size, facecolor=COLORS[domain],
                        edgecolor=edge_color, linewidth=edge_width,
                        alpha=0.95, zorder=10)
        ax.add_patch(circle)

        # Domain name
        name = model.DOMAIN_NAMES[domain]
        if '\n' not in name and len(name) > 12:
            name = name.replace(' ', '\n')

        ax.text(x, y + 0.02, name, ha='center', va='center',
               fontsize=10 if domain != 'structural_biology' else 11,
               fontweight='bold', color='white', zorder=11)

        # Acceleration value
        ax.text(x, y - 0.08, f'{accel:.1f}x', ha='center', va='center',
               fontsize=9, color='white', alpha=0.9, zorder=11)

    # Legend explaining the visual encoding
    legend_y = -1.35
    ax.text(-1.5, legend_y, 'Node size = Acceleration', fontsize=10,
           color='#616161')
    ax.text(0, legend_y, 'Arrow width = Spillover strength', fontsize=10,
           color='#616161')
    ax.text(1.2, legend_y, 'Green border = Hub domain', fontsize=10,
           color='#616161')

    # Key insight callout box
    insight_box = FancyBboxPatch((-1.7, 0.95), 1.4, 0.45,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#e8f5e9',
                                  edgecolor=COLORS['structural_biology'],
                                  linewidth=2)
    ax.add_patch(insight_box)
    ax.text(-1.0, 1.17, 'KEY INSIGHT', fontsize=10, fontweight='bold',
           ha='center', color=COLORS['structural_biology'])
    ax.text(-1.0, 1.05, 'Structural Biology is the keystone:\n+70% combined spillover\nto downstream domains',
           fontsize=10, ha='center', va='top', color='#2e7d32')

    add_headline(fig,
                 'Structural Biology Powers the Entire Ecosystem',
                 'Investments here generate +33% to Drug Discovery and +37% to Protein Design')

    add_source_note(fig)

    plt.savefig(OUTPUT_DIR / "fig3_spillover_network.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig3_spillover_network.png")


# =============================================================================
# FIGURE 4: SCENARIO MATRIX - Actionable
# Story: "Drug Discovery needs regulatory reform across ALL scenarios"
# =============================================================================

def fig4_scenario_matrix():
    """
    Figure 4: Scenario matrix with clear action implications.

    Key improvements:
    - Highlight the "bounded" story for Drug Discovery
    - Show scenario probabilities clearly
    - Actionable interpretation for each scenario
    """
    model = AIAccelerationModel()

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(COLORS['background'])

    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

    # Main heatmap
    ax = fig.add_subplot(gs[0])

    scenarios = ['pessimistic', 'conservative', 'baseline', 'optimistic', 'breakthrough']
    scenario_labels = ['Pessimistic\n(10% prob)', 'Conservative\n(20% prob)',
                      'Baseline\n(40% prob)', 'Optimistic\n(20% prob)',
                      'Breakthrough\n(10% prob)']

    # Order domains by baseline acceleration
    domain_order = ['structural_biology', 'protein_design', 'clinical_genomics',
                    'drug_discovery', 'materials_science']
    domain_names = [model.DOMAIN_NAMES[d] for d in domain_order]

    # Build data matrix
    data = np.zeros((len(domain_order), len(scenarios)))
    for i, domain in enumerate(domain_order):
        for j, scenario in enumerate(scenarios):
            f = model.forecast(domain, 2030, scenario)
            data[i, j] = f.acceleration

    # Custom colormap: white -> yellow -> green
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ['#ffebee', '#fff9c4', '#c8e6c9', '#81c784', '#2e7d32']
    cmap = LinearSegmentedColormap.from_list('acceleration', colors_cmap, N=256)

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.5, vmax=12)

    # Add text annotations
    for i in range(len(domain_order)):
        for j in range(len(scenarios)):
            val = data[i, j]
            # Dark text on light cells, light text on dark cells
            color = 'white' if val > 8 else '#212121'
            fontweight = 'bold' if j == 2 else 'normal'  # Baseline column bold
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center',
                   fontsize=12, fontweight=fontweight, color=color)

    # Highlight Drug Discovery row
    dd_idx = domain_order.index('drug_discovery')
    ax.add_patch(plt.Rectangle((-0.5, dd_idx - 0.5), len(scenarios), 1,
                               fill=False, edgecolor=COLORS['drug_discovery'],
                               linewidth=3, linestyle='--'))

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.set_yticks(range(len(domain_order)))
    ax.set_yticklabels(domain_names, fontsize=11)
    ax.tick_params(axis='both', length=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Acceleration (x)', fontsize=11)

    # Interpretation panel
    ax_right = fig.add_subplot(gs[1])
    ax_right.axis('off')

    interpretations = [
        ('STRUCTURAL\nBIOLOGY', '4.0x - 12.0x', 'Always leads\nRegardless of scenario', '#1a9850'),
        ('DRUG\nDISCOVERY', '1.6x - 4.8x', 'Bounded by trials\nNeeds regulatory reform', '#fc8d59'),
        ('MATERIALS\nSCIENCE', '0.9x - 2.7x', 'Synthesis bottleneck\nNeeds automation', '#d73027'),
    ]

    ax_right.text(0.1, 0.95, 'WHAT THIS MEANS', fontsize=12, fontweight='bold',
                 color='#212121')

    for i, (domain, range_val, interpretation, color) in enumerate(interpretations):
        y = 0.82 - i * 0.28
        ax_right.add_patch(FancyBboxPatch((0.02, y - 0.15), 0.96, 0.22,
                                          boxstyle="round,pad=0.02",
                                          facecolor='white',
                                          edgecolor=color, linewidth=2))
        ax_right.text(0.08, y + 0.02, domain, fontsize=9, fontweight='bold',
                     color=color)
        ax_right.text(0.08, y - 0.05, range_val, fontsize=11, fontweight='bold',
                     color='#212121')
        ax_right.text(0.08, y - 0.12, interpretation, fontsize=9, color='#616161')

    add_headline(fig,
                 'Drug Discovery Remains Bounded in Every Scenario',
                 'Regulatory reform is essential - technology alone cannot overcome clinical trial constraints')

    add_source_note(fig)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.savefig(OUTPUT_DIR / "fig4_scenario_heatmap.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig4_scenario_heatmap.png")


# =============================================================================
# FIGURE 5: WORKFORCE IMPACT - The Human Story
# Story: "Net positive, but transition support is critical"
# =============================================================================

def fig5_workforce_story():
    """
    Figure 5: Workforce impact with human-centered narrative.

    Key improvements:
    - Lead with net positive message
    - Show flow from displaced to created
    - Emphasize transition support need
    """
    model = AIAccelerationModel()
    snapshot = model.system_snapshot(2030)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(COLORS['background'])

    # Main sankey-style flow diagram
    ax = fig.add_axes([0.05, 0.15, 0.55, 0.68])

    total_displaced = snapshot.total_displaced
    total_created = snapshot.total_created
    net = snapshot.workforce_change

    # Visual: Two bars with connecting flow
    bar_width = 0.3

    # Displaced bar (left)
    ax.barh(0.5, total_displaced, height=bar_width, color=COLORS['negative'],
            alpha=0.8, label='Displaced')
    ax.text(total_displaced / 2, 0.5, f'{total_displaced:.2f}M\nDisplaced',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Created bar (right, starting from displaced)
    ax.barh(0.5, total_created, height=bar_width, left=total_displaced + 0.1,
            color=COLORS['positive'], alpha=0.8, label='Created')
    ax.text(total_displaced + 0.1 + total_created / 2, 0.5, f'{total_created:.2f}M\nCreated',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Net indicator
    ax.annotate('', xy=(total_displaced + 0.1 + total_created + 0.3, 0.5),
               xytext=(total_displaced + 0.1 + total_created + 0.1, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['positive'], lw=3))
    ax.text(total_displaced + 0.1 + total_created + 0.4, 0.5,
            f'NET: +{net:.2f}M', fontsize=16, fontweight='bold',
            color=COLORS['positive'], va='center')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Domain breakdown panel
    ax2 = fig.add_axes([0.62, 0.15, 0.35, 0.68])

    domains = list(model.domains)
    domain_names = [model.DOMAIN_NAMES[d].replace(' ', '\n') for d in domains]

    # Sort by net impact
    net_values = [(d, snapshot.domain_forecasts[d].net_jobs) for d in domains]
    net_values.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(domains))
    colors = [COLORS['positive'] if n > 0 else COLORS['negative']
              for _, n in net_values]

    bars = ax2.barh(y_pos, [n for _, n in net_values], color=colors, alpha=0.8,
                    height=0.6)

    for i, (d, n) in enumerate(net_values):
        offset = 0.05 if n > 0 else -0.05
        ha = 'left' if n > 0 else 'right'
        ax2.text(n + offset, i, f'{n:+.2f}M', va='center', ha=ha,
                fontsize=11, fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([model.DOMAIN_NAMES[d] for d, _ in net_values], fontsize=10)
    ax2.axvline(x=0, color='#212121', linewidth=1)
    ax2.set_xlabel('Net Jobs (Millions)', fontsize=11, fontweight='bold')
    ax2.set_title('By Domain', fontsize=12, fontweight='bold')
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', length=0)

    # Key message box at bottom
    fig.text(0.5, 0.05,
             'While net impact is positive, displaced workers need transition support: '
             'retraining programs, job placement services, and income bridges.',
             ha='center', fontsize=11, style='italic', color='#616161',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1',
                      edgecolor='#ffb300', alpha=0.9))

    add_headline(fig,
                 'AI Creates More Jobs Than It Displaces',
                 f'By 2030: {total_created:.2f}M new jobs vs {total_displaced:.2f}M displaced = Net +{net:.2f}M')

    add_source_note(fig)

    plt.savefig(OUTPUT_DIR / "fig5_workforce_summary.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig5_workforce_summary.png")


# =============================================================================
# FIGURE 6: EXECUTIVE DASHBOARD - One Page Decision Brief
# Story: "Here's what you need to know and do"
# =============================================================================

def fig6_executive_brief():
    """
    Figure 6: Executive dashboard for C-suite/policymakers.

    Key improvements:
    - Clear visual hierarchy: KPIs -> Details -> Actions
    - Traffic light system for quick assessment
    - Specific, actionable recommendations
    """
    model = AIAccelerationModel()
    snapshot = model.system_snapshot(2030)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(COLORS['background'])

    # Title banner
    fig.text(0.5, 0.96, 'AI Research Acceleration: Executive Brief',
            ha='center', fontsize=22, fontweight='bold', color='#212121')
    fig.text(0.5, 0.925, '2030 Baseline Projection  |  Decision Support Dashboard',
            ha='center', fontsize=13, color='#616161')

    # =========================
    # TOP ROW: Key Performance Indicators
    # =========================
    kpis = [
        (f'{snapshot.total_acceleration:.1f}x', 'System\nAcceleration',
         COLORS['highlight'], 'UP', 'Research proceeds 3.4x faster overall'),
        (f'+{snapshot.workforce_change:.1f}M', 'Net Jobs\nImpact',
         COLORS['positive'], 'UP', 'More jobs created than displaced'),
        (f'{snapshot.critical_actions}', 'Critical\nActions',
         COLORS['drug_discovery'], 'WARN', 'Policy actions needed for full gains'),
        (f'${120}B', 'Investment\nNeeded',
         '#5e35b1', 'INFO', 'Est. infrastructure investment'),
    ]

    kpi_width = 0.2
    for i, (value, label, color, status, tooltip) in enumerate(kpis):
        x = 0.1 + i * 0.23

        # KPI box
        fig.add_axes([x, 0.77, kpi_width, 0.12])
        ax_kpi = plt.gca()
        ax_kpi.axis('off')

        # Status indicator
        status_colors = {'UP': COLORS['positive'], 'WARN': '#ff9800', 'INFO': '#2196f3'}
        fig.text(x + 0.01, 0.87, '\u25CF', fontsize=12, color=status_colors[status])

        # Value and label
        fig.text(x + kpi_width/2, 0.83, value, ha='center', fontsize=32,
                fontweight='bold', color=color)
        fig.text(x + kpi_width/2, 0.775, label, ha='center', fontsize=11,
                color='#616161')

    # =========================
    # MIDDLE LEFT: Domain Performance
    # =========================
    ax_domains = fig.add_axes([0.05, 0.35, 0.4, 0.35])

    # Sort domains by acceleration
    domain_data = [(d, snapshot.domain_forecasts[d].acceleration)
                   for d in model.domains]
    domain_data.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(domain_data))
    bars = ax_domains.barh(y_pos, [a for _, a in domain_data],
                           color=[COLORS[d] for d, _ in domain_data],
                           alpha=0.85, height=0.6)

    for i, (d, a) in enumerate(domain_data):
        ax_domains.text(a + 0.15, i, f'{a:.1f}x', va='center', fontsize=11,
                       fontweight='bold')

    ax_domains.axvline(x=1, color=COLORS['neutral'], linestyle='--', linewidth=1.5)
    ax_domains.set_yticks(y_pos)
    ax_domains.set_yticklabels([model.DOMAIN_NAMES[d] for d, _ in domain_data],
                              fontsize=11)
    ax_domains.set_xlabel('Acceleration Factor', fontsize=11, fontweight='bold')
    ax_domains.set_title('Performance by Domain', fontsize=13, fontweight='bold',
                        pad=10)
    ax_domains.set_xlim(0, 9)
    ax_domains.spines['left'].set_visible(False)
    ax_domains.tick_params(axis='y', length=0)

    # =========================
    # MIDDLE RIGHT: 10-Year Trajectory
    # =========================
    ax_traj = fig.add_axes([0.52, 0.35, 0.43, 0.35])

    years = list(range(2025, 2036))
    system_traj = model.trajectory(None, 2025, 2035)
    total_accels = [s.total_acceleration for s in system_traj]
    ci_lows = [s.acceleration_ci_90[0] for s in system_traj]
    ci_highs = [s.acceleration_ci_90[1] for s in system_traj]

    ax_traj.fill_between(years, ci_lows, ci_highs, alpha=0.2, color=COLORS['highlight'])
    ax_traj.plot(years, total_accels, color=COLORS['highlight'], linewidth=3,
                marker='o', markersize=6)
    ax_traj.axhline(y=1, color=COLORS['neutral'], linestyle='--', linewidth=1.5)

    # Mark 2030
    ax_traj.axvline(x=2030, color='#e0e0e0', linewidth=2, linestyle=':')
    ax_traj.text(2030.1, 1.5, '2030', fontsize=10, color='#9e9e9e')

    ax_traj.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax_traj.set_ylabel('System Acceleration', fontsize=11, fontweight='bold')
    ax_traj.set_title('10-Year Trajectory (with 90% CI)', fontsize=13,
                     fontweight='bold', pad=10)
    ax_traj.set_xlim(2024.5, 2035.5)
    ax_traj.set_ylim(0, 7)

    # =========================
    # BOTTOM: Key Insights and Actions
    # =========================

    # Insights (left)
    insights_box = fig.add_axes([0.05, 0.05, 0.42, 0.25])
    insights_box.axis('off')

    fig.text(0.06, 0.28, 'KEY INSIGHTS', fontsize=13, fontweight='bold',
            color='#212121')

    insights = [
        (COLORS['positive'], f"Structural Biology leads at {snapshot.domain_forecasts['structural_biology'].acceleration:.1f}x - invest here first"),
        (COLORS['drug_discovery'], "Drug Discovery bounded by clinical trials in ALL scenarios"),
        (COLORS['positive'], f"Spillover effects add +30-40% to downstream domains"),
        (COLORS['highlight'], f"Net +{snapshot.workforce_change:.1f}M jobs, but transition support essential"),
    ]

    for i, (color, text) in enumerate(insights):
        fig.text(0.07, 0.24 - i * 0.045, '\u25CF', fontsize=11, color=color)
        fig.text(0.09, 0.24 - i * 0.045, text, fontsize=10, color='#424242')

    # Actions (right)
    actions_box = fig.add_axes([0.52, 0.05, 0.43, 0.25])
    actions_box.axis('off')

    fig.text(0.53, 0.28, 'PRIORITY ACTIONS', fontsize=13, fontweight='bold',
            color='#212121')

    recs = model.get_policy_recommendations(2030)

    # Priority colors
    priority_colors = {'critical': COLORS['negative'], 'high': '#ff9800', 'medium': '#2196f3'}

    for i, rec in enumerate(recs[:4]):
        y = 0.24 - i * 0.05
        color = priority_colors.get(rec.priority, '#757575')
        fig.text(0.54, y, f'[{rec.priority.upper()[:1]}]', fontsize=9,
                fontweight='bold', color=color)
        fig.text(0.57, y, f'{rec.title}', fontsize=10, color='#424242')
        fig.text(0.57, y - 0.018, f'{rec.timeline} | {rec.investment}',
                fontsize=8, color='#9e9e9e')

    add_source_note(fig, "AI Research Acceleration Model v1.0 | Baseline Scenario | Generated Jan 2026")

    plt.savefig(OUTPUT_DIR / "fig6_executive_dashboard.png", dpi=150,
                bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print("Generated: fig6_executive_dashboard.png")


# =============================================================================
# MAIN: Generate all improved figures
# =============================================================================

def generate_all_v2_figures():
    """Generate all improved v2 visualizations."""
    print("\n" + "=" * 70)
    print("Generating v1.0 Visualizations v2 - Expert-Level Storytelling")
    print("=" * 70 + "\n")

    fig1_headline_chart()
    fig2_trajectory_story()
    fig3_spillover_simplified()
    fig4_scenario_matrix()
    fig5_workforce_story()
    fig6_executive_brief()

    print(f"\n{'=' * 70}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)

    # Print storytelling summary
    print("\nNARRATIVE ARC:")
    print("  Fig 1: 'Structural Biology leads' - establishes the landscape")
    print("  Fig 2: 'Acceleration compounds' - creates urgency")
    print("  Fig 3: 'Invest in the hub' - strategic recommendation")
    print("  Fig 4: 'Reform is essential' - policy imperative")
    print("  Fig 5: 'Jobs grow, with care' - addresses human concern")
    print("  Fig 6: 'Here's what to do' - actionable summary")


if __name__ == "__main__":
    generate_all_v2_figures()
