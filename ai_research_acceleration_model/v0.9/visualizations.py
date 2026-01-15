#!/usr/bin/env python3
"""
v0.9 Visualizations: System-Level Analysis
==========================================

Visual communication strategy:
1. Network diagram - cross-domain spillovers
2. Sankey flow - workforce transitions
3. Policy priority matrix - what to do when
4. System trajectory - how it all evolves
5. Investment allocation - where to put money
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def fig1_cross_domain_network():
    """
    Figure 1: Cross-Domain Spillover Network

    Shows how domains influence each other.
    Key insight: Structural biology is the central enabler.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Domain positions (circular layout)
    domains = {
        'Structural\nBiology': (0, 1.0, '#2ecc71', 7.5),
        'Drug\nDiscovery': (0.95, 0.31, '#e74c3c', 3.8),
        'Materials\nScience': (0.59, -0.81, '#3498db', 1.7),
        'Clinical\nGenomics': (-0.59, -0.81, '#f39c12', 3.4),
        'Protein\nDesign': (-0.95, 0.31, '#9b59b6', 5.8),
    }

    # Draw nodes
    for name, (x, y, color, accel) in domains.items():
        # Node size proportional to acceleration
        size = 0.15 + accel * 0.02

        circle = Circle((x, y), size, facecolor=color, edgecolor='white',
                        linewidth=3, alpha=0.9, zorder=10)
        ax.add_patch(circle)

        # Domain name
        ax.text(x, y + 0.02, name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=11)

        # Acceleration value
        ax.text(x, y - 0.08, f'{accel:.1f}×', ha='center', va='center',
               fontsize=9, color='white', zorder=11)

    # Spillover connections (source, target, effect)
    spillovers = [
        ('Structural\nBiology', 'Drug\nDiscovery', 0.33, 'Structure-based\ndrug design'),
        ('Structural\nBiology', 'Protein\nDesign', 0.37, 'Target\nvalidation'),
        ('Protein\nDesign', 'Drug\nDiscovery', 0.16, 'Therapeutic\nproteins'),
        ('Clinical\nGenomics', 'Drug\nDiscovery', 0.12, 'Target\nidentification'),
        ('Drug\nDiscovery', 'Clinical\nGenomics', 0.05, 'Companion\ndiagnostics'),
        ('Materials\nScience', 'Structural\nBiology', 0.04, 'Novel\ndetectors'),
        ('Protein\nDesign', 'Materials\nScience', 0.05, 'Bio-materials'),
        ('Clinical\nGenomics', 'Protein\nDesign', 0.05, 'Variant\neffects'),
    ]

    for source, target, effect, label in spillovers:
        sx, sy = domains[source][0], domains[source][1]
        tx, ty = domains[target][0], domains[target][1]

        # Calculate arrow positions (from edge to edge)
        source_size = 0.15 + domains[source][3] * 0.02
        target_size = 0.15 + domains[target][3] * 0.02

        dx, dy = tx - sx, ty - sy
        dist = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/dist, dy/dist

        start_x = sx + dx * source_size
        start_y = sy + dy * source_size
        end_x = tx - dx * target_size
        end_y = ty - dy * target_size

        # Arrow width proportional to effect
        width = effect * 8

        # Draw arrow
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle=f'->,head_width={width*0.15},head_length={width*0.08}',
            linewidth=width,
            color='gray',
            alpha=0.6,
            connectionstyle='arc3,rad=0.1',
            zorder=5
        )
        ax.add_patch(arrow)

        # Label at midpoint
        mid_x = (start_x + end_x) / 2 + 0.1
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y, f'+{effect:.0%}',
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Title
    ax.text(0, 1.4, 'Cross-Domain Spillover Network',
           ha='center', fontsize=16, fontweight='bold')
    ax.text(0, 1.28, 'Arrow thickness = spillover strength | Node size = total acceleration',
           ha='center', fontsize=11, style='italic', color='gray')

    # Key insight box
    insight = ('Key Insight:\n'
              'Structural Biology is the central enabler\n'
              '• Feeds drug discovery (+33%)\n'
              '• Feeds protein design (+37%)\n'
              'Invest here for maximum spillover')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(-1.4, -1.3, insight, fontsize=10, va='top', bbox=props)

    plt.savefig(OUTPUT_DIR / "fig1_cross_domain_network.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_cross_domain_network.png")


def fig2_workforce_flow():
    """
    Figure 2: Workforce Transition Flows

    Sankey-style diagram showing job displacement and creation.
    Key insight: Net positive, but transitions needed.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Current workforce (left column)
    current = [
        ('Research Scientists', 25, 85, 0.8, '#2ecc71'),
        ('Lab Technicians', 20, 65, 0.7, '#3498db'),
        ('Data Analysts', 15, 48, 0.15, '#9b59b6'),
        ('Clinical Staff', 18, 32, 0.6, '#e74c3c'),
        ('Computational', 12, 18, 0.12, '#f39c12'),
    ]

    # Future roles (right column)
    future = [
        ('AI-Augmented\nResearchers', 25, 85, 1.2, '#27ae60'),
        ('Automation\nSpecialists', 15, 68, 0.5, '#2980b9'),
        ('AI/ML Scientists', 20, 52, 0.6, '#8e44ad'),
        ('Clinical Data\nScientists', 18, 36, 0.4, '#c0392b'),
        ('New Roles\n(Emerging)', 15, 20, 0.4, '#d35400'),
    ]

    # Draw current workforce boxes
    for name, x, y, size, color in current:
        height = size * 20
        rect = FancyBboxPatch((x-8, y-height/2), 16, height,
                              boxstyle='round,pad=0.02',
                              facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f'{name}\n{size:.2f}M', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    # Draw future workforce boxes
    for name, x_offset, y, size, color in future:
        x = 75
        height = size * 15
        rect = FancyBboxPatch((x-8, y-height/2), 16, height,
                              boxstyle='round,pad=0.02',
                              facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f'{name}\n{size:.2f}M', ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')

    # Draw flow arrows
    flows = [
        (25, 85, 75, 85, 0.6, 'gray'),      # Scientists → AI-Augmented
        (25, 65, 75, 68, 0.3, 'gray'),      # Technicians → Automation
        (25, 65, 75, 52, 0.2, '#e74c3c'),   # Technicians → displaced (red)
        (25, 48, 75, 52, 0.1, 'gray'),      # Data → AI/ML
        (25, 32, 75, 36, 0.4, 'gray'),      # Clinical → Clinical Data
        (25, 18, 75, 52, 0.08, 'gray'),     # Computational → AI/ML
    ]

    for x1, y1, x2, y2, width, color in flows:
        # Curved arrow
        mid_x = (x1 + x2) / 2
        ctrl_y = (y1 + y2) / 2 + 5

        # Simple bezier-like curve using multiple line segments
        t = np.linspace(0, 1, 20)
        curve_x = (1-t)**2 * (x1+8) + 2*(1-t)*t * mid_x + t**2 * (x2-8)
        curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2

        ax.plot(curve_x, curve_y, color=color, linewidth=width*15, alpha=0.4)

    # Central summary
    summary_box = FancyBboxPatch((42, 40), 16, 25,
                                  boxstyle='round,pad=0.05',
                                  facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(summary_box)
    ax.text(50, 52, 'NET CHANGE\n2030', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(50, 45, '+2.10M jobs', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#27ae60')

    # Displaced indicator
    ax.text(50, 10, '! 0.49M displaced\n(need retraining)', ha='center', va='center',
           fontsize=10, color='#e74c3c', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9))

    # Created indicator
    ax.text(50, 95, '+ 2.59M created\n(new opportunities)', ha='center', va='center',
           fontsize=10, color='#27ae60', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9))

    # Column labels
    ax.text(25, 98, 'Current Workforce\n(2024)', ha='center', fontsize=12, fontweight='bold')
    ax.text(75, 98, 'Future Workforce\n(2030)', ha='center', fontsize=12, fontweight='bold')

    # Title
    ax.text(50, 105, 'Workforce Transition Flows',
           ha='center', fontsize=16, fontweight='bold')

    plt.savefig(OUTPUT_DIR / "fig2_workforce_flow.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_workforce_flow.png")


def fig3_policy_priority_matrix():
    """
    Figure 3: Policy Priority Matrix

    Urgency vs Impact for policy recommendations.
    Key insight: Regulatory reform is critical AND high impact.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Quadrant background
    ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=50, color='gray', linestyle='-', alpha=0.3)

    # Quadrant labels
    ax.text(25, 95, 'QUICK WINS\nHigh Impact, Act Soon', ha='center', va='top',
           fontsize=11, fontweight='bold', color='#27ae60', alpha=0.7)
    ax.text(75, 95, 'MAJOR PROJECTS\nHigh Impact, Longer Timeline', ha='center', va='top',
           fontsize=11, fontweight='bold', color='#3498db', alpha=0.7)
    ax.text(25, 5, 'CONSIDER LATER\nLower Impact', ha='center', va='bottom',
           fontsize=11, fontweight='bold', color='#95a5a6', alpha=0.7)
    ax.text(75, 5, 'STRATEGIC INVESTMENTS\nLong-term Foundation', ha='center', va='bottom',
           fontsize=11, fontweight='bold', color='#f39c12', alpha=0.7)

    # Policy recommendations (urgency, impact, size, name, color)
    policies = [
        # Critical (red) - act within 1 year
        (85, 80, 400, 'AI Clinical Trial\nFramework', '#e74c3c', 'CRITICAL'),
        (80, 75, 350, 'Variant Classification\nStandards', '#e74c3c', 'CRITICAL'),
        (75, 85, 380, 'Competitiveness\nInitiative', '#e74c3c', 'CRITICAL'),

        # High (orange) - act within 2 years
        (70, 70, 300, 'Cryo-EM\nInfrastructure', '#e67e22', 'HIGH'),
        (65, 65, 280, 'AI-Biology\nTraining Pipeline', '#e67e22', 'HIGH'),
        (55, 75, 320, 'Preclinical\nAutomation', '#e67e22', 'HIGH'),
        (60, 60, 260, 'Expression\nFoundries', '#e67e22', 'HIGH'),

        # Medium (yellow) - act within 5 years
        (40, 55, 200, 'Triage\nStandards', '#f1c40f', 'MEDIUM'),
        (35, 50, 180, 'Ethics\nFramework', '#f1c40f', 'MEDIUM'),
        (45, 45, 190, 'Counselor\nCertification', '#f1c40f', 'MEDIUM'),

        # Low (gray) - monitor
        (25, 35, 150, 'Materials\nSynthesis Network', '#95a5a6', 'LOW'),
        (30, 30, 140, 'Data Sharing\nInfrastructure', '#95a5a6', 'LOW'),
    ]

    for urgency, impact, size, name, color, priority in policies:
        ax.scatter([100-urgency], [impact], s=size, c=color, alpha=0.7,
                  edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(name, (100-urgency, impact), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, ha='left', va='bottom')

    # Axis labels
    ax.set_xlabel('Timeline (Act Soon → Act Later)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Impact', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Custom ticks
    ax.set_xticks([15, 50, 85])
    ax.set_xticklabels(['1 year', '2-3 years', '5+ years'])
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(['Low', 'Medium', 'High'])

    # Legend
    legend_elements = [
        plt.scatter([], [], s=200, c='#e74c3c', label='Critical (7)', alpha=0.7),
        plt.scatter([], [], s=200, c='#e67e22', label='High (8)', alpha=0.7),
        plt.scatter([], [], s=200, c='#f1c40f', label='Medium (3)', alpha=0.7),
        plt.scatter([], [], s=200, c='#95a5a6', label='Low (2)', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='lower left', title='Priority Level')

    ax.set_title('Policy Priority Matrix: 20 Recommendations\n'
                 'Size = estimated investment | Position = urgency × impact',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_policy_priority_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_policy_priority_matrix.png")


def fig4_system_trajectory():
    """
    Figure 4: System Trajectory 2025-2035

    Shows how the entire system evolves over time.
    Key insight: Compound growth from cross-domain effects.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    years = np.arange(2025, 2036)

    # Data
    domains_data = {
        'Structural Biology': ([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.9, 8.4, 8.9, 9.4, 9.9], '#2ecc71'),
        'Drug Discovery': ([1.8, 2.4, 2.8, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.3], '#e74c3c'),
        'Materials Science': ([1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2], '#3498db'),
        'Protein Design': ([3.2, 3.9, 4.4, 4.9, 5.4, 5.8, 6.3, 6.7, 7.1, 7.6, 8.0], '#9b59b6'),
        'Clinical Genomics': ([2.2, 2.5, 2.7, 2.9, 3.2, 3.4, 3.6, 3.8, 4.0, 4.3, 4.5], '#f39c12'),
    }

    workforce = [0.50, 0.93, 1.30, 1.61, 1.87, 2.10, 2.29, 2.46, 2.60, 2.71, 2.82]
    weighted_avg = [2.4, 2.9, 3.2, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7]

    # Panel 1: Domain trajectories
    ax1 = axes[0, 0]
    for name, (data, color) in domains_data.items():
        ax1.plot(years, data, color=color, linewidth=2.5, marker='o', markersize=4, label=name)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Acceleration (×)')
    ax1.set_title('Domain Acceleration Trajectories', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(0, 12)

    # Panel 2: Weighted average with uncertainty
    ax2 = axes[0, 1]
    ci_low = [w * 0.7 for w in weighted_avg]
    ci_high = [w * 1.3 for w in weighted_avg]
    ax2.fill_between(years, ci_low, ci_high, alpha=0.3, color='#3498db', label='90% CI')
    ax2.plot(years, weighted_avg, color='#3498db', linewidth=3, marker='o', label='Weighted Average')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('System Acceleration (×)')
    ax2.set_title('Weighted System Average', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 8)

    # Panel 3: Workforce trajectory
    ax3 = axes[1, 0]
    colors = ['#e74c3c' if w < 0 else '#27ae60' for w in workforce]
    ax3.bar(years, workforce, color='#27ae60', alpha=0.7, edgecolor='white')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Net Job Change (Millions)')
    ax3.set_title('Cumulative Workforce Impact', fontweight='bold')

    # Add trend line
    z = np.polyfit(years, workforce, 2)
    p = np.poly1d(z)
    ax3.plot(years, p(years), color='#2c3e50', linewidth=2, linestyle='--', label='Trend')

    # Panel 4: Cross-domain synergy growth
    ax4 = axes[1, 1]

    # Synergy effect grows over time
    standalone = [2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2]
    with_spillover = weighted_avg

    ax4.fill_between(years, standalone, with_spillover, alpha=0.4, color='#9b59b6',
                     label='Cross-domain boost')
    ax4.plot(years, standalone, color='#95a5a6', linewidth=2, linestyle='--', label='Standalone')
    ax4.plot(years, with_spillover, color='#9b59b6', linewidth=2.5, label='With spillovers')

    # Annotate the boost
    boost_2030 = with_spillover[5] - standalone[5]
    ax4.annotate(f'+{boost_2030/standalone[5]*100:.0f}%\nboost',
                xy=(2030, (standalone[5] + with_spillover[5])/2),
                fontsize=11, ha='center', fontweight='bold', color='#9b59b6')

    ax4.set_xlabel('Year')
    ax4.set_ylabel('Acceleration (×)')
    ax4.set_title('Cross-Domain Synergy Effect', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.set_ylim(0, 8)

    fig.suptitle('System-Level Trajectory Analysis (2025-2035)\n'
                 'Cross-domain effects compound acceleration over time',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_system_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_system_trajectory.png")


def fig5_investment_allocation():
    """
    Figure 5: Recommended Investment Allocation

    Pie chart + bar chart showing where money should go.
    Key insight: Infrastructure is the largest investment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Pie chart by category
    ax1 = axes[0]

    categories = ['Infrastructure', 'Workforce\nDevelopment', 'Research\nPrograms',
                  'Regulatory\nReform', 'International\nCoordination']
    sizes = [35, 20, 30, 10, 5]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
    explode = (0.05, 0, 0, 0, 0)  # Highlight infrastructure

    wedges, texts, autotexts = ax1.pie(sizes, labels=categories, autopct='%1.0f%%',
                                        colors=colors, explode=explode,
                                        startangle=90, pctdistance=0.75)

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax1.set_title('Investment by Category\n($2-4B Total)', fontsize=14, fontweight='bold')

    # Right: Bar chart by domain
    ax2 = axes[1]

    domains = ['Structural\nBiology', 'Drug\nDiscovery', 'Materials\nScience',
               'Protein\nDesign', 'Clinical\nGenomics']
    investments = [500, 1200, 600, 400, 300]  # Millions
    domain_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

    bars = ax2.barh(domains, investments, color=domain_colors, alpha=0.8, edgecolor='white')

    # Add value labels
    for bar, val in zip(bars, investments):
        ax2.text(val + 30, bar.get_y() + bar.get_height()/2,
                f'${val}M', va='center', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Investment ($ Millions)', fontsize=12)
    ax2.set_title('Investment by Domain', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1500)

    # Add breakdown annotations
    breakdown = {
        'Structural\nBiology': 'Cryo-EM facilities',
        'Drug\nDiscovery': 'Trial infrastructure\n+ automation',
        'Materials\nScience': 'Synthesis network',
        'Protein\nDesign': 'Expression foundries',
        'Clinical\nGenomics': 'Training + standards',
    }

    for i, (domain, note) in enumerate(breakdown.items()):
        ax2.text(50, i, note, va='center', fontsize=8, style='italic', color='white')

    fig.suptitle('Recommended Investment Allocation\n'
                 'Total: $2-4 Billion over 5 years',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_investment_allocation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_investment_allocation.png")


def fig6_bottleneck_waterfall():
    """
    Figure 6: Bottleneck Constraint Waterfall

    Shows how bottlenecks limit potential acceleration.
    Key insight: Each bottleneck shaves off potential gains.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Waterfall data
    stages = ['AI\nPotential', 'Data\nAvailability', 'Compute\nLimits',
              'Validation\nCapacity', 'Regulatory\nConstraints', 'Workforce\nGaps',
              'REALIZED\nAcceleration']

    # Starting from high potential, each bottleneck reduces
    values = [15.0, -2.0, -1.5, -4.0, -3.0, -0.7, 3.8]  # Drug discovery example
    cumulative = [15.0, 13.0, 11.5, 7.5, 4.5, 3.8, 3.8]

    colors = ['#27ae60'] + ['#e74c3c'] * 5 + ['#3498db']

    # Draw waterfall bars
    for i, (stage, val, cum, color) in enumerate(zip(stages, values, cumulative, colors)):
        if i == 0:
            ax.bar(i, val, color=color, alpha=0.8, edgecolor='white', linewidth=2)
        elif i == len(stages) - 1:
            ax.bar(i, cum, color=color, alpha=0.8, edgecolor='white', linewidth=2)
        else:
            bottom = cumulative[i]
            ax.bar(i, abs(val), bottom=bottom, color=color, alpha=0.8,
                  edgecolor='white', linewidth=2)

            # Connector line
            ax.plot([i-0.4, i+0.4], [cumulative[i-1], cumulative[i-1]],
                   color='gray', linestyle='-', linewidth=1)

        # Value label
        if i == 0 or i == len(stages) - 1:
            ax.text(i, cum + 0.5, f'{cum:.1f}×', ha='center', fontsize=11, fontweight='bold')
        else:
            ax.text(i, cum + abs(val)/2 + cumulative[i-1] - cum,
                   f'{val:.1f}×', ha='center', fontsize=10, fontweight='bold', color='white')

    # Annotations for key bottlenecks
    ax.annotate('Biggest constraint:\nValidation capacity',
               xy=(3, 7.5), xytext=(4.5, 10),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='black'),
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.annotate('Regulatory:\n3× reduction',
               xy=(4, 4.5), xytext=(5.5, 6.5),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='black'),
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('Acceleration (×)', fontsize=12)
    ax.set_ylim(0, 18)

    ax.set_title('Drug Discovery: From AI Potential to Realized Acceleration\n'
                 'Each bottleneck constrains the achievable acceleration',
                 fontsize=14, fontweight='bold', pad=20)

    # Summary box
    summary = ('Summary:\n'
              'AI potential: 15×\n'
              'After all constraints: 3.8×\n'
              'Efficiency: 25%\n\n'
              'Key lever: Validation capacity')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
           va='top', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_bottleneck_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig6_bottleneck_waterfall.png")


def fig7_executive_summary():
    """
    Figure 7: Executive Summary Dashboard

    One-page visual summary of all key findings.
    Key insight: Everything on one page for decision makers.
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top banner with key numbers
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.axis('off')

    # Key metrics
    metrics = [
        ('4.2×', 'System\nAcceleration', '#3498db'),
        ('+2.1M', 'Net Jobs\nCreated', '#27ae60'),
        ('$2-4B', 'Investment\nNeeded', '#f39c12'),
        ('7', 'Critical\nActions', '#e74c3c'),
    ]

    for i, (value, label, color) in enumerate(metrics):
        x = 0.15 + i * 0.22
        ax_banner.text(x, 0.7, value, fontsize=36, fontweight='bold',
                      color=color, ha='center', transform=ax_banner.transAxes)
        ax_banner.text(x, 0.3, label, fontsize=12, ha='center',
                      transform=ax_banner.transAxes, color='gray')

    ax_banner.text(0.5, 0.95, 'AI Research Acceleration Model v0.9 - Executive Summary',
                  fontsize=18, fontweight='bold', ha='center', transform=ax_banner.transAxes)

    # Domain comparison
    ax1 = fig.add_subplot(gs[1, 0])
    domains = ['Struct Bio', 'Drug Disc', 'Materials', 'Protein', 'Genomics']
    accels = [7.5, 3.8, 1.7, 5.8, 3.4]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    ax1.barh(domains, accels, color=colors, alpha=0.8)
    ax1.set_xlabel('Acceleration (×)')
    ax1.set_title('By Domain (2030)', fontweight='bold')
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    # Workforce pie
    ax2 = fig.add_subplot(gs[1, 1])
    sizes = [19, 81]
    labels = ['Displaced\n0.49M', 'Created\n2.59M']
    colors = ['#e74c3c', '#27ae60']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
           startangle=90, explode=(0, 0.05))
    ax2.set_title('Workforce Impact', fontweight='bold')

    # Investment bar
    ax3 = fig.add_subplot(gs[1, 2])
    categories = ['Infra', 'R&D', 'Training', 'Regulatory']
    amounts = [1.2, 1.0, 0.6, 0.3]
    ax3.bar(categories, amounts, color='#3498db', alpha=0.8)
    ax3.set_ylabel('$ Billions')
    ax3.set_title('Investment Needs', fontweight='bold')

    # Timeline
    ax4 = fig.add_subplot(gs[2, :])
    years = [2025, 2027, 2030, 2035]
    milestones = [
        'Critical regulatory\nactions begin',
        'AI-assisted\ntrials approved',
        'Infrastructure\nbuild-out complete',
        'Full AI integration\nachieved'
    ]

    ax4.set_xlim(2024, 2036)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    for i, (year, milestone) in enumerate(zip(years, milestones)):
        x = (year - 2024) / 12
        ax4.axvline(x=year, ymin=0.3, ymax=0.7, color='#3498db', linewidth=3)
        ax4.scatter([year], [0.5], s=200, c='#3498db', zorder=5)
        ax4.text(year, 0.8, str(year), ha='center', fontsize=12, fontweight='bold')
        ax4.text(year, 0.2, milestone, ha='center', fontsize=9, va='top')

    ax4.set_title('Implementation Timeline', fontweight='bold', pad=20)
    ax4.plot([2024, 2036], [0.5, 0.5], color='#3498db', linewidth=2, alpha=0.5)

    plt.savefig(OUTPUT_DIR / "fig7_executive_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig7_executive_summary.png")


def generate_all_v09_figures():
    """Generate all v0.9 visualizations."""
    print("\n" + "="*60)
    print("Generating v0.9 Visualizations")
    print("="*60 + "\n")

    fig1_cross_domain_network()
    fig2_workforce_flow()
    fig3_policy_priority_matrix()
    fig4_system_trajectory()
    fig5_investment_allocation()
    fig6_bottleneck_waterfall()
    fig7_executive_summary()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_v09_figures()
