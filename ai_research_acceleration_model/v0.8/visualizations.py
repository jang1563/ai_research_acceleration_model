#!/usr/bin/env python3
"""
v0.8 Visualizations: Probabilistic Framework
=============================================

Visual communication strategy:
1. Uncertainty fans - show how predictions spread over time
2. Scenario comparison - side-by-side futures
3. Calibration plots - are we over/under confident?
4. Regulatory pathway decision tree - what drives drug discovery?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
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


def fig1_uncertainty_fans():
    """
    Figure 1: Uncertainty Fan Charts

    Shows how uncertainty grows over time for each domain.
    Key insight: Some domains have wider uncertainty than others.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    domains = [
        ("Structural Biology", 4.5, 0.4, "#2ecc71"),
        ("Drug Discovery", 1.4, 0.25, "#e74c3c"),
        ("Materials Science", 1.0, 0.5, "#3498db"),
        ("Protein Design", 2.5, 0.22, "#9b59b6"),
        ("Clinical Genomics", 2.0, 0.125, "#f39c12"),
    ]

    years = np.arange(2024, 2036)

    for idx, (name, base_accel, uncertainty, color) in enumerate(domains):
        ax = axes[idx]

        # Generate trajectories with growing uncertainty
        mean_traj = []
        ci_50_low, ci_50_high = [], []
        ci_90_low, ci_90_high = [], []

        for year in years:
            t = year - 2024
            # Mean grows toward ceiling
            mean = base_accel * (1 + 0.08 * t)
            mean_traj.append(mean)

            # Uncertainty grows over time
            width = mean * uncertainty * (1 + 0.1 * t)
            ci_50_low.append(mean - width * 0.5)
            ci_50_high.append(mean + width * 0.5)
            ci_90_low.append(mean - width)
            ci_90_high.append(mean + width)

        # Plot fans
        ax.fill_between(years, ci_90_low, ci_90_high, alpha=0.2, color=color, label='90% CI')
        ax.fill_between(years, ci_50_low, ci_50_high, alpha=0.4, color=color, label='50% CI')
        ax.plot(years, mean_traj, color=color, linewidth=2.5, label='Median')

        # Reference line at 1x
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Acceleration (×)')
        ax.set_xlim(2024, 2035)
        ax.set_ylim(0, max(ci_90_high) * 1.1)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)

    # Remove empty subplot
    axes[5].axis('off')

    # Add title
    fig.suptitle('Uncertainty Fan Charts: Acceleration Projections by Domain\n'
                 'Wider bands = higher uncertainty in projections',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_uncertainty_fans.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_uncertainty_fans.png")


def fig2_scenario_comparison():
    """
    Figure 2: Scenario Comparison Matrix

    Shows how different scenarios lead to different outcomes.
    Key insight: Even in optimistic scenarios, drug discovery is bounded.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    scenarios = ['Pessimistic\n(10%)', 'Conservative\n(20%)', 'Baseline\n(40%)',
                 'Optimistic\n(20%)', 'Breakthrough\n(10%)']
    domains = ['Structural\nBiology', 'Drug\nDiscovery', 'Materials\nScience',
               'Protein\nDesign', 'Clinical\nGenomics']

    # Acceleration values by scenario (2030)
    data = np.array([
        [2.0, 1.1, 0.8, 1.5, 1.3],   # Pessimistic
        [3.5, 1.2, 0.9, 2.0, 1.6],   # Conservative
        [4.5, 1.4, 1.0, 2.5, 2.0],   # Baseline
        [6.0, 1.6, 1.5, 3.5, 2.5],   # Optimistic
        [10.0, 2.5, 3.0, 6.0, 4.0],  # Breakthrough
    ])

    # Create heatmap
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=10)

    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(domains)):
            val = data[i, j]
            color = 'white' if val > 5 or val < 1.2 else 'black'
            ax.text(i, j, f'{val:.1f}×', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)

    # Labels
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=11)

    ax.set_xlabel('Scenario (probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Domain', fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Acceleration (×)', fontsize=11)

    # Title
    ax.set_title('Scenario Analysis: 2030 Acceleration by Domain\n'
                 'Drug Discovery remains bounded even in optimistic scenarios',
                 fontsize=14, fontweight='bold', pad=20)

    # Add annotation box
    textstr = 'Key Insight:\nDrug Discovery never exceeds 2.5×\ndue to clinical trial constraints'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.annotate(textstr, xy=(4, 1), xytext=(4.8, 2.5),
                fontsize=10, ha='left', va='center',
                bbox=props,
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_scenario_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_scenario_comparison.png")


def fig3_regulatory_pathways():
    """
    Figure 3: Regulatory Evolution Pathways

    Decision tree showing how regulatory scenarios unfold.
    Key insight: Transformative acceleration requires regulatory reform.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors for scenarios
    colors = {
        'status_quo': '#95a5a6',
        'incremental': '#3498db',
        'ai_assisted': '#2ecc71',
        'ai_primary': '#f39c12',
        'transformed': '#e74c3c',
    }

    # Draw timeline arrow
    ax.annotate('', xy=(95, 50), xytext=(5, 50),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(50, 45, 'Time →', ha='center', fontsize=12, color='gray')

    # Draw pathways
    pathways = [
        ('Status Quo', 30, 75, 85, 75, colors['status_quo'], '1.0×', '30%'),
        ('Incremental Reform', 30, 65, 85, 70, colors['incremental'], '1.2×', '35%'),
        ('AI-Assisted', 45, 55, 85, 60, colors['ai_assisted'], '1.5×', '20%'),
        ('AI-Primary', 55, 45, 85, 45, colors['ai_primary'], '2.5×', '10%'),
        ('Transformed', 70, 35, 85, 25, colors['transformed'], '8.0×', '5%'),
    ]

    for name, x1, y1, x2, y2, color, accel, prob in pathways:
        # Draw pathway line
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)

        # Add node at start
        ax.scatter([x1], [y1], s=150, c=color, zorder=5, edgecolors='white', linewidths=2)

        # Add endpoint box
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
        ax.text(x2+2, y2, f'{name}\n{accel} accel\nP={prob}',
               fontsize=9, va='center', bbox=bbox, color='white', fontweight='bold')

    # Year markers
    year_positions = [(15, 'Now'), (35, '2027'), (55, '2030'), (75, '2035'), (90, '2045')]
    for x, year in year_positions:
        ax.axvline(x=x, color='lightgray', linestyle='--', alpha=0.5)
        ax.text(x, 42, year, ha='center', fontsize=10, color='gray')

    # Decision points
    decisions = [
        (30, 70, 'Pilot\nPrograms'),
        (45, 55, 'AI Tools\nValidated'),
        (55, 45, 'Virtual\nTrials'),
        (70, 35, 'Paradigm\nShift'),
    ]

    for x, y, text in decisions:
        ax.annotate(text, xy=(x, y), xytext=(x-8, y+8),
                   fontsize=8, ha='center', va='bottom',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Title and key insight
    ax.text(50, 95, 'Regulatory Evolution Pathways for Drug Discovery',
           ha='center', fontsize=16, fontweight='bold')
    ax.text(50, 88, 'How regulatory reform unlocks (or constrains) AI acceleration',
           ha='center', fontsize=12, style='italic', color='gray')

    # Legend box with key insight
    insight_text = ('Key Insight:\n'
                   '• Status Quo (30%): No acceleration\n'
                   '• Incremental (35%): 20% faster by 2027\n'
                   '• AI-Assisted (20%): 50% faster by 2030\n'
                   '• AI-Primary (10%): 2.5× by 2035\n'
                   '• Transformed (5%): 8× requires paradigm shift')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(5, 25, insight_text, fontsize=9, va='top', bbox=props, family='monospace')

    plt.savefig(OUTPUT_DIR / "fig3_regulatory_pathways.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_regulatory_pathways.png")


def fig4_monte_carlo_distributions():
    """
    Figure 4: Monte Carlo Distribution Shapes

    Shows the actual distribution shape for each domain.
    Key insight: Some distributions are skewed (long tail of optimism).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    np.random.seed(42)

    domains = [
        ("Structural Biology", 4.5, 0.4, 1.31, "#2ecc71"),
        ("Drug Discovery", 1.4, 0.25, 0.64, "#e74c3c"),
        ("Materials Science", 1.0, 0.5, 2.69, "#3498db"),
        ("Protein Design", 2.5, 0.22, 0.40, "#9b59b6"),
        ("Clinical Genomics", 2.0, 0.125, 0.04, "#f39c12"),
    ]

    for idx, (name, mean, uncertainty, skewness, color) in enumerate(domains):
        ax = axes[idx]

        # Generate samples with appropriate skewness
        if skewness > 1:
            # Right-skewed (lognormal-like)
            samples = np.random.lognormal(np.log(mean) - 0.5*uncertainty**2, uncertainty, 10000)
        elif skewness > 0.3:
            # Moderately skewed
            samples = np.random.gamma(4, mean/4, 10000)
        else:
            # Nearly symmetric
            samples = np.random.normal(mean, mean * uncertainty, 10000)

        samples = np.clip(samples, 0.5, mean * 3)

        # Plot histogram
        n, bins, patches = ax.hist(samples, bins=50, density=True, alpha=0.7, color=color)

        # Add vertical lines for key statistics
        median = np.median(samples)
        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)

        ax.axvline(median, color='black', linewidth=2, label=f'Median: {median:.1f}×')
        ax.axvline(p5, color='gray', linewidth=1.5, linestyle='--', label=f'5th: {p5:.1f}×')
        ax.axvline(p95, color='gray', linewidth=1.5, linestyle='--', label=f'95th: {p95:.1f}×')

        ax.set_title(f'{name}\nSkewness: {skewness:.2f}', fontweight='bold')
        ax.set_xlabel('Acceleration (×)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)

        # Add skewness interpretation
        if skewness > 1:
            ax.text(0.95, 0.5, 'High\nupside\nuncertainty', transform=ax.transAxes,
                   ha='right', va='center', fontsize=9, style='italic', color='gray')
        elif skewness < 0.3:
            ax.text(0.95, 0.5, 'Symmetric\nuncertainty', transform=ax.transAxes,
                   ha='right', va='center', fontsize=9, style='italic', color='gray')

    # Remove empty subplot
    axes[5].axis('off')
    axes[5].text(0.5, 0.5, 'Monte Carlo:\n10,000 samples\nper domain\n\nShows full\nposterior\ndistribution',
                ha='center', va='center', fontsize=12, transform=axes[5].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Monte Carlo Posterior Distributions (2030)\n'
                 'Shape reveals nature of uncertainty - skewed = optimistic tail',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_monte_carlo_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_monte_carlo_distributions.png")


def fig5_calibration_target():
    """
    Figure 5: Calibration Target Plot

    Shows whether predictions are well-calibrated.
    Key insight: Model should be neither over nor under confident.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw target circles
    radii = [0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Perfect', 'Excellent', 'Good', 'Fair', 'Poor']
    colors = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c']

    for r, label, color in zip(radii, labels, colors):
        circle = plt.Circle((0.5, 0.5), r/2, fill=False, color=color, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.text(0.5 + r/2 * 0.7, 0.5 + r/2 * 0.7, label, fontsize=9, color=color)

    # Plot domain calibration points
    # (x, y) where x = expected coverage, y = observed coverage
    domains = [
        ("Structural Biology", 0.88, 0.85, "#2ecc71"),  # Slightly overconfident
        ("Drug Discovery", 0.90, 0.92, "#e74c3c"),      # Well calibrated
        ("Materials Science", 0.85, 0.70, "#3498db"),   # Overconfident
        ("Protein Design", 0.90, 0.88, "#9b59b6"),      # Well calibrated
        ("Clinical Genomics", 0.92, 0.95, "#f39c12"),   # Slightly underconfident
    ]

    for name, expected, observed, color in domains:
        # Convert to position (center is perfect calibration)
        error = abs(expected - observed)
        x = 0.5 + (observed - expected) * 2
        y = 0.5 + np.random.uniform(-0.1, 0.1)  # Small vertical jitter

        ax.scatter([x], [y], s=300, c=color, zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(name, (x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Add perfect calibration line
    ax.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)

    # Labels for axes interpretation
    ax.text(0.1, 0.5, '← Overconfident', va='center', fontsize=11, color='orange', fontweight='bold')
    ax.text(0.9, 0.5, 'Underconfident →', va='center', ha='right', fontsize=11, color='blue', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Model Calibration Assessment\n'
                 'Center = perfect calibration | Distance from center = calibration error',
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_text = ('Interpretation:\n'
                  '• Center: 90% CI contains 90% of outcomes\n'
                  '• Left: Too narrow CIs (overconfident)\n'
                  '• Right: Too wide CIs (underconfident)\n\n'
                  'Materials Science needs wider CIs')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.02, legend_text, fontsize=9, va='bottom', bbox=props, transform=ax.transAxes)

    plt.savefig(OUTPUT_DIR / "fig5_calibration_target.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_calibration_target.png")


def fig6_ensemble_forecast():
    """
    Figure 6: Probability-Weighted Ensemble Forecast

    Shows how scenarios combine into final forecast.
    Key insight: Ensemble averages across futures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Scenario contributions
    ax1 = axes[0]

    scenarios = ['Pessimistic', 'Conservative', 'Baseline', 'Optimistic', 'Breakthrough']
    probabilities = [0.10, 0.20, 0.40, 0.20, 0.10]
    accelerations = [1.1, 1.2, 1.4, 1.6, 2.5]  # Drug discovery
    contributions = [p * a for p, a in zip(probabilities, accelerations)]

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    # Stacked bar showing contributions
    bottom = 0
    for i, (scenario, prob, accel, contrib, color) in enumerate(zip(scenarios, probabilities, accelerations, contributions, colors)):
        ax1.bar(['Drug Discovery\nEnsemble'], [contrib], bottom=bottom, color=color,
               label=f'{scenario} ({prob:.0%}): {accel:.1f}× → +{contrib:.2f}', edgecolor='white')
        bottom += contrib

    ax1.axhline(y=sum(contributions), color='black', linestyle='--', linewidth=2)
    ax1.text(0.15, sum(contributions) + 0.05, f'Ensemble: {sum(contributions):.2f}×',
            fontsize=12, fontweight='bold')

    ax1.set_ylabel('Acceleration Contribution (×)', fontsize=12)
    ax1.set_title('Ensemble = Σ(probability × acceleration)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 2.0)

    # Right panel: Full distribution comparison
    ax2 = axes[1]

    x = np.linspace(0.5, 4, 100)

    # Individual scenario distributions (narrower)
    for scenario, prob, accel, color in zip(scenarios, probabilities, accelerations, colors):
        y = prob * np.exp(-0.5 * ((x - accel) / 0.2) ** 2)
        ax2.fill_between(x, y, alpha=0.3, color=color, label=scenario)
        ax2.plot(x, y, color=color, linewidth=1)

    # Ensemble distribution (wider, mixture)
    ensemble_mean = sum(p * a for p, a in zip(probabilities, accelerations))
    ensemble_y = np.zeros_like(x)
    for prob, accel in zip(probabilities, accelerations):
        ensemble_y += prob * np.exp(-0.5 * ((x - accel) / 0.3) ** 2)
    ax2.plot(x, ensemble_y, color='black', linewidth=3, label='Ensemble')

    ax2.axvline(x=ensemble_mean, color='black', linestyle='--', linewidth=2)
    ax2.text(ensemble_mean + 0.1, max(ensemble_y) * 0.9, f'Mean: {ensemble_mean:.2f}×',
            fontsize=11, fontweight='bold')

    ax2.set_xlabel('Acceleration (×)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Scenario Distributions → Ensemble', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)

    fig.suptitle('Probability-Weighted Ensemble Forecasting\n'
                 'Final forecast combines all scenarios weighted by probability',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_ensemble_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig6_ensemble_forecast.png")


def generate_all_v08_figures():
    """Generate all v0.8 visualizations."""
    print("\n" + "="*60)
    print("Generating v0.8 Visualizations")
    print("="*60 + "\n")

    fig1_uncertainty_fans()
    fig2_scenario_comparison()
    fig3_regulatory_pathways()
    fig4_monte_carlo_distributions()
    fig5_calibration_target()
    fig6_ensemble_forecast()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_v08_figures()
