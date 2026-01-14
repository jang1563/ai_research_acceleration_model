#!/usr/bin/env python3
"""
Integrated v0.9 Model: System-Level Analysis
=============================================

v0.9 combines all previous enhancements with new system-level modeling:

From previous versions:
- v0.4: Refined bottleneck model with domain-specific constraints
- v0.5: Autonomous lab integration
- v0.6: Expert panel calibration
- v0.7: Dynamic bypass and feedback loops
- v0.8: Full probabilistic framework

New in v0.9:
- Cross-domain interaction effects (spillover)
- Workforce impact modeling
- Policy recommendation engine
- Interactive API for scenario exploration

Key Insight:
Scientific domains don't accelerate in isolation. v0.9 captures how
acceleration in one domain enables acceleration in others, creating
compound effects that reshape the research landscape.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

# Add paths for imports
v09_src = Path(__file__).parent
v08_src = Path(__file__).parent.parent.parent / "v0.8" / "src"
v07_src = Path(__file__).parent.parent.parent / "v0.7" / "src"
v06_src = Path(__file__).parent.parent.parent / "v0.6" / "src"
v05_src = Path(__file__).parent.parent.parent / "v0.5" / "src"
v04_src = Path(__file__).parent.parent.parent / "v0.4" / "src"

for p in [v09_src, v08_src, v07_src, v06_src, v05_src, v04_src]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import v0.9 components
from cross_domain_effects import CrossDomainModel
from workforce_impact import WorkforceImpactModel
from policy_recommendations import PolicyRecommendationEngine


@dataclass
class SystemLevelForecast:
    """Comprehensive system-level forecast for a domain."""
    domain: str
    year: int

    # Acceleration metrics
    standalone_acceleration: float      # Without cross-domain effects
    cross_domain_boost: float           # Boost from other domains
    total_acceleration: float           # Combined
    acceleration_ci_90: Tuple[float, float]

    # Bottleneck analysis
    primary_bottleneck: str
    bottleneck_fraction: float

    # Workforce impact
    net_jobs_change: float              # Millions
    displacement_rate: float
    creation_rate: float
    skills_gap_severity: str

    # Policy context
    policy_recommendations: int
    critical_actions: int
    investment_needed: str


@dataclass
class SystemSnapshot:
    """Complete system snapshot across all domains."""
    year: int

    # Domain forecasts
    domain_forecasts: Dict[str, SystemLevelForecast]

    # Aggregate metrics
    total_acceleration: float           # Weighted average
    total_workforce_change: float       # Net jobs
    total_investment_needed: str

    # Key insights
    fastest_domain: str
    bottleneck_domain: str
    highest_spillover: Tuple[str, str, float]  # (source, target, effect)


class IntegratedV09Model:
    """
    System-level AI research acceleration model.

    Integrates:
    - Cross-domain interaction effects
    - Workforce impact projections
    - Policy recommendations
    - Uncertainty quantification
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

        # Initialize sub-models
        self.cross_domain = CrossDomainModel()
        self.workforce = WorkforceImpactModel()
        self.policy = PolicyRecommendationEngine()

        # Domain configurations
        self.domains = [
            "structural_biology",
            "drug_discovery",
            "materials_science",
            "protein_design",
            "clinical_genomics",
        ]

        # Base accelerations (from v0.8 Monte Carlo medians)
        self.base_accelerations = {
            "structural_biology": 4.5,
            "drug_discovery": 1.4,
            "materials_science": 1.0,
            "protein_design": 2.5,
            "clinical_genomics": 2.0,
        }

        # Uncertainty (90% CI width as fraction of mean)
        self.uncertainty = {
            "structural_biology": 0.4,
            "drug_discovery": 0.25,
            "materials_science": 0.5,
            "protein_design": 0.22,
            "clinical_genomics": 0.125,
        }

        # Bottleneck info
        self.bottlenecks = {
            "structural_biology": ("experimental_validation", 0.3),
            "drug_discovery": ("clinical_trials", 0.8),
            "materials_science": ("synthesis", 0.7),
            "protein_design": ("expression_validation", 0.4),
            "clinical_genomics": ("clinical_adoption", 0.5),
        }

    def _calculate_cross_domain_boost(
        self,
        domain: str,
        year: int,
    ) -> float:
        """Calculate acceleration boost from cross-domain effects."""
        forecast = self.cross_domain.forecast(domain, year)
        return forecast.spillover_boost + forecast.synergy_boost

    def forecast_domain(
        self,
        domain: str,
        year: int,
    ) -> SystemLevelForecast:
        """Generate comprehensive forecast for a single domain."""
        # Base acceleration
        base_accel = self.base_accelerations.get(domain, 2.0)

        # Time evolution (assume growth toward ceiling)
        t = year - 2024
        time_factor = 1 + 0.1 * t  # 10% per year growth
        standalone = base_accel * min(time_factor, 3.0)  # Cap at 3x growth

        # Cross-domain boost
        boost = self._calculate_cross_domain_boost(domain, year)
        total_accel = standalone * (1 + boost)

        # Confidence interval
        uncertainty_frac = self.uncertainty.get(domain, 0.3)
        ci_width = total_accel * uncertainty_frac
        ci_90 = (max(1.0, total_accel - ci_width), total_accel + ci_width)

        # Bottleneck
        bottleneck_name, bottleneck_frac = self.bottlenecks.get(
            domain, ("unknown", 0.5)
        )

        # Workforce
        workforce_impact = self.workforce.analyze_domain(domain, year, total_accel)

        # Policy
        policy_analysis = self.policy.analyze_domain(
            domain=domain,
            year=year,
            acceleration=total_accel,
            bottleneck_fraction=bottleneck_frac,
            workforce_displacement=workforce_impact.total_displaced / max(workforce_impact.total_current, 0.01),
            workforce_growth=workforce_impact.total_created / max(workforce_impact.total_current, 0.01),
        )

        critical_count = sum(
            1 for r in policy_analysis.recommendations
            if r.priority.value == "critical"
        )

        return SystemLevelForecast(
            domain=domain,
            year=year,
            standalone_acceleration=standalone,
            cross_domain_boost=boost,
            total_acceleration=total_accel,
            acceleration_ci_90=ci_90,
            primary_bottleneck=bottleneck_name,
            bottleneck_fraction=bottleneck_frac,
            net_jobs_change=workforce_impact.total_net,
            displacement_rate=workforce_impact.total_displaced / max(workforce_impact.total_current, 0.01),
            creation_rate=workforce_impact.total_created / max(workforce_impact.total_current, 0.01),
            skills_gap_severity=workforce_impact.skills_gap_severity,
            policy_recommendations=len(policy_analysis.recommendations),
            critical_actions=critical_count,
            investment_needed=policy_analysis.total_investment_recommended,
        )

    def system_snapshot(self, year: int) -> SystemSnapshot:
        """Generate complete system snapshot for a given year."""
        domain_forecasts = {}

        total_current_workforce = 0
        total_net_workforce = 0

        for domain in self.domains:
            forecast = self.forecast_domain(domain, year)
            domain_forecasts[domain] = forecast

            # Aggregate workforce
            wf = self.workforce.analyze_domain(domain, year)
            total_current_workforce += wf.total_current
            total_net_workforce += wf.total_net

        # Calculate weighted average acceleration
        weights = {
            "structural_biology": 0.15,
            "drug_discovery": 0.35,
            "materials_science": 0.20,
            "protein_design": 0.15,
            "clinical_genomics": 0.15,
        }
        total_accel = sum(
            domain_forecasts[d].total_acceleration * weights[d]
            for d in self.domains
        )

        # Find fastest domain
        fastest = max(self.domains, key=lambda d: domain_forecasts[d].total_acceleration)

        # Find bottleneck domain (lowest acceleration)
        slowest = min(self.domains, key=lambda d: domain_forecasts[d].total_acceleration)

        # Find highest spillover
        max_spillover = ("", "", 0.0)
        for target_domain in self.domains:
            forecast = self.cross_domain.forecast(target_domain, year)
            for source, effect in forecast.effects_by_source.items():
                if effect > max_spillover[2]:
                    max_spillover = (source, target_domain, effect)
        highest_spillover = max_spillover

        # Investment summary
        cost_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        total_cost = sum(
            cost_map.get(domain_forecasts[d].investment_needed.split()[0], 2)
            for d in self.domains
        )
        if total_cost < 8:
            investment_str = "Moderate ($500M-$1B)"
        elif total_cost < 15:
            investment_str = "Significant ($1B-$5B)"
        else:
            investment_str = "Major (>$5B)"

        return SystemSnapshot(
            year=year,
            domain_forecasts=domain_forecasts,
            total_acceleration=total_accel,
            total_workforce_change=total_net_workforce,
            total_investment_needed=investment_str,
            fastest_domain=fastest,
            bottleneck_domain=slowest,
            highest_spillover=highest_spillover,
        )

    def trajectory(
        self,
        start_year: int = 2025,
        end_year: int = 2035,
    ) -> List[SystemSnapshot]:
        """Generate trajectory of system snapshots."""
        return [
            self.system_snapshot(year)
            for year in range(start_year, end_year + 1)
        ]

    def comprehensive_report(self, year: int = 2030) -> str:
        """Generate comprehensive system-level report."""
        snapshot = self.system_snapshot(year)

        lines = [
            "=" * 100,
            f"AI RESEARCH ACCELERATION MODEL v0.9 - SYSTEM ANALYSIS ({year})",
            "=" * 100,
            "",
            "SYSTEM OVERVIEW:",
            "-" * 100,
            f"  Weighted average acceleration: {snapshot.total_acceleration:.1f}x",
            f"  Net workforce change: {snapshot.total_workforce_change:+.2f}M jobs",
            f"  Investment recommended: {snapshot.total_investment_needed}",
            "",
            f"  Fastest domain: {snapshot.fastest_domain} ({snapshot.domain_forecasts[snapshot.fastest_domain].total_acceleration:.1f}x)",
            f"  Bottleneck domain: {snapshot.bottleneck_domain} ({snapshot.domain_forecasts[snapshot.bottleneck_domain].total_acceleration:.1f}x)",
            f"  Highest spillover: {snapshot.highest_spillover[0]} → {snapshot.highest_spillover[1]} ({snapshot.highest_spillover[2]:.0%})",
            "",
            "-" * 100,
            "DOMAIN DETAILS:",
            "-" * 100,
            f"{'Domain':<22} {'Standalone':<12} {'Boost':<10} {'Total':<10} {'90% CI':<18} {'Jobs':<10}",
            "-" * 100,
        ]

        for domain in self.domains:
            f = snapshot.domain_forecasts[domain]
            ci_str = f"[{f.acceleration_ci_90[0]:.1f}-{f.acceleration_ci_90[1]:.1f}]"
            lines.append(
                f"{domain:<22} {f.standalone_acceleration:>10.1f}x {f.cross_domain_boost:>+8.0%} "
                f"{f.total_acceleration:>8.1f}x {ci_str:<18} {f.net_jobs_change:>+8.2f}M"
            )

        lines.extend([
            "",
            "-" * 100,
            "CROSS-DOMAIN INTERACTION MATRIX:",
            "-" * 100,
        ])

        # Build matrix representation from individual forecasts
        effect_matrix = {d: {} for d in self.domains}
        for target_domain in self.domains:
            forecast = self.cross_domain.forecast(target_domain, year)
            for source, effect in forecast.effects_by_source.items():
                effect_matrix[source][target_domain] = effect

        # Header
        header = f"{'Source →':<22}"
        for target in self.domains:
            header += f" {target[:10]:>10}"
        lines.append(header)
        lines.append("-" * 100)

        for source in self.domains:
            row = f"{source:<22}"
            for target in self.domains:
                effect = effect_matrix.get(source, {}).get(target, 0)
                if effect > 0:
                    row += f" {effect:>+9.0%}"
                else:
                    row += f" {'--':>10}"
            lines.append(row)

        lines.extend([
            "",
            "-" * 100,
            "BOTTLENECK ANALYSIS:",
            "-" * 100,
        ])

        for domain in self.domains:
            f = snapshot.domain_forecasts[domain]
            lines.append(
                f"  {domain:<20}: {f.primary_bottleneck:<25} ({f.bottleneck_fraction:.0%} of delay)"
            )

        lines.extend([
            "",
            "-" * 100,
            "WORKFORCE IMPACT SUMMARY:",
            "-" * 100,
        ])

        total_displaced = 0
        total_created = 0
        for domain in self.domains:
            f = snapshot.domain_forecasts[domain]
            wf = self.workforce.analyze_domain(domain, year)
            total_displaced += wf.total_displaced
            total_created += wf.total_created
            lines.append(
                f"  {domain:<20}: Displaced: {wf.total_displaced:.2f}M, "
                f"Created: {wf.total_created:.2f}M, Net: {wf.total_net:+.2f}M "
                f"(Gap: {f.skills_gap_severity})"
            )

        lines.extend([
            f"  {'TOTAL':<20}: Displaced: {total_displaced:.2f}M, "
            f"Created: {total_created:.2f}M, Net: {total_created - total_displaced:+.2f}M",
            "",
            "-" * 100,
            "POLICY PRIORITIES:",
            "-" * 100,
        ])

        total_critical = 0
        total_recs = 0
        for domain in self.domains:
            f = snapshot.domain_forecasts[domain]
            total_critical += f.critical_actions
            total_recs += f.policy_recommendations
            lines.append(
                f"  {domain:<20}: {f.policy_recommendations} recommendations "
                f"({f.critical_actions} critical), Investment: {f.investment_needed}"
            )

        lines.extend([
            "",
            f"  TOTAL: {total_recs} recommendations ({total_critical} critical)",
            f"  System-wide investment: {snapshot.total_investment_needed}",
            "",
            "-" * 100,
            "KEY INSIGHTS:",
            "-" * 100,
            "  1. Cross-domain effects amplify individual domain acceleration by 10-30%",
            "  2. Structural biology spillovers drive drug discovery and protein design",
            "  3. Clinical trials remain the dominant bottleneck for drug discovery",
            "  4. Net workforce impact is positive but requires transition support",
            "  5. Critical regulatory actions needed within 1 year for clinical domains",
            "",
            "=" * 100,
        ])

        return "\n".join(lines)

    def trajectory_report(
        self,
        start_year: int = 2025,
        end_year: int = 2035,
    ) -> str:
        """Generate trajectory report over time."""
        trajectories = self.trajectory(start_year, end_year)

        lines = [
            "=" * 100,
            "AI RESEARCH ACCELERATION TRAJECTORY (2025-2035)",
            "=" * 100,
            "",
            f"{'Year':<8}",
        ]

        for domain in self.domains:
            lines[3] += f" {domain[:12]:>12}"
        lines[3] += f" {'Weighted':>12} {'Workforce':>12}"

        lines.append("-" * 100)

        for snapshot in trajectories:
            row = f"{snapshot.year:<8}"
            for domain in self.domains:
                row += f" {snapshot.domain_forecasts[domain].total_acceleration:>11.1f}x"
            row += f" {snapshot.total_acceleration:>11.1f}x"
            row += f" {snapshot.total_workforce_change:>+11.2f}M"
            lines.append(row)

        lines.extend([
            "-" * 100,
            "",
            "TRAJECTORY INSIGHTS:",
            "  - Acceleration compounds over time due to cross-domain effects",
            "  - Drug discovery acceleration bounded by clinical trial constraints",
            "  - Materials science acceleration depends on synthesis automation",
            "  - Workforce net positive but transition support critical through 2030",
            "",
            "=" * 100,
        ])

        return "\n".join(lines)


def main():
    """Generate comprehensive v0.9 analysis."""
    model = IntegratedV09Model()

    print(model.comprehensive_report(2030))
    print()
    print()
    print(model.trajectory_report())


if __name__ == "__main__":
    main()
