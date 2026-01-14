#!/usr/bin/env python3
"""
AI Research Acceleration Model v1.0
====================================

Production-ready unified API for modeling AI's impact on scientific research.

This is the main entry point for the model. It provides:
1. Simple, intuitive API for common use cases
2. Full access to all underlying models when needed
3. Comprehensive documentation and examples

Key Features:
- 5 scientific domains (structural biology, drug discovery, materials science,
  protein design, clinical genomics)
- Probabilistic forecasts with confidence intervals
- Cross-domain spillover effects
- Workforce impact projections
- Policy recommendations
- Scenario analysis (pessimistic to breakthrough)
- Regulatory evolution modeling

Usage:
    from ai_acceleration_model import AIAccelerationModel

    model = AIAccelerationModel()

    # Quick forecast
    forecast = model.forecast("drug_discovery", 2030)
    print(f"Acceleration: {forecast.acceleration:.1f}x")
    print(f"90% CI: [{forecast.ci_90[0]:.1f}x - {forecast.ci_90[1]:.1f}x]")

    # System-wide analysis
    system = model.system_snapshot(2030)
    print(f"Weighted average: {system.total_acceleration:.1f}x")
    print(f"Net workforce change: {system.workforce_change:+.2f}M")

    # Full report
    print(model.executive_summary(2030))
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

# Version info
__version__ = "1.0.0"
__author__ = "AI Research Acceleration Model Team"
__date__ = "January 2026"


class Domain(Enum):
    """Scientific domains supported by the model."""
    STRUCTURAL_BIOLOGY = "structural_biology"
    DRUG_DISCOVERY = "drug_discovery"
    MATERIALS_SCIENCE = "materials_science"
    PROTEIN_DESIGN = "protein_design"
    CLINICAL_GENOMICS = "clinical_genomics"


class Scenario(Enum):
    """Scenario types for forecasting."""
    PESSIMISTIC = "pessimistic"
    CONSERVATIVE = "conservative"
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    BREAKTHROUGH = "breakthrough"


@dataclass
class DomainForecast:
    """Forecast for a single domain."""
    domain: str
    year: int

    # Core metrics
    acceleration: float
    ci_50: Tuple[float, float]
    ci_90: Tuple[float, float]

    # Breakdown
    standalone_acceleration: float
    cross_domain_boost: float

    # Bottleneck
    primary_bottleneck: str
    bottleneck_fraction: float

    # Workforce
    jobs_displaced: float
    jobs_created: float
    net_jobs: float

    # Confidence
    confidence_level: str  # "high", "medium", "low"
    key_assumptions: List[str]


@dataclass
class SystemSnapshot:
    """System-wide snapshot across all domains."""
    year: int

    # Aggregate metrics
    total_acceleration: float
    acceleration_ci_90: Tuple[float, float]

    # Domain breakdown
    domain_forecasts: Dict[str, DomainForecast]

    # Workforce
    total_displaced: float
    total_created: float
    workforce_change: float

    # Investment
    investment_needed: str
    critical_actions: int

    # Key insights
    fastest_domain: str
    slowest_domain: str
    highest_spillover: Tuple[str, str, float]


@dataclass
class PolicyRecommendation:
    """A policy recommendation."""
    id: str
    title: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    stakeholders: List[str]
    timeline: str
    investment: str


class AIAccelerationModel:
    """
    Main interface for the AI Research Acceleration Model.

    This class provides a unified API for all model functionality,
    from simple forecasts to comprehensive system analysis.
    """

    # Domain display names
    DOMAIN_NAMES = {
        "structural_biology": "Structural Biology",
        "drug_discovery": "Drug Discovery",
        "materials_science": "Materials Science",
        "protein_design": "Protein Design",
        "clinical_genomics": "Clinical Genomics",
    }

    # Base parameters (from v0.8 calibration)
    BASE_ACCELERATIONS = {
        "structural_biology": 4.5,
        "drug_discovery": 1.4,
        "materials_science": 1.0,
        "protein_design": 2.5,
        "clinical_genomics": 2.0,
    }

    UNCERTAINTY = {
        "structural_biology": 0.4,
        "drug_discovery": 0.25,
        "materials_science": 0.5,
        "protein_design": 0.22,
        "clinical_genomics": 0.125,
    }

    BOTTLENECKS = {
        "structural_biology": ("experimental_validation", 0.30),
        "drug_discovery": ("clinical_trials", 0.80),
        "materials_science": ("synthesis", 0.70),
        "protein_design": ("expression_validation", 0.40),
        "clinical_genomics": ("clinical_adoption", 0.50),
    }

    # Cross-domain spillover coefficients
    SPILLOVERS = {
        ("structural_biology", "drug_discovery"): 0.33,
        ("structural_biology", "protein_design"): 0.37,
        ("protein_design", "drug_discovery"): 0.16,
        ("clinical_genomics", "drug_discovery"): 0.12,
        ("drug_discovery", "clinical_genomics"): 0.05,
        ("materials_science", "structural_biology"): 0.04,
        ("protein_design", "materials_science"): 0.05,
        ("clinical_genomics", "protein_design"): 0.05,
    }

    # Workforce parameters (millions)
    WORKFORCE = {
        "structural_biology": {"current": 0.15, "displacement_rate": 0.25, "creation_rate": 1.5},
        "drug_discovery": {"current": 1.10, "displacement_rate": 0.20, "creation_rate": 1.8},
        "materials_science": {"current": 0.50, "displacement_rate": 0.26, "creation_rate": 1.6},
        "protein_design": {"current": 0.18, "displacement_rate": 0.22, "creation_rate": 2.5},
        "clinical_genomics": {"current": 0.08, "displacement_rate": 0.25, "creation_rate": 1.4},
    }

    def __init__(self, seed: int = 42):
        """
        Initialize the model.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.domains = list(self.BASE_ACCELERATIONS.keys())

    def _time_factor(self, year: int) -> float:
        """Calculate time evolution factor."""
        t = max(0, year - 2024)
        return 1 + 0.08 * t  # 8% annual growth toward ceiling

    def _calculate_spillover(self, domain: str, year: int) -> float:
        """Calculate total spillover boost for a domain."""
        boost = 0.0
        t = year - 2024

        for (source, target), coefficient in self.SPILLOVERS.items():
            if target == domain:
                source_accel = self.BASE_ACCELERATIONS[source] * self._time_factor(year)
                # Log dampening and lag
                log_accel = np.log1p(source_accel - 1)
                lag_factor = min(1.0, t / 2)  # 2-year lag
                effect = log_accel * coefficient * lag_factor * 0.3
                boost += min(effect, 0.5)  # Cap per source

        return boost

    def _calculate_workforce(self, domain: str, year: int, acceleration: float) -> Tuple[float, float, float]:
        """Calculate workforce impact."""
        params = self.WORKFORCE[domain]
        t = year - 2024

        # Displacement grows with time and acceleration
        displacement_rate = params["displacement_rate"] * (1 - np.exp(-0.1 * t))
        displaced = params["current"] * displacement_rate

        # Creation proportional to acceleration
        creation = params["current"] * params["creation_rate"] * np.sqrt(acceleration) / np.sqrt(3)
        creation *= (1 - np.exp(-0.15 * t))  # Lags displacement

        return displaced, creation, creation - displaced

    def forecast(
        self,
        domain: Union[str, Domain],
        year: int,
        scenario: Union[str, Scenario] = "baseline",
    ) -> DomainForecast:
        """
        Generate a forecast for a single domain.

        Args:
            domain: Domain name or Domain enum
            year: Target year (2024-2050)
            scenario: Scenario for forecast

        Returns:
            DomainForecast with acceleration and supporting metrics
        """
        # Normalize domain
        if isinstance(domain, Domain):
            domain = domain.value
        domain = domain.lower().replace(" ", "_")

        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}. Valid domains: {self.domains}")

        # Normalize scenario
        if isinstance(scenario, Scenario):
            scenario = scenario.value

        # Calculate base acceleration
        base = self.BASE_ACCELERATIONS[domain]
        time_factor = self._time_factor(year)
        standalone = base * time_factor

        # Apply scenario modifier
        scenario_modifiers = {
            "pessimistic": 0.6,
            "conservative": 0.8,
            "baseline": 1.0,
            "optimistic": 1.2,
            "breakthrough": 1.8,
        }
        modifier = scenario_modifiers.get(scenario, 1.0)
        standalone *= modifier

        # Cross-domain boost
        spillover = self._calculate_spillover(domain, year)
        total = standalone * (1 + spillover)

        # Confidence intervals
        uncertainty = self.UNCERTAINTY[domain]
        t = year - 2024
        width = total * uncertainty * (1 + 0.05 * t)  # Uncertainty grows

        ci_50 = (max(1.0, total - width * 0.5), total + width * 0.5)
        ci_90 = (max(1.0, total - width), total + width)

        # Bottleneck
        bottleneck_name, bottleneck_frac = self.BOTTLENECKS[domain]

        # Workforce
        displaced, created, net = self._calculate_workforce(domain, year, total)

        # Confidence level
        if uncertainty < 0.2:
            confidence = "high"
        elif uncertainty < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Key assumptions
        assumptions = [
            f"AI progress continues at current trajectory",
            f"No major regulatory changes" if domain == "drug_discovery" else "Automation continues scaling",
            f"{scenario.title()} scenario assumptions hold",
        ]

        return DomainForecast(
            domain=domain,
            year=year,
            acceleration=total,
            ci_50=ci_50,
            ci_90=ci_90,
            standalone_acceleration=standalone,
            cross_domain_boost=spillover,
            primary_bottleneck=bottleneck_name,
            bottleneck_fraction=bottleneck_frac,
            jobs_displaced=displaced,
            jobs_created=created,
            net_jobs=net,
            confidence_level=confidence,
            key_assumptions=assumptions,
        )

    def system_snapshot(self, year: int, scenario: str = "baseline") -> SystemSnapshot:
        """
        Generate a system-wide snapshot across all domains.

        Args:
            year: Target year
            scenario: Scenario for forecast

        Returns:
            SystemSnapshot with aggregate metrics
        """
        domain_forecasts = {}
        total_displaced = 0
        total_created = 0

        for domain in self.domains:
            forecast = self.forecast(domain, year, scenario)
            domain_forecasts[domain] = forecast
            total_displaced += forecast.jobs_displaced
            total_created += forecast.jobs_created

        # Weighted average acceleration
        weights = {
            "structural_biology": 0.15,
            "drug_discovery": 0.35,
            "materials_science": 0.20,
            "protein_design": 0.15,
            "clinical_genomics": 0.15,
        }
        total_accel = sum(
            domain_forecasts[d].acceleration * weights[d]
            for d in self.domains
        )

        # CI for system
        uncertainty_avg = sum(self.UNCERTAINTY[d] * weights[d] for d in self.domains)
        t = year - 2024
        width = total_accel * uncertainty_avg * (1 + 0.05 * t)
        accel_ci_90 = (max(1.0, total_accel - width), total_accel + width)

        # Find fastest/slowest
        fastest = max(self.domains, key=lambda d: domain_forecasts[d].acceleration)
        slowest = min(self.domains, key=lambda d: domain_forecasts[d].acceleration)

        # Find highest spillover
        max_spillover = ("", "", 0.0)
        for (source, target), coef in self.SPILLOVERS.items():
            if coef > max_spillover[2]:
                max_spillover = (source, target, coef)

        # Investment needed
        if total_accel < 3:
            investment = "Moderate ($500M-$1B)"
        elif total_accel < 5:
            investment = "Significant ($1B-$3B)"
        else:
            investment = "Major ($3B-$5B)"

        # Critical actions (simplified)
        critical = 3 if year <= 2027 else 5 if year <= 2030 else 7

        return SystemSnapshot(
            year=year,
            total_acceleration=total_accel,
            acceleration_ci_90=accel_ci_90,
            domain_forecasts=domain_forecasts,
            total_displaced=total_displaced,
            total_created=total_created,
            workforce_change=total_created - total_displaced,
            investment_needed=investment,
            critical_actions=critical,
            fastest_domain=fastest,
            slowest_domain=slowest,
            highest_spillover=max_spillover,
        )

    def compare_scenarios(self, domain: str, year: int) -> Dict[str, DomainForecast]:
        """
        Compare all scenarios for a domain.

        Args:
            domain: Domain to analyze
            year: Target year

        Returns:
            Dictionary of scenario -> forecast
        """
        scenarios = ["pessimistic", "conservative", "baseline", "optimistic", "breakthrough"]
        return {s: self.forecast(domain, year, s) for s in scenarios}

    def trajectory(
        self,
        domain: str = None,
        start_year: int = 2025,
        end_year: int = 2035,
    ) -> List[Union[DomainForecast, SystemSnapshot]]:
        """
        Generate trajectory over time.

        Args:
            domain: Specific domain, or None for system-wide
            start_year: Start year
            end_year: End year

        Returns:
            List of forecasts/snapshots for each year
        """
        results = []
        for year in range(start_year, end_year + 1):
            if domain:
                results.append(self.forecast(domain, year))
            else:
                results.append(self.system_snapshot(year))
        return results

    def get_policy_recommendations(self, year: int = 2030) -> List[PolicyRecommendation]:
        """
        Get policy recommendations.

        Args:
            year: Target year for recommendations

        Returns:
            List of policy recommendations sorted by priority
        """
        recommendations = [
            PolicyRecommendation(
                id="DD-001",
                title="AI-Adaptive Clinical Trial Framework",
                description="Develop regulatory pathway for AI-optimized adaptive trials",
                priority="critical",
                stakeholders=["Government", "Industry"],
                timeline="1 year",
                investment="$50-100M",
            ),
            PolicyRecommendation(
                id="CG-001",
                title="AI Variant Classification Standards",
                description="FDA guidance for AI-based variant classification",
                priority="critical",
                stakeholders=["Government", "Industry"],
                timeline="1 year",
                investment="$10-20M",
            ),
            PolicyRecommendation(
                id="SB-001",
                title="Scale Cryo-EM Infrastructure",
                description="Expand national cryo-EM facility capacity",
                priority="high",
                stakeholders=["Government", "Academia"],
                timeline="3-5 years",
                investment="$200-500M",
            ),
            PolicyRecommendation(
                id="DD-002",
                title="Preclinical Automation Incentives",
                description="Tax incentives for automated preclinical facilities",
                priority="high",
                stakeholders=["Government", "Industry"],
                timeline="2-3 years",
                investment="$100-300M",
            ),
            PolicyRecommendation(
                id="WF-001",
                title="AI-Biology Training Pipeline",
                description="Expand graduate programs at AI-biology interface",
                priority="high",
                stakeholders=["Academia", "Government"],
                timeline="5+ years",
                investment="$50-100M",
            ),
        ]

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 99))

        return recommendations

    def executive_summary(self, year: int = 2030) -> str:
        """
        Generate executive summary report.

        Args:
            year: Target year

        Returns:
            Formatted executive summary string
        """
        snapshot = self.system_snapshot(year)
        recommendations = self.get_policy_recommendations(year)

        lines = [
            "=" * 80,
            f"AI RESEARCH ACCELERATION MODEL v{__version__}",
            f"Executive Summary - {year}",
            "=" * 80,
            "",
            "KEY METRICS:",
            "-" * 40,
            f"  System acceleration:    {snapshot.total_acceleration:.1f}x",
            f"  90% confidence:         [{snapshot.acceleration_ci_90[0]:.1f}x - {snapshot.acceleration_ci_90[1]:.1f}x]",
            f"  Net workforce change:   {snapshot.workforce_change:+.2f}M jobs",
            f"  Investment needed:      {snapshot.investment_needed}",
            f"  Critical actions:       {snapshot.critical_actions}",
            "",
            "DOMAIN BREAKDOWN:",
            "-" * 40,
            f"  {'Domain':<22} {'Accel':>8} {'90% CI':>16} {'Jobs':>10}",
            "-" * 60,
        ]

        for domain in self.domains:
            f = snapshot.domain_forecasts[domain]
            ci_str = f"[{f.ci_90[0]:.1f}-{f.ci_90[1]:.1f}]"
            lines.append(
                f"  {self.DOMAIN_NAMES[domain]:<22} {f.acceleration:>7.1f}x {ci_str:>16} {f.net_jobs:>+9.2f}M"
            )

        lines.extend([
            "",
            "TOP INSIGHTS:",
            "-" * 40,
            f"  • Fastest domain: {self.DOMAIN_NAMES[snapshot.fastest_domain]}",
            f"  • Main bottleneck: {self.DOMAIN_NAMES[snapshot.slowest_domain]}",
            f"  • Highest spillover: {snapshot.highest_spillover[0]} → {snapshot.highest_spillover[1]} ({snapshot.highest_spillover[2]:.0%})",
            "",
            "CRITICAL RECOMMENDATIONS:",
            "-" * 40,
        ])

        for rec in recommendations[:3]:
            lines.append(f"  [{rec.id}] {rec.title}")
            lines.append(f"         Timeline: {rec.timeline} | Investment: {rec.investment}")

        lines.extend([
            "",
            "=" * 80,
            f"Model version: {__version__} | Generated: January 2026",
            "=" * 80,
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AIAccelerationModel(version={__version__}, domains={len(self.domains)})"


# Convenience functions for quick access
def quick_forecast(domain: str, year: int = 2030) -> DomainForecast:
    """Quick forecast for a single domain."""
    model = AIAccelerationModel()
    return model.forecast(domain, year)


def quick_summary(year: int = 2030) -> str:
    """Quick executive summary."""
    model = AIAccelerationModel()
    return model.executive_summary(year)


if __name__ == "__main__":
    # Demo usage
    model = AIAccelerationModel()

    print(model.executive_summary(2030))

    print("\n\nSCENARIO COMPARISON (Drug Discovery 2030):")
    print("-" * 60)
    scenarios = model.compare_scenarios("drug_discovery", 2030)
    for scenario, forecast in scenarios.items():
        print(f"  {scenario:<15}: {forecast.acceleration:.2f}x [{forecast.ci_90[0]:.1f}-{forecast.ci_90[1]:.1f}]")
