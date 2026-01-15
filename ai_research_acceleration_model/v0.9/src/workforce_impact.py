#!/usr/bin/env python3
"""
Workforce Impact Modeling for v0.9
==================================

Addresses Expert Review P1-P3: "Missing workforce implications"

Models how AI acceleration affects the scientific workforce:
1. Jobs displaced by AI/automation
2. Jobs created by new capabilities
3. Skill transitions required
4. Net employment effects
5. Policy recommendations for workforce development

Key Insight:
AI doesn't simply replace jobs - it transforms them. The net effect
depends on task composition, retraining capacity, and new opportunity
creation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class JobCategory(Enum):
    """Categories of scientific jobs."""
    RESEARCH_SCIENTIST = "research_scientist"
    LAB_TECHNICIAN = "lab_technician"
    DATA_ANALYST = "data_analyst"
    COMPUTATIONAL = "computational"
    CLINICAL = "clinical"
    REGULATORY = "regulatory"
    MANAGEMENT = "management"


@dataclass
class JobProfile:
    """Profile of a job category."""
    category: JobCategory
    name: str

    # Task composition (sums to 1.0)
    cognitive_fraction: float      # Thinking, analysis
    physical_fraction: float       # Lab work, experiments
    interpersonal_fraction: float  # Collaboration, communication

    # AI/automation exposure
    ai_exposure: float             # 0-1, fraction automatable by AI
    automation_exposure: float     # 0-1, fraction automatable by robotics

    # Workforce size (millions globally, approximate)
    current_workforce: float

    # Typical transition time (years to retrain)
    transition_time: float


# Define job profiles by domain
DOMAIN_JOB_PROFILES = {
    "structural_biology": [
        JobProfile(
            category=JobCategory.RESEARCH_SCIENTIST,
            name="Structural Biologist",
            cognitive_fraction=0.6,
            physical_fraction=0.3,
            interpersonal_fraction=0.1,
            ai_exposure=0.5,
            automation_exposure=0.4,
            current_workforce=0.05,
            transition_time=3.0,
        ),
        JobProfile(
            category=JobCategory.LAB_TECHNICIAN,
            name="Crystallography Technician",
            cognitive_fraction=0.2,
            physical_fraction=0.7,
            interpersonal_fraction=0.1,
            ai_exposure=0.3,
            automation_exposure=0.7,
            current_workforce=0.1,
            transition_time=1.5,
        ),
    ],

    "drug_discovery": [
        JobProfile(
            category=JobCategory.RESEARCH_SCIENTIST,
            name="Medicinal Chemist",
            cognitive_fraction=0.5,
            physical_fraction=0.4,
            interpersonal_fraction=0.1,
            ai_exposure=0.4,
            automation_exposure=0.3,
            current_workforce=0.3,
            transition_time=3.0,
        ),
        JobProfile(
            category=JobCategory.LAB_TECHNICIAN,
            name="HTS Technician",
            cognitive_fraction=0.2,
            physical_fraction=0.7,
            interpersonal_fraction=0.1,
            ai_exposure=0.2,
            automation_exposure=0.8,
            current_workforce=0.2,
            transition_time=1.0,
        ),
        JobProfile(
            category=JobCategory.CLINICAL,
            name="Clinical Research Associate",
            cognitive_fraction=0.4,
            physical_fraction=0.2,
            interpersonal_fraction=0.4,
            ai_exposure=0.3,
            automation_exposure=0.1,
            current_workforce=0.5,
            transition_time=2.0,
        ),
        JobProfile(
            category=JobCategory.REGULATORY,
            name="Regulatory Affairs Specialist",
            cognitive_fraction=0.5,
            physical_fraction=0.1,
            interpersonal_fraction=0.4,
            ai_exposure=0.3,
            automation_exposure=0.1,
            current_workforce=0.1,
            transition_time=2.0,
        ),
    ],

    "materials_science": [
        JobProfile(
            category=JobCategory.RESEARCH_SCIENTIST,
            name="Materials Scientist",
            cognitive_fraction=0.5,
            physical_fraction=0.4,
            interpersonal_fraction=0.1,
            ai_exposure=0.5,
            automation_exposure=0.3,
            current_workforce=0.2,
            transition_time=3.0,
        ),
        JobProfile(
            category=JobCategory.LAB_TECHNICIAN,
            name="Synthesis Technician",
            cognitive_fraction=0.2,
            physical_fraction=0.7,
            interpersonal_fraction=0.1,
            ai_exposure=0.2,
            automation_exposure=0.6,
            current_workforce=0.3,
            transition_time=1.0,
        ),
    ],

    "protein_design": [
        JobProfile(
            category=JobCategory.COMPUTATIONAL,
            name="Computational Biologist",
            cognitive_fraction=0.7,
            physical_fraction=0.2,
            interpersonal_fraction=0.1,
            ai_exposure=0.6,
            automation_exposure=0.2,
            current_workforce=0.08,
            transition_time=2.0,
        ),
        JobProfile(
            category=JobCategory.LAB_TECHNICIAN,
            name="Protein Expression Technician",
            cognitive_fraction=0.2,
            physical_fraction=0.7,
            interpersonal_fraction=0.1,
            ai_exposure=0.2,
            automation_exposure=0.7,
            current_workforce=0.1,
            transition_time=1.0,
        ),
    ],

    "clinical_genomics": [
        JobProfile(
            category=JobCategory.CLINICAL,
            name="Genetic Counselor",
            cognitive_fraction=0.4,
            physical_fraction=0.1,
            interpersonal_fraction=0.5,
            ai_exposure=0.4,
            automation_exposure=0.1,
            current_workforce=0.03,
            transition_time=2.0,
        ),
        JobProfile(
            category=JobCategory.DATA_ANALYST,
            name="Clinical Bioinformatician",
            cognitive_fraction=0.7,
            physical_fraction=0.1,
            interpersonal_fraction=0.2,
            ai_exposure=0.5,
            automation_exposure=0.3,
            current_workforce=0.05,
            transition_time=2.0,
        ),
    ],
}


@dataclass
class WorkforceImpact:
    """Impact analysis for a job category."""
    job_name: str
    domain: str

    # Current state
    current_workforce: float  # Millions

    # Displacement
    displacement_rate: float  # Fraction displaced by 2030
    displaced_workers: float  # Millions

    # Creation
    creation_rate: float      # New jobs relative to current
    created_jobs: float       # Millions

    # Net
    net_change: float         # Millions
    net_change_percent: float

    # Transition
    avg_transition_time: float  # Years
    retraining_cost: float      # $ per worker (estimate)


@dataclass
class DomainWorkforceImpact:
    """Aggregate workforce impact for a domain."""
    domain: str
    year: int

    # Totals
    total_current: float
    total_displaced: float
    total_created: float
    total_net: float

    # Breakdown
    job_impacts: List[WorkforceImpact]

    # Policy metrics
    transition_investment_needed: float  # $ millions
    peak_transition_year: int
    skills_gap_severity: str  # "low", "medium", "high"


class WorkforceImpactModel:
    """
    Models workforce impact of AI acceleration.

    Key dynamics:
    1. Task automation: AI/robots replace specific tasks
    2. Job transformation: Jobs shift to higher-value activities
    3. Job creation: New roles emerge (AI trainers, curators)
    4. Skill transitions: Workers retrain for new roles
    """

    def __init__(self):
        self.job_profiles = DOMAIN_JOB_PROFILES

        # New job creation rates by domain
        self.creation_rates = {
            "structural_biology": 1.5,    # 1.5x as many new jobs
            "drug_discovery": 1.8,        # Strong growth
            "materials_science": 1.6,
            "protein_design": 2.5,        # Highest growth
            "clinical_genomics": 1.4,
        }

    def _displacement_rate(
        self,
        profile: JobProfile,
        year: int,
        acceleration: float,
    ) -> float:
        """Calculate displacement rate for a job profile."""
        t = year - 2024

        # Base displacement from task automation
        ai_displacement = profile.ai_exposure * (1 - np.exp(-0.1 * t))
        auto_displacement = profile.automation_exposure * (1 - np.exp(-0.08 * t))

        # Combined (not additive - some overlap)
        combined = 1 - (1 - ai_displacement) * (1 - auto_displacement)

        # Modified by acceleration (higher acceleration = faster displacement)
        accel_factor = 1 + 0.1 * np.log(acceleration)

        # Physical tasks more resistant
        physical_resistance = profile.physical_fraction * 0.3
        interpersonal_resistance = profile.interpersonal_fraction * 0.5

        displacement = combined * accel_factor * (1 - physical_resistance - interpersonal_resistance)

        return min(max(displacement, 0), 0.8)  # Cap at 80%

    def _creation_rate(
        self,
        domain: str,
        year: int,
        acceleration: float,
    ) -> float:
        """Calculate job creation rate for a domain."""
        t = year - 2024

        base_rate = self.creation_rates.get(domain, 1.5)

        # Creation grows with acceleration (more activity = more jobs)
        # But with diminishing returns
        creation = base_rate * np.sqrt(acceleration) / np.sqrt(3)

        # Time factor (creation lags displacement)
        time_factor = 1 - np.exp(-0.15 * t)

        return creation * time_factor

    def analyze_job(
        self,
        profile: JobProfile,
        domain: str,
        year: int,
        acceleration: float,
    ) -> WorkforceImpact:
        """Analyze impact on a specific job category."""
        displacement = self._displacement_rate(profile, year, acceleration)
        displaced = profile.current_workforce * displacement

        # Creation is at domain level
        creation = self._creation_rate(domain, year, acceleration)
        # Attribute creation proportionally
        domain_profiles = self.job_profiles.get(domain, [])
        total_workforce = sum(p.current_workforce for p in domain_profiles)
        profile_share = profile.current_workforce / total_workforce if total_workforce > 0 else 0
        created = total_workforce * creation * profile_share

        net = created - displaced
        net_percent = (net / profile.current_workforce * 100) if profile.current_workforce > 0 else 0

        # Retraining cost estimate ($10k-$50k per worker)
        retraining_cost = 20000 + 10000 * profile.transition_time

        return WorkforceImpact(
            job_name=profile.name,
            domain=domain,
            current_workforce=profile.current_workforce,
            displacement_rate=displacement,
            displaced_workers=displaced,
            creation_rate=creation,
            created_jobs=created,
            net_change=net,
            net_change_percent=net_percent,
            avg_transition_time=profile.transition_time,
            retraining_cost=retraining_cost,
        )

    def analyze_domain(
        self,
        domain: str,
        year: int,
        acceleration: float = None,
    ) -> DomainWorkforceImpact:
        """Analyze workforce impact for an entire domain."""
        if acceleration is None:
            # Default accelerations
            default_accels = {
                "structural_biology": 15.0,
                "drug_discovery": 3.5,
                "materials_science": 3.8,
                "protein_design": 6.6,
                "clinical_genomics": 5.6,
            }
            acceleration = default_accels.get(domain, 3.0)

        profiles = self.job_profiles.get(domain, [])
        job_impacts = []

        total_current = 0
        total_displaced = 0
        total_created = 0
        total_transition_cost = 0

        for profile in profiles:
            impact = self.analyze_job(profile, domain, year, acceleration)
            job_impacts.append(impact)

            total_current += impact.current_workforce
            total_displaced += impact.displaced_workers
            total_created += impact.created_jobs
            total_transition_cost += impact.displaced_workers * 1e6 * impact.retraining_cost

        total_net = total_created - total_displaced

        # Skills gap severity
        if total_displaced > total_created * 1.5:
            severity = "high"
        elif total_displaced > total_created:
            severity = "medium"
        else:
            severity = "low"

        # Peak transition year (when displacement is highest)
        peak_year = 2028 + int(3 * np.log(acceleration))

        return DomainWorkforceImpact(
            domain=domain,
            year=year,
            total_current=total_current,
            total_displaced=total_displaced,
            total_created=total_created,
            total_net=total_net,
            job_impacts=job_impacts,
            transition_investment_needed=total_transition_cost / 1e6,  # Millions
            peak_transition_year=peak_year,
            skills_gap_severity=severity,
        )

    def workforce_report(self, year: int = 2030) -> str:
        """Generate comprehensive workforce impact report."""
        lines = [
            "=" * 100,
            f"WORKFORCE IMPACT ANALYSIS ({year})",
            "=" * 100,
            "",
            "DOMAIN SUMMARY:",
            "-" * 100,
            f"{'Domain':<22} {'Current':<10} {'Displaced':<12} {'Created':<10} {'Net':<10} {'Severity':<10}",
            "-" * 100,
        ]

        total_current = 0
        total_displaced = 0
        total_created = 0

        for domain in self.job_profiles.keys():
            impact = self.analyze_domain(domain, year)
            total_current += impact.total_current
            total_displaced += impact.total_displaced
            total_created += impact.total_created

            lines.append(
                f"{domain:<22} {impact.total_current:>8.2f}M {impact.total_displaced:>10.2f}M "
                f"{impact.total_created:>8.2f}M {impact.total_net:>+8.2f}M "
                f"{impact.skills_gap_severity:<10}"
            )

        lines.extend([
            "-" * 100,
            f"{'TOTAL':<22} {total_current:>8.2f}M {total_displaced:>10.2f}M "
            f"{total_created:>8.2f}M {total_created - total_displaced:>+8.2f}M",
            "",
            "JOB-LEVEL DETAIL (Drug Discovery):",
            "-" * 80,
        ])

        dd_impact = self.analyze_domain("drug_discovery", year)
        for job in dd_impact.job_impacts:
            lines.append(
                f"  {job.job_name:<30} Displaced: {job.displaced_workers:.2f}M "
                f"Created: {job.created_jobs:.2f}M Net: {job.net_change:+.2f}M"
            )

        lines.extend([
            "",
            "POLICY RECOMMENDATIONS:",
            f"  1. Total retraining investment needed: ${dd_impact.transition_investment_needed:.0f}M",
            f"  2. Peak transition pressure: {dd_impact.peak_transition_year}",
            f"  3. Focus areas: High-automation roles (HTS, synthesis)",
            f"  4. Growth areas: AI-augmented research, interpretation",
            "",
            "KEY INSIGHT: Net job impact is positive (+{:.2f}M) but distribution is uneven.".format(
                total_created - total_displaced
            ),
            "High-skill AI-augmented roles grow while routine lab work declines.",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    model = WorkforceImpactModel()
    print(model.workforce_report(2030))
