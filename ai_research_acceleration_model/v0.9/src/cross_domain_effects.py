#!/usr/bin/env python3
"""
Cross-Domain Interaction Effects for v0.9
==========================================

Models how advances in one domain accelerate others:

1. Structural Biology → Drug Discovery: Better targets, faster docking
2. Protein Design → Drug Discovery: Biologics, antibody therapeutics
3. Materials Science → All: Better instrumentation, sensors
4. Clinical Genomics → Drug Discovery: Better patient stratification

Key Insight:
Domains are not independent - breakthroughs cascade across fields.
AlphaFold accelerated drug discovery; ESM-3 enabled protein therapeutics.

This module captures:
- Direct spillover effects (technology transfer)
- Indirect effects (talent, funding, attention)
- Synergistic effects (compound acceleration)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DomainInteraction:
    """An interaction between two domains."""
    source_domain: str
    target_domain: str

    # Strength of interaction (0-1)
    direct_effect: float      # Technology/method transfer
    indirect_effect: float    # Funding/attention shift
    lag_years: float          # Years for effect to manifest

    # Mechanism
    mechanism: str
    examples: List[str]


# Define domain interactions
DOMAIN_INTERACTIONS = [
    # Structural Biology effects
    DomainInteraction(
        source_domain="structural_biology",
        target_domain="drug_discovery",
        direct_effect=0.4,
        indirect_effect=0.2,
        lag_years=1.0,
        mechanism="Structure-based drug design, target validation",
        examples=["AlphaFold → kinase inhibitors", "Cryo-EM → GPCR drugs"],
    ),
    DomainInteraction(
        source_domain="structural_biology",
        target_domain="protein_design",
        direct_effect=0.5,
        indirect_effect=0.1,
        lag_years=0.5,
        mechanism="Structure templates, scaffold design",
        examples=["AF2 → de novo design", "Structure → function prediction"],
    ),

    # Protein Design effects
    DomainInteraction(
        source_domain="protein_design",
        target_domain="drug_discovery",
        direct_effect=0.3,
        indirect_effect=0.15,
        lag_years=2.0,
        mechanism="Biologics, antibodies, ADCs",
        examples=["Designed antibodies", "Enzyme therapeutics"],
    ),
    DomainInteraction(
        source_domain="protein_design",
        target_domain="materials_science",
        direct_effect=0.1,
        indirect_effect=0.05,
        lag_years=3.0,
        mechanism="Biomaterials, protein scaffolds",
        examples=["Spider silk", "Protein-based materials"],
    ),

    # Materials Science effects
    DomainInteraction(
        source_domain="materials_science",
        target_domain="structural_biology",
        direct_effect=0.1,
        indirect_effect=0.05,
        lag_years=2.0,
        mechanism="Better detectors, cryo-EM grids",
        examples=["New detector materials", "Sample preparation"],
    ),
    DomainInteraction(
        source_domain="materials_science",
        target_domain="drug_discovery",
        direct_effect=0.05,
        indirect_effect=0.05,
        lag_years=3.0,
        mechanism="Drug delivery, sensors",
        examples=["Nanoparticle delivery", "Diagnostic sensors"],
    ),

    # Clinical Genomics effects
    DomainInteraction(
        source_domain="clinical_genomics",
        target_domain="drug_discovery",
        direct_effect=0.25,
        indirect_effect=0.1,
        lag_years=1.5,
        mechanism="Patient stratification, target discovery",
        examples=["GWAS targets", "Pharmacogenomics"],
    ),
    DomainInteraction(
        source_domain="clinical_genomics",
        target_domain="protein_design",
        direct_effect=0.1,
        indirect_effect=0.05,
        lag_years=2.0,
        mechanism="Variant interpretation, personalized therapeutics",
        examples=["Personalized AAV", "Gene therapy design"],
    ),

    # Drug Discovery effects (back-propagation)
    DomainInteraction(
        source_domain="drug_discovery",
        target_domain="clinical_genomics",
        direct_effect=0.15,
        indirect_effect=0.1,
        lag_years=2.0,
        mechanism="Clinical trial data, outcome validation",
        examples=["Drug response variants", "Trial genetic data"],
    ),
]


@dataclass
class CrossDomainForecast:
    """Forecast with cross-domain effects."""
    domain: str
    year: int

    # Base acceleration (without cross-domain)
    base_acceleration: float

    # Cross-domain boost
    spillover_boost: float
    synergy_boost: float

    # Final acceleration
    total_acceleration: float

    # Breakdown by source domain
    effects_by_source: Dict[str, float]


class CrossDomainModel:
    """
    Models cross-domain acceleration effects.

    Key dynamics:
    1. Spillover: Advances in domain A directly boost domain B
    2. Synergy: Combined advances create compound effects
    3. Lag: Effects take time to manifest (1-3 years)
    """

    def __init__(self):
        self.interactions = DOMAIN_INTERACTIONS

        # Build interaction matrix
        self.domains = ["structural_biology", "drug_discovery", "materials_science",
                        "protein_design", "clinical_genomics"]
        self.interaction_matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """Build domain interaction matrix."""
        n = len(self.domains)
        matrix = np.zeros((n, n))

        for interaction in self.interactions:
            i = self.domains.index(interaction.source_domain)
            j = self.domains.index(interaction.target_domain)
            total_effect = interaction.direct_effect + interaction.indirect_effect
            matrix[i, j] = total_effect

        return matrix

    def _base_acceleration(self, domain: str, year: int) -> float:
        """Get base acceleration (from earlier models)."""
        t = year - 2024

        # Simplified base accelerations
        base_2030 = {
            "structural_biology": 15.0,
            "drug_discovery": 3.5,
            "materials_science": 3.8,
            "protein_design": 6.6,
            "clinical_genomics": 5.6,
        }

        base = base_2030.get(domain, 3.0)
        # Logistic growth
        return 1 + (base - 1) * (1 - np.exp(-0.15 * t))

    def _calculate_spillovers(
        self,
        year: int,
        base_accels: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate spillover effects from each domain to each other."""
        spillovers = {d: {} for d in self.domains}

        for interaction in self.interactions:
            source = interaction.source_domain
            target = interaction.target_domain

            # Spillover proportional to source acceleration
            source_accel = base_accels.get(source, 1.0)

            # Account for lag
            effective_year = year - interaction.lag_years
            if effective_year < 2024:
                lag_factor = 0
            else:
                lag_factor = min(1.0, (year - 2024) / interaction.lag_years)

            # Spillover effect - use log scale to dampen extreme values
            # Spillover is proportional to log of source acceleration, not linear
            log_accel = np.log1p(source_accel - 1)  # log(1 + (accel-1))
            effect = log_accel * interaction.direct_effect * lag_factor * 0.3
            effect += log_accel * interaction.indirect_effect * lag_factor * 0.15

            # Cap at 50% boost per source
            spillovers[target][source] = min(max(0, effect), 0.5)

        return spillovers

    def _calculate_synergies(
        self,
        spillovers: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate synergistic effects from multiple spillovers."""
        synergies = {}

        for domain in self.domains:
            domain_spillovers = list(spillovers[domain].values())
            if len(domain_spillovers) >= 2:
                # Synergy when multiple domains contribute
                # Sqrt of product of top 2 effects
                sorted_effects = sorted(domain_spillovers, reverse=True)[:2]
                synergy = 0.3 * np.sqrt(sorted_effects[0] * sorted_effects[1])
            else:
                synergy = 0

            # Cap synergy at 20%
            synergies[domain] = min(synergy, 0.2)

        return synergies

    def forecast(self, domain: str, year: int) -> CrossDomainForecast:
        """Generate forecast with cross-domain effects."""
        # Get base accelerations for all domains
        base_accels = {d: self._base_acceleration(d, year) for d in self.domains}

        # Calculate spillovers
        spillovers = self._calculate_spillovers(year, base_accels)

        # Calculate synergies
        synergies = self._calculate_synergies(spillovers)

        # Total for this domain
        base = base_accels[domain]
        spillover_boost = sum(spillovers[domain].values())
        synergy_boost = synergies[domain]

        # Multiplicative combination
        total = base * (1 + spillover_boost) * (1 + synergy_boost)

        return CrossDomainForecast(
            domain=domain,
            year=year,
            base_acceleration=base,
            spillover_boost=spillover_boost,
            synergy_boost=synergy_boost,
            total_acceleration=total,
            effects_by_source=spillovers[domain],
        )

    def interaction_report(self, year: int = 2030) -> str:
        """Generate cross-domain interaction report."""
        lines = [
            "=" * 90,
            f"CROSS-DOMAIN INTERACTION ANALYSIS ({year})",
            "=" * 90,
            "",
            "INTERACTION MATRIX (effect strength):",
            "-" * 90,
            "From / To".ljust(20) + "".join(f"{d[:10]:>12}" for d in self.domains),
            "-" * 90,
        ]

        for i, source in enumerate(self.domains):
            row = f"{source:<20}"
            for j, target in enumerate(self.domains):
                if i == j:
                    row += f"{'--':>12}"
                else:
                    row += f"{self.interaction_matrix[i, j]:>11.2f}"
            lines.append(row)

        lines.extend([
            "-" * 90,
            "",
            "ACCELERATION BREAKDOWN:",
            "-" * 90,
            f"{'Domain':<22} {'Base':<10} {'Spillover':<12} {'Synergy':<10} {'Total':<10}",
            "-" * 90,
        ])

        for domain in self.domains:
            f = self.forecast(domain, year)
            lines.append(
                f"{domain:<22} {f.base_acceleration:>8.1f}x "
                f"{f.spillover_boost:>+10.2f} {f.synergy_boost:>+8.2f} "
                f"{f.total_acceleration:>8.1f}x"
            )

        lines.extend([
            "-" * 90,
            "",
            "TOP SPILLOVER EFFECTS:",
        ])

        # Find top spillovers
        all_spillovers = []
        for target in self.domains:
            f = self.forecast(target, year)
            for source, effect in f.effects_by_source.items():
                if effect > 0.1:
                    all_spillovers.append((source, target, effect))

        all_spillovers.sort(key=lambda x: x[2], reverse=True)
        for source, target, effect in all_spillovers[:5]:
            lines.append(f"  {source} → {target}: +{effect:.2f}")

        lines.extend([
            "",
            "KEY INSIGHT: Drug discovery benefits most from cross-domain effects",
            "(structural biology, protein design, clinical genomics all contribute).",
        ])

        return "\n".join(lines)


def analyze_cross_domain():
    """Run cross-domain analysis."""
    model = CrossDomainModel()
    print(model.interaction_report(2030))


if __name__ == "__main__":
    analyze_cross_domain()
