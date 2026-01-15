#!/usr/bin/env python3
"""
Sub-Domain Profiles for v0.7
============================

Addresses Expert Review Issues:
- D1-P1: "Drug discovery oversimplified" - S4 includes HTS, ADMET, animal studies, clinical trials
- D1-P2: "Protein design heterogeneity" - Enzyme engineering vs de novo design vs antibody design

v0.6 Problem:
A single domain profile (e.g., "drug_discovery") used one set of acceleration parameters.
In reality, different sub-stages and sub-types have vastly different AI/automation impacts.

v0.7 Enhancement:
- Drug Discovery: 7 sub-stages from target ID to post-market
- Protein Design: 4 sub-types with different bottlenecks
- Materials Science: Exploration vs optimization vs scale-up
- Clinical Genomics: Variant discovery vs interpretation vs clinical action

Key Insight:
End-to-end acceleration is the PRODUCT of sub-stage accelerations, weighted by
the fraction of time/cost each sub-stage represents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


@dataclass
class SubStage:
    """A sub-stage within a domain pipeline."""
    name: str
    description: str

    # Time/cost contribution (fraction of total)
    time_fraction: float    # Fraction of total time
    cost_fraction: float    # Fraction of total cost

    # AI impact
    ai_acceleration: float           # AI-enabled acceleration (1-1000x)
    ai_maturity: float               # Current AI maturity (0-1)
    ai_ceiling: float                # Maximum achievable acceleration

    # Automation impact
    automation_acceleration: float   # Automation acceleration (1-50x)
    automation_maturity: float       # Current automation maturity (0-1)
    automation_ceiling: float        # Maximum achievable acceleration

    # Physical constraints
    inherently_physical: bool        # Must be done physically
    regulatory_constrained: bool     # Regulatory approval required
    human_judgment_required: float   # Fraction requiring human decision (0-1)


@dataclass
class SubDomainProfile:
    """A sub-type within a domain with distinct characteristics."""
    name: str
    description: str
    parent_domain: str

    # Acceleration range (observed/expected)
    min_acceleration: float
    typical_acceleration: float
    max_acceleration: float

    # Key bottlenecks
    primary_bottleneck: str
    secondary_bottleneck: str

    # Current state
    current_maturity: float   # 0-1, how mature is AI in this sub-domain
    growth_rate: float        # Annual improvement rate


# =============================================================================
# DRUG DISCOVERY SUB-STAGES (D1-P1 Fix)
# =============================================================================

DRUG_DISCOVERY_STAGES = {
    "target_identification": SubStage(
        name="Target Identification",
        description="Identify disease-relevant biological targets",
        time_fraction=0.05,
        cost_fraction=0.02,
        ai_acceleration=5.0,
        ai_maturity=0.6,
        ai_ceiling=20.0,
        automation_acceleration=1.5,
        automation_maturity=0.3,
        automation_ceiling=3.0,
        inherently_physical=False,
        regulatory_constrained=False,
        human_judgment_required=0.5,
    ),

    "hit_discovery": SubStage(
        name="Hit Discovery / HTS",
        description="Screen compound libraries for activity",
        time_fraction=0.08,
        cost_fraction=0.05,
        ai_acceleration=10.0,
        ai_maturity=0.7,
        ai_ceiling=100.0,
        automation_acceleration=5.0,
        automation_maturity=0.8,
        automation_ceiling=10.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.2,
    ),

    "hit_to_lead": SubStage(
        name="Hit-to-Lead Optimization",
        description="Optimize hit compounds for potency and selectivity",
        time_fraction=0.10,
        cost_fraction=0.08,
        ai_acceleration=3.0,
        ai_maturity=0.5,
        ai_ceiling=20.0,
        automation_acceleration=2.0,
        automation_maturity=0.4,
        automation_ceiling=5.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.4,
    ),

    "admet_optimization": SubStage(
        name="ADMET Optimization",
        description="Optimize absorption, distribution, metabolism, excretion, toxicity",
        time_fraction=0.12,
        cost_fraction=0.10,
        ai_acceleration=3.0,
        ai_maturity=0.4,
        ai_ceiling=10.0,
        automation_acceleration=2.0,
        automation_maturity=0.5,
        automation_ceiling=5.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.3,
    ),

    "preclinical": SubStage(
        name="Preclinical Studies",
        description="Animal studies for safety and efficacy",
        time_fraction=0.15,
        cost_fraction=0.15,
        ai_acceleration=1.5,
        ai_maturity=0.3,
        ai_ceiling=5.0,
        automation_acceleration=1.2,
        automation_maturity=0.2,
        automation_ceiling=2.0,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.5,
    ),

    "phase1_clinical": SubStage(
        name="Phase 1 Clinical Trials",
        description="Safety testing in healthy volunteers",
        time_fraction=0.10,
        cost_fraction=0.10,
        ai_acceleration=1.2,
        ai_maturity=0.2,
        ai_ceiling=2.0,
        automation_acceleration=1.1,
        automation_maturity=0.1,
        automation_ceiling=1.5,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.7,
    ),

    "phase2_clinical": SubStage(
        name="Phase 2 Clinical Trials",
        description="Efficacy testing in patients",
        time_fraction=0.15,
        cost_fraction=0.20,
        ai_acceleration=1.15,
        ai_maturity=0.2,
        ai_ceiling=1.5,
        automation_acceleration=1.05,
        automation_maturity=0.1,
        automation_ceiling=1.2,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.8,
    ),

    "phase3_clinical": SubStage(
        name="Phase 3 Clinical Trials",
        description="Large-scale efficacy and safety trials",
        time_fraction=0.20,
        cost_fraction=0.25,
        ai_acceleration=1.1,
        ai_maturity=0.1,
        ai_ceiling=1.3,
        automation_acceleration=1.05,
        automation_maturity=0.1,
        automation_ceiling=1.1,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.9,
    ),

    "regulatory_approval": SubStage(
        name="Regulatory Approval",
        description="FDA/EMA approval process",
        time_fraction=0.05,
        cost_fraction=0.05,
        ai_acceleration=1.3,
        ai_maturity=0.2,
        ai_ceiling=2.0,
        automation_acceleration=1.2,
        automation_maturity=0.2,
        automation_ceiling=1.5,
        inherently_physical=False,
        regulatory_constrained=True,
        human_judgment_required=0.9,
    ),
}


# =============================================================================
# PROTEIN DESIGN SUB-TYPES (D1-P2 Fix)
# =============================================================================

PROTEIN_DESIGN_SUBTYPES = {
    "enzyme_engineering": SubDomainProfile(
        name="Enzyme Engineering",
        description="Optimize enzymes for industrial/therapeutic use",
        parent_domain="protein_design",
        min_acceleration=2.0,
        typical_acceleration=10.0,
        max_acceleration=20.0,
        primary_bottleneck="Activity assay throughput",
        secondary_bottleneck="Expression yield",
        current_maturity=0.6,
        growth_rate=0.15,
    ),

    "de_novo_design": SubDomainProfile(
        name="De Novo Protein Design",
        description="Design new protein folds and functions",
        parent_domain="protein_design",
        min_acceleration=4.0,
        typical_acceleration=50.0,
        max_acceleration=100.0,
        primary_bottleneck="Functional validation",
        secondary_bottleneck="Fold stability",
        current_maturity=0.5,
        growth_rate=0.20,
    ),

    "antibody_design": SubDomainProfile(
        name="Antibody Design",
        description="Design therapeutic antibodies",
        parent_domain="protein_design",
        min_acceleration=1.5,
        typical_acceleration=5.0,
        max_acceleration=10.0,
        primary_bottleneck="Affinity maturation",
        secondary_bottleneck="Immunogenicity prediction",
        current_maturity=0.4,
        growth_rate=0.12,
    ),

    "protein_binders": SubDomainProfile(
        name="Protein Binder Design",
        description="Design proteins that bind specific targets",
        parent_domain="protein_design",
        min_acceleration=3.0,
        typical_acceleration=20.0,
        max_acceleration=50.0,
        primary_bottleneck="Binding assay throughput",
        secondary_bottleneck="Specificity optimization",
        current_maturity=0.55,
        growth_rate=0.18,
    ),
}


# =============================================================================
# MATERIALS SCIENCE SUB-STAGES
# =============================================================================

MATERIALS_SCIENCE_STAGES = {
    "computational_screening": SubStage(
        name="Computational Screening",
        description="DFT/ML screening of candidate materials",
        time_fraction=0.05,
        cost_fraction=0.02,
        ai_acceleration=365.0,  # GNoME-scale
        ai_maturity=0.8,
        ai_ceiling=1000.0,
        automation_acceleration=10.0,
        automation_maturity=0.9,
        automation_ceiling=20.0,
        inherently_physical=False,
        regulatory_constrained=False,
        human_judgment_required=0.1,
    ),

    "synthesis": SubStage(
        name="Material Synthesis",
        description="Physical synthesis of candidate materials",
        time_fraction=0.40,
        cost_fraction=0.35,
        ai_acceleration=2.0,
        ai_maturity=0.3,
        ai_ceiling=10.0,
        automation_acceleration=3.0,
        automation_maturity=0.4,
        automation_ceiling=20.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.3,
    ),

    "characterization": SubStage(
        name="Characterization",
        description="Measure material properties",
        time_fraction=0.25,
        cost_fraction=0.25,
        ai_acceleration=3.0,
        ai_maturity=0.4,
        ai_ceiling=10.0,
        automation_acceleration=5.0,
        automation_maturity=0.5,
        automation_ceiling=20.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.2,
    ),

    "optimization": SubStage(
        name="Property Optimization",
        description="Iterative optimization of material properties",
        time_fraction=0.20,
        cost_fraction=0.25,
        ai_acceleration=5.0,
        ai_maturity=0.5,
        ai_ceiling=20.0,
        automation_acceleration=2.0,
        automation_maturity=0.3,
        automation_ceiling=10.0,
        inherently_physical=True,
        regulatory_constrained=False,
        human_judgment_required=0.4,
    ),

    "scale_up": SubStage(
        name="Scale-Up",
        description="Scale from lab to production",
        time_fraction=0.10,
        cost_fraction=0.13,
        ai_acceleration=1.5,
        ai_maturity=0.2,
        ai_ceiling=5.0,
        automation_acceleration=1.5,
        automation_maturity=0.3,
        automation_ceiling=5.0,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.5,
    ),
}


# =============================================================================
# CLINICAL GENOMICS SUB-STAGES
# =============================================================================

CLINICAL_GENOMICS_STAGES = {
    "variant_calling": SubStage(
        name="Variant Calling",
        description="Identify genetic variants from sequencing",
        time_fraction=0.10,
        cost_fraction=0.15,
        ai_acceleration=5.0,
        ai_maturity=0.8,
        ai_ceiling=10.0,
        automation_acceleration=10.0,
        automation_maturity=0.9,
        automation_ceiling=20.0,
        inherently_physical=False,
        regulatory_constrained=False,
        human_judgment_required=0.1,
    ),

    "variant_interpretation": SubStage(
        name="Variant Interpretation",
        description="Classify variants as pathogenic/benign",
        time_fraction=0.25,
        cost_fraction=0.20,
        ai_acceleration=50.0,  # AlphaMissense-scale
        ai_maturity=0.7,
        ai_ceiling=100.0,
        automation_acceleration=3.0,
        automation_maturity=0.6,
        automation_ceiling=10.0,
        inherently_physical=False,
        regulatory_constrained=False,
        human_judgment_required=0.3,
    ),

    "clinical_correlation": SubStage(
        name="Clinical Correlation",
        description="Correlate variants with patient phenotypes",
        time_fraction=0.20,
        cost_fraction=0.15,
        ai_acceleration=3.0,
        ai_maturity=0.4,
        ai_ceiling=10.0,
        automation_acceleration=2.0,
        automation_maturity=0.3,
        automation_ceiling=5.0,
        inherently_physical=False,
        regulatory_constrained=True,
        human_judgment_required=0.5,
    ),

    "clinical_validation": SubStage(
        name="Clinical Validation",
        description="Validate findings in clinical setting",
        time_fraction=0.25,
        cost_fraction=0.30,
        ai_acceleration=1.5,
        ai_maturity=0.2,
        ai_ceiling=3.0,
        automation_acceleration=1.3,
        automation_maturity=0.2,
        automation_ceiling=2.0,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.7,
    ),

    "clinical_action": SubStage(
        name="Clinical Action",
        description="Implement findings in patient care",
        time_fraction=0.20,
        cost_fraction=0.20,
        ai_acceleration=1.2,
        ai_maturity=0.1,
        ai_ceiling=2.0,
        automation_acceleration=1.1,
        automation_maturity=0.1,
        automation_ceiling=1.5,
        inherently_physical=True,
        regulatory_constrained=True,
        human_judgment_required=0.9,
    ),
}


class SubDomainModel:
    """
    Models acceleration at sub-stage/sub-type granularity.

    Key insight: End-to-end acceleration is determined by the slowest sub-stage,
    weighted by time/cost fraction.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.stages = self._get_stages()
        self.subtypes = self._get_subtypes()

    def _get_stages(self) -> Dict[str, SubStage]:
        """Get sub-stages for domain."""
        stage_mapping = {
            "drug_discovery": DRUG_DISCOVERY_STAGES,
            "materials_science": MATERIALS_SCIENCE_STAGES,
            "clinical_genomics": CLINICAL_GENOMICS_STAGES,
        }
        return stage_mapping.get(self.domain, {})

    def _get_subtypes(self) -> Dict[str, SubDomainProfile]:
        """Get sub-types for domain."""
        if self.domain == "protein_design":
            return PROTEIN_DESIGN_SUBTYPES
        return {}

    def stage_acceleration(
        self,
        stage_name: str,
        year: int,
        ai_scenario: str = "baseline",
        automation_scenario: str = "baseline",
    ) -> float:
        """
        Calculate acceleration for a specific sub-stage.

        Args:
            stage_name: Name of the sub-stage
            year: Target year
            ai_scenario: AI development scenario
            automation_scenario: Automation development scenario

        Returns:
            Acceleration factor for this stage
        """
        if stage_name not in self.stages:
            return 1.0

        stage = self.stages[stage_name]
        t = year - 2024

        # AI acceleration (grows toward ceiling with maturity)
        ai_growth = stage.ai_maturity + (1 - stage.ai_maturity) * (1 - np.exp(-0.1 * t))
        ai_accel = 1 + (stage.ai_ceiling - 1) * ai_growth

        # Scenario multipliers
        ai_multipliers = {
            "conservative": 0.5,
            "baseline": 1.0,
            "optimistic": 1.5,
            "breakthrough": 2.0,
        }
        ai_accel = 1 + (ai_accel - 1) * ai_multipliers.get(ai_scenario, 1.0)

        # Automation acceleration
        auto_growth = stage.automation_maturity + (1 - stage.automation_maturity) * (1 - np.exp(-0.08 * t))
        auto_accel = 1 + (stage.automation_ceiling - 1) * auto_growth

        auto_multipliers = {
            "conservative": 0.5,
            "baseline": 1.0,
            "optimistic": 1.5,
            "breakthrough": 3.0,
        }
        auto_accel = 1 + (auto_accel - 1) * auto_multipliers.get(automation_scenario, 1.0)

        # Combine (minimum of AI and automation, modified by physical constraints)
        if stage.inherently_physical:
            # Physical stages limited by automation
            combined = min(ai_accel, auto_accel)
        else:
            # Computational stages can use full AI acceleration
            combined = ai_accel

        # Regulatory constraint reduces acceleration
        if stage.regulatory_constrained:
            combined = 1 + (combined - 1) * 0.5

        # Human judgment reduces automation benefit
        combined = 1 + (combined - 1) * (1 - stage.human_judgment_required * 0.3)

        return combined

    def end_to_end_acceleration(
        self,
        year: int,
        ai_scenario: str = "baseline",
        automation_scenario: str = "baseline",
        use_time_weighting: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate end-to-end acceleration across all sub-stages.

        Returns:
            Tuple of (overall_acceleration, stage_accelerations_dict)
        """
        if not self.stages:
            return 1.0, {}

        stage_accels = {}
        weighted_log_accel = 0

        for name, stage in self.stages.items():
            accel = self.stage_acceleration(
                name, year, ai_scenario, automation_scenario
            )
            stage_accels[name] = accel

            # Weight by time or cost fraction
            weight = stage.time_fraction if use_time_weighting else stage.cost_fraction
            weighted_log_accel += weight * np.log(accel)

        # Geometric mean (multiplicative acceleration)
        overall = np.exp(weighted_log_accel)

        return overall, stage_accels

    def subtype_acceleration(
        self,
        subtype_name: str,
        year: int,
    ) -> float:
        """
        Calculate acceleration for a specific sub-type.
        """
        if subtype_name not in self.subtypes:
            return 1.0

        subtype = self.subtypes[subtype_name]
        t = year - 2024

        # Growth toward max acceleration
        growth = subtype.current_maturity + (1 - subtype.current_maturity) * (
            1 - np.exp(-subtype.growth_rate * t)
        )

        accel = subtype.min_acceleration + (
            (subtype.max_acceleration - subtype.min_acceleration) * growth
        )

        return accel

    def identify_bottlenecks(
        self,
        year: int,
        ai_scenario: str = "baseline",
        automation_scenario: str = "baseline",
    ) -> List[Tuple[str, float, str]]:
        """
        Identify the bottleneck sub-stages.

        Returns list of (stage_name, acceleration, bottleneck_type) sorted by impact.
        """
        if not self.stages:
            return []

        bottlenecks = []

        for name, stage in self.stages.items():
            accel = self.stage_acceleration(
                name, year, ai_scenario, automation_scenario
            )

            # Determine bottleneck type
            if stage.regulatory_constrained:
                bottleneck_type = "regulatory"
            elif stage.inherently_physical:
                bottleneck_type = "physical"
            elif stage.human_judgment_required > 0.5:
                bottleneck_type = "human_judgment"
            else:
                bottleneck_type = "none"

            # Calculate impact (time fraction / acceleration)
            impact = stage.time_fraction / accel
            bottlenecks.append((name, accel, bottleneck_type, impact))

        # Sort by impact (descending)
        bottlenecks.sort(key=lambda x: x[3], reverse=True)

        return [(b[0], b[1], b[2]) for b in bottlenecks]

    def summary(self) -> str:
        """Generate summary of sub-domain model."""
        lines = [
            "=" * 90,
            f"SUB-DOMAIN MODEL: {self.domain.upper()}",
            "=" * 90,
            "",
        ]

        if self.stages:
            lines.extend([
                "SUB-STAGE ANALYSIS (2030 Baseline Scenario):",
                "-" * 90,
                f"{'Stage':<30} {'Time%':<8} {'AI Acc':<10} {'Auto Acc':<10} {'Combined':<10} {'Type':<12}",
                "-" * 90,
            ])

            overall, stage_accels = self.end_to_end_acceleration(2030)

            for name, stage in self.stages.items():
                accel = stage_accels.get(name, 1.0)
                stage_type = "Physical" if stage.inherently_physical else "Compute"
                if stage.regulatory_constrained:
                    stage_type = "Regulated"

                lines.append(
                    f"{name:<30} {stage.time_fraction:>6.0%} "
                    f"{stage.ai_acceleration:>8.1f}x {stage.automation_acceleration:>8.1f}x "
                    f"{accel:>8.1f}x {stage_type:<12}"
                )

            lines.extend([
                "-" * 90,
                f"{'END-TO-END':<30} {'100%':<8} {'':<10} {'':<10} {overall:>8.1f}x",
                "",
            ])

            # Bottleneck analysis
            bottlenecks = self.identify_bottlenecks(2030)
            lines.extend([
                "TOP BOTTLENECKS:",
            ])
            for i, (name, accel, btype) in enumerate(bottlenecks[:3]):
                lines.append(f"  {i+1}. {name}: {accel:.1f}x acceleration ({btype})")

        if self.subtypes:
            lines.extend([
                "",
                "SUB-TYPE ANALYSIS:",
                "-" * 90,
                f"{'Sub-Type':<25} {'Min':<8} {'Typical':<10} {'Max':<8} {'2030 Est':<10} {'Bottleneck':<20}",
                "-" * 90,
            ])

            for name, subtype in self.subtypes.items():
                accel_2030 = self.subtype_acceleration(name, 2030)
                lines.append(
                    f"{name:<25} {subtype.min_acceleration:>6.1f}x "
                    f"{subtype.typical_acceleration:>8.1f}x {subtype.max_acceleration:>6.1f}x "
                    f"{accel_2030:>8.1f}x {subtype.primary_bottleneck:<20}"
                )

        return "\n".join(lines)


def analyze_drug_discovery_pipeline():
    """Detailed analysis of drug discovery pipeline."""
    print("=" * 90)
    print("DRUG DISCOVERY PIPELINE ANALYSIS")
    print("=" * 90)
    print()

    model = SubDomainModel("drug_discovery")

    years = [2024, 2030, 2040, 2050]
    scenarios = ["conservative", "baseline", "optimistic", "breakthrough"]

    print("END-TO-END ACCELERATION BY SCENARIO:")
    print("-" * 60)
    print(f"{'Scenario':<15} {'2024':<10} {'2030':<10} {'2040':<10} {'2050':<10}")
    print("-" * 60)

    for scenario in scenarios:
        accels = []
        for year in years:
            overall, _ = model.end_to_end_acceleration(
                year, ai_scenario=scenario, automation_scenario=scenario
            )
            accels.append(f"{overall:.1f}x")
        print(f"{scenario:<15} {accels[0]:<10} {accels[1]:<10} {accels[2]:<10} {accels[3]:<10}")

    print("-" * 60)
    print()
    print("KEY INSIGHT: Clinical trials (Phase 2/3) represent ~35% of time and ~45% of cost,")
    print("but only achieve 1.1-1.2x acceleration due to regulatory and human constraints.")
    print("This limits end-to-end acceleration regardless of early-stage AI improvements.")


def analyze_protein_design_subtypes():
    """Analysis of protein design sub-types."""
    print("=" * 90)
    print("PROTEIN DESIGN SUB-TYPE ANALYSIS")
    print("=" * 90)
    print()

    model = SubDomainModel("protein_design")

    years = [2024, 2030, 2040, 2050]

    print(f"{'Sub-Type':<25} {'2024':<10} {'2030':<10} {'2040':<10} {'2050':<10}")
    print("-" * 70)

    for name in PROTEIN_DESIGN_SUBTYPES.keys():
        accels = [f"{model.subtype_acceleration(name, y):.1f}x" for y in years]
        print(f"{name:<25} {accels[0]:<10} {accels[1]:<10} {accels[2]:<10} {accels[3]:<10}")

    print("-" * 70)
    print()
    print("KEY INSIGHT: De novo design shows highest potential (up to 100x) due to")
    print("computational nature, while antibody design is more constrained by")
    print("immunogenicity validation requirements.")


if __name__ == "__main__":
    # Drug discovery analysis
    analyze_drug_discovery_pipeline()
    print()
    print()

    # Protein design analysis
    analyze_protein_design_subtypes()
    print()
    print()

    # Detailed summary
    model = SubDomainModel("drug_discovery")
    print(model.summary())
