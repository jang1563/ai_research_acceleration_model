#!/usr/bin/env python3
"""
Disease Models Module for AI-Accelerated Biological Discovery Model - v1.1

Updates from v1.0:
- P2-17: Vaccine pipeline pathway with faster timelines
- P2-12: Disease-specific Phase II M_max overrides
- P1-8: Global access factors integration

Version: 1.1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class DiseaseCategory(Enum):
    """Major disease categories with distinct development characteristics."""
    BREAST_CANCER = "breast_cancer"
    LUNG_CANCER = "lung_cancer"
    PANCREATIC_CANCER = "pancreatic_cancer"
    LEUKEMIA = "leukemia"
    ALZHEIMERS = "alzheimers"
    PARKINSONS = "parkinsons"
    ALS = "als"
    PANDEMIC_NOVEL = "pandemic_novel"
    HIV = "hiv"
    TUBERCULOSIS = "tuberculosis"
    DIABETES_T2 = "diabetes_t2"
    HEART_FAILURE = "heart_failure"
    RARE_GENETIC = "rare_genetic"
    # P2-17: New vaccine pathway
    VACCINE_PLATFORM = "vaccine_platform"


@dataclass
class DiseaseProfile:
    """Disease-specific parameters for modeling time-to-cure."""
    name: str
    category: DiseaseCategory
    starting_stage: int
    advances_needed: int
    baseline_p_modifier: float
    ai_potential_modifier: float
    current_therapies: int
    unmet_need_score: float
    description: str
    stage_p_modifiers: Dict[int, float] = field(default_factory=dict)
    stage_M_modifiers: Dict[int, float] = field(default_factory=dict)
    # P2-12: Disease-specific Phase II M_max override
    phase2_M_max_override: Optional[float] = None
    # P1-8: Global access factor
    global_access_factor: float = 0.5


# Disease profiles including P2-17 vaccine pathway
DISEASE_PROFILES = {
    DiseaseCategory.BREAST_CANCER: DiseaseProfile(
        name="Breast Cancer",
        category=DiseaseCategory.BREAST_CANCER,
        starting_stage=7,
        advances_needed=3,
        baseline_p_modifier=1.2,
        ai_potential_modifier=1.4,
        current_therapies=25,
        unmet_need_score=5.0,
        description="Well-understood biology, strong biomarker landscape.",
        stage_p_modifiers={7: 1.3, 8: 1.2},
        phase2_M_max_override=3.5,  # P2-12: Biomarker-driven
        global_access_factor=0.4,   # P1-8
    ),

    DiseaseCategory.PANCREATIC_CANCER: DiseaseProfile(
        name="Pancreatic Cancer",
        category=DiseaseCategory.PANCREATIC_CANCER,
        starting_stage=6,
        advances_needed=4,
        baseline_p_modifier=0.20,
        ai_potential_modifier=1.1,
        current_therapies=5,
        unmet_need_score=9.5,
        description="Highly aggressive, late diagnosis, poor response.",
        stage_p_modifiers={7: 0.15, 8: 0.20},
        phase2_M_max_override=2.0,  # P2-12: Limited AI benefit
        global_access_factor=0.3,
    ),

    DiseaseCategory.ALZHEIMERS: DiseaseProfile(
        name="Alzheimer's Disease",
        category=DiseaseCategory.ALZHEIMERS,
        starting_stage=4,
        advances_needed=3,
        baseline_p_modifier=0.35,
        ai_potential_modifier=0.9,
        current_therapies=2,
        unmet_need_score=10.0,
        description="Complex biology, post-lecanemab era improving.",
        stage_p_modifiers={6: 0.5, 7: 0.20, 8: 0.35},
        stage_M_modifiers={7: 0.8},
        phase2_M_max_override=2.0,  # P2-12: CNS complexity
        global_access_factor=0.3,
    ),

    DiseaseCategory.PANDEMIC_NOVEL: DiseaseProfile(
        name="Novel Pandemic Pathogen",
        category=DiseaseCategory.PANDEMIC_NOVEL,
        starting_stage=2,
        advances_needed=2,
        baseline_p_modifier=1.5,
        ai_potential_modifier=2.0,
        current_therapies=0,
        unmet_need_score=10.0,
        description="COVID-19 demonstrated AI potential: 100 days vaccine target.",
        stage_p_modifiers={3: 1.5, 6: 1.3, 7: 1.4},
        stage_M_modifiers={3: 2.0, 6: 2.0, 7: 1.5},
        phase2_M_max_override=5.0,  # P2-12: Emergency acceleration
        global_access_factor=0.8,   # P1-8: COVAX distribution
    ),

    # P2-17: New vaccine platform pathway
    DiseaseCategory.VACCINE_PLATFORM: DiseaseProfile(
        name="Vaccine Platform Technology",
        category=DiseaseCategory.VACCINE_PLATFORM,
        starting_stage=3,
        advances_needed=1,
        baseline_p_modifier=1.8,  # Platform validated
        ai_potential_modifier=2.2,  # Highest AI potential
        current_therapies=5,  # mRNA, viral vector, etc.
        unmet_need_score=9.0,
        description="P2-17: Platform technologies (mRNA, viral vector) enable rapid "
                   "development. COVID-19 showed 100-day timeline possible.",
        stage_p_modifiers={3: 1.8, 6: 1.5, 7: 1.6, 8: 1.4},
        stage_M_modifiers={3: 2.5, 6: 2.0, 7: 1.8, 8: 1.5, 9: 1.5},
        phase2_M_max_override=5.0,  # Fast-tracked
        global_access_factor=0.7,
    ),

    DiseaseCategory.RARE_GENETIC: DiseaseProfile(
        name="Rare Genetic Disease",
        category=DiseaseCategory.RARE_GENETIC,
        starting_stage=4,
        advances_needed=1,
        baseline_p_modifier=1.4,
        ai_potential_modifier=1.6,
        current_therapies=0,
        unmet_need_score=9.0,
        description="Gene therapy and ASOs promising. AI excellent for target ID.",
        stage_p_modifiers={6: 1.3, 7: 1.5, 8: 1.4},
        stage_M_modifiers={9: 1.5},
        phase2_M_max_override=4.0,  # P2-12: Targeted development
        global_access_factor=0.2,   # P1-8: Specialized centers only
    ),

    DiseaseCategory.LUNG_CANCER: DiseaseProfile(
        name="Lung Cancer (NSCLC)",
        category=DiseaseCategory.LUNG_CANCER,
        starting_stage=6,
        advances_needed=4,
        baseline_p_modifier=1.1,
        ai_potential_modifier=1.3,
        current_therapies=20,
        unmet_need_score=7.0,
        description="Immunotherapy revolution ongoing.",
        stage_p_modifiers={6: 1.2, 7: 1.1},
        phase2_M_max_override=3.5,
        global_access_factor=0.4,
    ),

    DiseaseCategory.LEUKEMIA: DiseaseProfile(
        name="Leukemia (AML/ALL)",
        category=DiseaseCategory.LEUKEMIA,
        starting_stage=7,
        advances_needed=3,
        baseline_p_modifier=1.0,
        ai_potential_modifier=1.5,
        current_therapies=15,
        unmet_need_score=6.0,
        description="Highly heterogeneous. AI excellent for subtype classification.",
        stage_p_modifiers={7: 1.1},
        phase2_M_max_override=3.5,
        global_access_factor=0.4,
    ),

    DiseaseCategory.PARKINSONS: DiseaseProfile(
        name="Parkinson's Disease",
        category=DiseaseCategory.PARKINSONS,
        starting_stage=5,
        advances_needed=2,
        baseline_p_modifier=0.4,
        ai_potential_modifier=1.0,
        current_therapies=10,
        unmet_need_score=8.0,
        description="Symptomatic treatments exist but no disease-modifying therapy.",
        stage_p_modifiers={7: 0.25, 8: 0.4},
        phase2_M_max_override=2.0,
        global_access_factor=0.3,
    ),

    DiseaseCategory.ALS: DiseaseProfile(
        name="ALS (Lou Gehrig's Disease)",
        category=DiseaseCategory.ALS,
        starting_stage=5,
        advances_needed=2,
        baseline_p_modifier=0.3,
        ai_potential_modifier=1.1,
        current_therapies=3,
        unmet_need_score=9.5,
        description="Fatal within 2-5 years. SOD1 mutations tractable.",
        stage_p_modifiers={7: 0.2, 8: 0.35},
        phase2_M_max_override=2.0,
        global_access_factor=0.3,
    ),

    DiseaseCategory.HIV: DiseaseProfile(
        name="HIV/AIDS (Cure)",
        category=DiseaseCategory.HIV,
        starting_stage=4,
        advances_needed=1,
        baseline_p_modifier=0.6,
        ai_potential_modifier=1.2,
        current_therapies=30,
        unmet_need_score=7.0,
        description="Excellent treatment but no cure. Latent reservoir is key challenge.",
        stage_p_modifiers={7: 0.5},
        phase2_M_max_override=2.5,
        global_access_factor=0.6,
    ),

    DiseaseCategory.TUBERCULOSIS: DiseaseProfile(
        name="Tuberculosis (MDR-TB)",
        category=DiseaseCategory.TUBERCULOSIS,
        starting_stage=6,
        advances_needed=2,
        baseline_p_modifier=0.8,
        ai_potential_modifier=1.3,
        current_therapies=5,
        unmet_need_score=8.0,
        description="Drug-resistant TB is growing threat.",
        stage_p_modifiers={7: 0.7, 8: 0.9},
        phase2_M_max_override=3.0,
        global_access_factor=0.7,  # Global Fund support
    ),

    DiseaseCategory.DIABETES_T2: DiseaseProfile(
        name="Type 2 Diabetes",
        category=DiseaseCategory.DIABETES_T2,
        starting_stage=7,
        advances_needed=2,
        baseline_p_modifier=1.3,
        ai_potential_modifier=1.2,
        current_therapies=15,
        unmet_need_score=6.0,
        description="GLP-1 agonists revolutionary.",
        stage_p_modifiers={7: 1.4, 8: 1.3},
        phase2_M_max_override=3.5,
        global_access_factor=0.5,
    ),

    DiseaseCategory.HEART_FAILURE: DiseaseProfile(
        name="Heart Failure",
        category=DiseaseCategory.HEART_FAILURE,
        starting_stage=6,
        advances_needed=2,
        baseline_p_modifier=1.0,
        ai_potential_modifier=1.3,
        current_therapies=12,
        unmet_need_score=7.0,
        description="SGLT2 inhibitors recent success.",
        stage_p_modifiers={7: 1.1},
        phase2_M_max_override=3.0,
        global_access_factor=0.5,
    ),
}


@dataclass
class DiseaseModelConfig:
    """Configuration for disease modeling."""
    diseases: List[DiseaseCategory] = field(default_factory=lambda: [
        DiseaseCategory.BREAST_CANCER,
        DiseaseCategory.ALZHEIMERS,
        DiseaseCategory.PANDEMIC_NOVEL,
        DiseaseCategory.VACCINE_PLATFORM,  # P2-17
    ])
    cure_horizon_years: int = 26
    n_samples: int = 1000
    discount_rate: float = 0.03


class DiseaseModelModule:
    """Models disease-specific time-to-cure and progress metrics."""

    def __init__(self, config: Optional[DiseaseModelConfig] = None):
        self.config = config or DiseaseModelConfig()
        self.profiles = DISEASE_PROFILES

    def get_disease_profile(self, disease: DiseaseCategory) -> DiseaseProfile:
        """Get profile for a specific disease."""
        return self.profiles[disease]

    def get_phase2_M_max(self, disease: DiseaseCategory) -> float:
        """
        P2-12: Get disease-specific Phase II M_max override.

        Returns the disease-specific override if set, otherwise default.
        """
        profile = self.profiles.get(disease)
        if profile and profile.phase2_M_max_override:
            return profile.phase2_M_max_override
        return 2.8  # Default from v1.0

    def compute_expected_time_to_cure(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
    ) -> Tuple[float, Dict]:
        """Compute expected time to cure for a disease under a scenario."""
        profile = self.profiles[disease]
        stage_times = {}
        total_time = 0.0

        for stage in range(profile.starting_stage, 11):
            base_dur = base_durations.get(stage, 12)

            M_i = scenario_multipliers.get(stage, 1.0)
            M_modifier = profile.stage_M_modifiers.get(stage, profile.ai_potential_modifier)
            effective_M = 1 + (M_i - 1) * M_modifier

            accel_dur = base_dur / effective_M

            p_i = scenario_p_success.get(stage, 0.5)
            p_modifier = profile.stage_p_modifiers.get(stage, profile.baseline_p_modifier)
            effective_p = min(p_i * p_modifier, 0.99)

            expected_attempts = min(1 / effective_p, 10)
            stage_time = accel_dur * expected_attempts

            stage_times[stage] = {
                'base_duration': base_dur,
                'accelerated_duration': accel_dur,
                'effective_M': effective_M,
                'effective_p': effective_p,
                'expected_attempts': expected_attempts,
                'expected_time': stage_time,
            }
            total_time += stage_time

        total_years = total_time / 12
        total_years_all_advances = total_years * profile.advances_needed

        return total_years_all_advances, {
            'disease': disease.value,
            'disease_name': profile.name,
            'starting_stage': profile.starting_stage,
            'advances_needed': profile.advances_needed,
            'time_per_advance_years': total_years,
            'total_time_years': total_years_all_advances,
            'stage_breakdown': stage_times,
            'global_access_factor': profile.global_access_factor,
        }

    def compute_cure_probability(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
        horizon_years: Optional[int] = None,
    ) -> float:
        """Compute probability of achieving cure within horizon."""
        if horizon_years is None:
            horizon_years = self.config.cure_horizon_years

        profile = self.profiles[disease]
        cures = 0

        np.random.seed(42)

        for _ in range(self.config.n_samples):
            total_time = 0.0

            for _ in range(profile.advances_needed):
                for stage in range(profile.starting_stage, 11):
                    base_dur = base_durations.get(stage, 12)

                    M_i = scenario_multipliers.get(stage, 1.0)
                    M_modifier = profile.stage_M_modifiers.get(stage, profile.ai_potential_modifier)
                    effective_M = 1 + (M_i - 1) * M_modifier * np.random.lognormal(0, 0.2)

                    accel_dur = base_dur / max(effective_M, 1.0)

                    p_i = scenario_p_success.get(stage, 0.5)
                    p_modifier = profile.stage_p_modifiers.get(stage, profile.baseline_p_modifier)
                    effective_p = min(p_i * p_modifier * np.random.lognormal(0, 0.15), 0.99)

                    attempts = np.random.geometric(effective_p)
                    attempts = min(attempts, 5)

                    total_time += accel_dur * attempts

            if total_time / 12 <= horizon_years:
                cures += 1

        return cures / self.config.n_samples

    def compute_patients_impacted(
        self,
        disease: DiseaseCategory,
        cure_probability: float,
        horizon_years: int = 26,
    ) -> Dict:
        """Estimate patients impacted, adjusted by global access factor."""
        prevalence = {
            DiseaseCategory.BREAST_CANCER: 2_300_000,
            DiseaseCategory.LUNG_CANCER: 2_200_000,
            DiseaseCategory.PANCREATIC_CANCER: 500_000,
            DiseaseCategory.LEUKEMIA: 475_000,
            DiseaseCategory.ALZHEIMERS: 32_000_000,
            DiseaseCategory.PARKINSONS: 8_500_000,
            DiseaseCategory.ALS: 30_000,
            DiseaseCategory.PANDEMIC_NOVEL: 100_000_000,
            DiseaseCategory.HIV: 1_300_000,
            DiseaseCategory.TUBERCULOSIS: 10_600_000,
            DiseaseCategory.DIABETES_T2: 500_000_000,
            DiseaseCategory.HEART_FAILURE: 64_000_000,
            DiseaseCategory.RARE_GENETIC: 30_000,
            DiseaseCategory.VACCINE_PLATFORM: 50_000_000,  # P2-17
        }

        annual_cases = prevalence.get(disease, 100_000)
        profile = self.profiles[disease]

        # P1-8: Apply global access factor
        access_adjusted_cases = annual_cases * profile.global_access_factor

        expected_beneficiaries = 0
        for year in range(horizon_years):
            discount = (1 + self.config.discount_rate) ** (-year)
            expected_beneficiaries += access_adjusted_cases * cure_probability * discount

        return {
            'disease': disease.value,
            'disease_name': profile.name,
            'annual_cases': annual_cases,
            'global_access_factor': profile.global_access_factor,
            'access_adjusted_cases': access_adjusted_cases,
            'cure_probability': cure_probability,
            'expected_beneficiaries': int(expected_beneficiaries),
        }

    def get_all_profiles_summary(self) -> pd.DataFrame:
        """Get summary of all disease profiles."""
        rows = []
        for category, profile in self.profiles.items():
            rows.append({
                'disease': profile.name,
                'category': category.value,
                'starting_stage': profile.starting_stage,
                'advances_needed': profile.advances_needed,
                'p_modifier': profile.baseline_p_modifier,
                'ai_potential': profile.ai_potential_modifier,
                'phase2_M_max': profile.phase2_M_max_override or 2.8,
                'global_access': profile.global_access_factor,
                'unmet_need': profile.unmet_need_score,
            })
        return pd.DataFrame(rows)


def create_default_module() -> DiseaseModelModule:
    """Create disease model module with default configuration."""
    return DiseaseModelModule(DiseaseModelConfig())


if __name__ == "__main__":
    print("=" * 70)
    print("Disease Model Module - v1.1")
    print("=" * 70)

    module = create_default_module()

    print("\nDisease Profiles Summary (with v1.1 updates):")
    print("-" * 70)
    summary = module.get_all_profiles_summary()
    print(summary.to_string(index=False))

    print("\n\nP2-17 Vaccine Platform Profile:")
    vaccine_profile = module.get_disease_profile(DiseaseCategory.VACCINE_PLATFORM)
    print(f"  Name: {vaccine_profile.name}")
    print(f"  Starting stage: {vaccine_profile.starting_stage}")
    print(f"  AI potential: {vaccine_profile.ai_potential_modifier}")
    print(f"  Phase II M_max: {vaccine_profile.phase2_M_max_override}")
    print(f"  Global access: {vaccine_profile.global_access_factor}")

    print("\nModule loaded successfully.")
