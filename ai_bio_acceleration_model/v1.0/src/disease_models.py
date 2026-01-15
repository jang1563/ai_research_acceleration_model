#!/usr/bin/env python3
"""
Disease Models Module for AI-Accelerated Biological Discovery Model

This module implements disease-specific modeling for time-to-cure calculations
and case studies. Different diseases have varying:
- Starting stages in the pipeline (some already have drug candidates)
- Number of advances needed for meaningful progress
- Success rate modifiers (biological complexity)
- AI acceleration potential (data availability, biomarker potential)

Case Studies:
1. Cancer (various subtypes) - High AI potential, biomarker-driven
2. Alzheimer's Disease - Low success rates, complex biology
3. Pandemic Preparedness - Rapid response, platform technologies

================================================================================
MATHEMATICAL FRAMEWORK
================================================================================

1. TIME-TO-CURE CALCULATION
---------------------------
Expected time to achieve cure (all advances):

    T_cure = advances_needed × Σ_{i=start}^{10} E[T_i]

Where E[T_i] = expected time at stage i:

    E[T_i] = (τ_i / M_eff_i) × E[attempts_i]

Components:
- τ_i = baseline duration (months) from DiMasi et al. (2016)
- M_eff_i = effective AI multiplier = 1 + (M_i - 1) × disease_M_modifier
- E[attempts_i] = expected attempts = min(1/p_eff_i, 10) [capped]
- p_eff_i = effective success prob = min(p_i × disease_p_modifier, 0.99)

The 1/p formulation comes from geometric distribution: if each attempt
succeeds with probability p, expected attempts until success = 1/p.
Reference: Ross, S. (2014). "A First Course in Probability"

2. CURE PROBABILITY (Monte Carlo)
---------------------------------
P(cure by horizon) estimated via simulation:

    For n=1000 samples:
        1. Draw M_i ~ LogNormal(log(M_i), 0.2) for uncertainty
        2. Draw p_i ~ LogNormal(log(p_i), 0.15) for uncertainty
        3. Simulate attempts at each stage (geometric draws)
        4. Sum total time across all stages and advances
        5. Count if total_time ≤ horizon_years

    P(cure) = count(successes) / n

Coefficient of variation (CV) values from:
- Uncertainty in AI capability projections: CV=20% (Epoch AI, 2024)
- Uncertainty in clinical success rates: CV=15% (Wong et al., 2019)

3. EXPECTED BENEFICIARIES (Health Economics)
--------------------------------------------
Discounted expected beneficiaries over horizon (UPDATED per Expert D1/D2):

    E[B] = P(cure) × uptake × Σ_{y=0}^{H} [cases_0 × (1+g)^y / (1+r)^y]

Where:
- P(cure) = probability of cure by horizon
- uptake = treatment uptake rate (0-1), accounts for access barriers
- cases_0 = baseline annual cases
- g = incidence growth rate (disease-specific, 0-2%/year)
- r = discount rate = 3% (standard for health economics)
- H = horizon years (26 for 2024-2050)

Incidence Growth Rates (Expert D1 - Dr. Okonkwo):
- Alzheimer's: +2%/year (aging population, +50% by 2050)
- Pancreatic cancer: +1%/year
- Breast cancer: +0.5%/year
- Pandemic: 0% (episodic, not growth-based)

Treatment Uptake Rates (Expert D2 - Dr. Okonkwo):
- Pandemic vaccine: 70% (based on COVID-19 uptake)
- Cancer treatment: 85% (high-income countries)
- Alzheimer's: 60% (diagnosis + access barriers)
- Rare genetic: 50% (specialized centers only)

Discount rate of 3% follows NICE (UK) and ICER (US) guidelines:
- NICE (2013). "Guide to the methods of technology appraisal"
- Sanders et al. (2016). "Recommendations for Conduct of Cost-Effectiveness
  Analysis: Second Panel on Cost-Effectiveness in Health and Medicine"

================================================================================
DISEASE PARAMETER SOURCES
================================================================================

Success Rate Modifiers (baseline_p_modifier):
- Oncology: Wong et al. (2019) Phase II success = 21% vs 33% average → 0.63x
- CNS: Wong et al. (2019) Phase II success = 15% vs 33% average → 0.46x
- Alzheimer's: Cummings et al. (2019) reports 99% failure rate → 0.25x
- Rare Disease: FDA Orphan Drug data shows higher success → 1.4x

AI Potential Modifiers (ai_potential_modifier):
- High biomarker diseases (oncology): Topol (2019) → 1.3-1.5x
- Genomics-driven (rare genetic): Estimated from gene therapy success → 1.6x
- Complex biology (CNS): Limited data availability → 0.9x
- Pandemic: COVID-19 demonstrated AI potential → 2.0x (100-day vaccine)

Disease Prevalence/Incidence:
- WHO Global Health Estimates (2024)
- GBD 2019 Study (Lancet, 2020)
- Disease-specific registries (NCI, Alzheimer's Association, etc.)

TERMINOLOGY NOTE (Expert E2 - Dr. Kim):
---------------------------------------
"Cure" in this model refers to achieving transformative therapeutic outcomes,
which varies by disease type:
- Curative: Disease eliminated (rare genetic, some early cancers)
- Disease-modifying: Significant slowing/halting progression (Alzheimer's, most cancers)
- Functional: Normal life with ongoing treatment (HIV model)

For chronic conditions like Alzheimer's and metastatic cancer, "cure" should be
interpreted as "transformative therapy" achieving clinically meaningful outcomes,
not complete disease elimination.

================================================================================
REFERENCES
================================================================================

Clinical Development:
- Wong CH, Siah KW, Lo AW (2019). "Estimation of clinical trial success rates
  and related parameters." Biostatistics 20(2):273-286.
- DiMasi JA, Grabowski HG, Hansen RW (2016). "Innovation in the pharmaceutical
  industry: New estimates of R&D costs." J Health Econ 47:20-33.
- Hay M, Thomas DW, Craighead JL, et al. (2014). "Clinical development success
  rates for investigational drugs." Nat Biotechnol 32(1):40-51.

Disease-Specific:
- Cummings J, Lee G, Ritter A, et al. (2019). "Alzheimer's disease drug
  development pipeline: 2019." Alzheimers Dement (N Y) 5:272-293.
- Mullard A (2021). "2020 FDA drug approvals." Nat Rev Drug Discov 20:85-90.

Health Economics:
- NICE (2013). "Guide to the methods of technology appraisal."
- Sanders GD, et al. (2016). "Recommendations for Conduct, Methodological
  Practices, and Reporting of Cost-effectiveness Analyses." JAMA 316(10):1093.

AI in Drug Discovery:
- Topol EJ (2019). "High-performance medicine: the convergence of human and
  artificial intelligence." Nat Med 25(1):44-56.
- Schneider P, et al. (2020). "Rethinking drug design in the artificial
  intelligence era." Nat Rev Drug Discov 19(5):353-364.

Probability Theory:
- Ross SM (2014). "A First Course in Probability." 9th ed. Pearson.
  [Geometric distribution: E[X] = 1/p for attempts until first success]

Version: 0.8
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class DiseaseCategory(Enum):
    """Major disease categories with distinct development characteristics."""
    # Cancers
    BREAST_CANCER = "breast_cancer"
    LUNG_CANCER = "lung_cancer"
    PANCREATIC_CANCER = "pancreatic_cancer"
    LEUKEMIA = "leukemia"

    # Neurodegenerative
    ALZHEIMERS = "alzheimers"
    PARKINSONS = "parkinsons"
    ALS = "als"

    # Infectious
    PANDEMIC_NOVEL = "pandemic_novel"
    HIV = "hiv"
    TUBERCULOSIS = "tuberculosis"

    # Other
    DIABETES_T2 = "diabetes_t2"
    HEART_FAILURE = "heart_failure"
    RARE_GENETIC = "rare_genetic"


@dataclass
class DiseaseProfile:
    """
    Disease-specific parameters for modeling time-to-cure.

    Attributes:
        name: Display name
        category: DiseaseCategory enum
        starting_stage: Which pipeline stage is the current frontier
        advances_needed: Number of successful therapies needed for "cure"
        baseline_p_modifier: Multiplier on baseline success rates
        ai_potential_modifier: Multiplier on AI acceleration (M_max)
        current_therapies: Number of approved therapies
        unmet_need_score: 0-10 scale of unmet medical need
        description: Brief description of development challenges
    """
    name: str
    category: DiseaseCategory
    starting_stage: int  # 1-10, where in pipeline is current frontier
    advances_needed: int  # How many successful drugs to consider "solved"
    baseline_p_modifier: float  # Multiplier on p_success (1.0 = average)
    ai_potential_modifier: float  # Multiplier on M_max (1.0 = average)
    current_therapies: int  # Approved drugs as of 2024
    unmet_need_score: float  # 0-10
    description: str

    # Stage-specific overrides (optional)
    stage_p_modifiers: Dict[int, float] = field(default_factory=dict)
    stage_M_modifiers: Dict[int, float] = field(default_factory=dict)


# Disease profiles based on literature
DISEASE_PROFILES = {
    # =========================================================================
    # CANCERS - Generally high AI potential due to biomarkers, genomics
    # =========================================================================
    DiseaseCategory.BREAST_CANCER: DiseaseProfile(
        name="Breast Cancer",
        category=DiseaseCategory.BREAST_CANCER,
        starting_stage=7,  # Many candidates in Phase II+
        advances_needed=3,  # Need 3 more breakthrough therapies
        baseline_p_modifier=1.2,  # Better than average (biomarkers)
        ai_potential_modifier=1.4,  # High AI potential (genomics, imaging)
        current_therapies=25,  # Many approved
        unmet_need_score=5.0,  # Moderate - metastatic still challenging
        description="Well-understood biology, strong biomarker landscape. "
                   "AI excels at subtype identification and treatment matching.",
        stage_p_modifiers={7: 1.3, 8: 1.2},  # Higher Phase II/III success
    ),

    DiseaseCategory.LUNG_CANCER: DiseaseProfile(
        name="Lung Cancer (NSCLC)",
        category=DiseaseCategory.LUNG_CANCER,
        starting_stage=6,  # Active Phase I pipeline
        advances_needed=4,
        baseline_p_modifier=1.1,
        ai_potential_modifier=1.3,  # Good genomic data
        current_therapies=20,
        unmet_need_score=7.0,  # High - still leading cause of cancer death
        description="Immunotherapy revolution ongoing. AI valuable for "
                   "biomarker discovery and combination optimization.",
        stage_p_modifiers={6: 1.2, 7: 1.1},
    ),

    # UPDATED per Expert B2 (Dr. Williams): starting_stage 5→6
    # UPDATED per Expert H2 (Dr. Huang): p_modifier 0.50→0.20 (Phase II ~5%)
    DiseaseCategory.PANCREATIC_CANCER: DiseaseProfile(
        name="Pancreatic Cancer",
        category=DiseaseCategory.PANCREATIC_CANCER,
        starting_stage=6,  # Many drugs in Phase I/II (gemcitabine, PARP, immunotherapy)
        advances_needed=4,  # Reduced from 5 - need quality breakthroughs
        baseline_p_modifier=0.20,  # Phase II success ~5% (was 0.5, too optimistic)
        ai_potential_modifier=1.1,  # Moderate AI potential
        current_therapies=5,
        unmet_need_score=9.5,  # Critical - 5-year survival ~10%
        description="Highly aggressive, late diagnosis, poor response to "
                   "therapy. Early detection is key AI opportunity.",
        stage_p_modifiers={7: 0.15, 8: 0.20},  # Very low Phase II/III success
    ),

    DiseaseCategory.LEUKEMIA: DiseaseProfile(
        name="Leukemia (AML/ALL)",
        category=DiseaseCategory.LEUKEMIA,
        starting_stage=7,
        advances_needed=3,
        baseline_p_modifier=1.0,
        ai_potential_modifier=1.5,  # High - genomic classification
        current_therapies=15,
        unmet_need_score=6.0,
        description="Highly heterogeneous. AI excellent for subtype "
                   "classification and minimal residual disease monitoring.",
        stage_p_modifiers={7: 1.1},
    ),

    # =========================================================================
    # NEURODEGENERATIVE - Low success rates, complex biology
    # =========================================================================
    # UPDATED per Expert B1 (Dr. Williams): p_modifier 0.25→0.35 (post-lecanemab era)
    DiseaseCategory.ALZHEIMERS: DiseaseProfile(
        name="Alzheimer's Disease",
        category=DiseaseCategory.ALZHEIMERS,
        starting_stage=4,  # Still need better targets
        advances_needed=3,  # Disease-modifying therapies
        baseline_p_modifier=0.35,  # Updated: post-lecanemab era, better target selection
        ai_potential_modifier=0.9,  # Lower AI potential (complex biology)
        current_therapies=2,  # Aducanumab, lecanemab (controversial)
        unmet_need_score=10.0,  # Critical unmet need
        description="99% failure rate historically, but improving with validated "
                   "targets. Lecanemab success suggests better target selection.",
        stage_p_modifiers={6: 0.5, 7: 0.20, 8: 0.35},  # Slightly improved success
        stage_M_modifiers={7: 0.8},  # Slower trials (disease progression)
    ),

    DiseaseCategory.PARKINSONS: DiseaseProfile(
        name="Parkinson's Disease",
        category=DiseaseCategory.PARKINSONS,
        starting_stage=5,
        advances_needed=2,  # Disease-modifying therapy
        baseline_p_modifier=0.4,
        ai_potential_modifier=1.0,
        current_therapies=10,  # Symptomatic only
        unmet_need_score=8.0,
        description="Symptomatic treatments exist but no disease-modifying "
                   "therapy. Alpha-synuclein targeting in trials.",
        stage_p_modifiers={7: 0.25, 8: 0.4},
    ),

    DiseaseCategory.ALS: DiseaseProfile(
        name="ALS (Lou Gehrig's Disease)",
        category=DiseaseCategory.ALS,
        starting_stage=5,
        advances_needed=2,
        baseline_p_modifier=0.3,
        ai_potential_modifier=1.1,  # Some AI potential (genetic subtypes)
        current_therapies=3,  # Limited efficacy
        unmet_need_score=9.5,
        description="Fatal within 2-5 years. SOD1 mutations tractable, "
                   "sporadic ALS remains challenging.",
        stage_p_modifiers={7: 0.2, 8: 0.35},
    ),

    # =========================================================================
    # INFECTIOUS - Variable, pandemic response critical
    # =========================================================================
    DiseaseCategory.PANDEMIC_NOVEL: DiseaseProfile(
        name="Novel Pandemic Pathogen",
        category=DiseaseCategory.PANDEMIC_NOVEL,
        starting_stage=2,  # Start from hypothesis
        advances_needed=2,  # Vaccine + therapeutic
        baseline_p_modifier=1.5,  # Platform technologies help
        ai_potential_modifier=2.0,  # Highest AI potential (COVID showed this)
        current_therapies=0,  # Novel pathogen
        unmet_need_score=10.0,  # Emergency
        description="COVID-19 demonstrated AI potential: protein structure, "
                   "mRNA design, drug repurposing. 100 days vaccine target.",
        stage_p_modifiers={3: 1.5, 6: 1.3, 7: 1.4},
        stage_M_modifiers={3: 2.0, 6: 2.0, 7: 1.5},  # Emergency acceleration
    ),

    DiseaseCategory.HIV: DiseaseProfile(
        name="HIV/AIDS (Cure)",
        category=DiseaseCategory.HIV,
        starting_stage=4,
        advances_needed=1,  # Functional cure
        baseline_p_modifier=0.6,
        ai_potential_modifier=1.2,
        current_therapies=30,  # Many ARVs, but no cure
        unmet_need_score=7.0,
        description="Excellent treatment but no cure. Latent reservoir "
                   "is key challenge. Gene therapy approaches emerging.",
        stage_p_modifiers={7: 0.5},
    ),

    DiseaseCategory.TUBERCULOSIS: DiseaseProfile(
        name="Tuberculosis (MDR-TB)",
        category=DiseaseCategory.TUBERCULOSIS,
        starting_stage=6,
        advances_needed=2,
        baseline_p_modifier=0.8,
        ai_potential_modifier=1.3,
        current_therapies=5,  # Need new mechanisms
        unmet_need_score=8.0,
        description="Drug-resistant TB is growing threat. Need shorter "
                   "treatment regimens. AI for drug combination optimization.",
        stage_p_modifiers={7: 0.7, 8: 0.9},
    ),

    # =========================================================================
    # OTHER MAJOR DISEASES
    # =========================================================================
    DiseaseCategory.DIABETES_T2: DiseaseProfile(
        name="Type 2 Diabetes",
        category=DiseaseCategory.DIABETES_T2,
        starting_stage=7,
        advances_needed=2,  # Disease reversal
        baseline_p_modifier=1.3,  # Good success rates
        ai_potential_modifier=1.2,
        current_therapies=15,
        unmet_need_score=6.0,
        description="GLP-1 agonists revolutionary. AI valuable for "
                   "precision medicine and complication prediction.",
        stage_p_modifiers={7: 1.4, 8: 1.3},
    ),

    DiseaseCategory.HEART_FAILURE: DiseaseProfile(
        name="Heart Failure",
        category=DiseaseCategory.HEART_FAILURE,
        starting_stage=6,
        advances_needed=2,
        baseline_p_modifier=1.0,
        ai_potential_modifier=1.3,  # Imaging, biomarkers
        current_therapies=12,
        unmet_need_score=7.0,
        description="SGLT2 inhibitors recent success. AI for imaging "
                   "analysis and patient stratification.",
        stage_p_modifiers={7: 1.1},
    ),

    DiseaseCategory.RARE_GENETIC: DiseaseProfile(
        name="Rare Genetic Disease (Generic)",
        category=DiseaseCategory.RARE_GENETIC,
        starting_stage=4,
        advances_needed=1,  # Single therapy often sufficient
        baseline_p_modifier=1.4,  # Higher success (targeted)
        ai_potential_modifier=1.6,  # High AI potential (genomics)
        current_therapies=0,  # Disease-specific
        unmet_need_score=9.0,
        description="7,000+ rare diseases, most untreated. Gene therapy "
                   "and ASOs promising. AI excellent for target identification.",
        stage_p_modifiers={6: 1.3, 7: 1.5, 8: 1.4},
        stage_M_modifiers={9: 1.5},  # Accelerated approval pathways
    ),
}


@dataclass
class DiseaseModelConfig:
    """Configuration for disease modeling."""

    # Selected diseases for analysis
    diseases: List[DiseaseCategory] = field(default_factory=lambda: [
        DiseaseCategory.BREAST_CANCER,
        DiseaseCategory.ALZHEIMERS,
        DiseaseCategory.PANDEMIC_NOVEL,
    ])

    # Time horizon for cure probability
    cure_horizon_years: int = 26  # 2024-2050

    # Monte Carlo samples for uncertainty
    n_samples: int = 1000

    # Discount rate for future cures (health economics)
    discount_rate: float = 0.03


class DiseaseModelModule:
    """
    Models disease-specific time-to-cure and progress metrics.

    Key features:
    1. Disease-specific starting stages and success modifiers
    2. Time-to-cure calculations under different scenarios
    3. Probability of cure by 2050
    4. Expected patients impacted
    """

    def __init__(self, config: Optional[DiseaseModelConfig] = None):
        """Initialize disease model module."""
        self.config = config or DiseaseModelConfig()
        self.profiles = DISEASE_PROFILES

    def get_disease_profile(self, disease: DiseaseCategory) -> DiseaseProfile:
        """Get profile for a specific disease."""
        return self.profiles[disease]

    def compute_expected_time_to_cure(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
    ) -> Tuple[float, Dict]:
        """
        Compute expected time to cure for a disease under a scenario.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease to model
        scenario_multipliers : Dict[int, float]
            AI acceleration multipliers by stage (from model)
        scenario_p_success : Dict[int, float]
            Success probabilities by stage (from model)
        base_durations : Dict[int, float]
            Baseline durations in months by stage

        Returns
        -------
        Tuple[float, Dict]
            Expected time to cure (years) and detailed breakdown
        """
        profile = self.profiles[disease]

        # Compute expected time for each stage from starting point
        stage_times = {}
        total_time = 0.0

        for stage in range(profile.starting_stage, 11):
            # Get base duration
            base_dur = base_durations.get(stage, 12)  # Default 12 months

            # Apply AI multiplier (with disease-specific modifier)
            M_i = scenario_multipliers.get(stage, 1.0)
            M_modifier = profile.stage_M_modifiers.get(stage, profile.ai_potential_modifier)
            effective_M = 1 + (M_i - 1) * M_modifier

            # Accelerated duration
            accel_dur = base_dur / effective_M

            # Apply success probability (with disease-specific modifier)
            p_i = scenario_p_success.get(stage, 0.5)
            p_modifier = profile.stage_p_modifiers.get(stage, profile.baseline_p_modifier)
            effective_p = min(p_i * p_modifier, 0.99)  # Cap at 99%

            # Expected attempts = 1/p (geometric distribution)
            # But cap at reasonable number
            expected_attempts = min(1 / effective_p, 10)

            # Expected time for this stage
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

        # Convert to years
        total_years = total_time / 12

        # Multiply by advances needed
        total_years_all_advances = total_years * profile.advances_needed

        return total_years_all_advances, {
            'disease': disease.value,
            'disease_name': profile.name,
            'starting_stage': profile.starting_stage,
            'advances_needed': profile.advances_needed,
            'time_per_advance_years': total_years,
            'total_time_years': total_years_all_advances,
            'stage_breakdown': stage_times,
        }

    def compute_cure_probability(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
        horizon_years: Optional[int] = None,
    ) -> float:
        """
        Compute probability of achieving cure within horizon.

        Uses Monte Carlo simulation with uncertainty in parameters.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease to model
        scenario_multipliers : Dict[int, float]
            AI acceleration multipliers
        scenario_p_success : Dict[int, float]
            Success probabilities
        base_durations : Dict[int, float]
            Baseline durations
        horizon_years : int, optional
            Time horizon (default: config value)

        Returns
        -------
        float
            Probability of cure (0-1)
        """
        if horizon_years is None:
            horizon_years = self.config.cure_horizon_years

        profile = self.profiles[disease]
        cures = 0

        np.random.seed(42)  # Reproducibility

        for _ in range(self.config.n_samples):
            # Simulate time to achieve all advances
            total_time = 0.0
            advances_achieved = 0

            for _ in range(profile.advances_needed):
                # Simulate one drug development cycle
                for stage in range(profile.starting_stage, 11):
                    base_dur = base_durations.get(stage, 12)

                    # Add uncertainty to multipliers (CV = 20%)
                    M_i = scenario_multipliers.get(stage, 1.0)
                    M_modifier = profile.stage_M_modifiers.get(stage, profile.ai_potential_modifier)
                    effective_M = 1 + (M_i - 1) * M_modifier * np.random.lognormal(0, 0.2)

                    accel_dur = base_dur / max(effective_M, 1.0)

                    # Success probability with uncertainty
                    p_i = scenario_p_success.get(stage, 0.5)
                    p_modifier = profile.stage_p_modifiers.get(stage, profile.baseline_p_modifier)
                    effective_p = min(p_i * p_modifier * np.random.lognormal(0, 0.15), 0.99)

                    # Simulate attempts (geometric)
                    attempts = np.random.geometric(effective_p)
                    attempts = min(attempts, 5)  # Cap at 5 attempts

                    total_time += accel_dur * attempts

                advances_achieved += 1

            # Check if achieved within horizon
            if total_time / 12 <= horizon_years:
                cures += 1

        return cures / self.config.n_samples

    def compute_time_to_cure_distribution(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
    ) -> Dict:
        """
        Compute time-to-cure distribution with confidence intervals.

        ADDED per Expert F1 (Dr. Nakamura): Uncertainty bands for time estimates.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease to model
        scenario_multipliers : Dict[int, float]
            AI acceleration multipliers
        scenario_p_success : Dict[int, float]
            Success probabilities
        base_durations : Dict[int, float]
            Baseline durations

        Returns
        -------
        Dict
            Distribution statistics including mean, median, CI
        """
        profile = self.profiles[disease]
        times = []

        np.random.seed(42)

        for _ in range(self.config.n_samples):
            total_time = 0.0

            for _ in range(profile.advances_needed):
                for stage in range(profile.starting_stage, 11):
                    base_dur = base_durations.get(stage, 12)

                    # Add uncertainty to multipliers (CV = 20%)
                    M_i = scenario_multipliers.get(stage, 1.0)
                    M_modifier = profile.stage_M_modifiers.get(stage, profile.ai_potential_modifier)
                    effective_M = 1 + (M_i - 1) * M_modifier * np.random.lognormal(0, 0.2)

                    accel_dur = base_dur / max(effective_M, 1.0)

                    # Success probability with uncertainty (CV = 15%)
                    p_i = scenario_p_success.get(stage, 0.5)
                    p_modifier = profile.stage_p_modifiers.get(stage, profile.baseline_p_modifier)
                    effective_p = min(p_i * p_modifier * np.random.lognormal(0, 0.15), 0.99)

                    # Simulate attempts (geometric)
                    attempts = np.random.geometric(effective_p)
                    attempts = min(attempts, 5)

                    total_time += accel_dur * attempts

            times.append(total_time / 12)  # Convert to years

        times = np.array(times)

        return {
            'disease': disease.value,
            'disease_name': profile.name,
            'mean': float(np.mean(times)),
            'median': float(np.median(times)),
            'std': float(np.std(times)),
            'ci_5': float(np.percentile(times, 5)),
            'ci_25': float(np.percentile(times, 25)),
            'ci_75': float(np.percentile(times, 75)),
            'ci_95': float(np.percentile(times, 95)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
        }

    def compute_cure_probability_trajectory(
        self,
        disease: DiseaseCategory,
        scenario_multipliers: Dict[int, float],
        scenario_p_success: Dict[int, float],
        base_durations: Dict[int, float],
        years: List[int] = None,
    ) -> Dict:
        """
        Compute cure probability trajectory over time.

        ADDED per Expert F3 (Dr. Nakamura): P(cure) evolution from 2024 to 2050.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease to model
        scenario_multipliers : Dict[int, float]
            AI acceleration multipliers
        scenario_p_success : Dict[int, float]
            Success probabilities
        base_durations : Dict[int, float]
            Baseline durations
        years : List[int], optional
            Years to compute P(cure) for (default: 2024-2050)

        Returns
        -------
        Dict
            Trajectory with year -> P(cure) mapping
        """
        if years is None:
            years = list(range(2024, 2051))

        profile = self.profiles[disease]
        trajectory = {}

        for target_year in years:
            horizon = target_year - 2024
            if horizon <= 0:
                trajectory[target_year] = 0.0
                continue

            # Compute P(cure) for this horizon
            p_cure = self.compute_cure_probability(
                disease, scenario_multipliers, scenario_p_success,
                base_durations, horizon_years=horizon
            )
            trajectory[target_year] = p_cure

        return {
            'disease': disease.value,
            'disease_name': profile.name,
            'trajectory': trajectory,
        }

    def compute_patients_impacted(
        self,
        disease: DiseaseCategory,
        cure_probability: float,
        horizon_years: int = 26,
    ) -> Dict:
        """
        Estimate patients impacted by achieving cure.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease
        cure_probability : float
            Probability of cure
        horizon_years : int
            Time horizon

        Returns
        -------
        Dict
            Impact metrics
        """
        # Disease prevalence/incidence estimates (annual, global)
        # Source: WHO Global Health Estimates 2024, GBD 2019
        # UPDATED per Expert G1 (Dr. Santos - WHO Epidemiology)
        prevalence = {
            DiseaseCategory.BREAST_CANCER: 2_300_000,  # New cases/year (WHO 2024)
            DiseaseCategory.LUNG_CANCER: 2_200_000,    # New cases/year
            DiseaseCategory.PANCREATIC_CANCER: 500_000,  # New cases/year
            DiseaseCategory.LEUKEMIA: 475_000,         # New cases/year
            DiseaseCategory.ALZHEIMERS: 32_000_000,    # Living with AD (WHO 2024) - was 10M, corrected
            DiseaseCategory.PARKINSONS: 8_500_000,     # Living with
            DiseaseCategory.ALS: 30_000,               # New cases/year
            DiseaseCategory.PANDEMIC_NOVEL: 100_000_000,  # Potential (variable 10M-1B)
            DiseaseCategory.HIV: 1_300_000,            # New infections/year
            DiseaseCategory.TUBERCULOSIS: 10_600_000,  # Incidence/year (prevalence 7.5M)
            DiseaseCategory.DIABETES_T2: 500_000_000,  # Living with
            DiseaseCategory.HEART_FAILURE: 64_000_000, # Living with
            DiseaseCategory.RARE_GENETIC: 30_000,      # Per disease
        }

        annual_cases = prevalence.get(disease, 100_000)

        # UPDATED per Expert D1: Incidence growth rates
        # Accounts for demographic changes (aging population, etc.)
        incidence_growth = {
            DiseaseCategory.ALZHEIMERS: 0.02,        # 2% annual increase (aging)
            DiseaseCategory.PANCREATIC_CANCER: 0.01, # 1% annual increase
            DiseaseCategory.BREAST_CANCER: 0.005,    # 0.5% annual increase
            DiseaseCategory.LUNG_CANCER: 0.005,      # 0.5% annual increase
            DiseaseCategory.PANDEMIC_NOVEL: 0.0,     # Episodic, not growth-based
            DiseaseCategory.RARE_GENETIC: 0.0,       # Stable incidence
            DiseaseCategory.DIABETES_T2: 0.015,      # 1.5% (obesity epidemic)
            DiseaseCategory.HEART_FAILURE: 0.01,     # 1% (aging)
        }
        growth_rate = incidence_growth.get(disease, 0.005)  # Default 0.5%

        # UPDATED per Expert D2: Treatment uptake rates
        # Not all patients receive cure even if available (access, diagnosis, eligibility)
        uptake_rates = {
            DiseaseCategory.PANDEMIC_NOVEL: 0.70,    # Based on COVID vaccine uptake
            DiseaseCategory.BREAST_CANCER: 0.85,    # High-income country cancer care
            DiseaseCategory.LUNG_CANCER: 0.80,      # Slightly lower (late diagnosis)
            DiseaseCategory.PANCREATIC_CANCER: 0.75, # Often diagnosed late
            DiseaseCategory.LEUKEMIA: 0.85,         # Cancer treatment
            DiseaseCategory.ALZHEIMERS: 0.60,       # Diagnosis + access barriers
            DiseaseCategory.PARKINSONS: 0.70,       # Better diagnosis than AD
            DiseaseCategory.ALS: 0.80,              # Specialty centers
            DiseaseCategory.HIV: 0.75,              # Treatment access varies
            DiseaseCategory.TUBERCULOSIS: 0.65,     # LMIC access issues
            DiseaseCategory.DIABETES_T2: 0.80,      # Good treatment availability
            DiseaseCategory.HEART_FAILURE: 0.80,    # Good treatment availability
            DiseaseCategory.RARE_GENETIC: 0.50,     # Specialized centers only
        }
        uptake = uptake_rates.get(disease, 0.75)  # Default 75%

        # Expected patients benefiting (UPDATED formula per Expert D1/D2):
        # E[B] = P(cure) × uptake × Σ_{y=0}^{H} [cases_0 × (1+g)^y / (1+r)^y]
        expected_beneficiaries = 0
        for year in range(horizon_years):
            growth = (1 + growth_rate) ** year
            discount = (1 + self.config.discount_rate) ** (-year)
            expected_beneficiaries += annual_cases * growth * cure_probability * uptake * discount

        profile = self.profiles[disease]

        return {
            'disease': disease.value,
            'disease_name': profile.name,
            'annual_cases': annual_cases,
            'incidence_growth_rate': growth_rate,
            'treatment_uptake_rate': uptake,
            'cure_probability': cure_probability,
            'horizon_years': horizon_years,
            'expected_beneficiaries': int(expected_beneficiaries),
            'unmet_need_score': profile.unmet_need_score,
        }

    def generate_case_study(
        self,
        disease: DiseaseCategory,
        scenario_results: Dict[str, Dict],
        base_durations: Dict[int, float],
    ) -> pd.DataFrame:
        """
        Generate comprehensive case study for a disease.

        Parameters
        ----------
        disease : DiseaseCategory
            The disease to analyze
        scenario_results : Dict[str, Dict]
            Results from model scenarios (multipliers, p_success)
        base_durations : Dict[int, float]
            Baseline stage durations

        Returns
        -------
        pd.DataFrame
            Case study comparison across scenarios
        """
        profile = self.profiles[disease]
        rows = []

        for scenario_name, scenario_data in scenario_results.items():
            multipliers = scenario_data.get('multipliers', {})
            p_success = scenario_data.get('p_success', {})

            # Time to cure
            time_years, details = self.compute_expected_time_to_cure(
                disease, multipliers, p_success, base_durations
            )

            # Cure probability
            cure_prob = self.compute_cure_probability(
                disease, multipliers, p_success, base_durations
            )

            # Patients impacted
            impact = self.compute_patients_impacted(disease, cure_prob)

            rows.append({
                'scenario': scenario_name,
                'disease': profile.name,
                'expected_time_years': time_years,
                'time_per_advance': details['time_per_advance_years'],
                'cure_probability_26yr': cure_prob,
                'expected_beneficiaries': impact['expected_beneficiaries'],
                'starting_stage': profile.starting_stage,
                'advances_needed': profile.advances_needed,
                'unmet_need_score': profile.unmet_need_score,
            })

        return pd.DataFrame(rows)

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
                'current_therapies': profile.current_therapies,
                'unmet_need': profile.unmet_need_score,
            })
        return pd.DataFrame(rows)


def create_default_module() -> DiseaseModelModule:
    """Create disease model module with default configuration."""
    return DiseaseModelModule(DiseaseModelConfig())


if __name__ == "__main__":
    # Test the module
    print("=" * 70)
    print("Disease Model Module Test")
    print("=" * 70)

    module = create_default_module()

    print("\nDisease Profiles Summary:")
    print("-" * 70)
    summary = module.get_all_profiles_summary()
    print(summary.to_string(index=False))

    # Test with example scenario data
    example_multipliers = {
        1: 10.0, 2: 8.0, 3: 3.0, 4: 20.0, 5: 3.0,
        6: 2.5, 7: 2.0, 8: 2.2, 9: 1.8, 10: 3.0
    }
    example_p_success = {
        1: 0.95, 2: 0.90, 3: 0.35, 4: 0.95, 5: 0.55,
        6: 0.70, 7: 0.40, 8: 0.65, 9: 0.92, 10: 0.95
    }
    base_durations = {
        1: 6, 2: 3, 3: 12, 4: 2, 5: 8,
        6: 12, 7: 24, 8: 36, 9: 12, 10: 12
    }

    print("\n\nCase Study: Alzheimer's Disease")
    print("-" * 70)
    time_years, details = module.compute_expected_time_to_cure(
        DiseaseCategory.ALZHEIMERS,
        example_multipliers,
        example_p_success,
        base_durations
    )
    print(f"  Expected time to cure: {time_years:.1f} years")
    print(f"  Starting stage: {details['starting_stage']}")
    print(f"  Advances needed: {details['advances_needed']}")
    print(f"  Time per advance: {details['time_per_advance_years']:.1f} years")

    print("\n\nCase Study: Pandemic Preparedness")
    print("-" * 70)
    time_years, details = module.compute_expected_time_to_cure(
        DiseaseCategory.PANDEMIC_NOVEL,
        example_multipliers,
        example_p_success,
        base_durations
    )
    print(f"  Expected time to cure: {time_years:.1f} years")
    print(f"  Starting stage: {details['starting_stage']}")
    print(f"  Advances needed: {details['advances_needed']}")
    print(f"  Time per advance: {details['time_per_advance_years']:.1f} years")

    print("\n\nCase Study: Breast Cancer")
    print("-" * 70)
    time_years, details = module.compute_expected_time_to_cure(
        DiseaseCategory.BREAST_CANCER,
        example_multipliers,
        example_p_success,
        base_durations
    )
    print(f"  Expected time to cure: {time_years:.1f} years")
    print(f"  Starting stage: {details['starting_stage']}")
    print(f"  Advances needed: {details['advances_needed']}")
    print(f"  Time per advance: {details['time_per_advance_years']:.1f} years")
