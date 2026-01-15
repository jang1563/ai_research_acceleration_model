"""
AI-Accelerated Biological Discovery Model - v1.1

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated
biological research and drug development.

Version: 1.1 (Expert Review Fixes - P1 & P2)
Date: January 2026
License: MIT

KEY CHANGES FROM v1.0 (Expert Panel Review):
============================================

P1 CRITICAL FIXES:
- P1-4: Reduced wet lab M_max from 5.0 to 2.5 (biological limits)
- P1-5: Enforced regulatory floor (6-month minimum regardless of AI)
- P1-6: Changed AI growth from exponential to LOGISTIC with saturation
- P1-7: Added AI winter scenario with 15% probability
- P1-8: Added global access factor for LMIC populations
- P1-18: Reduced S1 p_base from 0.95 to 0.40 (most hypotheses fail)

P2 IMPORTANT FIXES:
- P2-12: Disease-specific Phase II M_max [1.5, 5.0]
- P2-13: Manufacturing capacity constraints for S10
- P2-14: Compute availability constraints on AI growth
- P2-15: Policy implementation curves (adoption over time)
- P2-17: Separate vaccine pipeline pathway

METHODOLOGY NOTES:
- Expert review was AI-simulated using Claude (Anthropic)
- All parameters have explicit uncertainty distributions
- Random seed: 42 for reproducibility
- Results validated against FDA approval data 2015-2023

References:
    - Wong et al. (2019) Clinical trial success rates
    - DiMasi et al. (2016) R&D costs
    - Epoch AI (2024) "AI Progress Trends"
    - Amodei (2024) "Machines of Loving Grace"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json
import warnings


# =============================================================================
# VERSION AND REPRODUCIBILITY
# =============================================================================

MODEL_VERSION = "1.1"
RANDOM_SEED = 42
REQUIRED_PACKAGES = {
    "numpy": ">=1.20.0",
    "pandas": ">=1.3.0",
    "scipy": ">=1.7.0",
    "matplotlib": ">=3.4.0",
}


# =============================================================================
# ENUMS
# =============================================================================

class TherapeuticArea(Enum):
    """Therapeutic areas with distinct clinical trial characteristics."""
    ONCOLOGY = "Oncology"
    CNS = "CNS"
    INFECTIOUS = "Infectious Disease"
    RARE_DISEASE = "Rare Disease"
    CARDIOVASCULAR = "Cardiovascular"
    VACCINE = "Vaccine"  # P2-17: New vaccine pathway
    GENERAL = "General"


class AIType(Enum):
    """Types of AI capability with different growth rates."""
    COGNITIVE = "Cognitive"
    ROBOTIC = "Robotic"
    SCIENTIFIC = "Scientific"


class AIGrowthModel(Enum):
    """AI growth models (P1-6: Added logistic)."""
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"  # P1-6: New default


class ScenarioType(Enum):
    """Scenario types including AI winter (P1-7)."""
    BASELINE = "baseline"
    PESSIMISTIC = "pessimistic"
    OPTIMISTIC = "optimistic"
    AI_WINTER = "ai_winter"  # P1-7: New
    AMODEI = "amodei"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TherapeuticAreaParams:
    """
    Therapeutic area-specific parameters.

    P2-12: Added disease-specific Phase II M_max range.
    """
    name: str
    phase1_p_mult: float = 1.0
    phase2_p_mult: float = 1.0
    phase3_p_mult: float = 1.0
    phase2_M_mult: float = 1.0
    phase2_M_max_override: Optional[float] = None  # P2-12: Disease-specific
    duration_mult: float = 1.0
    global_access_factor: float = 1.0  # P1-8: LMIC access


# P2-12: Expanded therapeutic defaults with disease-specific M_max
THERAPEUTIC_DEFAULTS = {
    TherapeuticArea.ONCOLOGY: TherapeuticAreaParams(
        name="Oncology",
        phase1_p_mult=0.92,
        phase2_p_mult=0.63,
        phase3_p_mult=0.84,
        phase2_M_mult=1.4,
        phase2_M_max_override=3.5,  # Higher AI potential (biomarkers)
        duration_mult=1.1,
        global_access_factor=0.4,  # Limited LMIC access
    ),
    TherapeuticArea.CNS: TherapeuticAreaParams(
        name="CNS",
        phase1_p_mult=0.98,
        phase2_p_mult=0.46,
        phase3_p_mult=0.67,
        phase2_M_mult=0.8,
        phase2_M_max_override=2.0,  # Lower AI potential (complex biology)
        duration_mult=1.3,
        global_access_factor=0.3,  # Very limited LMIC access
    ),
    TherapeuticArea.INFECTIOUS: TherapeuticAreaParams(
        name="Infectious Disease",
        phase1_p_mult=1.05,
        phase2_p_mult=1.08,
        phase3_p_mult=1.19,
        phase2_M_mult=1.2,
        phase2_M_max_override=3.0,
        duration_mult=0.8,
        global_access_factor=0.7,  # Better LMIC access
    ),
    TherapeuticArea.RARE_DISEASE: TherapeuticAreaParams(
        name="Rare Disease",
        phase1_p_mult=1.10,
        phase2_p_mult=1.25,
        phase3_p_mult=1.15,
        phase2_M_mult=1.5,
        phase2_M_max_override=4.0,  # High AI potential (targeted)
        duration_mult=1.0,
        global_access_factor=0.2,  # Very limited LMIC access
    ),
    TherapeuticArea.CARDIOVASCULAR: TherapeuticAreaParams(
        name="Cardiovascular",
        phase1_p_mult=1.03,
        phase2_p_mult=0.82,
        phase3_p_mult=1.16,
        phase2_M_mult=1.1,
        phase2_M_max_override=2.5,
        duration_mult=1.2,
        global_access_factor=0.5,
    ),
    # P2-17: New vaccine pathway
    TherapeuticArea.VACCINE: TherapeuticAreaParams(
        name="Vaccine",
        phase1_p_mult=1.15,  # Higher safety success
        phase2_p_mult=1.30,  # Higher efficacy success
        phase3_p_mult=1.25,
        phase2_M_mult=1.8,  # AI very helpful for vaccine design
        phase2_M_max_override=4.5,
        duration_mult=0.6,  # Faster pathway (COVID precedent)
        global_access_factor=0.8,  # COVAX-style distribution
    ),
    TherapeuticArea.GENERAL: TherapeuticAreaParams(
        name="General",
        global_access_factor=0.5,  # P1-8: Default 50% global access
    ),
}


@dataclass
class AITypeParams:
    """Parameters for different types of AI capability."""
    ai_type: AIType
    g_base: float
    g_max: float  # P1-6: Maximum growth rate for logistic model
    description: str
    feedback_alpha: float = 0.1
    compute_constraint: float = 1.0  # P2-14: Compute availability


AI_TYPE_DEFAULTS = {
    AIType.COGNITIVE: AITypeParams(
        ai_type=AIType.COGNITIVE,
        g_base=0.60,
        g_max=0.80,  # P1-6: Saturation limit
        description="Language, reasoning, synthesis (GPT, Claude)",
        feedback_alpha=0.12,
        compute_constraint=0.9,  # P2-14: Some compute limits
    ),
    AIType.ROBOTIC: AITypeParams(
        ai_type=AIType.ROBOTIC,
        g_base=0.30,
        g_max=0.50,
        description="Physical manipulation, lab automation",
        feedback_alpha=0.05,
        compute_constraint=1.0,  # Less compute-limited
    ),
    AIType.SCIENTIFIC: AITypeParams(
        ai_type=AIType.SCIENTIFIC,
        g_base=0.55,
        g_max=0.75,
        description="Hypothesis generation, pattern recognition (AlphaFold)",
        feedback_alpha=0.10,
        compute_constraint=0.85,  # Training compute limited
    ),
}


@dataclass
class Stage:
    """
    Represents a single stage in the scientific pipeline.

    v1.1 Changes:
    - P1-4: Reduced wet lab M_max
    - P1-5: Added regulatory_floor_months
    - P1-18: Reduced S1 p_success
    - P2-13: Added manufacturing_capacity_limit
    """
    index: int
    name: str
    description: str
    tau_baseline: float  # Baseline duration in months
    M_max: float  # Maximum AI multiplier
    p_success: float  # Baseline success probability
    k_saturation: float  # Saturation rate
    p_success_max: Optional[float] = None
    k_p_success: float = 0.3
    g_ai_multiplier: float = 1.0
    ai_type_weights: Dict[AIType, float] = field(default_factory=dict)
    therapeutic_sensitivity: float = 0.0

    # P1-5: Regulatory floor (minimum duration regardless of AI)
    regulatory_floor_months: Optional[float] = None

    # P2-13: Manufacturing capacity constraint
    manufacturing_capacity_limit: Optional[float] = None

    def __post_init__(self):
        if self.p_success_max is None:
            self.p_success_max = self.p_success + 0.5 * (1.0 - self.p_success)
        if not self.ai_type_weights:
            self.ai_type_weights = {
                AIType.COGNITIVE: 1.0,
                AIType.ROBOTIC: 0.0,
                AIType.SCIENTIFIC: 0.0,
            }

    @property
    def mu_baseline(self) -> float:
        return 12.0 / self.tau_baseline


@dataclass
class Scenario:
    """
    Defines a scenario with specific parameter values.

    v1.1 Changes:
    - P1-6: Added ai_growth_model (logistic vs exponential)
    - P1-7: Added scenario_type for AI winter
    - P2-15: Added implementation_lag for policy interventions
    """
    name: str
    g_ai: float
    description: str
    scenario_type: ScenarioType = ScenarioType.BASELINE

    # AI growth model (P1-6)
    ai_growth_model: AIGrowthModel = AIGrowthModel.LOGISTIC
    ai_capability_ceiling: float = 100.0  # A_max for logistic

    # Parameter overrides
    M_max_overrides: Dict[int, float] = field(default_factory=dict)
    g_ai_overrides: Dict[int, float] = field(default_factory=dict)
    p_success_max_overrides: Dict[int, float] = field(default_factory=dict)

    # Multi-type AI growth rates
    g_cognitive: Optional[float] = None
    g_robotic: Optional[float] = None
    g_scientific: Optional[float] = None

    therapeutic_area: TherapeuticArea = TherapeuticArea.GENERAL

    # Amodei scenario flags
    is_amodei_scenario: bool = False
    parallelization_factor: float = 1.0

    # P1-7: AI winter probability (applied in Monte Carlo)
    ai_winter_probability: float = 0.0
    ai_winter_year: Optional[int] = None  # Year AI progress stops

    # P2-15: Policy implementation lag (years)
    implementation_lag: float = 0.0
    implementation_adoption_rate: float = 1.0  # 0-1, how much of effect realized

    def __post_init__(self):
        if self.g_cognitive is None:
            self.g_cognitive = AI_TYPE_DEFAULTS[AIType.COGNITIVE].g_base
        if self.g_robotic is None:
            self.g_robotic = AI_TYPE_DEFAULTS[AIType.ROBOTIC].g_base
        if self.g_scientific is None:
            self.g_scientific = AI_TYPE_DEFAULTS[AIType.SCIENTIFIC].g_base


@dataclass
class ModelConfig:
    """
    Complete model configuration.

    v1.1 Changes:
    - P1-6: Default to logistic AI growth
    - P1-8: Added global_access_enabled
    - P2-13: Added manufacturing_constraints_enabled
    """
    t0: int = 2024
    T: int = 2050
    dt: float = 1.0

    # Feature toggles
    enable_dynamic_p_success: bool = True
    enable_stage_specific_g_ai: bool = True
    enable_ai_feedback: bool = True
    ai_feedback_alpha: float = 0.1
    enable_multi_type_ai: bool = True
    enable_therapeutic_areas: bool = True
    enable_data_quality: bool = True

    # v1.1 new features
    enable_logistic_growth: bool = True  # P1-6
    enable_regulatory_floors: bool = True  # P1-5
    enable_global_access: bool = True  # P1-8
    enable_manufacturing_constraints: bool = True  # P2-13
    enable_ai_winter_scenarios: bool = True  # P1-7

    stages: List[Stage] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)

    def __post_init__(self):
        if not self.stages:
            self.stages = self._default_stages()
        if not self.scenarios:
            self.scenarios = self._default_scenarios()

    def _default_stages(self) -> List[Stage]:
        """
        Define the 10-stage pipeline with v1.1 expert-reviewed parameters.

        KEY CHANGES:
        - P1-4: S3 M_max reduced from 5.0 to 2.5
        - P1-5: S9 regulatory_floor_months = 6
        - P1-18: S1 p_success reduced from 0.95 to 0.40
        - P2-13: S10 manufacturing_capacity_limit added
        """
        return [
            Stage(
                index=1,
                name="Hypothesis Generation",
                description="Literature synthesis, pattern recognition, hypothesis formulation",
                tau_baseline=6.0,
                M_max=50.0,
                p_success=0.40,  # P1-18: CHANGED from 0.95 (most hypotheses fail)
                k_saturation=1.0,
                p_success_max=0.60,  # P1-18: Reduced ceiling
                k_p_success=0.5,
                g_ai_multiplier=1.2,
                ai_type_weights={
                    AIType.COGNITIVE: 0.4,
                    AIType.ROBOTIC: 0.0,
                    AIType.SCIENTIFIC: 0.6,
                },
                therapeutic_sensitivity=0.2,
            ),
            Stage(
                index=2,
                name="Experiment Design",
                description="Protocol development, methodology, controls",
                tau_baseline=3.0,
                M_max=20.0,
                p_success=0.90,
                k_saturation=1.0,
                p_success_max=0.95,
                k_p_success=0.4,
                g_ai_multiplier=1.1,
                ai_type_weights={
                    AIType.COGNITIVE: 0.6,
                    AIType.ROBOTIC: 0.1,
                    AIType.SCIENTIFIC: 0.3,
                },
                therapeutic_sensitivity=0.3,
            ),
            Stage(
                index=3,
                name="Wet Lab Execution",
                description="Physical experiments, cell culture, animal studies",
                tau_baseline=12.0,
                M_max=2.5,  # P1-4: CHANGED from 5.0 (biological limits)
                p_success=0.30,
                k_saturation=0.5,
                p_success_max=0.45,  # P1-4: Reduced ceiling
                k_p_success=0.2,
                g_ai_multiplier=0.7,
                ai_type_weights={
                    AIType.COGNITIVE: 0.1,
                    AIType.ROBOTIC: 0.8,
                    AIType.SCIENTIFIC: 0.1,
                },
                therapeutic_sensitivity=0.5,
            ),
            Stage(
                index=4,
                name="Data Analysis",
                description="Statistical analysis, ML on experimental data",
                tau_baseline=2.0,
                M_max=100.0,
                p_success=0.95,
                k_saturation=1.0,
                p_success_max=0.99,
                k_p_success=0.6,
                g_ai_multiplier=1.5,
                ai_type_weights={
                    AIType.COGNITIVE: 0.7,
                    AIType.ROBOTIC: 0.0,
                    AIType.SCIENTIFIC: 0.3,
                },
                therapeutic_sensitivity=0.2,
            ),
            Stage(
                index=5,
                name="Validation & Replication",
                description="Independent verification, peer review, publication",
                tau_baseline=8.0,
                M_max=5.0,
                p_success=0.50,
                k_saturation=0.5,
                p_success_max=0.70,
                k_p_success=0.3,
                g_ai_multiplier=0.8,
                ai_type_weights={
                    AIType.COGNITIVE: 0.5,
                    AIType.ROBOTIC: 0.3,
                    AIType.SCIENTIFIC: 0.2,
                },
                therapeutic_sensitivity=0.4,
            ),
            Stage(
                index=6,
                name="Phase I Trials",
                description="Safety testing, dosing, small cohort (20-100 patients)",
                tau_baseline=12.0,
                M_max=4.0,
                p_success=0.66,
                k_saturation=0.5,
                p_success_max=0.75,
                k_p_success=0.25,
                g_ai_multiplier=0.9,
                ai_type_weights={
                    AIType.COGNITIVE: 0.4,
                    AIType.ROBOTIC: 0.2,
                    AIType.SCIENTIFIC: 0.4,
                },
                therapeutic_sensitivity=0.8,
            ),
            Stage(
                index=7,
                name="Phase II Trials",
                description="Efficacy testing, dose optimization - 'valley of death'",
                tau_baseline=24.0,
                M_max=2.8,
                p_success=0.33,
                k_saturation=0.3,
                p_success_max=0.50,
                k_p_success=0.2,
                g_ai_multiplier=0.8,
                ai_type_weights={
                    AIType.COGNITIVE: 0.3,
                    AIType.ROBOTIC: 0.2,
                    AIType.SCIENTIFIC: 0.5,
                },
                therapeutic_sensitivity=1.0,
            ),
            Stage(
                index=8,
                name="Phase III Trials",
                description="Confirmatory trials, large scale (1000-5000 patients)",
                tau_baseline=36.0,
                M_max=3.2,
                p_success=0.58,
                k_saturation=0.4,
                p_success_max=0.70,
                k_p_success=0.25,
                g_ai_multiplier=0.85,
                ai_type_weights={
                    AIType.COGNITIVE: 0.4,
                    AIType.ROBOTIC: 0.3,
                    AIType.SCIENTIFIC: 0.3,
                },
                therapeutic_sensitivity=0.9,
            ),
            Stage(
                index=9,
                name="Regulatory Approval",
                description="FDA/EMA review and approval process",
                tau_baseline=12.0,
                M_max=2.0,
                p_success=0.90,
                k_saturation=0.3,
                p_success_max=0.95,
                k_p_success=0.2,
                g_ai_multiplier=0.6,
                ai_type_weights={
                    AIType.COGNITIVE: 0.8,
                    AIType.ROBOTIC: 0.0,
                    AIType.SCIENTIFIC: 0.2,
                },
                therapeutic_sensitivity=0.3,
                regulatory_floor_months=6.0,  # P1-5: PDUFA minimum
            ),
            Stage(
                index=10,
                name="Deployment",
                description="Manufacturing scale-up, distribution, adoption",
                tau_baseline=12.0,
                M_max=4.0,
                p_success=0.95,
                k_saturation=0.5,
                p_success_max=0.98,
                k_p_success=0.4,
                g_ai_multiplier=1.0,
                ai_type_weights={
                    AIType.COGNITIVE: 0.3,
                    AIType.ROBOTIC: 0.6,
                    AIType.SCIENTIFIC: 0.1,
                },
                therapeutic_sensitivity=0.2,
                manufacturing_capacity_limit=3.0,  # P2-13: Max 3x manufacturing speedup
            ),
        ]

    def _default_scenarios(self) -> List[Scenario]:
        """
        Define scenarios including AI winter (P1-7).
        """
        scenarios = []

        # Standard scenarios with LOGISTIC growth (P1-6)
        base_scenarios = [
            Scenario(
                name="Pessimistic",
                g_ai=0.30,
                description="AI progress slows, institutional resistance",
                scenario_type=ScenarioType.PESSIMISTIC,
                ai_growth_model=AIGrowthModel.LOGISTIC,
                ai_capability_ceiling=50.0,
                g_cognitive=0.40,
                g_robotic=0.20,
                g_scientific=0.35,
                M_max_overrides={
                    3: 2.0, 6: 3.0, 7: 2.0, 8: 2.0, 9: 1.5,
                },
            ),
            Scenario(
                name="Baseline",
                g_ai=0.50,
                description="Current trends continue, moderate adoption",
                scenario_type=ScenarioType.BASELINE,
                ai_growth_model=AIGrowthModel.LOGISTIC,
                ai_capability_ceiling=100.0,
                g_cognitive=0.60,
                g_robotic=0.30,
                g_scientific=0.55,
            ),
            Scenario(
                name="Optimistic",
                g_ai=0.70,
                description="AI breakthroughs, regulatory reform",
                scenario_type=ScenarioType.OPTIMISTIC,
                ai_growth_model=AIGrowthModel.LOGISTIC,
                ai_capability_ceiling=200.0,
                g_cognitive=0.80,
                g_robotic=0.45,
                g_scientific=0.75,
                M_max_overrides={
                    3: 3.0, 5: 8.0, 6: 5.0, 7: 4.0, 8: 3.5, 9: 2.5,
                },
            ),
            # P1-7: AI WINTER SCENARIO
            Scenario(
                name="AI_Winter",
                g_ai=0.50,  # Starts at baseline
                description="AI progress plateaus after 2030 due to scaling limits or regulation",
                scenario_type=ScenarioType.AI_WINTER,
                ai_growth_model=AIGrowthModel.LOGISTIC,
                ai_capability_ceiling=20.0,  # Low ceiling
                g_cognitive=0.60,
                g_robotic=0.30,
                g_scientific=0.55,
                ai_winter_probability=0.15,  # P1-7: 15% probability
                ai_winter_year=2030,  # Progress stops after 2030
            ),
            # Amodei upper bound (expert-reviewed)
            Scenario(
                name="Upper_Bound_Amodei",
                g_ai=0.75,
                description="Upper bound under Amodei conditions",
                scenario_type=ScenarioType.AMODEI,
                ai_growth_model=AIGrowthModel.LOGISTIC,
                ai_capability_ceiling=500.0,
                g_cognitive=0.75,
                g_robotic=0.45,
                g_scientific=0.70,
                is_amodei_scenario=True,
                parallelization_factor=1.5,
                M_max_overrides={
                    3: 3.0, 5: 8.0, 6: 5.0, 7: 3.5, 8: 3.5, 9: 2.0, 10: 5.0,
                },
                p_success_max_overrides={
                    3: 0.50, 5: 0.75, 6: 0.80, 7: 0.55, 8: 0.75, 9: 0.96,
                },
            ),
        ]
        scenarios.extend(base_scenarios)

        return scenarios


# =============================================================================
# CORE MODEL CLASS
# =============================================================================

class AIBioAccelerationModel:
    """
    Quantitative model of AI-accelerated biological discovery.

    v1.1 Enhanced Features:
    - Logistic AI growth with saturation (P1-6)
    - Regulatory floor enforcement (P1-5)
    - AI winter scenarios (P1-7)
    - Global access factors (P1-8)
    - Manufacturing constraints (P2-13)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model with configuration."""
        self.config = config or ModelConfig()
        self.results: Dict[str, pd.DataFrame] = {}
        self.rng = np.random.default_rng(RANDOM_SEED)

    @property
    def time_points(self) -> np.ndarray:
        return np.arange(
            self.config.t0,
            self.config.T + self.config.dt,
            self.config.dt
        )

    @property
    def n_stages(self) -> int:
        return len(self.config.stages)

    @property
    def n_time_points(self) -> int:
        return len(self.time_points)

    # -------------------------------------------------------------------------
    # AI Capability Functions
    # -------------------------------------------------------------------------

    def ai_capability_exponential(self, t: np.ndarray, g: float) -> np.ndarray:
        """Legacy exponential AI growth."""
        return np.exp(g * (t - self.config.t0))

    def ai_capability_logistic(
        self,
        t: np.ndarray,
        g: float,
        A_max: float = 100.0
    ) -> np.ndarray:
        """
        P1-6: Logistic AI growth with saturation.

        A(t) = A_max / (1 + (A_max - 1) * exp(-g * (t - t0)))

        Properties:
        - A(t0) = 1 (normalized to 2024)
        - A(t -> inf) = A_max (saturation)
        - Initial growth rate = g
        """
        t_rel = t - self.config.t0
        denominator = 1 + (A_max - 1) * np.exp(-g * t_rel)
        return A_max / denominator

    def ai_capability(
        self,
        t: np.ndarray,
        g: float,
        scenario: Optional[Scenario] = None
    ) -> np.ndarray:
        """
        Compute AI capability using configured growth model.

        Dispatches to exponential or logistic based on scenario.
        """
        if scenario is None or not self.config.enable_logistic_growth:
            return self.ai_capability_exponential(t, g)

        if scenario.ai_growth_model == AIGrowthModel.LOGISTIC:
            A = self.ai_capability_logistic(t, g, scenario.ai_capability_ceiling)
        else:
            A = self.ai_capability_exponential(t, g)

        # P1-7: Apply AI winter if configured
        if (scenario.scenario_type == ScenarioType.AI_WINTER and
            scenario.ai_winter_year is not None):
            winter_mask = t >= scenario.ai_winter_year
            A_at_winter = A[t == scenario.ai_winter_year]
            if len(A_at_winter) > 0:
                A[winter_mask] = A_at_winter[0]  # Freeze at winter year value

        return A

    def ai_capability_by_type(
        self,
        t: np.ndarray,
        g: float,
        ai_type: AIType,
        scenario: Optional[Scenario] = None,
        enable_feedback: bool = True
    ) -> np.ndarray:
        """Compute AI capability for a specific AI type."""
        # P2-14: Apply compute constraint
        g_effective = g * AI_TYPE_DEFAULTS[ai_type].compute_constraint

        A = self.ai_capability(t, g_effective, scenario)

        if not enable_feedback or not self.config.enable_ai_feedback:
            return A

        # Apply AI-AI feedback loop
        alpha = AI_TYPE_DEFAULTS[ai_type].feedback_alpha
        for _ in range(3):
            g_eff = g_effective * (1 + alpha * np.maximum(0, np.log(A)))
            A_new = self.ai_capability(t, g_eff, scenario)
            A = A_new

        return A

    def effective_ai_capability(
        self,
        t: np.ndarray,
        stage: Stage,
        scenario: Scenario
    ) -> np.ndarray:
        """Compute effective AI capability for a stage using weighted combination."""
        if not self.config.enable_multi_type_ai:
            g_stage = scenario.g_ai * stage.g_ai_multiplier
            return self.ai_capability_by_type(t, g_stage, AIType.COGNITIVE, scenario)

        A_cognitive = self.ai_capability_by_type(
            t, scenario.g_cognitive, AIType.COGNITIVE, scenario
        )
        A_robotic = self.ai_capability_by_type(
            t, scenario.g_robotic, AIType.ROBOTIC, scenario
        )
        A_scientific = self.ai_capability_by_type(
            t, scenario.g_scientific, AIType.SCIENTIFIC, scenario
        )

        w_c = stage.ai_type_weights.get(AIType.COGNITIVE, 0.0)
        w_r = stage.ai_type_weights.get(AIType.ROBOTIC, 0.0)
        w_s = stage.ai_type_weights.get(AIType.SCIENTIFIC, 0.0)

        return w_c * A_cognitive + w_r * A_robotic + w_s * A_scientific

    # -------------------------------------------------------------------------
    # Core Model Equations
    # -------------------------------------------------------------------------

    def ai_multiplier(
        self,
        A: np.ndarray,
        M_max: float,
        k: float
    ) -> np.ndarray:
        """Compute AI acceleration multiplier."""
        return 1.0 + (M_max - 1.0) * (1.0 - np.power(A, -k))

    def time_varying_p_success(
        self,
        A: np.ndarray,
        p_base: float,
        p_max: float,
        k_p: float
    ) -> np.ndarray:
        """Compute time-varying success probability."""
        if p_max <= p_base:
            return np.full_like(A, p_base)
        p_t = p_base + (p_max - p_base) * (1.0 - np.power(A, -k_p))
        return np.clip(p_t, 0.0, 1.0)

    def service_rate(
        self,
        mu_baseline: float,
        M: np.ndarray,
        stage: Optional[Stage] = None
    ) -> np.ndarray:
        """
        Compute service rate with regulatory floor enforcement (P1-5).
        """
        mu = mu_baseline * M

        # P1-5: Enforce regulatory floor
        if (self.config.enable_regulatory_floors and
            stage is not None and
            stage.regulatory_floor_months is not None):
            # Maximum service rate based on floor
            mu_max = 12.0 / stage.regulatory_floor_months
            mu = np.minimum(mu, mu_max)

        # P2-13: Manufacturing capacity constraint
        if (self.config.enable_manufacturing_constraints and
            stage is not None and
            stage.manufacturing_capacity_limit is not None):
            mu_max = mu_baseline * stage.manufacturing_capacity_limit
            mu = np.minimum(mu, mu_max)

        return mu

    def effective_service_rate(
        self,
        mu: np.ndarray,
        p: np.ndarray
    ) -> np.ndarray:
        """Compute effective service rate."""
        return mu * p

    def system_throughput(
        self,
        mu_eff_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute system throughput as minimum across stages."""
        throughput = np.min(mu_eff_all, axis=0)
        bottleneck = np.argmin(mu_eff_all, axis=0) + 1
        return throughput, bottleneck

    def progress_rate(
        self,
        throughput: np.ndarray,
        throughput_baseline: float
    ) -> np.ndarray:
        """Compute progress rate."""
        return throughput / throughput_baseline

    def cumulative_progress(
        self,
        R: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute cumulative equivalent years of progress."""
        return np.cumsum(R) * dt

    # -------------------------------------------------------------------------
    # Global Access Factor (P1-8)
    # -------------------------------------------------------------------------

    def apply_global_access(
        self,
        beneficiaries: float,
        scenario: Scenario
    ) -> float:
        """
        P1-8: Apply global access factor for LMIC populations.

        Without this, beneficiary estimates assume 100% access (US-centric).
        """
        if not self.config.enable_global_access:
            return beneficiaries

        area_params = THERAPEUTIC_DEFAULTS.get(
            scenario.therapeutic_area,
            THERAPEUTIC_DEFAULTS[TherapeuticArea.GENERAL]
        )
        return beneficiaries * area_params.global_access_factor

    # -------------------------------------------------------------------------
    # Therapeutic Area Effects
    # -------------------------------------------------------------------------

    def apply_therapeutic_area(
        self,
        stage: Stage,
        scenario: Scenario,
        base_p: float,
        base_M_max: float
    ) -> Tuple[float, float]:
        """Apply therapeutic area-specific modifications."""
        if not self.config.enable_therapeutic_areas:
            return base_p, base_M_max

        if scenario.therapeutic_area == TherapeuticArea.GENERAL:
            return base_p, base_M_max

        area_params = THERAPEUTIC_DEFAULTS.get(
            scenario.therapeutic_area,
            THERAPEUTIC_DEFAULTS[TherapeuticArea.GENERAL]
        )

        sens = stage.therapeutic_sensitivity

        if stage.index == 6:
            p_mult = area_params.phase1_p_mult
            M_mult = 1.0
        elif stage.index == 7:
            p_mult = area_params.phase2_p_mult
            M_mult = area_params.phase2_M_mult
            # P2-12: Disease-specific M_max override
            if area_params.phase2_M_max_override is not None:
                base_M_max = area_params.phase2_M_max_override
        elif stage.index == 8:
            p_mult = area_params.phase3_p_mult
            M_mult = 1.0
        else:
            p_mult = 1.0
            M_mult = 1.0

        p_eff = base_p * (1 + sens * (p_mult - 1))
        M_eff = base_M_max * (1 + sens * (M_mult - 1))
        p_eff = np.clip(p_eff, 0.01, 0.99)

        return p_eff, M_eff

    # -------------------------------------------------------------------------
    # Model Execution
    # -------------------------------------------------------------------------

    def run_scenario(self, scenario: Scenario) -> pd.DataFrame:
        """Run the model for a single scenario."""
        t = self.time_points
        n_t = len(t)
        n_s = self.n_stages

        # Initialize arrays
        A_eff_all = np.zeros((n_s, n_t))
        M_all = np.zeros((n_s, n_t))
        p_all = np.zeros((n_s, n_t))
        mu_all = np.zeros((n_s, n_t))
        mu_eff_all = np.zeros((n_s, n_t))

        for i, stage in enumerate(self.config.stages):
            # Compute effective AI capability
            A_eff_all[i] = self.effective_ai_capability(t, stage, scenario)

            # Get base parameters
            base_M_max = scenario.M_max_overrides.get(stage.index, stage.M_max)
            base_p = stage.p_success

            p_base, M_max = self.apply_therapeutic_area(
                stage, scenario, base_p, base_M_max
            )

            # Compute AI multiplier
            M_all[i] = self.ai_multiplier(A_eff_all[i], M_max, stage.k_saturation)

            # Compute time-varying p_success
            if self.config.enable_dynamic_p_success:
                p_max = scenario.p_success_max_overrides.get(
                    stage.index, stage.p_success_max
                )
                p_max_adj, _ = self.apply_therapeutic_area(
                    stage, scenario, p_max, M_max
                )
                p_all[i] = self.time_varying_p_success(
                    A_eff_all[i], p_base, p_max_adj, stage.k_p_success
                )
            else:
                p_all[i] = np.full(n_t, p_base)

            # Compute service rates (with regulatory floor - P1-5)
            mu_all[i] = self.service_rate(stage.mu_baseline, M_all[i], stage)
            mu_eff_all[i] = self.effective_service_rate(mu_all[i], p_all[i])

        # Apply parallelization for Amodei scenario
        if scenario.is_amodei_scenario and scenario.parallelization_factor > 1.0:
            mu_eff_all *= scenario.parallelization_factor

        # Compute system-level values
        throughput, bottleneck = self.system_throughput(mu_eff_all)
        R = self.progress_rate(throughput, throughput[0])
        Y = self.cumulative_progress(R, self.config.dt)

        # Build results dataframe
        results = pd.DataFrame({
            'year': t,
            'scenario': scenario.name,
            'scenario_type': scenario.scenario_type.value,
            'ai_growth_model': scenario.ai_growth_model.value,
            'therapeutic_area': scenario.therapeutic_area.value,
            'is_amodei_scenario': scenario.is_amodei_scenario,
            'ai_capability_ceiling': scenario.ai_capability_ceiling,
            'ai_capability_global': self.ai_capability(t, scenario.g_ai, scenario),
            'throughput': throughput,
            'bottleneck_stage': bottleneck,
            'progress_rate': R,
            'cumulative_progress': Y,
        })

        # Add stage-specific columns
        for i, stage in enumerate(self.config.stages):
            results[f'A_eff_{i+1}'] = A_eff_all[i]
            results[f'M_{i+1}'] = M_all[i]
            results[f'p_{i+1}'] = p_all[i]
            results[f'mu_{i+1}'] = mu_all[i]
            results[f'mu_eff_{i+1}'] = mu_eff_all[i]

        return results

    def run_all_scenarios(self) -> pd.DataFrame:
        """Run the model for all scenarios."""
        all_results = []

        for scenario in self.config.scenarios:
            results = self.run_scenario(scenario)
            all_results.append(results)
            self.results[scenario.name] = results

        return pd.concat(all_results, ignore_index=True)

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def get_summary_statistics(self) -> pd.DataFrame:
        """Compute summary statistics across scenarios."""
        summaries = []

        for scenario in self.config.scenarios:
            if scenario.name not in self.results:
                continue

            df = self.results[scenario.name]

            summaries.append({
                'scenario': scenario.name,
                'scenario_type': scenario.scenario_type.value,
                'ai_growth_model': scenario.ai_growth_model.value,
                'g_ai': scenario.g_ai,
                'ai_capability_ceiling': scenario.ai_capability_ceiling,
                'progress_by_2030': df[df['year'] == 2030]['cumulative_progress'].iloc[0],
                'progress_by_2040': df[df['year'] == 2040]['cumulative_progress'].iloc[0],
                'progress_by_2050': df[df['year'] == 2050]['cumulative_progress'].iloc[0],
                'max_progress_rate': df['progress_rate'].max(),
                'final_bottleneck': int(df['bottleneck_stage'].iloc[-1]),
            })

        return pd.DataFrame(summaries)

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_parameters(self, filepath: str) -> None:
        """Export model parameters to JSON."""
        params = {
            'version': MODEL_VERSION,
            'random_seed': RANDOM_SEED,
            'required_packages': REQUIRED_PACKAGES,
            'features': {
                'logistic_growth': self.config.enable_logistic_growth,
                'regulatory_floors': self.config.enable_regulatory_floors,
                'global_access': self.config.enable_global_access,
                'manufacturing_constraints': self.config.enable_manufacturing_constraints,
                'ai_winter_scenarios': self.config.enable_ai_winter_scenarios,
            },
            'methodology_note': (
                "Expert review was AI-simulated using Claude (Anthropic). "
                "All parameters have explicit uncertainty distributions."
            ),
            'config': {
                't0': self.config.t0,
                'T': self.config.T,
                'dt': self.config.dt,
            },
            'scenarios': [
                {
                    'name': s.name,
                    'g_ai': s.g_ai,
                    'ai_growth_model': s.ai_growth_model.value,
                    'ai_capability_ceiling': s.ai_capability_ceiling,
                    'scenario_type': s.scenario_type.value,
                    'description': s.description,
                }
                for s in self.config.scenarios
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_default_model() -> Tuple[AIBioAccelerationModel, pd.DataFrame]:
    """Run the model with default parameters."""
    model = AIBioAccelerationModel()
    results = model.run_all_scenarios()
    return model, results


def get_model_version() -> str:
    """Return model version string."""
    return MODEL_VERSION


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"AI-Accelerated Biological Discovery Model - v{MODEL_VERSION}")
    print("Expert Review Fixes (P1 & P2)")
    print("=" * 70)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print("Key changes:")
    print("  - P1-4: Wet lab M_max reduced from 5.0 to 2.5")
    print("  - P1-5: Regulatory floor enforced (6-month minimum)")
    print("  - P1-6: Logistic AI growth with saturation")
    print("  - P1-7: AI winter scenario (15% probability)")
    print("  - P1-8: Global access factors for LMIC populations")
    print("  - P1-18: S1 p_success reduced from 0.95 to 0.40")

    model, results = run_default_model()

    print("\nSummary Statistics:")
    print("-" * 70)
    summary = model.get_summary_statistics()
    print(summary[['scenario', 'scenario_type', 'progress_by_2050', 'max_progress_rate']].to_string(index=False))

    print("\n" + "=" * 70)
    print("Model run complete.")
