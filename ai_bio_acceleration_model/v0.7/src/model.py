"""
AI-Accelerated Biological Discovery Model - v0.7

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated
biological research and drug development.

Version: 0.7 (Pipeline Iteration + Amodei Scenario)
Date: January 2026
License: MIT

Key Changes from v0.6:
- Pipeline Iteration Module: Models failure/rework dynamics
- Amodei Scenario: Optimistic upper-bound matching Dario Amodei's predictions
  - Based on "Machines of Loving Grace" (Oct 2024)
  - 10x acceleration target with regulatory reform assumptions
- Semi-Markov process for stage transitions
- Effective throughput accounting for rework cycles

v0.6 Features (retained):
- Data Quality Module D(t): Models how data quality affects all stages
- Multi-type AI: Cognitive (g_c), Robotic (g_r), Scientific (g_s)
- Therapeutic area differentiation
- Dynamic success rates

References:
    - Amodei (2024) "Machines of Loving Grace" - 10x acceleration prediction
    - Epoch AI (2024) "AI Progress Trends"
    - Wong et al. (2019) Clinical trial success rates
    - DiMasi et al. (2016) R&D costs
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json

# Import modules
from data_quality import DataQualityModule, DataQualityConfig
from pipeline_iteration import (
    PipelineIterationModule,
    PipelineIterationConfig,
    ReworkConfig,
    DEFAULT_REWORK_CONFIG
)


# =============================================================================
# ENUMS FOR THERAPEUTIC AREAS AND AI TYPES
# =============================================================================

class TherapeuticArea(Enum):
    """Therapeutic areas with distinct clinical trial characteristics."""
    ONCOLOGY = "Oncology"
    CNS = "CNS"
    INFECTIOUS = "Infectious Disease"
    RARE_DISEASE = "Rare Disease"
    CARDIOVASCULAR = "Cardiovascular"
    GENERAL = "General"


class AIType(Enum):
    """Types of AI capability with different growth rates."""
    COGNITIVE = "Cognitive"
    ROBOTIC = "Robotic"
    SCIENTIFIC = "Scientific"


# =============================================================================
# DATA CLASSES FOR MODEL PARAMETERS
# =============================================================================

@dataclass
class TherapeuticAreaParams:
    """Therapeutic area-specific parameters."""
    name: str
    phase1_p_mult: float = 1.0
    phase2_p_mult: float = 1.0
    phase3_p_mult: float = 1.0
    phase2_M_mult: float = 1.0
    duration_mult: float = 1.0


THERAPEUTIC_DEFAULTS = {
    TherapeuticArea.ONCOLOGY: TherapeuticAreaParams(
        name="Oncology",
        phase1_p_mult=0.92,
        phase2_p_mult=0.63,
        phase3_p_mult=0.84,
        phase2_M_mult=1.4,
        duration_mult=1.1,
    ),
    TherapeuticArea.CNS: TherapeuticAreaParams(
        name="CNS",
        phase1_p_mult=0.98,
        phase2_p_mult=0.46,
        phase3_p_mult=0.67,
        phase2_M_mult=0.8,
        duration_mult=1.3,
    ),
    TherapeuticArea.INFECTIOUS: TherapeuticAreaParams(
        name="Infectious Disease",
        phase1_p_mult=1.05,
        phase2_p_mult=1.08,
        phase3_p_mult=1.19,
        phase2_M_mult=1.2,
        duration_mult=0.8,
    ),
    TherapeuticArea.RARE_DISEASE: TherapeuticAreaParams(
        name="Rare Disease",
        phase1_p_mult=1.10,
        phase2_p_mult=1.25,
        phase3_p_mult=1.15,
        phase2_M_mult=1.5,
        duration_mult=1.0,
    ),
    TherapeuticArea.CARDIOVASCULAR: TherapeuticAreaParams(
        name="Cardiovascular",
        phase1_p_mult=1.03,
        phase2_p_mult=0.82,
        phase3_p_mult=1.16,
        phase2_M_mult=1.1,
        duration_mult=1.2,
    ),
    TherapeuticArea.GENERAL: TherapeuticAreaParams(
        name="General",
    ),
}


@dataclass
class AITypeParams:
    """Parameters for different types of AI capability."""
    ai_type: AIType
    g_base: float
    description: str
    feedback_alpha: float = 0.1


AI_TYPE_DEFAULTS = {
    AIType.COGNITIVE: AITypeParams(
        ai_type=AIType.COGNITIVE,
        g_base=0.60,
        description="Language, reasoning, synthesis (GPT, Claude)",
        feedback_alpha=0.12,
    ),
    AIType.ROBOTIC: AITypeParams(
        ai_type=AIType.ROBOTIC,
        g_base=0.30,
        description="Physical manipulation, lab automation",
        feedback_alpha=0.05,
    ),
    AIType.SCIENTIFIC: AITypeParams(
        ai_type=AIType.SCIENTIFIC,
        g_base=0.55,
        description="Hypothesis generation, pattern recognition (AlphaFold)",
        feedback_alpha=0.10,
    ),
}


@dataclass
class Stage:
    """Represents a single stage in the scientific pipeline."""
    index: int
    name: str
    description: str
    tau_baseline: float
    M_max: float
    p_success: float
    k_saturation: float
    p_success_max: Optional[float] = None
    k_p_success: float = 0.3
    g_ai_multiplier: float = 1.0
    ai_type_weights: Dict[AIType, float] = field(default_factory=dict)
    therapeutic_sensitivity: float = 0.0

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

    v0.7 NEW: Added 'Amodei' scenario type for optimistic upper-bound
    """
    name: str
    g_ai: float
    description: str
    M_max_overrides: Dict[int, float] = field(default_factory=dict)
    g_ai_overrides: Dict[int, float] = field(default_factory=dict)
    p_success_max_overrides: Dict[int, float] = field(default_factory=dict)
    g_cognitive: Optional[float] = None
    g_robotic: Optional[float] = None
    g_scientific: Optional[float] = None
    therapeutic_area: TherapeuticArea = TherapeuticArea.GENERAL

    # v0.7 NEW: Scenario type flag
    is_amodei_scenario: bool = False

    # v0.7 NEW: Parallelization factor (Amodei emphasizes "massive parallelization")
    parallelization_factor: float = 1.0

    def __post_init__(self):
        if self.g_cognitive is None:
            self.g_cognitive = AI_TYPE_DEFAULTS[AIType.COGNITIVE].g_base
        if self.g_robotic is None:
            self.g_robotic = AI_TYPE_DEFAULTS[AIType.ROBOTIC].g_base
        if self.g_scientific is None:
            self.g_scientific = AI_TYPE_DEFAULTS[AIType.SCIENTIFIC].g_base


@dataclass
class ModelConfig:
    """Complete model configuration."""
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
    data_quality_config: DataQualityConfig = field(default_factory=DataQualityConfig)

    # v0.7 NEW: Pipeline iteration
    enable_pipeline_iteration: bool = True
    pipeline_iteration_config: PipelineIterationConfig = field(
        default_factory=PipelineIterationConfig
    )

    stages: List[Stage] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)

    def __post_init__(self):
        if not self.stages:
            self.stages = self._default_stages()
        if not self.scenarios:
            self.scenarios = self._default_scenarios()

    def _default_stages(self) -> List[Stage]:
        """Define the 10-stage pipeline."""
        return [
            Stage(
                index=1,
                name="Hypothesis Generation",
                description="Literature synthesis, pattern recognition, hypothesis formulation",
                tau_baseline=6.0,
                M_max=50.0,
                p_success=0.95,
                k_saturation=1.0,
                p_success_max=0.98,
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
                M_max=5.0,
                p_success=0.30,
                k_saturation=0.5,
                p_success_max=0.50,
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
            ),
        ]

    def _default_scenarios(self) -> List[Scenario]:
        """
        Define scenarios including the new Amodei Scenario.

        v0.7: Adds Amodei Scenario based on "Machines of Loving Grace"
        Key assumptions from Amodei:
        - 10x acceleration is achievable
        - Regulatory reform enables faster trials
        - Drugs with larger effect sizes streamline approval
        - Massive parallelization of R&D
        """
        scenarios = []

        # Base scenarios (unchanged from v0.6)
        base_scenarios = [
            Scenario(
                name="Pessimistic",
                g_ai=0.30,
                description="AI progress slows, institutional resistance",
                g_cognitive=0.40,
                g_robotic=0.20,
                g_scientific=0.35,
                therapeutic_area=TherapeuticArea.GENERAL,
                M_max_overrides={
                    3: 3.5, 6: 3.0, 7: 2.0, 8: 2.0, 9: 1.5,
                },
            ),
            Scenario(
                name="Baseline",
                g_ai=0.50,
                description="Current trends continue, moderate adoption",
                g_cognitive=0.60,
                g_robotic=0.30,
                g_scientific=0.55,
                therapeutic_area=TherapeuticArea.GENERAL,
            ),
            Scenario(
                name="Optimistic",
                g_ai=0.70,
                description="AI breakthroughs, regulatory reform",
                g_cognitive=0.80,
                g_robotic=0.45,
                g_scientific=0.75,
                therapeutic_area=TherapeuticArea.GENERAL,
                M_max_overrides={
                    3: 8.0, 5: 8.0, 6: 5.0, 7: 5.0, 8: 3.5, 9: 3.0,
                },
            ),
            # ===================================================================
            # v0.7 REVISED: UPPER BOUND (AMODEI CONDITIONS)
            # Based on Dario Amodei's "Machines of Loving Grace" (Oct 2024)
            # Target: 10x acceleration (50-100 years progress in 5-10 years)
            #
            # EXPERT REVIEW REVISIONS:
            # - A1: g_cognitive capped at 0.75 (was 0.90) - more realistic
            # - C1: Phase II M_max capped at 3.5 (was 5.0) - regulatory limits
            # - C3: Regulatory M_max capped at 2.0 (was 4.0) - PDUFA constraints
            # - A2: Parallelization with diminishing returns: 1.5 effective (was 2.0)
            # ===================================================================
            Scenario(
                name="Upper_Bound_Amodei",
                g_ai=0.75,
                description="Upper bound under Amodei conditions (regulatory reform + breakthrough drugs)",
                g_cognitive=0.75,   # REVISED: Capped per Expert A1 (was 0.90)
                g_robotic=0.45,     # REVISED: Slightly reduced (was 0.50)
                g_scientific=0.70,  # REVISED: More conservative (was 0.85)
                therapeutic_area=TherapeuticArea.GENERAL,
                is_amodei_scenario=True,
                parallelization_factor=1.5,  # REVISED: Diminishing returns (was 2.0)
                # REVISED M_max values per Expert C1, C3
                M_max_overrides={
                    3: 8.0,    # Lab automation (was 10.0)
                    5: 8.0,    # AI-accelerated validation (was 10.0)
                    6: 5.0,    # Phase I - adaptive designs (was 6.0)
                    7: 3.5,    # REVISED: Phase II capped per Expert C1 (was 5.0)
                    8: 3.5,    # Phase III - adaptive designs (was 4.5)
                    9: 2.0,    # REVISED: Regulatory capped per Expert C3 (was 4.0)
                    10: 5.0,   # Manufacturing automation (was 6.0)
                },
                # Higher success rate ceilings (better drugs, better targeting)
                p_success_max_overrides={
                    3: 0.55,   # Better experiment design (was 0.60)
                    5: 0.75,   # AI improves reproducibility (was 0.80)
                    6: 0.80,   # Better patient selection (was 0.85)
                    7: 0.55,   # REVISED: More realistic Phase II (was 0.65)
                    8: 0.75,   # Better Phase III success (was 0.80)
                    9: 0.96,   # Faster approval for effective drugs (was 0.98)
                },
            ),
        ]
        scenarios.extend(base_scenarios)

        # Therapeutic area-specific Baseline scenarios
        for area in [TherapeuticArea.ONCOLOGY, TherapeuticArea.CNS,
                     TherapeuticArea.INFECTIOUS, TherapeuticArea.RARE_DISEASE]:
            scenarios.append(Scenario(
                name=f"Baseline_{area.value.replace(' ', '_')}",
                g_ai=0.50,
                description=f"Baseline scenario for {area.value}",
                g_cognitive=0.60,
                g_robotic=0.30,
                g_scientific=0.55,
                therapeutic_area=area,
            ))

        return scenarios


# =============================================================================
# CORE MODEL CLASS
# =============================================================================

class AIBioAccelerationModel:
    """
    Quantitative model of AI-accelerated biological discovery.

    v0.7 Enhanced Features:
    - Pipeline iteration with rework dynamics
    - Amodei scenario for 10x acceleration upper-bound
    - Effective throughput accounting for failure cycles
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model with configuration."""
        self.config = config or ModelConfig()
        self.results: Dict[str, pd.DataFrame] = {}

        # v0.6: Initialize data quality module
        if self.config.enable_data_quality:
            self.data_quality = DataQualityModule(self.config.data_quality_config)
        else:
            self.data_quality = None

        # v0.7: Initialize pipeline iteration module
        if self.config.enable_pipeline_iteration:
            self.pipeline_iteration = PipelineIterationModule(
                self.config.pipeline_iteration_config
            )
        else:
            self.pipeline_iteration = None

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
    # Multi-Type AI Capability
    # -------------------------------------------------------------------------

    def ai_capability_by_type(
        self,
        t: np.ndarray,
        g: float,
        ai_type: AIType,
        enable_feedback: bool = True
    ) -> np.ndarray:
        """Compute AI capability for a specific AI type."""
        if not enable_feedback or not self.config.enable_ai_feedback:
            return np.exp(g * (t - self.config.t0))

        alpha = AI_TYPE_DEFAULTS[ai_type].feedback_alpha
        A = np.exp(g * (t - self.config.t0))

        for _ in range(5):
            g_eff = g * (1 + alpha * np.maximum(0, np.log(A)))
            dt = self.config.dt
            A_new = np.ones_like(t)
            for i in range(1, len(t)):
                A_new[i] = A_new[i-1] * np.exp(g_eff[i-1] * dt)
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
            return self.ai_capability_by_type(t, g_stage, AIType.COGNITIVE)

        A_cognitive = self.ai_capability_by_type(
            t, scenario.g_cognitive, AIType.COGNITIVE
        )
        A_robotic = self.ai_capability_by_type(
            t, scenario.g_robotic, AIType.ROBOTIC
        )
        A_scientific = self.ai_capability_by_type(
            t, scenario.g_scientific, AIType.SCIENTIFIC
        )

        w_c = stage.ai_type_weights.get(AIType.COGNITIVE, 0.0)
        w_r = stage.ai_type_weights.get(AIType.ROBOTIC, 0.0)
        w_s = stage.ai_type_weights.get(AIType.SCIENTIFIC, 0.0)

        A_eff = w_c * A_cognitive + w_r * A_robotic + w_s * A_scientific
        return A_eff

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
    # Core Model Equations
    # -------------------------------------------------------------------------

    def ai_capability(self, t: np.ndarray, g: float) -> np.ndarray:
        """Legacy: Compute single AI capability."""
        return self.ai_capability_by_type(t, g, AIType.COGNITIVE)

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
        M: np.ndarray
    ) -> np.ndarray:
        """Compute service rate."""
        return mu_baseline * M

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
    # Model Execution
    # -------------------------------------------------------------------------

    def run_scenario(self, scenario: Scenario) -> pd.DataFrame:
        """
        Run the model for a single scenario.

        v0.7 Changes:
        - Applies parallelization factor for Amodei scenario
        - Computes effective throughput with pipeline iteration
        """
        t = self.time_points
        n_t = len(t)
        n_s = self.n_stages

        # Initialize arrays
        A_eff_all = np.zeros((n_s, n_t))
        M_all = np.zeros((n_s, n_t))
        p_all = np.zeros((n_s, n_t))
        mu_all = np.zeros((n_s, n_t))
        mu_eff_all = np.zeros((n_s, n_t))
        DQM_all = np.zeros((n_s, n_t))

        # v0.6: Compute data quality trajectory
        D_t = np.ones(n_t)
        if self.config.enable_data_quality and self.data_quality is not None:
            A_global = self.ai_capability(t, scenario.g_ai)
            for j in range(n_t):
                D_t[j] = self.data_quality.compute_D(t[j], A_global[j])

        for i, stage in enumerate(self.config.stages):
            # Step 1: Compute effective AI capability
            A_eff_all[i] = self.effective_ai_capability(t, stage, scenario)

            # Step 2: Get base parameters with therapeutic area adjustment
            base_M_max = scenario.M_max_overrides.get(stage.index, stage.M_max)
            base_p = stage.p_success

            p_base, M_max = self.apply_therapeutic_area(
                stage, scenario, base_p, base_M_max
            )

            # Step 3: Compute AI multiplier
            M_all[i] = self.ai_multiplier(A_eff_all[i], M_max, stage.k_saturation)

            # Step 4: Compute time-varying p_success
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

            # Step 5: Compute data quality multiplier (v0.6)
            if self.config.enable_data_quality and self.data_quality is not None:
                for j in range(n_t):
                    DQM_all[i, j] = self.data_quality.compute_DQM(stage.index, D_t[j])
            else:
                DQM_all[i] = np.ones(n_t)

            # Step 6: Compute service rates
            mu_all[i] = self.service_rate(stage.mu_baseline, M_all[i]) * DQM_all[i]
            mu_eff_all[i] = self.effective_service_rate(mu_all[i], p_all[i])

        # v0.7: Apply parallelization factor for Amodei scenario
        if scenario.is_amodei_scenario and scenario.parallelization_factor > 1.0:
            mu_eff_all *= scenario.parallelization_factor

        # v0.7: Compute pipeline iteration overhead
        rework_overhead = np.ones(n_t)
        if self.config.enable_pipeline_iteration and self.pipeline_iteration is not None:
            for j in range(n_t):
                # Get p_success at this time point
                p_success_t = {i+1: p_all[i, j] for i in range(n_s)}
                _, stats = self.pipeline_iteration.compute_effective_throughput(
                    1.0, p_success_t, n_s
                )
                rework_overhead[j] = stats['overhead_factor']

        # Compute system-level values
        throughput, bottleneck = self.system_throughput(mu_eff_all)

        # v0.7: Adjust throughput for rework overhead
        if self.config.enable_pipeline_iteration:
            throughput_adjusted = throughput / rework_overhead
        else:
            throughput_adjusted = throughput

        R = self.progress_rate(throughput_adjusted, throughput_adjusted[0])
        Y = self.cumulative_progress(R, self.config.dt)

        # Build results dataframe
        results = pd.DataFrame({
            'year': t,
            'scenario': scenario.name,
            'therapeutic_area': scenario.therapeutic_area.value,
            'is_amodei_scenario': scenario.is_amodei_scenario,
            'parallelization_factor': scenario.parallelization_factor,
            'ai_capability_global': self.ai_capability(t, scenario.g_ai),
            'ai_capability_cognitive': self.ai_capability_by_type(t, scenario.g_cognitive, AIType.COGNITIVE),
            'ai_capability_robotic': self.ai_capability_by_type(t, scenario.g_robotic, AIType.ROBOTIC),
            'ai_capability_scientific': self.ai_capability_by_type(t, scenario.g_scientific, AIType.SCIENTIFIC),
            'data_quality': D_t,
            'rework_overhead': rework_overhead,
            'throughput_raw': throughput,
            'throughput': throughput_adjusted,
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
            results[f'DQM_{i+1}'] = DQM_all[i]

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

    def get_bottleneck_transitions(
        self,
        scenario_name: str
    ) -> pd.DataFrame:
        """Identify when bottleneck shifts between stages."""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found.")

        df = self.results[scenario_name]
        transitions = []
        prev_bottleneck = df['bottleneck_stage'].iloc[0]

        for _, row in df.iterrows():
            if row['bottleneck_stage'] != prev_bottleneck:
                transitions.append({
                    'year': row['year'],
                    'from_stage': prev_bottleneck,
                    'to_stage': int(row['bottleneck_stage']),
                    'from_name': self.config.stages[prev_bottleneck - 1].name,
                    'to_name': self.config.stages[int(row['bottleneck_stage']) - 1].name,
                })
                prev_bottleneck = int(row['bottleneck_stage'])

        return pd.DataFrame(transitions)

    def get_summary_statistics(self) -> pd.DataFrame:
        """Compute summary statistics across scenarios."""
        summaries = []

        for scenario in self.config.scenarios:
            if scenario.name not in self.results:
                continue

            df = self.results[scenario.name]

            progress_10 = df[df['cumulative_progress'] >= 10]['year'].min()
            progress_50 = df[df['cumulative_progress'] >= 50]['year'].min()
            progress_100 = df[df['cumulative_progress'] >= 100]['year'].min()

            summaries.append({
                'scenario': scenario.name,
                'therapeutic_area': scenario.therapeutic_area.value,
                'is_amodei': scenario.is_amodei_scenario,
                'g_ai': scenario.g_ai,
                'g_cognitive': scenario.g_cognitive,
                'g_robotic': scenario.g_robotic,
                'g_scientific': scenario.g_scientific,
                'parallelization_factor': scenario.parallelization_factor,
                'progress_by_2030': df[df['year'] == 2030]['cumulative_progress'].iloc[0],
                'progress_by_2040': df[df['year'] == 2040]['cumulative_progress'].iloc[0],
                'progress_by_2050': df[df['year'] == 2050]['cumulative_progress'].iloc[0],
                'max_progress_rate': df['progress_rate'].max(),
                'year_10_equiv_years': progress_10 if pd.notna(progress_10) else '>2050',
                'year_50_equiv_years': progress_50 if pd.notna(progress_50) else '>2050',
                'year_100_equiv_years': progress_100 if pd.notna(progress_100) else '>2050',
                'final_bottleneck': int(df['bottleneck_stage'].iloc[-1]),
            })

        return pd.DataFrame(summaries)

    def compare_with_amodei(self) -> pd.DataFrame:
        """
        Compare model scenarios against Amodei's predictions (v0.7 NEW).

        Amodei predicts:
        - 10x acceleration (50-100 years in 5-10 years)
        - Could potentially reach 1000 years in 5-10 years
        - Skeptical of 100 years in 1 year
        """
        comparisons = []

        amodei_target_10yr = 50.0  # 50-100 years progress in 10 years
        amodei_target_high = 100.0

        for scenario in self.config.scenarios:
            if scenario.name not in self.results:
                continue

            df = self.results[scenario.name]

            # Progress in first 10 years (2024-2034)
            progress_10yr = df[df['year'] == 2034]['cumulative_progress'].iloc[0]
            progress_2050 = df[df['year'] == 2050]['cumulative_progress'].iloc[0]

            # Acceleration factor
            calendar_years_10 = 10
            calendar_years_26 = 26
            acceleration_10yr = progress_10yr / calendar_years_10
            acceleration_26yr = progress_2050 / calendar_years_26

            comparisons.append({
                'scenario': scenario.name,
                'is_amodei_scenario': scenario.is_amodei_scenario,
                'progress_10yr': progress_10yr,
                'progress_2050': progress_2050,
                'acceleration_10yr': acceleration_10yr,
                'acceleration_26yr': acceleration_26yr,
                'meets_amodei_low': progress_10yr >= amodei_target_10yr,
                'meets_amodei_high': progress_10yr >= amodei_target_high,
                'vs_amodei_target': progress_10yr / amodei_target_10yr,
            })

        return pd.DataFrame(comparisons)

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_parameters(self, filepath: str) -> None:
        """Export model parameters to JSON."""
        params = {
            'version': '0.7',
            'features': {
                'pipeline_iteration': self.config.enable_pipeline_iteration,
                'data_quality': self.config.enable_data_quality,
                'multi_type_ai': self.config.enable_multi_type_ai,
                'therapeutic_areas': self.config.enable_therapeutic_areas,
            },
            'config': {
                't0': self.config.t0,
                'T': self.config.T,
                'dt': self.config.dt,
            },
            'scenarios': [
                {
                    'name': s.name,
                    'g_ai': s.g_ai,
                    'g_cognitive': s.g_cognitive,
                    'g_robotic': s.g_robotic,
                    'g_scientific': s.g_scientific,
                    'is_amodei_scenario': s.is_amodei_scenario,
                    'parallelization_factor': s.parallelization_factor,
                    'description': s.description,
                }
                for s in self.config.scenarios
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    def export_results(self, filepath: str) -> None:
        """Export all results to CSV."""
        if not self.results:
            raise ValueError("No results to export.")

        combined = pd.concat(self.results.values(), ignore_index=True)
        combined.to_csv(filepath, index=False)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_default_model() -> Tuple[AIBioAccelerationModel, pd.DataFrame]:
    """Run the model with default parameters."""
    model = AIBioAccelerationModel()
    results = model.run_all_scenarios()
    return model, results


def compare_amodei_prediction() -> pd.DataFrame:
    """
    Compare our model against Amodei's "10x in 5-10 years" prediction.

    Returns DataFrame showing which scenarios match Amodei's expectations.
    """
    model = AIBioAccelerationModel()
    model.run_all_scenarios()
    return model.compare_with_amodei()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model - v0.7")
    print("Pipeline Iteration + Amodei Scenario")
    print("=" * 70)

    model, results = run_default_model()

    print("\nSummary Statistics:")
    print("-" * 70)
    summary = model.get_summary_statistics()
    print(summary[['scenario', 'is_amodei', 'progress_by_2050', 'max_progress_rate']].to_string(index=False))

    print("\n\nAmodei Comparison (10x target in 10 years):")
    print("-" * 70)
    amodei_comparison = model.compare_with_amodei()
    print(amodei_comparison[['scenario', 'progress_10yr', 'acceleration_10yr', 'meets_amodei_low']].to_string(index=False))

    print("\n" + "=" * 70)
    print("Model run complete.")
