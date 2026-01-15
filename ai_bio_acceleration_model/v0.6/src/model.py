"""
AI-Accelerated Biological Discovery Model - v0.6

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated
biological research and drug development.

Version: 0.6 (Data Quality Module)
Date: January 2026
License: MIT

Key Changes from v0.5:
- Data Quality Module D(t): Models how data quality affects all stages
- D(t) grows with AI capability (AI improves data generation/annotation)
- Stage-specific data quality elasticities
- Data production feedback loop (stages produce data that improves D)

v0.5 Features (retained):
- Multi-type AI: Cognitive (g_c), Robotic (g_r), Scientific (g_s)
- Therapeutic area differentiation: Oncology, CNS, Infectious Disease, Rare Disease
- Area-specific p_success and M_max parameters based on empirical data
- Historical AI calibration against Epoch AI benchmarks

Expert Review Issues Addressed:
- B1: Therapeutic area differentiation (Expert B - Drug Development)
- A1: Historical AI calibration (Expert A - AI Capabilities)
- Roadmap: Multi-type AI split from single g_ai

Mathematical Framework (v0.5):
- Three AI capability types:
  - A_c(t): Cognitive AI (reasoning, synthesis) - g_c = 0.60
  - A_r(t): Robotic AI (physical manipulation) - g_r = 0.30
  - A_s(t): Scientific AI (hypothesis generation) - g_s = 0.55
- Stage-AI mapping: Each stage uses weighted combination of AI types
- Therapeutic-specific parameters: p_success and M_max vary by therapeutic area

References:
    - Epoch AI (2024) "AI Progress Trends" - Historical calibration
    - Wong et al. (2019) Clinical trial success rates by therapeutic area
    - DiMasi et al. (2016) R&D costs by therapeutic area
    - Topol (2019) "High-performance medicine"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json

# Import data quality module
from data_quality import DataQualityModule, DataQualityConfig


# =============================================================================
# ENUMS FOR THERAPEUTIC AREAS AND AI TYPES
# =============================================================================

class TherapeuticArea(Enum):
    """Therapeutic areas with distinct clinical trial characteristics."""
    ONCOLOGY = "Oncology"
    CNS = "CNS"  # Central Nervous System
    INFECTIOUS = "Infectious Disease"
    RARE_DISEASE = "Rare Disease"
    CARDIOVASCULAR = "Cardiovascular"
    GENERAL = "General"  # Default/average


class AIType(Enum):
    """Types of AI capability with different growth rates."""
    COGNITIVE = "Cognitive"      # Reasoning, synthesis, language
    ROBOTIC = "Robotic"          # Physical manipulation, lab automation
    SCIENTIFIC = "Scientific"    # Hypothesis generation, pattern recognition


# =============================================================================
# DATA CLASSES FOR MODEL PARAMETERS
# =============================================================================

@dataclass
class TherapeuticAreaParams:
    """
    Therapeutic area-specific parameters.

    Based on Wong et al. (2019) and DiMasi et al. (2016):
    - Oncology: Lower Phase II success (20.9%), higher M_max (biomarkers)
    - CNS: Lowest Phase II success (15.3%), longest trials
    - Infectious: Higher success rates (44.8% overall), faster trials
    - Rare Disease: Higher approval (33% vs 13%), smaller trials

    References:
    - Wong et al. (2019) Nat Rev Drug Discov 18:495-497
    - Thomas et al. (2016) Nat Rev Drug Discov 15:817-818
    """
    name: str

    # Phase-specific success rate multipliers (relative to baseline)
    phase1_p_mult: float = 1.0
    phase2_p_mult: float = 1.0
    phase3_p_mult: float = 1.0

    # AI acceleration potential multipliers
    phase2_M_mult: float = 1.0  # Biomarker-driven design potential

    # Trial duration multipliers
    duration_mult: float = 1.0


# Default therapeutic area parameters
THERAPEUTIC_DEFAULTS = {
    TherapeuticArea.ONCOLOGY: TherapeuticAreaParams(
        name="Oncology",
        phase1_p_mult=0.92,    # 60.7% vs 66% baseline
        phase2_p_mult=0.63,    # 20.9% vs 33% baseline
        phase3_p_mult=0.84,    # 48.8% vs 58% baseline
        phase2_M_mult=1.4,     # High biomarker potential
        duration_mult=1.1,     # Longer endpoint assessment
    ),
    TherapeuticArea.CNS: TherapeuticAreaParams(
        name="CNS",
        phase1_p_mult=0.98,    # 64.5% vs 66% baseline
        phase2_p_mult=0.46,    # 15.3% vs 33% baseline (LOWEST)
        phase3_p_mult=0.67,    # 38.8% vs 58% baseline
        phase2_M_mult=0.8,     # Lower AI potential (complex biology)
        duration_mult=1.3,     # Longest trials
    ),
    TherapeuticArea.INFECTIOUS: TherapeuticAreaParams(
        name="Infectious Disease",
        phase1_p_mult=1.05,    # 69.3% vs 66% baseline
        phase2_p_mult=1.08,    # 35.7% vs 33% baseline
        phase3_p_mult=1.19,    # 69.0% vs 58% baseline
        phase2_M_mult=1.2,     # Good biomarker potential
        duration_mult=0.8,     # Faster trials
    ),
    TherapeuticArea.RARE_DISEASE: TherapeuticAreaParams(
        name="Rare Disease",
        phase1_p_mult=1.10,    # Higher due to targeted development
        phase2_p_mult=1.25,    # Better patient selection
        phase3_p_mult=1.15,    # Orphan drug advantage
        phase2_M_mult=1.5,     # High AI potential (small n, rich data)
        duration_mult=1.0,     # Similar duration
    ),
    TherapeuticArea.CARDIOVASCULAR: TherapeuticAreaParams(
        name="Cardiovascular",
        phase1_p_mult=1.03,    # 67.6% vs 66% baseline
        phase2_p_mult=0.82,    # 27.0% vs 33% baseline
        phase3_p_mult=1.16,    # 67.0% vs 58% baseline
        phase2_M_mult=1.1,     # Moderate biomarker potential
        duration_mult=1.2,     # Long-term outcomes
    ),
    TherapeuticArea.GENERAL: TherapeuticAreaParams(
        name="General",
        phase1_p_mult=1.0,
        phase2_p_mult=1.0,
        phase3_p_mult=1.0,
        phase2_M_mult=1.0,
        duration_mult=1.0,
    ),
}


@dataclass
class AITypeParams:
    """
    Parameters for different types of AI capability.

    Based on Epoch AI (2024) trends:
    - Cognitive: Fastest growth (LLMs, reasoning)
    - Robotic: Slowest growth (hardware-limited)
    - Scientific: Intermediate (AlphaFold, structure prediction)

    References:
    - Epoch AI (2024) "Trends in Machine Learning"
    - METR (2024) Task horizon data
    """
    ai_type: AIType
    g_base: float              # Base growth rate (year^-1)
    description: str

    # Feedback parameters (from v0.4.1)
    feedback_alpha: float = 0.1  # How much AI accelerates itself


# Default AI type parameters
AI_TYPE_DEFAULTS = {
    AIType.COGNITIVE: AITypeParams(
        ai_type=AIType.COGNITIVE,
        g_base=0.60,  # Fastest growth
        description="Language, reasoning, synthesis (GPT, Claude)",
        feedback_alpha=0.12,  # Highest self-improvement
    ),
    AIType.ROBOTIC: AITypeParams(
        ai_type=AIType.ROBOTIC,
        g_base=0.30,  # Slowest growth (hardware-limited)
        description="Physical manipulation, lab automation",
        feedback_alpha=0.05,  # Limited self-improvement
    ),
    AIType.SCIENTIFIC: AITypeParams(
        ai_type=AIType.SCIENTIFIC,
        g_base=0.55,  # Intermediate
        description="Hypothesis generation, pattern recognition (AlphaFold)",
        feedback_alpha=0.10,  # Moderate self-improvement
    ),
}


@dataclass
class Stage:
    """
    Represents a single stage in the scientific pipeline.

    v0.5 Extensions:
    - ai_type_weights: Dict mapping AIType -> weight for this stage
    - therapeutic_sensitivity: How much therapeutic area affects this stage
    """

    index: int                  # Stage number (1-10)
    name: str                   # Stage name
    description: str            # Brief description
    tau_baseline: float         # Baseline duration (months)
    M_max: float               # Maximum AI acceleration multiplier
    p_success: float           # Baseline success probability (at t0)
    k_saturation: float        # Saturation rate for AI multiplier

    # v0.4 parameters
    p_success_max: Optional[float] = None
    k_p_success: float = 0.3
    g_ai_multiplier: float = 1.0

    # v0.5 NEW: Multi-type AI weights
    # Maps AIType -> weight (should sum to 1.0)
    ai_type_weights: Dict[AIType, float] = field(default_factory=dict)

    # v0.5 NEW: Therapeutic area sensitivity (0 = no effect, 1 = full effect)
    therapeutic_sensitivity: float = 0.0

    def __post_init__(self):
        """Set defaults."""
        if self.p_success_max is None:
            self.p_success_max = self.p_success + 0.5 * (1.0 - self.p_success)

        # Default AI type weights if not specified
        if not self.ai_type_weights:
            # Default: purely cognitive for most stages
            self.ai_type_weights = {
                AIType.COGNITIVE: 1.0,
                AIType.ROBOTIC: 0.0,
                AIType.SCIENTIFIC: 0.0,
            }

    @property
    def mu_baseline(self) -> float:
        """Baseline service rate (projects per year)."""
        return 12.0 / self.tau_baseline


@dataclass
class Scenario:
    """
    Defines a scenario with specific parameter values.

    v0.5 Extensions:
    - AI type growth rates (g_cognitive, g_robotic, g_scientific)
    - Therapeutic area to simulate
    """

    name: str                   # Scenario name
    g_ai: float                # Global AI growth rate (legacy, for compatibility)
    description: str           # Scenario description

    # Scenario-specific M_max overrides
    M_max_overrides: Dict[int, float] = field(default_factory=dict)

    # v0.4 parameters
    g_ai_overrides: Dict[int, float] = field(default_factory=dict)
    p_success_max_overrides: Dict[int, float] = field(default_factory=dict)

    # v0.5 NEW: AI type growth rate overrides
    g_cognitive: Optional[float] = None
    g_robotic: Optional[float] = None
    g_scientific: Optional[float] = None

    # v0.5 NEW: Therapeutic area for this scenario
    therapeutic_area: TherapeuticArea = TherapeuticArea.GENERAL

    def __post_init__(self):
        """Set defaults for AI type growth rates."""
        if self.g_cognitive is None:
            self.g_cognitive = AI_TYPE_DEFAULTS[AIType.COGNITIVE].g_base
        if self.g_robotic is None:
            self.g_robotic = AI_TYPE_DEFAULTS[AIType.ROBOTIC].g_base
        if self.g_scientific is None:
            self.g_scientific = AI_TYPE_DEFAULTS[AIType.SCIENTIFIC].g_base


@dataclass
class ModelConfig:
    """Complete model configuration."""

    t0: int = 2024             # Baseline year
    T: int = 2050              # Horizon year
    dt: float = 1.0            # Time step (years)

    # v0.4 parameters
    enable_dynamic_p_success: bool = True
    enable_stage_specific_g_ai: bool = True
    enable_ai_feedback: bool = True
    ai_feedback_alpha: float = 0.1

    # v0.5 NEW: Enable multi-type AI
    enable_multi_type_ai: bool = True

    # v0.5 NEW: Enable therapeutic area effects
    enable_therapeutic_areas: bool = True

    # v0.6 NEW: Data quality module
    enable_data_quality: bool = True
    data_quality_config: DataQualityConfig = field(default_factory=DataQualityConfig)

    # Default stages and scenarios
    stages: List[Stage] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default stages and scenarios."""
        if not self.stages:
            self.stages = self._default_stages()
        if not self.scenarios:
            self.scenarios = self._default_scenarios()

    def _default_stages(self) -> List[Stage]:
        """
        Define the 10-stage pipeline with v0.5 parameters.

        v0.5 Changes:
        - Added ai_type_weights for multi-type AI
        - Added therapeutic_sensitivity for area-specific effects
        """
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
                    AIType.COGNITIVE: 0.4,   # Language understanding
                    AIType.ROBOTIC: 0.0,     # No physical component
                    AIType.SCIENTIFIC: 0.6,  # Pattern recognition dominant
                },
                therapeutic_sensitivity=0.2,  # Slight area effect
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
                    AIType.COGNITIVE: 0.6,   # Planning, reasoning
                    AIType.ROBOTIC: 0.1,     # Equipment selection
                    AIType.SCIENTIFIC: 0.3,  # Domain knowledge
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
                    AIType.COGNITIVE: 0.1,   # Minimal
                    AIType.ROBOTIC: 0.8,     # Dominant (lab automation)
                    AIType.SCIENTIFIC: 0.1,  # Analysis guidance
                },
                therapeutic_sensitivity=0.5,  # Moderate area effect
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
                    AIType.COGNITIVE: 0.7,   # Core AI domain
                    AIType.ROBOTIC: 0.0,     # No physical component
                    AIType.SCIENTIFIC: 0.3,  # Domain interpretation
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
                    AIType.COGNITIVE: 0.5,   # Review, verification
                    AIType.ROBOTIC: 0.3,     # Replication experiments
                    AIType.SCIENTIFIC: 0.2,  # Interpretation
                },
                therapeutic_sensitivity=0.4,
            ),
            # === CLINICAL TRIALS (High therapeutic sensitivity) ===
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
                    AIType.COGNITIVE: 0.4,   # Dosing optimization
                    AIType.ROBOTIC: 0.2,     # Patient monitoring
                    AIType.SCIENTIFIC: 0.4,  # Safety prediction
                },
                therapeutic_sensitivity=0.8,  # HIGH: varies by area
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
                    AIType.COGNITIVE: 0.3,   # Trial design
                    AIType.ROBOTIC: 0.2,     # Patient monitoring
                    AIType.SCIENTIFIC: 0.5,  # Biomarker discovery
                },
                therapeutic_sensitivity=1.0,  # HIGHEST: critical bottleneck
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
                    AIType.COGNITIVE: 0.4,   # Adaptive design
                    AIType.ROBOTIC: 0.3,     # Site management
                    AIType.SCIENTIFIC: 0.3,  # Endpoint analysis
                },
                therapeutic_sensitivity=0.9,  # HIGH
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
                    AIType.COGNITIVE: 0.8,   # Documentation, NLP
                    AIType.ROBOTIC: 0.0,     # No physical component
                    AIType.SCIENTIFIC: 0.2,  # Evidence synthesis
                },
                therapeutic_sensitivity=0.3,  # Low: process-driven
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
                    AIType.COGNITIVE: 0.3,   # Supply chain optimization
                    AIType.ROBOTIC: 0.6,     # Manufacturing automation
                    AIType.SCIENTIFIC: 0.1,  # Quality control
                },
                therapeutic_sensitivity=0.2,  # Low
            ),
        ]

    def _default_scenarios(self) -> List[Scenario]:
        """
        Define scenarios including therapeutic area variations.

        v0.5: Adds therapeutic area-specific scenarios
        """
        scenarios = []

        # Base scenarios (General therapeutic area)
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
        ]
        scenarios.extend(base_scenarios)

        # Therapeutic area-specific Baseline scenarios
        for area in [TherapeuticArea.ONCOLOGY, TherapeuticArea.CNS,
                     TherapeuticArea.INFECTIOUS, TherapeuticArea.RARE_DISEASE]:
            params = THERAPEUTIC_DEFAULTS[area]
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

    v0.5 Enhanced Mathematical Framework:
    ------------------------------------
    1. Multi-type AI Capability:
       A_eff_i(t) = Σ_k w_ik * A_k(t)
       where w_ik is weight of AI type k for stage i

    2. AI Type Capabilities:
       A_c(t) = exp(g_c * (t - t0))  [Cognitive]
       A_r(t) = exp(g_r * (t - t0))  [Robotic]
       A_s(t) = exp(g_s * (t - t0))  [Scientific]

    3. Therapeutic Area Effects:
       p_i^eff = p_i * (1 - sens_i) + p_i * area_mult * sens_i
       M_max_i^eff = M_max_i * (1 - sens_i) + M_max_i * area_mult * sens_i
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

    @property
    def time_points(self) -> np.ndarray:
        """Array of time points from t0 to T."""
        return np.arange(
            self.config.t0,
            self.config.T + self.config.dt,
            self.config.dt
        )

    @property
    def n_stages(self) -> int:
        """Number of pipeline stages."""
        return len(self.config.stages)

    @property
    def n_time_points(self) -> int:
        """Number of time points."""
        return len(self.time_points)

    # -------------------------------------------------------------------------
    # Multi-Type AI Capability (v0.5)
    # -------------------------------------------------------------------------

    def ai_capability_by_type(
        self,
        t: np.ndarray,
        g: float,
        ai_type: AIType,
        enable_feedback: bool = True
    ) -> np.ndarray:
        """
        Compute AI capability for a specific AI type.

        Parameters
        ----------
        t : np.ndarray
            Time points
        g : float
            Growth rate for this AI type
        ai_type : AIType
            Type of AI (for feedback parameter lookup)
        enable_feedback : bool
            Whether to enable AI-AI feedback loop

        Returns
        -------
        np.ndarray
            AI capability values
        """
        if not enable_feedback or not self.config.enable_ai_feedback:
            return np.exp(g * (t - self.config.t0))

        # Get type-specific feedback alpha
        alpha = AI_TYPE_DEFAULTS[ai_type].feedback_alpha

        # Iterative solution for AI-AI feedback
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
        """
        Compute effective AI capability for a stage using weighted combination.

        A_eff_i(t) = Σ_k w_ik * A_k(t)

        Parameters
        ----------
        t : np.ndarray
            Time points
        stage : Stage
            Stage object with ai_type_weights
        scenario : Scenario
            Scenario with AI type growth rates

        Returns
        -------
        np.ndarray
            Effective AI capability for this stage
        """
        if not self.config.enable_multi_type_ai:
            # Fall back to single AI type (v0.4 behavior)
            g_stage = scenario.g_ai * stage.g_ai_multiplier
            return self.ai_capability_by_type(t, g_stage, AIType.COGNITIVE)

        # Compute capability for each AI type
        A_cognitive = self.ai_capability_by_type(
            t, scenario.g_cognitive, AIType.COGNITIVE
        )
        A_robotic = self.ai_capability_by_type(
            t, scenario.g_robotic, AIType.ROBOTIC
        )
        A_scientific = self.ai_capability_by_type(
            t, scenario.g_scientific, AIType.SCIENTIFIC
        )

        # Weighted combination
        w_c = stage.ai_type_weights.get(AIType.COGNITIVE, 0.0)
        w_r = stage.ai_type_weights.get(AIType.ROBOTIC, 0.0)
        w_s = stage.ai_type_weights.get(AIType.SCIENTIFIC, 0.0)

        A_eff = w_c * A_cognitive + w_r * A_robotic + w_s * A_scientific

        return A_eff

    # -------------------------------------------------------------------------
    # Therapeutic Area Effects (v0.5)
    # -------------------------------------------------------------------------

    def apply_therapeutic_area(
        self,
        stage: Stage,
        scenario: Scenario,
        base_p: float,
        base_M_max: float
    ) -> Tuple[float, float]:
        """
        Apply therapeutic area-specific modifications to stage parameters.

        The therapeutic area multipliers represent relative success rates
        compared to the overall baseline (General). For example, CNS has
        phase2_p_mult=0.46, meaning Phase II success is only 46% of baseline.

        Parameters
        ----------
        stage : Stage
            Stage object with therapeutic_sensitivity
        scenario : Scenario
            Scenario with therapeutic_area
        base_p : float
            Base success probability
        base_M_max : float
            Base maximum AI multiplier

        Returns
        -------
        Tuple[float, float]
            (modified_p, modified_M_max)
        """
        if not self.config.enable_therapeutic_areas:
            return base_p, base_M_max

        # For General area, no modification
        if scenario.therapeutic_area == TherapeuticArea.GENERAL:
            return base_p, base_M_max

        area_params = THERAPEUTIC_DEFAULTS.get(
            scenario.therapeutic_area,
            THERAPEUTIC_DEFAULTS[TherapeuticArea.GENERAL]
        )

        sens = stage.therapeutic_sensitivity

        # Determine which phase modifier to use based on stage
        if stage.index == 6:  # Phase I
            p_mult = area_params.phase1_p_mult
            M_mult = 1.0
        elif stage.index == 7:  # Phase II
            p_mult = area_params.phase2_p_mult
            M_mult = area_params.phase2_M_mult
        elif stage.index == 8:  # Phase III
            p_mult = area_params.phase3_p_mult
            M_mult = 1.0
        else:
            p_mult = 1.0
            M_mult = 1.0

        # Apply sensitivity-weighted modification
        # When p_mult < 1 (harder area like CNS), p decreases
        # p_eff = p * (1 + sens * (mult - 1))
        # Example: CNS Phase II: p_mult=0.46, sens=1.0, base_p=0.33
        #          p_eff = 0.33 * (1 + 1.0 * (0.46 - 1)) = 0.33 * 0.46 = 0.15
        p_eff = base_p * (1 + sens * (p_mult - 1))
        M_eff = base_M_max * (1 + sens * (M_mult - 1))

        # Clip p to valid range
        p_eff = np.clip(p_eff, 0.01, 0.99)

        return p_eff, M_eff

    # -------------------------------------------------------------------------
    # Core Model Equations
    # -------------------------------------------------------------------------

    def ai_capability(self, t: np.ndarray, g: float) -> np.ndarray:
        """Legacy: Compute single AI capability (v0.4 compatibility)."""
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

        v0.6 Changes:
        - Added data quality module integration
        - Service rates modified by data quality multiplier

        v0.5 Changes:
        - Uses multi-type AI capability
        - Applies therapeutic area effects
        """
        t = self.time_points
        n_t = len(t)
        n_s = self.n_stages

        # Initialize arrays
        A_eff_all = np.zeros((n_s, n_t))   # Effective AI capability per stage
        M_all = np.zeros((n_s, n_t))
        p_all = np.zeros((n_s, n_t))
        mu_all = np.zeros((n_s, n_t))
        mu_eff_all = np.zeros((n_s, n_t))
        DQM_all = np.zeros((n_s, n_t))     # v0.6: Data quality multipliers

        # v0.6: Compute data quality trajectory (uses global AI capability)
        D_t = np.ones(n_t)  # Default: no data quality effect
        if self.config.enable_data_quality and self.data_quality is not None:
            # Use average AI capability for data quality computation
            A_global = self.ai_capability(t, scenario.g_ai)
            for j in range(n_t):
                D_t[j] = self.data_quality.compute_D(t[j], A_global[j])

        for i, stage in enumerate(self.config.stages):
            # Step 1: Compute effective AI capability (multi-type weighted)
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
                # Also adjust p_max for therapeutic area
                # Note: apply_therapeutic_area returns (p_adj, M_adj)
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

            # Step 6: Compute service rates (now includes data quality)
            # mu_i(t) = mu_i^0 * M_i(t) * DQM_i(t)
            mu_all[i] = self.service_rate(stage.mu_baseline, M_all[i]) * DQM_all[i]
            mu_eff_all[i] = self.effective_service_rate(mu_all[i], p_all[i])

        # Compute system-level values
        throughput, bottleneck = self.system_throughput(mu_eff_all)
        R = self.progress_rate(throughput, throughput[0])
        Y = self.cumulative_progress(R, self.config.dt)

        # Build results dataframe
        results = pd.DataFrame({
            'year': t,
            'scenario': scenario.name,
            'therapeutic_area': scenario.therapeutic_area.value,
            'ai_capability_global': self.ai_capability(t, scenario.g_ai),
            'ai_capability_cognitive': self.ai_capability_by_type(t, scenario.g_cognitive, AIType.COGNITIVE),
            'ai_capability_robotic': self.ai_capability_by_type(t, scenario.g_robotic, AIType.ROBOTIC),
            'ai_capability_scientific': self.ai_capability_by_type(t, scenario.g_scientific, AIType.SCIENTIFIC),
            'data_quality': D_t,  # v0.6: Data quality index
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
            results[f'DQM_{i+1}'] = DQM_all[i]  # v0.6: Data quality multiplier

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
                'g_ai': scenario.g_ai,
                'g_cognitive': scenario.g_cognitive,
                'g_robotic': scenario.g_robotic,
                'g_scientific': scenario.g_scientific,
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

    def get_therapeutic_area_comparison(self) -> pd.DataFrame:
        """
        Compare progress across therapeutic areas (v0.5 NEW).

        Returns DataFrame comparing 2050 progress for different areas.
        """
        comparisons = []

        for scenario in self.config.scenarios:
            if scenario.name not in self.results:
                continue

            df = self.results[scenario.name]
            progress_2050 = df[df['year'] == 2050]['cumulative_progress'].iloc[0]

            comparisons.append({
                'scenario': scenario.name,
                'therapeutic_area': scenario.therapeutic_area.value,
                'progress_2050': progress_2050,
            })

        return pd.DataFrame(comparisons)

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_parameters(self, filepath: str) -> None:
        """Export model parameters to JSON."""
        params = {
            'version': '0.5',
            'config': {
                't0': self.config.t0,
                'T': self.config.T,
                'dt': self.config.dt,
                'enable_dynamic_p_success': self.config.enable_dynamic_p_success,
                'enable_stage_specific_g_ai': self.config.enable_stage_specific_g_ai,
                'enable_multi_type_ai': self.config.enable_multi_type_ai,
                'enable_therapeutic_areas': self.config.enable_therapeutic_areas,
                'enable_ai_feedback': self.config.enable_ai_feedback,
            },
            'ai_types': {
                ai_type.value: {
                    'g_base': params.g_base,
                    'feedback_alpha': params.feedback_alpha,
                    'description': params.description,
                }
                for ai_type, params in AI_TYPE_DEFAULTS.items()
            },
            'therapeutic_areas': {
                area.value: {
                    'phase1_p_mult': params.phase1_p_mult,
                    'phase2_p_mult': params.phase2_p_mult,
                    'phase3_p_mult': params.phase3_p_mult,
                    'phase2_M_mult': params.phase2_M_mult,
                    'duration_mult': params.duration_mult,
                }
                for area, params in THERAPEUTIC_DEFAULTS.items()
            },
            'stages': [
                {
                    'index': s.index,
                    'name': s.name,
                    'tau_baseline': s.tau_baseline,
                    'M_max': s.M_max,
                    'p_success': s.p_success,
                    'p_success_max': s.p_success_max,
                    'k_saturation': s.k_saturation,
                    'therapeutic_sensitivity': s.therapeutic_sensitivity,
                    'ai_type_weights': {k.value: v for k, v in s.ai_type_weights.items()},
                }
                for s in self.config.stages
            ],
            'scenarios': [
                {
                    'name': s.name,
                    'g_ai': s.g_ai,
                    'g_cognitive': s.g_cognitive,
                    'g_robotic': s.g_robotic,
                    'g_scientific': s.g_scientific,
                    'therapeutic_area': s.therapeutic_area.value,
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


def compare_therapeutic_areas() -> pd.DataFrame:
    """
    Compare model results across therapeutic areas.

    Returns DataFrame with progress by 2050 for each area.
    """
    model = AIBioAccelerationModel()
    results = model.run_all_scenarios()

    comparison = model.get_therapeutic_area_comparison()

    # Filter to baseline scenarios only
    baseline_scenarios = comparison[comparison['scenario'].str.startswith('Baseline')]

    return baseline_scenarios


def compare_ai_types() -> pd.DataFrame:
    """
    Compare AI capability growth by type.

    Returns DataFrame showing AI capabilities over time.
    """
    config = ModelConfig()
    model = AIBioAccelerationModel(config)

    t = model.time_points

    results = {
        'year': t,
        'Cognitive': model.ai_capability_by_type(t, 0.60, AIType.COGNITIVE),
        'Robotic': model.ai_capability_by_type(t, 0.30, AIType.ROBOTIC),
        'Scientific': model.ai_capability_by_type(t, 0.55, AIType.SCIENTIFIC),
    }

    return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model - v0.5")
    print("Multi-Type AI + Therapeutic Areas")
    print("=" * 70)

    # Run model
    model, results = run_default_model()

    # Print summary
    print("\nSummary Statistics:")
    print("-" * 70)
    summary = model.get_summary_statistics()
    print(summary[['scenario', 'therapeutic_area', 'progress_by_2050']].to_string(index=False))

    # Compare therapeutic areas
    print("\n\nTherapeutic Area Comparison (Baseline Scenarios):")
    print("-" * 70)
    area_comparison = compare_therapeutic_areas()
    print(area_comparison.to_string(index=False))

    # Compare AI types
    print("\n\nAI Type Capabilities (2050):")
    print("-" * 70)
    ai_comparison = compare_ai_types()
    print(f"  Cognitive (g=0.60): {ai_comparison[ai_comparison['year'] == 2050]['Cognitive'].iloc[0]:.1f}x")
    print(f"  Robotic (g=0.30):   {ai_comparison[ai_comparison['year'] == 2050]['Robotic'].iloc[0]:.1f}x")
    print(f"  Scientific (g=0.55): {ai_comparison[ai_comparison['year'] == 2050]['Scientific'].iloc[0]:.1f}x")

    print("\n" + "=" * 70)
    print("Model run complete.")
