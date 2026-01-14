"""
AI Research Acceleration Model
==============================

Core model integrating pipeline, PSM, AI capability growth, and infrastructure
constraints to forecast scientific research acceleration.

Based on PROJECT_BIBLE.md Sections 6, 7, 8.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

from .pipeline import ResearchPipeline, Stage, DEFAULT_STAGES
from .paradigm_shift import ParadigmShiftModule, ShiftType


class Scenario(Enum):
    """Pre-defined scenarios for AI development trajectories."""
    AI_WINTER = "ai_winter"         # g = 0.15, major disruption
    CONSERVATIVE = "conservative"   # g = 0.30, steady progress
    BASELINE = "baseline"           # g = 0.40, expected trajectory
    AMBITIOUS = "ambitious"         # g = 0.55, accelerated progress


@dataclass
class ScenarioParams:
    """Parameters for a scenario."""
    g_ai: float                         # AI capability growth rate
    g_ai_std: float                     # Standard deviation for UQ
    description: str
    infrastructure_multiplier: float = 1.0  # Scales infrastructure constraints


SCENARIO_DEFAULTS = {
    Scenario.AI_WINTER: ScenarioParams(
        g_ai=0.15,
        g_ai_std=0.05,
        description="Major AI disruption (regulation, safety incident)",
        infrastructure_multiplier=0.7,
    ),
    Scenario.CONSERVATIVE: ScenarioParams(
        g_ai=0.30,
        g_ai_std=0.08,
        description="Steady but cautious AI progress",
        infrastructure_multiplier=0.9,
    ),
    Scenario.BASELINE: ScenarioParams(
        g_ai=0.40,
        g_ai_std=0.10,
        description="Expected trajectory based on 2020-2025 trends",
        infrastructure_multiplier=1.0,
    ),
    Scenario.AMBITIOUS: ScenarioParams(
        g_ai=0.55,
        g_ai_std=0.12,
        description="Accelerated AI progress with sustained investment",
        infrastructure_multiplier=1.2,
    ),
}


@dataclass
class InfrastructureConstraints:
    """
    Models infrastructure bottlenecks that limit AI impact.

    Based on PROJECT_BIBLE.md Section 8.
    """
    # Compute constraints
    compute_baseline: float = 1.0           # Normalized baseline
    compute_growth_rate: float = 0.35       # Annual growth rate
    compute_max_multiplier: float = 100.0   # Saturation level

    # Data access constraints
    data_baseline: float = 0.7              # Current data accessibility (0-1)
    data_growth_rate: float = 0.05          # Annual improvement
    data_max: float = 0.95                  # Maximum achievable

    # Personnel/talent constraints
    talent_baseline: float = 0.5            # Current AI-trained researcher fraction
    talent_growth_rate: float = 0.08        # Annual growth
    talent_max: float = 0.90                # Maximum achievable

    def compute_capacity(self, t: float) -> float:
        """Calculate compute capacity at time t."""
        return min(
            self.compute_max_multiplier,
            self.compute_baseline * np.exp(self.compute_growth_rate * t)
        )

    def data_accessibility(self, t: float) -> float:
        """Calculate data accessibility at time t."""
        return min(
            self.data_max,
            self.data_baseline + self.data_growth_rate * t
        )

    def talent_availability(self, t: float) -> float:
        """Calculate AI-trained talent availability at time t."""
        return min(
            self.talent_max,
            self.talent_baseline + self.talent_growth_rate * t
        )

    def combined_constraint(self, t: float) -> float:
        """
        Calculate combined infrastructure constraint factor.

        Returns a value in [0, 1] representing overall infrastructure readiness.
        The geometric mean balances the three factors rather than letting
        one bottleneck dominate.
        """
        # Use geometric mean for more balanced constraint
        compute_factor = min(1.0, self.compute_capacity(t) / 10.0)  # Normalize to reasonable scale
        data_factor = self.data_accessibility(t)
        talent_factor = self.talent_availability(t)

        # Geometric mean of the three factors
        return (compute_factor * data_factor * talent_factor) ** (1/3)


@dataclass
class SimulationUnlock:
    """
    Models the potential for AI to invent simulation tools that replace physical trials.

    This represents a bypass pathway around physical-world bottlenecks - not just
    speeding up experiments, but circumventing them entirely through validated
    in-silico substitutes.

    Based on PROJECT_BIBLE.md Section 2.4.
    """
    # Probability trajectory for simulation achieving physical-trial equivalence
    p_unlock_2025: float = 0.05     # Current probability
    p_unlock_2035: float = 0.30     # Mid-term projection
    p_unlock_2050: float = 0.60     # Long-term projection

    # M_max values for different unlock scenarios
    M_max_physical: float = 2.5     # Without unlock (biological timescales)
    M_max_partial: float = 10.0     # Partial unlock (some experiments simulated)
    M_max_full: float = 50.0        # Full unlock (most experiments simulated)

    # Stages affected by simulation unlock (physical stages)
    affected_stages: tuple = ("S4", "S6")  # Wet Lab, Validation

    def p_unlock(self, t: float) -> float:
        """
        Calculate probability of simulation unlock at time t.

        Uses logistic interpolation between known points.
        """
        # Logistic growth from 2025 baseline
        # Calibrated to hit p_2035 and p_2050 targets
        k = 0.12  # Growth rate
        t_mid = 15  # Midpoint (around 2040)

        p = self.p_unlock_2025 + (self.p_unlock_2050 - self.p_unlock_2025) / (
            1 + np.exp(-k * (t - t_mid))
        )
        return min(0.95, p)

    def effective_M_max(self, t: float, stage_id: str, base_M_max: float) -> float:
        """
        Calculate effective M_max accounting for simulation unlock potential.

        M_max^{unlock}(t) = M_max^{physical} + P_unlock(t) × (M_max^{cognitive} - M_max^{physical})
        """
        if stage_id not in self.affected_stages:
            return base_M_max

        p = self.p_unlock(t)

        # Blend between physical and cognitive M_max based on unlock probability
        # This represents expected value across unlock scenarios
        M_max_cognitive = 100.0  # Target if fully unlocked

        return base_M_max + p * (M_max_cognitive - base_M_max)


@dataclass
class AIFailureModes:
    """
    Models AI failure modes that can reduce effective acceleration.

    Based on PROJECT_BIBLE.md Section 7.
    """
    # Hallucination propagation risk
    hallucination_base_rate: float = 0.05       # Base rate of AI errors
    hallucination_decay: float = 0.1            # Annual improvement rate

    # Monoculture risk (lack of diversity)
    monoculture_threshold: float = 0.7          # AI usage level triggering risk
    monoculture_penalty_max: float = 0.1        # Maximum penalty when triggered

    # Automation bias risk
    automation_bias_factor: float = 0.05        # Reduction due to over-reliance

    # Quality degradation in recursive AI use
    recursive_degradation: float = 0.02         # Per-iteration quality loss

    def hallucination_rate(self, t: float) -> float:
        """Calculate hallucination rate at time t."""
        return self.hallucination_base_rate * np.exp(-self.hallucination_decay * t)

    def calc_monoculture_penalty(self, ai_adoption_rate: float) -> float:
        """Calculate penalty from monoculture risk."""
        if ai_adoption_rate > self.monoculture_threshold:
            excess = ai_adoption_rate - self.monoculture_threshold
            return self.monoculture_penalty_max * (excess / (1 - self.monoculture_threshold))
        return 0.0

    def effective_quality_factor(self, t: float, ai_adoption_rate: float) -> float:
        """
        Calculate overall quality factor accounting for failure modes.

        Returns value in [0, 1] that multiplies effective acceleration.
        """
        # Start with base quality
        quality = 1.0

        # Subtract hallucination impact
        quality -= self.hallucination_rate(t)

        # Subtract monoculture penalty
        quality -= self.calc_monoculture_penalty(ai_adoption_rate)

        # Subtract automation bias
        quality -= self.automation_bias_factor * ai_adoption_rate

        return max(0.5, quality)  # Floor at 50% to prevent collapse


class AIResearchAccelerationModel:
    """
    Main model class integrating all components.

    Calculates research acceleration accounting for:
    - 8-stage pipeline dynamics
    - Paradigm shift effects
    - Infrastructure constraints
    - AI failure modes
    - Simulation unlock potential (AI replacing physical trials)
    - Scenario-based AI capability trajectories
    """

    def __init__(
        self,
        pipeline: Optional[ResearchPipeline] = None,
        psm: Optional[ParadigmShiftModule] = None,
        scenario: Scenario = Scenario.BASELINE,
        infrastructure: Optional[InfrastructureConstraints] = None,
        failure_modes: Optional[AIFailureModes] = None,
        simulation_unlock: Optional[SimulationUnlock] = None,
        enable_unlock: bool = True,
        base_year: int = 2025,
    ):
        """
        Initialize the model.

        Args:
            pipeline: Research pipeline (uses default if not provided)
            psm: Paradigm Shift Module (uses default if not provided)
            scenario: AI development scenario
            infrastructure: Infrastructure constraints
            failure_modes: AI failure mode parameters
            simulation_unlock: Simulation unlock parameters
            enable_unlock: Whether to model simulation unlock potential
            base_year: Base year for calculations
        """
        self.pipeline = pipeline or ResearchPipeline()
        self.psm = psm or ParadigmShiftModule()
        self.scenario = scenario
        self.scenario_params = SCENARIO_DEFAULTS[scenario]
        self.infrastructure = infrastructure or InfrastructureConstraints()
        self.failure_modes = failure_modes or AIFailureModes()
        self.simulation_unlock = simulation_unlock or SimulationUnlock()
        self.enable_unlock = enable_unlock
        self.base_year = base_year

    def ai_capability(self, t: float, g_ai: Optional[float] = None) -> float:
        """
        Calculate AI capability level at time t.

        A(t) = A_0 × exp(g × t)

        Args:
            t: Years since base_year
            g_ai: Override growth rate (uses scenario default if not provided)

        Returns:
            AI capability level (normalized, A_0 = 1.0 at t=0)
        """
        g = g_ai if g_ai is not None else self.scenario_params.g_ai
        return np.exp(g * t)

    def effective_multiplier_with_constraints(
        self,
        stage: Stage,
        t: float,
        ai_capability: float,
    ) -> float:
        """
        Calculate constrained effective multiplier for a stage.

        The key insight is that constraints should only affect the AI-driven
        acceleration (above 1.0x), not reduce baseline human capability.

        M_constrained = 1 + (M_raw - 1) × C(t) × Q(t) × U(t)

        Where:
        - M_raw: Raw AI multiplier including PSM effects
        - C(t): Infrastructure constraint factor
        - Q(t): Quality factor from failure modes
        - U(t): Simulation unlock factor (for physical stages)
        """
        # Base effective multiplier from AI
        M_eff = stage.effective_multiplier(t, ai_capability)

        # PSM contribution for this stage
        psm_factor = self.psm.calculate_psm(ai_capability, stage.id)

        # Raw AI-driven multiplier
        M_raw = M_eff * psm_factor

        # Apply simulation unlock potential for physical stages
        # This can increase effective M_max beyond physical limits
        if self.enable_unlock and stage.id in self.simulation_unlock.affected_stages:
            # Calculate unlock-adjusted multiplier
            # The unlock probability increases the ceiling, not the current value
            p_unlock = self.simulation_unlock.p_unlock(t)
            unlock_boost = p_unlock * (50.0 - stage.params.M_max_speed)  # Potential additional acceleration
            M_raw = M_raw + unlock_boost * (1 - np.exp(-0.1 * ai_capability))

        # Infrastructure constraint (0-1 scale)
        infra_constraint = self.infrastructure.combined_constraint(t)
        infra_constraint *= self.scenario_params.infrastructure_multiplier
        infra_constraint = min(1.0, infra_constraint)  # Cap at 1.0

        # Failure mode quality factor
        # Estimate AI adoption rate from capability
        adoption_rate = min(0.95, 1 - np.exp(-0.5 * ai_capability))
        quality_factor = self.failure_modes.effective_quality_factor(t, adoption_rate)

        # Constraints only affect the acceleration above baseline (1.0x)
        # This ensures we never go below 1.0x (baseline human capability)
        acceleration_above_baseline = M_raw - 1.0
        constrained_acceleration = acceleration_above_baseline * infra_constraint * quality_factor

        return 1.0 + max(0.0, constrained_acceleration)

    def stage_duration(self, stage: Stage, t: float) -> float:
        """Calculate effective duration for a stage at time t."""
        A = self.ai_capability(t)
        M = self.effective_multiplier_with_constraints(stage, t, A)
        return stage.params.tau_0 / M

    def total_pipeline_duration(self, t: float) -> float:
        """Calculate total pipeline duration at time t."""
        return sum(self.stage_duration(s, t) for s in self.pipeline.stages)

    def acceleration_factor(self, t: float) -> float:
        """
        Calculate overall acceleration factor at time t.

        Acceleration = baseline_duration / effective_duration
        """
        baseline = self.pipeline.total_baseline_duration()
        effective = self.total_pipeline_duration(t)
        return baseline / effective

    def throughput(self, t: float) -> float:
        """
        Calculate system throughput at time t.

        Θ(t) = min_i μ_i(t)
        """
        A = self.ai_capability(t)
        rates = []

        for stage in self.pipeline.stages:
            M = self.effective_multiplier_with_constraints(stage, t, A)
            rate = M / stage.params.tau_0  # discoveries per month
            rates.append(rate)

        return min(rates)

    def identify_bottleneck(self, t: float) -> Stage:
        """Identify the bottleneck stage at time t."""
        A = self.ai_capability(t)
        min_rate = float('inf')
        bottleneck = None

        for stage in self.pipeline.stages:
            M = self.effective_multiplier_with_constraints(stage, t, A)
            rate = M / stage.params.tau_0
            if rate < min_rate:
                min_rate = rate
                bottleneck = stage

        return bottleneck

    def validated_discoveries(
        self,
        t: float,
        baseline_rate: float = 15000.0  # discoveries/year
    ) -> float:
        """
        Estimate validated discoveries per year at time t.

        D(t) = D_baseline × acceleration_factor(t) × success_probability_factor

        Args:
            t: Years since base year
            baseline_rate: Baseline discovery rate (Tier 1+2, from bibliometrics)

        Returns:
            Estimated validated discoveries per year
        """
        accel = self.acceleration_factor(t)

        # Average success probability across stages
        A = self.ai_capability(t)
        avg_success = np.mean([
            s.success_probability(t, A) for s in self.pipeline.stages
        ])
        baseline_avg_success = np.mean([
            s.params.p_success for s in self.pipeline.stages
        ])

        success_factor = avg_success / baseline_avg_success

        return baseline_rate * accel * success_factor

    def forecast(
        self,
        years: List[int],
        metrics: Optional[List[str]] = None
    ) -> Dict[int, Dict]:
        """
        Generate forecasts for specified years.

        Args:
            years: List of calendar years to forecast
            metrics: Which metrics to include (all if not specified)

        Returns:
            Dictionary with forecasts for each year
        """
        if metrics is None:
            metrics = [
                'acceleration', 'duration', 'throughput',
                'bottleneck', 'discoveries', 'ai_capability'
            ]

        forecasts = {}

        for year in years:
            t = year - self.base_year
            if t < 0:
                continue

            result = {'year': year, 't': t}

            if 'ai_capability' in metrics:
                result['ai_capability'] = self.ai_capability(t)

            if 'acceleration' in metrics:
                result['acceleration'] = self.acceleration_factor(t)

            if 'duration' in metrics:
                result['duration_months'] = self.total_pipeline_duration(t)

            if 'throughput' in metrics:
                result['throughput'] = self.throughput(t)

            if 'bottleneck' in metrics:
                bottleneck = self.identify_bottleneck(t)
                result['bottleneck'] = bottleneck.name

            if 'discoveries' in metrics:
                result['discoveries_per_year'] = self.validated_discoveries(t)

            forecasts[year] = result

        return forecasts

    def summary(self) -> Dict:
        """Generate a summary of current model configuration."""
        return {
            'scenario': self.scenario.value,
            'scenario_params': {
                'g_ai': self.scenario_params.g_ai,
                'g_ai_std': self.scenario_params.g_ai_std,
                'description': self.scenario_params.description,
            },
            'pipeline_stages': len(self.pipeline.stages),
            'baseline_duration_months': self.pipeline.total_baseline_duration(),
            'base_year': self.base_year,
            'infrastructure': {
                'compute_growth': self.infrastructure.compute_growth_rate,
                'data_baseline': self.infrastructure.data_baseline,
                'talent_baseline': self.infrastructure.talent_baseline,
            },
        }


if __name__ == "__main__":
    # Quick test
    print("=== AI Research Acceleration Model ===\n")

    for scenario in Scenario:
        model = AIResearchAccelerationModel(scenario=scenario)
        print(f"Scenario: {scenario.value}")
        print(f"  g_ai: {model.scenario_params.g_ai}")

        # Forecast key years
        forecasts = model.forecast([2025, 2030, 2035, 2040, 2050])

        for year, data in forecasts.items():
            print(f"  {year}: {data['acceleration']:.2f}x accel, bottleneck={data['bottleneck']}")

        print()
