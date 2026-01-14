"""
Research Pipeline Module
========================

Defines the 8-stage research pipeline from hypothesis to publication,
with stage-specific parameters for AI acceleration modeling.

Based on PROJECT_BIBLE.md Section 4.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class StageType(Enum):
    """Classification of stage types for acceleration modeling."""
    COGNITIVE = "cognitive"      # Pure thinking/analysis - high AI potential
    PHYSICAL = "physical"        # Wet lab, experiments - limited by physical world
    HYBRID = "hybrid"            # Mix of cognitive and physical constraints
    SOCIAL = "social"            # Peer review, publication - human processes


@dataclass
class StageParameters:
    """
    Parameters for a single pipeline stage.

    Attributes:
        tau_0: Baseline duration in months (pre-AI)
        M_max_speed: Maximum AI multiplier for processing speed
        M_max_quality: Maximum AI multiplier for output quality
        p_success: Baseline probability of stage success
        k: AI adoption rate parameter (higher = faster adoption)
        stage_type: Classification for constraint modeling
        reliability_2025: Initial AI reliability factor r(t) for 2025
        reliability_growth: Annual improvement in reliability
    """
    tau_0: float                    # months
    M_max_speed: float              # dimensionless
    M_max_quality: float            # dimensionless
    p_success: float                # probability [0, 1]
    k: float                        # adoption rate
    stage_type: StageType
    reliability_2025: float = 0.7   # r(t) at t=0
    reliability_growth: float = 0.05  # annual improvement

    # Distribution parameters for uncertainty quantification
    tau_0_std: float = 0.0          # standard deviation for tau_0
    M_max_speed_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    M_max_quality_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class Stage:
    """
    A single stage in the research pipeline.

    Represents one of the 8 stages from hypothesis generation to publication.
    """
    id: str                         # S1, S2, ..., S8
    name: str                       # Human-readable name
    description: str                # Detailed description
    params: StageParameters         # Stage parameters

    def effective_multiplier(self, t: float, ai_capability: float) -> float:
        """
        Calculate effective AI multiplier at time t.

        M_i^eff(t) = sqrt(M_i^speed(t) × M_i^quality(t)) × r_i(t)

        Args:
            t: Years since 2025
            ai_capability: Current AI capability level A(t)

        Returns:
            Effective multiplier combining speed, quality, and reliability
        """
        # Speed and quality multipliers with saturation
        M_speed = 1 + (self.params.M_max_speed - 1) * (1 - np.exp(-self.params.k * ai_capability))
        M_quality = 1 + (self.params.M_max_quality - 1) * (1 - np.exp(-self.params.k * ai_capability))

        # Reliability factor (improves over time)
        r_t = min(1.0, self.params.reliability_2025 + self.params.reliability_growth * t)

        # Combined effective multiplier
        M_eff = np.sqrt(M_speed * M_quality) * r_t

        return M_eff

    def service_rate(self, t: float, ai_capability: float, baseline_rate: float = 1.0) -> float:
        """
        Calculate service rate μ_i(t) for this stage.

        μ_i(t) = μ_i^0 × M_i^eff(t)

        Args:
            t: Years since 2025
            ai_capability: Current AI capability level
            baseline_rate: Baseline service rate (1/tau_0 by default)

        Returns:
            Service rate (discoveries per month)
        """
        if baseline_rate == 1.0:
            baseline_rate = 1.0 / self.params.tau_0

        return baseline_rate * self.effective_multiplier(t, ai_capability)

    def success_probability(self, t: float, ai_capability: float) -> float:
        """
        Calculate enhanced success probability with AI assistance.

        Simple model: p(t) = p_0 + (1 - p_0) × (1 - exp(-k × A(t))) × improvement_factor

        The improvement is bounded to prevent unrealistic 100% success rates.
        """
        improvement_factor = 0.3  # AI can improve success by up to 30% of the gap
        p_0 = self.params.p_success

        improvement = (1 - p_0) * (1 - np.exp(-self.params.k * ai_capability)) * improvement_factor

        return min(0.98, p_0 + improvement)  # Cap at 98%


# Default stage definitions based on PROJECT_BIBLE.md Section 4
DEFAULT_STAGES = [
    Stage(
        id="S1",
        name="Literature Synthesis",
        description="Comprehensive review and synthesis of existing literature to identify gaps and opportunities",
        params=StageParameters(
            tau_0=3.0,
            M_max_speed=100.0,
            M_max_quality=30.0,
            p_success=0.95,
            k=1.2,
            stage_type=StageType.COGNITIVE,
            reliability_2025=0.85,
            reliability_growth=0.03,
            tau_0_std=1.0,
            M_max_speed_range=(50.0, 200.0),
            M_max_quality_range=(15.0, 50.0),
        )
    ),
    Stage(
        id="S2",
        name="Hypothesis Generation",
        description="Formulation of testable hypotheses based on literature gaps and theoretical frameworks",
        params=StageParameters(
            tau_0=6.0,
            M_max_speed=100.0,
            M_max_quality=20.0,
            p_success=0.85,
            k=1.0,
            stage_type=StageType.COGNITIVE,
            reliability_2025=0.70,
            reliability_growth=0.05,
            tau_0_std=2.0,
            M_max_speed_range=(50.0, 200.0),
            M_max_quality_range=(10.0, 40.0),
        )
    ),
    Stage(
        id="S3",
        name="Experimental Design",
        description="Design of experiments, selection of methods, power analysis, protocol development",
        params=StageParameters(
            tau_0=2.0,
            M_max_speed=30.0,
            M_max_quality=20.0,
            p_success=0.90,
            k=1.0,
            stage_type=StageType.COGNITIVE,
            reliability_2025=0.75,
            reliability_growth=0.04,
            tau_0_std=0.5,
            M_max_speed_range=(15.0, 50.0),
            M_max_quality_range=(10.0, 35.0),
        )
    ),
    Stage(
        id="S4",
        name="Data Generation (Wet Lab)",
        description="Physical experiments, data collection, sample processing - constrained by biological timescales",
        params=StageParameters(
            tau_0=12.0,
            M_max_speed=2.5,
            M_max_quality=2.5,
            p_success=0.30,
            k=0.5,
            stage_type=StageType.PHYSICAL,
            reliability_2025=0.90,
            reliability_growth=0.02,
            tau_0_std=4.0,
            M_max_speed_range=(1.5, 4.0),
            M_max_quality_range=(1.5, 4.0),
        )
    ),
    Stage(
        id="S5",
        name="Data Analysis",
        description="Statistical analysis, computational modeling, pattern recognition in experimental data",
        params=StageParameters(
            tau_0=3.5,
            M_max_speed=100.0,
            M_max_quality=50.0,
            p_success=0.95,
            k=1.2,
            stage_type=StageType.COGNITIVE,
            reliability_2025=0.80,
            reliability_growth=0.04,
            tau_0_std=1.5,
            M_max_speed_range=(50.0, 200.0),
            M_max_quality_range=(25.0, 80.0),
        )
    ),
    Stage(
        id="S6",
        name="Validation & Replication",
        description="Independent validation, replication attempts, robustness testing",
        params=StageParameters(
            tau_0=8.0,
            M_max_speed=2.5,
            M_max_quality=2.5,
            p_success=0.50,
            k=0.5,
            stage_type=StageType.HYBRID,
            reliability_2025=0.85,
            reliability_growth=0.03,
            tau_0_std=3.0,
            M_max_speed_range=(1.5, 4.0),
            M_max_quality_range=(1.5, 4.0),
        )
    ),
    Stage(
        id="S7",
        name="Writing & Peer Review",
        description="Manuscript preparation, submission, peer review process",
        params=StageParameters(
            tau_0=6.0,
            M_max_speed=10.0,
            M_max_quality=5.0,
            p_success=0.70,
            k=0.6,
            stage_type=StageType.SOCIAL,
            reliability_2025=0.75,
            reliability_growth=0.04,
            tau_0_std=2.0,
            M_max_speed_range=(5.0, 20.0),
            M_max_quality_range=(2.5, 10.0),
        )
    ),
    Stage(
        id="S8",
        name="Publication & Dissemination",
        description="Final publication, open access, preprints, knowledge dissemination",
        params=StageParameters(
            tau_0=3.0,
            M_max_speed=20.0,
            M_max_quality=10.0,
            p_success=0.95,
            k=0.8,
            stage_type=StageType.SOCIAL,
            reliability_2025=0.85,
            reliability_growth=0.03,
            tau_0_std=1.0,
            M_max_speed_range=(10.0, 40.0),
            M_max_quality_range=(5.0, 20.0),
        )
    ),
]


class ResearchPipeline:
    """
    The complete 8-stage research pipeline.

    Models the flow of research from hypothesis generation to publication,
    with AI acceleration at each stage.
    """

    def __init__(self, stages: Optional[List[Stage]] = None):
        """
        Initialize the research pipeline.

        Args:
            stages: List of Stage objects. Uses DEFAULT_STAGES if not provided.
        """
        self.stages = stages if stages is not None else DEFAULT_STAGES.copy()
        self._stage_map = {s.id: s for s in self.stages}

    def get_stage(self, stage_id: str) -> Stage:
        """Get a stage by its ID."""
        return self._stage_map[stage_id]

    def total_baseline_duration(self) -> float:
        """Calculate total baseline pipeline duration in months."""
        return sum(s.params.tau_0 for s in self.stages)

    def effective_duration(self, t: float, ai_capability: float) -> float:
        """
        Calculate effective pipeline duration with AI acceleration.

        Args:
            t: Years since 2025
            ai_capability: Current AI capability level

        Returns:
            Total duration in months
        """
        total = 0.0
        for stage in self.stages:
            M_eff = stage.effective_multiplier(t, ai_capability)
            total += stage.params.tau_0 / M_eff
        return total

    def acceleration_factor(self, t: float, ai_capability: float) -> float:
        """
        Calculate overall acceleration factor.

        Acceleration = baseline_duration / effective_duration
        """
        baseline = self.total_baseline_duration()
        effective = self.effective_duration(t, ai_capability)
        return baseline / effective

    def throughput(self, t: float, ai_capability: float) -> float:
        """
        Calculate system throughput (bottleneck-constrained).

        Θ(t) = min_i μ_i(t)

        Returns:
            Throughput in discoveries per month
        """
        service_rates = [
            stage.service_rate(t, ai_capability)
            for stage in self.stages
        ]
        return min(service_rates)

    def identify_bottleneck(self, t: float, ai_capability: float) -> Stage:
        """
        Identify the current bottleneck stage.

        Returns:
            The stage with the lowest service rate
        """
        min_rate = float('inf')
        bottleneck = None

        for stage in self.stages:
            rate = stage.service_rate(t, ai_capability)
            if rate < min_rate:
                min_rate = rate
                bottleneck = stage

        return bottleneck

    def stage_summary(self, t: float, ai_capability: float) -> Dict[str, Dict]:
        """
        Generate a summary of all stages at given time and AI capability.

        Returns:
            Dictionary with stage metrics
        """
        summary = {}
        for stage in self.stages:
            M_eff = stage.effective_multiplier(t, ai_capability)
            summary[stage.id] = {
                'name': stage.name,
                'type': stage.params.stage_type.value,
                'baseline_duration': stage.params.tau_0,
                'effective_duration': stage.params.tau_0 / M_eff,
                'multiplier': M_eff,
                'service_rate': stage.service_rate(t, ai_capability),
                'success_prob': stage.success_probability(t, ai_capability),
            }
        return summary


if __name__ == "__main__":
    # Quick test
    pipeline = ResearchPipeline()

    print("=== Research Pipeline Summary ===\n")
    print(f"Total baseline duration: {pipeline.total_baseline_duration():.1f} months")
    print(f"Number of stages: {len(pipeline.stages)}\n")

    # Test at different time points
    for t, A in [(0, 1.0), (5, 2.5), (10, 5.0), (25, 15.0)]:
        print(f"Year {2025 + t} (AI capability = {A}):")
        print(f"  Effective duration: {pipeline.effective_duration(t, A):.1f} months")
        print(f"  Acceleration factor: {pipeline.acceleration_factor(t, A):.2f}x")
        bottleneck = pipeline.identify_bottleneck(t, A)
        print(f"  Bottleneck: {bottleneck.name}")
        print()
