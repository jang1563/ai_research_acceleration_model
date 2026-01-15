"""
AI-Accelerated Biological Discovery Model - v0.4.1

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated
biological research and drug development.

Version: 0.4.1 (AI-AI Feedback Loop)
Date: January 2026
License: MIT

Key Changes from v0.4:
- AI-AI feedback loop: AI accelerates its own development (Issue A1 from Expert Review)
- g(t) = g_0 * (1 + alpha * log(A(t))) creates superexponential growth
- Configurable feedback strength (alpha parameter, default 0.1)
- Can be disabled for v0.4 compatibility

Key Changes from v0.3:
- Time-varying p_success: Success probabilities improve with AI capability (Issue B4)
- Stage-specific AI growth rates: Different g_ai per stage reflecting domain adoption

Mathematical Framework (v0.4.1):
- AI Capability with feedback: A(t) solved iteratively with g(t) = g_0 * (1 + α*log(A))
- p_success(t) = p_base + (p_max - p_base) * (1 - A(t)^(-k_p))
- A_i(t) = exp(g_i * (t - t0)) where g_i is stage-specific

References:
    - Bostrom (2014) "Superintelligence" - recursive self-improvement
    - Grace et al. (2018) "When Will AI Exceed Human Performance?"
    - Topol (2019) "High-performance medicine: convergence of AI and healthcare"
    - Harrer et al. (2019) "AI for clinical trial design: opportunities and barriers"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import json


# =============================================================================
# DATA CLASSES FOR MODEL PARAMETERS
# =============================================================================

@dataclass
class Stage:
    """
    Represents a single stage in the scientific pipeline.

    v0.4 Extensions:
    - p_success_max: Maximum achievable success rate with AI (for time-varying p)
    - k_p_success: Saturation rate for success probability improvement
    - g_ai_multiplier: Stage-specific AI growth rate multiplier (1.0 = global rate)
    """

    index: int                  # Stage number (1-10)
    name: str                   # Stage name
    description: str            # Brief description
    tau_baseline: float         # Baseline duration (months)
    M_max: float               # Maximum AI acceleration multiplier
    p_success: float           # Baseline success probability (at t0)
    k_saturation: float        # Saturation rate for AI multiplier

    # v0.4 NEW: Time-varying success probability parameters
    p_success_max: Optional[float] = None  # Maximum p_success with AI (None = static)
    k_p_success: float = 0.3               # Saturation rate for p improvement

    # v0.4 NEW: Stage-specific AI growth rate multiplier
    g_ai_multiplier: float = 1.0           # Multiplier on global g_ai

    def __post_init__(self):
        """Set defaults for p_success_max if not provided."""
        if self.p_success_max is None:
            # Default: p_success can improve by up to 50% toward 1.0
            # e.g., if p=0.30, max becomes 0.30 + 0.5*(1.0-0.30) = 0.65
            self.p_success_max = self.p_success + 0.5 * (1.0 - self.p_success)

    @property
    def mu_baseline(self) -> float:
        """Baseline service rate (projects per year)."""
        return 12.0 / self.tau_baseline


@dataclass
class Scenario:
    """
    Defines a scenario with specific parameter values.

    v0.4 Extensions:
    - g_ai_overrides: Stage-specific AI growth rate overrides
    - p_success_max_overrides: Stage-specific max success probability overrides
    """

    name: str                   # Scenario name
    g_ai: float                # Global AI capability growth rate (year^-1)
    description: str           # Scenario description

    # Scenario-specific M_max overrides (stage_index -> M_max)
    M_max_overrides: Dict[int, float] = field(default_factory=dict)

    # v0.4 NEW: Stage-specific g_ai overrides (stage_index -> g_ai)
    g_ai_overrides: Dict[int, float] = field(default_factory=dict)

    # v0.4 NEW: Stage-specific p_success_max overrides
    p_success_max_overrides: Dict[int, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Complete model configuration."""

    t0: int = 2024             # Baseline year
    T: int = 2050              # Horizon year
    dt: float = 1.0            # Time step (years)

    # v0.4 NEW: Enable/disable time-varying p_success
    enable_dynamic_p_success: bool = True

    # v0.4 NEW: Enable/disable stage-specific g_ai
    enable_stage_specific_g_ai: bool = True

    # v0.4.1 NEW: Enable AI-AI feedback loop (recursive improvement)
    enable_ai_feedback: bool = True

    # v0.4.1 NEW: AI feedback strength parameter
    # g(t) = g_0 * (1 + alpha * log(A(t)))
    # alpha = 0 means no feedback (v0.4 behavior)
    # alpha = 0.1 means 10% increase in g per e-fold of AI capability
    ai_feedback_alpha: float = 0.1

    # Default stages (can be overridden)
    stages: List[Stage] = field(default_factory=list)

    # Default scenarios (can be overridden)
    scenarios: List[Scenario] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default stages and scenarios if not provided."""
        if not self.stages:
            self.stages = self._default_stages()
        if not self.scenarios:
            self.scenarios = self._default_scenarios()

    def _default_stages(self) -> List[Stage]:
        """
        Define the 10-stage pipeline with default parameters.

        v0.4 Changes:
        - Added p_success_max for stages where AI can improve success rates
        - Added g_ai_multiplier for differential AI adoption rates
        - Clinical trial stages have lower p_success_max (biological limits)
        - Computational stages have higher g_ai_multiplier (faster AI adoption)

        References:
        - Topol (2019) for AI impact on clinical success rates
        - Harrer et al. (2019) for AI in clinical trial design
        - Wong et al. (2019) for baseline clinical trial success rates
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
                p_success_max=0.98,        # Already high, limited improvement
                k_p_success=0.5,
                g_ai_multiplier=1.2        # Fast AI adoption (pure computation)
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
                g_ai_multiplier=1.1        # Good AI adoption
            ),
            Stage(
                index=3,
                name="Wet Lab Execution",
                description="Physical experiments, cell culture, animal studies",
                tau_baseline=12.0,
                M_max=5.0,
                p_success=0.30,
                k_saturation=0.5,
                p_success_max=0.50,        # AI improves experiment success (Ref: Topol 2019)
                k_p_success=0.2,           # Slow improvement (physical limits)
                g_ai_multiplier=0.7        # Slower AI adoption (hardware-limited)
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
                g_ai_multiplier=1.5        # Fastest AI adoption (core AI domain)
            ),
            Stage(
                index=5,
                name="Validation & Replication",
                description="Independent verification, peer review, publication",
                tau_baseline=8.0,
                M_max=5.0,
                p_success=0.50,
                k_saturation=0.5,
                p_success_max=0.70,        # AI detects irreproducible results earlier
                k_p_success=0.3,
                g_ai_multiplier=0.8        # Moderate adoption (social processes)
            ),
            # === CLINICAL TRIALS ===
            Stage(
                index=6,
                name="Phase I Trials",
                description="Safety testing, dosing, small cohort (20-100 patients)",
                tau_baseline=12.0,
                M_max=4.0,
                p_success=0.66,
                k_saturation=0.5,
                p_success_max=0.75,        # AI patient selection + dosing (Harrer 2019)
                k_p_success=0.25,
                g_ai_multiplier=0.9        # Moderate adoption (regulatory caution)
            ),
            Stage(
                index=7,
                name="Phase II Trials",
                description="Efficacy testing, dose optimization - 'valley of death'",
                tau_baseline=24.0,
                M_max=2.8,
                p_success=0.33,
                k_saturation=0.3,
                p_success_max=0.50,        # AI biomarker discovery + patient stratification
                k_p_success=0.2,           # Slow (fundamental biology challenges)
                g_ai_multiplier=0.8        # Slower adoption (high stakes)
            ),
            Stage(
                index=8,
                name="Phase III Trials",
                description="Confirmatory trials, large scale (1000-5000 patients)",
                tau_baseline=36.0,
                M_max=3.2,
                p_success=0.58,
                k_saturation=0.4,
                p_success_max=0.70,        # AI adaptive designs + RWE integration
                k_p_success=0.25,
                g_ai_multiplier=0.85       # Moderate adoption
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
                g_ai_multiplier=0.6        # Slow adoption (institutional)
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
                g_ai_multiplier=1.0        # Standard adoption
            ),
        ]

    def _default_scenarios(self) -> List[Scenario]:
        """
        Define the three standard scenarios.

        v0.4 Changes:
        - Added g_ai_overrides for stage-specific AI growth
        - Added p_success_max_overrides for scenario-specific success limits

        References:
        - Pessimistic: Regulatory resistance, limited AI validation
        - Baseline: Current trends extrapolated
        - Optimistic: Rapid AI adoption, regulatory reform
        """
        return [
            Scenario(
                name="Pessimistic",
                g_ai=0.30,
                description="AI progress slows, institutional resistance, limited adoption",
                M_max_overrides={
                    3: 3.5,   # Wet Lab: limited automation
                    6: 3.0,   # Phase I: conservative
                    7: 2.0,   # Phase II: strong resistance
                    8: 2.0,   # Phase III: very conservative
                    9: 1.5,   # Regulatory: institutional inertia
                },
                g_ai_overrides={
                    6: 0.20,  # Phase I: slower AI adoption
                    7: 0.15,  # Phase II: very slow adoption
                    8: 0.15,  # Phase III: very slow adoption
                    9: 0.10,  # Regulatory: minimal AI impact
                },
                p_success_max_overrides={
                    7: 0.40,  # Phase II: limited AI benefit
                    8: 0.62,  # Phase III: limited AI benefit
                }
            ),
            Scenario(
                name="Baseline",
                g_ai=0.50,
                description="Current trends continue, moderate adoption",
                M_max_overrides={},
                g_ai_overrides={},         # Use stage g_ai_multiplier * global g_ai
                p_success_max_overrides={}  # Use default p_success_max
            ),
            Scenario(
                name="Optimistic",
                g_ai=0.70,
                description="AI breakthroughs, regulatory reform, rapid adoption",
                M_max_overrides={
                    3: 8.0,   # Wet Lab: automation revolution
                    5: 8.0,   # Validation: AI replication
                    6: 5.0,   # Phase I: AI dosing + digital twins
                    7: 5.0,   # Phase II: biomarker-driven designs
                    8: 3.5,   # Phase III: RWE integration
                    9: 3.0,   # Regulatory: AI-assisted review
                },
                g_ai_overrides={
                    4: 0.90,  # Data Analysis: rapid advancement
                    6: 0.75,  # Phase I: faster adoption with reform
                    7: 0.65,  # Phase II: accelerated with breakthroughs
                    8: 0.60,  # Phase III: regulatory acceptance
                },
                p_success_max_overrides={
                    3: 0.55,  # Wet Lab: better experiment design
                    7: 0.55,  # Phase II: breakthrough biomarkers
                    8: 0.75,  # Phase III: better patient selection
                }
            ),
        ]


# =============================================================================
# CORE MODEL CLASS
# =============================================================================

class AIBioAccelerationModel:
    """
    Quantitative model of AI-accelerated biological discovery.

    v0.4 Enhanced Mathematical Framework:
    ------------------------------------
    1. AI Capability (stage-specific):
       A_i(t) = exp(g_i * (t - t0))
       where g_i = g_global * g_ai_multiplier_i (or override)

    2. AI Multiplier:
       M_i(t) = 1 + (M_max_i - 1) * (1 - A_i(t)^(-k_i))

    3. Time-varying Success Probability (NEW):
       p_i(t) = p_base_i + (p_max_i - p_base_i) * (1 - A_i(t)^(-k_p_i))

    4. Service Rate:
       μ_i(t) = μ_i^0 * M_i(t)

    5. Effective Rate:
       μ_i^eff(t) = μ_i(t) * p_i(t)  [p_i now time-varying!]

    6. Throughput:
       Θ(t) = min_i {μ_i^eff(t)}

    7. Progress Rate:
       R(t) = Θ(t) / Θ(t0)

    8. Cumulative Progress:
       Y(T) = Σ R(t) * dt
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model with configuration."""
        self.config = config or ModelConfig()
        self.results: Dict[str, pd.DataFrame] = {}

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
    # Core Model Equations
    # -------------------------------------------------------------------------

    def ai_capability(self, t: np.ndarray, g: float) -> np.ndarray:
        """
        Compute AI capability at time t.

        Without feedback: A(t) = exp(g * (t - t0))

        With AI-AI feedback (v0.4.1):
        - AI accelerates its own development
        - g(t) = g_0 * (1 + alpha * log(A(t)))
        - This creates superexponential growth

        The feedback is solved iteratively since A depends on g which depends on A.

        References:
        - Bostrom (2014) "Superintelligence" - recursive self-improvement
        - Grace et al. (2018) "When Will AI Exceed Human Performance?"
        """
        if not self.config.enable_ai_feedback or self.config.ai_feedback_alpha == 0:
            # Original exponential model
            return np.exp(g * (t - self.config.t0))

        # Iterative solution for AI-AI feedback
        # Start with base exponential
        A = np.exp(g * (t - self.config.t0))

        # Iterate to find fixed point (usually converges in 3-5 iterations)
        alpha = self.config.ai_feedback_alpha
        for _ in range(5):
            # Effective growth rate increases with log(A)
            g_eff = g * (1 + alpha * np.maximum(0, np.log(A)))
            # Recompute A with enhanced growth
            # Use cumulative integration for time-varying g
            dt = self.config.dt
            A_new = np.ones_like(t)
            for i in range(1, len(t)):
                # A(t+dt) = A(t) * exp(g_eff(t) * dt)
                A_new[i] = A_new[i-1] * np.exp(g_eff[i-1] * dt)
            A = A_new

        return A

    def ai_multiplier(
        self,
        A: np.ndarray,
        M_max: float,
        k: float
    ) -> np.ndarray:
        """
        Compute AI acceleration multiplier for a stage.

        M(t) = 1 + (M_max - 1) * (1 - A(t)^(-k))
        """
        return 1.0 + (M_max - 1.0) * (1.0 - np.power(A, -k))

    def time_varying_p_success(
        self,
        A: np.ndarray,
        p_base: float,
        p_max: float,
        k_p: float
    ) -> np.ndarray:
        """
        Compute time-varying success probability (v0.4 NEW).

        p(t) = p_base + (p_max - p_base) * (1 - A(t)^(-k_p))

        Properties:
        - At t=t0: A=1, so p=p_base (baseline success rate)
        - As t→∞: A→∞, so p→p_max (maximum achievable with AI)
        - k_p controls rate of improvement

        Parameters
        ----------
        A : np.ndarray
            AI capability values
        p_base : float
            Baseline success probability (at t0)
        p_max : float
            Maximum achievable success probability
        k_p : float
            Saturation rate for probability improvement

        Returns
        -------
        np.ndarray
            Time-varying success probability (p_base to p_max)
        """
        if p_max <= p_base:
            # No improvement possible, return static value
            return np.full_like(A, p_base)

        p_t = p_base + (p_max - p_base) * (1.0 - np.power(A, -k_p))
        return np.clip(p_t, 0.0, 1.0)  # Ensure valid probability

    def service_rate(
        self,
        mu_baseline: float,
        M: np.ndarray
    ) -> np.ndarray:
        """Compute service rate: μ(t) = μ_baseline * M(t)"""
        return mu_baseline * M

    def effective_service_rate(
        self,
        mu: np.ndarray,
        p: np.ndarray
    ) -> np.ndarray:
        """
        Compute effective service rate with time-varying p.

        μ_eff(t) = μ(t) * p(t)

        v0.4: p is now an array (time-varying), not a scalar
        """
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
        """Compute progress rate: R(t) = Θ(t) / Θ(t0)"""
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

        v0.4 Changes:
        - Computes stage-specific AI capability using g_ai_overrides
        - Computes time-varying p_success for each stage
        - Stores p_success trajectories in results
        """
        t = self.time_points
        n_t = len(t)
        n_s = self.n_stages

        # Initialize arrays
        A_all = np.zeros((n_s, n_t))       # Stage-specific AI capability
        M_all = np.zeros((n_s, n_t))
        p_all = np.zeros((n_s, n_t))       # v0.4: Time-varying p_success
        mu_all = np.zeros((n_s, n_t))
        mu_eff_all = np.zeros((n_s, n_t))

        for i, stage in enumerate(self.config.stages):
            # Step 1: Compute stage-specific AI growth rate
            if self.config.enable_stage_specific_g_ai:
                # Check for scenario override first, then use multiplier
                if stage.index in scenario.g_ai_overrides:
                    g_stage = scenario.g_ai_overrides[stage.index]
                else:
                    g_stage = scenario.g_ai * stage.g_ai_multiplier
            else:
                g_stage = scenario.g_ai

            # Step 2: Compute stage-specific AI capability
            A_all[i] = self.ai_capability(t, g_stage)

            # Step 3: Compute AI multiplier (using scenario override if available)
            M_max = scenario.M_max_overrides.get(stage.index, stage.M_max)
            M_all[i] = self.ai_multiplier(A_all[i], M_max, stage.k_saturation)

            # Step 4: Compute time-varying p_success (v0.4)
            if self.config.enable_dynamic_p_success:
                p_max = scenario.p_success_max_overrides.get(
                    stage.index, stage.p_success_max
                )
                p_all[i] = self.time_varying_p_success(
                    A_all[i], stage.p_success, p_max, stage.k_p_success
                )
            else:
                p_all[i] = np.full(n_t, stage.p_success)

            # Step 5: Compute service rates
            mu_all[i] = self.service_rate(stage.mu_baseline, M_all[i])
            mu_eff_all[i] = self.effective_service_rate(mu_all[i], p_all[i])

        # Compute system-level values
        throughput, bottleneck = self.system_throughput(mu_eff_all)
        R = self.progress_rate(throughput, throughput[0])
        Y = self.cumulative_progress(R, self.config.dt)

        # Build results dataframe
        results = pd.DataFrame({
            'year': t,
            'scenario': scenario.name,
            'ai_capability_global': self.ai_capability(t, scenario.g_ai),
            'throughput': throughput,
            'bottleneck_stage': bottleneck,
            'progress_rate': R,
            'cumulative_progress': Y,
        })

        # Add stage-specific columns
        for i, stage in enumerate(self.config.stages):
            results[f'A_{i+1}'] = A_all[i]
            results[f'M_{i+1}'] = M_all[i]
            results[f'p_{i+1}'] = p_all[i]        # v0.4: Time-varying p
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

    def get_bottleneck_transitions(
        self,
        scenario_name: str
    ) -> pd.DataFrame:
        """Identify when bottleneck shifts between stages."""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found. Run model first.")

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
                'g_ai': scenario.g_ai,
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

    def get_p_success_evolution(self, scenario_name: str) -> pd.DataFrame:
        """
        Get the evolution of success probabilities over time (v0.4 NEW).

        Returns DataFrame showing how p_success changes for each stage.
        """
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found.")

        df = self.results[scenario_name]

        p_cols = ['year'] + [f'p_{i+1}' for i in range(self.n_stages)]
        return df[p_cols].copy()

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_parameters(self, filepath: str) -> None:
        """Export model parameters to JSON."""
        params = {
            'version': '0.4',
            'config': {
                't0': self.config.t0,
                'T': self.config.T,
                'dt': self.config.dt,
                'enable_dynamic_p_success': self.config.enable_dynamic_p_success,
                'enable_stage_specific_g_ai': self.config.enable_stage_specific_g_ai,
            },
            'stages': [
                {
                    'index': s.index,
                    'name': s.name,
                    'description': s.description,
                    'tau_baseline': s.tau_baseline,
                    'M_max': s.M_max,
                    'p_success': s.p_success,
                    'p_success_max': s.p_success_max,
                    'k_saturation': s.k_saturation,
                    'k_p_success': s.k_p_success,
                    'g_ai_multiplier': s.g_ai_multiplier,
                    'mu_baseline': s.mu_baseline,
                }
                for s in self.config.stages
            ],
            'scenarios': [
                {
                    'name': s.name,
                    'g_ai': s.g_ai,
                    'description': s.description,
                    'M_max_overrides': s.M_max_overrides,
                    'g_ai_overrides': s.g_ai_overrides,
                    'p_success_max_overrides': s.p_success_max_overrides,
                }
                for s in self.config.scenarios
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    def export_results(self, filepath: str) -> None:
        """Export all results to CSV."""
        if not self.results:
            raise ValueError("No results to export. Run model first.")

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


def compute_equivalent_years(
    calendar_years: int,
    scenario: str = 'Baseline'
) -> float:
    """Quick computation of equivalent years of progress."""
    model, results = run_default_model()
    df = results[results['scenario'] == scenario]
    target_year = 2024 + calendar_years

    if target_year > 2050:
        raise ValueError("Target year exceeds model horizon (2050)")

    return df[df['year'] == target_year]['cumulative_progress'].iloc[0]


def compare_v03_v04() -> pd.DataFrame:
    """
    Compare v0.3 (static p_success) vs v0.4 (dynamic p_success).

    Useful for understanding the impact of the new features.
    """
    # v0.4 with all features
    config_v04 = ModelConfig(
        enable_dynamic_p_success=True,
        enable_stage_specific_g_ai=True
    )
    model_v04 = AIBioAccelerationModel(config_v04)
    results_v04 = model_v04.run_all_scenarios()

    # v0.3 equivalent (static p, uniform g)
    config_v03 = ModelConfig(
        enable_dynamic_p_success=False,
        enable_stage_specific_g_ai=False
    )
    model_v03 = AIBioAccelerationModel(config_v03)
    results_v03 = model_v03.run_all_scenarios()

    # Compare key metrics
    comparison = []
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        v03 = results_v03[results_v03['scenario'] == scenario]
        v04 = results_v04[results_v04['scenario'] == scenario]

        comparison.append({
            'scenario': scenario,
            'v03_2050': v03[v03['year'] == 2050]['cumulative_progress'].iloc[0],
            'v04_2050': v04[v04['year'] == 2050]['cumulative_progress'].iloc[0],
            'improvement_pct': (
                (v04[v04['year'] == 2050]['cumulative_progress'].iloc[0] -
                 v03[v03['year'] == 2050]['cumulative_progress'].iloc[0]) /
                v03[v03['year'] == 2050]['cumulative_progress'].iloc[0] * 100
            ),
        })

    return pd.DataFrame(comparison)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model - v0.4")
    print("Dynamic p_success + Stage-specific g_ai")
    print("=" * 70)

    # Run model
    model, results = run_default_model()

    # Print summary
    print("\nSummary Statistics:")
    print("-" * 70)
    summary = model.get_summary_statistics()
    print(summary.to_string(index=False))

    # Print bottleneck transitions
    print("\nBottleneck Transitions (Baseline Scenario):")
    print("-" * 70)
    transitions = model.get_bottleneck_transitions('Baseline')
    if len(transitions) > 0:
        print(transitions.to_string(index=False))
    else:
        print("No bottleneck transitions detected")

    # Show p_success evolution for Phase II (key bottleneck)
    print("\nPhase II Success Probability Evolution (Baseline):")
    print("-" * 70)
    p_evolution = model.get_p_success_evolution('Baseline')
    print(f"  2024: {p_evolution['p_7'].iloc[0]:.3f}")
    print(f"  2030: {p_evolution[p_evolution['year'] == 2030]['p_7'].iloc[0]:.3f}")
    print(f"  2040: {p_evolution[p_evolution['year'] == 2040]['p_7'].iloc[0]:.3f}")
    print(f"  2050: {p_evolution[p_evolution['year'] == 2050]['p_7'].iloc[0]:.3f}")

    # Compare v0.3 vs v0.4
    print("\nv0.3 vs v0.4 Comparison:")
    print("-" * 70)
    comparison = compare_v03_v04()
    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("Model run complete.")
