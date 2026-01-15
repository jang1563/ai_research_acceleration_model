"""
AI-Accelerated Biological Discovery Model - v0.2

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated
biological research and drug development.

Version: 0.2 (Split Clinical Trials)
Date: January 2026
License: MIT

Key Changes from v0.1:
- Clinical Trials split into Phase I, Phase II, Phase III
- 10-stage pipeline (was 8)
- More granular bottleneck analysis possible

Reference:
    "Bottleneck Dynamics in AI-Accelerated Biological Discovery:
     A Quantitative Scenario Analysis"
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
    """Represents a single stage in the scientific pipeline."""
    
    index: int                  # Stage number (1-8)
    name: str                   # Stage name
    description: str            # Brief description
    tau_baseline: float         # Baseline duration (months)
    M_max: float               # Maximum AI acceleration multiplier
    p_success: float           # Success probability
    k_saturation: float        # Saturation rate for AI multiplier
    
    @property
    def mu_baseline(self) -> float:
        """Baseline service rate (projects per year)."""
        return 12.0 / self.tau_baseline


@dataclass
class Scenario:
    """Defines a scenario with specific parameter values."""

    name: str                   # Scenario name
    g_ai: float                # AI capability growth rate (year^-1)
    description: str           # Scenario description
    # Scenario-specific M_max overrides (stage_index -> M_max)
    M_max_overrides: Dict[int, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Complete model configuration."""
    
    t0: int = 2024             # Baseline year
    T: int = 2050              # Horizon year
    dt: float = 1.0            # Time step (years)
    
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

        v0.2 Change: Clinical Trials split into Phase I/II/III

        References:
        - Phase I duration/success: DiMasi et al. (2016) "Innovation in the pharmaceutical industry"
        - Phase II duration/success: Wong et al. (2019) "Estimation of clinical trial success rates"
        - Phase III duration/success: Thomas et al. (2016) "Clinical Development Success Rates"
        - AI acceleration potential: Harrer et al. (2019) "AI for Clinical Trial Design"
        """
        return [
            Stage(
                index=1,
                name="Hypothesis Generation",
                description="Literature synthesis, pattern recognition, hypothesis formulation",
                tau_baseline=6.0,      # 6 months
                M_max=50.0,            # High AI potential (cognitive task)
                p_success=0.95,        # Most hypotheses can be formulated
                k_saturation=1.0       # Standard saturation rate
            ),
            Stage(
                index=2,
                name="Experiment Design",
                description="Protocol development, methodology, controls",
                tau_baseline=3.0,      # 3 months
                M_max=20.0,            # Good AI potential
                p_success=0.90,        # Most designs are feasible
                k_saturation=1.0
            ),
            Stage(
                index=3,
                name="Wet Lab Execution",
                description="Physical experiments, cell culture, animal studies",
                tau_baseline=12.0,     # 12 months
                M_max=5.0,             # Limited by physical constraints
                p_success=0.30,        # High failure rate in experiments
                k_saturation=0.5       # Slower saturation (physical limits)
            ),
            Stage(
                index=4,
                name="Data Analysis",
                description="Statistical analysis, ML on experimental data",
                tau_baseline=2.0,      # 2 months
                M_max=100.0,           # Very high AI potential (computation)
                p_success=0.95,        # Analysis usually completes
                k_saturation=1.0
            ),
            Stage(
                index=5,
                name="Validation & Replication",
                description="Independent verification, peer review, publication",
                tau_baseline=8.0,      # 8 months
                M_max=5.0,             # Limited by social processes
                p_success=0.50,        # Replication crisis
                k_saturation=0.5
            ),
            # === CLINICAL TRIALS SPLIT INTO PHASES (v0.2.1) ===
            # v0.2.1 FIX: Adjusted for realistic bottleneck timing
            # Phase II should remain bottleneck until ~2030-2032, not 2026
            Stage(
                index=6,
                name="Phase I Trials",
                description="Safety testing, dosing, small cohort (20-100 patients)",
                tau_baseline=12.0,     # 12 months (DiMasi et al. 2016)
                M_max=4.0,             # AI patient selection, adaptive dosing
                p_success=0.66,        # ~66% success rate (Thomas et al. 2016)
                k_saturation=0.5       # Moderate AI adoption speed
            ),
            Stage(
                index=7,
                name="Phase II Trials",
                description="Efficacy testing, dose optimization - 'valley of death'",
                tau_baseline=24.0,     # 24 months (DiMasi et al. 2016)
                M_max=2.8,             # REDUCED from 3.5 - hardest to accelerate
                p_success=0.33,        # ADJUSTED for ~12.6% combined rate
                k_saturation=0.3       # REDUCED - slow adoption, high stakes
            ),
            Stage(
                index=8,
                name="Phase III Trials",
                description="Confirmatory trials, large scale (1000-5000 patients)",
                tau_baseline=36.0,     # 36 months (DiMasi et al. 2016)
                M_max=3.2,             # INCREASED from 2.5 - adaptive/seamless designs
                p_success=0.58,        # ~58% success rate (Thomas et al. 2016)
                k_saturation=0.4       # INCREASED - regulatory acceptance growing
            ),
            # Combined success: 0.66 * 0.33 * 0.58 = 0.126 (~12.6%)
            # === END CLINICAL TRIALS SPLIT ===
            Stage(
                index=9,
                name="Regulatory Approval",
                description="FDA/EMA review and approval process",
                tau_baseline=12.0,     # 12 months
                M_max=2.0,             # Limited by institutional capacity
                p_success=0.90,        # Most Phase III successes approved
                k_saturation=0.3
            ),
            Stage(
                index=10,
                name="Deployment",
                description="Manufacturing scale-up, distribution, adoption",
                tau_baseline=12.0,     # 12 months
                M_max=4.0,             # Moderate automation potential
                p_success=0.95,        # Deployment usually succeeds
                k_saturation=0.5
            ),
        ]
    
    def _default_scenarios(self) -> List[Scenario]:
        """
        Define the three standard scenarios.

        v0.2.1: Updated with more realistic M_max overrides and Wet Lab consideration.

        References for scenario assumptions:
        - Pessimistic: Mullard (2021) "2020 FDA drug approvals" - slowing approval rates
        - Baseline: DiMasi et al. (2016) - historical trends continue
        - Optimistic: FDA (2019) Adaptive Designs guidance + EU regulatory reform proposals
        """
        return [
            Scenario(
                name="Pessimistic",
                g_ai=0.30,
                description="AI progress slows, institutional resistance, limited adoption",
                M_max_overrides={
                    3: 3.5,   # Wet Lab: limited automation adoption
                    6: 3.0,   # Phase I: conservative
                    7: 2.0,   # Phase II: strong resistance to change
                    8: 2.0,   # Phase III: very conservative regulators
                    9: 1.5,   # Regulatory: institutional inertia
                }
            ),
            Scenario(
                name="Baseline",
                g_ai=0.50,
                description="Current trends continue, moderate adoption",
                M_max_overrides={}  # Use default M_max values
            ),
            Scenario(
                name="Optimistic",
                g_ai=0.70,
                description="AI breakthroughs, regulatory reform, rapid adoption",
                M_max_overrides={
                    3: 8.0,   # Wet Lab: lab automation revolution (Cloud Labs, etc.)
                    5: 8.0,   # Validation: AI-powered replication checks
                    6: 5.0,   # Phase I: AI dosing + digital twins
                    7: 5.0,   # Phase II: biomarker-driven seamless designs (aggressive)
                    8: 3.5,   # Phase III: RWE helps but still needs large samples
                    9: 3.0,   # Regulatory: AI-assisted review, global harmonization
                }
                # This creates transition: S7 ceiling=0.825 > S8 ceiling=0.677
                # So Phase III becomes bottleneck as Phase II is accelerated
            ),
        ]


# =============================================================================
# CORE MODEL CLASS
# =============================================================================

class AIBioAccelerationModel:
    """
    Quantitative model of AI-accelerated biological discovery.
    
    This model computes:
    - AI capability growth over time
    - Stage-specific acceleration multipliers
    - Pipeline throughput and bottleneck identification
    - Cumulative progress in equivalent years
    
    Mathematical Framework:
    -----------------------
    1. AI Capability: A(t) = exp(g * (t - t0))
    2. AI Multiplier: M_i(t) = 1 + (M_max_i - 1) * (1 - A(t)^(-k_i))
    3. Service Rate: μ_i(t) = μ_i^0 * M_i(t)
    4. Effective Rate: μ_i^eff(t) = μ_i(t) * p_i
    5. Throughput: Θ(t) = min_i {μ_i^eff(t)}
    6. Progress Rate: R(t) = Θ(t) / Θ(t0)
    7. Cumulative Progress: Y(T) = Σ R(t) * dt
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model with configuration.
        
        Parameters
        ----------
        config : ModelConfig, optional
            Model configuration. If None, uses defaults.
        """
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
        
        A(t) = exp(g * (t - t0))
        
        Parameters
        ----------
        t : np.ndarray
            Time points (years)
        g : float
            AI growth rate (year^-1)
            
        Returns
        -------
        np.ndarray
            AI capability (normalized, A(t0) = 1)
        """
        return np.exp(g * (t - self.config.t0))
    
    def ai_multiplier(
        self, 
        A: np.ndarray, 
        M_max: float, 
        k: float
    ) -> np.ndarray:
        """
        Compute AI acceleration multiplier for a stage.
        
        M(t) = 1 + (M_max - 1) * (1 - A(t)^(-k))
        
        Properties:
        - At t=t0: A=1, so M=1 (no acceleration)
        - As t→∞: A→∞, so M→M_max (full saturation)
        - k controls rate of approach to saturation
        
        Parameters
        ----------
        A : np.ndarray
            AI capability values
        M_max : float
            Maximum multiplier (saturation limit)
        k : float
            Saturation rate parameter
            
        Returns
        -------
        np.ndarray
            AI multiplier values (1 to M_max)
        """
        return 1.0 + (M_max - 1.0) * (1.0 - np.power(A, -k))
    
    def service_rate(
        self, 
        mu_baseline: float, 
        M: np.ndarray
    ) -> np.ndarray:
        """
        Compute service rate for a stage.
        
        μ(t) = μ_baseline * M(t)
        
        Parameters
        ----------
        mu_baseline : float
            Baseline service rate (projects/year)
        M : np.ndarray
            AI multiplier values
            
        Returns
        -------
        np.ndarray
            Service rate (projects/year)
        """
        return mu_baseline * M
    
    def effective_service_rate(
        self, 
        mu: np.ndarray, 
        p: float
    ) -> np.ndarray:
        """
        Compute effective service rate accounting for success probability.
        
        μ_eff(t) = μ(t) * p
        
        Parameters
        ----------
        mu : np.ndarray
            Service rate values
        p : float
            Success probability
            
        Returns
        -------
        np.ndarray
            Effective service rate (successful projects/year)
        """
        return mu * p
    
    def system_throughput(
        self, 
        mu_eff_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute system throughput as minimum across stages.
        
        Θ(t) = min_i {μ_i^eff(t)}
        i*(t) = argmin_i {μ_i^eff(t)}
        
        Parameters
        ----------
        mu_eff_all : np.ndarray
            Effective service rates, shape (n_stages, n_time_points)
            
        Returns
        -------
        throughput : np.ndarray
            System throughput at each time point
        bottleneck : np.ndarray
            Bottleneck stage index at each time point (1-indexed)
        """
        throughput = np.min(mu_eff_all, axis=0)
        bottleneck = np.argmin(mu_eff_all, axis=0) + 1  # 1-indexed
        return throughput, bottleneck
    
    def progress_rate(
        self, 
        throughput: np.ndarray, 
        throughput_baseline: float
    ) -> np.ndarray:
        """
        Compute progress rate relative to baseline.
        
        R(t) = Θ(t) / Θ(t0)
        
        Parameters
        ----------
        throughput : np.ndarray
            System throughput at each time point
        throughput_baseline : float
            Throughput at t0
            
        Returns
        -------
        np.ndarray
            Progress rate (dimensionless, R(t0) = 1)
        """
        return throughput / throughput_baseline
    
    def cumulative_progress(
        self, 
        R: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """
        Compute cumulative equivalent years of progress.
        
        Y(t) = Σ_{τ=t0}^{t} R(τ) * dt
        
        Parameters
        ----------
        R : np.ndarray
            Progress rate at each time point
        dt : float
            Time step (years)
            
        Returns
        -------
        np.ndarray
            Cumulative progress (equivalent years)
        """
        return np.cumsum(R) * dt
    
    # -------------------------------------------------------------------------
    # Model Execution
    # -------------------------------------------------------------------------
    
    def run_scenario(self, scenario: Scenario) -> pd.DataFrame:
        """
        Run the model for a single scenario.
        
        Parameters
        ----------
        scenario : Scenario
            Scenario to run
            
        Returns
        -------
        pd.DataFrame
            Results dataframe with all computed values
        """
        t = self.time_points
        n_t = len(t)
        n_s = self.n_stages
        
        # Step 1: Compute AI capability
        A = self.ai_capability(t, scenario.g_ai)
        
        # Step 2: Compute stage-specific values
        M_all = np.zeros((n_s, n_t))
        mu_all = np.zeros((n_s, n_t))
        mu_eff_all = np.zeros((n_s, n_t))
        
        for i, stage in enumerate(self.config.stages):
            # Use scenario-specific M_max override if available
            M_max = scenario.M_max_overrides.get(stage.index, stage.M_max)
            M_all[i] = self.ai_multiplier(A, M_max, stage.k_saturation)
            mu_all[i] = self.service_rate(stage.mu_baseline, M_all[i])
            mu_eff_all[i] = self.effective_service_rate(mu_all[i], stage.p_success)
        
        # Step 3: Compute system-level values
        throughput, bottleneck = self.system_throughput(mu_eff_all)
        R = self.progress_rate(throughput, throughput[0])
        Y = self.cumulative_progress(R, self.config.dt)
        
        # Step 4: Build results dataframe
        results = pd.DataFrame({
            'year': t,
            'scenario': scenario.name,
            'ai_capability': A,
            'throughput': throughput,
            'bottleneck_stage': bottleneck,
            'progress_rate': R,
            'cumulative_progress': Y,
        })
        
        # Add stage-specific columns
        for i, stage in enumerate(self.config.stages):
            results[f'M_{i+1}'] = M_all[i]
            results[f'mu_{i+1}'] = mu_all[i]
            results[f'mu_eff_{i+1}'] = mu_eff_all[i]
        
        return results
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run the model for all scenarios.
        
        Returns
        -------
        pd.DataFrame
            Combined results for all scenarios
        """
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
        """
        Identify when bottleneck shifts between stages.
        
        Parameters
        ----------
        scenario_name : str
            Name of scenario to analyze
            
        Returns
        -------
        pd.DataFrame
            Bottleneck transitions with year and stage changes
        """
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
        """
        Compute summary statistics across scenarios.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for each scenario
        """
        summaries = []
        
        for scenario in self.config.scenarios:
            if scenario.name not in self.results:
                continue
            
            df = self.results[scenario.name]
            
            # Find key milestone years
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
    
    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------
    
    def export_parameters(self, filepath: str) -> None:
        """Export model parameters to JSON."""
        params = {
            'config': {
                't0': self.config.t0,
                'T': self.config.T,
                'dt': self.config.dt,
            },
            'stages': [
                {
                    'index': s.index,
                    'name': s.name,
                    'description': s.description,
                    'tau_baseline': s.tau_baseline,
                    'M_max': s.M_max,
                    'p_success': s.p_success,
                    'k_saturation': s.k_saturation,
                    'mu_baseline': s.mu_baseline,
                }
                for s in self.config.stages
            ],
            'scenarios': [
                {
                    'name': s.name,
                    'g_ai': s.g_ai,
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
            raise ValueError("No results to export. Run model first.")
        
        combined = pd.concat(self.results.values(), ignore_index=True)
        combined.to_csv(filepath, index=False)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_default_model() -> Tuple[AIBioAccelerationModel, pd.DataFrame]:
    """
    Run the model with default parameters.
    
    Returns
    -------
    model : AIBioAccelerationModel
        The model instance
    results : pd.DataFrame
        Combined results for all scenarios
    """
    model = AIBioAccelerationModel()
    results = model.run_all_scenarios()
    return model, results


def compute_equivalent_years(
    calendar_years: int, 
    scenario: str = 'Baseline'
) -> float:
    """
    Quick computation of equivalent years of progress.
    
    Parameters
    ----------
    calendar_years : int
        Number of calendar years from 2024
    scenario : str
        Scenario name ('Pessimistic', 'Baseline', or 'Optimistic')
        
    Returns
    -------
    float
        Equivalent years of scientific progress
    """
    model, results = run_default_model()
    df = results[results['scenario'] == scenario]
    target_year = 2024 + calendar_years
    
    if target_year > 2050:
        raise ValueError("Target year exceeds model horizon (2050)")
    
    return df[df['year'] == target_year]['cumulative_progress'].iloc[0]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI-Accelerated Biological Discovery Model - v0.1")
    print("=" * 70)
    
    # Run model
    model, results = run_default_model()
    
    # Print summary
    print("\nSummary Statistics:")
    print("-" * 70)
    summary = model.get_summary_statistics()
    print(summary.to_string(index=False))
    
    # Print bottleneck transitions for baseline
    print("\nBottleneck Transitions (Baseline Scenario):")
    print("-" * 70)
    transitions = model.get_bottleneck_transitions('Baseline')
    if len(transitions) > 0:
        print(transitions.to_string(index=False))
    else:
        print("No bottleneck transitions detected")
    
    # Print key results
    print("\nKey Results:")
    print("-" * 70)
    for scenario in ['Pessimistic', 'Baseline', 'Optimistic']:
        df = results[results['scenario'] == scenario]
        y_2050 = df[df['year'] == 2050]['cumulative_progress'].iloc[0]
        print(f"  {scenario}: {y_2050:.1f} equivalent years by 2050")
    
    print("\n" + "=" * 70)
    print("Model run complete.")
