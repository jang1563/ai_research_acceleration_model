#!/usr/bin/env python3
"""
Policy Analysis Module for AI-Accelerated Biological Discovery Model

This module evaluates policy interventions that could accelerate AI-driven
biological discovery. Each intervention affects specific model parameters.

================================================================================
INTERVENTION FRAMEWORK
================================================================================

Policy interventions are modeled as parameter modifications:
    ΔProgress = f(θ_modified) - f(θ_baseline)

Where θ represents model parameters (g_ai, M_max, p_success, etc.)

INTERVENTION CATEGORIES:
1. AI Investment - Increase g_ai growth rate
2. Regulatory Reform - Increase M_max for clinical stages
3. Data Infrastructure - Improve D(t) data quality
4. Talent Development - Increase stage-specific multipliers
5. Research Funding - Improve p_success rates
6. International Coordination - Parallelization benefits

ROI CALCULATION:
    ROI = ΔBeneficiaries × QALY_value / Cost

Where:
- ΔBeneficiaries = additional patients helped by intervention
- QALY_value = $50,000-150,000 per QALY (ICER threshold)
- Cost = estimated policy implementation cost

================================================================================
REFERENCES
================================================================================

Policy Cost Estimates:
- Congressional Budget Office (2024). "Federal Research Funding Costs"
- FDA (2023). "User Fee Programs: Budget and Performance"
- NIH (2024). "Annual Budget Request"

Value of Statistical Life / QALY:
- Viscusi WK (2018). "Best Estimate Selection Bias in the Value of a
  Statistical Life." J Benefit Cost Anal 9(2):205-246.
- ICER (2024). "2024 Value Assessment Framework"

Regulatory Reform Impact:
- Tufts CSDD (2022). "Impact of Regulatory Pathways on Drug Development"
- FDA (2021). "Expedited Programs for Serious Conditions"

Version: 0.9
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd


class InterventionCategory(Enum):
    """Categories of policy interventions."""
    AI_INVESTMENT = "ai_investment"
    REGULATORY_REFORM = "regulatory_reform"
    DATA_INFRASTRUCTURE = "data_infrastructure"
    TALENT_DEVELOPMENT = "talent_development"
    RESEARCH_FUNDING = "research_funding"
    INTERNATIONAL_COORDINATION = "international_coordination"


@dataclass
class PolicyIntervention:
    """
    Defines a policy intervention and its effects on model parameters.

    Attributes:
        name: Short intervention name
        category: InterventionCategory
        description: Detailed description
        cost_usd: Estimated implementation cost (annual, USD)
        cost_uncertainty: Uncertainty range (low_factor, high_factor)
        parameter_effects: Dict of parameter -> modification function
        implementation_lag_years: Time to full effect
        duration_years: How long intervention lasts
        evidence_quality: 1-5 scale (5 = strong evidence)
    """
    name: str
    category: InterventionCategory
    description: str
    cost_usd: float
    cost_uncertainty: Tuple[float, float]  # (low_factor, high_factor)
    parameter_effects: Dict[str, Callable]  # param_name -> modifier function
    implementation_lag_years: float = 1.0
    duration_years: float = 10.0
    evidence_quality: int = 3
    notes: str = ""


# =============================================================================
# POLICY INTERVENTIONS CATALOG
# =============================================================================

POLICY_INTERVENTIONS = {
    # -------------------------------------------------------------------------
    # AI INVESTMENT INTERVENTIONS
    # -------------------------------------------------------------------------
    "ai_research_doubling": PolicyIntervention(
        name="Double AI Research Funding",
        category=InterventionCategory.AI_INVESTMENT,
        description="Double federal funding for AI in biomedical research "
                   "(NIH AI/ML programs, NSF Bio+AI). Expected to increase "
                   "g_ai by 25-50% through accelerated capability development.",
        cost_usd=5_000_000_000,  # $5B/year additional
        cost_uncertainty=(0.8, 1.3),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.35, 0.90),  # 35% increase, capped
            "g_cognitive": lambda g: min(g * 1.30, 0.85),
            "g_scientific": lambda g: min(g * 1.25, 0.80),
        },
        implementation_lag_years=2.0,
        duration_years=10.0,
        evidence_quality=3,
        notes="Based on NIH AI/ML spending ~$3B/year. Doubling would be ~$5B additional."
    ),

    "ai_compute_infrastructure": PolicyIntervention(
        name="National AI Compute Infrastructure",
        category=InterventionCategory.AI_INVESTMENT,
        description="Build dedicated AI compute infrastructure for biomedical "
                   "research (similar to NAIRR proposal). Reduces compute "
                   "bottlenecks for academic researchers.",
        cost_usd=2_000_000_000,  # $2B/year
        cost_uncertainty=(0.7, 1.5),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.15, 0.90),
            "g_cognitive": lambda g: min(g * 1.20, 0.85),
        },
        implementation_lag_years=3.0,
        duration_years=15.0,
        evidence_quality=2,
        notes="NAIRR proposal estimated $2.6B over 6 years. Ongoing costs ~$0.5B/year."
    ),

    # -------------------------------------------------------------------------
    # REGULATORY REFORM INTERVENTIONS
    # -------------------------------------------------------------------------
    # UPDATED per Expert B1 (Dr. Zhao): Cost $50M → $200M
    "adaptive_trials_expansion": PolicyIntervention(
        name="Expand Adaptive Trial Designs",
        category=InterventionCategory.REGULATORY_REFORM,
        description="FDA guidance expansion for adaptive and Bayesian trial "
                   "designs. Allows more efficient trials with smaller samples "
                   "and faster iteration.",
        cost_usd=200_000_000,  # $200M/year (UPDATED from $50M - includes IT, training)
        cost_uncertainty=(0.7, 1.5),
        parameter_effects={
            "M_max_phase2": lambda m: min(m * 1.20, 4.0),  # 20% faster Phase II
            "M_max_phase3": lambda m: min(m * 1.15, 3.0),  # 15% faster Phase III
            "p_phase2": lambda p: min(p * 1.10, 0.60),  # Better signal detection
        },
        implementation_lag_years=1.5,
        duration_years=15.0,
        evidence_quality=4,
        notes="FDA CIRTD program ~$15M. Full expansion requires guidance, IT, reviewer training."
    ),

    "accelerated_approval_expansion": PolicyIntervention(
        name="Expand Accelerated Approval Pathways",
        category=InterventionCategory.REGULATORY_REFORM,
        description="Broaden criteria for accelerated approval, breakthrough "
                   "therapy designation, and priority review. Includes AI-based "
                   "surrogate endpoint validation.",
        cost_usd=100_000_000,  # $100M/year
        cost_uncertainty=(0.6, 1.5),
        parameter_effects={
            "M_max_regulatory": lambda m: min(m * 1.30, 2.5),  # 30% faster approval
            "M_max_phase3": lambda m: min(m * 1.10, 3.0),
        },
        implementation_lag_years=1.0,
        duration_years=10.0,
        evidence_quality=4,
        notes="21st Century Cures Act showed significant acceleration for designated products."
    ),

    # UPDATED per Expert C2 (Dr. Whitfield): implementation_lag 2.0 → 4.0 years
    "real_world_evidence": PolicyIntervention(
        name="Real-World Evidence Integration",
        category=InterventionCategory.REGULATORY_REFORM,
        description="Expand use of real-world data/evidence for regulatory "
                   "decisions. Allows supplementary evidence from EHRs, claims "
                   "data, and patient registries.",
        cost_usd=200_000_000,  # $200M/year (infrastructure + FDA programs)
        cost_uncertainty=(0.7, 1.4),
        parameter_effects={
            "M_max_phase3": lambda m: min(m * 1.25, 3.5),
            "M_max_phase2": lambda m: min(m * 1.15, 4.0),
            "p_regulatory": lambda p: min(p * 1.05, 0.98),
        },
        implementation_lag_years=4.0,  # UPDATED from 2.0 - RWE is 10-year journey
        duration_years=15.0,
        evidence_quality=3,
        notes="FDA RWE Framework (2018) enables this. Evidentiary standards still developing."
    ),

    # -------------------------------------------------------------------------
    # DATA INFRASTRUCTURE INTERVENTIONS
    # -------------------------------------------------------------------------
    "federated_health_data": PolicyIntervention(
        name="Federated Health Data Network",
        category=InterventionCategory.DATA_INFRASTRUCTURE,
        description="Create national federated learning infrastructure for "
                   "health data. Enables AI training on distributed EHR data "
                   "without centralization.",
        cost_usd=1_500_000_000,  # $1.5B/year
        cost_uncertainty=(0.8, 1.5),
        parameter_effects={
            "D_improvement": lambda d: min(d * 1.30, 1.0),  # 30% data quality boost
            "g_ai": lambda g: min(g * 1.10, 0.90),
            "g_cognitive": lambda g: min(g * 1.15, 0.85),
        },
        implementation_lag_years=3.0,
        duration_years=20.0,
        evidence_quality=2,
        notes="Similar to UK Biobank impact but larger scale. High uncertainty."
    ),

    "biobank_expansion": PolicyIntervention(
        name="National Biobank Expansion",
        category=InterventionCategory.DATA_INFRASTRUCTURE,
        description="Expand All of Us and similar biobanks. Target 10M+ "
                   "participants with comprehensive phenotyping and genomics.",
        cost_usd=800_000_000,  # $800M/year
        cost_uncertainty=(0.9, 1.3),
        parameter_effects={
            "D_improvement": lambda d: min(d * 1.20, 1.0),
            "p_phase1": lambda p: min(p * 1.05, 0.80),  # Better target validation
            "p_phase2": lambda p: min(p * 1.08, 0.55),
        },
        implementation_lag_years=2.0,
        duration_years=15.0,
        evidence_quality=4,
        notes="All of Us currently ~$500M/year. UK Biobank has demonstrated clear value."
    ),

    # -------------------------------------------------------------------------
    # TALENT DEVELOPMENT INTERVENTIONS
    # -------------------------------------------------------------------------
    "ai_bio_training": PolicyIntervention(
        name="AI-Biology Cross-Training Program",
        category=InterventionCategory.TALENT_DEVELOPMENT,
        description="Fund 5,000 additional PhD/postdoc positions at AI-biology "
                   "intersection. Includes computational biology, ML for drug "
                   "discovery, and clinical AI.",
        cost_usd=500_000_000,  # $500M/year
        cost_uncertainty=(0.8, 1.2),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.08, 0.90),
            "g_scientific": lambda g: min(g * 1.12, 0.80),
            "M_max_preclinical": lambda m: min(m * 1.10, 25.0),
        },
        implementation_lag_years=4.0,  # Time to train
        duration_years=20.0,
        evidence_quality=3,
        notes="~5000 positions × $100K/year. Long lag but sustained effect."
    ),

    "industry_academia_partnerships": PolicyIntervention(
        name="Industry-Academia AI Partnerships",
        category=InterventionCategory.TALENT_DEVELOPMENT,
        description="Fund collaborative centers where pharma and academic "
                   "researchers work together on AI drug discovery.",
        cost_usd=300_000_000,  # $300M/year
        cost_uncertainty=(0.7, 1.4),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.05, 0.90),
            "M_max_preclinical": lambda m: min(m * 1.15, 25.0),
            "p_preclinical": lambda p: min(p * 1.05, 0.65),
        },
        implementation_lag_years=2.0,
        duration_years=10.0,
        evidence_quality=3,
        notes="Similar to ATOM consortium model. Moderate evidence of effectiveness."
    ),

    # -------------------------------------------------------------------------
    # RESEARCH FUNDING INTERVENTIONS
    # -------------------------------------------------------------------------
    "target_validation_initiative": PolicyIntervention(
        name="Target Validation Initiative",
        category=InterventionCategory.RESEARCH_FUNDING,
        description="Major initiative to improve target validation using AI, "
                   "genetics, and multi-omics. Address key source of Phase II "
                   "failure.",
        cost_usd=1_000_000_000,  # $1B/year
        cost_uncertainty=(0.8, 1.3),
        parameter_effects={
            "p_phase2": lambda p: min(p * 1.25, 0.55),  # 25% better target selection
            "p_phase1": lambda p: min(p * 1.10, 0.80),
            "g_scientific": lambda g: min(g * 1.10, 0.80),
        },
        implementation_lag_years=3.0,
        duration_years=15.0,
        evidence_quality=3,
        notes="Phase II failure is ~60% due to lack of efficacy (wrong targets)."
    ),

    "translational_science_centers": PolicyIntervention(
        name="Translational Science Centers",
        category=InterventionCategory.RESEARCH_FUNDING,
        description="Expand NCATS-like translational science centers with "
                   "AI-enabled high-throughput screening and optimization.",
        cost_usd=600_000_000,  # $600M/year
        cost_uncertainty=(0.7, 1.3),
        parameter_effects={
            "M_max_preclinical": lambda m: min(m * 1.25, 25.0),
            "p_preclinical": lambda p: min(p * 1.08, 0.65),
        },
        implementation_lag_years=2.0,
        duration_years=15.0,
        evidence_quality=3,
        notes="NCATS budget ~$850M. Expansion would add significant capacity."
    ),

    # -------------------------------------------------------------------------
    # INTERNATIONAL COORDINATION INTERVENTIONS
    # -------------------------------------------------------------------------
    "harmonized_regulations": PolicyIntervention(
        name="International Regulatory Harmonization",
        category=InterventionCategory.INTERNATIONAL_COORDINATION,
        description="ICH harmonization expansion with mutual recognition of "
                   "clinical trial data. Reduces duplicative trials across "
                   "US/EU/Japan.",
        cost_usd=50_000_000,  # $50M/year (diplomatic + regulatory)
        cost_uncertainty=(0.5, 2.0),
        parameter_effects={
            "M_max_phase3": lambda m: min(m * 1.20, 3.5),
            "parallelization": lambda p: min(p * 1.15, 1.8),
        },
        implementation_lag_years=3.0,
        duration_years=20.0,
        evidence_quality=3,
        notes="ICH process slow but impactful. High uncertainty on timeline."
    ),

    "pandemic_preparedness": PolicyIntervention(
        name="100 Days Mission Infrastructure",
        category=InterventionCategory.INTERNATIONAL_COORDINATION,
        description="Build on CEPI 100 Days Mission for pandemic vaccines. "
                   "Includes platform technologies, clinical trial networks, "
                   "and manufacturing capacity.",
        cost_usd=3_000_000_000,  # $3B/year globally (US share)
        cost_uncertainty=(0.7, 1.5),
        parameter_effects={
            "M_max_pandemic": lambda m: min(m * 1.50, 4.0),
            "p_pandemic_phase2": lambda p: min(p * 1.30, 0.70),
            "parallelization_pandemic": lambda p: min(p * 1.50, 2.5),
        },
        implementation_lag_years=2.0,
        duration_years=15.0,
        evidence_quality=4,
        notes="COVID demonstrated feasibility. CEPI 100 Days target is ambitious but achievable."
    ),
}


@dataclass
class PolicyAnalysisConfig:
    """Configuration for policy analysis."""

    # Economic parameters
    value_per_qaly: float = 100_000  # $100K/QALY (ICER midpoint)
    discount_rate: float = 0.03  # 3% annual discount

    # UPDATED per Expert D1 (Dr. Sharma): Disease-specific QALY weights
    # Using weighted average for mixed beneficiary populations
    qaly_per_cure: float = 4.0  # Reduced from 10.0 - weighted average

    # Disease-specific QALY weights for detailed calculations
    # Source: GBD 2019 disability weights, ICER assessments
    qaly_weights: dict = None  # Set in __post_init__

    # Analysis horizon
    analysis_years: int = 26  # 2024-2050

    # Uncertainty settings
    n_monte_carlo: int = 1000

    # Baseline scenario
    baseline_scenario: str = "Baseline"

    def __post_init__(self):
        """Initialize disease-specific QALY weights."""
        if self.qaly_weights is None:
            # UPDATED per Expert D1 (Dr. Sharma)
            self.qaly_weights = {
                'cancer_early_cure': 15.0,      # Young patients, good prognosis
                'cancer_late_treatment': 3.0,   # Metastatic, limited extension
                'alzheimers_therapy': 2.0,      # Quality improvement, limited quantity
                'pandemic_vaccine': 0.3,        # Prevention, per dose
                'rare_disease_cure': 12.0,      # Often pediatric, long life ahead
                'infectious_cure': 8.0,         # HIV/TB cure value
                'default': 4.0,                 # Weighted average
            }


class PolicyAnalysisModule:
    """
    Analyzes policy interventions for accelerating AI-driven biological discovery.

    Key capabilities:
    1. Compute effect of each intervention on model outcomes
    2. Calculate ROI per dollar spent
    3. Rank interventions by cost-effectiveness
    4. Analyze intervention combinations
    5. Recommend optimal policy portfolio
    """

    def __init__(self, config: Optional[PolicyAnalysisConfig] = None):
        """Initialize policy analysis module."""
        self.config = config or PolicyAnalysisConfig()
        self.interventions = POLICY_INTERVENTIONS

    def get_intervention(self, name: str) -> PolicyIntervention:
        """Get intervention by name."""
        return self.interventions[name]

    def list_interventions(self) -> pd.DataFrame:
        """List all available interventions."""
        rows = []
        for key, intervention in self.interventions.items():
            rows.append({
                'key': key,
                'name': intervention.name,
                'category': intervention.category.value,
                'cost_usd_millions': intervention.cost_usd / 1e6,
                'implementation_lag_years': intervention.implementation_lag_years,
                'evidence_quality': intervention.evidence_quality,
            })
        return pd.DataFrame(rows)

    def compute_intervention_effect(
        self,
        intervention_key: str,
        baseline_results: Dict,
        model_runner: Callable,
    ) -> Dict:
        """
        Compute effect of a single intervention.

        Parameters
        ----------
        intervention_key : str
            Key of intervention to analyze
        baseline_results : Dict
            Results from baseline scenario
        model_runner : Callable
            Function that runs model with modified parameters

        Returns
        -------
        Dict
            Effect metrics including delta_progress, delta_beneficiaries, ROI
        """
        intervention = self.interventions[intervention_key]

        # Run model with intervention effects
        modified_results = model_runner(intervention.parameter_effects)

        # Compute deltas
        baseline_progress = baseline_results.get('cumulative_progress_2050', 0)
        modified_progress = modified_results.get('cumulative_progress_2050', 0)
        delta_progress = modified_progress - baseline_progress

        baseline_beneficiaries = baseline_results.get('total_beneficiaries', 0)
        modified_beneficiaries = modified_results.get('total_beneficiaries', 0)
        delta_beneficiaries = modified_beneficiaries - baseline_beneficiaries

        # Compute economic value
        value_generated = (
            delta_beneficiaries *
            self.config.qaly_per_cure *
            self.config.value_per_qaly
        )

        # Total cost over analysis period (discounted)
        total_cost = 0
        for year in range(int(intervention.implementation_lag_years),
                         min(int(intervention.duration_years), self.config.analysis_years)):
            discount = (1 + self.config.discount_rate) ** (-year)
            total_cost += intervention.cost_usd * discount

        # ROI calculation
        roi = value_generated / total_cost if total_cost > 0 else 0

        return {
            'intervention_key': intervention_key,
            'intervention_name': intervention.name,
            'category': intervention.category.value,
            'cost_usd': intervention.cost_usd,
            'total_cost_discounted': total_cost,
            'baseline_progress': baseline_progress,
            'modified_progress': modified_progress,
            'delta_progress': delta_progress,
            'delta_progress_pct': (delta_progress / baseline_progress * 100) if baseline_progress > 0 else 0,
            'baseline_beneficiaries': baseline_beneficiaries,
            'modified_beneficiaries': modified_beneficiaries,
            'delta_beneficiaries': delta_beneficiaries,
            'value_generated': value_generated,
            'roi': roi,
            'cost_per_beneficiary': total_cost / delta_beneficiaries if delta_beneficiaries > 0 else float('inf'),
            'implementation_lag_years': intervention.implementation_lag_years,
            'evidence_quality': intervention.evidence_quality,
        }

    def estimate_intervention_effect_simple(
        self,
        intervention_key: str,
        baseline_acceleration: float = 5.7,  # Baseline scenario
        baseline_beneficiaries: float = 500_000_000,  # 500M total
    ) -> Dict:
        """
        Simple estimation of intervention effect without full model run.

        Uses heuristic that parameter changes translate approximately
        linearly to outcome changes for small perturbations.

        Parameters
        ----------
        intervention_key : str
            Key of intervention
        baseline_acceleration : float
            Baseline acceleration factor (e.g., 5.7x)
        baseline_beneficiaries : float
            Baseline total beneficiaries

        Returns
        -------
        Dict
            Estimated effect metrics
        """
        intervention = self.interventions[intervention_key]

        # Estimate acceleration improvement from parameter changes
        # Weight different parameters by their impact
        acceleration_boost = 1.0

        for param, modifier in intervention.parameter_effects.items():
            if 'g_ai' in param or 'g_' in param:
                # Growth rate changes have multiplicative effect
                # Assume baseline g = 0.50
                old_g = 0.50
                new_g = modifier(old_g)
                boost = (new_g / old_g) ** 0.5  # Square root for diminishing returns
                acceleration_boost *= boost
            elif 'M_max' in param:
                # M_max changes have direct effect
                old_m = 2.5
                new_m = modifier(old_m)
                boost = (new_m / old_m) ** 0.3  # Smaller exponent
                acceleration_boost *= boost
            elif 'p_' in param:
                # Success rate changes
                old_p = 0.30
                new_p = modifier(old_p)
                boost = (new_p / old_p) ** 0.4
                acceleration_boost *= boost
            elif 'D_improvement' in param:
                # Data quality
                old_d = 0.7
                new_d = modifier(old_d)
                boost = (new_d / old_d) ** 0.3
                acceleration_boost *= boost

        # Apply implementation lag penalty
        lag_penalty = 1.0 - (intervention.implementation_lag_years / self.config.analysis_years) * 0.5
        acceleration_boost *= lag_penalty

        # Compute outcomes
        modified_acceleration = baseline_acceleration * acceleration_boost
        delta_acceleration = modified_acceleration - baseline_acceleration

        # Beneficiaries scale approximately with acceleration
        modified_beneficiaries = baseline_beneficiaries * (modified_acceleration / baseline_acceleration)
        delta_beneficiaries = modified_beneficiaries - baseline_beneficiaries

        # Economic value
        value_generated = (
            delta_beneficiaries *
            self.config.qaly_per_cure *
            self.config.value_per_qaly
        )

        # Total cost
        total_cost = 0
        for year in range(int(intervention.implementation_lag_years),
                         min(int(intervention.duration_years), self.config.analysis_years)):
            discount = (1 + self.config.discount_rate) ** (-year)
            total_cost += intervention.cost_usd * discount

        roi = value_generated / total_cost if total_cost > 0 else 0

        return {
            'intervention_key': intervention_key,
            'intervention_name': intervention.name,
            'category': intervention.category.value,
            'annual_cost_usd': intervention.cost_usd,
            'total_cost_discounted': total_cost,
            'baseline_acceleration': baseline_acceleration,
            'modified_acceleration': modified_acceleration,
            'delta_acceleration': delta_acceleration,
            'acceleration_boost_factor': acceleration_boost,
            'baseline_beneficiaries': baseline_beneficiaries,
            'modified_beneficiaries': modified_beneficiaries,
            'delta_beneficiaries': delta_beneficiaries,
            'value_generated_usd': value_generated,
            'roi': roi,
            'cost_per_beneficiary': total_cost / delta_beneficiaries if delta_beneficiaries > 0 else float('inf'),
            'cost_per_qaly': (total_cost / (delta_beneficiaries * self.config.qaly_per_cure)
                            if delta_beneficiaries > 0 else float('inf')),
            'implementation_lag_years': intervention.implementation_lag_years,
            'evidence_quality': intervention.evidence_quality,
        }

    def rank_interventions(
        self,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
        rank_by: str = 'roi',
    ) -> pd.DataFrame:
        """
        Rank all interventions by specified metric.

        Parameters
        ----------
        baseline_acceleration : float
            Baseline scenario acceleration
        baseline_beneficiaries : float
            Baseline total beneficiaries
        rank_by : str
            Metric to rank by ('roi', 'delta_beneficiaries', 'cost_per_qaly')

        Returns
        -------
        pd.DataFrame
            Ranked interventions
        """
        results = []
        for key in self.interventions.keys():
            effect = self.estimate_intervention_effect_simple(
                key, baseline_acceleration, baseline_beneficiaries
            )
            results.append(effect)

        df = pd.DataFrame(results)

        # Sort by rank metric
        if rank_by == 'cost_per_qaly':
            df = df.sort_values(rank_by, ascending=True)
        else:
            df = df.sort_values(rank_by, ascending=False)

        df['rank'] = range(1, len(df) + 1)

        return df

    def analyze_portfolio(
        self,
        intervention_keys: List[str],
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
    ) -> Dict:
        """
        Analyze a portfolio of interventions (with interaction effects).

        Parameters
        ----------
        intervention_keys : List[str]
            Keys of interventions to combine
        baseline_acceleration : float
            Baseline acceleration
        baseline_beneficiaries : float
            Baseline beneficiaries

        Returns
        -------
        Dict
            Portfolio analysis results
        """
        # Combine parameter effects (with diminishing returns)
        combined_boost = 1.0
        total_cost = 0

        individual_effects = []

        for i, key in enumerate(intervention_keys):
            effect = self.estimate_intervention_effect_simple(
                key, baseline_acceleration, baseline_beneficiaries
            )
            individual_effects.append(effect)

            # Apply diminishing returns for combinations
            diminishing_factor = 0.8 ** i  # Each additional intervention less effective
            combined_boost *= (1 + (effect['acceleration_boost_factor'] - 1) * diminishing_factor)
            total_cost += effect['total_cost_discounted']

        # Compute combined outcomes
        modified_acceleration = baseline_acceleration * combined_boost
        modified_beneficiaries = baseline_beneficiaries * (modified_acceleration / baseline_acceleration)
        delta_beneficiaries = modified_beneficiaries - baseline_beneficiaries

        value_generated = (
            delta_beneficiaries *
            self.config.qaly_per_cure *
            self.config.value_per_qaly
        )

        return {
            'interventions': intervention_keys,
            'n_interventions': len(intervention_keys),
            'total_cost_discounted': total_cost,
            'combined_boost_factor': combined_boost,
            'baseline_acceleration': baseline_acceleration,
            'modified_acceleration': modified_acceleration,
            'delta_acceleration': modified_acceleration - baseline_acceleration,
            'baseline_beneficiaries': baseline_beneficiaries,
            'modified_beneficiaries': modified_beneficiaries,
            'delta_beneficiaries': delta_beneficiaries,
            'value_generated_usd': value_generated,
            'roi': value_generated / total_cost if total_cost > 0 else 0,
            'individual_effects': individual_effects,
        }

    def recommend_portfolio(
        self,
        budget_usd: float,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
        min_evidence_quality: int = 2,
    ) -> Dict:
        """
        Recommend optimal portfolio given budget constraint.

        Uses greedy algorithm to select highest ROI interventions
        within budget.

        Parameters
        ----------
        budget_usd : float
            Total annual budget available
        baseline_acceleration : float
            Baseline acceleration
        baseline_beneficiaries : float
            Baseline beneficiaries
        min_evidence_quality : int
            Minimum evidence quality (1-5)

        Returns
        -------
        Dict
            Recommended portfolio
        """
        # Rank by ROI
        rankings = self.rank_interventions(
            baseline_acceleration, baseline_beneficiaries, rank_by='roi'
        )

        # Filter by evidence quality
        rankings = rankings[rankings['evidence_quality'] >= min_evidence_quality]

        # Greedy selection
        selected = []
        remaining_budget = budget_usd

        for _, row in rankings.iterrows():
            if row['annual_cost_usd'] <= remaining_budget:
                selected.append(row['intervention_key'])
                remaining_budget -= row['annual_cost_usd']

        # Analyze selected portfolio
        if selected:
            portfolio = self.analyze_portfolio(
                selected, baseline_acceleration, baseline_beneficiaries
            )
        else:
            portfolio = {'interventions': [], 'roi': 0}

        return {
            'budget_usd': budget_usd,
            'selected_interventions': selected,
            'remaining_budget': remaining_budget,
            'portfolio_analysis': portfolio,
        }

    def sensitivity_analysis(
        self,
        intervention_key: str,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
    ) -> pd.DataFrame:
        """
        Sensitivity analysis on key assumptions.

        Parameters
        ----------
        intervention_key : str
            Intervention to analyze
        baseline_acceleration : float
            Baseline acceleration
        baseline_beneficiaries : float
            Baseline beneficiaries

        Returns
        -------
        pd.DataFrame
            Sensitivity results
        """
        intervention = self.interventions[intervention_key]
        base_effect = self.estimate_intervention_effect_simple(
            intervention_key, baseline_acceleration, baseline_beneficiaries
        )

        sensitivities = []

        # Vary QALY value
        for qaly_value in [50_000, 100_000, 150_000]:
            self.config.value_per_qaly = qaly_value
            effect = self.estimate_intervention_effect_simple(
                intervention_key, baseline_acceleration, baseline_beneficiaries
            )
            sensitivities.append({
                'parameter': 'value_per_qaly',
                'value': qaly_value,
                'roi': effect['roi'],
                'delta_beneficiaries': effect['delta_beneficiaries'],
            })
        self.config.value_per_qaly = 100_000  # Reset

        # Vary cost (using uncertainty bounds)
        for cost_factor in [intervention.cost_uncertainty[0], 1.0, intervention.cost_uncertainty[1]]:
            modified_cost = intervention.cost_usd * cost_factor
            # Temporarily modify
            old_cost = intervention.cost_usd
            intervention.cost_usd = modified_cost
            effect = self.estimate_intervention_effect_simple(
                intervention_key, baseline_acceleration, baseline_beneficiaries
            )
            intervention.cost_usd = old_cost
            sensitivities.append({
                'parameter': 'cost_factor',
                'value': cost_factor,
                'roi': effect['roi'],
                'delta_beneficiaries': effect['delta_beneficiaries'],
            })

        return pd.DataFrame(sensitivities)


def create_default_module() -> PolicyAnalysisModule:
    """Create policy analysis module with default configuration."""
    return PolicyAnalysisModule(PolicyAnalysisConfig())


if __name__ == "__main__":
    print("=" * 70)
    print("Policy Analysis Module Test")
    print("=" * 70)

    module = create_default_module()

    print("\nAvailable Interventions:")
    print("-" * 70)
    interventions = module.list_interventions()
    print(interventions.to_string(index=False))

    print("\n\nIntervention Rankings (by ROI):")
    print("-" * 70)
    rankings = module.rank_interventions()
    cols = ['rank', 'intervention_name', 'annual_cost_usd', 'roi',
            'delta_beneficiaries', 'evidence_quality']
    print(rankings[cols].to_string(index=False))

    print("\n\nTop 3 Portfolio Analysis:")
    print("-" * 70)
    top_3 = rankings.head(3)['intervention_key'].tolist()
    portfolio = module.analyze_portfolio(top_3)
    print(f"  Interventions: {portfolio['interventions']}")
    print(f"  Combined boost: {portfolio['combined_boost_factor']:.2f}x")
    print(f"  Modified acceleration: {portfolio['modified_acceleration']:.1f}x")
    print(f"  Delta beneficiaries: {portfolio['delta_beneficiaries']/1e6:.1f}M")
    print(f"  Portfolio ROI: {portfolio['roi']:.1f}")

    print("\n\nBudget-Constrained Recommendation ($5B/year):")
    print("-" * 70)
    recommendation = module.recommend_portfolio(5_000_000_000)
    print(f"  Selected: {recommendation['selected_interventions']}")
    print(f"  Remaining budget: ${recommendation['remaining_budget']/1e6:.0f}M")
    if recommendation['portfolio_analysis'].get('roi'):
        print(f"  Portfolio ROI: {recommendation['portfolio_analysis']['roi']:.1f}")
