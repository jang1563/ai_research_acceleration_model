#!/usr/bin/env python3
"""
Policy Analysis Module for AI-Accelerated Biological Discovery Model - v1.1

Updates from v1.0:
- P2-15: Policy implementation curves with lag and adoption rate
- P2-16: Expanded QALY range ($50K-$200K)

Version: 1.1
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
class ImplementationCurve:
    """P2-15: Policy implementation curve parameters."""
    lag_years: float  # Time to start having effect
    ramp_up_years: float  # Time from start to full effect
    adoption_rate: float  # Final adoption rate (0-1)
    curve_type: str = 'logistic'  # 'linear', 'logistic', 'step'

    def get_effect_multiplier(self, years_since_start: float) -> float:
        """Get effect multiplier at given time since policy start."""
        if years_since_start < self.lag_years:
            return 0.0

        years_active = years_since_start - self.lag_years

        if self.curve_type == 'step':
            return self.adoption_rate

        elif self.curve_type == 'linear':
            progress = min(years_active / self.ramp_up_years, 1.0)
            return progress * self.adoption_rate

        elif self.curve_type == 'logistic':
            # S-curve: slow start, fast middle, slow end
            if self.ramp_up_years <= 0:
                return self.adoption_rate
            midpoint = self.ramp_up_years / 2
            steepness = 4 / self.ramp_up_years  # Controls curve shape
            progress = 1 / (1 + np.exp(-steepness * (years_active - midpoint)))
            return progress * self.adoption_rate

        return self.adoption_rate


@dataclass
class PolicyIntervention:
    """Defines a policy intervention and its effects on model parameters."""
    name: str
    category: InterventionCategory
    description: str
    cost_usd: float
    cost_uncertainty: Tuple[float, float]
    parameter_effects: Dict[str, Callable]
    implementation_lag_years: float = 1.0
    duration_years: float = 10.0
    evidence_quality: int = 3
    notes: str = ""
    # P2-15: Implementation curve
    implementation_curve: Optional[ImplementationCurve] = None


# Policy interventions with P2-15 implementation curves
POLICY_INTERVENTIONS = {
    "ai_research_doubling": PolicyIntervention(
        name="Double AI Research Funding",
        category=InterventionCategory.AI_INVESTMENT,
        description="Double federal funding for AI in biomedical research.",
        cost_usd=3_000_000_000,  # Updated from $5B
        cost_uncertainty=(0.8, 1.3),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.35, 0.90),
            "g_cognitive": lambda g: min(g * 1.30, 0.85),
            "g_scientific": lambda g: min(g * 1.25, 0.80),
        },
        implementation_lag_years=2.0,
        duration_years=10.0,
        evidence_quality=3,
        # P2-15: Gradual ramp-up
        implementation_curve=ImplementationCurve(
            lag_years=2.0, ramp_up_years=3.0, adoption_rate=0.9, curve_type='logistic'
        ),
    ),

    "adaptive_trials_expansion": PolicyIntervention(
        name="Expand Adaptive Trial Designs",
        category=InterventionCategory.REGULATORY_REFORM,
        description="FDA guidance expansion for adaptive and Bayesian trials.",
        cost_usd=200_000_000,
        cost_uncertainty=(0.7, 1.5),
        parameter_effects={
            "M_max_phase2": lambda m: min(m * 1.20, 4.0),
            "M_max_phase3": lambda m: min(m * 1.15, 3.0),
            "p_phase2": lambda p: min(p * 1.10, 0.60),
        },
        implementation_lag_years=1.5,
        duration_years=15.0,
        evidence_quality=4,
        implementation_curve=ImplementationCurve(
            lag_years=1.5, ramp_up_years=2.0, adoption_rate=0.85, curve_type='logistic'
        ),
    ),

    "accelerated_approval_expansion": PolicyIntervention(
        name="Expand Accelerated Approval Pathways",
        category=InterventionCategory.REGULATORY_REFORM,
        description="Broaden criteria for accelerated approval.",
        cost_usd=150_000_000,
        cost_uncertainty=(0.6, 1.5),
        parameter_effects={
            "M_max_regulatory": lambda m: min(m * 1.30, 2.5),
            "M_max_phase3": lambda m: min(m * 1.10, 3.0),
        },
        implementation_lag_years=1.0,
        duration_years=10.0,
        evidence_quality=4,
        implementation_curve=ImplementationCurve(
            lag_years=1.0, ramp_up_years=2.0, adoption_rate=0.90, curve_type='logistic'
        ),
    ),

    "real_world_evidence": PolicyIntervention(
        name="Real-World Evidence Integration",
        category=InterventionCategory.REGULATORY_REFORM,
        description="Expand use of RWD/RWE for regulatory decisions.",
        cost_usd=400_000_000,
        cost_uncertainty=(0.7, 1.4),
        parameter_effects={
            "M_max_phase3": lambda m: min(m * 1.25, 3.5),
            "M_max_phase2": lambda m: min(m * 1.15, 4.0),
            "p_regulatory": lambda p: min(p * 1.05, 0.98),
        },
        implementation_lag_years=4.0,
        duration_years=15.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=4.0, ramp_up_years=5.0, adoption_rate=0.70, curve_type='logistic'
        ),
    ),

    "federated_health_data": PolicyIntervention(
        name="Federated Health Data Network",
        category=InterventionCategory.DATA_INFRASTRUCTURE,
        description="National federated learning infrastructure for health data.",
        cost_usd=1_500_000_000,
        cost_uncertainty=(0.8, 1.5),
        parameter_effects={
            "D_improvement": lambda d: min(d * 1.30, 1.0),
            "g_ai": lambda g: min(g * 1.10, 0.90),
            "g_cognitive": lambda g: min(g * 1.15, 0.85),
        },
        implementation_lag_years=3.0,
        duration_years=20.0,
        evidence_quality=2,
        implementation_curve=ImplementationCurve(
            lag_years=3.0, ramp_up_years=5.0, adoption_rate=0.60, curve_type='logistic'
        ),
    ),

    "biobank_expansion": PolicyIntervention(
        name="National Biobank Expansion",
        category=InterventionCategory.DATA_INFRASTRUCTURE,
        description="Expand All of Us to 10M+ participants.",
        cost_usd=800_000_000,
        cost_uncertainty=(0.9, 1.3),
        parameter_effects={
            "D_improvement": lambda d: min(d * 1.20, 1.0),
            "p_phase1": lambda p: min(p * 1.05, 0.80),
            "p_phase2": lambda p: min(p * 1.08, 0.55),
        },
        implementation_lag_years=2.0,
        duration_years=15.0,
        evidence_quality=4,
        implementation_curve=ImplementationCurve(
            lag_years=2.0, ramp_up_years=4.0, adoption_rate=0.85, curve_type='logistic'
        ),
    ),

    "ai_bio_training": PolicyIntervention(
        name="AI-Biology Cross-Training Program",
        category=InterventionCategory.TALENT_DEVELOPMENT,
        description="Fund 5,000 additional PhD/postdoc positions.",
        cost_usd=500_000_000,
        cost_uncertainty=(0.8, 1.2),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.08, 0.90),
            "g_scientific": lambda g: min(g * 1.12, 0.80),
            "M_max_preclinical": lambda m: min(m * 1.10, 25.0),
        },
        implementation_lag_years=4.0,
        duration_years=20.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=4.0, ramp_up_years=6.0, adoption_rate=0.80, curve_type='logistic'
        ),
    ),

    "target_validation_initiative": PolicyIntervention(
        name="Target Validation Initiative",
        category=InterventionCategory.RESEARCH_FUNDING,
        description="Major initiative to improve target validation using AI.",
        cost_usd=1_000_000_000,
        cost_uncertainty=(0.8, 1.3),
        parameter_effects={
            "p_phase2": lambda p: min(p * 1.25, 0.55),
            "p_phase1": lambda p: min(p * 1.10, 0.80),
            "g_scientific": lambda g: min(g * 1.10, 0.80),
        },
        implementation_lag_years=3.0,
        duration_years=15.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=3.0, ramp_up_years=4.0, adoption_rate=0.75, curve_type='logistic'
        ),
    ),

    "translational_science_centers": PolicyIntervention(
        name="Translational Science Centers",
        category=InterventionCategory.RESEARCH_FUNDING,
        description="Expand NCATS-like centers with AI-enabled screening.",
        cost_usd=1_200_000_000,
        cost_uncertainty=(0.7, 1.3),
        parameter_effects={
            "M_max_preclinical": lambda m: min(m * 1.25, 25.0),
            "p_preclinical": lambda p: min(p * 1.08, 0.65),
        },
        implementation_lag_years=2.0,
        duration_years=15.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=2.0, ramp_up_years=3.0, adoption_rate=0.80, curve_type='logistic'
        ),
    ),

    "harmonized_regulations": PolicyIntervention(
        name="International Regulatory Harmonization",
        category=InterventionCategory.INTERNATIONAL_COORDINATION,
        description="ICH harmonization with mutual recognition.",
        cost_usd=300_000_000,
        cost_uncertainty=(0.5, 2.0),
        parameter_effects={
            "M_max_phase3": lambda m: min(m * 1.20, 3.5),
            "parallelization": lambda p: min(p * 1.15, 1.8),
        },
        implementation_lag_years=3.0,
        duration_years=20.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=3.0, ramp_up_years=5.0, adoption_rate=0.60, curve_type='logistic'
        ),
    ),

    "ai_compute_infrastructure": PolicyIntervention(
        name="National AI Compute Infrastructure",
        category=InterventionCategory.AI_INVESTMENT,
        description="Build dedicated AI compute for biomedical research.",
        cost_usd=2_000_000_000,
        cost_uncertainty=(0.7, 1.5),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.15, 0.90),
            "g_cognitive": lambda g: min(g * 1.20, 0.85),
        },
        implementation_lag_years=3.0,
        duration_years=15.0,
        evidence_quality=2,
        implementation_curve=ImplementationCurve(
            lag_years=3.0, ramp_up_years=2.0, adoption_rate=0.85, curve_type='logistic'
        ),
    ),

    "immigration_reform": PolicyIntervention(
        name="STEM Immigration Reform",
        category=InterventionCategory.TALENT_DEVELOPMENT,
        description="Expand H-1B and green card pathways for AI/bio talent.",
        cost_usd=100_000_000,
        cost_uncertainty=(0.5, 2.0),
        parameter_effects={
            "g_ai": lambda g: min(g * 1.05, 0.90),
            "g_scientific": lambda g: min(g * 1.08, 0.80),
        },
        implementation_lag_years=2.0,
        duration_years=20.0,
        evidence_quality=3,
        implementation_curve=ImplementationCurve(
            lag_years=2.0, ramp_up_years=3.0, adoption_rate=0.70, curve_type='logistic'
        ),
    ),
}


@dataclass
class PolicyAnalysisConfig:
    """Configuration for policy analysis."""
    # P2-16: Expanded QALY range
    value_per_qaly_low: float = 50_000   # NICE lower bound
    value_per_qaly_mid: float = 100_000  # ICER midpoint
    value_per_qaly_high: float = 200_000 # US willingness-to-pay
    value_per_qaly: float = 100_000  # Default

    discount_rate: float = 0.03
    qaly_per_cure: float = 4.0
    analysis_years: int = 26
    n_monte_carlo: int = 1000
    baseline_scenario: str = "Baseline"

    qaly_weights: dict = None

    def __post_init__(self):
        if self.qaly_weights is None:
            self.qaly_weights = {
                'cancer_early_cure': 15.0,
                'cancer_late_treatment': 3.0,
                'alzheimers_therapy': 2.0,
                'pandemic_vaccine': 0.3,
                'rare_disease_cure': 12.0,
                'infectious_cure': 8.0,
                'default': 4.0,
            }


class PolicyAnalysisModule:
    """Analyzes policy interventions with implementation curves."""

    def __init__(self, config: Optional[PolicyAnalysisConfig] = None):
        self.config = config or PolicyAnalysisConfig()
        self.interventions = POLICY_INTERVENTIONS

    def get_intervention(self, name: str) -> PolicyIntervention:
        """Get intervention by name."""
        return self.interventions[name]

    def get_implementation_effect(
        self,
        intervention_key: str,
        years_since_start: float
    ) -> float:
        """
        P2-15: Get implementation effect multiplier at given time.

        Accounts for lag, ramp-up, and adoption rate.
        """
        intervention = self.interventions[intervention_key]
        if intervention.implementation_curve:
            return intervention.implementation_curve.get_effect_multiplier(years_since_start)
        # Fallback: step function after lag
        if years_since_start >= intervention.implementation_lag_years:
            return 1.0
        return 0.0

    def estimate_intervention_effect_simple(
        self,
        intervention_key: str,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
        value_per_qaly: Optional[float] = None,
    ) -> Dict:
        """Simple estimation of intervention effect."""
        intervention = self.interventions[intervention_key]

        if value_per_qaly is None:
            value_per_qaly = self.config.value_per_qaly

        acceleration_boost = 1.0

        for param, modifier in intervention.parameter_effects.items():
            if 'g_ai' in param or 'g_' in param:
                old_g = 0.50
                new_g = modifier(old_g)
                boost = (new_g / old_g) ** 0.5
                acceleration_boost *= boost
            elif 'M_max' in param:
                old_m = 2.5
                new_m = modifier(old_m)
                boost = (new_m / old_m) ** 0.3
                acceleration_boost *= boost
            elif 'p_' in param:
                old_p = 0.30
                new_p = modifier(old_p)
                boost = (new_p / old_p) ** 0.4
                acceleration_boost *= boost

        # P2-15: Apply implementation curve effect (average over horizon)
        if intervention.implementation_curve:
            avg_effect = 0
            for year in range(self.config.analysis_years):
                avg_effect += self.get_implementation_effect(intervention_key, year)
            avg_effect /= self.config.analysis_years
            acceleration_boost = 1 + (acceleration_boost - 1) * avg_effect

        modified_acceleration = baseline_acceleration * acceleration_boost
        delta_acceleration = modified_acceleration - baseline_acceleration

        modified_beneficiaries = baseline_beneficiaries * (modified_acceleration / baseline_acceleration)
        delta_beneficiaries = modified_beneficiaries - baseline_beneficiaries

        value_generated = (
            delta_beneficiaries *
            self.config.qaly_per_cure *
            value_per_qaly
        )

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
            'implementation_lag_years': intervention.implementation_lag_years,
            'evidence_quality': intervention.evidence_quality,
        }

    def rank_interventions(
        self,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
        rank_by: str = 'roi',
    ) -> pd.DataFrame:
        """Rank all interventions by specified metric."""
        results = []
        for key in self.interventions.keys():
            effect = self.estimate_intervention_effect_simple(
                key, baseline_acceleration, baseline_beneficiaries
            )
            results.append(effect)

        df = pd.DataFrame(results)

        if rank_by == 'cost_per_qaly':
            df = df.sort_values(rank_by, ascending=True)
        else:
            df = df.sort_values(rank_by, ascending=False)

        df['rank'] = range(1, len(df) + 1)
        return df

    def sensitivity_to_qaly_value(
        self,
        intervention_key: str,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
    ) -> pd.DataFrame:
        """P2-16: Sensitivity analysis across QALY value range."""
        qaly_values = [
            self.config.value_per_qaly_low,
            self.config.value_per_qaly_mid,
            self.config.value_per_qaly_high,
        ]

        results = []
        for qaly_value in qaly_values:
            effect = self.estimate_intervention_effect_simple(
                intervention_key,
                baseline_acceleration,
                baseline_beneficiaries,
                value_per_qaly=qaly_value
            )
            results.append({
                'intervention': intervention_key,
                'value_per_qaly': qaly_value,
                'roi': effect['roi'],
                'value_generated_usd': effect['value_generated_usd'],
            })

        return pd.DataFrame(results)

    def recommend_portfolio(
        self,
        budget_usd: float,
        baseline_acceleration: float = 5.7,
        baseline_beneficiaries: float = 500_000_000,
        min_evidence_quality: int = 2,
    ) -> Dict:
        """Recommend optimal portfolio given budget constraint."""
        rankings = self.rank_interventions(
            baseline_acceleration, baseline_beneficiaries, rank_by='roi'
        )
        rankings = rankings[rankings['evidence_quality'] >= min_evidence_quality]

        selected = []
        remaining_budget = budget_usd

        for _, row in rankings.iterrows():
            if row['annual_cost_usd'] <= remaining_budget:
                selected.append(row['intervention_key'])
                remaining_budget -= row['annual_cost_usd']

        return {
            'budget_usd': budget_usd,
            'selected_interventions': selected,
            'remaining_budget': remaining_budget,
        }


def create_default_module() -> PolicyAnalysisModule:
    """Create policy analysis module with default configuration."""
    return PolicyAnalysisModule(PolicyAnalysisConfig())


if __name__ == "__main__":
    print("=" * 70)
    print("Policy Analysis Module - v1.1")
    print("=" * 70)

    module = create_default_module()

    print("\nP2-15 Implementation Curves:")
    print("-" * 70)
    for key, intervention in module.interventions.items():
        if intervention.implementation_curve:
            curve = intervention.implementation_curve
            print(f"  {key[:30]:30s}: lag={curve.lag_years}yr, ramp={curve.ramp_up_years}yr, "
                  f"adoption={curve.adoption_rate:.0%}")

    print("\n\nP2-16 QALY Value Range:")
    print("-" * 70)
    print(f"  Low (NICE):  ${module.config.value_per_qaly_low:,}")
    print(f"  Mid (ICER):  ${module.config.value_per_qaly_mid:,}")
    print(f"  High (US):   ${module.config.value_per_qaly_high:,}")

    print("\n\nIntervention Rankings (by ROI):")
    print("-" * 70)
    rankings = module.rank_interventions()
    cols = ['rank', 'intervention_name', 'annual_cost_usd', 'roi', 'evidence_quality']
    print(rankings[cols].head(10).to_string(index=False))

    print("\nModule loaded successfully.")
