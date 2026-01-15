#!/usr/bin/env python3
"""
Enhanced Features Module for AI Research Acceleration Model v1.1
================================================================

Adds features identified in gap analysis:
1. Policy ROI calculations (HIGH priority)
2. Bottleneck transition timeline (MEDIUM priority)
3. Multi-type AI breakdown (LOW priority)
4. Data quality module (LOW priority)

These extend the base model without modifying it.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Import base model
try:
    from ai_acceleration_model import AIAccelerationModel, ParameterSource
except ImportError:
    AIAccelerationModel = None


# =============================================================================
# 1. POLICY ROI CALCULATIONS (HIGH PRIORITY)
# =============================================================================

@dataclass
class PolicyIntervention:
    """
    A policy intervention with cost and effect parameters.

    Based on ai_bio_acceleration_model v0.9 policy module.
    """
    id: str
    name: str
    description: str
    category: str

    # Cost parameters
    cost_annual_millions: float  # $M/year
    duration_years: int
    implementation_lag_years: float = 1.0

    # Effect parameters (multipliers on domain acceleration)
    domain_effects: Dict[str, float] = field(default_factory=dict)

    # Meta
    evidence_quality: int = 3  # 1-5 scale
    evidence_source: str = ""


# Policy interventions library (from ai_bio_acceleration_model v0.9)
POLICY_INTERVENTIONS = [
    # Regulatory Reform (highest ROI)
    PolicyIntervention(
        id="REG-001",
        name="Adaptive Trial Expansion",
        description="Expand FDA adaptive trial pathways for AI-optimized designs",
        category="Regulatory Reform",
        cost_annual_millions=200,
        duration_years=5,
        implementation_lag_years=2.0,
        domain_effects={"drug_discovery": 1.15, "clinical_genomics": 1.05},
        evidence_quality=4,
        evidence_source="FDA Modernization Act 2.0; adaptive trial literature"
    ),
    PolicyIntervention(
        id="REG-002",
        name="Accelerated Approval Expansion",
        description="Extend accelerated approval to more disease areas",
        category="Regulatory Reform",
        cost_annual_millions=150,
        duration_years=5,
        implementation_lag_years=1.5,
        domain_effects={"drug_discovery": 1.10},
        evidence_quality=4,
        evidence_source="FDA CDER reports"
    ),
    PolicyIntervention(
        id="REG-003",
        name="Real-World Evidence Integration",
        description="Integrate RWE into regulatory decision-making",
        category="Regulatory Reform",
        cost_annual_millions=400,
        duration_years=7,
        implementation_lag_years=4.0,
        domain_effects={"drug_discovery": 1.12, "clinical_genomics": 1.08},
        evidence_quality=3,
        evidence_source="21st Century Cures Act; RWE guidance"
    ),

    # Infrastructure
    PolicyIntervention(
        id="INF-001",
        name="Cryo-EM Infrastructure Expansion",
        description="Fund national cryo-EM facility network expansion",
        category="Infrastructure",
        cost_annual_millions=100,
        duration_years=5,
        implementation_lag_years=2.0,
        domain_effects={"structural_biology": 1.20, "protein_design": 1.05},
        evidence_quality=5,
        evidence_source="NIH cryo-EM working group; structural biology surveys"
    ),
    PolicyIntervention(
        id="INF-002",
        name="Autonomous Synthesis Facilities",
        description="A-Lab style automated materials synthesis facilities",
        category="Infrastructure",
        cost_annual_millions=60,
        duration_years=5,
        implementation_lag_years=3.0,
        domain_effects={"materials_science": 1.35},
        evidence_quality=4,
        evidence_source="A-Lab publications; GNoME backlog analysis"
    ),
    PolicyIntervention(
        id="INF-003",
        name="Federated Health Data Network",
        description="National federated learning infrastructure for health data",
        category="Infrastructure",
        cost_annual_millions=300,
        duration_years=5,
        implementation_lag_years=3.0,
        domain_effects={"drug_discovery": 1.08, "clinical_genomics": 1.15},
        evidence_quality=3,
        evidence_source="NIH data sharing initiatives"
    ),

    # AI Investment
    PolicyIntervention(
        id="AI-001",
        name="AI Biology Research Doubling",
        description="Double federal AI-biology research funding",
        category="AI Investment",
        cost_annual_millions=600,
        duration_years=5,
        implementation_lag_years=1.0,
        domain_effects={
            "structural_biology": 1.10,
            "drug_discovery": 1.08,
            "materials_science": 1.08,
            "protein_design": 1.12,
            "clinical_genomics": 1.10
        },
        evidence_quality=4,
        evidence_source="NSF/NIH AI research trends"
    ),
    PolicyIntervention(
        id="AI-002",
        name="AI Compute Infrastructure",
        description="Dedicated compute for biology AI models",
        category="AI Investment",
        cost_annual_millions=400,
        duration_years=5,
        implementation_lag_years=1.5,
        domain_effects={
            "structural_biology": 1.08,
            "protein_design": 1.10,
            "materials_science": 1.06
        },
        evidence_quality=3,
        evidence_source="National AI Research Resource plans"
    ),

    # Workforce
    PolicyIntervention(
        id="WF-001",
        name="AI-Biology Training Programs",
        description="Graduate and postdoc programs at AI-biology interface",
        category="Workforce",
        cost_annual_millions=100,
        duration_years=10,
        implementation_lag_years=3.0,
        domain_effects={
            "structural_biology": 1.05,
            "drug_discovery": 1.05,
            "protein_design": 1.08,
            "clinical_genomics": 1.05
        },
        evidence_quality=4,
        evidence_source="NSF CAREER awards; industry hiring data"
    ),

    # International
    PolicyIntervention(
        id="INT-001",
        name="Regulatory Harmonization",
        description="International regulatory harmonization for AI-derived therapies",
        category="International",
        cost_annual_millions=60,
        duration_years=5,
        implementation_lag_years=3.0,
        domain_effects={"drug_discovery": 1.08},
        evidence_quality=3,
        evidence_source="ICH guidelines; FDA/EMA cooperation"
    ),
]


@dataclass
class PolicyROIResult:
    """Results of policy ROI calculation."""
    intervention: PolicyIntervention
    baseline_acceleration: float
    enhanced_acceleration: float
    delta_acceleration: float
    total_cost_millions: float
    roi_per_billion: float  # Acceleration gain per $1B
    npv_acceleration: float  # NPV of acceleration gains
    payback_years: float
    affected_domains: List[str]


class PolicyROICalculator:
    """
    Calculate ROI for policy interventions.

    Methodology from ai_bio_acceleration_model v0.9.
    """

    def __init__(self, model: 'AIAccelerationModel' = None):
        """Initialize with model instance."""
        if model is None and AIAccelerationModel is not None:
            model = AIAccelerationModel()
        self.model = model
        self.discount_rate = 0.03  # 3% per NICE/ICER

    def calculate_roi(
        self,
        intervention: PolicyIntervention,
        year: int = 2030
    ) -> PolicyROIResult:
        """Calculate ROI for a single intervention."""
        # Baseline acceleration
        baseline = self.model.system_snapshot(year)
        baseline_accel = baseline.total_acceleration

        # Enhanced acceleration (apply domain effects)
        enhanced_accel = self._calculate_enhanced_acceleration(
            intervention, year
        )

        delta_accel = enhanced_accel - baseline_accel

        # Total cost
        total_cost = intervention.cost_annual_millions * intervention.duration_years

        # ROI metrics
        roi_per_billion = delta_accel / (total_cost / 1000) if total_cost > 0 else 0

        # NPV of acceleration gains
        npv = self._calculate_npv(delta_accel, intervention, year)

        # Payback (years to recover investment in acceleration terms)
        annual_gain = delta_accel / intervention.duration_years
        payback = total_cost / (annual_gain * 1000) if annual_gain > 0 else float('inf')

        return PolicyROIResult(
            intervention=intervention,
            baseline_acceleration=baseline_accel,
            enhanced_acceleration=enhanced_accel,
            delta_acceleration=delta_accel,
            total_cost_millions=total_cost,
            roi_per_billion=roi_per_billion,
            npv_acceleration=npv,
            payback_years=payback,
            affected_domains=list(intervention.domain_effects.keys())
        )

    def _calculate_enhanced_acceleration(
        self,
        intervention: PolicyIntervention,
        year: int
    ) -> float:
        """Calculate system acceleration with intervention applied."""
        # Get domain forecasts
        domain_accels = {}
        for domain in self.model.domains:
            forecast = self.model.forecast(domain, year)

            # Apply intervention effect if applicable
            effect = intervention.domain_effects.get(domain, 1.0)

            # Account for implementation lag
            effective_years = year - 2024 - intervention.implementation_lag_years
            if effective_years <= 0:
                effect = 1.0  # Not yet effective
            elif effective_years < intervention.duration_years:
                # Partial effect during ramp-up
                ramp_fraction = effective_years / intervention.duration_years
                effect = 1.0 + (effect - 1.0) * ramp_fraction

            domain_accels[domain] = forecast.acceleration * effect

        # Calculate system acceleration (geometric mean with OECD weights)
        weights = {
            "structural_biology": 0.12,
            "drug_discovery": 0.45,
            "materials_science": 0.18,
            "protein_design": 0.15,
            "clinical_genomics": 0.10,
        }

        log_weighted = sum(
            weights[d] * np.log(domain_accels[d])
            for d in self.model.domains
        )
        return np.exp(log_weighted)

    def _calculate_npv(
        self,
        delta_accel: float,
        intervention: PolicyIntervention,
        year: int
    ) -> float:
        """Calculate NPV of acceleration gains."""
        npv = 0
        for t in range(intervention.duration_years):
            year_t = year + t
            # Discount future gains
            discount = (1 + self.discount_rate) ** t
            npv += delta_accel / discount
        return npv

    def rank_interventions(
        self,
        year: int = 2030,
        budget_millions: float = None
    ) -> List[PolicyROIResult]:
        """
        Rank all interventions by ROI.

        If budget provided, returns optimal portfolio under constraint.
        """
        results = []
        for intervention in POLICY_INTERVENTIONS:
            try:
                result = self.calculate_roi(intervention, year)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not calculate ROI for {intervention.name}: {e}")

        # Sort by ROI
        results.sort(key=lambda x: x.roi_per_billion, reverse=True)

        if budget_millions is not None:
            # Greedy selection under budget
            selected = []
            remaining_budget = budget_millions
            for result in results:
                if result.total_cost_millions <= remaining_budget:
                    selected.append(result)
                    remaining_budget -= result.total_cost_millions
            return selected

        return results

    def portfolio_analysis(
        self,
        budget_millions: float = 10000,
        year: int = 2030
    ) -> Dict:
        """Analyze optimal portfolio under budget constraint."""
        portfolio = self.rank_interventions(year, budget_millions)

        total_cost = sum(r.total_cost_millions for r in portfolio)
        total_delta = sum(r.delta_acceleration for r in portfolio)

        # Get baseline
        baseline = self.model.system_snapshot(year).total_acceleration

        return {
            "budget": budget_millions,
            "interventions_selected": len(portfolio),
            "total_cost_millions": total_cost,
            "baseline_acceleration": baseline,
            "portfolio_acceleration": baseline + total_delta,
            "total_delta_acceleration": total_delta,
            "portfolio_roi": total_delta / (total_cost / 1000) if total_cost > 0 else 0,
            "interventions": [r.intervention.name for r in portfolio],
            "by_category": self._group_by_category(portfolio),
        }

    def _group_by_category(self, results: List[PolicyROIResult]) -> Dict:
        """Group results by intervention category."""
        categories = {}
        for r in results:
            cat = r.intervention.category
            if cat not in categories:
                categories[cat] = {"count": 0, "cost": 0, "delta": 0}
            categories[cat]["count"] += 1
            categories[cat]["cost"] += r.total_cost_millions
            categories[cat]["delta"] += r.delta_acceleration
        return categories


# =============================================================================
# 2. BOTTLENECK TRANSITION TIMELINE (MEDIUM PRIORITY)
# =============================================================================

@dataclass
class BottleneckTransition:
    """Records when bottleneck shifts from one domain to another."""
    year: int
    from_domain: str
    to_domain: str
    from_bottleneck: str
    to_bottleneck: str


class BottleneckAnalyzer:
    """
    Analyze bottleneck transitions over time.

    Tracks when the binding constraint shifts between domains.
    """

    def __init__(self, model: 'AIAccelerationModel' = None):
        if model is None and AIAccelerationModel is not None:
            model = AIAccelerationModel()
        self.model = model

    def get_bottleneck_timeline(
        self,
        start_year: int = 2024,
        end_year: int = 2040
    ) -> List[Dict]:
        """
        Get timeline of bottleneck status by year.

        Returns list of {year, domain_bottlenecks, system_bottleneck}
        """
        timeline = []

        for year in range(start_year, end_year + 1):
            snapshot = self.model.system_snapshot(year)

            # Domain-level bottlenecks
            domain_bottlenecks = {}
            for domain, forecast in snapshot.domain_forecasts.items():
                domain_bottlenecks[domain] = {
                    "bottleneck": forecast.primary_bottleneck,
                    "fraction": forecast.bottleneck_fraction,
                    "acceleration": forecast.acceleration,
                    "headroom": self._calculate_headroom(domain, year)
                }

            # System bottleneck (slowest domain)
            system_bottleneck = min(
                snapshot.domain_forecasts.items(),
                key=lambda x: x[1].acceleration
            )[0]

            timeline.append({
                "year": year,
                "domain_bottlenecks": domain_bottlenecks,
                "system_bottleneck": system_bottleneck,
                "system_acceleration": snapshot.total_acceleration
            })

        return timeline

    def _calculate_headroom(self, domain: str, year: int) -> float:
        """Calculate how much acceleration headroom remains (vs ceiling)."""
        params = self.model.TIME_EVOLUTION[domain]
        ceiling = params["ceiling"]
        base = self.model.BASE_PARAMETERS[domain].value
        current = self.model.forecast(domain, year).acceleration

        # Headroom as fraction of ceiling
        return (ceiling * base - current) / (ceiling * base)

    def detect_transitions(
        self,
        start_year: int = 2024,
        end_year: int = 2040
    ) -> List[BottleneckTransition]:
        """Detect when system bottleneck shifts between domains."""
        timeline = self.get_bottleneck_timeline(start_year, end_year)
        transitions = []

        prev_bottleneck = None
        for entry in timeline:
            current_bottleneck = entry["system_bottleneck"]

            if prev_bottleneck is not None and current_bottleneck != prev_bottleneck:
                # Transition detected
                prev_entry = timeline[timeline.index(entry) - 1]

                transitions.append(BottleneckTransition(
                    year=entry["year"],
                    from_domain=prev_bottleneck,
                    to_domain=current_bottleneck,
                    from_bottleneck=prev_entry["domain_bottlenecks"][prev_bottleneck]["bottleneck"],
                    to_bottleneck=entry["domain_bottlenecks"][current_bottleneck]["bottleneck"]
                ))

            prev_bottleneck = current_bottleneck

        return transitions

    def get_bottleneck_summary(self, year: int = 2030) -> Dict:
        """Get summary of bottleneck status for a specific year."""
        snapshot = self.model.system_snapshot(year)

        bottleneck_ranking = sorted(
            snapshot.domain_forecasts.items(),
            key=lambda x: x[1].acceleration
        )

        return {
            "year": year,
            "system_bottleneck": bottleneck_ranking[0][0],
            "bottleneck_acceleration": bottleneck_ranking[0][1].acceleration,
            "fastest_domain": bottleneck_ranking[-1][0],
            "fastest_acceleration": bottleneck_ranking[-1][1].acceleration,
            "bottleneck_gap": (
                bottleneck_ranking[-1][1].acceleration /
                bottleneck_ranking[0][1].acceleration
            ),
            "ranking": [(d, f.acceleration) for d, f in bottleneck_ranking]
        }


# =============================================================================
# 3. MULTI-TYPE AI BREAKDOWN (LOW PRIORITY)
# =============================================================================

class AIType(Enum):
    """Types of AI capability with different characteristics."""
    COGNITIVE = "cognitive"      # Language, reasoning (GPT, Claude)
    ROBOTIC = "robotic"         # Physical manipulation, lab automation
    SCIENTIFIC = "scientific"   # Hypothesis, prediction (AlphaFold)


@dataclass
class AITypeParams:
    """Parameters for each AI type."""
    ai_type: AIType
    base_growth_rate: float
    max_growth_rate: float
    current_capability: float
    description: str


# AI type parameters (from ai_bio_acceleration_model)
AI_TYPE_DEFAULTS = {
    AIType.COGNITIVE: AITypeParams(
        ai_type=AIType.COGNITIVE,
        base_growth_rate=0.60,
        max_growth_rate=0.80,
        current_capability=1.0,
        description="Language, reasoning, synthesis (GPT, Claude)"
    ),
    AIType.ROBOTIC: AITypeParams(
        ai_type=AIType.ROBOTIC,
        base_growth_rate=0.30,
        max_growth_rate=0.50,
        current_capability=1.0,
        description="Physical manipulation, lab automation"
    ),
    AIType.SCIENTIFIC: AITypeParams(
        ai_type=AIType.SCIENTIFIC,
        base_growth_rate=0.55,
        max_growth_rate=0.75,
        current_capability=1.0,
        description="Hypothesis generation, prediction (AlphaFold)"
    ),
}


# Domain-to-AI-type weights
DOMAIN_AI_WEIGHTS = {
    "structural_biology": {
        AIType.COGNITIVE: 0.2,
        AIType.ROBOTIC: 0.2,  # Cryo-EM operation
        AIType.SCIENTIFIC: 0.6  # AlphaFold-like
    },
    "drug_discovery": {
        AIType.COGNITIVE: 0.3,
        AIType.ROBOTIC: 0.4,  # HTS, clinical trials
        AIType.SCIENTIFIC: 0.3
    },
    "materials_science": {
        AIType.COGNITIVE: 0.2,
        AIType.ROBOTIC: 0.5,  # Synthesis
        AIType.SCIENTIFIC: 0.3
    },
    "protein_design": {
        AIType.COGNITIVE: 0.3,
        AIType.ROBOTIC: 0.3,  # Expression, validation
        AIType.SCIENTIFIC: 0.4
    },
    "clinical_genomics": {
        AIType.COGNITIVE: 0.4,  # Interpretation
        AIType.ROBOTIC: 0.2,  # Sequencing
        AIType.SCIENTIFIC: 0.4  # Variant prediction
    }
}


class MultiTypeAIAnalyzer:
    """
    Analyze AI impact by type (cognitive, robotic, scientific).

    Shows which AI capabilities drive acceleration in each domain.
    """

    def __init__(self, model: 'AIAccelerationModel' = None):
        if model is None and AIAccelerationModel is not None:
            model = AIAccelerationModel()
        self.model = model

    def get_ai_type_contributions(
        self,
        domain: str,
        year: int = 2030
    ) -> Dict[AIType, float]:
        """
        Calculate contribution of each AI type to domain acceleration.

        Returns dict mapping AI type to contribution (fraction of total).
        """
        weights = DOMAIN_AI_WEIGHTS.get(domain, {})
        total_accel = self.model.forecast(domain, year).acceleration

        contributions = {}
        for ai_type, weight in weights.items():
            # AI type contribution = weight × type capability growth
            params = AI_TYPE_DEFAULTS[ai_type]
            t = year - 2024
            type_capability = np.exp(params.base_growth_rate * t)
            contributions[ai_type] = weight * type_capability

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def get_ai_type_summary(self, year: int = 2030) -> Dict:
        """Get summary of AI type contributions across all domains."""
        summary = {
            "year": year,
            "by_domain": {},
            "by_type": {t: 0 for t in AIType},
            "limiting_type": None
        }

        for domain in self.model.domains:
            contributions = self.get_ai_type_contributions(domain, year)
            summary["by_domain"][domain] = contributions

            # Aggregate across domains
            for ai_type, contrib in contributions.items():
                summary["by_type"][ai_type] += contrib

        # Normalize
        total = sum(summary["by_type"].values())
        if total > 0:
            summary["by_type"] = {k: v / total for k, v in summary["by_type"].items()}

        # Identify limiting type (lowest growth rate)
        summary["limiting_type"] = min(
            AI_TYPE_DEFAULTS.items(),
            key=lambda x: x[1].base_growth_rate
        )[0]

        return summary


# =============================================================================
# 4. DATA QUALITY MODULE (LOW PRIORITY)
# =============================================================================

@dataclass
class DataQualityParams:
    """Parameters for data quality evolution."""
    D0: float = 1.0  # Baseline data quality (2024)
    gamma: float = 0.08  # Growth coefficient
    domain_elasticities: Dict[str, float] = field(default_factory=dict)


# Domain elasticities to data quality
DATA_QUALITY_ELASTICITIES = {
    "structural_biology": 0.6,  # Moderate - needs experimental validation
    "drug_discovery": 0.4,      # Lower - clinical trials standardized
    "materials_science": 0.5,   # Moderate - characterization data important
    "protein_design": 0.7,      # High - design success depends on training data
    "clinical_genomics": 0.8,   # Highest - variant interpretation is data-driven
}


class DataQualityModule:
    """
    Model data quality improvement and its effect on acceleration.

    Based on ai_bio_acceleration_model v0.6 data quality module.
    """

    def __init__(
        self,
        model: 'AIAccelerationModel' = None,
        params: DataQualityParams = None
    ):
        if model is None and AIAccelerationModel is not None:
            model = AIAccelerationModel()
        self.model = model
        self.params = params or DataQualityParams(
            domain_elasticities=DATA_QUALITY_ELASTICITIES
        )

    def data_quality_index(self, year: int) -> float:
        """
        Calculate data quality index D(t).

        Formula: D(t) = D0 × (1 + gamma × log(A(t)))
        where A(t) is AI capability
        """
        t = year - 2024
        # Simple AI capability proxy
        A_t = np.exp(0.5 * t)  # ~50% annual AI improvement

        D_t = self.params.D0 * (1 + self.params.gamma * np.log(max(1, A_t)))
        return D_t

    def data_quality_multiplier(self, domain: str, year: int) -> float:
        """
        Calculate data quality multiplier for a domain.

        Formula: DQM = (D(t) / D0) ^ elasticity
        """
        D_t = self.data_quality_index(year)
        elasticity = self.params.domain_elasticities.get(domain, 0.5)

        DQM = (D_t / self.params.D0) ** elasticity
        return DQM

    def get_data_quality_summary(self, year: int = 2030) -> Dict:
        """Get summary of data quality effects."""
        D_t = self.data_quality_index(year)

        multipliers = {}
        for domain in self.model.domains:
            multipliers[domain] = self.data_quality_multiplier(domain, year)

        return {
            "year": year,
            "data_quality_index": D_t,
            "baseline_index": self.params.D0,
            "improvement_factor": D_t / self.params.D0,
            "domain_multipliers": multipliers,
            "highest_impact_domain": max(multipliers, key=multipliers.get),
            "lowest_impact_domain": min(multipliers, key=multipliers.get),
        }

    def adjust_forecast(self, domain: str, year: int, base_acceleration: float) -> float:
        """Adjust acceleration for data quality effects."""
        DQM = self.data_quality_multiplier(domain, year)
        # Data quality provides additive boost (not multiplicative)
        return base_acceleration * (1 + (DQM - 1) * 0.5)


# =============================================================================
# CONVENIENCE: GET ALL ENHANCEMENTS
# =============================================================================

def get_enhanced_analysis(model: 'AIAccelerationModel' = None, year: int = 2030) -> Dict:
    """
    Run all enhanced analyses and return comprehensive results.
    """
    if model is None and AIAccelerationModel is not None:
        model = AIAccelerationModel()

    results = {
        "year": year,
        "policy_roi": None,
        "bottleneck_analysis": None,
        "ai_type_breakdown": None,
        "data_quality": None,
    }

    # 1. Policy ROI
    try:
        roi_calc = PolicyROICalculator(model)
        results["policy_roi"] = {
            "top_interventions": [
                {
                    "name": r.intervention.name,
                    "roi": r.roi_per_billion,
                    "cost_millions": r.total_cost_millions,
                    "delta_accel": r.delta_acceleration
                }
                for r in roi_calc.rank_interventions(year)[:5]
            ],
            "portfolio_10B": roi_calc.portfolio_analysis(10000, year)
        }
    except Exception as e:
        results["policy_roi"] = {"error": str(e)}

    # 2. Bottleneck analysis
    try:
        bottleneck = BottleneckAnalyzer(model)
        results["bottleneck_analysis"] = {
            "summary": bottleneck.get_bottleneck_summary(year),
            "transitions": [
                {"year": t.year, "from": t.from_domain, "to": t.to_domain}
                for t in bottleneck.detect_transitions(2024, 2040)
            ]
        }
    except Exception as e:
        results["bottleneck_analysis"] = {"error": str(e)}

    # 3. AI type breakdown
    try:
        ai_analyzer = MultiTypeAIAnalyzer(model)
        results["ai_type_breakdown"] = ai_analyzer.get_ai_type_summary(year)
    except Exception as e:
        results["ai_type_breakdown"] = {"error": str(e)}

    # 4. Data quality
    try:
        dq_module = DataQualityModule(model)
        results["data_quality"] = dq_module.get_data_quality_summary(year)
    except Exception as e:
        results["data_quality"] = {"error": str(e)}

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Enhanced Features Module for AI Research Acceleration Model v1.1")
    print("=" * 70)

    if AIAccelerationModel is None:
        print("Warning: Base model not available. Using mock data.")
    else:
        model = AIAccelerationModel()
        results = get_enhanced_analysis(model, 2030)

        print("\n1. POLICY ROI ANALYSIS")
        print("-" * 40)
        if "error" not in results["policy_roi"]:
            print("Top 5 Interventions by ROI:")
            for i, interv in enumerate(results["policy_roi"]["top_interventions"], 1):
                print(f"  {i}. {interv['name']}: ROI={interv['roi']:.1f}/B, Cost=${interv['cost_millions']:.0f}M")

            portfolio = results["policy_roi"]["portfolio_10B"]
            print(f"\n$10B Portfolio: {portfolio['portfolio_acceleration']:.2f}x (vs {portfolio['baseline_acceleration']:.2f}x baseline)")

        print("\n2. BOTTLENECK ANALYSIS")
        print("-" * 40)
        if "error" not in results["bottleneck_analysis"]:
            summary = results["bottleneck_analysis"]["summary"]
            print(f"System bottleneck: {summary['system_bottleneck']}")
            print(f"Bottleneck acceleration: {summary['bottleneck_acceleration']:.2f}x")
            print(f"Fastest domain: {summary['fastest_domain']} ({summary['fastest_acceleration']:.2f}x)")
            print(f"Bottleneck gap: {summary['bottleneck_gap']:.1f}x")

        print("\n3. AI TYPE BREAKDOWN")
        print("-" * 40)
        if "error" not in results["ai_type_breakdown"]:
            by_type = results["ai_type_breakdown"]["by_type"]
            for ai_type, contrib in by_type.items():
                print(f"  {ai_type.value}: {contrib:.1%}")
            print(f"Limiting type: {results['ai_type_breakdown']['limiting_type'].value}")

        print("\n4. DATA QUALITY")
        print("-" * 40)
        if "error" not in results["data_quality"]:
            dq = results["data_quality"]
            print(f"Data quality index: {dq['data_quality_index']:.2f} (baseline: {dq['baseline_index']:.2f})")
            print(f"Improvement factor: {dq['improvement_factor']:.2f}x")
            print(f"Highest impact: {dq['highest_impact_domain']}")
