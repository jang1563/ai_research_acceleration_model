#!/usr/bin/env python3
"""
Policy Recommendation Engine for v0.9
======================================

Addresses Expert Review P1-P3: "Missing policy implications"

Generates actionable policy recommendations based on:
1. Domain-specific acceleration forecasts
2. Bottleneck analysis
3. Workforce impact projections
4. Cross-domain interaction effects
5. Uncertainty quantification

Target Audiences:
- Research funders (NIH, NSF, UKRI, etc.)
- Policymakers (science policy, workforce)
- Industry strategists
- Academic leadership
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class PolicyDomain(Enum):
    """Policy domains for recommendations."""
    RESEARCH_FUNDING = "research_funding"
    WORKFORCE_DEVELOPMENT = "workforce_development"
    REGULATORY = "regulatory"
    INFRASTRUCTURE = "infrastructure"
    INTERNATIONAL = "international"
    ETHICS = "ethics"


class PolicyPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"      # Act within 1 year
    HIGH = "high"              # Act within 2 years
    MEDIUM = "medium"          # Act within 5 years
    LOW = "low"                # Monitor and consider


class StakeholderType(Enum):
    """Types of stakeholders for targeted recommendations."""
    GOVERNMENT = "government"
    ACADEMIA = "academia"
    INDUSTRY = "industry"
    NONPROFIT = "nonprofit"
    INTERNATIONAL = "international"


@dataclass
class PolicyRecommendation:
    """A specific policy recommendation."""
    id: str
    title: str
    description: str

    # Classification
    domain: PolicyDomain
    priority: PolicyPriority
    target_stakeholders: List[StakeholderType]

    # Justification
    rationale: str
    evidence_base: List[str]

    # Impact
    expected_impact: str
    impact_timeline: str  # "immediate", "1-2 years", "3-5 years", "5+ years"

    # Implementation
    implementation_steps: List[str]
    estimated_cost: str  # "low", "medium", "high", "very_high"
    barriers: List[str]

    # Metrics
    success_metrics: List[str]

    # Uncertainty
    confidence: float  # 0-1
    key_assumptions: List[str]


@dataclass
class DomainPolicyAnalysis:
    """Policy analysis for a specific scientific domain."""
    domain: str
    year: int

    # Acceleration context
    current_acceleration: float
    projected_acceleration: float
    primary_bottleneck: str

    # Recommendations
    recommendations: List[PolicyRecommendation]

    # Summary metrics
    total_investment_recommended: str
    workforce_transition_support: str
    regulatory_changes_needed: int


class PolicyRecommendationEngine:
    """
    Generates policy recommendations based on model outputs.

    Combines insights from:
    - Bottleneck analysis
    - Workforce impact projections
    - Cross-domain effects
    - Regulatory scenarios
    """

    def __init__(self):
        self.recommendation_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load recommendation templates by domain."""
        return {
            "structural_biology": [
                {
                    "trigger": "high_acceleration",
                    "threshold": 5.0,
                    "template": PolicyRecommendation(
                        id="SB-001",
                        title="Scale Cryo-EM Infrastructure",
                        description="Expand national cryo-EM facility capacity to validate AI predictions",
                        domain=PolicyDomain.INFRASTRUCTURE,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.ACADEMIA],
                        rationale="AI structure prediction is 10-100x faster than experiments. Validation becomes the bottleneck.",
                        evidence_base=[
                            "AlphaFold2 accuracy metrics (2021-2024)",
                            "Cryo-EM facility utilization data",
                            "Backlog growth in structural validation",
                        ],
                        expected_impact="Reduce validation backlog by 50%",
                        impact_timeline="3-5 years",
                        implementation_steps=[
                            "Assess current facility utilization",
                            "Identify expansion sites",
                            "Fund facility upgrades and new instruments",
                            "Train additional operators",
                        ],
                        estimated_cost="high",
                        barriers=["Facility construction time", "Trained personnel shortage"],
                        success_metrics=["Validation throughput (structures/year)", "Average queue time"],
                        confidence=0.75,
                        key_assumptions=["AI prediction accuracy continues improving"],
                    ),
                },
                {
                    "trigger": "workforce_displacement",
                    "threshold": 0.3,
                    "template": PolicyRecommendation(
                        id="SB-002",
                        title="Crystallographer Transition Program",
                        description="Reskilling program for crystallographers to become AI-augmented structural analysts",
                        domain=PolicyDomain.WORKFORCE_DEVELOPMENT,
                        priority=PolicyPriority.MEDIUM,
                        target_stakeholders=[StakeholderType.ACADEMIA, StakeholderType.NONPROFIT],
                        rationale="Traditional crystallography skills less in demand; AI interpretation skills needed",
                        evidence_base=[
                            "Workforce impact projections",
                            "Job posting trend analysis",
                            "Graduate student career outcomes",
                        ],
                        expected_impact="Successful career transition for 70% of affected workers",
                        impact_timeline="1-2 years",
                        implementation_steps=[
                            "Develop curriculum for AI-augmented structural biology",
                            "Partner with professional societies",
                            "Create fellowship programs for retraining",
                            "Establish industry partnerships for placement",
                        ],
                        estimated_cost="medium",
                        barriers=["Resistance to change", "Funding for stipends"],
                        success_metrics=["Transition rate", "Post-transition employment", "Salary maintenance"],
                        confidence=0.65,
                        key_assumptions=["Demand for AI-augmented analysts grows as predicted"],
                    ),
                },
            ],

            "drug_discovery": [
                {
                    "trigger": "regulatory_bottleneck",
                    "threshold": 0.8,  # When regulatory is >80% of bottleneck
                    "template": PolicyRecommendation(
                        id="DD-001",
                        title="AI-Adaptive Clinical Trial Framework",
                        description="Develop regulatory pathway for AI-optimized adaptive trials",
                        domain=PolicyDomain.REGULATORY,
                        priority=PolicyPriority.CRITICAL,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.INDUSTRY],
                        rationale="Clinical trials are the dominant bottleneck. AI can optimize but not replace human trials under current regulations.",
                        evidence_base=[
                            "v0.8 regulatory scenario analysis",
                            "FDA adaptive trial guidance",
                            "AI trial optimization case studies",
                        ],
                        expected_impact="20-30% reduction in Phase 2/3 timelines",
                        impact_timeline="3-5 years",
                        implementation_steps=[
                            "Convene FDA-industry working group",
                            "Pilot AI-optimized adaptive designs",
                            "Develop validation standards for AI predictions",
                            "Update guidance documents",
                        ],
                        estimated_cost="medium",
                        barriers=["Regulatory conservatism", "Safety concerns", "Industry adoption"],
                        success_metrics=["Time to approval", "Trial efficiency", "Safety maintenance"],
                        confidence=0.55,
                        key_assumptions=["FDA/EMA willing to engage", "Safety not compromised"],
                    ),
                },
                {
                    "trigger": "high_acceleration",
                    "threshold": 2.5,
                    "template": PolicyRecommendation(
                        id="DD-002",
                        title="Preclinical Automation Incentives",
                        description="Tax incentives for automated preclinical testing facilities",
                        domain=PolicyDomain.RESEARCH_FUNDING,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.INDUSTRY],
                        rationale="AI identifies candidates faster than they can be tested. Automation closes the gap.",
                        evidence_base=[
                            "Preclinical throughput data",
                            "Automation ROI studies",
                            "International competitiveness analysis",
                        ],
                        expected_impact="3x increase in preclinical throughput",
                        impact_timeline="3-5 years",
                        implementation_steps=[
                            "Design tax credit structure",
                            "Define qualifying investments",
                            "Create certification process",
                            "Monitor and evaluate impact",
                        ],
                        estimated_cost="high",
                        barriers=["Legislative process", "Budget constraints"],
                        success_metrics=["Automation adoption rate", "Preclinical throughput", "Time to IND"],
                        confidence=0.60,
                        key_assumptions=["Industry responds to incentives"],
                    ),
                },
                {
                    "trigger": "workforce_growth",
                    "threshold": 0.2,
                    "template": PolicyRecommendation(
                        id="DD-003",
                        title="AI-Biology Training Pipeline",
                        description="Expand graduate programs at AI-biology interface",
                        domain=PolicyDomain.WORKFORCE_DEVELOPMENT,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.ACADEMIA, StakeholderType.GOVERNMENT],
                        rationale="Growth in AI-augmented drug discovery requires hybrid skillsets",
                        evidence_base=[
                            "Job market projections",
                            "Skills gap analysis",
                            "Industry hiring surveys",
                        ],
                        expected_impact="Double pipeline of AI-biology graduates",
                        impact_timeline="5+ years",
                        implementation_steps=[
                            "Fund new interdisciplinary programs",
                            "Create industry fellowship programs",
                            "Develop online modules for working professionals",
                            "Establish competency standards",
                        ],
                        estimated_cost="medium",
                        barriers=["Faculty hiring", "Curriculum development time"],
                        success_metrics=["Graduate enrollment", "Employment outcomes", "Skills assessments"],
                        confidence=0.75,
                        key_assumptions=["Demand continues as projected"],
                    ),
                },
            ],

            "materials_science": [
                {
                    "trigger": "synthesis_bottleneck",
                    "threshold": 0.7,
                    "template": PolicyRecommendation(
                        id="MS-001",
                        title="Materials Synthesis Automation Network",
                        description="Create national network of automated synthesis facilities",
                        domain=PolicyDomain.INFRASTRUCTURE,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.ACADEMIA],
                        rationale="AI predicts 100x more materials than can be synthesized. Automation essential.",
                        evidence_base=[
                            "GNoME discovery rate",
                            "Synthesis throughput data",
                            "Backlog growth projections",
                        ],
                        expected_impact="10x synthesis throughput",
                        impact_timeline="5+ years",
                        implementation_steps=[
                            "Identify hub locations",
                            "Develop shared access model",
                            "Fund facility construction",
                            "Create training programs",
                        ],
                        estimated_cost="very_high",
                        barriers=["Capital costs", "Regional competition", "Coordination"],
                        success_metrics=["Synthesis throughput", "User satisfaction", "Discovery-to-publication time"],
                        confidence=0.55,
                        key_assumptions=["Coordination across institutions achievable"],
                    ),
                },
                {
                    "trigger": "high_acceleration",
                    "threshold": 2.0,
                    "template": PolicyRecommendation(
                        id="MS-002",
                        title="AI Materials Triage Standards",
                        description="Develop standards for AI-based prioritization of candidate materials",
                        domain=PolicyDomain.REGULATORY,
                        priority=PolicyPriority.MEDIUM,
                        target_stakeholders=[StakeholderType.ACADEMIA, StakeholderType.INDUSTRY],
                        rationale="Efficient triage essential to manage prediction backlog",
                        evidence_base=[
                            "Triage efficiency modeling",
                            "Industry best practices",
                            "Publication standards analysis",
                        ],
                        expected_impact="Standardized, reproducible prioritization",
                        impact_timeline="1-2 years",
                        implementation_steps=[
                            "Convene standards working group",
                            "Develop initial framework",
                            "Pilot with major labs",
                            "Iterate and publish",
                        ],
                        estimated_cost="low",
                        barriers=["Community buy-in", "IP concerns"],
                        success_metrics=["Adoption rate", "Reproducibility", "Discovery efficiency"],
                        confidence=0.70,
                        key_assumptions=["Community willing to standardize"],
                    ),
                },
            ],

            "protein_design": [
                {
                    "trigger": "high_acceleration",
                    "threshold": 4.0,
                    "template": PolicyRecommendation(
                        id="PD-001",
                        title="Protein Expression Foundries",
                        description="Fund shared protein expression facilities for designed proteins",
                        domain=PolicyDomain.INFRASTRUCTURE,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.ACADEMIA],
                        rationale="Design outpaces validation. Shared facilities enable broader access.",
                        evidence_base=[
                            "Design vs expression throughput",
                            "Facility utilization data",
                            "Community surveys",
                        ],
                        expected_impact="Democratize access to protein validation",
                        impact_timeline="3-5 years",
                        implementation_steps=[
                            "Assess existing facility landscape",
                            "Design shared access model",
                            "Fund regional hubs",
                            "Develop training and protocols",
                        ],
                        estimated_cost="high",
                        barriers=["Regional politics", "Operating costs"],
                        success_metrics=["Expression throughput", "User diversity", "Cost per protein"],
                        confidence=0.65,
                        key_assumptions=["Demand continues growing"],
                    ),
                },
                {
                    "trigger": "ethics_concern",
                    "threshold": 0.5,
                    "template": PolicyRecommendation(
                        id="PD-002",
                        title="Protein Design Ethics Framework",
                        description="Develop ethical guidelines for de novo protein design",
                        domain=PolicyDomain.ETHICS,
                        priority=PolicyPriority.MEDIUM,
                        target_stakeholders=[StakeholderType.ACADEMIA, StakeholderType.GOVERNMENT, StakeholderType.INTERNATIONAL],
                        rationale="Powerful design capabilities require ethical guardrails",
                        evidence_base=[
                            "Biosecurity analyses",
                            "Dual-use research precedents",
                            "International governance examples",
                        ],
                        expected_impact="Responsible innovation framework",
                        impact_timeline="1-2 years",
                        implementation_steps=[
                            "Convene ethics working group",
                            "Review existing frameworks",
                            "Draft guidelines",
                            "Community consultation",
                            "Publish and implement",
                        ],
                        estimated_cost="low",
                        barriers=["International coordination", "Enforcement"],
                        success_metrics=["Adoption rate", "Incident prevention"],
                        confidence=0.60,
                        key_assumptions=["Community engagement achievable"],
                    ),
                },
            ],

            "clinical_genomics": [
                {
                    "trigger": "clinical_adoption",
                    "threshold": 0.5,
                    "template": PolicyRecommendation(
                        id="CG-001",
                        title="AI Variant Classification Standards",
                        description="FDA guidance for AI-based variant classification in clinical settings",
                        domain=PolicyDomain.REGULATORY,
                        priority=PolicyPriority.CRITICAL,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.INDUSTRY],
                        rationale="Clinical adoption requires regulatory clarity",
                        evidence_base=[
                            "AlphaMissense accuracy data",
                            "Current VUS classification challenges",
                            "FDA AI/ML guidance precedents",
                        ],
                        expected_impact="Clear pathway for clinical AI tools",
                        impact_timeline="1-2 years",
                        implementation_steps=[
                            "FDA stakeholder meeting",
                            "Draft guidance document",
                            "Comment period",
                            "Final guidance",
                        ],
                        estimated_cost="low",
                        barriers=["Regulatory complexity", "Liability concerns"],
                        success_metrics=["Guidance publication", "Tool approvals", "Clinical adoption rate"],
                        confidence=0.70,
                        key_assumptions=["FDA prioritizes this area"],
                    ),
                },
                {
                    "trigger": "workforce_growth",
                    "threshold": 0.3,
                    "template": PolicyRecommendation(
                        id="CG-002",
                        title="AI Genetic Counselor Certification",
                        description="Certification for genetic counselors in AI tool usage",
                        domain=PolicyDomain.WORKFORCE_DEVELOPMENT,
                        priority=PolicyPriority.MEDIUM,
                        target_stakeholders=[StakeholderType.ACADEMIA, StakeholderType.NONPROFIT],
                        rationale="AI augments counselor capabilities but requires training",
                        evidence_base=[
                            "Genetic counselor workforce projections",
                            "AI tool proliferation",
                            "Training needs assessments",
                        ],
                        expected_impact="Competency assurance for AI-augmented counseling",
                        impact_timeline="1-2 years",
                        implementation_steps=[
                            "Partner with NSGC/ABGC",
                            "Develop curriculum",
                            "Pilot certification exam",
                            "Full implementation",
                        ],
                        estimated_cost="low",
                        barriers=["Professional society coordination"],
                        success_metrics=["Certification rate", "Competency scores"],
                        confidence=0.75,
                        key_assumptions=["Professional societies engaged"],
                    ),
                },
            ],

            "cross_cutting": [
                {
                    "trigger": "international_competition",
                    "threshold": 0.6,
                    "template": PolicyRecommendation(
                        id="CC-001",
                        title="AI Biology Competitiveness Initiative",
                        description="National initiative to maintain leadership in AI-accelerated biology",
                        domain=PolicyDomain.INTERNATIONAL,
                        priority=PolicyPriority.CRITICAL,
                        target_stakeholders=[StakeholderType.GOVERNMENT],
                        rationale="International competition for AI biology leadership intensifying",
                        evidence_base=[
                            "International R&D investment comparison",
                            "Publication and patent trends",
                            "Talent flow analysis",
                        ],
                        expected_impact="Maintain/extend leadership position",
                        impact_timeline="5+ years",
                        implementation_steps=[
                            "Comprehensive competitiveness assessment",
                            "Develop national strategy",
                            "Coordinate agency investments",
                            "Track progress annually",
                        ],
                        estimated_cost="very_high",
                        barriers=["Political will", "Coordination across agencies"],
                        success_metrics=["Global ranking", "Talent retention", "Commercial impact"],
                        confidence=0.50,
                        key_assumptions=["Political support achievable"],
                    ),
                },
                {
                    "trigger": "data_sharing",
                    "threshold": 0.4,
                    "template": PolicyRecommendation(
                        id="CC-002",
                        title="Open Science Data Infrastructure",
                        description="Expand infrastructure for sharing AI training data in biology",
                        domain=PolicyDomain.INFRASTRUCTURE,
                        priority=PolicyPriority.HIGH,
                        target_stakeholders=[StakeholderType.GOVERNMENT, StakeholderType.ACADEMIA],
                        rationale="AI acceleration depends on data availability",
                        evidence_base=[
                            "Data access barriers survey",
                            "Open data success stories",
                            "International data sharing agreements",
                        ],
                        expected_impact="Accelerate AI model development",
                        impact_timeline="3-5 years",
                        implementation_steps=[
                            "Audit existing repositories",
                            "Identify gaps and needs",
                            "Fund infrastructure expansion",
                            "Develop sharing standards",
                        ],
                        estimated_cost="high",
                        barriers=["IP concerns", "Privacy", "Sustainability"],
                        success_metrics=["Data deposits", "Usage metrics", "Model improvements"],
                        confidence=0.65,
                        key_assumptions=["Community willing to share"],
                    ),
                },
            ],
        }

    def analyze_domain(
        self,
        domain: str,
        year: int,
        acceleration: float,
        bottleneck_fraction: float,
        workforce_displacement: float,
        workforce_growth: float,
    ) -> DomainPolicyAnalysis:
        """Generate policy analysis for a domain."""
        recommendations = []

        templates = self.recommendation_templates.get(domain, [])
        templates.extend(self.recommendation_templates.get("cross_cutting", []))

        for template_info in templates:
            trigger = template_info["trigger"]
            threshold = template_info["threshold"]
            template = template_info["template"]

            triggered = False
            if trigger == "high_acceleration" and acceleration >= threshold:
                triggered = True
            elif trigger == "workforce_displacement" and workforce_displacement >= threshold:
                triggered = True
            elif trigger == "workforce_growth" and workforce_growth >= threshold:
                triggered = True
            elif trigger == "regulatory_bottleneck" and bottleneck_fraction >= threshold:
                triggered = True
            elif trigger == "synthesis_bottleneck" and bottleneck_fraction >= threshold:
                triggered = True
            elif trigger == "clinical_adoption" and acceleration >= 2.0:
                triggered = True
            elif trigger == "ethics_concern" and acceleration >= 3.0:
                triggered = True
            elif trigger == "international_competition":
                triggered = True  # Always relevant
            elif trigger == "data_sharing":
                triggered = True  # Always relevant

            if triggered:
                recommendations.append(template)

        # Sort by priority
        priority_order = {
            PolicyPriority.CRITICAL: 0,
            PolicyPriority.HIGH: 1,
            PolicyPriority.MEDIUM: 2,
            PolicyPriority.LOW: 3,
        }
        recommendations.sort(key=lambda r: priority_order[r.priority])

        # Calculate summary metrics
        cost_values = {"low": 1, "medium": 5, "high": 20, "very_high": 100}
        total_cost = sum(cost_values.get(r.estimated_cost, 5) for r in recommendations)

        if total_cost < 10:
            investment_str = "Low (<$50M)"
        elif total_cost < 50:
            investment_str = "Medium ($50M-$200M)"
        elif total_cost < 100:
            investment_str = "High ($200M-$500M)"
        else:
            investment_str = "Very High (>$500M)"

        # Workforce support based on displacement
        if workforce_displacement > 0.3:
            workforce_str = "Major transition support needed"
        elif workforce_displacement > 0.1:
            workforce_str = "Moderate retraining programs"
        else:
            workforce_str = "Standard career development"

        # Count regulatory recommendations
        regulatory_count = sum(1 for r in recommendations
                              if r.domain == PolicyDomain.REGULATORY)

        return DomainPolicyAnalysis(
            domain=domain,
            year=year,
            current_acceleration=acceleration * 0.5,  # Rough estimate
            projected_acceleration=acceleration,
            primary_bottleneck="clinical_trials" if domain == "drug_discovery" else "validation",
            recommendations=recommendations,
            total_investment_recommended=investment_str,
            workforce_transition_support=workforce_str,
            regulatory_changes_needed=regulatory_count,
        )

    def generate_comprehensive_report(
        self,
        domain_analyses: List[DomainPolicyAnalysis],
    ) -> str:
        """Generate comprehensive policy report across all domains."""
        lines = [
            "=" * 100,
            "POLICY RECOMMENDATIONS FOR AI-ACCELERATED BIOLOGY",
            "=" * 100,
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 100,
        ]

        # Count recommendations by priority
        all_recs = []
        for analysis in domain_analyses:
            all_recs.extend(analysis.recommendations)

        # Deduplicate by ID
        seen_ids = set()
        unique_recs = []
        for rec in all_recs:
            if rec.id not in seen_ids:
                seen_ids.add(rec.id)
                unique_recs.append(rec)

        critical = sum(1 for r in unique_recs if r.priority == PolicyPriority.CRITICAL)
        high = sum(1 for r in unique_recs if r.priority == PolicyPriority.HIGH)
        medium = sum(1 for r in unique_recs if r.priority == PolicyPriority.MEDIUM)

        lines.extend([
            f"Total recommendations: {len(unique_recs)}",
            f"  Critical (act within 1 year): {critical}",
            f"  High (act within 2 years): {high}",
            f"  Medium (act within 5 years): {medium}",
            "",
        ])

        # Domain summaries
        lines.extend([
            "DOMAIN SUMMARIES:",
            "-" * 100,
        ])

        for analysis in domain_analyses:
            lines.extend([
                f"\n{analysis.domain.upper()}:",
                f"  Current → Projected acceleration: {analysis.current_acceleration:.1f}x → {analysis.projected_acceleration:.1f}x",
                f"  Primary bottleneck: {analysis.primary_bottleneck}",
                f"  Investment recommended: {analysis.total_investment_recommended}",
                f"  Workforce support: {analysis.workforce_transition_support}",
                f"  Regulatory changes: {analysis.regulatory_changes_needed}",
            ])

        # Critical recommendations
        lines.extend([
            "",
            "-" * 100,
            "CRITICAL RECOMMENDATIONS (Act within 1 year):",
            "-" * 100,
        ])

        for rec in unique_recs:
            if rec.priority == PolicyPriority.CRITICAL:
                lines.extend([
                    f"\n[{rec.id}] {rec.title}",
                    f"  {rec.description}",
                    f"  Target: {', '.join(s.value for s in rec.target_stakeholders)}",
                    f"  Impact: {rec.expected_impact}",
                    f"  Confidence: {rec.confidence:.0%}",
                ])

        # High priority recommendations
        lines.extend([
            "",
            "-" * 100,
            "HIGH PRIORITY RECOMMENDATIONS (Act within 2 years):",
            "-" * 100,
        ])

        for rec in unique_recs:
            if rec.priority == PolicyPriority.HIGH:
                lines.extend([
                    f"\n[{rec.id}] {rec.title}",
                    f"  {rec.description}",
                    f"  Impact: {rec.expected_impact}",
                ])

        # Investment summary
        lines.extend([
            "",
            "-" * 100,
            "INVESTMENT SUMMARY:",
            "-" * 100,
            "",
            "By Domain:",
        ])

        by_domain = {}
        for rec in unique_recs:
            d = rec.domain.value
            by_domain[d] = by_domain.get(d, 0) + 1

        for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
            lines.append(f"  {domain}: {count} recommendations")

        lines.extend([
            "",
            "KEY INSIGHTS:",
            "  1. Clinical trial reform is the critical enabler for drug discovery",
            "  2. Infrastructure investments needed to close validation gaps",
            "  3. Workforce transition support essential to avoid displacement harm",
            "  4. International coordination needed for ethics and standards",
            "",
            "=" * 100,
        ])

        return "\n".join(lines)


def generate_sample_report():
    """Generate a sample policy report."""
    engine = PolicyRecommendationEngine()

    # Sample domain analyses
    domains_data = [
        ("structural_biology", 15.0, 0.3, 0.25, 0.1),
        ("drug_discovery", 3.5, 0.8, 0.15, 0.3),
        ("materials_science", 3.8, 0.7, 0.20, 0.2),
        ("protein_design", 6.6, 0.4, 0.15, 0.4),
        ("clinical_genomics", 5.6, 0.5, 0.10, 0.35),
    ]

    analyses = []
    for domain, accel, bottleneck, displacement, growth in domains_data:
        analysis = engine.analyze_domain(
            domain=domain,
            year=2030,
            acceleration=accel,
            bottleneck_fraction=bottleneck,
            workforce_displacement=displacement,
            workforce_growth=growth,
        )
        analyses.append(analysis)

    return engine.generate_comprehensive_report(analyses)


if __name__ == "__main__":
    print(generate_sample_report())
