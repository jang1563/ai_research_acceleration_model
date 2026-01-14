"""
GNoME Case Study (DeepMind, 2023)
=================================

Detailed case study of Graph Networks for Materials Exploration (GNoME)
for model validation.

GNoME represents a Type I (Scale) shift in materials science,
predicting 2.2 million new stable materials (800x expansion of known space).

Key validation questions:
1. Does our model capture the scale acceleration correctly?
2. Is experimental synthesis identified as the bottleneck?
3. What does GNoME tell us about Type I vs Type III shifts?
"""

from case_study_framework import (
    CaseStudy,
    CaseStudyMetrics,
    StageMetrics,
    ShiftType,
)


def create_gnome_case_study() -> CaseStudy:
    """
    Create the GNoME case study with comprehensive metrics.

    Data sources:
    - Merchant et al. (2023) Nature - GNoME paper
    - Materials Project database statistics
    - A-Lab autonomous synthesis paper
    """

    # Stage-level metrics
    stage_metrics = {
        "S1": StageMetrics(
            stage_name="Literature Review",
            stage_id="S1",
            time_before=60,              # Days to survey materials literature
            time_after=7,                # With ML-curated databases
            is_bottleneck=False,
        ),
        "S2": StageMetrics(
            stage_name="Hypothesis Generation",
            stage_id="S2",
            # Core GNoME impact - generating candidate materials
            time_before=30,              # Days to identify candidate materials
            time_after=0.001,            # Seconds with GNoME
            cost_before=10000,           # DFT calculations
            cost_after=1,                # ML inference
            is_bottleneck=False,
        ),
        "S3": StageMetrics(
            stage_name="Computational Analysis",
            stage_id="S3",
            # Stability prediction
            time_before=7,               # Days for DFT stability calculation per material
            time_after=0.0001,           # Milliseconds with GNoME
            quality_before=0.90,         # DFT accuracy
            quality_after=0.85,          # GNoME accuracy
            is_bottleneck=False,
        ),
        "S4": StageMetrics(
            stage_name="Experimental Synthesis",
            stage_id="S4",
            # KEY BOTTLENECK
            time_before=90,              # Days for traditional synthesis
            time_after=1,                # A-Lab can attempt 1 per day
            is_bottleneck=True,
            bottleneck_reason=(
                "2.2M predicted materials, but A-Lab can only synthesize "
                "~350/year (17 days each). Would take 6,000+ years to validate all."
            ),
        ),
        "S5": StageMetrics(
            stage_name="Property Characterization",
            stage_id="S5",
            time_before=30,              # Days for full characterization
            time_after=7,                # Automated characterization
            is_bottleneck=True,
            bottleneck_reason="Each synthesized material needs characterization",
        ),
        "S6": StageMetrics(
            stage_name="Validation & Publication",
            stage_id="S6",
            time_before=180,             # Peer review for novel materials
            time_after=90,               # Pre-validated by model
            is_bottleneck=False,
        ),
    }

    # Compute derived metrics
    for stage in stage_metrics.values():
        stage.compute_derived()

    # Overall metrics
    metrics = CaseStudyMetrics(
        # Time impact (for hypothesis generation stage)
        total_time_before=365,           # ~1 year to identify promising materials
        total_time_after=1,              # ~1 day with GNoME
        overall_acceleration=365,        # 365x for candidate generation

        # Scale impact (primary metric for Type I shift)
        scale_before=28000,              # Known stable inorganic crystals (2023)
        scale_after=2200000,             # GNoME predictions
        scale_increase=78.6,             # ~80x expansion of material space

        # Quality impact
        quality_improvement=None,        # Focus is scale, not quality

        # Cost impact
        cost_reduction_factor=10000,     # DFT → ML inference

        # Stage breakdown
        stage_metrics=stage_metrics,

        # Bottleneck analysis
        primary_bottleneck="S4",         # Experimental synthesis
        secondary_bottlenecks=["S5"],    # Property characterization

        # Data sources
        sources=[
            "Merchant et al. (2023) Nature",
            "Materials Project database",
            "A-Lab autonomous synthesis (Szymanski et al. 2023)",
            "ICSD database statistics",
        ],
        data_quality="high",
    )

    metrics.compute_derived()

    return CaseStudy(
        name="GNoME",
        domain="Materials Science",
        organization="DeepMind",
        year=2023,

        shift_type=ShiftType.TYPE_I,     # Scale acceleration
        affected_stages=["S2", "S3"],

        metrics=metrics,

        description=(
            "GNoME predicted 2.2 million new stable materials, expanding the "
            "known crystal structure space by ~80x. This represents hypothesis "
            "generation at unprecedented scale, but validation remains bottlenecked "
            "by experimental synthesis."
        ),

        key_insight=(
            "Perfect example of Type I (Scale) shift: AI generates hypotheses "
            "at massive scale (2.2M candidates), but physical validation "
            "(synthesis) becomes the binding constraint. A-Lab can only "
            "validate ~350/year → 6,000 years to test all predictions."
        ),

        problem_solved="Materials discovery and stability prediction at scale",
        problem_duration_years=100,      # Since X-ray crystallography

        limitations=[
            "Only predicts stability, not synthesizability",
            "Does not predict synthesis routes",
            "Property predictions (conductivity, etc.) separate",
            "Many predictions may be metastable, not truly stable",
        ],

        remaining_bottlenecks=[
            "Experimental synthesis (6,000 year backlog)",
            "Property characterization",
            "Finding synthesis routes",
            "Application-specific testing",
        ],

        primary_paper="Merchant, A., et al. (2023). Nature, 624, 80-85.",
        additional_refs=[
            "Szymanski, N.J., et al. (2023). Nature - A-Lab",
            "Jain, A., et al. (2013). APL Materials - Materials Project",
        ],
    )


# Pre-built case study for easy import
GNoMECaseStudy = create_gnome_case_study()


# Additional analysis functions

def gnome_synthesis_bottleneck_analysis() -> dict:
    """
    Detailed analysis of the synthesis bottleneck.

    This is a critical validation for our model's prediction that
    physical stages (S4) limit overall acceleration.
    """
    return {
        "gnome_predictions": 2200000,
        "a_lab_synthesis_rate": {
            "materials_per_day": 1,
            "success_rate": 0.71,        # 71% success rate
            "materials_per_year": 350,
        },
        "validation_backlog": {
            "years_to_validate_all": 6286,  # 2.2M / 350
            "practical_subset": 0.01,       # Maybe 1% worth pursuing
            "years_for_subset": 63,
        },
        "bottleneck_implications": [
            "Synthesis, not prediction, is the rate-limiting step",
            "Need autonomous synthesis scaling (more A-Labs)",
            "Need better triage to identify most promising candidates",
            "Physical constraints cannot be bypassed by AI alone",
        ],
        "model_validation": (
            "Our model predicts S4 (wet lab) has M_max ~ 2.5x acceleration ceiling. "
            "GNoME shows S2/S3 achieved ~100,000x+ but S4 remains ~1x "
            "(one material synthesized per day, same as before). "
            "Overall pipeline limited by S4, exactly as model predicts."
        ),
    }


def compare_gnome_to_alphafold() -> dict:
    """
    Compare GNoME (Type I) to AlphaFold (Type III) shifts.

    Key insight: Different shift types have different bottleneck patterns.
    """
    return {
        "alphafold": {
            "shift_type": "Type III (Capability)",
            "primary_impact": "Do what was impossible (fold proteins)",
            "acceleration_stage": "S3 (Analysis/Prediction)",
            "stage_acceleration": 36500,  # X-ray → AlphaFold
            "end_to_end_acceleration": 24,
            "bottleneck": "S6 (Validation) - community adoption",
        },
        "gnome": {
            "shift_type": "Type I (Scale)",
            "primary_impact": "Do much more of same thing (predict stability)",
            "acceleration_stage": "S2/S3 (Hypothesis/Analysis)",
            "stage_acceleration": 100000,  # DFT → GNoME
            "end_to_end_acceleration": 1,  # Still 1 synthesis/day!
            "bottleneck": "S4 (Synthesis) - physical process",
        },
        "key_insight": (
            "Both achieve massive acceleration in computational stages "
            "(10,000-100,000x), but end-to-end impact differs dramatically "
            "based on where the physical bottleneck lies. "
            "AlphaFold's bottleneck (validation) is partially social → 24x. "
            "GNoME's bottleneck (synthesis) is fully physical → ~1x for each material."
        ),
        "model_implication": (
            "Our model should differentiate between bottleneck types: "
            "Social bottlenecks (S6) may compress over time. "
            "Physical bottlenecks (S4) require infrastructure investment."
        ),
    }


def gnome_triage_requirements() -> dict:
    """
    Document the triage problem created by scale acceleration.

    When you generate 2.2M candidates but can only test 350/year,
    prioritization becomes critical.
    """
    return {
        "problem": (
            "GNoME created a new problem: how to select the best ~1,000 "
            "materials from 2.2M candidates for experimental validation?"
        ),
        "triage_criteria": [
            "Predicted stability confidence",
            "Synthesizability estimate",
            "Property predictions (conductivity, hardness, etc.)",
            "Application relevance",
            "Novelty relative to known materials",
        ],
        "triage_tools_needed": [
            "ML models for synthesizability",
            "Property prediction models",
            "Application-specific screening",
            "Expert curation",
        ],
        "meta_insight": (
            "AI at scale creates new bottlenecks: selection/triage of candidates. "
            "This is a second-order effect our model doesn't yet capture. "
            "Consider adding 'triage overhead' to future versions."
        ),
    }
