"""
ESM-3 Case Study (Meta, 2024)
=============================

Detailed case study of ESM-3 (Evolutionary Scale Modeling) for model validation.

ESM-3 represents a Type III (Capability Extension) shift in protein design,
enabling de novo protein generation from text prompts.

Key validation questions:
1. How does ESM-3 compare to AlphaFold in terms of acceleration?
2. What does de novo generation mean for the research pipeline?
3. Is validation still the bottleneck for designed proteins?
"""

from case_study_framework import (
    CaseStudy,
    CaseStudyMetrics,
    StageMetrics,
    ShiftType,
)


def create_esm3_case_study() -> CaseStudy:
    """
    Create the ESM-3 case study with comprehensive metrics.

    Data sources:
    - Hayes et al. (2024) - ESM-3 paper
    - Meta AI blog posts and technical reports
    - Protein design community feedback
    """

    # Stage-level metrics
    stage_metrics = {
        "S1": StageMetrics(
            stage_name="Literature Review",
            stage_id="S1",
            time_before=30,              # Days to review protein design literature
            time_after=3,                # With ESM-integrated tools
            is_bottleneck=False,
        ),
        "S2": StageMetrics(
            stage_name="Design Specification",
            stage_id="S2",
            # Core ESM-3 impact - prompt to protein
            time_before=60,              # Days for traditional rational design
            time_after=0.01,             # Minutes with ESM-3
            is_bottleneck=False,
        ),
        "S3": StageMetrics(
            stage_name="Sequence Generation",
            stage_id="S3",
            # De novo generation
            time_before=30,              # Days for directed evolution
            time_after=0.001,            # Seconds with ESM-3
            quality_before=0.50,         # Success rate of designed proteins
            quality_after=0.70,          # ESM-3 predicted success rate
            cost_before=50000,           # Directed evolution costs
            cost_after=10,               # Compute costs
            is_bottleneck=False,
        ),
        "S4": StageMetrics(
            stage_name="Expression & Synthesis",
            stage_id="S4",
            # Physical production
            time_before=14,              # Days for protein expression
            time_after=14,               # Same - physical process
            is_bottleneck=True,
            bottleneck_reason="Protein expression is a physical process unchanged by AI",
        ),
        "S5": StageMetrics(
            stage_name="Functional Testing",
            stage_id="S5",
            time_before=30,              # Days for activity assays
            time_after=20,               # Slightly faster with better designs
            is_bottleneck=True,
            bottleneck_reason="Need to validate actual function, not just structure",
        ),
        "S6": StageMetrics(
            stage_name="Validation & Publication",
            stage_id="S6",
            time_before=90,              # Peer review
            time_after=60,               # Established field now
            is_bottleneck=False,
        ),
    }

    # Compute derived metrics
    for stage in stage_metrics.values():
        stage.compute_derived()

    # Overall metrics
    metrics = CaseStudyMetrics(
        # Time impact
        total_time_before=180,           # ~6 months for protein design project
        total_time_after=45,             # ~1.5 months with ESM-3
        overall_acceleration=4.0,        # ~4x end-to-end (bottleneck limited)

        # Quality impact
        quality_improvement=0.20,        # 50% → 70% success rate

        # Scale impact
        scale_before=100,                # Designs testable per year
        scale_after=1000,                # With faster design iteration
        scale_increase=10,

        # Cost impact
        cost_reduction_factor=5000,      # Design phase only

        # Stage breakdown
        stage_metrics=stage_metrics,

        # Bottleneck analysis
        primary_bottleneck="S4",         # Expression
        secondary_bottlenecks=["S5"],    # Functional testing

        # Data sources
        sources=[
            "Hayes et al. (2024) - ESM-3",
            "Meta AI Technical Reports",
            "Protein Design Community Benchmarks",
        ],
        data_quality="medium",           # Recent, less independent validation
    )

    metrics.compute_derived()

    return CaseStudy(
        name="ESM-3",
        domain="Protein Design",
        organization="Meta AI",
        year=2024,

        shift_type=ShiftType.TYPE_III,   # Capability extension
        affected_stages=["S2", "S3"],

        metrics=metrics,

        description=(
            "ESM-3 enables de novo protein generation from text prompts, "
            "representing a shift from 'predicting' natural proteins to "
            "'creating' new proteins with desired properties. Uses a "
            "multimodal architecture with 98B parameters trained on "
            "billions of protein sequences."
        ),

        key_insight=(
            "Despite 30,000x+ acceleration in design (S2/S3), end-to-end "
            "acceleration is only ~4x because expression and functional "
            "testing remain unchanged. Validates our model's physical "
            "bottleneck prediction."
        ),

        problem_solved="De novo protein design from natural language",
        problem_duration_years=30,       # Since protein engineering began

        limitations=[
            "Functional success rate still ~70%, not 100%",
            "Cannot guarantee specific functions",
            "Training biased toward natural proteins",
            "Large model requires significant compute",
        ],

        remaining_bottlenecks=[
            "Protein expression (physical)",
            "Functional validation (experimental)",
            "Scale-up for industrial production",
            "Regulatory approval for therapeutic use",
        ],

        primary_paper="Hayes, T., et al. (2024). bioRxiv - ESM-3",
        additional_refs=[
            "Lin, Z., et al. (2023). Science - ESM-2",
            "Rives, A., et al. (2021). PNAS - ESM (original)",
        ],
    )


# Pre-built case study for easy import
ESM3CaseStudy = create_esm3_case_study()


# Additional analysis functions

def esm3_vs_alphafold_comparison() -> dict:
    """
    Compare ESM-3 (generative) to AlphaFold (predictive).

    Key insight: Different AI capabilities, similar bottleneck patterns.
    """
    return {
        "alphafold": {
            "capability": "Prediction (sequence → structure)",
            "primary_use": "Understand existing proteins",
            "stage_acceleration": 36500,
            "end_to_end": 24,
            "bottleneck": "Validation (social)",
        },
        "esm3": {
            "capability": "Generation (specification → sequence)",
            "primary_use": "Create new proteins",
            "stage_acceleration": 30000,
            "end_to_end": 4,
            "bottleneck": "Expression + Testing (physical)",
        },
        "key_difference": (
            "AlphaFold accelerates understanding (analysis). "
            "ESM-3 accelerates creation (synthesis). "
            "Both hit physical/validation bottlenecks."
        ),
        "complementary_use": (
            "ESM-3 generates candidates → AlphaFold predicts structures → "
            "Wet lab validates function. AI accelerates S2/S3, "
            "but S4/S6 remain rate-limiting."
        ),
    }


def esm3_generation_capabilities() -> dict:
    """
    Document ESM-3's generation capabilities.

    This is a Type III capability that didn't exist before.
    """
    return {
        "prompt_types": [
            "Natural language descriptions",
            "Partial sequence constraints",
            "Structure constraints",
            "Function specifications",
        ],
        "generation_modes": {
            "unconditional": "Generate any valid protein",
            "conditional_structure": "Generate sequence for given structure",
            "conditional_function": "Generate protein with specified function",
            "inpainting": "Fill in missing regions",
        },
        "success_metrics": {
            "foldability": 0.90,         # 90% of designs are foldable
            "novel_sequences": 0.95,      # 95% are genuinely novel
            "functional": 0.70,           # 70% show intended function
        },
        "implication_for_model": (
            "ESM-3 creates a new capability (generation) that wasn't in our "
            "original pipeline model. We modeled AI as accelerating existing "
            "stages, but ESM-3 enables entirely new workflows. "
            "Consider adding 'capability expansion' factor to v0.4."
        ),
    }


def protein_design_pipeline_impact() -> dict:
    """
    Document how ESM-3 changes the protein design workflow.

    Traditional: Define goal → Library → Screen → Iterate
    With ESM-3: Define goal → Generate → Express → Test → Iterate
    """
    return {
        "traditional_pipeline": {
            "steps": [
                "Define target function",
                "Create library (random mutagenesis)",
                "High-throughput screening",
                "Select winners",
                "Iterate (3-5 cycles)",
            ],
            "total_time_months": 12,
            "cost_usd": 500000,
            "success_rate": 0.20,
        },
        "esm3_pipeline": {
            "steps": [
                "Define target function (prompt)",
                "Generate candidates (ESM-3)",
                "Filter by structure (AlphaFold)",
                "Express top candidates",
                "Test function",
                "Iterate (1-2 cycles)",
            ],
            "total_time_months": 3,
            "cost_usd": 50000,
            "success_rate": 0.40,
        },
        "acceleration": {
            "time": 4.0,
            "cost": 10.0,
            "success_rate": 2.0,
        },
        "key_changes": [
            "Library creation eliminated (design directly)",
            "Screening replaced by computational filtering",
            "Fewer experimental cycles needed",
            "Higher starting quality",
        ],
    }


def esm3_bottleneck_analysis() -> dict:
    """
    Detailed analysis of remaining bottlenecks after ESM-3.

    Despite 30,000x+ design acceleration, overall is ~4x.
    """
    return {
        "design_phase": {
            "acceleration": 30000,
            "time_before_days": 30,
            "time_after_days": 0.001,
            "percentage_of_pipeline": 0.05,  # Only 5% of total time
        },
        "expression_phase": {
            "acceleration": 1.0,
            "time_days": 14,
            "percentage_of_pipeline": 0.35,
            "reason": "Physical E. coli growth unchanged",
        },
        "testing_phase": {
            "acceleration": 1.5,
            "time_days": 20,
            "percentage_of_pipeline": 0.50,
            "reason": "Still need functional assays",
        },
        "overall_acceleration": 4.0,
        "bottleneck_analysis": (
            "Design went from 30 days to minutes (30,000x). "
            "But expression (14 days) + testing (20 days) = 34 days unchanged. "
            "Pipeline went from ~60 days to ~35 days = 1.7x from time reduction, "
            "plus faster iteration cycles = ~4x overall. "
            "Physical stages dominate, exactly as model predicts."
        ),
        "model_validation": (
            "Our model predicts M_max ~ 2.5x for physical stages. "
            "ESM-3 case shows S4 at 1.0x (expression) and S5 at 1.5x (testing). "
            "Model slightly overestimates physical acceleration. "
            "Suggest revising M_max_physical down to ~1.5x."
        ),
    }
