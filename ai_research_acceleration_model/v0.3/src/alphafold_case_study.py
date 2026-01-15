"""
AlphaFold Case Study (2021-2024)
================================

Detailed case study of AlphaFold 2 and AlphaFold 3 for model validation.

AlphaFold represents a Type III (Capability Extension) shift in structural biology,
solving the 50-year protein folding problem.

Key validation questions:
1. Does our model predict the ~1000x acceleration in structure prediction?
2. Does it correctly identify validation as the remaining bottleneck?
3. What does AlphaFold tell us about AI ceiling effects?
"""

from case_study_framework import (
    CaseStudy,
    CaseStudyMetrics,
    StageMetrics,
    ShiftType,
)


def create_alphafold_case_study() -> CaseStudy:
    """
    Create the AlphaFold case study with comprehensive metrics.

    Data sources:
    - Jumper et al. (2021) Nature - AlphaFold 2 paper
    - Abramson et al. (2024) Nature - AlphaFold 3 paper
    - RCSB PDB statistics
    - Community benchmarks and surveys
    """

    # Stage-level metrics
    stage_metrics = {
        "S1": StageMetrics(
            stage_name="Literature Review",
            stage_id="S1",
            time_before=30,              # Days to review structural biology literature
            time_after=2,                # With AlphaFold DB + ChatGPT
            is_bottleneck=False,
        ),
        "S2": StageMetrics(
            stage_name="Hypothesis Generation",
            stage_id="S2",
            time_before=14,              # Days to identify target structures
            time_after=1,                # AlphaFold predicts structures directly
            is_bottleneck=False,
        ),
        "S3": StageMetrics(
            stage_name="Structure Prediction",
            stage_id="S3",
            # Core AlphaFold impact
            time_before=365,             # ~1 year for X-ray crystallography
            time_after=0.01,             # Minutes with AlphaFold
            quality_before=0.95,         # X-ray crystallography accuracy
            quality_after=0.92,          # AlphaFold 2 median accuracy
            cost_before=100000,          # $100K typical crystallography
            cost_after=10,               # Compute costs
            is_bottleneck=False,
        ),
        "S4": StageMetrics(
            stage_name="Wet Lab Validation",
            stage_id="S4",
            time_before=180,             # 6 months experimental validation
            time_after=120,              # Still need some validation
            is_bottleneck=True,
            bottleneck_reason="Physical experiments still required for high-confidence applications",
        ),
        "S5": StageMetrics(
            stage_name="Results Interpretation",
            stage_id="S5",
            time_before=30,
            time_after=7,                # Faster with confident structures
            is_bottleneck=False,
        ),
        "S6": StageMetrics(
            stage_name="Validation & Publication",
            stage_id="S6",
            time_before=90,              # Peer review + replication
            time_after=60,               # Community still skeptical initially
            is_bottleneck=True,
            bottleneck_reason="Scientific community validation and adoption",
        ),
    }

    # Compute derived metrics
    for stage in stage_metrics.values():
        stage.compute_derived()

    # Overall metrics
    metrics = CaseStudyMetrics(
        # Time impact
        total_time_before=365 * 2,       # ~2 years for structure-based research project
        total_time_after=30,             # ~1 month with AlphaFold
        overall_acceleration=24.3,       # ~24x end-to-end (bottleneck limited)

        # Quality impact
        quality_improvement=0.92,        # 92% GDT-TS on CASP14

        # Scale impact
        scale_before=15000,              # Structures solved per year (PDB growth rate)
        scale_after=200000000,           # 200M+ structures in AlphaFold DB
        scale_increase=13333,            # ~13,000x scale increase

        # Cost impact
        cost_reduction_factor=10000,     # $100K → $10

        # Stage breakdown
        stage_metrics=stage_metrics,

        # Bottleneck analysis
        primary_bottleneck="S4",         # Wet lab validation
        secondary_bottlenecks=["S6"],    # Community validation

        # Data sources
        sources=[
            "Jumper et al. (2021) Nature",
            "Abramson et al. (2024) Nature",
            "RCSB PDB Annual Reports",
            "AlphaFold Database Statistics",
            "CASP14/15 Assessment Reports",
        ],
        data_quality="high",
    )

    metrics.compute_derived()

    return CaseStudy(
        name="AlphaFold 2/3",
        domain="Structural Biology",
        organization="DeepMind / Isomorphic Labs",
        year=2021,

        shift_type=ShiftType.TYPE_III,   # Capability extension
        affected_stages=["S2", "S3", "S5"],

        metrics=metrics,

        description=(
            "AlphaFold 2 solved the 50-year protein folding problem, achieving "
            "experimental-quality structure predictions in minutes instead of years. "
            "AlphaFold 3 extended this to protein complexes and other biomolecules."
        ),

        key_insight=(
            "Massive acceleration in the prediction stage (~36,500x for S3 alone), "
            "but end-to-end acceleration bottlenecked by wet lab validation (~24x). "
            "This validates our model's prediction that physical stages limit overall gains."
        ),

        problem_solved="Protein structure prediction from sequence",
        problem_duration_years=50,       # Since Anfinsen's dogma (1973)

        limitations=[
            "Does not predict protein dynamics",
            "Confidence varies by protein family",
            "Complex assemblies less accurate (pre-AF3)",
            "Does not replace all experimental structural biology",
        ],

        remaining_bottlenecks=[
            "Experimental validation for drug discovery",
            "Functional annotation still needed",
            "Dynamics and conformational changes",
            "Post-translational modifications",
        ],

        primary_paper="Jumper, J., et al. (2021). Nature, 596, 583-589.",
        additional_refs=[
            "Abramson, J., et al. (2024). Nature - AlphaFold 3",
            "Varadi, M., et al. (2022). NAR - AlphaFold DB",
            "Tunyasuvunakool, K., et al. (2021). Nature - Proteome coverage",
        ],
    )


# Pre-built case study for easy import
AlphaFoldCaseStudy = create_alphafold_case_study()


# Additional metrics and analysis functions

def alphafold_impact_by_year() -> dict:
    """
    Track AlphaFold's impact metrics over time.

    Returns dict with year-by-year metrics.
    """
    return {
        2020: {
            "event": "AlphaFold 2 wins CASP14",
            "structures_predicted": 0,
            "citations": 0,
            "adoption": "announcement only",
        },
        2021: {
            "event": "AlphaFold 2 paper + code release",
            "structures_predicted": 350000,  # Initial DB
            "citations": 500,
            "adoption": "early adopters",
        },
        2022: {
            "event": "Full proteome coverage (200M+)",
            "structures_predicted": 200000000,
            "citations": 10000,
            "adoption": "widespread",
        },
        2023: {
            "event": "Integration into drug discovery pipelines",
            "structures_predicted": 200000000,
            "citations": 20000,
            "adoption": "standard tool",
        },
        2024: {
            "event": "AlphaFold 3 (complexes, modifications)",
            "structures_predicted": 200000000,  # + complexes
            "citations": 30000,
            "adoption": "essential infrastructure",
        },
    }


def compare_to_baseline() -> dict:
    """
    Compare AlphaFold acceleration to pre-AI computational methods.

    Key insight from H5 reviewer: AI should be compared to Rosetta baseline,
    not manual methods.
    """
    return {
        "manual_methods": {
            "method": "X-ray Crystallography / Cryo-EM",
            "time_per_structure_days": 365,
            "success_rate": 0.60,
            "cost_usd": 100000,
        },
        "rosetta_2005": {
            "method": "Rosetta ab initio",
            "time_per_structure_days": 30,
            "success_rate": 0.30,  # For small proteins
            "cost_usd": 1000,     # Compute costs
        },
        "alphafold_2021": {
            "method": "AlphaFold 2",
            "time_per_structure_days": 0.01,  # Minutes
            "success_rate": 0.92,
            "cost_usd": 10,
        },
        "acceleration_vs_manual": 36500,   # ~36,500x
        "acceleration_vs_rosetta": 3000,   # ~3,000x
        "key_insight": (
            "AlphaFold achieves ~3,000x over Rosetta (computational baseline), "
            "or ~36,500x over experimental methods. The H5 reviewer correctly "
            "notes that computational-to-computational comparison is more relevant "
            "for understanding AI's marginal contribution."
        ),
    }


def alphafold_validation_requirements() -> dict:
    """
    Document when AlphaFold predictions need experimental validation.

    This captures the remaining bottleneck.
    """
    return {
        "high_confidence_uses": {
            "description": "AlphaFold sufficient without validation",
            "examples": [
                "Initial hypothesis generation",
                "Homology model replacement",
                "Education and visualization",
                "Large-scale bioinformatics",
            ],
            "confidence_threshold": 70,  # pLDDT
        },
        "medium_confidence_uses": {
            "description": "AlphaFold guides experiments",
            "examples": [
                "Drug binding site identification",
                "Mutagenesis design",
                "Protein engineering",
            ],
            "validation_required": "targeted experiments",
        },
        "low_confidence_uses": {
            "description": "Experimental validation essential",
            "examples": [
                "Clinical drug development",
                "Novel fold discovery",
                "Disordered region analysis",
                "Protein dynamics",
            ],
            "validation_required": "full experimental characterization",
        },
        "bottleneck_implication": (
            "Drug discovery and clinical applications still require experimental "
            "validation, which takes months. This is why end-to-end acceleration "
            "is ~24x despite ~36,500x in the prediction stage alone."
        ),
    }
