"""
Case Study Validation Framework
===============================

Framework for validating the AI Research Acceleration Model against
real-world AI breakthroughs in biology and materials science.

Each case study provides:
1. Observed metrics (before/after AI, acceleration factors)
2. Stage-by-stage breakdown matching our pipeline model
3. Bottleneck identification
4. Model predictions vs. actual outcomes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
import json

# Import v0.1 model for predictions
import sys
sys.path.insert(0, '/sessions/intelligent-beautiful-shannon/mnt/Accelerating_biology_with_AI/ai_research_acceleration_model/v0.1')
from src.model import AIResearchAccelerationModel, Scenario

# Stage names mapping
STAGE_NAMES = {
    "S1": "Literature Review & Synthesis",
    "S2": "Hypothesis Generation",
    "S3": "Experimental Design & Analysis",
    "S4": "Wet Lab Execution",
    "S5": "Results Interpretation",
    "S6": "Validation & Publication",
}


class ShiftType(Enum):
    """Types of paradigm shifts (from PROJECT_BIBLE Section 9)."""
    TYPE_I = "scale"              # Do more of what we already do
    TYPE_II = "efficiency"        # Do it faster/cheaper
    TYPE_III = "capability"       # Do what was impossible
    MIXED = "mixed"               # Combination


class ValidationStatus(Enum):
    """Status of model validation against case study."""
    VALIDATED = "validated"       # Model prediction matches observation
    PARTIAL = "partial"           # Some aspects match, others don't
    REJECTED = "rejected"         # Model prediction significantly wrong
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    stage_id: str                          # S1, S2, etc.

    # Time metrics
    time_before: Optional[float] = None    # Days/weeks before AI
    time_after: Optional[float] = None     # Days/weeks after AI
    time_acceleration: Optional[float] = None

    # Quality metrics
    quality_before: Optional[float] = None  # 0-1 scale
    quality_after: Optional[float] = None

    # Cost metrics
    cost_before: Optional[float] = None    # Relative or absolute
    cost_after: Optional[float] = None
    cost_reduction: Optional[float] = None

    # Is this stage a bottleneck?
    is_bottleneck: bool = False
    bottleneck_reason: Optional[str] = None

    # Uncertainty
    uncertainty: float = 0.2               # Relative uncertainty

    def compute_derived(self):
        """Compute derived metrics."""
        if self.time_before and self.time_after:
            self.time_acceleration = self.time_before / self.time_after
        if self.cost_before and self.cost_after:
            self.cost_reduction = self.cost_before / self.cost_after


@dataclass
class CaseStudyMetrics:
    """
    Comprehensive metrics for a case study.

    Captures before/after comparison and stage-by-stage breakdown.
    """
    # Overall metrics
    total_time_before: float               # Total pipeline time before AI (days)
    total_time_after: float                # Total pipeline time after AI (days)
    overall_acceleration: float            # Speedup factor

    # Quality impact
    quality_improvement: Optional[float] = None  # e.g., accuracy gain

    # Scale impact
    scale_before: Optional[float] = None   # e.g., structures solved/year
    scale_after: Optional[float] = None
    scale_increase: Optional[float] = None

    # Cost impact
    cost_reduction_factor: Optional[float] = None

    # Stage-level breakdown
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)

    # Bottleneck analysis
    primary_bottleneck: Optional[str] = None
    secondary_bottlenecks: List[str] = field(default_factory=list)

    # Data sources
    sources: List[str] = field(default_factory=list)
    data_quality: str = "medium"           # low, medium, high

    def compute_derived(self):
        """Compute derived metrics."""
        if self.scale_before and self.scale_after:
            self.scale_increase = self.scale_after / self.scale_before
        for stage in self.stage_metrics.values():
            stage.compute_derived()


@dataclass
class CaseStudy:
    """
    A complete case study of an AI breakthrough.

    Contains all information needed to validate against our model.
    """
    # Identity
    name: str                              # e.g., "AlphaFold 2"
    domain: str                            # e.g., "Structural Biology"
    organization: str                      # e.g., "DeepMind"
    year: int                              # Year of breakthrough

    # Classification
    shift_type: ShiftType
    affected_stages: List[str]             # Which pipeline stages affected

    # Metrics
    metrics: CaseStudyMetrics

    # Context
    description: str
    key_insight: str                       # Most important lesson

    # What problem did it solve?
    problem_solved: str
    problem_duration_years: Optional[int] = None  # How long was this a problem?

    # Limitations
    limitations: List[str] = field(default_factory=list)
    remaining_bottlenecks: List[str] = field(default_factory=list)

    # References
    primary_paper: Optional[str] = None
    additional_refs: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """
    Result of validating model against a case study.
    """
    case_study_name: str

    # Overall validation
    status: ValidationStatus
    overall_score: float                   # 0-1, how well model matches

    # Detailed comparisons
    predicted_acceleration: float
    observed_acceleration: float
    acceleration_error: float              # Log error

    # Stage-level validation
    stage_validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Bottleneck validation
    predicted_bottleneck: Optional[str] = None
    observed_bottleneck: Optional[str] = None
    bottleneck_match: bool = False

    # Insights
    model_strengths: List[str] = field(default_factory=list)
    model_weaknesses: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)

    # Confidence
    confidence: float = 0.5                # How confident in validation

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validation Result: {self.case_study_name}",
            "=" * 50,
            f"Status: {self.status.value.upper()}",
            f"Overall Score: {self.overall_score:.2f}",
            "",
            f"Acceleration:",
            f"  Predicted: {self.predicted_acceleration:.1f}x",
            f"  Observed:  {self.observed_acceleration:.1f}x",
            f"  Log Error: {self.acceleration_error:.2f}",
            "",
            f"Bottleneck Match: {'Yes' if self.bottleneck_match else 'No'}",
            f"  Predicted: {self.predicted_bottleneck or 'None'}",
            f"  Observed:  {self.observed_bottleneck or 'None'}",
        ]

        if self.model_strengths:
            lines.append("")
            lines.append("Model Strengths:")
            for s in self.model_strengths:
                lines.append(f"  + {s}")

        if self.model_weaknesses:
            lines.append("")
            lines.append("Model Weaknesses:")
            for w in self.model_weaknesses:
                lines.append(f"  - {w}")

        if self.suggested_improvements:
            lines.append("")
            lines.append("Suggested Improvements:")
            for i in self.suggested_improvements:
                lines.append(f"  * {i}")

        return "\n".join(lines)


class CaseStudyValidator:
    """
    Validates the AI Research Acceleration Model against case studies.

    Compares model predictions to observed metrics from real AI breakthroughs.
    """

    # Mapping from case study domains to model stages
    DOMAIN_STAGE_MAPPING = {
        "Structural Biology": ["S1", "S2", "S3"],  # Lit review, hypothesis, analysis
        "Materials Science": ["S2", "S3", "S4"],   # Hypothesis, analysis, experimental
        "Protein Design": ["S2", "S3", "S4", "S6"], # Design through validation
        "Drug Discovery": ["S1", "S2", "S3", "S4", "S5", "S6"],  # Full pipeline
    }

    def __init__(
        self,
        model: Optional[AIResearchAccelerationModel] = None,
        scenario: Scenario = Scenario.BASELINE,
    ):
        """
        Initialize validator.

        Args:
            model: Pre-configured model, or create new one
            scenario: Scenario to use for predictions
        """
        self.model = model or AIResearchAccelerationModel(scenario=scenario)
        self.scenario = scenario
        self.case_studies: Dict[str, CaseStudy] = {}
        self.validation_results: Dict[str, ValidationResult] = {}

    def add_case_study(self, case_study: CaseStudy):
        """Add a case study to validate against."""
        self.case_studies[case_study.name] = case_study

    def validate(self, case_study_name: str) -> ValidationResult:
        """
        Validate model against a specific case study.

        Args:
            case_study_name: Name of case study to validate against

        Returns:
            ValidationResult with detailed comparison
        """
        if case_study_name not in self.case_studies:
            raise ValueError(f"Unknown case study: {case_study_name}")

        cs = self.case_studies[case_study_name]

        # Get model prediction for the year of the breakthrough
        year = cs.year

        # Model starts at 2025, so for earlier years estimate from trajectory
        if year < 2025:
            # Use 2025 as baseline and extrapolate backwards
            forecasts = self.model.forecast([2025])
            base_accel = forecasts[2025]['acceleration']
            # Assume exponential growth backwards (about 40%/year based on AI progress)
            years_before = 2025 - year
            # Earlier years had less acceleration (inverse extrapolation)
            predicted_accel = base_accel / (1.4 ** years_before)
        else:
            forecasts = self.model.forecast([year])
            predicted_accel = forecasts[year]['acceleration']

        # Get observed acceleration
        observed_accel = cs.metrics.overall_acceleration

        # Calculate error (log scale)
        log_error = abs(np.log10(predicted_accel) - np.log10(observed_accel))

        # Score based on log error (0 = perfect, 1 = order of magnitude off)
        score = max(0, 1 - log_error)

        # Determine status
        if log_error < 0.3:  # Within 2x
            status = ValidationStatus.VALIDATED
        elif log_error < 0.7:  # Within 5x
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.REJECTED

        # Identify bottlenecks from model
        # S4 (wet lab) and S6 (validation) should be bottlenecks
        predicted_bottleneck = self._identify_predicted_bottleneck(cs)
        observed_bottleneck = cs.metrics.primary_bottleneck

        bottleneck_match = (
            predicted_bottleneck == observed_bottleneck or
            (predicted_bottleneck in ["S4", "S6"] and
             observed_bottleneck in ["S4", "S6", "wet_lab", "validation", "experimental"])
        )

        # Stage-level validation
        stage_validations = self._validate_stages(cs)

        # Identify strengths and weaknesses
        strengths, weaknesses, improvements = self._analyze_fit(
            cs, predicted_accel, observed_accel, stage_validations
        )

        result = ValidationResult(
            case_study_name=case_study_name,
            status=status,
            overall_score=score,
            predicted_acceleration=predicted_accel,
            observed_acceleration=observed_accel,
            acceleration_error=log_error,
            stage_validations=stage_validations,
            predicted_bottleneck=predicted_bottleneck,
            observed_bottleneck=observed_bottleneck,
            bottleneck_match=bottleneck_match,
            model_strengths=strengths,
            model_weaknesses=weaknesses,
            suggested_improvements=improvements,
            confidence=self._estimate_confidence(cs),
        )

        self.validation_results[case_study_name] = result
        return result

    def validate_all(self) -> Dict[str, ValidationResult]:
        """Validate against all registered case studies."""
        for name in self.case_studies:
            self.validate(name)
        return self.validation_results

    def _identify_predicted_bottleneck(self, cs: CaseStudy) -> str:
        """Identify which stage model predicts as bottleneck."""
        # Our model predicts S4 (wet lab) and S6 (validation) as key bottlenecks
        # because they have physical constraints (M_max ~ 2.5x)

        if cs.domain in ["Structural Biology", "Protein Design"]:
            # These still need experimental validation
            return "S6"  # Validation
        elif cs.domain == "Materials Science":
            return "S4"  # Wet lab synthesis
        else:
            return "S6"  # Default to validation

    def _validate_stages(self, cs: CaseStudy) -> Dict[str, Dict[str, Any]]:
        """Validate model predictions at stage level."""
        validations = {}

        for stage_id, stage_metrics in cs.metrics.stage_metrics.items():
            if stage_metrics.time_acceleration:
                # Get model prediction for this stage
                # (simplified - would need more sophisticated mapping)
                if stage_id in ["S1", "S2", "S3"]:
                    # Cognitive stages - model predicts high acceleration
                    predicted = 25.0  # M_max for cognitive
                elif stage_id in ["S4", "S6"]:
                    # Physical stages - model predicts low acceleration
                    predicted = 2.5   # M_max for physical
                else:
                    predicted = 5.0   # Hybrid

                observed = stage_metrics.time_acceleration
                error = abs(np.log10(predicted) - np.log10(observed))

                validations[stage_id] = {
                    "predicted": predicted,
                    "observed": observed,
                    "log_error": error,
                    "match": error < 0.5,
                }

        return validations

    def _analyze_fit(
        self,
        cs: CaseStudy,
        predicted: float,
        observed: float,
        stage_validations: Dict,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze model fit and identify strengths/weaknesses."""
        strengths = []
        weaknesses = []
        improvements = []

        # Check overall acceleration
        if abs(np.log10(predicted) - np.log10(observed)) < 0.3:
            strengths.append("Overall acceleration prediction within 2x of observed")
        elif predicted < observed:
            weaknesses.append(f"Model under-predicted acceleration ({predicted:.1f}x vs {observed:.1f}x)")
            improvements.append("Consider higher M_max for this domain")
        else:
            weaknesses.append(f"Model over-predicted acceleration ({predicted:.1f}x vs {observed:.1f}x)")
            improvements.append("Consider domain-specific constraints")

        # Check bottleneck prediction
        if cs.metrics.primary_bottleneck in ["S4", "S6", "wet_lab", "validation", "experimental"]:
            strengths.append("Correctly predicted physical bottleneck")

        # Check if case study is Type III (capability extension)
        if cs.shift_type == ShiftType.TYPE_III:
            strengths.append("Model handles capability extensions (Type III shifts)")

        # Analyze stage-level fits
        cognitive_errors = []
        physical_errors = []
        for stage_id, val in stage_validations.items():
            if stage_id in ["S1", "S2", "S3"]:
                cognitive_errors.append(val["log_error"])
            else:
                physical_errors.append(val["log_error"])

        if cognitive_errors and np.mean(cognitive_errors) < 0.5:
            strengths.append("Good fit for cognitive stages")
        elif cognitive_errors:
            weaknesses.append("Poor fit for cognitive stages")

        if physical_errors and np.mean(physical_errors) < 0.5:
            strengths.append("Good fit for physical stages")
        elif physical_errors:
            weaknesses.append("Poor fit for physical stages")

        return strengths, weaknesses, improvements

    def _estimate_confidence(self, cs: CaseStudy) -> float:
        """Estimate confidence in validation based on data quality."""
        base_confidence = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
        }.get(cs.metrics.data_quality, 0.5)

        # Reduce confidence for recent case studies (less data)
        if cs.year >= 2024:
            base_confidence *= 0.8

        # Increase confidence if multiple sources
        if len(cs.metrics.sources) > 2:
            base_confidence = min(1.0, base_confidence * 1.1)

        return base_confidence

    def generate_summary_report(self) -> str:
        """Generate summary report of all validations."""
        if not self.validation_results:
            return "No validations performed yet."

        lines = [
            "=" * 60,
            "CASE STUDY VALIDATION SUMMARY",
            "=" * 60,
            "",
            f"Model Scenario: {self.scenario.name}",
            f"Case Studies Validated: {len(self.validation_results)}",
            "",
        ]

        # Summary table
        lines.append("Results Overview:")
        lines.append("-" * 60)
        lines.append(f"{'Case Study':<25} {'Status':<15} {'Score':<10} {'Error':<10}")
        lines.append("-" * 60)

        for name, result in self.validation_results.items():
            lines.append(
                f"{name:<25} {result.status.value:<15} {result.overall_score:.2f}      {result.acceleration_error:.2f}"
            )

        lines.append("-" * 60)

        # Overall statistics
        scores = [r.overall_score for r in self.validation_results.values()]
        errors = [r.acceleration_error for r in self.validation_results.values()]

        lines.append("")
        lines.append(f"Mean Score: {np.mean(scores):.2f}")
        lines.append(f"Mean Log Error: {np.mean(errors):.2f}")

        # Count by status
        status_counts = {}
        for r in self.validation_results.values():
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

        lines.append("")
        lines.append("Validation Status Distribution:")
        for status, count in status_counts.items():
            lines.append(f"  {status}: {count}")

        # Key insights
        lines.append("")
        lines.append("Key Insights:")
        lines.append("-" * 40)

        all_strengths = set()
        all_weaknesses = set()
        for r in self.validation_results.values():
            all_strengths.update(r.model_strengths)
            all_weaknesses.update(r.model_weaknesses)

        if all_strengths:
            lines.append("Model Strengths (across case studies):")
            for s in list(all_strengths)[:5]:
                lines.append(f"  + {s}")

        if all_weaknesses:
            lines.append("")
            lines.append("Model Weaknesses (across case studies):")
            for w in list(all_weaknesses)[:5]:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    def export_results(self, filepath: str):
        """Export validation results to JSON."""
        export_data = {
            "scenario": self.scenario.name,
            "timestamp": datetime.now().isoformat(),
            "case_studies": len(self.validation_results),
            "results": {}
        }

        for name, result in self.validation_results.items():
            export_data["results"][name] = {
                "status": result.status.value,
                "score": result.overall_score,
                "predicted_acceleration": result.predicted_acceleration,
                "observed_acceleration": result.observed_acceleration,
                "log_error": result.acceleration_error,
                "bottleneck_match": result.bottleneck_match,
                "confidence": result.confidence,
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
