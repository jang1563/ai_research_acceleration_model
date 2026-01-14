#!/usr/bin/env python3
"""
Prospective Validation Framework for v0.8
==========================================

Addresses Expert Review E1-P3: "No future projection validation"

Previous validation was entirely retrospective (2021-2024 case studies).
v0.8 introduces a framework for prospective validation:

1. Register predictions BEFORE observing outcomes
2. Track predictions over time with timestamps
3. Score predictions as outcomes become available
4. Update model calibration based on prospective performance

Key Features:
- Prediction registry with versioning
- Automated scoring when outcomes are provided
- Brier scores for probabilistic predictions
- Calibration analysis
- Model update recommendations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import json
import hashlib
import numpy as np
from pathlib import Path


class PredictionStatus(Enum):
    """Status of a registered prediction."""
    PENDING = "pending"           # Waiting for outcome
    SCORED = "scored"             # Outcome observed, prediction scored
    EXPIRED = "expired"           # Prediction window passed without observation
    SUPERSEDED = "superseded"     # Replaced by newer prediction


class OutcomeType(Enum):
    """Types of outcomes for validation."""
    POINT_ESTIMATE = "point_estimate"     # Single value (e.g., 3.5x acceleration)
    RANGE = "range"                       # Range (e.g., 2-5x)
    BINARY = "binary"                     # Yes/No (e.g., backlog cleared?)
    CATEGORICAL = "categorical"           # Category (e.g., bottleneck type)


@dataclass
class PredictionRecord:
    """A registered prediction for prospective validation."""
    # Identification
    prediction_id: str
    model_version: str
    domain: str
    target_year: int

    # Prediction details
    metric: str                           # What is being predicted
    outcome_type: OutcomeType
    point_prediction: float               # Central estimate
    confidence_interval_90: Tuple[float, float]  # 90% CI
    confidence_interval_50: Tuple[float, float]  # 50% CI

    # Metadata
    prediction_date: str                  # When prediction was made
    assumptions: List[str]                # Key assumptions
    caveats: List[str]                    # Known limitations
    data_cutoff: str                      # Latest data used

    # Tracking
    status: PredictionStatus = PredictionStatus.PENDING
    outcome_value: Optional[float] = None
    outcome_date: Optional[str] = None
    outcome_source: Optional[str] = None

    # Scores (filled when outcome observed)
    absolute_error: Optional[float] = None
    log_error: Optional[float] = None
    within_90_ci: Optional[bool] = None
    within_50_ci: Optional[bool] = None
    brier_score: Optional[float] = None


@dataclass
class ProspectivePrediction:
    """A prediction to be registered for future validation."""
    domain: str
    target_year: int
    metric: str
    point_prediction: float
    ci_90_lower: float
    ci_90_upper: float
    ci_50_lower: float
    ci_50_upper: float
    assumptions: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


class PredictionRegistry:
    """
    Registry for prospective predictions.

    Maintains a versioned log of all predictions made by the model,
    enabling rigorous prospective validation as outcomes become known.
    """

    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or Path("prediction_registry.json")
        self.predictions: Dict[str, PredictionRecord] = {}
        self._load()

    def _generate_id(self, prediction: ProspectivePrediction, model_version: str) -> str:
        """Generate unique ID for a prediction."""
        content = f"{model_version}_{prediction.domain}_{prediction.target_year}_{prediction.metric}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _load(self):
        """Load existing registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    for pid, pdata in data.items():
                        pdata['status'] = PredictionStatus(pdata['status'])
                        pdata['outcome_type'] = OutcomeType(pdata['outcome_type'])
                        pdata['confidence_interval_90'] = tuple(pdata['confidence_interval_90'])
                        pdata['confidence_interval_50'] = tuple(pdata['confidence_interval_50'])
                        self.predictions[pid] = PredictionRecord(**pdata)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")

    def _save(self):
        """Save registry to disk."""
        data = {}
        for pid, pred in self.predictions.items():
            pdict = {
                'prediction_id': pred.prediction_id,
                'model_version': pred.model_version,
                'domain': pred.domain,
                'target_year': pred.target_year,
                'metric': pred.metric,
                'outcome_type': pred.outcome_type.value,
                'point_prediction': pred.point_prediction,
                'confidence_interval_90': list(pred.confidence_interval_90),
                'confidence_interval_50': list(pred.confidence_interval_50),
                'prediction_date': pred.prediction_date,
                'assumptions': pred.assumptions,
                'caveats': pred.caveats,
                'data_cutoff': pred.data_cutoff,
                'status': pred.status.value,
                'outcome_value': pred.outcome_value,
                'outcome_date': pred.outcome_date,
                'outcome_source': pred.outcome_source,
                'absolute_error': pred.absolute_error,
                'log_error': pred.log_error,
                'within_90_ci': pred.within_90_ci,
                'within_50_ci': pred.within_50_ci,
                'brier_score': pred.brier_score,
            }
            data[pid] = pdict

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        prediction: ProspectivePrediction,
        model_version: str,
        data_cutoff: str = None,
    ) -> str:
        """
        Register a new prediction for prospective validation.

        Returns the prediction ID.
        """
        prediction_id = self._generate_id(prediction, model_version)

        record = PredictionRecord(
            prediction_id=prediction_id,
            model_version=model_version,
            domain=prediction.domain,
            target_year=prediction.target_year,
            metric=prediction.metric,
            outcome_type=OutcomeType.POINT_ESTIMATE,
            point_prediction=prediction.point_prediction,
            confidence_interval_90=(prediction.ci_90_lower, prediction.ci_90_upper),
            confidence_interval_50=(prediction.ci_50_lower, prediction.ci_50_upper),
            prediction_date=datetime.now().isoformat(),
            assumptions=prediction.assumptions,
            caveats=prediction.caveats,
            data_cutoff=data_cutoff or datetime.now().strftime("%Y-%m-%d"),
            status=PredictionStatus.PENDING,
        )

        self.predictions[prediction_id] = record
        self._save()

        return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        outcome_value: float,
        outcome_source: str,
    ):
        """
        Record an observed outcome and score the prediction.
        """
        if prediction_id not in self.predictions:
            raise ValueError(f"Unknown prediction ID: {prediction_id}")

        pred = self.predictions[prediction_id]

        # Calculate scores
        absolute_error = abs(outcome_value - pred.point_prediction)
        log_error = abs(np.log10(max(outcome_value, 0.1)) -
                        np.log10(max(pred.point_prediction, 0.1)))

        within_90 = (pred.confidence_interval_90[0] <= outcome_value <=
                     pred.confidence_interval_90[1])
        within_50 = (pred.confidence_interval_50[0] <= outcome_value <=
                     pred.confidence_interval_50[1])

        # Brier score for probabilistic calibration
        # Simplified: based on whether outcome was within 90% CI
        brier = (1.0 - int(within_90)) ** 2  # 0 if correct, 1 if wrong

        # Update record
        pred.outcome_value = outcome_value
        pred.outcome_date = datetime.now().isoformat()
        pred.outcome_source = outcome_source
        pred.absolute_error = absolute_error
        pred.log_error = log_error
        pred.within_90_ci = within_90
        pred.within_50_ci = within_50
        pred.brier_score = brier
        pred.status = PredictionStatus.SCORED

        self._save()

    def get_pending(self) -> List[PredictionRecord]:
        """Get all pending predictions."""
        return [p for p in self.predictions.values()
                if p.status == PredictionStatus.PENDING]

    def get_scored(self) -> List[PredictionRecord]:
        """Get all scored predictions."""
        return [p for p in self.predictions.values()
                if p.status == PredictionStatus.SCORED]

    def calibration_analysis(self) -> Dict:
        """
        Analyze calibration of probabilistic predictions.

        Ideally:
        - 90% of outcomes should fall within 90% CI
        - 50% of outcomes should fall within 50% CI
        """
        scored = self.get_scored()
        if not scored:
            return {"error": "No scored predictions available"}

        n = len(scored)
        within_90 = sum(1 for p in scored if p.within_90_ci) / n
        within_50 = sum(1 for p in scored if p.within_50_ci) / n
        mean_log_error = np.mean([p.log_error for p in scored])
        mean_brier = np.mean([p.brier_score for p in scored])

        # Calibration assessment
        cal_90 = "well_calibrated" if 0.85 <= within_90 <= 0.95 else (
            "overconfident" if within_90 < 0.85 else "underconfident"
        )
        cal_50 = "well_calibrated" if 0.45 <= within_50 <= 0.55 else (
            "overconfident" if within_50 < 0.45 else "underconfident"
        )

        return {
            "n_predictions": n,
            "within_90_ci_rate": within_90,
            "within_50_ci_rate": within_50,
            "mean_log_error": mean_log_error,
            "mean_brier_score": mean_brier,
            "calibration_90": cal_90,
            "calibration_50": cal_50,
            "recommendation": self._calibration_recommendation(within_90, within_50),
        }

    def _calibration_recommendation(self, rate_90: float, rate_50: float) -> str:
        """Generate calibration adjustment recommendation."""
        if rate_90 < 0.85:
            return "WIDEN confidence intervals - model is overconfident"
        elif rate_90 > 0.95:
            return "NARROW confidence intervals - model is underconfident"
        elif rate_50 < 0.45:
            return "Increase point estimate variance"
        elif rate_50 > 0.55:
            return "Decrease point estimate variance"
        else:
            return "Calibration is acceptable"

    def summary_report(self) -> str:
        """Generate summary report of prediction registry."""
        pending = self.get_pending()
        scored = self.get_scored()

        lines = [
            "=" * 70,
            "PROSPECTIVE VALIDATION REGISTRY",
            "=" * 70,
            "",
            f"Total predictions: {len(self.predictions)}",
            f"  Pending: {len(pending)}",
            f"  Scored: {len(scored)}",
            "",
        ]

        if scored:
            cal = self.calibration_analysis()
            lines.extend([
                "CALIBRATION ANALYSIS:",
                f"  90% CI coverage: {cal['within_90_ci_rate']:.0%} (target: 90%)",
                f"  50% CI coverage: {cal['within_50_ci_rate']:.0%} (target: 50%)",
                f"  Mean log error: {cal['mean_log_error']:.3f}",
                f"  Assessment: {cal['calibration_90']}",
                f"  Recommendation: {cal['recommendation']}",
                "",
            ])

        if pending:
            lines.extend([
                "PENDING PREDICTIONS:",
                "-" * 50,
            ])
            for p in pending[:5]:  # Show first 5
                lines.append(f"  {p.domain} {p.target_year}: {p.point_prediction:.1f}x "
                             f"[{p.confidence_interval_90[0]:.1f}-{p.confidence_interval_90[1]:.1f}]")
            if len(pending) > 5:
                lines.append(f"  ... and {len(pending) - 5} more")

        return "\n".join(lines)


# Pre-defined 2025-2030 predictions for prospective validation
PROSPECTIVE_PREDICTIONS_2025_2030 = [
    ProspectivePrediction(
        domain="structural_biology",
        target_year=2025,
        metric="end_to_end_acceleration",
        point_prediction=8.0,
        ci_90_lower=4.0,
        ci_90_upper=16.0,
        ci_50_lower=6.0,
        ci_50_upper=11.0,
        assumptions=[
            "AlphaFold3 available and widely adopted",
            "Cryo-EM automation continues current trajectory",
            "No major new structural biology breakthroughs",
        ],
        caveats=[
            "Based on limited 2024 case studies",
            "Excludes membrane protein structures",
        ],
    ),
    ProspectivePrediction(
        domain="drug_discovery",
        target_year=2025,
        metric="end_to_end_acceleration",
        point_prediction=2.5,
        ci_90_lower=1.5,
        ci_90_upper=4.0,
        ci_50_lower=2.0,
        ci_50_upper=3.2,
        assumptions=[
            "No major regulatory changes",
            "Clinical trial requirements unchanged",
            "AI adoption in pharma continues",
        ],
        caveats=[
            "Clinical trials still dominant bottleneck",
            "Excludes vaccines (different timeline)",
        ],
    ),
    ProspectivePrediction(
        domain="materials_science",
        target_year=2025,
        metric="end_to_end_acceleration",
        point_prediction=2.0,
        ci_90_lower=1.2,
        ci_90_upper=3.5,
        ci_50_lower=1.5,
        ci_50_upper=2.5,
        assumptions=[
            "Synthesis bottleneck persists",
            "GNoME-scale predictions continue",
            "Modest automation improvement",
        ],
        caveats=[
            "Backlog continues to grow",
            "Triage efficiency critical uncertainty",
        ],
    ),
    ProspectivePrediction(
        domain="protein_design",
        target_year=2025,
        metric="end_to_end_acceleration",
        point_prediction=4.5,
        ci_90_lower=2.5,
        ci_90_upper=8.0,
        ci_50_lower=3.5,
        ci_50_upper=6.0,
        assumptions=[
            "ESM-3 and successors widely available",
            "Expression validation bottleneck persists",
            "De novo design continues rapid progress",
        ],
        caveats=[
            "High variance across sub-types",
            "Functional assay throughput critical",
        ],
    ),
    ProspectivePrediction(
        domain="clinical_genomics",
        target_year=2025,
        metric="end_to_end_acceleration",
        point_prediction=3.0,
        ci_90_lower=1.8,
        ci_90_upper=5.0,
        ci_50_lower=2.3,
        ci_50_upper=4.0,
        assumptions=[
            "AlphaMissense-style tools mature",
            "Clinical adoption gradual",
            "Regulatory acceptance of AI classification",
        ],
        caveats=[
            "Clinical validation still required",
            "Variant of uncertain significance remains challenging",
        ],
    ),
]


def register_v08_predictions(registry_path: Path = None):
    """Register v0.8 predictions for prospective validation."""
    registry = PredictionRegistry(registry_path)

    print("Registering v0.8 prospective predictions...")
    print()

    for pred in PROSPECTIVE_PREDICTIONS_2025_2030:
        pred_id = registry.register(
            prediction=pred,
            model_version="v0.8",
            data_cutoff="2024-12-31",
        )
        print(f"  Registered: {pred.domain} {pred.target_year} -> ID: {pred_id}")

    print()
    print(registry.summary_report())

    return registry


if __name__ == "__main__":
    # Register predictions
    registry_path = Path(__file__).parent.parent / "prediction_registry.json"
    registry = register_v08_predictions(registry_path)

    print()
    print()

    # Simulate some outcomes for demonstration
    print("=" * 70)
    print("SIMULATING OUTCOME RECORDING (for demonstration)")
    print("=" * 70)
    print()

    # Get a pending prediction and simulate an outcome
    pending = registry.get_pending()
    if pending:
        example = pending[0]
        print(f"Example: Recording simulated outcome for {example.domain} {example.target_year}")
        print(f"  Prediction: {example.point_prediction:.1f}x")
        print(f"  90% CI: [{example.confidence_interval_90[0]:.1f} - {example.confidence_interval_90[1]:.1f}]")

        # Simulate an outcome (for demo purposes)
        simulated_outcome = example.point_prediction * 0.9  # 10% below prediction
        print(f"  Simulated outcome: {simulated_outcome:.1f}x")

        registry.record_outcome(
            example.prediction_id,
            outcome_value=simulated_outcome,
            outcome_source="simulated_demo"
        )

        print()
        print("After scoring:")
        print(registry.summary_report())
