"""
Paradigm Shift Module (PSM)
===========================

Models capability extensions and methodological shifts enabled by AI,
distinguishing from conceptual paradigm shifts which remain unpredictable.

Based on PROJECT_BIBLE.md Section 5.

Key Distinction (per expert review):
- Capability Extensions: Expanding what can be done within existing frameworks
- Methodological Shifts: Changing how research is conducted
- Conceptual Paradigm Shifts: Changing fundamental assumptions (NOT modeled - unpredictable)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ShiftType(Enum):
    """
    Types of capability extensions and methodological shifts.

    Note: True conceptual paradigm shifts (e.g., germ theory) are not modeled
    as they are fundamentally unpredictable.
    """
    SCALE = "type_i"           # 10-1000x increase in throughput/scale
    ACCESSIBILITY = "type_ii"  # Democratization of specialized capabilities
    CAPABILITY = "type_iii"    # Novel capabilities previously impossible


@dataclass
class ShiftParameters:
    """
    Parameters for a specific type of capability extension.

    Attributes:
        weight: Base contribution to PSM (95% CI provided)
        weight_ci: 95% confidence interval for weight
        threshold: AI capability threshold for activation
        growth_rate: Rate of impact growth after threshold
        saturation: Maximum impact level
        stages_affected: Which pipeline stages are affected
    """
    weight: float                           # Base weight
    weight_ci: Tuple[float, float]          # 95% CI (lower, upper)
    threshold: float                        # A(t) threshold for activation
    growth_rate: float                      # Impact growth rate
    saturation: float                       # Maximum impact
    stages_affected: List[str]              # Stage IDs affected


# Default shift parameters based on PROJECT_BIBLE.md Section 5
DEFAULT_SHIFT_PARAMS = {
    ShiftType.SCALE: ShiftParameters(
        weight=0.40,
        weight_ci=(0.30, 0.50),
        threshold=1.0,
        growth_rate=0.3,
        saturation=10.0,
        stages_affected=["S1", "S5", "S8"],  # Literature, Analysis, Dissemination
    ),
    ShiftType.ACCESSIBILITY: ShiftParameters(
        weight=0.35,
        weight_ci=(0.25, 0.45),
        threshold=1.5,
        growth_rate=0.25,
        saturation=5.0,
        stages_affected=["S2", "S3", "S5", "S7"],  # Hypothesis, Design, Analysis, Writing
    ),
    ShiftType.CAPABILITY: ShiftParameters(
        weight=0.25,
        weight_ci=(0.15, 0.35),
        threshold=2.0,
        growth_rate=0.2,
        saturation=20.0,
        stages_affected=["S2", "S4", "S5", "S6"],  # Hypothesis, Wet Lab, Analysis, Validation
    ),
}


@dataclass
class HistoricalShift:
    """
    A historical capability extension or methodological shift for calibration.

    Used to calibrate PSM parameters against known historical impacts.
    """
    name: str
    year: int
    shift_type: ShiftType
    measured_impact: float          # Empirically measured acceleration
    impact_ci: Tuple[float, float]  # 95% CI for impact
    domain: str                     # Scientific domain
    notes: str = ""


# Historical shifts for calibration (from PROJECT_BIBLE.md Section 3)
HISTORICAL_SHIFTS = [
    HistoricalShift(
        name="Microscopy",
        year=1670,
        shift_type=ShiftType.CAPABILITY,
        measured_impact=5.0,
        impact_ci=(3.0, 8.0),
        domain="Biology",
        notes="Enabled observation of microorganisms, cells"
    ),
    HistoricalShift(
        name="Telescope",
        year=1609,
        shift_type=ShiftType.CAPABILITY,
        measured_impact=4.0,
        impact_ci=(2.5, 6.0),
        domain="Astronomy",
        notes="Enabled planetary observation, challenged geocentric model"
    ),
    HistoricalShift(
        name="PCR",
        year=1985,
        shift_type=ShiftType.SCALE,
        measured_impact=50.0,
        impact_ci=(30.0, 80.0),
        domain="Molecular Biology",
        notes="Enabled DNA amplification at massive scale"
    ),
    HistoricalShift(
        name="Human Genome Project",
        year=2003,
        shift_type=ShiftType.SCALE,
        measured_impact=100.0,
        impact_ci=(50.0, 200.0),
        domain="Genomics",
        notes="Reference genome enabled comparative genomics"
    ),
    HistoricalShift(
        name="Next-Gen Sequencing",
        year=2007,
        shift_type=ShiftType.SCALE,
        measured_impact=1000.0,
        impact_ci=(500.0, 2000.0),
        domain="Genomics",
        notes="Cost dropped from $100M to $1000 per genome"
    ),
    HistoricalShift(
        name="CRISPR-Cas9",
        year=2012,
        shift_type=ShiftType.ACCESSIBILITY,
        measured_impact=10.0,
        impact_ci=(5.0, 20.0),
        domain="Genetics",
        notes="Democratized gene editing to any lab"
    ),
    HistoricalShift(
        name="AlphaFold",
        year=2020,
        shift_type=ShiftType.CAPABILITY,
        measured_impact=50.0,
        impact_ci=(25.0, 100.0),
        domain="Structural Biology",
        notes="Solved 50-year protein folding problem"
    ),
]


class ParadigmShiftModule:
    """
    Models capability extensions and methodological shifts from AI.

    PSM(t) = 1 + Σ w_j × f_j(A(t))

    Where:
    - w_j = weight for shift type j (with uncertainty)
    - f_j(A(t)) = impact function based on AI capability

    Note: This module explicitly does NOT model true conceptual paradigm shifts
    (e.g., germ theory, central dogma) as these are fundamentally unpredictable.
    """

    def __init__(
        self,
        shift_params: Optional[Dict[ShiftType, ShiftParameters]] = None,
        historical_shifts: Optional[List[HistoricalShift]] = None
    ):
        """
        Initialize the Paradigm Shift Module.

        Args:
            shift_params: Parameters for each shift type. Uses defaults if not provided.
            historical_shifts: Historical data for calibration. Uses defaults if not provided.
        """
        self.shift_params = shift_params or DEFAULT_SHIFT_PARAMS.copy()
        self.historical_shifts = historical_shifts or HISTORICAL_SHIFTS.copy()

    def impact_function(
        self,
        shift_type: ShiftType,
        ai_capability: float,
        use_uncertainty: bool = False,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Calculate impact of a shift type given AI capability.

        f_j(A) = saturation × (1 - exp(-growth × (A - threshold))) for A > threshold
               = 0 for A <= threshold

        Args:
            shift_type: The type of shift
            ai_capability: Current AI capability level A(t)
            use_uncertainty: Whether to sample from uncertainty distributions
            rng: Random number generator for uncertainty sampling

        Returns:
            Impact value
        """
        params = self.shift_params[shift_type]

        if ai_capability <= params.threshold:
            return 0.0

        # Base impact calculation
        impact = params.saturation * (
            1 - np.exp(-params.growth_rate * (ai_capability - params.threshold))
        )

        return impact

    def calculate_psm(
        self,
        ai_capability: float,
        stage_id: Optional[str] = None,
        use_uncertainty: bool = False,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Calculate the Paradigm Shift Module value.

        PSM(t) = 1 + Σ w_j × f_j(A(t))

        If stage_id is provided, only includes shifts affecting that stage.

        Args:
            ai_capability: Current AI capability level
            stage_id: Optional stage to filter by
            use_uncertainty: Whether to sample from uncertainty distributions
            rng: Random number generator

        Returns:
            PSM multiplier (≥ 1.0)
        """
        total_impact = 1.0

        for shift_type, params in self.shift_params.items():
            # Skip if stage specified and not affected
            if stage_id and stage_id not in params.stages_affected:
                continue

            # Get weight (potentially with uncertainty)
            if use_uncertainty and rng is not None:
                # Sample from truncated normal within CI
                weight = rng.uniform(params.weight_ci[0], params.weight_ci[1])
            else:
                weight = params.weight

            # Calculate impact
            impact = self.impact_function(shift_type, ai_capability, use_uncertainty, rng)

            # Add weighted contribution
            total_impact += weight * impact

        return total_impact

    def stage_psm_breakdown(
        self,
        ai_capability: float,
        stage_id: str
    ) -> Dict[str, float]:
        """
        Get breakdown of PSM contributions for a specific stage.

        Args:
            ai_capability: Current AI capability level
            stage_id: Stage to analyze

        Returns:
            Dictionary with contribution from each shift type
        """
        breakdown = {'base': 1.0}

        for shift_type, params in self.shift_params.items():
            if stage_id not in params.stages_affected:
                breakdown[shift_type.value] = 0.0
                continue

            impact = self.impact_function(shift_type, ai_capability)
            contribution = params.weight * impact
            breakdown[shift_type.value] = contribution

        breakdown['total'] = sum(breakdown.values())
        return breakdown

    def validate_against_historical(self) -> Dict[str, Dict]:
        """
        Validate PSM predictions against historical data.

        Returns:
            Dictionary with validation results for each historical shift
        """
        results = {}

        for shift in self.historical_shifts:
            params = self.shift_params.get(shift.shift_type)
            if not params:
                continue

            # Estimate what A(t) would have been needed
            # For historical, we assume they represent saturation-level impacts
            predicted_saturation = params.saturation

            results[shift.name] = {
                'year': shift.year,
                'type': shift.shift_type.value,
                'measured_impact': shift.measured_impact,
                'measured_ci': shift.impact_ci,
                'predicted_saturation': predicted_saturation,
                'within_ci': (
                    shift.impact_ci[0] <= predicted_saturation <= shift.impact_ci[1]
                ),
            }

        return results

    def summary(self, ai_capability: float) -> Dict:
        """
        Generate a summary of PSM at given AI capability.

        Args:
            ai_capability: Current AI capability level

        Returns:
            Summary dictionary
        """
        return {
            'ai_capability': ai_capability,
            'total_psm': self.calculate_psm(ai_capability),
            'by_type': {
                st.value: {
                    'weight': p.weight,
                    'weight_ci': p.weight_ci,
                    'impact': self.impact_function(st, ai_capability),
                    'contribution': p.weight * self.impact_function(st, ai_capability),
                    'stages_affected': p.stages_affected,
                }
                for st, p in self.shift_params.items()
            },
        }


if __name__ == "__main__":
    # Quick test
    psm = ParadigmShiftModule()

    print("=== Paradigm Shift Module Summary ===\n")

    # Test at different AI capability levels
    for A in [1.0, 2.0, 3.0, 5.0, 10.0]:
        print(f"AI Capability A = {A}:")
        print(f"  Total PSM: {psm.calculate_psm(A):.2f}x")

        summary = psm.summary(A)
        for type_name, data in summary['by_type'].items():
            if data['impact'] > 0:
                print(f"  - {type_name}: impact={data['impact']:.2f}, contrib={data['contribution']:.2f}")
        print()

    # Validate against historical
    print("=== Historical Validation ===\n")
    validation = psm.validate_against_historical()
    for name, result in validation.items():
        status = "✓" if result['within_ci'] else "✗"
        print(f"{status} {name} ({result['year']}): measured={result['measured_impact']:.0f}, predicted_sat={result['predicted_saturation']:.0f}")
