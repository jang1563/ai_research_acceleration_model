#!/usr/bin/env python3
"""
Outcome Translation Module for AI-Accelerated Biological Discovery Model

This module translates abstract model outputs (equivalent years, progress rates)
into concrete, policy-relevant outcomes that non-specialists can understand.

Created based on expert reviewer feedback (Dr. Rachel Kim, MIT Media Lab):
- "93.5 equivalent years is meaningless to policymakers"
- "Need concrete translations: therapies, patients, diseases"

Version: 0.5.1
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class OutcomeTranslation:
    """Container for translated outcomes."""
    # Core metrics
    acceleration_factor: float
    calendar_years: int
    equivalent_years: float

    # Therapy projections
    baseline_therapies: int
    projected_therapies: int
    additional_therapies: int

    # Patient impact (conservative estimates)
    patients_impacted_millions: float

    # Disease-specific projections
    disease_projections: Dict[str, str]

    # Plain English summary
    summary: str
    policy_statement: str


class OutcomeTranslator:
    """
    Translates model outputs to policy-relevant outcomes.

    Key assumptions (conservative estimates based on literature):
    - Baseline: ~50 novel therapies approved 2024-2050 at current pace
    - Each novel therapy benefits ~500K-2M patients over lifecycle
    - Not all accelerated discoveries translate to approvals (0.7 factor)

    References:
    - FDA Novel Drug Approvals (2019-2023): ~45 per year
    - DiMasi et al. (2016): Drug development success rates
    - IQVIA (2023): Patient populations for approved drugs
    """

    # Baseline assumptions
    BASELINE_THERAPIES_2024_2050 = 50  # Conservative estimate
    TRANSLATION_EFFICIENCY = 0.7  # Not all progress = approvals
    PATIENTS_PER_THERAPY_MILLIONS = 1.0  # Average lifecycle impact

    # Disease-specific acceleration potential
    DISEASE_ACCELERATION = {
        'cancer': {
            'baseline_treatments': 15,
            'ai_multiplier': 1.4,  # Oncology benefits most (M_mult from model)
            'description': 'new targeted cancer therapies'
        },
        'rare_diseases': {
            'baseline_treatments': 8,
            'ai_multiplier': 1.2,
            'description': 'rare disease treatments (vs ~2 per year currently)'
        },
        'infectious': {
            'baseline_treatments': 10,
            'ai_multiplier': 1.15,
            'description': 'new antibiotics and antivirals'
        },
        'cns': {
            'baseline_treatments': 5,
            'ai_multiplier': 0.85,  # CNS is hardest
            'description': 'neurological disease treatments (Alzheimer\'s, Parkinson\'s)'
        },
        'cardiovascular': {
            'baseline_treatments': 7,
            'ai_multiplier': 1.1,
            'description': 'cardiovascular therapies'
        }
    }

    def __init__(self,
                 baseline_year: int = 2024,
                 horizon_year: int = 2050):
        self.baseline_year = baseline_year
        self.horizon_year = horizon_year
        self.calendar_years = horizon_year - baseline_year

    def translate(self,
                  equivalent_years: float,
                  scenario_name: str = 'Baseline',
                  therapeutic_area: Optional[str] = None) -> OutcomeTranslation:
        """
        Translate equivalent years to concrete outcomes.

        Parameters
        ----------
        equivalent_years : float
            Model output: cumulative equivalent years of progress
        scenario_name : str
            Scenario name for context
        therapeutic_area : str, optional
            Specific therapeutic area for targeted projections

        Returns
        -------
        OutcomeTranslation
            Container with all translated outcomes
        """
        # Calculate acceleration factor
        acceleration_factor = equivalent_years / self.calendar_years

        # Project therapies (conservative)
        projected_therapies = int(
            self.BASELINE_THERAPIES_2024_2050 *
            acceleration_factor *
            self.TRANSLATION_EFFICIENCY
        )
        additional_therapies = projected_therapies - self.BASELINE_THERAPIES_2024_2050

        # Patient impact
        patients_impacted = additional_therapies * self.PATIENTS_PER_THERAPY_MILLIONS

        # Disease-specific projections
        disease_projections = self._project_by_disease(acceleration_factor)

        # Generate summaries
        summary = self._generate_summary(
            acceleration_factor,
            projected_therapies,
            additional_therapies,
            scenario_name
        )

        policy_statement = self._generate_policy_statement(
            acceleration_factor,
            additional_therapies,
            patients_impacted
        )

        return OutcomeTranslation(
            acceleration_factor=acceleration_factor,
            calendar_years=self.calendar_years,
            equivalent_years=equivalent_years,
            baseline_therapies=self.BASELINE_THERAPIES_2024_2050,
            projected_therapies=projected_therapies,
            additional_therapies=additional_therapies,
            patients_impacted_millions=patients_impacted,
            disease_projections=disease_projections,
            summary=summary,
            policy_statement=policy_statement
        )

    def _project_by_disease(self, acceleration_factor: float) -> Dict[str, str]:
        """Generate disease-specific projections."""
        projections = {}

        for disease, params in self.DISEASE_ACCELERATION.items():
            baseline = params['baseline_treatments']
            multiplier = params['ai_multiplier']
            desc = params['description']

            # Apply both overall acceleration and disease-specific multiplier
            projected = int(baseline * acceleration_factor * multiplier * self.TRANSLATION_EFFICIENCY)
            additional = projected - baseline

            if additional > 0:
                projections[disease] = f"+{additional} {desc} (vs {baseline} at current pace)"
            else:
                projections[disease] = f"{projected} {desc} (challenging area)"

        return projections

    def _generate_summary(self,
                          acceleration_factor: float,
                          projected_therapies: int,
                          additional_therapies: int,
                          scenario_name: str) -> str:
        """Generate plain English summary."""

        if acceleration_factor < 2.0:
            pace_description = "modestly faster"
        elif acceleration_factor < 3.0:
            pace_description = "significantly faster"
        elif acceleration_factor < 4.0:
            pace_description = "dramatically faster"
        else:
            pace_description = "transformatively faster"

        return f"""
WHAT THIS MEANS ({scenario_name} Scenario)
{'=' * 50}

PACE OF DISCOVERY: {acceleration_factor:.1f}x current speed
By 2050, biological research will move {pace_description} than today.

THERAPY IMPACT: ~{projected_therapies} new therapies
Compared to ~{self.BASELINE_THERAPIES_2024_2050} at current pace,
this represents {additional_therapies:+d} additional breakthrough treatments.

IN HUMAN TERMS:
Each additional therapy typically benefits 500,000 - 2 million patients
over its lifecycle. This acceleration could impact tens of millions of lives.
""".strip()

    def _generate_policy_statement(self,
                                   acceleration_factor: float,
                                   additional_therapies: int,
                                   patients_impacted: float) -> str:
        """Generate policy-focused statement."""

        return f"""
POLICY IMPLICATIONS
{'=' * 50}

KEY FINDING: AI could accelerate biological discovery by {acceleration_factor:.1f}x,
potentially delivering {additional_therapies:+d} additional therapies by 2050.

BOTTOM LINE: Even accounting for physical constraints (clinical trials,
regulatory processes), AI-driven acceleration is substantial and real.

CRITICAL BOTTLENECK: Phase II clinical trials remain the limiting factor.
Policy interventions targeting trial efficiency have highest ROI.

PATIENT IMPACT: Estimated {patients_impacted:.0f} million additional patients
could benefit from accelerated therapeutic development.
""".strip()

    def format_uncertainty_for_policy(self,
                                      median: float,
                                      ci_low: float,
                                      ci_high: float,
                                      confidence: int = 90) -> str:
        """
        Reframe confidence intervals for policymaker understanding.

        Instead of: "90% CI: [70, 115]"
        Say: "We're 90% confident acceleration will be between 2.7x and 4.4x"
        """
        median_factor = median / self.calendar_years
        low_factor = ci_low / self.calendar_years
        high_factor = ci_high / self.calendar_years

        return f"""
UNCERTAINTY STATEMENT
{'=' * 50}

EXPECTED OUTCOME: {median_factor:.1f}x acceleration ({median:.0f} equivalent years)

CONFIDENCE RANGE: We are {confidence}% confident that:
- Acceleration will be at least {low_factor:.1f}x (even in pessimistic conditions)
- Acceleration could reach {high_factor:.1f}x (in favorable conditions)

INTERPRETATION:
- There is only a {(100-confidence)//2}% chance acceleration falls below {low_factor:.1f}x
- Even the conservative estimate represents substantial progress
""".strip()

    def compare_therapeutic_areas(self,
                                  area_results: Dict[str, float]) -> str:
        """
        Generate comparative statement for therapeutic areas.

        Parameters
        ----------
        area_results : Dict[str, float]
            {therapeutic_area: equivalent_years}
        """
        # Sort by progress
        sorted_areas = sorted(area_results.items(), key=lambda x: x[1], reverse=True)

        best_area, best_progress = sorted_areas[0]
        worst_area, worst_progress = sorted_areas[-1]

        best_factor = best_progress / self.calendar_years
        worst_factor = worst_progress / self.calendar_years
        gap = best_progress - worst_progress
        gap_pct = (best_progress / worst_progress - 1) * 100

        lines = [
            "THERAPEUTIC AREA COMPARISON",
            "=" * 50,
            "",
            "RANKING BY 2050 PROGRESS:",
        ]

        for i, (area, progress) in enumerate(sorted_areas, 1):
            factor = progress / self.calendar_years
            lines.append(f"  {i}. {area.title():15} {progress:6.1f} yr ({factor:.1f}x)")

        lines.extend([
            "",
            f"KEY INSIGHT:",
            f"  {best_area.title()} will see {gap_pct:.0f}% more progress than {worst_area.title()}",
            f"  ({best_progress:.0f} vs {worst_progress:.0f} equivalent years)",
            "",
            "WHY THE GAP:",
            f"  - {best_area.title()}: Benefits from biomarker-driven trial designs",
            f"  - {worst_area.title()}: Complex biology, lower trial success rates",
            "",
            "POLICY IMPLICATION:",
            f"  Extra investment in {worst_area.title()} infrastructure may be warranted",
            f"  to close the gap with {best_area.title()}.",
        ])

        return "\n".join(lines)


def translate_model_results(results_df,
                           scenario: str = 'Baseline',
                           year: int = 2050) -> OutcomeTranslation:
    """
    Convenience function to translate model results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model results with 'scenario', 'year', 'cumulative_progress' columns
    scenario : str
        Scenario to translate
    year : int
        Target year

    Returns
    -------
    OutcomeTranslation
    """
    translator = OutcomeTranslator(baseline_year=2024, horizon_year=year)

    row = results_df[
        (results_df['scenario'] == scenario) &
        (results_df['year'] == year)
    ].iloc[0]

    equiv_years = row['cumulative_progress']

    return translator.translate(equiv_years, scenario)


def generate_outcome_report(results_df,
                           scenarios: List[str] = None,
                           output_path: str = None) -> str:
    """
    Generate comprehensive outcome report for all scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model results
    scenarios : List[str], optional
        Scenarios to include (default: all)
    output_path : str, optional
        Path to save report

    Returns
    -------
    str
        Complete report text
    """
    if scenarios is None:
        scenarios = ['Pessimistic', 'Baseline', 'Optimistic']

    translator = OutcomeTranslator()

    report_lines = [
        "=" * 70,
        "AI-ACCELERATED BIOLOGICAL DISCOVERY: OUTCOME TRANSLATION REPORT",
        "=" * 70,
        "",
        "This report translates model outputs into concrete, policy-relevant",
        "outcomes that non-specialists can understand.",
        "",
        "-" * 70,
    ]

    for scenario in scenarios:
        try:
            row = results_df[
                (results_df['scenario'] == scenario) &
                (results_df['year'] == 2050)
            ].iloc[0]

            equiv_years = row['cumulative_progress']
            translation = translator.translate(equiv_years, scenario)

            report_lines.extend([
                "",
                translation.summary,
                "",
                "-" * 70,
            ])
        except (IndexError, KeyError):
            continue

    # Add therapeutic area comparison if available
    area_scenarios = ['Baseline_Oncology', 'Baseline_CNS', 'Baseline_Infectious',
                      'Baseline_Rare_Disease']

    area_results = {}
    for area_scenario in area_scenarios:
        try:
            row = results_df[
                (results_df['scenario'] == area_scenario) &
                (results_df['year'] == 2050)
            ].iloc[0]
            area_name = area_scenario.replace('Baseline_', '').replace('_', ' ')
            area_results[area_name] = row['cumulative_progress']
        except (IndexError, KeyError):
            continue

    # Add General baseline
    try:
        general_row = results_df[
            (results_df['scenario'] == 'Baseline') &
            (results_df['year'] == 2050)
        ].iloc[0]
        area_results['General'] = general_row['cumulative_progress']
    except (IndexError, KeyError):
        pass

    if len(area_results) > 1:
        report_lines.extend([
            "",
            translator.compare_therapeutic_areas(area_results),
            "",
            "-" * 70,
        ])

    # Policy statement
    baseline_translation = translator.translate(
        results_df[
            (results_df['scenario'] == 'Baseline') &
            (results_df['year'] == 2050)
        ].iloc[0]['cumulative_progress'],
        'Baseline'
    )

    report_lines.extend([
        "",
        baseline_translation.policy_statement,
        "",
        "=" * 70,
    ])

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


# Glossary of key terms
GLOSSARY = """
GLOSSARY OF KEY TERMS
=====================

EQUIVALENT YEARS
    A measure of how much scientific progress is made compared to the
    2024 baseline pace. For example, "93.5 equivalent years by 2050"
    means that in the 26 calendar years from 2024-2050, we make as much
    progress as would take 93.5 years at the 2024 pace. This represents
    ~3.6x acceleration.

ACCELERATION FACTOR
    The ratio of equivalent years to calendar years. An acceleration
    factor of 3x means discoveries happen three times faster than today.

BOTTLENECK
    The slowest step in the discovery pipeline that limits overall
    throughput. Like a narrow section of pipe that restricts water flow,
    the bottleneck determines how fast the entire system can operate.
    Currently: Phase II clinical trials.

SERVICE RATE
    How quickly each stage of the pipeline processes scientific projects.
    Measured in projects per year. Higher service rate = faster stage.

M_MAX (Maximum AI Multiplier)
    The theoretical maximum speedup AI can provide for each stage.
    For example, M_max = 5 for wet lab means AI can at most make
    experiments 5x faster (due to physical constraints like cell
    division time).

PROGRESS RATE
    How fast scientific discovery is happening relative to 2024.
    Progress rate of 2.0 means discoveries are happening twice as fast.

CONFIDENCE INTERVAL (CI)
    A range that captures uncertainty. A "90% CI of [70, 115]" means
    we're 90% confident the true value falls between 70 and 115.

THERAPEUTIC AREA
    A category of diseases/conditions:
    - Oncology: Cancer treatments
    - CNS: Brain/nervous system diseases (Alzheimer's, Parkinson's)
    - Infectious: Bacterial/viral diseases
    - Rare Disease: Conditions affecting <200,000 people

MONTE CARLO SIMULATION
    A technique that runs the model thousands of times with slightly
    different inputs to understand how uncertainty in assumptions
    affects conclusions.
"""


if __name__ == "__main__":
    # Example usage
    print(GLOSSARY)

    # Example translation
    translator = OutcomeTranslator()

    # Baseline scenario
    baseline_result = translator.translate(93.5, 'Baseline')
    print(baseline_result.summary)
    print()
    print(baseline_result.policy_statement)

    # Uncertainty example
    print()
    print(translator.format_uncertainty_for_policy(93.5, 70, 115))

    # Therapeutic area comparison
    print()
    area_results = {
        'Oncology': 128.5,
        'Infectious': 109.1,
        'Rare Disease': 95.9,
        'General': 93.5,
        'CNS': 76.0
    }
    print(translator.compare_therapeutic_areas(area_results))
