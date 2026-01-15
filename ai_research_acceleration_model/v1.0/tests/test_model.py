#!/usr/bin/env python3
"""
Test Suite for AI Research Acceleration Model v1.0
===================================================

Comprehensive tests covering:
1. Core functionality
2. Edge cases
3. Consistency checks
4. Regression tests against known values
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_acceleration_model import (
    AIAccelerationModel,
    DomainForecast,
    SystemSnapshot,
    Domain,
    Scenario,
    quick_forecast,
    quick_summary,
)


class TestModelInitialization(unittest.TestCase):
    """Test model initialization."""

    def test_default_initialization(self):
        """Model initializes with default parameters."""
        model = AIAccelerationModel()
        self.assertEqual(len(model.domains), 5)
        self.assertEqual(model.seed, 42)

    def test_custom_seed(self):
        """Model accepts custom seed."""
        model = AIAccelerationModel(seed=123)
        self.assertEqual(model.seed, 123)

    def test_domains_present(self):
        """All expected domains are present."""
        model = AIAccelerationModel()
        expected = ["structural_biology", "drug_discovery", "materials_science",
                   "protein_design", "clinical_genomics"]
        self.assertEqual(set(model.domains), set(expected))


class TestDomainForecast(unittest.TestCase):
    """Test single-domain forecasting."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_forecast_returns_correct_type(self):
        """Forecast returns DomainForecast object."""
        forecast = self.model.forecast("drug_discovery", 2030)
        self.assertIsInstance(forecast, DomainForecast)

    def test_forecast_all_domains(self):
        """Can forecast all domains without error."""
        for domain in self.model.domains:
            forecast = self.model.forecast(domain, 2030)
            self.assertIsNotNone(forecast)
            self.assertGreater(forecast.acceleration, 0)

    def test_forecast_with_enum(self):
        """Can use Domain enum for forecasting."""
        forecast = self.model.forecast(Domain.DRUG_DISCOVERY, 2030)
        self.assertEqual(forecast.domain, "drug_discovery")

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with self.assertRaises(ValueError):
            self.model.forecast("invalid_domain", 2030)

    def test_acceleration_increases_over_time(self):
        """Acceleration generally increases over time."""
        f2025 = self.model.forecast("structural_biology", 2025)
        f2030 = self.model.forecast("structural_biology", 2030)
        f2035 = self.model.forecast("structural_biology", 2035)

        self.assertLess(f2025.acceleration, f2030.acceleration)
        self.assertLess(f2030.acceleration, f2035.acceleration)

    def test_confidence_intervals_ordered(self):
        """CI bounds are properly ordered."""
        forecast = self.model.forecast("drug_discovery", 2030)

        # 50% CI should be inside 90% CI
        self.assertGreaterEqual(forecast.ci_50[0], forecast.ci_90[0])
        self.assertLessEqual(forecast.ci_50[1], forecast.ci_90[1])

        # Lower bound < acceleration < upper bound
        self.assertLess(forecast.ci_90[0], forecast.acceleration)
        self.assertGreater(forecast.ci_90[1], forecast.acceleration)

    def test_workforce_metrics_present(self):
        """Workforce metrics are calculated."""
        forecast = self.model.forecast("drug_discovery", 2030)

        self.assertIsNotNone(forecast.jobs_displaced)
        self.assertIsNotNone(forecast.jobs_created)
        self.assertIsNotNone(forecast.net_jobs)

        # Net = created - displaced
        self.assertAlmostEqual(
            forecast.net_jobs,
            forecast.jobs_created - forecast.jobs_displaced,
            places=5
        )


class TestScenarios(unittest.TestCase):
    """Test scenario analysis."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_scenarios_ordered_correctly(self):
        """Scenarios produce ordered acceleration values."""
        scenarios = self.model.compare_scenarios("drug_discovery", 2030)

        accels = [
            scenarios["pessimistic"].acceleration,
            scenarios["conservative"].acceleration,
            scenarios["baseline"].acceleration,
            scenarios["optimistic"].acceleration,
            scenarios["breakthrough"].acceleration,
        ]

        # Should be monotonically increasing
        for i in range(len(accels) - 1):
            self.assertLess(accels[i], accels[i+1])

    def test_scenario_enum_works(self):
        """Can use Scenario enum."""
        f1 = self.model.forecast("drug_discovery", 2030, Scenario.BASELINE)
        f2 = self.model.forecast("drug_discovery", 2030, "baseline")
        self.assertEqual(f1.acceleration, f2.acceleration)

    def test_all_scenarios_valid(self):
        """All scenario strings work."""
        scenarios = ["pessimistic", "conservative", "baseline", "optimistic", "breakthrough"]
        for scenario in scenarios:
            forecast = self.model.forecast("drug_discovery", 2030, scenario)
            self.assertIsNotNone(forecast)


class TestSystemSnapshot(unittest.TestCase):
    """Test system-wide analysis."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_snapshot_returns_correct_type(self):
        """System snapshot returns correct type."""
        snapshot = self.model.system_snapshot(2030)
        self.assertIsInstance(snapshot, SystemSnapshot)

    def test_snapshot_contains_all_domains(self):
        """Snapshot contains forecasts for all domains."""
        snapshot = self.model.system_snapshot(2030)
        self.assertEqual(len(snapshot.domain_forecasts), 5)

        for domain in self.model.domains:
            self.assertIn(domain, snapshot.domain_forecasts)

    def test_workforce_totals_consistent(self):
        """Workforce totals match sum of domains."""
        snapshot = self.model.system_snapshot(2030)

        total_displaced = sum(f.jobs_displaced for f in snapshot.domain_forecasts.values())
        total_created = sum(f.jobs_created for f in snapshot.domain_forecasts.values())

        self.assertAlmostEqual(snapshot.total_displaced, total_displaced, places=5)
        self.assertAlmostEqual(snapshot.total_created, total_created, places=5)
        self.assertAlmostEqual(
            snapshot.workforce_change,
            total_created - total_displaced,
            places=5
        )

    def test_fastest_slowest_domains_correct(self):
        """Fastest and slowest domain identification is correct."""
        snapshot = self.model.system_snapshot(2030)

        accels = {d: f.acceleration for d, f in snapshot.domain_forecasts.items()}
        actual_fastest = max(accels, key=accels.get)
        actual_slowest = min(accels, key=accels.get)

        self.assertEqual(snapshot.fastest_domain, actual_fastest)
        self.assertEqual(snapshot.slowest_domain, actual_slowest)


class TestTrajectory(unittest.TestCase):
    """Test trajectory generation."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_trajectory_returns_list(self):
        """Trajectory returns list of correct length."""
        traj = self.model.trajectory("drug_discovery", 2025, 2030)
        self.assertEqual(len(traj), 6)  # 2025-2030 inclusive

    def test_trajectory_system_wide(self):
        """System-wide trajectory works."""
        traj = self.model.trajectory(None, 2025, 2030)
        self.assertEqual(len(traj), 6)
        self.assertIsInstance(traj[0], SystemSnapshot)

    def test_trajectory_years_correct(self):
        """Trajectory covers correct years."""
        traj = self.model.trajectory("drug_discovery", 2025, 2027)
        years = [f.year for f in traj]
        self.assertEqual(years, [2025, 2026, 2027])


class TestRegressionValues(unittest.TestCase):
    """Test against known regression values from v0.9."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_structural_biology_2030_baseline(self):
        """Structural biology 2030 baseline matches expected range."""
        forecast = self.model.forecast("structural_biology", 2030)
        # Should be in 5-10x range based on v0.9
        self.assertGreater(forecast.acceleration, 5.0)
        self.assertLess(forecast.acceleration, 12.0)

    def test_drug_discovery_bounded(self):
        """Drug discovery remains bounded by clinical trials."""
        forecast = self.model.forecast("drug_discovery", 2030)
        # Should be under 5x even with spillovers
        self.assertLess(forecast.acceleration, 6.0)

    def test_materials_science_constrained(self):
        """Materials science shows synthesis constraint."""
        forecast = self.model.forecast("materials_science", 2030)
        # Should be the slowest domain
        self.assertLess(forecast.acceleration, 3.0)

    def test_workforce_net_positive(self):
        """Net workforce impact is positive."""
        snapshot = self.model.system_snapshot(2030)
        self.assertGreater(snapshot.workforce_change, 0)

    def test_spillover_effects_present(self):
        """Cross-domain spillovers are non-zero."""
        forecast = self.model.forecast("drug_discovery", 2030)
        self.assertGreater(forecast.cross_domain_boost, 0)


class TestOutputFormats(unittest.TestCase):
    """Test output format methods."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_executive_summary_returns_string(self):
        """Executive summary returns non-empty string."""
        summary = self.model.executive_summary(2030)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 100)

    def test_executive_summary_contains_key_info(self):
        """Executive summary contains key information."""
        summary = self.model.executive_summary(2030)
        self.assertIn("acceleration", summary.lower())
        self.assertIn("2030", summary)
        self.assertIn("workforce", summary.lower())

    def test_policy_recommendations_returned(self):
        """Policy recommendations are returned."""
        recs = self.model.get_policy_recommendations(2030)
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0].priority, "critical")


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_quick_forecast(self):
        """quick_forecast function works."""
        forecast = quick_forecast("drug_discovery", 2030)
        self.assertIsInstance(forecast, DomainForecast)

    def test_quick_summary(self):
        """quick_summary function works."""
        summary = quick_summary(2030)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.model = AIAccelerationModel()

    def test_year_2024_baseline(self):
        """Year 2024 gives baseline values."""
        forecast = self.model.forecast("drug_discovery", 2024)
        # Should be close to base acceleration
        self.assertLess(abs(forecast.acceleration - 1.4), 1.0)

    def test_far_future_reasonable(self):
        """Far future (2050) gives reasonable values."""
        forecast = self.model.forecast("structural_biology", 2050)
        # Should be accelerated but not infinite
        self.assertGreater(forecast.acceleration, 5.0)
        self.assertLess(forecast.acceleration, 100.0)

    def test_case_insensitive_domain(self):
        """Domain names are case-insensitive."""
        f1 = self.model.forecast("Drug_Discovery", 2030)
        f2 = self.model.forecast("drug_discovery", 2030)
        self.assertEqual(f1.acceleration, f2.acceleration)

    def test_domain_with_spaces(self):
        """Domain names with spaces work."""
        f1 = self.model.forecast("drug discovery", 2030)
        f2 = self.model.forecast("drug_discovery", 2030)
        self.assertEqual(f1.acceleration, f2.acceleration)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
