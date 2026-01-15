# Changelog - v1.1

## Version 1.1 (January 2026)

### Expert Panel Review Implementation

This version implements fixes from a 15-expert simulated review panel.
**Note: Expert review was AI-simulated using Claude (Anthropic).**

---

## P1 Critical Fixes

### P1-1: Sobol Indices Labeled as Approximate
- All Sobol sensitivity indices now explicitly marked as "APPROXIMATE"
- Added `is_approximate` flag to SobolIndices dataclass
- Documented that correlation-based proxy is used, not full Saltelli method
- **Impact**: Transparency improvement, no numerical change

### P1-2: Calibrated g_ai Distribution
- Increased uncertainty: σ = 0.25 → σ = 0.50 (doubled)
- Expanded bounds: [0.2, 0.9] → [0.15, 1.0]
- Calibration source: Epoch AI + AI Impacts Survey
- **Impact**: Wider confidence intervals on all outputs

### P1-3: Historical Validation Module
- New `historical_validation.py` module
- Validates against FDA approvals 2015-2023
- Compares to Wong et al. (2019) Phase success rates
- Computes MAE, MAPE, RMSE, R² metrics
- **Impact**: Model credibility assessment

### P1-4: Reduced Wet Lab M_max
- S3 (Wet Lab) M_max: 5.0 → 2.5
- Rationale: Biological timescales are irreducible (cell division ~24hr)
- Source: Expert C3 (robotics professor)
- **Impact**: Lower acceleration for physical stages

### P1-5: Regulatory Floor Enforcement
- S9 now has `regulatory_floor_months = 6`
- PDUFA minimum review times cannot be bypassed by AI
- AI multiplier capped at 2.0 for regulatory stage
- **Impact**: More realistic regulatory timelines

### P1-6: Logistic AI Growth Model
- Default growth model changed from exponential to logistic
- `A(t) = A_max / (1 + (A_max - 1) * exp(-g * (t - t0)))`
- Saturation parameter `ai_capability_ceiling` per scenario
- **Impact**: Earlier saturation, lower long-term acceleration

### P1-7: AI Winter Scenario
- New scenario type: `ScenarioType.AI_WINTER`
- 15% probability of progress plateau after 2030
- `ai_winter_probability` parameter for Monte Carlo
- **Impact**: Captures tail risk of AI stagnation

### P1-8: Global Access Factor
- New `global_access_factor` per therapeutic area
- Oncology: 0.4, CNS: 0.3, Infectious: 0.7, Rare: 0.2
- Adjusts beneficiary estimates for LMIC populations
- **Impact**: More realistic global health impact

### P1-9: Methodology Disclosure
- Explicit statement that expert review was AI-simulated
- Added to model docstrings and REPRODUCIBILITY.md
- Recommended external validation steps documented
- **Impact**: Transparency improvement

### P1-10: Reproducibility Artifacts
- Added `requirements.txt` with pinned versions
- Added `REPRODUCIBILITY.md` with seed documentation
- Random seed: 42 for all stochastic components
- Verification checksums provided
- **Impact**: Exact reproducibility enabled

---

## P2 Important Fixes

### P2-11: Bootstrap CIs on Sobol Indices
- 1000 bootstrap samples for confidence intervals
- 90% CI reported for all Sobol indices
- **Impact**: Uncertainty quantification on sensitivity

### P2-12: Disease-Specific Phase II M_max
- New `phase2_M_max_override` in TherapeuticAreaParams
- Range: [1.5, 5.0] depending on therapeutic area
- Oncology: 3.5 (biomarker-driven), CNS: 2.0 (complex biology)
- **Impact**: More nuanced therapeutic predictions

### P2-13: Manufacturing Constraints
- S10 now has `manufacturing_capacity_limit = 3.0`
- Novel modalities (cell/gene therapy) have supply constraints
- **Impact**: Realistic deployment bottleneck

### P2-14: Compute Constraints on AI
- New `compute_constraint` parameter per AI type
- Cognitive: 0.9, Robotic: 1.0, Scientific: 0.85
- Reflects training compute limitations
- **Impact**: Slightly lower cognitive AI growth

### P2-15: Policy Implementation Curves
- New `implementation_lag` and `implementation_adoption_rate` parameters
- Policies don't reach full effect immediately
- **Impact**: More realistic policy analysis

### P2-16: Expanded QALY Range
- Value per QALY: $50K - $200K (was $100K fixed)
- Reflects NICE (UK) to US willingness-to-pay range
- **Impact**: Wider uncertainty on economic value

### P2-17: Vaccine Pipeline Pathway
- New `TherapeuticArea.VACCINE` with distinct parameters
- Faster timelines (COVID precedent)
- Higher global access (COVAX-style)
- **Impact**: Better pandemic preparedness modeling

### P2-18: Reduced S1 p_success
- S1 (Hypothesis) p_success: 0.95 → 0.40
- Rationale: 90%+ of biological hypotheses fail to translate
- Source: Expert B2 (computational biology)
- **Impact**: Lower overall progress, more realistic

---

## Summary of Numerical Changes

| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| S3 M_max | 5.0 | 2.5 | -50% |
| S1 p_success | 0.95 | 0.40 | -58% |
| g_ai σ | 0.25 | 0.50 | +100% |
| AI growth model | Exponential | Logistic | - |
| Regulatory floor | None | 6 months | New |
| AI winter prob | 0% | 15% | New |
| Global access | 100% | 20-80% | New |

## Breaking Changes

1. **API**: `ai_growth_model` parameter required for scenarios
2. **Output**: Cumulative progress values ~20-30% lower
3. **Uncertainty**: 95% CI approximately 2x wider
4. **Scenarios**: New "AI_Winter" scenario in defaults

## Migration Guide

```python
# v1.0 code
scenario = Scenario(name="Baseline", g_ai=0.50, ...)

# v1.1 code (explicit growth model)
scenario = Scenario(
    name="Baseline",
    g_ai=0.50,
    ai_growth_model=AIGrowthModel.LOGISTIC,
    ai_capability_ceiling=100.0,
    ...
)
```

## Files Changed

- `src/model.py` - Core model with all P1/P2 fixes
- `src/uncertainty_quantification.py` - Calibrated distributions
- `src/historical_validation.py` - New validation module
- `requirements.txt` - Pinned dependencies
- `REPRODUCIBILITY.md` - Seeds and verification
- `CHANGELOG.md` - This file
