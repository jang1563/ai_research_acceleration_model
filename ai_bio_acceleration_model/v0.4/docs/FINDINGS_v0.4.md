# Key Findings - v0.4 (Monte Carlo Uncertainty Quantification)

## Executive Summary

Version 0.4 introduces Monte Carlo uncertainty quantification, providing 90% confidence intervals on all projections. This enables proper communication of model uncertainty and risk assessment.

**Bottom Line:** By 2050, the Baseline scenario projects **62.9 equivalent years** (90% CI: [48.6, 76.7]), representing a 2.4x acceleration with substantial uncertainty.

---

## Key Results

### Scenario Projections with Uncertainty (2050)

| Scenario | Median | 90% CI | Spread |
|----------|--------|--------|--------|
| Pessimistic | 43.4 yr | [34.7, 51.1] | ±19% |
| Baseline | 62.9 yr | [48.6, 76.7] | ±22% |
| Optimistic | 98.4 yr | [71.6, 120.1] | ±25% |

### Uncertainty by Time Horizon

| Year | Baseline Median | 90% CI |
|------|-----------------|--------|
| 2030 | 11.2 yr | [9.5, 13.1] |
| 2040 | 35.7 yr | [28.4, 43.2] |
| 2050 | 62.9 yr | [48.6, 76.7] |

**Key Observation:** Uncertainty grows over time due to compounding parameter uncertainty.

---

## Critical Insights

### 1. Uncertainty is Substantial but Bounded

The 90% CI spans roughly ±20-25% of the median, indicating:
- Model is informative (not all outcomes equally likely)
- But significant uncertainty remains
- Policy conclusions should be robust to this uncertainty

### 2. Scenarios Are Distinguishable

Even with uncertainty, the scenarios remain distinguishable:
- Pessimistic 90th percentile (51.1) < Baseline median (62.9)
- Baseline 90th percentile (76.7) < Optimistic median (98.4)

### 3. Key Uncertainty Drivers

Parameters contributing most to output variance:
1. S7_M_max (Phase II max multiplier)
2. g_ai (AI growth rate)
3. S8_M_max (Phase III max multiplier)
4. S7_p_success (Phase II success rate)

### 4. Fat Tails in Optimistic Scenario

The Optimistic scenario shows larger uncertainty spread (±25%) due to:
- More aggressive parameter assumptions
- Longer extrapolation period
- Compounding effects of high growth rates

---

## Monte Carlo Methodology

### Parameter Distributions

Each parameter sampled from a distribution reflecting uncertainty:

| Parameter Type | Distribution | Width |
|----------------|--------------|-------|
| M_max values | LogNormal | σ = 0.2 |
| p_success | Beta | α, β from literature |
| g_ai | Normal | σ = 0.05 |
| k_saturation | Normal | σ = 0.1 |

### Simulation Details

- **Samples:** 500 per scenario
- **Correlation:** Independent sampling (conservative)
- **Convergence:** Verified stable at 500 samples

### Confidence Interval Calculation

$$CI_{90\%} = [P_{5}, P_{95}]$$

Where $P_k$ is the k-th percentile of the Monte Carlo distribution.

---

## Visualization Improvements (v0.4.2)

### New/Improved Figures

1. **Tornado Diagram** - Bidirectional bars with proper baseline reference
2. **Combined Fan Chart** - All scenarios with 50%/90% confidence intervals
3. **Bottleneck Heatmap** - Time × stage constraint matrix
4. **Summary Dashboard** - Four informative panels

### Key Improvements

- Proper uncertainty bands (not just point estimates)
- Clear visual hierarchy
- Publication-quality formatting

---

## Policy Implications

### 1. Robust Conclusions

Despite uncertainty, some conclusions are robust:
- Phase II remains bottleneck in all scenarios
- Clinical trials limit overall acceleration
- AI investment has positive expected value

### 2. Risk Assessment

Decision-makers can use CIs for risk assessment:
- 5th percentile: "worst reasonable case"
- 95th percentile: "best reasonable case"
- Median: "most likely outcome"

### 3. Value of Information

High-uncertainty parameters are candidates for further research:
- Better Phase II M_max estimates
- More precise g_ai measurements
- Therapeutic area-specific parameters

---

## Limitations & Next Steps

### Current Limitations

1. Independent parameter sampling (ignores correlations)
2. Symmetric distributions for some skewed parameters
3. No uncertainty on structural model assumptions

### Planned for v0.5

- Multi-type AI differentiation (Cognitive/Robotic/Scientific)
- Therapeutic area modeling
- Stage-specific AI type contributions

---

## References

- Saltelli A, Ratto M, Andres T, et al. (2008) "Global Sensitivity Analysis: The Primer" *Wiley*. [DOI: 10.1002/9780470725184](https://doi.org/10.1002/9780470725184)
- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends Pharmacol Sci* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005)
