# Key Findings - v0.3 (Sensitivity Analysis)

## Executive Summary

Version 0.3 introduces comprehensive sensitivity analysis to identify which parameters have the greatest impact on model outcomes. This enables evidence-based policy prioritization.

**Bottom Line:** Phase II M_max (S7_M_max) has the highest sensitivity index (**0.811**), making it the top policy target for accelerating biological discovery.

---

## Key Results

### Parameter Sensitivity Rankings

| Rank | Parameter | Sensitivity Index | Elasticity | Policy Implication |
|------|-----------|-------------------|------------|-------------------|
| 1 | S7_M_max (Phase II) | **0.811** | 0.89 | Highest leverage - target Phase II automation |
| 2 | g_ai | 0.427 | 0.15 | AI R&D investment has strong returns |
| 3 | S8_M_max (Phase III) | 0.260 | 0.00 | Secondary target after Phase II |
| 4 | S7_p_success | 0.181 | 0.00 | Improve Phase II success with biomarkers |
| 5 | S8_p_success | 0.181 | -0.07 | Improve Phase III methods |

### Scenario Results (Baseline)

| Year | Cumulative Progress | Acceleration |
|------|---------------------|--------------|
| 2030 | 11.2 equiv. years | 1.9x |
| 2040 | 35.7 equiv. years | 2.2x |
| 2050 | 62.9 equiv. years | 2.4x |

---

## Critical Insights

### 1. Phase II is the Highest-Leverage Intervention Point

The sensitivity analysis confirms that Phase II trials (S7) are the critical bottleneck:
- Highest sensitivity index (0.811)
- Highest elasticity (0.89)
- Small improvements in Phase II M_max yield large gains in overall progress

### 2. AI Growth Rate Matters, But Less Than Clinical Trial Acceleration

While g_ai has a moderate sensitivity (0.427), it's less impactful than Phase II acceleration:
- Doubling AI growth rate increases progress by ~15%
- Doubling Phase II M_max increases progress by ~89%

### 3. Success Rates Have Secondary Impact

Improving p_success has lower sensitivity than M_max:
- Better biomarker selection (higher p_success) helps
- But faster trials (higher M_max) help more

### 4. Diminishing Returns at High M_max

Once Phase II M_max exceeds ~4x, the bottleneck shifts to Phase III, reducing further gains from Phase II improvements.

---

## Policy Recommendations

### Priority 1: Phase II Trial Acceleration

Invest in technologies that accelerate Phase II trials:
- Biomarker-driven patient selection
- Adaptive trial designs
- AI-powered dose optimization
- Real-world evidence integration

### Priority 2: Fundamental AI Research

Continue investment in AI R&D:
- Foundation model improvements
- Scientific AI (AlphaFold-type systems)
- Lab automation robotics

### Priority 3: Phase III After Phase II is Resolved

Once Phase II is no longer the bottleneck:
- Focus on Phase III acceleration
- Seamless Phase II/III trial designs
- Regulatory reform for faster approval

---

## Methodology

### One-at-a-Time (OAT) Sensitivity Analysis

Each parameter varied by ±20% from baseline while holding others constant:

$$S_i = \frac{|Y(p_i^+) - Y(p_i^-)|}{Y_{baseline}}$$

### Elasticity Calculation

$$\varepsilon_i = \frac{\Delta Y / Y}{\Delta p_i / p_i}$$

### Tornado Diagram

Visual representation of parameter impact, sorted by sensitivity index.

---

## Limitations & Next Steps

### Current Limitations

1. OAT sensitivity doesn't capture interactions
2. Linear approximation for elasticity
3. No uncertainty on sensitivity estimates

### Planned for v0.4

- Monte Carlo uncertainty quantification
- Global sensitivity analysis (Sobol indices)
- Parameter interaction effects

---

## References

- Saltelli A, Ratto M, Andres T, et al. (2008) "Global Sensitivity Analysis: The Primer" *Wiley*. [DOI: 10.1002/9780470725184](https://doi.org/10.1002/9780470725184)
- Pianosi F, Beven K, Freer J, et al. (2016) "Sensitivity analysis of environmental models: A systematic review with practical workflow" *Environ Model Softw* 79:214-232. [DOI: 10.1016/j.envsoft.2016.02.008](https://doi.org/10.1016/j.envsoft.2016.02.008)
- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)
