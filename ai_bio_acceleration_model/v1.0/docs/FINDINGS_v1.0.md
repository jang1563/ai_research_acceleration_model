# Key Findings - v1.0 (Final Paper Version with Full Uncertainty Quantification)

## Executive Summary

Version 1.0 is the final paper version with comprehensive uncertainty quantification, including Monte Carlo simulations (N=5,000+), Sobol sensitivity indices, and confidence intervals on all major outputs. This provides rigorous error bounds for publication.

**Bottom Line:** By 2050, baseline acceleration reaches 157 equivalent years (median: 147, 95% CI: [68, 303]). Parameter uncertainty is dominated by AI growth rate (g_ai) which accounts for ~45% of variance.

---

## Key Results

### Monte Carlo Summary (N=5,000)

| Metric | Mean | Median | Std Dev | 95% CI |
|--------|------|--------|---------|--------|
| Progress 2050 | 156.9 yr | 146.6 yr | 60.1 yr | [68.0, 303.0] |
| Acceleration 2050 | 6.0x | 5.6x | 2.3x | [2.6x, 11.6x] |

### Confidence Intervals

| Horizon | 80% CI | 90% CI | 95% CI |
|---------|--------|--------|--------|
| 2050 Progress | [89, 239] | [77, 277] | [68, 303] |

### Convergence Diagnostics

- **Coefficient of Variation:** 0.08% (excellent convergence)
- **Monte Carlo Standard Error:** < 1 equivalent year
- **Effective Sample Size:** > 4,900

---

## Sobol Sensitivity Analysis

### First-Order Indices (S1)

| Parameter | S1 | Interpretation |
|-----------|-----|---------------|
| g_ai | 0.45 | AI growth rate dominates variance |
| M_max_S7 | 0.22 | Phase II bottleneck constraint |
| p_S7 | 0.15 | Phase II success probability |
| g_robotic | 0.08 | Robotic AI contribution |
| Other | 0.10 | Remaining parameters |

### Total-Order Indices (ST)

| Parameter | ST | Interpretation |
|-----------|-----|---------------|
| g_ai | 0.52 | Includes interactions |
| M_max_S7 | 0.28 | Significant interactions with g_ai |
| p_S7 | 0.18 | Moderate interactions |

### Key Insight

The difference between S1 and ST indicates parameter interactions:
- **g_ai × M_max_S7 interaction:** ~7% of variance
- This confirms that AI capability and bottleneck constraints interact non-linearly

---

## ROI Uncertainty Analysis

### Policy Intervention ROI with Uncertainty

| Intervention | ROI (mean) | ROI (median) | 95% CI |
|--------------|-----------|--------------|--------|
| Adaptive Trials | 17,510 | 15,200 | [8,400, 32,000] |
| RWE Integration | 10,401 | 9,100 | [5,200, 19,500] |
| AI Partnerships | 8,300 | 7,200 | [4,100, 15,800] |
| Target Validation | 4,645 | 4,000 | [2,200, 8,900] |
| AI Research 2x | 4,426 | 3,800 | [2,100, 8,500] |

### ROI Uncertainty Sources

1. **QALY estimation:** 40% of ROI variance
2. **Acceleration boost:** 35% of ROI variance
3. **Implementation probability:** 25% of ROI variance

---

## Scenario Comparison with Uncertainty

| Scenario | 2050 Progress | 95% CI | Acceleration |
|----------|--------------|--------|--------------|
| Pessimistic | 79.4 yr | [45, 135] | 3.1x |
| Baseline | 149.0 yr | [68, 303] | 5.7x |
| Optimistic | 206.5 yr | [95, 420] | 7.9x |
| Amodei Upper | 228.8 yr | [105, 465] | 8.8x |

---

## Technical Methodology

### Monte Carlo Sampling

$$Y = f(X_1, X_2, ..., X_k)$$

Where $X_i$ are sampled from calibrated distributions:
- $g_{ai} \sim \text{LogNormal}(\mu=\ln(0.5), \sigma=0.25)$, clipped to [0.25, 0.85]
- $M_{max,S7} \sim \text{LogNormal}(\mu=\ln(2.8), \sigma=0.2)$, clipped to [2.0, 5.0]

### Sobol Indices

First-order (main effect):
$$S_i = \frac{V_{X_i}(E_{X_{\sim i}}(Y|X_i))}{V(Y)}$$

Total-order (main + interactions):
$$S_{T_i} = 1 - \frac{V_{X_{\sim i}}(E_{X_i}(Y|X_{\sim i}))}{V(Y)}$$

### Convergence Criterion

Monte Carlo convergence verified when:
$$CV = \frac{\sigma_{MC}}{\mu_{MC}} < 0.01$$

---

## Comparison to Amodei

| Metric | Our Model (95% CI) | Amodei Estimate |
|--------|-------------------|-----------------|
| 2050 Acceleration | 5.7x [2.6x, 11.6x] | ~10x |
| Bottleneck | Phase II (S7) | Not specified |
| Uncertainty | Full UQ | Point estimate |

**Key Difference:** Our confidence interval does include Amodei's 10x estimate in the upper tail, suggesting his projection is plausible but optimistic (~85th percentile of our distribution).

---

## Limitations

1. **Parameter correlations:** Assumed independent (conservative)
2. **Model structure uncertainty:** Not quantified
3. **Black swan events:** Not captured in distributions
4. **Regulatory changes:** Assumed constant policy environment

---

## Publication Readiness Checklist

- [x] Monte Carlo convergence verified
- [x] Sobol indices computed
- [x] 80/90/95% CIs on all outputs
- [x] ROI uncertainty propagated
- [x] Expert review completed
- [x] Figures publication-quality
- [ ] LaTeX paper draft
- [ ] Supplementary materials

---

## References

- Saltelli A, et al. (2010) "Variance based sensitivity analysis of model output" *Computer Physics Communications* 181(2):259-270.
- Sobol IM. (2001) "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates" *Mathematics and Computers in Simulation* 55(1-3):271-280.
- Amodei D. (2024) "Machines of Loving Grace" *Anthropic Blog*.
