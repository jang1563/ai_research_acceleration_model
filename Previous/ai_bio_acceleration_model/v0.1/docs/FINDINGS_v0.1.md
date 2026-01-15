# v0.1 Findings and Observations

## Summary

This document captures key findings, insights, and issues identified from running the v0.1 pilot model. These inform the direction of subsequent iterations.

---

## 1. Key Results

### 1.1 Cumulative Progress by 2050

| Scenario | Equivalent Years | Calendar Years | Acceleration |
|----------|------------------|----------------|--------------|
| Pessimistic | 51.6 | 26 | 2.0x |
| Baseline | 56.9 | 26 | 2.2x |
| Optimistic | 59.6 | 26 | 2.3x |

**Interpretation:** Under baseline assumptions, 26 calendar years (2024-2050) yield ~57 equivalent years of scientific progress, a 2.2x acceleration.

### 1.2 Bottleneck Analysis

**Finding:** S6 (Clinical Trials) is the bottleneck throughout the entire simulation period for all scenarios.

**Effective Service Rates at t=2024:**

| Stage | μ_eff (projects/year) |
|-------|----------------------|
| S1 | 1.90 |
| S2 | 3.60 |
| S3 | 0.30 |
| S4 | 5.70 |
| S5 | 0.75 |
| **S6** | **0.02** |
| S7 | 0.90 |
| S8 | 0.95 |

S6 has an effective rate ~15x lower than S3 (next lowest), making it the dominant bottleneck.

---

## 2. Model Behavior Analysis

### 2.1 Why Clinical Trials Dominate

The combination of:
- Very long baseline duration (72 months)
- Very low success rate (12%)
- Very limited AI acceleration potential (M_max = 2.5x)

creates an effective rate so low that no other stage becomes rate-limiting even as AI capability grows exponentially.

**Mathematical insight:**
- At t=2024: μ_6^eff = 0.167 × 1.0 × 0.12 = 0.02
- At t=2050: μ_6^eff = 0.167 × 2.5 × 0.12 = 0.05 (maximum)
- Other stages grow much faster, but S6 remains lowest

### 2.2 Limited Scenario Differentiation

The three scenarios yield similar results (52-60 equivalent years) because:
1. The bottleneck (S6) has low M_max (2.5x)
2. S6's k_saturation is low (0.3), meaning slow approach to maximum
3. Even with faster AI growth, S6 only moves from 1x to ~2.5x

**Implication:** The model is insensitive to AI growth rate when the bottleneck has low ceiling.

### 2.3 Saturation Behavior

Stages with high M_max (S1, S4) quickly saturate:
- By 2035: S1 reaches ~40x (80% of M_max=50)
- By 2035: S4 reaches ~80x (80% of M_max=100)

But these don't matter because they're not bottlenecks.

---

## 3. Model Validity Assessment

### 3.1 What the Model Gets Right

1. **Clinical trials as major bottleneck** - Aligns with real-world drug development experience
2. **Limited AI impact on physical processes** - Reflects genuine constraints
3. **Saturation dynamics** - Captures diminishing returns appropriately

### 3.2 Potential Issues

1. **Bottleneck too dominant** - S6 may be over-constrained
2. **No bottleneck transitions** - Surprising given expected shifts
3. **Low scenario sensitivity** - May not capture real uncertainty

### 3.3 Questions for v0.2

1. Should clinical trials be split into phases (as originally planned)?
2. Are the success rates too pessimistic?
3. Should M_max for clinical trials be higher with adaptive designs?
4. Is the 72-month duration appropriate for all drug types?

---

## 4. Recommendations for v0.2

### 4.1 Must Do

1. **Calibrate S6 parameters more carefully**
   - Split into Phase I/II/III with different parameters
   - Research adaptive trial acceleration potential
   - Differentiate by drug type (small molecule vs. biologic vs. gene therapy)

2. **Add literature citations for all M_max values**
   - Provide explicit justification
   - Include uncertainty ranges

### 4.2 Should Consider

1. **Increase S6 M_max** from 2.5x to perhaps 3-4x
   - Adaptive trial designs
   - Better biomarkers for patient selection
   - AI-optimized dosing

2. **Adjust S3 parameters**
   - Currently S3 is distant second bottleneck
   - May become bottleneck if S6 is relieved

3. **Test scenario sensitivity**
   - Vary M_max values by scenario
   - Optimistic scenario: regulatory reform increases S6 M_max

---

## 5. Insights for Paper

### 5.1 Key Finding to Highlight

"Clinical trials represent a persistent bottleneck that limits the impact of AI acceleration in earlier pipeline stages. Even with transformative AI capabilities in hypothesis generation and analysis, the ceiling on clinical trial acceleration constrains overall progress."

### 5.2 Policy Implication

"Investments in AI for early-stage research yield diminishing returns while clinical trial reform could have outsized impact on overall pipeline throughput."

### 5.3 Novel Contribution

The model provides a framework for quantifying which bottlenecks matter most and when they matter, enabling evidence-based policy prioritization.

---

## 6. Technical Notes

### 6.1 Numerical Behavior

- Model runs stably with no numerical issues
- Exponential growth in A(t) handled correctly
- Saturation function behaves as expected

### 6.2 Computational Performance

- Full 3-scenario run: <1 second
- All visualizations: ~5 seconds
- No optimization needed at current scale

### 6.3 Code Quality

- Clean separation of concerns (model vs. visualization)
- Well-documented functions
- Easy to extend for future iterations

---

## 7. Next Steps

1. **Review this document** and decide on parameter adjustments
2. **Research clinical trial acceleration** literature for better M_max estimates
3. **Implement v0.2** with calibrated parameters
4. **Add sensitivity analysis** to understand parameter importance
5. **Consider splitting S6** into phases to capture more nuanced dynamics
