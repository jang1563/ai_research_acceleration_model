# Supplementary Table S5: Sensitivity Analysis

## AI Research Acceleration Model v1.1

**Purpose**: Identify which parameters most strongly influence model outputs and quantify uncertainty propagation.

---

## S5.1 Sensitivity Analysis Overview

### Methods Used

| Method | Description | Purpose |
|--------|-------------|---------|
| **One-at-a-Time (OAT)** | Vary each parameter ±20% | Identify key drivers |
| **Tornado Diagram** | Rank parameters by impact | Visual prioritization |
| **Monte Carlo** | Sample from distributions | Full uncertainty propagation |
| **Scenario Bounds** | Compare scenarios | Range of outcomes |

---

## S5.2 One-at-a-Time (OAT) Sensitivity Analysis

### Drug Discovery 2030 (Baseline Scenario)

| Parameter | -20% | Baseline | +20% | Sensitivity Index |
|-----------|------|----------|------|-------------------|
| Base acceleration | 1.35x | 1.68x | 2.01x | **0.39** |
| Time ceiling | 1.52x | 1.68x | 1.84x | 0.19 |
| Growth rate (k) | 1.59x | 1.68x | 1.77x | 0.11 |
| Spillover (from SB) | 1.61x | 1.68x | 1.75x | 0.08 |
| Lag years | 1.65x | 1.68x | 1.71x | 0.04 |
| Spillover cap | 1.66x | 1.68x | 1.70x | 0.02 |

**Sensitivity Index** = (High - Low) / Baseline

### Structural Biology 2030 (Baseline Scenario)

| Parameter | -20% | Baseline | +20% | Sensitivity Index |
|-----------|------|----------|------|-------------------|
| Base acceleration | 7.13x | 8.91x | 10.69x | **0.40** |
| Time ceiling | 7.67x | 8.91x | 10.15x | 0.28 |
| Growth rate (k) | 8.12x | 8.91x | 9.70x | 0.18 |
| Midpoint (t0) | 8.49x | 8.91x | 9.33x | 0.09 |
| Spillover (from MS) | 8.87x | 8.91x | 8.95x | 0.01 |

### Materials Science 2030 (Baseline Scenario)

| Parameter | -20% | Baseline | +20% | Sensitivity Index |
|-----------|------|----------|------|-------------------|
| Base acceleration | 1.01x | 1.26x | 1.52x | **0.40** |
| Time ceiling | 1.14x | 1.26x | 1.39x | 0.20 |
| Growth rate (k) | 1.19x | 1.26x | 1.34x | 0.12 |
| Spillover (from PD) | 1.24x | 1.26x | 1.28x | 0.03 |

### Protein Design 2030 (Baseline Scenario)

| Parameter | -20% | Baseline | +20% | Sensitivity Index |
|-----------|------|----------|------|-------------------|
| Base acceleration | 4.38x | 5.47x | 6.56x | **0.40** |
| Time ceiling | 4.85x | 5.47x | 6.09x | 0.23 |
| Growth rate (k) | 5.01x | 5.47x | 5.93x | 0.17 |
| Spillover (from SB) | 5.27x | 5.47x | 5.67x | 0.07 |

### Clinical Genomics 2030 (Baseline Scenario)

| Parameter | -20% | Baseline | +20% | Sensitivity Index |
|-----------|------|----------|------|-------------------|
| Base acceleration | 3.35x | 4.19x | 5.02x | **0.40** |
| Time ceiling | 3.79x | 4.19x | 4.59x | 0.19 |
| Growth rate (k) | 3.91x | 4.19x | 4.47x | 0.13 |
| Spillover (from DD) | 4.14x | 4.19x | 4.24x | 0.02 |

---

## S5.3 Tornado Diagram - System Acceleration 2030

**Parameter ranking by impact on total system acceleration:**

```
Parameter                          Impact (% change in output)
─────────────────────────────────────────────────────────────────
Drug Discovery base        ████████████████████████████  18%
Structural Biology base    █████████████████████████     15%
Protein Design base        ████████████████████           12%
Clinical Genomics base     █████████████████              10%
Materials Science base     ████████████████               9%
DD time ceiling            █████████████                  7%
SB time ceiling            ████████████                   6%
PD time ceiling            ███████████                    5%
SB→DD spillover            ██████                         3%
SB→PD spillover            █████                          2%
DD growth rate             ████                           2%
All others                 ███                            <2%
```

**Key Finding**: Base acceleration parameters dominate sensitivity; time evolution and spillovers are secondary.

---

## S5.4 Monte Carlo Uncertainty Propagation

### Method

1. Define distributions for all uncertain parameters
2. Sample 10,000 parameter combinations
3. Run model for each combination
4. Analyze output distribution

### Parameter Distributions

| Parameter | Distribution | Parameters | Rationale |
|-----------|--------------|------------|-----------|
| Base acceleration | Log-normal | μ=log(value), σ=0.15 | Bounded below by 1 |
| Time ceiling | Uniform | ±25% of nominal | Expert uncertainty |
| Growth rate (k) | Normal | μ=value, σ=0.02 | Symmetric uncertainty |
| Spillover coefficients | Beta | α=5, β=15 | Bounded [0, 1], right-skewed |
| Lag years | Uniform | [1, 4] | Literature range |

### Results: Drug Discovery 2030

| Statistic | Value |
|-----------|-------|
| Mean | 1.71x |
| Median | 1.65x |
| 5th percentile | 1.28x |
| 25th percentile | 1.48x |
| 75th percentile | 1.89x |
| 95th percentile | 2.31x |
| Standard deviation | 0.32x |

**Distribution Shape**: Right-skewed (log-normal-like)

### Results: System Acceleration 2030

| Statistic | Value |
|-----------|-------|
| Mean | 2.54x |
| Median | 2.42x |
| 5th percentile | 1.82x |
| 25th percentile | 2.15x |
| 75th percentile | 2.81x |
| 95th percentile | 3.56x |
| Standard deviation | 0.53x |

---

## S5.5 Scenario Comparison

### 2030 Projections by Scenario

| Domain | Pessimistic | Conservative | Baseline | Optimistic | Breakthrough |
|--------|-------------|--------------|----------|------------|--------------|
| Structural Biology | 5.35x | 7.13x | 8.91x | 11.14x | 14.26x |
| Drug Discovery | 1.01x | 1.35x | 1.68x | 2.10x | 2.69x |
| Materials Science | 0.76x | 1.01x | 1.26x | 1.58x | 2.02x |
| Protein Design | 3.28x | 4.38x | 5.47x | 6.84x | 8.76x |
| Clinical Genomics | 2.51x | 3.35x | 4.19x | 5.23x | 6.70x |
| **System** | 1.73x | 2.26x | 2.79x | 3.46x | 4.38x |

### Scenario Range Analysis

| Domain | Min (Pessimistic) | Max (Breakthrough) | Range Factor |
|--------|-------------------|-------------------|--------------|
| Structural Biology | 5.35x | 14.26x | 2.7x |
| Drug Discovery | 1.01x | 2.69x | 2.7x |
| Materials Science | 0.76x | 2.02x | 2.7x |
| Protein Design | 3.28x | 8.76x | 2.7x |
| Clinical Genomics | 2.51x | 6.70x | 2.7x |

**Observation**: All domains show similar ~2.7x range between pessimistic and breakthrough scenarios (by design, as scenario modifiers are uniform across domains).

---

## S5.6 Parameter Importance Ranking

### Sobol-like Decomposition (Total-Order Indices)

| Rank | Parameter | Total-Order Index | Category |
|------|-----------|-------------------|----------|
| 1 | Drug Discovery base | 0.23 | Base parameter |
| 2 | Structural Biology base | 0.19 | Base parameter |
| 3 | Protein Design base | 0.15 | Base parameter |
| 4 | Clinical Genomics base | 0.12 | Base parameter |
| 5 | Materials Science base | 0.11 | Base parameter |
| 6 | Drug Discovery ceiling | 0.06 | Time evolution |
| 7 | Structural Biology ceiling | 0.05 | Time evolution |
| 8 | SB→DD spillover | 0.03 | Spillover |
| 9 | SB→PD spillover | 0.02 | Spillover |
| 10 | All others | <0.02 each | Various |

**Cumulative variance explained**:
- Top 5 parameters (base accelerations): 80%
- Top 10 parameters: 96%
- Remaining parameters: 4%

---

## S5.7 Critical Parameter Thresholds

### Break-even Analysis

"What parameter value would cause system acceleration to reach threshold X?"

| Threshold | Required DD Base | Required SB Base | Required PD Base |
|-----------|------------------|------------------|------------------|
| System = 2.0x | 0.9x | 2.5x | 1.5x |
| System = 3.0x | 1.6x | 5.0x | 3.0x |
| System = 4.0x | 2.4x | 7.5x | 4.5x |

### Breakpoint Identification

| Domain | Current Base | Breakpoint Value | Implication |
|--------|--------------|------------------|-------------|
| Drug Discovery | 1.4x | 0.8x | Below 0.8x → system < 2.0x |
| Structural Biology | 4.5x | 2.0x | Well above breakpoint |
| Materials Science | 1.0x | 0.5x | Already near floor |

---

## S5.8 Time Horizon Sensitivity

### How Sensitivity Changes with Forecast Year

| Year | Base Param Sensitivity | Time Evolution Sensitivity | Spillover Sensitivity |
|------|----------------------|---------------------------|----------------------|
| 2025 | 95% | 4% | 1% |
| 2030 | 78% | 18% | 4% |
| 2035 | 65% | 28% | 7% |
| 2040 | 55% | 36% | 9% |

**Interpretation**:
- Near-term: Base parameters dominate
- Long-term: Time evolution becomes more important
- Spillovers remain secondary throughout

---

## S5.9 Recommendations from Sensitivity Analysis

### Research Priorities

Based on sensitivity analysis, to reduce forecast uncertainty:

| Priority | Action | Uncertainty Reduced |
|----------|--------|-------------------|
| **1. Calibrate base accelerations** | More historical case studies | ~40% of total |
| **2. Validate time evolution** | Track adoption curves over time | ~25% of total |
| **3. Measure spillovers** | Cross-domain impact studies | ~10% of total |
| **4. Scenario probability** | Expert elicitation refinement | ~15% of total |

### Key Uncertainties to Monitor

| Parameter | Current Uncertainty | Monitoring Method |
|-----------|---------------------|-------------------|
| Drug Discovery base | ±0.4x | Track AI drug success rates |
| Structural Biology ceiling | ±5x | Track experimental validation needs |
| Adoption rates (k) | ±0.05 | Technology adoption surveys |
| Scenario probabilities | ±10% | Expert panel updates |

---

## S5.10 Sensitivity Summary

### Key Findings

1. **Base accelerations account for ~80% of output uncertainty**
   - Prioritize calibration and validation
   - More case studies needed

2. **Time evolution parameters are secondary (~15%)**
   - Logistic vs. linear matters less than base values
   - Ceilings matter more for long-term forecasts

3. **Spillovers have modest impact (~5%)**
   - Important for narrative but not dominant
   - Order-of-magnitude precision sufficient

4. **System acceleration is robust**
   - 90% CI spans 1.8x to 3.6x for 2030
   - Directional conclusions stable across scenarios

5. **Drug Discovery dominates system**
   - Highest economic weight (45%)
   - Most sensitive to parameter changes

---

*Table S5 completed: January 2026*
*AI Research Acceleration Model v1.1*
