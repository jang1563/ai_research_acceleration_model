# Technical Specification: AI-Accelerated Biological Discovery Model

**Version:** 0.1 (Pilot Model)  
**Date:** January 2026  
**Status:** Development

---

## 1. Overview

This document provides the complete mathematical specification for v0.1 of the AI-Accelerated Biological Discovery Model. This pilot version implements a minimal viable model to validate the core framework before adding complexity in subsequent iterations.

### 1.1 Model Purpose

The model quantifies how AI acceleration affects the end-to-end biological discovery pipeline, identifying:

1. **Rate-limiting bottlenecks** at each point in time
2. **System throughput** accounting for multi-stage constraints
3. **Cumulative progress** in equivalent years of scientific advancement

### 1.2 Version 0.1 Scope

| Feature | v0.1 Status |
|---------|-------------|
| Pipeline stages | 8 stages (simplified) |
| AI capability types | 1 (aggregate) |
| AI growth dynamics | Constant exponential |
| Iteration/failure | Simple success probabilities |
| Data quality module | Deferred |
| Uncertainty quantification | 3 scenarios only |

---

## 2. Notation

### 2.1 Core Symbols

| Symbol | Description | Units |
|--------|-------------|-------|
| $t$ | Time | years |
| $t_0$ | Baseline year (2024) | years |
| $T$ | Horizon year (2050) | years |
| $i$ | Stage index | $i \in \{1, 2, ..., 8\}$ |
| $A(t)$ | AI capability | dimensionless |
| $g$ | AI growth rate | year⁻¹ |
| $M_i(t)$ | AI multiplier for stage $i$ | dimensionless |
| $M_i^{\max}$ | Maximum AI multiplier for stage $i$ | dimensionless |
| $k_i$ | Saturation rate for stage $i$ | dimensionless |
| $\tau_i^0$ | Baseline duration of stage $i$ | months |
| $\mu_i^0$ | Baseline service rate of stage $i$ | projects/year |
| $\mu_i(t)$ | Service rate of stage $i$ at time $t$ | projects/year |
| $p_i$ | Success probability for stage $i$ | probability |
| $\mu_i^{\text{eff}}(t)$ | Effective service rate | projects/year |
| $\Theta(t)$ | System throughput | projects/year |
| $i^*(t)$ | Bottleneck stage | index |
| $R(t)$ | Progress rate | dimensionless |
| $Y(t)$ | Cumulative progress | equivalent years |

---

## 3. Mathematical Formulation

### 3.1 AI Capability Growth

AI capability grows exponentially from a normalized baseline:

$$A(t) = \exp(g \cdot (t - t_0))$$

**Properties:**
- $A(t_0) = 1$ (normalized baseline)
- $A(t)$ grows without bound as $t \to \infty$
- $g$ determines the growth rate (scenario-dependent)

**Parameter values by scenario:**

| Scenario | $g$ (year⁻¹) | Interpretation |
|----------|--------------|----------------|
| Pessimistic | 0.30 | ~35% capability increase per year |
| Baseline | 0.50 | ~65% capability increase per year |
| Optimistic | 0.70 | ~100% capability increase per year |

### 3.2 AI Acceleration Multiplier

The AI multiplier for stage $i$ saturates as capability grows:

$$M_i(t) = 1 + (M_i^{\max} - 1) \cdot \left(1 - A(t)^{-k_i}\right)$$

**Properties:**
- At $t = t_0$: $A = 1$, so $M_i = 1$ (no acceleration)
- As $t \to \infty$: $A \to \infty$, so $M_i \to M_i^{\max}$ (saturation)
- $k_i$ controls the rate of approach to saturation

**Derivation of saturation form:**

We want a function that:
1. Equals 1 when $A = 1$
2. Approaches $M^{\max}$ as $A \to \infty$
3. Is monotonically increasing
4. Has a smooth, diminishing-returns shape

The form $1 - A^{-k}$ satisfies:
- At $A = 1$: $1 - 1^{-k} = 0$ ✓
- As $A \to \infty$: $1 - 0 = 1$ ✓
- Monotonically increasing for $k > 0$ ✓

### 3.3 Service Rate

The service rate of stage $i$ is accelerated by the AI multiplier:

$$\mu_i(t) = \mu_i^0 \cdot M_i(t)$$

Where the baseline service rate is derived from baseline duration:

$$\mu_i^0 = \frac{12}{\tau_i^0}$$

**Units:** If $\tau_i^0$ is in months, then $\mu_i^0$ is in projects per year.

### 3.4 Effective Service Rate

The effective service rate accounts for stage success probability:

$$\mu_i^{\text{eff}}(t) = \mu_i(t) \cdot p_i$$

**Interpretation:** This represents the rate at which projects *successfully complete* stage $i$, accounting for failures that don't proceed.

### 3.5 System Throughput

System throughput is the minimum effective service rate across all stages:

$$\Theta(t) = \min_{i \in \{1, ..., 8\}} \mu_i^{\text{eff}}(t)$$

**Rationale:** In a sequential pipeline, the overall flow rate is limited by the slowest stage (bottleneck).

### 3.6 Bottleneck Identification

The bottleneck stage at time $t$ is:

$$i^*(t) = \underset{i}{\arg\min} \, \mu_i^{\text{eff}}(t)$$

### 3.7 Progress Rate

The progress rate measures acceleration relative to baseline:

$$R(t) = \frac{\Theta(t)}{\Theta(t_0)}$$

**Properties:**
- $R(t_0) = 1$ by definition
- $R(t) > 1$ indicates acceleration
- $R(t) \leq M_{i^*}^{\max}$ (bounded by bottleneck saturation)

### 3.8 Cumulative Progress

Cumulative equivalent years of progress:

$$Y(T) = \sum_{t=t_0}^{T-\Delta t} R(t) \cdot \Delta t$$

For continuous formulation (with $\Delta t = 1$ year):

$$Y(T) = \int_{t_0}^{T} R(t) \, dt$$

**Interpretation:** By year $T$, we have achieved $Y(T)$ years worth of scientific progress (at baseline rates), in $T - t_0$ calendar years.

---

## 4. Pipeline Definition

### 4.1 Stage Specifications

| Stage | Name | $\tau_i^0$ (mo) | $\mu_i^0$ (proj/yr) | $M_i^{\max}$ | $p_i$ | $k_i$ |
|-------|------|----------------|---------------------|--------------|-------|-------|
| S1 | Hypothesis Generation | 6 | 2.0 | 50 | 0.95 | 1.0 |
| S2 | Experiment Design | 3 | 4.0 | 20 | 0.90 | 1.0 |
| S3 | Wet Lab Execution | 12 | 1.0 | 5 | 0.30 | 0.5 |
| S4 | Data Analysis | 2 | 6.0 | 100 | 0.95 | 1.0 |
| S5 | Validation & Replication | 8 | 1.5 | 5 | 0.50 | 0.5 |
| S6 | Clinical Trials | 72 | 0.167 | 2.5 | 0.12 | 0.3 |
| S7 | Regulatory Approval | 12 | 1.0 | 2 | 0.90 | 0.3 |
| S8 | Deployment | 12 | 1.0 | 4 | 0.95 | 0.5 |

### 4.2 Parameter Justifications

**S1 - Hypothesis Generation (M_max = 50x):**
- Primarily cognitive task (literature synthesis, pattern recognition)
- AlphaFold demonstrated >1000x speedup for protein structure
- Bounded by need to verify hypothesis quality

**S2 - Experiment Design (M_max = 20x):**
- AI can rapidly generate experimental protocols
- Bounded by feasibility checking and domain expertise

**S3 - Wet Lab Execution (M_max = 5x):**
- Cell division ~24 hours (irreducible)
- Mouse studies require weeks (biological timescales)
- Gains come from parallelization and automation

**S4 - Data Analysis (M_max = 100x):**
- Pure computation; AI excels
- Bounded by human interpretation needs

**S5 - Validation & Replication (M_max = 5x):**
- Peer review: social process (months)
- Replication requires physical time
- AI can accelerate but not eliminate

**S6 - Clinical Trials (M_max = 2.5x):**
- Human metabolism sets floor
- Phase III requires statistical power (sample size)
- Adaptive trial designs offer modest gains

**S7 - Regulatory Approval (M_max = 2x):**
- Institutional capacity constraints
- Human decision-making required
- Political/legal factors

**S8 - Deployment (M_max = 4x):**
- Manufacturing automation possible
- Logistics optimization
- Healthcare system adoption friction

### 4.3 Success Probability Derivations

**S3 - Wet Lab (p = 0.30):**
- Historical experiment failure rates ~70%
- Most hypotheses don't survive wet lab test

**S5 - Validation (p = 0.50):**
- Replication crisis: ~50% of studies don't replicate
- Publication bias compounds issues

**S6 - Clinical Trials (p = 0.12):**
- Phase I success: ~65%
- Phase II success: ~30%
- Phase III success: ~60%
- Combined: 0.65 × 0.30 × 0.60 ≈ 0.12

---

## 5. Computational Implementation

### 5.1 Algorithm

```
Input: Configuration (t0, T, dt, stages, scenarios)
Output: Results dataframe with all computed values

For each scenario:
    1. Initialize time array: t = [t0, t0+dt, ..., T]
    2. Compute AI capability: A(t) = exp(g * (t - t0))
    
    For each stage i:
        3. Compute AI multiplier: M_i(t)
        4. Compute service rate: μ_i(t) = μ_i^0 * M_i(t)
        5. Compute effective rate: μ_i^eff(t) = μ_i(t) * p_i
    
    6. Compute throughput: Θ(t) = min_i {μ_i^eff(t)}
    7. Identify bottleneck: i*(t) = argmin_i {μ_i^eff(t)}
    8. Compute progress rate: R(t) = Θ(t) / Θ(t0)
    9. Compute cumulative progress: Y(t) = cumsum(R * dt)
    
    10. Store results in dataframe
```

### 5.2 Numerical Considerations

- **Time step:** $\Delta t = 1$ year (sufficient for this model)
- **Precision:** Standard float64 precision
- **Overflow:** AI capability grows large but multiplier is bounded
- **Stability:** No feedback loops in v0.1; stable by construction

---

## 6. Model Outputs

### 6.1 Primary Outputs

| Output | Symbol | Description |
|--------|--------|-------------|
| Progress Rate | $R(t)$ | Acceleration factor at time $t$ |
| Cumulative Progress | $Y(T)$ | Equivalent years by horizon |
| Bottleneck Stage | $i^*(t)$ | Rate-limiting stage at time $t$ |

### 6.2 Secondary Outputs

| Output | Description |
|--------|-------------|
| AI Capability Trajectory | $A(t)$ over time |
| Stage-Specific Multipliers | $M_i(t)$ for each stage |
| Effective Service Rates | $\mu_i^{\text{eff}}(t)$ for each stage |
| Bottleneck Transitions | When $i^*$ changes |

---

## 7. Limitations and Future Work

### 7.1 Known Limitations (v0.1)

1. **Single AI capability type:** Does not distinguish cognitive/robotic/scientific
2. **Constant growth rate:** No feedback from AI-accelerated AI research
3. **No iteration/feedback:** Pipeline is strictly sequential
4. **No data quality module:** Data as bottleneck not explicitly modeled
5. **Point estimates only:** No uncertainty quantification beyond scenarios

### 7.2 Planned Improvements

| Version | Focus |
|---------|-------|
| v0.2 | Parameter calibration with literature sources |
| v0.3 | Full scenario analysis |
| v0.4 | AI feedback loop |
| v0.5 | Multi-type AI (cognitive/robotic/scientific) |
| v0.6 | Data quality module |
| v0.7 | Pipeline iteration/failure |
| v0.8 | Disease-specific time-to-cure |
| v0.9 | Policy intervention analysis |
| v1.0 | Monte Carlo uncertainty quantification |

---

## 8. References

1. Amodei, D. (2024). "Machines of Loving Grace." Anthropic Blog.
2. DeepMind. (2024). "A New Golden Age of Discovery." Nature.
3. Epoch AI. (2024). "AI Trends." epoch.ai/trends
4. FDA Annual Reports. Clinical trial statistics.
5. ClinicalTrials.gov. Trial success rate data.

---

## Appendix A: Derivation of Saturation Function

We seek a function $f(A)$ such that the AI multiplier $M = 1 + (M^{\max} - 1) \cdot f(A)$ satisfies:

1. $f(1) = 0$ (no acceleration at baseline)
2. $f(\infty) = 1$ (full saturation as AI → ∞)
3. $f'(A) > 0$ for all $A > 0$ (monotonically increasing)
4. $f''(A) < 0$ for all $A > 0$ (diminishing returns)

**Candidate:** $f(A) = 1 - A^{-k}$ for $k > 0$

**Verification:**
1. $f(1) = 1 - 1^{-k} = 1 - 1 = 0$ ✓
2. $\lim_{A \to \infty} f(A) = 1 - 0 = 1$ ✓
3. $f'(A) = k \cdot A^{-(k+1)} > 0$ for $A > 0$, $k > 0$ ✓
4. $f''(A) = -k(k+1) \cdot A^{-(k+2)} < 0$ for $A > 0$, $k > 0$ ✓

**Role of $k$:**
- Larger $k$ → faster approach to saturation
- At $A = 2$: $f(2) = 1 - 2^{-k}$
  - $k = 0.5$: $f(2) = 0.29$ (slow saturation)
  - $k = 1.0$: $f(2) = 0.50$ (moderate saturation)
  - $k = 2.0$: $f(2) = 0.75$ (fast saturation)

---

## Appendix B: Baseline Bottleneck Analysis

At $t = t_0$, all $M_i(t_0) = 1$, so effective service rates are:

$$\mu_i^{\text{eff}}(t_0) = \mu_i^0 \cdot p_i$$

| Stage | $\mu_i^0$ | $p_i$ | $\mu_i^{\text{eff}}(t_0)$ |
|-------|-----------|-------|--------------------------|
| S1 | 2.0 | 0.95 | 1.90 |
| S2 | 4.0 | 0.90 | 3.60 |
| S3 | 1.0 | 0.30 | **0.30** |
| S4 | 6.0 | 0.95 | 5.70 |
| S5 | 1.5 | 0.50 | 0.75 |
| S6 | 0.167 | 0.12 | **0.02** |
| S7 | 1.0 | 0.90 | 0.90 |
| S8 | 1.0 | 0.95 | 0.95 |

**Initial bottleneck:** S6 (Clinical Trials) with $\mu_6^{\text{eff}} = 0.02$

This reflects the reality that clinical trials are the primary bottleneck in drug development.
