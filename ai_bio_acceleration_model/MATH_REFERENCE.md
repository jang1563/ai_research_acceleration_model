# Mathematical Reference Sheet

> Quick reference for all equations, parameters, and derivations.

---

## 1. CORE EQUATIONS

### 1.1 AI Capability Growth

```
A(t) = exp(g × (t - t₀))
```

Where:
- `A(t)` = AI capability at time t (dimensionless, normalized)
- `g` = growth rate (year⁻¹)
- `t₀` = 2024 (baseline year)

**Properties:**
- A(2024) = 1
- A(2030) = exp(6g) ≈ 20x for g=0.5
- A(2050) = exp(26g) ≈ 400,000x for g=0.5

### 1.2 AI Acceleration Multiplier

```
M_i(t) = 1 + (M_max_i - 1) × (1 - A(t)^(-k_i))
```

Where:
- `M_i(t)` = acceleration multiplier for stage i
- `M_max_i` = maximum possible acceleration (saturation limit)
- `k_i` = saturation rate (how fast it approaches max)

**Properties:**
- M_i(t₀) = 1 (no acceleration at baseline)
- M_i(∞) → M_max_i (full saturation)
- Higher k → faster saturation

**Saturation behavior at A=10 (≈2029 for g=0.5):**
- k=0.3: M reaches ~45% of (M_max - 1)
- k=0.5: M reaches ~68% of (M_max - 1)
- k=1.0: M reaches ~90% of (M_max - 1)

### 1.3 Service Rate

```
μ_i(t) = μ_i⁰ × M_i(t)
```

Where:
- `μ_i(t)` = service rate (projects/year)
- `μ_i⁰` = baseline service rate = 12 / τ_i⁰

### 1.4 Effective Service Rate

```
μ_i_eff(t) = μ_i(t) × p_i
```

Where:
- `p_i` = success probability for stage i

### 1.5 System Throughput

```
Θ(t) = min{μ_i_eff(t)} for all i
```

The bottleneck stage is:
```
i*(t) = argmin{μ_i_eff(t)}
```

### 1.6 Progress Rate

```
R(t) = Θ(t) / Θ(t₀)
```

**Interpretation:** R(t) = 2 means science progresses 2x faster than baseline.

### 1.7 Cumulative Progress

```
Y(T) = Σ R(t) × Δt  for t = t₀ to T-1
```

**Interpretation:** Y(T) = 50 means we've achieved 50 equivalent years of progress.

---

## 2. PARAMETER VALUES (v0.1)

### 2.1 Stage Parameters

| Stage | Name | τ⁰ (mo) | μ⁰ | M_max | p | k |
|-------|------|---------|-----|-------|-----|-----|
| 1 | Hypothesis | 6 | 2.0 | 50 | 0.95 | 1.0 |
| 2 | Design | 3 | 4.0 | 20 | 0.90 | 1.0 |
| 3 | Wet Lab | 12 | 1.0 | 5 | 0.30 | 0.5 |
| 4 | Analysis | 2 | 6.0 | 100 | 0.95 | 1.0 |
| 5 | Validation | 8 | 1.5 | 5 | 0.50 | 0.5 |
| 6 | Clinical | 72 | 0.167 | 2.5 | 0.12 | 0.3 |
| 7 | Regulatory | 12 | 1.0 | 2 | 0.90 | 0.3 |
| 8 | Deployment | 12 | 1.0 | 4 | 0.95 | 0.5 |

### 2.2 Computed Baseline Values

| Stage | μ⁰ | p | μ_eff⁰ |
|-------|-----|-----|--------|
| 1 | 2.0 | 0.95 | 1.90 |
| 2 | 4.0 | 0.90 | 3.60 |
| 3 | 1.0 | 0.30 | 0.30 |
| 4 | 6.0 | 0.95 | 5.70 |
| 5 | 1.5 | 0.50 | 0.75 |
| 6 | 0.167 | 0.12 | **0.02** ← Bottleneck |
| 7 | 1.0 | 0.90 | 0.90 |
| 8 | 1.0 | 0.95 | 0.95 |

### 2.3 Scenario Parameters

| Scenario | g |
|----------|---|
| Pessimistic | 0.30 |
| Baseline | 0.50 |
| Optimistic | 0.70 |

---

## 3. PLANNED EXTENSIONS

### 3.1 AI Feedback Loop (v0.4)

```
g(t) = g₀ + (g_max - g₀) × (1 - exp(-α × Y(t)))
```

Or differential form:
```
dg/dt = α × (g_max - g(t)) × (A(t) - 1)
```

Parameters:
- α = 0.10 [0.05, 0.20]: feedback strength
- g_max = 0.80 [0.60, 1.00]: maximum growth rate

### 3.2 Multi-Type AI (v0.5)

Three capability types with different growth rates:

| Type | Symbol | g | Description |
|------|--------|---|-------------|
| Cognitive | A_c | 0.60 | Reasoning, synthesis |
| Robotic | A_r | 0.30 | Physical manipulation |
| Scientific | A_s | 0.50 | Hypothesis generation |

Stage-to-type mapping:
- S1, S5: Scientific
- S2, S4, S7: Cognitive
- S3, S6, S8: Robotic

### 3.3 Data Quality Module (v0.6)

```
D(t) = D(t₀) × exp(δ × (t - t₀))
```

Data quality multiplier:
```
DQM_i(t) = (D(t) / D(t₀))^ε_i
```

Where ε_i is elasticity (0.1 to 0.5 depending on stage).

### 3.4 Pipeline Iteration (v0.7)

Effective throughput with rework:
```
μ_i_eff(t) = μ_i(t) × p_i × (1 / (1 + Σ q_ij × (1-p_j)))
```

Where q_ij = probability that failure at j sends work back to i.

### 3.5 Time-to-Cure (v0.8)

For disease d:
```
E[T_cure_d] = (1/π_d) × Σ τ_k(t)  for k = s_d to n
```

Where:
- s_d = starting stage for disease d
- π_d = success probability per attempt
- τ_k(t) = 1/μ_k(t) = expected time at stage k

---

## 4. DERIVATIONS

### 4.1 Saturation Function

**Goal:** Find f(A) such that M = 1 + (M_max - 1) × f(A) has:
1. f(1) = 0
2. f(∞) = 1
3. f'(A) > 0
4. f''(A) < 0

**Solution:** f(A) = 1 - A^(-k)

**Proof:**
1. f(1) = 1 - 1^(-k) = 1 - 1 = 0 ✓
2. lim_{A→∞} f(A) = 1 - 0 = 1 ✓
3. f'(A) = k × A^(-(k+1)) > 0 for A > 0, k > 0 ✓
4. f''(A) = -k(k+1) × A^(-(k+2)) < 0 for A > 0, k > 0 ✓

### 4.2 Bottleneck as Rate Limiter

For sequential pipeline, steady-state throughput equals minimum stage capacity.

**Proof (Little's Law):**
- Work-in-progress at stage i: W_i = λ / μ_i
- If λ > μ_i for any i, queue grows unboundedly
- Therefore, sustainable λ ≤ min{μ_i}
- Throughput = min{μ_i_eff}

### 4.3 Progress Accumulation

If R(t) is piecewise constant over Δt = 1 year:
```
Y(T) = Σ R(t) × 1 = Σ R(t)
```

For continuous R(t):
```
Y(T) = ∫_{t₀}^{T} R(t) dt
```

---

## 5. NUMERICAL EXAMPLES

### 5.1 Baseline Scenario at 2035

```
t = 2035
g = 0.50
A(2035) = exp(0.50 × 11) = exp(5.5) ≈ 245

For S6 (Clinical Trials):
M_6(2035) = 1 + (2.5 - 1) × (1 - 245^(-0.3))
         = 1 + 1.5 × (1 - 0.197)
         = 1 + 1.5 × 0.803
         = 1 + 1.20
         = 2.20

μ_6(2035) = 0.167 × 2.20 = 0.37 projects/year
μ_6_eff(2035) = 0.37 × 0.12 = 0.044 projects/year
```

### 5.2 Comparing Stages at 2040

| Stage | M(2040) | μ(2040) | μ_eff(2040) |
|-------|---------|---------|-------------|
| S1 | 49.5 | 99.0 | 94.1 |
| S2 | 19.8 | 79.2 | 71.3 |
| S3 | 4.7 | 4.7 | 1.4 |
| S4 | 99.5 | 597 | 567 |
| S5 | 4.7 | 7.1 | 3.5 |
| S6 | 2.3 | 0.39 | **0.047** |
| S7 | 1.9 | 1.9 | 1.7 |
| S8 | 3.7 | 3.7 | 3.5 |

**Bottleneck:** Still S6 at 0.047 projects/year

---

## 6. UNIT CONVERSIONS

| From | To | Formula |
|------|-----|---------|
| Duration (months) | Service rate (proj/year) | μ = 12 / τ |
| Service rate | Duration | τ = 12 / μ |
| Growth rate (year⁻¹) | Doubling time (years) | T_d = ln(2) / g |
| Doubling time | Growth rate | g = ln(2) / T_d |

**Common doubling times:**
- g = 0.30 → T_d = 2.3 years
- g = 0.50 → T_d = 1.4 years
- g = 0.70 → T_d = 1.0 years

---

## 7. QUICK FORMULAS FOR CLAUDE

### Check if parameters make sense:

```python
# Baseline effective rate
mu_eff_baseline = (12 / tau_baseline) * p_success

# Should be positive and reasonable (0.01 to 10 typically)

# Time to saturation (80% of max)
# Solve: 1 - A^(-k) = 0.8
# A = 5^(1/k)
# t_80 = t0 + ln(5^(1/k)) / g = t0 + ln(5) / (k × g)

t_80_percent = 2024 + 1.61 / (k * g)  # years to reach 80% saturation
```

### Check bottleneck:

```python
# At any time t, bottleneck is stage with minimum mu_eff
# mu_eff_i(t) = (12 / tau_i) * M_i(t) * p_i

# If S6 always bottleneck, it means:
# (12/72) * M_6_max * 0.12 < other stages' minimums
# 0.167 * 2.5 * 0.12 = 0.05 < other minimums
```

---

*This reference sheet accompanies PROJECT_BIBLE.md*
