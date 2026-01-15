# Iteration Planning Document

> Detailed specifications for each of the 10 planned iterations.

---

## ITERATION OVERVIEW

```
v0.1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅ COMPLETE
v0.2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 NEXT
v0.3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.7 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.8 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v0.9 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Planned
v1.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔲 Final
```

---

## v0.1: CORE FRAMEWORK ✅ COMPLETE

### What Was Built
- 8-stage pipeline model
- Single AI capability type with exponential growth
- Saturation-based AI multiplier
- System throughput as minimum of effective rates
- 3 scenarios (pessimistic/baseline/optimistic)
- 7 publication-quality figures
- Technical specification document

### Key Results
- Baseline: 56.9 equivalent years by 2050 (2.2x)
- S6 (Clinical Trials) is persistent bottleneck
- Limited scenario differentiation

### Issues Identified
- S6 too dominant (no bottleneck transitions)
- May need to split clinical trials
- M_max values need literature grounding

### Files Delivered
```
v0.1/
├── src/model.py
├── src/visualize.py
├── docs/TECHNICAL_SPECIFICATION.md
├── docs/FINDINGS_v0.1.md
├── outputs/results.csv
├── outputs/parameters.json
└── outputs/*.png, *.pdf (7 figures)
```

---

## v0.2: PARAMETER CALIBRATION 🔲 NEXT

### Objectives
1. Split clinical trials into Phase I/II/III
2. Ground all parameters in literature
3. Improve scenario differentiation
4. Add source citations

### Specific Changes

#### 1. Split S6 into Three Stages

**Current (v0.1):**
```
S6: Clinical Trials
  τ = 72 months
  M_max = 2.5
  p = 0.12
  k = 0.3
```

**New (v0.2):**
```
S6a: Phase I (Safety/Dose-Finding)
  τ = 12 months
  M_max = 3.0    # Adaptive designs, AI-optimized dosing
  p = 0.65       # Historical Phase I success rate
  k = 0.3

S6b: Phase II (Proof of Concept)
  τ = 24 months
  M_max = 3.0    # Better biomarker selection
  p = 0.30       # Historical Phase II success rate
  k = 0.3

S6c: Phase III (Pivotal)
  τ = 36 months
  M_max = 2.0    # Limited by statistical requirements
  p = 0.60       # Historical Phase III success rate
  k = 0.2        # Slowest to saturate
```

#### 2. Literature Sources to Add

| Parameter | Source | Value |
|-----------|--------|-------|
| Phase I duration | Wong et al. 2019 | 12-18 months |
| Phase I success | BIO Industry Analysis 2021 | 63.2% |
| Phase II duration | DiMasi et al. 2016 | 24-30 months |
| Phase II success | BIO Industry Analysis 2021 | 30.7% |
| Phase III duration | FDA analysis | 30-42 months |
| Phase III success | BIO Industry Analysis 2021 | 58.1% |
| AI growth rate | Epoch AI 2024 | 0.4-0.8 year⁻¹ |

#### 3. Code Changes Required

**In model.py:**
```python
# Update Stage list to have 10 stages instead of 8
# Renumber S7 → S8, S8 → S9, etc.
# Add S6a, S6b, S6c

# Update n_stages references
# Update bottleneck naming
```

**In visualize.py:**
```python
# Update STAGE_COLORS (need 10 colors)
# Update figure layouts
# Update labels
```

### Expected Outcomes
- More bottleneck transitions (S6a → S6b → S6c likely)
- Better scenario differentiation
- More realistic dynamics

### Validation Criteria
- [ ] Model runs without errors
- [ ] Baseline results plausible (40-80 equiv years by 2050)
- [ ] At least one bottleneck transition occurs
- [ ] All parameters have citations

### Deliverables
```
v0.2/
├── src/model.py (updated)
├── src/visualize.py (updated)
├── docs/TECHNICAL_SPECIFICATION.md (updated)
├── docs/FINDINGS_v0.2.md (new)
├── docs/PARAMETER_SOURCES.md (new)
├── outputs/results.csv
└── outputs/*.png, *.pdf
```

---

## v0.3: SCENARIO ANALYSIS 🔲 Planned

### Objectives
1. Full parameter sensitivity analysis
2. Scenario-specific M_max values
3. Confidence intervals on outputs
4. Identify key uncertainty drivers

### Specific Changes

#### 1. Expanded Scenarios

| Parameter | Pessimistic | Baseline | Optimistic |
|-----------|-------------|----------|------------|
| g | 0.30 | 0.50 | 0.70 |
| M_max (S6) | 2.0 | 2.5 | 4.0 |
| M_max (S3) | 3 | 5 | 10 |
| p (S6b) | 0.25 | 0.30 | 0.40 |

#### 2. Sensitivity Analysis

Compute ∂Y/∂θ for each parameter θ:
- AI growth rate (g)
- Each M_max value
- Each success probability
- Each baseline duration

#### 3. Parameter Sweeps

```python
# Sweep g from 0.2 to 0.8 in 0.1 increments
# Sweep M_max_S6 from 1.5 to 5.0 in 0.5 increments
# Generate heatmap of Y(2050) vs (g, M_max_S6)
```

### Expected Outcomes
- Clear identification of most important parameters
- Better uncertainty characterization
- More dramatic scenario differentiation

### Deliverables
- Tornado diagram (parameter sensitivity)
- 2D heatmaps (parameter interactions)
- Expanded results tables

---

## v0.4: AI FEEDBACK LOOP 🔲 Planned

### Objectives
1. Make AI growth rate time-varying
2. Model AI-accelerated AI research
3. Ensure stability (no explosion)

### Mathematical Addition

**Dynamic growth rate:**
```
g(t) = g₀ + (g_max - g₀) × (1 - exp(-α × Y(t)))
```

**Parameters:**
- g₀ = initial growth rate (scenario-dependent)
- g_max = maximum growth rate (0.8 to 1.0)
- α = feedback strength (0.05 to 0.15)

### Implementation

```python
def compute_growth_rate(self, Y_cumulative, g0, g_max, alpha):
    """Dynamic growth rate with feedback."""
    return g0 + (g_max - g0) * (1 - np.exp(-alpha * Y_cumulative))

# In main loop:
for t in time_points:
    g_t = self.compute_growth_rate(Y[t-1], g0, g_max, alpha)
    A[t] = A[t-1] * np.exp(g_t * dt)
    # ... rest of computation
```

### Stability Analysis
- Verify g(t) ≤ g_max for all t
- Verify A(t) remains finite
- Compare with and without feedback

---

## v0.5: MULTI-TYPE AI 🔲 Planned

### Objectives
1. Decompose AI into cognitive/robotic/scientific
2. Different growth rates per type
3. Map stages to AI types

### Three AI Types

| Type | Symbol | g | Description | Relevant Stages |
|------|--------|---|-------------|-----------------|
| Cognitive | A_c | 0.60 | Reasoning, synthesis | S2, S4, S7 |
| Robotic | A_r | 0.30 | Physical manipulation | S3, S6a-c, S8 |
| Scientific | A_s | 0.50 | Hypothesis generation | S1, S5 |

### Implementation

```python
@dataclass
class AICapability:
    cognitive: float
    robotic: float
    scientific: float

def ai_capability(self, t, scenario):
    return AICapability(
        cognitive=np.exp(scenario.g_cognitive * (t - t0)),
        robotic=np.exp(scenario.g_robotic * (t - t0)),
        scientific=np.exp(scenario.g_scientific * (t - t0)),
    )
```

### Expected Impact
- Robotic-dependent stages (S3, S6) lag behind
- Cognitive stages accelerate fastest
- More realistic bottleneck dynamics

---

## v0.6: DATA QUALITY MODULE 🔲 Planned

### Objectives
1. Add data quality as cross-cutting factor
2. Model AI impact on data quality
3. Data quality affects stage effectiveness

### Mathematical Addition

**Data quality growth:**
```
D(t) = D(t₀) × exp(δ × (t - t₀))
```

**Data quality multiplier:**
```
DQM_i(t) = (D(t) / D(t₀))^ε_i
```

**Updated service rate:**
```
μ_i(t) = μ_i⁰ × M_i(t) × DQM_i(t)
```

### Parameters

| Stage | ε_i (elasticity) | Rationale |
|-------|------------------|-----------|
| S1 | 0.3 | Better data → better hypotheses |
| S2 | 0.2 | Design less data-dependent |
| S3 | 0.1 | Execution is physical |
| S4 | 0.4 | Analysis quality depends on input |
| S5 | 0.3 | Replication needs good data |
| S6 | 0.2 | Clinical protocols standardized |
| S7 | 0.2 | Regulatory review quality |
| S8 | 0.1 | Deployment mostly operational |

---

## v0.7: PIPELINE ITERATION 🔲 Planned

### Objectives
1. Model failure and rework
2. Projects can return to earlier stages
3. AI can improve success rates

### Mathematical Addition

**Failure routing matrix Q:**
```
Q[i,j] = probability failure at i returns to j
```

**Example for S6b (Phase II failure):**
```
Q[6b, 1] = 0.1   # Return to hypothesis
Q[6b, 2] = 0.1   # Return to design
Q[6b, 3] = 0.3   # Return to wet lab
Q[6b, abandon] = 0.5  # Project abandoned
```

**Effective throughput:**
```
Θ_eff = Θ × Π(p_i) × (1 + rework_factor)^(-1)
```

---

## v0.8: DISEASE MODELS 🔲 Planned

### Objectives
1. Disease-specific time-to-cure
2. Case studies: Cancer, Alzheimer's, Pandemic
3. Probability distributions

### Disease Parameters

| Disease | Start Stage | Advances Needed | Success/Advance |
|---------|-------------|-----------------|-----------------|
| Cancer (general) | S5 | 4 | 0.3 |
| Alzheimer's | S3 | 6 | 0.2 |
| Pandemic (next) | S1 | 2 | 0.7 |

### Implementation

```python
@dataclass
class Disease:
    name: str
    start_stage: int
    advances_needed: int
    success_per_advance: float

def time_to_cure(self, disease, scenario):
    """Compute expected time to cure."""
    # ...
```

---

## v0.9: POLICY ANALYSIS 🔲 Planned

### Objectives
1. Define intervention effects
2. Compute ROI per intervention
3. Rank by value per dollar
4. Timing recommendations

### Interventions

| Intervention | Cost ($B) | Effect |
|--------------|-----------|--------|
| Lab automation | 50 | M_max_S3: 5 → 10 |
| Clinical trial reform | 10 | τ_S6: -30% |
| Open data mandates | 5 | δ: +50% |
| AI R&D funding | 100 | g: +20% |
| Regulatory expansion | 20 | μ_S7: +50% |

### ROI Computation

```
ROI_I = (Y(2050|I) - Y(2050|baseline)) / cost_I
```

---

## v1.0: UNCERTAINTY QUANTIFICATION 🔲 Final

### Objectives
1. Full Monte Carlo simulation
2. Parameter distributions
3. Sobol sensitivity indices
4. Publication-ready uncertainty

### Implementation

```python
def monte_carlo(self, n_samples=10000):
    results = []
    for _ in range(n_samples):
        params = self.sample_parameters()
        model = AIBioAccelerationModel(params)
        result = model.run()
        results.append(result)
    return self.compute_statistics(results)
```

### Outputs
- 80% confidence intervals
- Sobol indices for each parameter
- Distribution plots for key outputs

---

## ITERATION CHECKLIST TEMPLATE

For each iteration, verify:

```
□ Code runs without errors
□ Unit tests pass (if applicable)
□ Results are plausible
□ Figures generate correctly
□ Documentation updated
□ FINDINGS document created
□ CHANGELOG updated
□ Git committed
□ Zip created
```

---

*This document guides the 10-iteration development process.*
