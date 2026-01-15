# Claude Desktop Session Context

Copy and paste this at the start of each Claude Desktop session to establish context.

---

## Session Starter (Copy This)

```
I'm working on a research project: AI-Accelerated Biological Discovery Model

PROJECT GOAL:
Quantitative model analyzing how AI accelerates biological science,
identifying bottlenecks in the discovery pipeline.

CURRENT STATUS:
- Version: v0.1 complete (pilot model)
- Next: v0.2 (parameter calibration, split clinical trials)

MODEL OVERVIEW:
- 8-stage pipeline: Hypothesis → Experiment Design → Wet Lab → Analysis 
  → Validation → Clinical Trials → Regulatory → Deployment
- AI multiplier with saturation: M(t) = 1 + (M_max - 1) * (1 - A(t)^(-k))
- System throughput = min of all stage effective rates
- 3 scenarios: Pessimistic (g=0.3), Baseline (g=0.5), Optimistic (g=0.7)

KEY v0.1 FINDING:
Clinical Trials (S6) is persistent bottleneck with M_max=2.5x.
Limited scenario differentiation because bottleneck ceiling constrains all.

ITERATION ROADMAP:
v0.1 ✅ Core framework
v0.2 → Parameter calibration, split clinical into Phase I/II/III
v0.3 → Full scenario analysis
v0.4 → AI feedback loop
v0.5 → Multi-type AI (cognitive/robotic/scientific)
v0.6 → Data quality module
v0.7 → Pipeline iteration/failure
v0.8 → Disease time-to-cure
v0.9 → Policy intervention ROI
v1.0 → Monte Carlo uncertainty

I need help with: [DESCRIBE YOUR TASK HERE]
```

---

## Quick Reference: Key Equations

**AI Capability:** A(t) = exp(g * (t - 2024))

**AI Multiplier:** M_i(t) = 1 + (M_max_i - 1) * (1 - A(t)^(-k_i))

**Service Rate:** μ_i(t) = (12/τ_i) * M_i(t)

**Effective Rate:** μ_i_eff(t) = μ_i(t) * p_i

**Throughput:** Θ(t) = min{μ_i_eff(t)}

**Progress Rate:** R(t) = Θ(t) / Θ(2024)

**Cumulative Progress:** Y(T) = Σ R(t)

---

## Quick Reference: Current Parameters (v0.1)

| Stage | τ (months) | M_max | p | k |
|-------|------------|-------|---|---|
| S1 Hypothesis | 6 | 50 | 0.95 | 1.0 |
| S2 Design | 3 | 20 | 0.90 | 1.0 |
| S3 Wet Lab | 12 | 5 | 0.30 | 0.5 |
| S4 Analysis | 2 | 100 | 0.95 | 1.0 |
| S5 Validation | 8 | 5 | 0.50 | 0.5 |
| S6 Clinical | 72 | 2.5 | 0.12 | 0.3 |
| S7 Regulatory | 12 | 2 | 0.90 | 0.3 |
| S8 Deployment | 12 | 4 | 0.95 | 0.5 |

---

## Common Tasks & Prompts

### Modify Parameters
```
I want to change the parameters for stage S3 (Wet Lab):
- Current: M_max=5, p=0.30, k=0.5
- New: M_max=8, p=0.35, k=0.6

Please update the model.py code for this change.
```

### Add New Stage
```
I want to split S6 (Clinical Trials) into three separate stages:
- S6a: Phase I (12 months, M_max=3, p=0.65)
- S6b: Phase II (24 months, M_max=3, p=0.30)  
- S6c: Phase III (36 months, M_max=2, p=0.60)

Please provide the updated Stage definitions and any necessary code changes.
```

### Analyze Results
```
I ran the model with new parameters. Here are the results:

[PASTE TERMINAL OUTPUT]

Questions:
1. Is the bottleneck transition timing reasonable?
2. Should we adjust any parameters?
3. What does this imply for the paper's conclusions?
```

### Debug Error
```
I got this error when running the model:

[PASTE ERROR MESSAGE]

Here's the code section that might be relevant:

[PASTE CODE]
```

### Add New Feature
```
I want to add a new module for [DESCRIPTION].
Current model structure: [EXPLAIN]
The new feature should: [REQUIREMENTS]

Please provide the implementation.
```
