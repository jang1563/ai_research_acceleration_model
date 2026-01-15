# Integration Plan: Unified AI Biology Acceleration Model

## Overview

This document outlines how to integrate the two parallel model tracks into a unified framework.

---

## Two Model Tracks

### Track A: Pipeline Model (`ai_bio_acceleration_model/v1.1`)

**Focus**: Drug development pipeline stages
**Unit**: 10 stages (S1-S10)
**Strength**: Detailed bottleneck dynamics, therapeutic areas, policy ROI

| Stage | Name | Key Parameters |
|-------|------|----------------|
| S1 | Hypothesis Generation | M_max=50, p=0.40 |
| S2 | Experiment Design | M_max=20, p=0.90 |
| S3 | Wet Lab Execution | M_max=2.5, p=0.30 |
| S4 | Data Analysis | M_max=100, p=0.95 |
| S5 | Validation & Replication | M_max=5, p=0.50 |
| S6 | Phase I Trials | M_max=4, p=0.66 |
| S7 | Phase II Trials | M_max=2.8, p=0.33 (BOTTLENECK) |
| S8 | Phase III Trials | M_max=3.2, p=0.58 |
| S9 | Regulatory Approval | M_max=2, p=0.90, 6-mo floor |
| S10 | Deployment | M_max=4, p=0.95 |

### Track B: Domain Model (`ai_research_acceleration_model/v1.1`)

**Focus**: Research acceleration across scientific domains
**Unit**: 5 domains
**Strength**: Cross-domain spillovers, pipeline discount factors, validation

| Domain | Base Accel | Ceiling | Bottleneck |
|--------|------------|---------|------------|
| Structural Biology | 4.5× | 15× | Experimental validation (30%) |
| Drug Discovery | 1.4× | 4× | Clinical trials (75%) |
| Materials Science | 1.0× | 5× | Synthesis (65%) |
| Protein Design | 2.5× | 10× | Expression validation (45%) |
| Clinical Genomics | 2.0× | 6× | Clinical adoption (50%) |

---

## Integration Architecture

### Unified Model Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED AI BIOLOGY MODEL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │  DOMAIN     │      │  PIPELINE   │      │  DISEASE    │    │
│   │  MODULE     │ ───► │  MODULE     │ ───► │  MODULE     │    │
│   │ (5 domains) │      │ (10 stages) │      │ (outcomes)  │    │
│   └─────────────┘      └─────────────┘      └─────────────┘    │
│         │                    │                    │             │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              SPILLOVER & INTEGRATION LAYER           │      │
│   │  • Cross-domain spillovers                          │      │
│   │  • Domain → Stage mapping                           │      │
│   │  • Pipeline discount factors                        │      │
│   └─────────────────────────────────────────────────────┘      │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              POLICY & OUTPUT LAYER                   │      │
│   │  • Policy ROI calculations                          │      │
│   │  • Workforce projections                            │      │
│   │  • Scenario comparisons                             │      │
│   │  • Uncertainty quantification                       │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Domain-to-Stage Mapping

The key integration insight: **Domains map to subsets of pipeline stages**

| Domain | Primary Stages | Secondary Stages | Notes |
|--------|---------------|------------------|-------|
| **Structural Biology** | S1, S2, S4 | S3, S5 | Hypothesis + Analysis heavy |
| **Drug Discovery** | S6, S7, S8, S9 | All stages | Full pipeline |
| **Materials Science** | S1, S2, S3, S4 | S5 | No clinical stages |
| **Protein Design** | S1, S2, S3, S4, S5 | - | Pre-clinical focus |
| **Clinical Genomics** | S1, S4, S5 | S9 | Diagnostic pathway |

### Mapping Formula

For domain d acceleration at time t:
```
Accel_domain(d, t) = Σ w_ds × Accel_stage(s, t)

where:
- w_ds = weight of stage s in domain d
- Accel_stage(s, t) = M_s(t) × DQM_s(t) × p_s(t) / p_s(0)
```

---

## Integration Components

### 1. Domain Module (from Track B)

**Keep**:
- 5-domain structure
- Cross-domain spillover network
- S-curve time evolution (logistic)
- Economic weighting (OECD)
- Validation framework (15 cases)

**Add from Track A**:
- Multi-type AI (Cognitive/Robotic/Scientific)
- Data quality module D(t)

### 2. Pipeline Module (from Track A)

**Keep**:
- 10-stage structure
- Stage-specific M_max and p_success
- AI type weights per stage
- Therapeutic area parameters
- Regulatory floors

**Add from Track B**:
- Pipeline discount factors
- Domain-level bottleneck fractions

### 3. Disease Module (from Track A)

**Keep**:
- 13 disease profiles
- Time-to-cure calculations
- Patient impact projections
- Therapeutic area modifiers

**Enhance**:
- Link to domain acceleration for indirect effects
- Add spillover benefits

### 4. Policy Module (from Track A)

**Keep**:
- 12 policy interventions
- ROI calculations
- Budget-constrained optimization
- Implementation curves

**Enhance**:
- Domain-specific policy effects
- Spillover-aware ROI

---

## Unified Data Classes

```python
@dataclass
class UnifiedForecast:
    """Combined forecast from both models."""

    # Domain-level (from Track B)
    domain: str
    domain_acceleration: float
    domain_ci_90: Tuple[float, float]
    spillover_boost: float
    pipeline_discount: float

    # Stage-level (from Track A)
    bottleneck_stage: str
    stage_accelerations: Dict[str, float]
    rework_overhead: float

    # Disease-level (combined)
    therapeutic_area: str
    time_to_cure: Optional[float]
    cure_probability: Optional[float]
    expected_beneficiaries: Optional[float]

    # Workforce (from Track B)
    jobs_displaced: float
    jobs_created: float
    net_jobs: float

    # Policy relevance (from Track A)
    policy_recommendations: List[str]
    top_intervention_roi: float
```

---

## Key Integration Equations

### 1. Domain Acceleration with Stage Detail

```
Accel_domain(d, t) = Base_d × TimeEvol_d(t) × (1 + Spillover_d(t)) / PipelineDiscount_d

where:
  TimeEvol_d(t) = 1 + (Ceiling_d - 1) / (1 + exp(-k_d × (t - t0_d)))
  Spillover_d(t) = Σ_j log(1 + Accel_j - 1) × Coef_jd × LagFactor(t)
  PipelineDiscount_d = Σ_s (w_ds × BottleneckFrac_s)
```

### 2. Stage Acceleration with Domain Context

```
Accel_stage(s, t) = M_s(t) × DQM_s(t) × (p_s(t) / p_s_base)

where:
  M_s(t) = 1 + (M_max_s - 1) × (1 - A(t)^(-k_s)) × DomainBoost_s(d)
  DomainBoost_s(d) = 1 + Σ_j Spillover_js  # Spillover to this stage
```

### 3. Unified System Acceleration

```
Accel_system(t) = exp(Σ_d w_d × log(Accel_domain(d, t)))  # Geometric mean

where:
  w_d = OECD economic weight for domain d
```

### 4. Disease Outcome with Both Models

```
TimeToCure(disease, t) = Σ_s (τ_s / Accel_stage(s, t)) × (1 / p_s_eff) × DomainFactor

where:
  DomainFactor = 1 / Accel_domain(primary_domain, t)  # Domain context
```

---

## Implementation Plan

### Phase 1: Core Integration (Week 1-2)

1. **Create unified data structures**
   - Merge ParameterSource with Stage dataclass
   - Add domain-stage mapping
   - Create UnifiedForecast dataclass

2. **Implement domain-stage mapping**
   - Define w_ds weights
   - Implement bi-directional acceleration flow

3. **Merge spillover networks**
   - Domain spillovers (Track B)
   - Stage spillovers (via domain mapping)

### Phase 2: Feature Merge (Week 2-3)

4. **Add Track A features to Track B**
   - Multi-type AI (Cognitive/Robotic/Scientific)
   - Data quality module D(t)
   - Therapeutic area parameters

5. **Add Track B features to Track A**
   - Pipeline discount factors
   - Prospective validation framework
   - Economic weighting

### Phase 3: Policy Integration (Week 3-4)

6. **Unified policy module**
   - Domain-aware interventions
   - Stage-specific effects
   - Spillover-aware ROI

7. **Disease model enhancement**
   - Link to domain acceleration
   - Cross-domain cure pathways

### Phase 4: Validation & Documentation (Week 4)

8. **Validation**
   - Cross-validate predictions
   - Check consistency between levels
   - Expand validation cases

9. **Documentation**
   - Unified supplementary tables
   - Integration methodology document
   - Updated figures

---

## File Structure for Unified Model

```
ai_unified_acceleration_model/
├── v2.0/
│   ├── src/
│   │   ├── unified_model.py          # Main integrated model
│   │   ├── domain_module.py          # Domain-level (from Track B)
│   │   ├── pipeline_module.py        # Stage-level (from Track A)
│   │   ├── disease_module.py         # Disease outcomes
│   │   ├── policy_module.py          # Policy ROI
│   │   ├── spillover_module.py       # Cross-domain/stage effects
│   │   ├── uncertainty_module.py     # Monte Carlo, Sobol
│   │   └── validation_module.py      # Historical validation
│   │
│   ├── supplementary/
│   │   ├── TABLE_S1_PARAMETERS.md    # All parameters (merged)
│   │   ├── TABLE_S2_SPILLOVERS.md    # Spillover network
│   │   ├── TABLE_S3_VALIDATION.md    # Validation cases
│   │   ├── TABLE_S4_DOMAIN_STAGE_MAP.md  # Domain-stage mapping
│   │   ├── TABLE_S5_DISEASE_PROFILES.md  # Disease models
│   │   └── TABLE_S6_POLICY_ROI.md    # Policy interventions
│   │
│   ├── figures/
│   │   └── [publication figures]
│   │
│   ├── README.md
│   ├── METHODOLOGY.md
│   └── VALIDATION.md
```

---

## Benefits of Integration

| Benefit | Description |
|---------|-------------|
| **Comprehensiveness** | Domain + Stage + Disease + Policy in one model |
| **Consistency** | Single set of parameters, no contradictions |
| **Cross-validation** | Compare domain vs stage predictions |
| **Policy depth** | Domain effects + Stage mechanisms + ROI |
| **Storytelling** | Multiple levels of insight for different audiences |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complexity explosion | Modular design, clear interfaces |
| Parameter conflicts | Use Track B as primary, Track A as detail |
| Validation difficulty | Maintain separate validation at each level |
| Over-engineering | Start with minimum viable integration |

---

## Decision: Start Integration?

**Recommended Approach**:

1. **For Manuscript**: Use Track B (v1.1) as-is
   - Already manuscript-ready
   - Simpler story for publication
   - Can reference Track A as "detailed pipeline model available"

2. **For Comprehensive Tool**: Integrate both
   - Create v2.0 unified model
   - More powerful for policy analysis
   - Better for stakeholder tools

**Suggested Path**:
- Submit manuscript with Track B (v1.1)
- Develop v2.0 integrated model in parallel
- Publish second paper on unified framework

---

## Quick Win: Add Policy ROI to v1.1

If full integration is too much, we can add just the policy ROI framework from Track A:

```python
# Add to ai_acceleration_model.py

@dataclass
class PolicyIntervention:
    name: str
    cost_annual: float  # $M/year
    duration_years: int
    domain_effects: Dict[str, float]  # Multiplier by domain
    evidence_quality: int  # 1-5

INTERVENTIONS = [
    PolicyIntervention("Adaptive Trial Expansion", 200, 5,
                       {"drug_discovery": 1.15}, 4),
    PolicyIntervention("Cryo-EM Infrastructure", 500, 10,
                       {"structural_biology": 1.20}, 5),
    PolicyIntervention("Synthesis Automation", 300, 5,
                       {"materials_science": 1.30}, 3),
    # ...
]

def calculate_policy_roi(intervention, model, year=2030):
    """Calculate ROI for a policy intervention."""
    baseline = model.system_snapshot(year)

    # Apply intervention effects
    enhanced_model = model.copy()
    for domain, mult in intervention.domain_effects.items():
        enhanced_model.BASE_PARAMETERS[domain].value *= mult

    enhanced = enhanced_model.system_snapshot(year)

    delta_accel = enhanced.total_acceleration - baseline.total_acceleration
    delta_workforce = enhanced.workforce_change - baseline.workforce_change

    total_cost = intervention.cost_annual * intervention.duration_years

    # Simple ROI: delta acceleration per $B
    roi = delta_accel / (total_cost / 1000)

    return {
        "intervention": intervention.name,
        "cost_total": total_cost,
        "delta_acceleration": delta_accel,
        "delta_workforce": delta_workforce,
        "roi": roi,
    }
```

---

## Conclusion

**Integration is possible and valuable**, but should be approached strategically:

1. **Short-term**: Publish v1.1 Track B as-is (domain model)
2. **Medium-term**: Add policy ROI from Track A
3. **Long-term**: Build v2.0 unified model

The two models are **complementary, not contradictory**:
- Track A answers "How do pipeline stages limit drug development?"
- Track B answers "How does AI accelerate research across domains?"
- Together: "How does AI reshape biology research and what should we do about it?"

---

*Integration Plan completed: January 2026*
