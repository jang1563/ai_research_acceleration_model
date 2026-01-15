# Implementation Plan: v0.5 → v0.6+
## Based on 6-Expert Review Panel Feedback

**Created:** January 13, 2026
**Status:** Ready for Implementation

---

## EXECUTIVE SUMMARY

The 6-expert panel identified **24 improvements** across three categories:
- **Communication (Rachel Kim):** 5 recommendations
- **Visualization (David Nakamura):** 7 recommendations
- **Model Enhancements (Returning 4 experts):** 12 recommendations

### Prioritization Matrix

```
                    HIGH IMPACT
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
     │   T1.1, T1.5     │   T1.2, T1.4     │
     │   Quick Wins     │   Strategic      │
     │                  │   Priorities     │
LOW ─┼──────────────────┼──────────────────┼─ HIGH
EFFORT│                 │                  │  EFFORT
     │   T4.1-T4.4      │   T3.1-T3.5      │
     │   Nice-to-Have   │   Long-term      │
     │                  │   Investments    │
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
                    LOW IMPACT
```

---

## PHASE 1: COMMUNICATION QUICK WINS (v0.5.1)
**Effort:** 1 day | **Impact:** HIGH | **Dependencies:** None

### 1.1 Add Outcome Translations

**Current Problem:**
> "93.5 equivalent years" is meaningless to policymakers

**Implementation:**
```python
# Add to summary outputs
def translate_progress_to_outcomes(equiv_years, calendar_years=26):
    """Convert equivalent years to tangible outcomes."""
    acceleration_factor = equiv_years / calendar_years

    # Baseline: ~50 novel therapies approved 2024-2050 at current pace
    baseline_therapies = 50
    projected_therapies = int(baseline_therapies * acceleration_factor * 0.7)  # Conservative

    return {
        'acceleration_factor': acceleration_factor,
        'projected_therapies': projected_therapies,
        'vs_baseline': projected_therapies - baseline_therapies
    }
```

**Deliverables:**
- [ ] Add `outcomes.py` module with translation functions
- [ ] Update `summary.txt` with outcome translations
- [ ] Add callout boxes to key figures

### 1.2 Reframe Uncertainty Communication

**Current:** "90% CI: [70, 115]"
**Better:** "We're 90% confident acceleration will be between 2.7x and 4.4x"

**Implementation:**
- [ ] Add `format_uncertainty_for_policy()` function
- [ ] Create probabilistic statement templates
- [ ] Update all CI displays in outputs

### 1.3 Create Glossary

**Key Terms to Define:**
| Term | Plain English |
|------|---------------|
| Equivalent years | How much progress compared to 2024 pace |
| Bottleneck | The slowest step limiting overall speed |
| Service rate | How fast each step processes projects |
| M_max | Maximum possible AI speedup for each step |

**Deliverable:**
- [ ] Add `GLOSSARY.md` to outputs folder

---

## PHASE 2: VISUALIZATION OVERHAUL (v0.5.2)
**Effort:** 2-3 days | **Impact:** HIGH | **Dependencies:** Phase 1

### 2.1 Hero Figure Design

**Specification:**
```python
def create_hero_figure(results):
    """
    Single most important visualization.

    Design principles:
    - One clear message
    - Minimal cognitive load
    - Self-explanatory annotations
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline trajectory only (prominence)
    baseline = results[results['scenario'] == 'Baseline']
    ax.plot(baseline['year'], baseline['progress_rate'],
            linewidth=3, color='#1f77b4', label='Expected trajectory')

    # Uncertainty band (subtle)
    ax.fill_between(baseline['year'],
                    baseline['progress_rate_p10'],
                    baseline['progress_rate_p90'],
                    alpha=0.2, color='#1f77b4')

    # Key milestone annotations
    ax.annotate('2x faster\nby 2035', xy=(2035, 2.0),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('3x faster\nby 2043', xy=(2043, 3.0),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Clean axis labels
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Speed of Discovery\n(1x = 2024 pace)', fontsize=14)
    ax.set_title('AI-Accelerated Biological Discovery: Expected Trajectory',
                 fontsize=16, fontweight='bold')

    return fig
```

### 2.2 Colorblind-Safe Palette

**Replace:**
```python
# OLD (problematic)
COLORS = {'pessimistic': 'blue', 'baseline': 'green', 'optimistic': 'red'}

# NEW (accessible)
COLORS = {
    'pessimistic': '#4575b4',  # Blue
    'baseline': '#fdae61',      # Orange
    'optimistic': '#d73027',    # Red-orange
    'general': '#1a1a1a',       # Dark gray
    'oncology': '#91bfdb',      # Light blue
    'cns': '#fc8d59',           # Coral
    'infectious': '#91cf60',    # Green
    'rare': '#d9ef8b'           # Yellow-green
}
```

### 2.3 Executive Summary Figure (New)

**2-Panel Design:**
```
┌─────────────────────────────────────────────────────────┐
│ [LEFT PANEL: Progress Over Time]  [RIGHT: By Area]      │
│                                                         │
│  Speed │      ╭──────                   Oncology █████  │
│   4x   │     ╱                          Infectious ████ │
│   3x   │   ╱         ← 3x by 2045       Rare Dis. ███   │
│   2x   │ ╱                              General ███     │
│   1x   │──────                          CNS ██          │
│        └──────────────────              ──────────────  │
│        2024  2030  2040  2050           Progress 2050   │
│                                                         │
│ Key Insight: Physical bottlenecks limit AI acceleration │
│ but Oncology benefits most from AI-driven biomarkers    │
└─────────────────────────────────────────────────────────┘
```

### 2.4 Figure Hierarchy Implementation

**Tier 1 - Hero Figures (main paper):**
- `fig_hero_progress.png` - Progress trajectory with annotations
- `fig_executive_summary.png` - 2-panel overview

**Tier 2 - Supporting Figures (paper):**
- `fig_bottleneck_timeline.png` - Simplified, baseline only
- `fig_therapeutic_comparison.png` - Slope chart redesign
- `fig_ai_types.png` - Multi-type AI explanation

**Tier 3 - Technical Figures (supplementary):**
- `fig_tornado.png` - Sensitivity analysis
- `fig_uncertainty_bands.png` - Monte Carlo results
- `fig_service_rates.png` - Detailed stage analysis
- `summary_dashboard.png` - All metrics combined

### 2.5 Annotation Strategy

**Every figure should answer:**
1. What am I looking at? (title)
2. What's the key finding? (annotation)
3. Why does it matter? (callout)

**Example for Bottleneck Timeline:**
```python
# Add annotation
ax.annotate(
    'Phase II remains bottleneck\nuntil mid-2040s',
    xy=(2042, 7), xytext=(2035, 8.5),
    fontsize=11, ha='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
)
```

---

## PHASE 3: MODEL ENHANCEMENT - DATA QUALITY (v0.6)
**Effort:** 3-5 days | **Impact:** HIGH | **Dependencies:** Phases 1-2

### 3.1 Data Quality Module D(t)

**Mathematical Framework:**
```python
class DataQualityModule:
    """
    Models how data quality improves over time and affects all stages.

    D(t) = D_0 * (1 + gamma * log(A(t)))

    Where:
    - D_0 = 1.0 (normalized baseline)
    - gamma = data quality growth coefficient
    - A(t) = AI capability

    Effect on stages:
    DQM_i(t) = (D(t) / D_0)^epsilon_i

    Where epsilon_i is stage-specific data quality elasticity
    """

    def __init__(self, gamma=0.15):
        self.gamma = gamma
        self.D_0 = 1.0

        # Stage-specific elasticities
        self.elasticities = {
            1: 0.8,   # Hypothesis generation - high dependence
            2: 0.6,   # Experiment design
            3: 0.3,   # Wet lab - moderate
            4: 0.9,   # Data analysis - highest
            5: 0.5,   # Validation
            6: 0.4,   # Phase I
            7: 0.6,   # Phase II - trial design benefits
            8: 0.5,   # Phase III
            9: 0.2,   # Regulatory - low
            10: 0.3   # Deployment
        }

    def D(self, t, A_t):
        """Data quality at time t."""
        return self.D_0 * (1 + self.gamma * np.log(A_t))

    def data_quality_multiplier(self, stage_idx, t, A_t):
        """Multiplier for stage throughput due to data quality."""
        D_t = self.D(t, A_t)
        epsilon = self.elasticities.get(stage_idx, 0.5)
        return (D_t / self.D_0) ** epsilon
```

### 3.2 Integration with Main Model

```python
# In AIBioAccelerationModel.compute_stage_trajectory():

# After computing M_i(t), apply data quality multiplier
if self.config.enable_data_quality:
    dq_mult = self.data_quality.data_quality_multiplier(
        stage.index, t, A_eff
    )
    M_adjusted = M * dq_mult
```

### 3.3 New Outputs for v0.6

- `fig_data_quality.png` - D(t) trajectory
- `fig_dq_impact.png` - Data quality effect by stage
- Updated sensitivity analysis including gamma parameter

---

## PHASE 4: ADVANCED FEATURES (v0.7-0.8)
**Effort:** 1-2 weeks | **Impact:** MEDIUM-HIGH

### 4.1 AI Type Correlation (v0.7)

**Current:** Independent sampling of g_c, g_r, g_s
**Improved:** Correlated sampling with correlation matrix

```python
# Correlation structure
CORRELATION_MATRIX = np.array([
    [1.0, 0.3, 0.7],   # Cognitive correlates with Scientific
    [0.3, 1.0, 0.2],   # Robotic less correlated
    [0.7, 0.2, 1.0]    # Scientific correlates with Cognitive
])

def sample_correlated_growth_rates(n_samples, base_rates, cv):
    """Sample g_c, g_r, g_s with correlation structure."""
    from scipy.stats import multivariate_normal

    means = list(base_rates.values())
    stds = [r * cv for r in means]

    # Convert correlation to covariance
    cov = np.outer(stds, stds) * CORRELATION_MATRIX

    samples = multivariate_normal.rvs(mean=means, cov=cov, size=n_samples)
    return samples
```

### 4.2 Geographic Regulatory Variation (v0.7)

**Three Regulatory Scenarios:**
| Region | Phase I M_max | Phase II M_max | Phase III M_max | Approval M_max |
|--------|---------------|----------------|-----------------|----------------|
| US FDA | 4.0 | 2.8 | 3.2 | 2.0 |
| EU EMA | 3.5 | 2.5 | 2.8 | 2.5 |
| China NMPA | 4.5 | 3.0 | 3.5 | 3.0 |

### 4.3 Pipeline Iteration (v0.8)

**Rework Probabilities:**
```python
REWORK_MATRIX = {
    # From stage -> To stage: probability
    (3, 2): 0.15,   # Wet lab failure → redesign experiment
    (5, 3): 0.20,   # Validation failure → redo wet lab
    (7, 6): 0.25,   # Phase II failure → back to Phase I redesign
    (8, 7): 0.15,   # Phase III failure → redesign Phase II
}
```

---

## PHASE 5: PAPER FINALIZATION (v0.9-1.0)
**Effort:** 2-3 weeks | **Impact:** CRITICAL for publication

### 5.1 Policy Analysis Module (v0.9)

**Interventions to Model:**
1. Increase wet lab automation funding (+$1B)
2. Regulatory reform (adaptive trials)
3. Data sharing mandates
4. AI safety framework
5. Clinical trial infrastructure

**ROI Calculation:**
```python
def compute_intervention_roi(intervention, cost, baseline_progress):
    """
    Compute return on investment for policy intervention.

    ROI = (ΔProgress * value_per_year) / cost
    """
    modified_progress = run_model_with_intervention(intervention)
    delta_progress = modified_progress - baseline_progress

    # Assume $50B value per equivalent year of progress
    value = delta_progress * 50e9
    roi = value / cost

    return roi
```

### 5.2 Full Uncertainty Quantification (v1.0)

**Monte Carlo Specifications:**
- N = 10,000 samples (up from 500)
- Sobol sensitivity indices
- Parameter uncertainty propagation
- Scenario probability weights

---

## TIMELINE SUMMARY

```
Week 1:  Phase 1 (Communication) + Phase 2 Start (Visualization)
Week 2:  Phase 2 Complete + Phase 3 Start (Data Quality)
Week 3:  Phase 3 Complete + Review
Week 4:  Phase 4.1 (AI Correlation)
Week 5:  Phase 4.2-4.3 (Geographic, Iteration)
Week 6:  Phase 5.1 (Policy Analysis)
Week 7:  Phase 5.2 (Full UQ)
Week 8:  Paper Writing + Figures
Week 9:  Internal Review
Week 10: Submission to bioRxiv
```

---

## ACCEPTANCE CRITERIA

### v0.5.1 (Communication)
- [ ] All figures have plain-English annotations
- [ ] Outcome translations in summary.txt
- [ ] Glossary created
- [ ] Uncertainty reframed for policymakers

### v0.5.2 (Visualization)
- [ ] Hero figure created and prominently positioned
- [ ] Colorblind-safe palette implemented
- [ ] Executive summary figure created
- [ ] Figure hierarchy documented
- [ ] All fonts ≥10pt

### v0.6 (Data Quality)
- [ ] D(t) module implemented
- [ ] Elasticity parameters calibrated
- [ ] Sensitivity analysis updated
- [ ] New visualization for data quality impact

### v0.7+ (Advanced)
- [ ] AI correlation structure in Monte Carlo
- [ ] Geographic scenarios implemented
- [ ] Rework probabilities added
- [ ] Policy ROI rankings computed

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Visualization redesign takes longer | Medium | Low | Prioritize hero figure, defer others |
| Data quality module adds complexity | Medium | Medium | Keep elasticities simple initially |
| Expert feedback conflicts | Low | Medium | Defer to communication experts for audience concerns |
| Parameter calibration for new modules | High | Medium | Use literature defaults, sensitivity analysis |

---

*Plan created: January 13, 2026*
*Next review: After Phase 2 completion*
