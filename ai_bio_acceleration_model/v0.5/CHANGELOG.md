# Changelog

All notable changes to the AI-Accelerated Biological Discovery Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5] - 2026-01-13

### Major Changes
- **Multi-Type AI Differentiation**: Model now distinguishes between three AI types
  - Cognitive AI (g=0.60): Language models, reasoning, hypothesis generation
  - Robotic AI (g=0.30): Lab automation, physical experiments
  - Scientific AI (g=0.55): AlphaFold-type specialized scientific models

- **Therapeutic Area Modeling**: Area-specific success rates and acceleration factors
  - Oncology: p_success modifier 1.25x (biomarker-driven)
  - CNS: p_success modifier 0.65x (complex mechanisms)
  - Infectious Disease: p_success modifier 1.40x (clear targets)
  - Rare Disease: p_success modifier 0.95x (regulatory advantages)

- **Stage-Specific AI Weights**: Each stage has weighted contributions from AI types
  - `ai_type_weights` parameter defines Cognitive/Robotic/Scientific mix
  - Wet Lab (S3): 90% Robotic, 10% Cognitive
  - Hypothesis (S1): 80% Cognitive, 20% Scientific
  - Data Analysis (S4): 50% Cognitive, 50% Scientific

### Added
- `TherapeuticArea` enum in model.py
- `AIType` enum in model.py
- `compare_therapeutic_areas()` function
- `compare_ai_types()` function
- `get_therapeutic_area_comparison()` method
- Therapeutic area-specific scenario generation
- Stage attribute `therapeutic_sensitivity` for area modulation
- Improved visualizations from v0.4.2 (tornado, fan chart, heatmap, dashboard)

### Key Findings

**Therapeutic Area Impact (Baseline, 2050):**
| Area | Equivalent Years | Relative to Baseline |
|------|-----------------|---------------------|
| Infectious Disease | 72.3 | +15% |
| Oncology | 68.5 | +9% |
| Rare Disease | 61.2 | -3% |
| CNS | 48.7 | -23% |

**AI Type Contributions:**
- Cognitive AI dominates early stages (hypothesis, design, analysis)
- Robotic AI is critical bottleneck in wet lab and validation stages
- Scientific AI provides specialized acceleration in technical stages
- By 2050, Cognitive AI reaches 3,814x capability vs Robotic at 62x

### References
- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)
- Epoch AI. (2024) "AI Trends" [https://epoch.ai/trends](https://epoch.ai/trends)
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends Pharmacol Sci* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005)

---

## [0.4.2] - 2026-01-13

### Improved Visualizations
- **Tornado Diagram**: Bidirectional bars with proper baseline reference
- **Combined Fan Chart**: All scenarios with 50%/90% confidence intervals
- **Bottleneck Heatmap**: Time × stage constraint matrix
- **Summary Dashboard**: Four informative panels replacing uninformative Panel D

### Added
- `visualize_improved.py` module with publication-quality figures
- `generate_improved_figures.py` script
- Fixed column name compatibility (sensitivity_index → sensitivity, p50 → median)

### Fixed
- Tornado diagram negative elasticity labels confusion
- Summary dashboard Panel D (was solid block, now shows heatmap)
- Overlapping fills in cumulative progress charts
- Version comparison endpoint label overlap

---

## [0.4] - 2026-01-13

### Major Changes
- **Monte Carlo Uncertainty Quantification**: Full uncertainty analysis
  - 500+ samples per scenario
  - 90% confidence intervals on all projections
  - Parameter uncertainty propagation

### Added
- `uncertainty.py` module for Monte Carlo simulations
- Confidence interval CSV exports
- Uncertainty band visualizations
- Histogram plots for 2050 projections

### Results with Uncertainty (90% CI)
| Scenario | 2050 Median | 90% CI |
|----------|------------|--------|
| Pessimistic | 43.4 yr | [34.7, 51.1] |
| Baseline | 62.9 yr | [48.6, 76.7] |
| Optimistic | 98.4 yr | [71.6, 120.1] |

---

## [0.3] - 2026-01-13

### Major Changes
- **Full Sensitivity Analysis Module** (`sensitivity.py`):
  - One-at-a-time (OAT) sensitivity analysis
  - Parameter sweep functionality
  - Tornado diagrams for parameter importance ranking
  - Automatic identification of highest-leverage parameters

### Key Findings from Sensitivity Analysis

| Parameter | Sensitivity Index | Elasticity | Policy Implication |
|-----------|-------------------|------------|-------------------|
| S7_M_max (Phase II) | **0.811** | 0.89 | Highest leverage - target Phase II automation |
| g_ai | 0.427 | 0.15 | AI R&D investment has strong returns |
| S8_M_max (Phase III) | 0.260 | 0.00 | Secondary target after Phase II |
| S7_p_success | 0.181 | 0.00 | Improve Phase II success with biomarkers |
| S8_p_success | 0.181 | -0.07 | Improve Phase III methods |

### Policy Recommendations
1. **Priority 1:** Develop AI tools for Phase II trials (biomarker selection, adaptive designs)
2. **Priority 2:** Invest in fundamental AI research (g_ai growth)
3. **Priority 3:** Focus on Phase III acceleration after Phase II is resolved

### Added
- `src/sensitivity.py` - Complete sensitivity analysis module
- `outputs/sensitivity_summary.csv` - Parameter rankings
- `outputs/fig_tornado.png` - Visual parameter importance
- Command-line argument `--skip-sensitivity` for faster runs

---

## [0.2.1] - 2026-01-13

### Changed
- **Phase II parameters tuned for realistic "valley of death" behavior:**
  - M_max: 3.5x → 2.8x (hardest phase to accelerate)
  - p_success: 0.31 → 0.33 (combined rate ~12.6%)
  - k_saturation: 0.4 → 0.3 (slow adoption due to high stakes)

- **Phase III parameters adjusted for regulatory reform scenarios:**
  - M_max: 2.5x → 3.2x (adaptive/seamless trial designs)
  - k_saturation: 0.3 → 0.4 (growing regulatory acceptance)

### Results (v0.2.1)
| Scenario | Equiv. Years by 2050 |
|----------|---------------------|
| Pessimistic | 43.4 |
| Baseline | 62.9 |
| Optimistic | 98.4 |

---

## [0.2] - 2026-01-13

### Major Changes
- **Clinical Trials split into three phases:**
  - S6: Phase I Trials (12 mo, M_max=4.0x, p=0.66)
  - S7: Phase II Trials (24 mo, M_max=3.5x, p=0.31)
  - S8: Phase III Trials (36 mo, M_max=2.5x, p=0.58)

- **Pipeline expanded from 8 to 10 stages:**
  - S1-S5: Same as v0.1 (Hypothesis → Validation)
  - S6-S8: Split clinical trial phases (NEW)
  - S9: Regulatory Approval (was S7)
  - S10: Deployment (was S8)

### References
- DiMasi et al. (2016) "Innovation in the pharmaceutical industry"
- Wong et al. (2019) "Estimation of clinical trial success rates"
- Thomas et al. (2016) "Clinical Development Success Rates"

---

## [0.1] - 2026-01-13

### Added
- **Core Model Framework**
  - 8-stage scientific pipeline (hypothesis → deployment)
  - AI capability growth with exponential dynamics
  - AI acceleration multipliers with saturation
  - System throughput as minimum of stage capacities
  - Bottleneck identification algorithm
  - Progress rate and cumulative progress calculations

- **Pipeline Stages**
  - S1: Hypothesis Generation (M_max=50x)
  - S2: Experiment Design (M_max=20x)
  - S3: Wet Lab Execution (M_max=5x)
  - S4: Data Analysis (M_max=100x)
  - S5: Validation & Replication (M_max=5x)
  - S6: Clinical Trials (M_max=2.5x)
  - S7: Regulatory Approval (M_max=2x)
  - S8: Deployment (M_max=4x)

- **Visualization Suite**
  - 7 publication-quality figures
  - Summary dashboard

### Key Results (v0.1)
- Baseline scenario: ~57 equivalent years by 2050 (2.2x acceleration)
- S6 (Clinical Trials) identified as persistent bottleneck

---

## Version Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| 0.1 | Core framework | ✅ Complete |
| 0.2 | Parameter calibration | ✅ Complete |
| 0.3 | Sensitivity analysis | ✅ Complete |
| 0.4 | Monte Carlo uncertainty | ✅ Complete |
| 0.5 | Multi-type AI + Therapeutic areas | ✅ Complete |
| 0.6 | Data quality module | 🔲 Planned |
| 0.7 | Pipeline iteration | 🔲 Planned |
| 0.8 | Disease models | 🔲 Planned |
| 0.9 | Policy analysis | 🔲 Planned |
| 1.0 | Final paper version | 🔲 Planned |
