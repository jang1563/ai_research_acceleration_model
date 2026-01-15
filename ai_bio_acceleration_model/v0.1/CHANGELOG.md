# Changelog

All notable changes to the AI-Accelerated Biological Discovery Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.1] - 2026-01-13

### Changed
- **S6 Clinical Trials parameters revised** based on critical review:
  - M_max: 2.5x → 3.0x (adaptive trial designs, AI patient selection)
  - p_success: 0.12 → 0.15 (AI-improved biomarker selection)
  - k_saturation: 0.3 → 0.4 (faster adoption with proven benefits)

- **Scenario-specific M_max overrides** implemented:
  - Pessimistic: S6 M_max = 2.5x (conservative, institutional resistance)
  - Baseline: S6 M_max = 3.0x (default, moderate adoption)
  - Optimistic: S6 M_max = 4.0x (regulatory reform, adaptive trials)

### Added
- Scenario dataclass now supports `M_max_overrides` dictionary
- Better scenario differentiation for policy analysis

### References
- Adaptive trial acceleration: Berry et al. (2016) "Bayesian Adaptive Methods for Clinical Trials"
- AI patient selection: Harrer et al. (2019) "Artificial Intelligence for Clinical Trial Design"
- Regulatory reform potential: FDA (2019) "Adaptive Designs for Clinical Trials"
- Success rate improvements: Wong et al. (2019) "Estimation of clinical trial success rates"

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

- **Scenarios**
  - Pessimistic (g=0.30)
  - Baseline (g=0.50)
  - Optimistic (g=0.70)

- **Visualization Suite**
  - Figure 1: AI Capability Growth
  - Figure 2: Stage-Specific AI Multipliers
  - Figure 3: Effective Service Rates
  - Figure 4: Bottleneck Timeline
  - Figure 5: Progress Rate
  - Figure 6: Cumulative Progress
  - Summary Dashboard

- **Documentation**
  - Technical Specification (mathematical formulation)
  - README with quick start guide
  - API documentation

- **Data Export**
  - Results CSV (all scenarios, all time points)
  - Parameters JSON (model configuration)
  - Summary TXT (key statistics)

### Key Results (v0.1)
- Baseline scenario: ~57 equivalent years by 2050 (2.2x acceleration)
- S6 (Clinical Trials) identified as persistent bottleneck
- All three scenarios show similar results due to bottleneck constraint

### Known Issues
- S6 dominates as bottleneck throughout; may need parameter adjustment
- Limited scenario differentiation suggests bottleneck M_max drives results
- Minor matplotlib warning in dashboard generation

### Technical Notes
- Python 3.x required
- Dependencies: numpy, pandas, matplotlib
- Tested on Ubuntu 24

---

## Planned for [0.2]

### To Add
- Literature-grounded parameter calibration
- Source citations for all M_max values
- Improved scenario differentiation

### To Fix
- Dashboard tick label warning
- Bottleneck dominance analysis

---

## Planned for [0.3]

### To Add
- Full scenario analysis with parameter sweeps
- Sensitivity analysis on key parameters
- Confidence intervals on outputs

---

## Planned for [0.4]

### To Add
- AI feedback loop (recursive improvement)
- Dynamic growth rates
- Stability analysis

---

## Version Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| 0.1 | Core framework | ✅ Complete |
| 0.2 | Parameter calibration | 🔲 Planned |
| 0.3 | Scenario analysis | 🔲 Planned |
| 0.4 | AI feedback loop | 🔲 Planned |
| 0.5 | Multi-type AI | 🔲 Planned |
| 0.6 | Data quality module | 🔲 Planned |
| 0.7 | Pipeline iteration | 🔲 Planned |
| 0.8 | Disease models | 🔲 Planned |
| 0.9 | Policy analysis | 🔲 Planned |
| 1.0 | Uncertainty quantification | 🔲 Planned |
