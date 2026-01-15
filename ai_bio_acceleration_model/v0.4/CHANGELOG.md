# Changelog

All notable changes to the AI-Accelerated Biological Discovery Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

### References
- Saltelli A, Ratto M, Andres T, et al. (2008) "Global Sensitivity Analysis: The Primer" *Wiley*. [DOI: 10.1002/9780470725184](https://doi.org/10.1002/9780470725184)
- Pianosi F, Beven K, Freer J, et al. (2016) "Sensitivity analysis of environmental models: A systematic review with practical workflow" *Environ Model Softw* 79:214-232. [DOI: 10.1016/j.envsoft.2016.02.008](https://doi.org/10.1016/j.envsoft.2016.02.008)

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

- **Scenario M_max overrides significantly expanded:**
  - Pessimistic: Added Wet Lab (3.5x) and Regulatory (1.5x) constraints
  - Optimistic: Added Wet Lab (8.0x), Validation (8.0x) acceleration
  - Optimistic Phase II: 5.0x (biomarker-driven seamless designs)

### Fixed
- Dashboard title now correctly shows "v0.2"
- Bottleneck transitions now occur in Optimistic scenario (Phase II → Phase III in 2027)
- Phase II correctly remains bottleneck in Baseline (valley of death behavior)

### Results (v0.2.1)
| Scenario | Equiv. Years by 2050 |
|----------|---------------------|
| Pessimistic | 43.4 |
| Baseline | 62.9 |
| Optimistic | 98.4 |

### References
- Butler D. (2008) "Translational research: Crossing the valley of death" *Nature* 453:840-842. [DOI: 10.1038/453840a](https://doi.org/10.1038/453840a)
- Arrowsmith J, Miller P. (2013) "Phase II and Phase III attrition rates 2011-2012" *Nat Rev Drug Discov* 12:569. [DOI: 10.1038/nrd4090](https://doi.org/10.1038/nrd4090)
- Stallard N, Todd S, Parashar A, et al. (2019) "On the need to understand benefits and risks of adaptive designs in clinical trials" *Ther Innov Regul Sci* 54:1310-1316. [DOI: 10.1007/s43441-019-00014-2](https://doi.org/10.1007/s43441-019-00014-2)

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

### Added
- Phase-specific success rates from literature
- Phase-specific AI acceleration potentials
- Updated scenario M_max overrides for each clinical phase

### References
- DiMasi JA, Grabowski HG, Hansen RW. (2016) "Innovation in the pharmaceutical industry: New estimates of R&D costs" *J Health Econ* 47:20-33. [DOI: 10.1016/j.jhealeco.2016.01.012](https://doi.org/10.1016/j.jhealeco.2016.01.012)
- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)
- Thomas DW, Burns J, Audette J, et al. (2016) "Clinical Development Success Rates 2006-2015" *BIO Industry Analysis*. [PDF](https://www.bio.org/sites/default/files/legacy/bioorg/docs/Clinical%20Development%20Success%20Rates%202006-2015%20-%20BIO,%20Biomedtracker,%20Amplion%202016.pdf)
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends Pharmacol Sci* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005)
- FDA. (2019) "Adaptive Designs for Clinical Trials of Drugs and Biologics: Guidance for Industry" [PDF](https://www.fda.gov/media/78495/download)

### Expected Impact
- More granular bottleneck analysis possible
- Bottleneck transitions between phases should be visible
- Better policy differentiation (which phase to accelerate?)

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
- Berry SM, Carlin BP, Lee JJ, Muller P. (2010) "Bayesian Adaptive Methods for Clinical Trials" *CRC Press*. [DOI: 10.1201/EBK1439825488](https://doi.org/10.1201/EBK1439825488)
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends Pharmacol Sci* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005)
- FDA. (2019) "Adaptive Designs for Clinical Trials of Drugs and Biologics: Guidance for Industry" [PDF](https://www.fda.gov/media/78495/download)
- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)

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
