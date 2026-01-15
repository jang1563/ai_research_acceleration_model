# AI-Accelerated Biological Discovery Model

**Version 0.5 (Multi-Type AI + Therapeutic Areas)**

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated biological research and drug development.

---

## Overview

This model quantifies how AI capabilities affect the end-to-end biological discovery pipeline, from hypothesis generation through drug deployment. Version 0.5 introduces **multi-type AI** differentiation and **therapeutic area-specific** modeling.

### Key Features (v0.5)

- **10-stage scientific pipeline** modeling the full discovery-to-deployment process
- **Multi-type AI differentiation:**
  - Cognitive AI (g=0.60): Language models, reasoning, hypothesis generation
  - Robotic AI (g=0.30): Lab automation, physical experiments
  - Scientific AI (g=0.55): AlphaFold-type specialized scientific models
- **Therapeutic area modeling:**
  - Oncology (higher success rates, biomarker-driven)
  - CNS (lower success rates, complex mechanisms)
  - Infectious Disease (fastest success, clear targets)
  - Rare Disease (small populations, regulatory incentives)
- **Stage-specific AI contributions** based on task composition
- **Monte Carlo uncertainty quantification** (500+ samples per scenario)
- **Sensitivity analysis** with tornado diagrams
- **Publication-quality visualizations** (improved in v0.4.2)

### Model Outputs

1. **Progress Rate** - How much faster science advances relative to 2024
2. **Cumulative Progress** - Equivalent years of scientific advancement
3. **Bottleneck Timeline** - Which stage limits progress over time
4. **Therapeutic Area Comparison** - Progress by disease category
5. **AI Type Contributions** - Role of cognitive vs robotic vs scientific AI

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n ai_bio_acceleration_model python=3.9 numpy pandas matplotlib scipy
conda activate ai_bio_acceleration_model
```

### Run the Model

```bash
cd v0.5
python run_model.py
```

### Command-line Options

```bash
# Full run (sensitivity + Monte Carlo)
python run_model.py

# Skip sensitivity analysis (faster)
python run_model.py --skip-sensitivity

# Skip Monte Carlo (faster)
python run_model.py --skip-monte-carlo

# Custom Monte Carlo samples
python run_model.py --mc-samples 1000
```

### Expected Output

```
Key Results (Baseline Scenario):
  By 2030: 11.2 equiv. years (6 calendar -> 1.9x)
  By 2040: 35.7 equiv. years (16 calendar -> 2.2x)
  By 2050: 62.9 equiv. years (26 calendar -> 2.4x)

Therapeutic Area Comparison (Baseline, 2050):
  Infectious Disease   :   72.3 years
  Oncology             :   68.5 years
  Rare Disease         :   61.2 years
  CNS                  :   48.7 years

AI Type Capabilities (2050, Baseline):
  Cognitive (g=0.60):  3814x
  Robotic (g=0.30):    62x
  Scientific (g=0.55): 1808x
```

---

## What's New in v0.5

### Multi-Type AI Model

Different AI types progress at different rates and contribute differently to each stage:

| AI Type | Growth Rate | Primary Contribution |
|---------|-------------|---------------------|
| Cognitive | 60%/year | Hypothesis, design, analysis |
| Robotic | 30%/year | Wet lab, validation, manufacturing |
| Scientific | 55%/year | Specialized scientific models |

Each stage has a weighted mix of AI types (defined by `ai_type_weights`).

### Therapeutic Area Differentiation

Success rates and AI acceleration potential vary by therapeutic area:

| Area | Phase II p_success | Phase III p_success | Notes |
|------|-------------------|--------------------|----|
| Oncology | 0.39 | 0.65 | Biomarker-driven, precision medicine |
| CNS | 0.15 | 0.50 | Blood-brain barrier, complex mechanisms |
| Infectious | 0.45 | 0.75 | Clear targets, measurable endpoints |
| Rare Disease | 0.30 | 0.60 | Small populations, regulatory advantages |

---

## Model Structure

### Pipeline Stages (v0.5)

| Stage | Description | Duration | Max AI Speedup | AI Type Weights |
|-------|-------------|----------|----------------|-----------------|
| S1 | Hypothesis Generation | 6 mo | 50x | Cog: 0.8, Sci: 0.2 |
| S2 | Experiment Design | 3 mo | 20x | Cog: 0.6, Sci: 0.4 |
| S3 | Wet Lab Execution | 12 mo | 5x | Rob: 0.9, Cog: 0.1 |
| S4 | Data Analysis | 2 mo | 100x | Cog: 0.5, Sci: 0.5 |
| S5 | Validation & Replication | 8 mo | 5x | Rob: 0.7, Cog: 0.3 |
| S6 | Phase I Trials | 12 mo | 4x | Rob: 0.4, Cog: 0.4, Sci: 0.2 |
| S7 | Phase II Trials | 24 mo | 2.8x | Cog: 0.5, Rob: 0.3, Sci: 0.2 |
| S8 | Phase III Trials | 36 mo | 3.2x | Rob: 0.5, Cog: 0.3, Sci: 0.2 |
| S9 | Regulatory Approval | 12 mo | 3x | Cog: 0.9, Rob: 0.1 |
| S10 | Deployment | 12 mo | 4x | Rob: 0.6, Cog: 0.4 |

### Scenarios

| Scenario | g_cognitive | g_robotic | g_scientific | Description |
|----------|------------|-----------|--------------|-------------|
| Pessimistic | 0.40 | 0.20 | 0.35 | Slower AI progress |
| Baseline | 0.60 | 0.30 | 0.55 | Current trends |
| Optimistic | 0.80 | 0.45 | 0.75 | Breakthroughs |

---

## File Structure

```
v0.5/
├── run_model.py                 # Main execution script
├── generate_improved_figures.py # Improved visualization generator
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── requirements.txt             # Python dependencies
├── src/
│   ├── model.py                # Core model (multi-type AI, therapeutic areas)
│   ├── visualize.py            # Standard visualization module
│   ├── visualize_improved.py   # Improved publication-quality figures
│   ├── sensitivity.py          # Sensitivity analysis module
│   └── uncertainty.py          # Monte Carlo uncertainty module
├── docs/
│   ├── TECHNICAL_SPECIFICATION.md  # Mathematical specification
│   ├── FINDINGS_v0.1.md            # v0.1 findings
│   └── FINDINGS_v0.2.md            # v0.2 findings
└── outputs/                    # Generated outputs (after running)
    ├── results.csv             # Complete results data
    ├── parameters.json         # Model parameters
    ├── summary.txt             # Summary statistics
    ├── fig*.png/pdf            # Visualization figures
    ├── fig_tornado_improved.png
    ├── fig_combined_fan_chart.png
    ├── fig_bottleneck_heatmap.png
    └── summary_dashboard_improved.png
```

---

## Mathematical Framework

### Multi-Type AI Capability

Each AI type grows at its own rate:
$$A_c(t) = \exp(g_c \cdot (t - t_0))$$ (Cognitive)
$$A_r(t) = \exp(g_r \cdot (t - t_0))$$ (Robotic)
$$A_s(t) = \exp(g_s \cdot (t - t_0))$$ (Scientific)

### Stage-Specific Effective AI

Weighted combination based on task composition:
$$A_i^{eff}(t) = w_i^c \cdot A_c(t) + w_i^r \cdot A_r(t) + w_i^s \cdot A_s(t)$$

### Therapeutic Area Modification

Success rates adjusted by area-specific factors:
$$p_i^{area} = p_i^{base} \cdot f_{area}$$

### System Throughput

$$\Theta(t) = \min_i \left\{ \mu_i(t) \cdot p_i^{area} \right\}$$

See `docs/TECHNICAL_SPECIFICATION.md` for complete mathematical details.

---

## Key Results (v0.5)

### Summary Statistics

| Scenario | By 2030 | By 2040 | By 2050 | 90% CI (2050) |
|----------|---------|---------|---------|---------------|
| Pessimistic | 8.5 yr | 24.6 yr | 43.4 yr | [34.7, 51.1] |
| Baseline | 11.2 yr | 35.7 yr | 62.9 yr | [48.6, 76.7] |
| Optimistic | 16.9 yr | 55.0 yr | 98.4 yr | [71.6, 120.1] |

### Therapeutic Area Impact

Infectious diseases see fastest acceleration due to clear targets and endpoints.
CNS disorders remain challenging due to complex mechanisms and limited biomarkers.

### AI Type Contributions

- **Early stages (S1-S4)**: Dominated by Cognitive AI
- **Lab stages (S3, S5)**: Robotic AI is critical bottleneck
- **Clinical stages (S6-S8)**: Mixed contributions, all AI types matter
- **Regulatory/Deployment (S9-S10)**: Cognitive AI for documentation and logistics

---

## Improved Visualizations (v0.4.2+)

This version includes improved publication-quality figures:

1. **Tornado Diagram** - Bidirectional bars showing parameter sensitivity
2. **Combined Fan Chart** - All scenarios with 50%/90% uncertainty bands
3. **Bottleneck Heatmap** - Time × stage constraint matrix
4. **Summary Dashboard** - Four informative panels

Generate improved figures:
```bash
python generate_improved_figures.py
```

---

## API Usage

```python
from src.model import AIBioAccelerationModel, TherapeuticArea, compare_ai_types

# Initialize with default parameters
model = AIBioAccelerationModel()

# Run all scenarios
results = model.run_all_scenarios()

# Compare therapeutic areas
area_comparison = model.get_therapeutic_area_comparison()

# Compare AI types
ai_comparison = compare_ai_types()

# Get summary statistics
summary = model.get_summary_statistics()
```

---

## References

- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069) - Clinical trial success by therapeutic area
- DiMasi JA, Grabowski HG, Hansen RW. (2016) "Innovation in the pharmaceutical industry: New estimates of R&D costs" *Journal of Health Economics* 47:20-33. [DOI: 10.1016/j.jhealeco.2016.01.012](https://doi.org/10.1016/j.jhealeco.2016.01.012) - R&D costs and timelines
- Epoch AI. (2024) "AI Trends" [https://epoch.ai/trends](https://epoch.ai/trends) - AI capability growth rates
- Thomas DW, Burns J, Audette J, et al. (2016) "Clinical Development Success Rates 2006-2015" *BIO Industry Analysis*. [PDF](https://www.bio.org/sites/default/files/legacy/bioorg/docs/Clinical%20Development%20Success%20Rates%202006-2015%20-%20BIO,%20Biomedtracker,%20Amplion%202016.pdf) - Phase-specific success rates
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends in Pharmacological Sciences* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005) - AI applications in clinical trials
- Amodei D. (2024) "Machines of Loving Grace" *Anthropic Blog*. [Link](https://www.anthropic.com/news/machines-of-loving-grace) - AI acceleration potential in biology

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v0.5 | Jan 2026 | Multi-type AI + therapeutic area differentiation |
| v0.4.2 | Jan 2026 | Improved publication-quality visualizations |
| v0.4 | Jan 2026 | AI feedback loop, Monte Carlo uncertainty |
| v0.3 | Jan 2026 | Sensitivity analysis, 10-stage pipeline |
| v0.2 | Jan 2026 | Parameter calibration, clinical trial phases |
| v0.1 | Jan 2026 | Pilot model: 8-stage pipeline |

---

## License

MIT License

---

## Contact

For questions or feedback, please open an issue on GitHub.
