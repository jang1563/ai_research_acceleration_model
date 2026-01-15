# AI-Accelerated Biological Discovery Model

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated biological research.

## Project Goal

A quantitative model analyzing how AI accelerates biological science, with:
- End-to-end pipeline modeling
- Bottleneck identification and transition forecasting
- Scenario-based analysis
- Policy intervention ROI rankings

## Quick Start

```bash
cd v1.0
pip install -r requirements.txt
python run_model.py
```

## Project Structure

```
ai_bio_acceleration_model/
├── PROJECT_BIBLE.md           # Master reference document
├── HANDOFF_GUIDE.md           # How to continue with Claude Desktop
├── CLAUDE_DESKTOP_CONTEXT.md  # Copy-paste context for Claude sessions
├── README.md                  # This file
├── webpage/                   # Interactive web dashboard
│   ├── index.html            # Main dashboard
│   ├── supplementary.html    # Supplementary materials
│   ├── app.js                # Application logic
│   ├── charts.js             # Chart visualizations
│   ├── styles.css            # Styling
│   └── data/                 # JSON data files
├── v0.1/                      # Initial pilot model
├── v0.2/                      # Parameter calibration
├── v0.3/                      # Scenario analysis (sensitivity tornado diagrams)
├── v0.4/                      # Monte Carlo uncertainty quantification
├── v0.5/                      # Multi-type AI + Therapeutic areas
├── v0.6/                      # Data Quality Module (DQM)
├── v0.7/                      # Pipeline Iteration with Rework Dynamics
├── v0.8/                      # Disease-Specific Time-to-Cure Models
├── v0.9/                      # Policy Analysis + Intervention ROI
└── v1.0/                      # Current version (Final Paper with Full UQ)
    ├── run_model.py           # Main execution script
    ├── requirements.txt       # Python dependencies
    ├── src/
    │   ├── model.py          # Core model
    │   ├── uncertainty_quantification.py  # Monte Carlo + Sobol
    │   ├── policy_analysis.py # Policy intervention analysis
    │   ├── disease_models.py  # Disease-specific projections
    │   └── visualize_v2.py   # Communication-optimized figures
    ├── docs/
    │   ├── TECHNICAL_SPECIFICATION.md
    │   └── FINDINGS_v1.0.md
    └── outputs/               # UQ results, Sobol indices, figures
```

## Iteration Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | Core pipeline model | ✅ Complete |
| v0.2 | Parameter calibration | ✅ Complete |
| v0.3 | Scenario analysis | ✅ Complete |
| v0.4 | Monte Carlo + Improved visualizations | ✅ Complete |
| v0.5 | Multi-type AI + Therapeutic areas | ✅ Complete |
| v0.6 | Data Quality Module (DQM) | ✅ Complete |
| v0.7 | Pipeline Iteration with Rework Dynamics | ✅ Complete |
| v0.8 | Disease-Specific Time-to-Cure Models | ✅ Complete |
| v0.9 | Policy Analysis + Intervention ROI | ✅ Complete |
| v1.0 | Final Paper with Full Uncertainty Quantification | ✅ Complete |

## Working with Claude Desktop

See `HANDOFF_GUIDE.md` for detailed instructions on:
- How to set up local development
- How to communicate with Claude Desktop
- Workflow for each iteration
- Tips for efficient development

See `CLAUDE_DESKTOP_CONTEXT.md` for:
- Copy-paste session starters
- Quick reference for equations and parameters
- Common task prompts

## Interactive Dashboard

Open `webpage/index.html` in a browser to explore:
- Interactive scenario comparisons
- Policy intervention analysis
- Disease time-to-cure projections
- Monte Carlo uncertainty distributions

## Key Results (v1.0)

- **Full uncertainty quantification** with Monte Carlo (N=10,000) and Sobol indices
- **80/90/95% confidence intervals** on all major outputs
- **Policy intervention ROI rankings** with uncertainty propagation
- **Multi-type AI**: Cognitive (g=0.60), Robotic (g=0.30), Scientific (g=0.55)
- **Expert panel review** with validated methodology

### Monte Carlo Results (Baseline 2050)

| Metric | Mean | Median | 95% CI |
|--------|------|--------|--------|
| Progress | 157.8 yr | 147.1 yr | [68, 309] |
| Acceleration | 6.1x | 5.7x | [2.6x, 11.9x] |

### Sobol Sensitivity Indices

| Parameter | S1 (Main Effect) | ST (Total) |
|-----------|-----------------|------------|
| g_ai | 0.45 | 0.52 |
| M_max_S7 | 0.22 | 0.28 |
| p_S7 | 0.15 | 0.18 |

### Top Policy Interventions by ROI

| Rank | Intervention | Annual Cost | ROI |
|------|-------------|-------------|-----|
| 1 | Expand Adaptive Trial Designs | $200M | 17,510 |
| 2 | Real-World Evidence Integration | $200M | 10,401 |
| 3 | Industry-Academia AI Partnerships | $300M | 8,300 |

- **AI growth rate (g_ai)** dominates variance (~45%)
- **Phase II Trials (S7)** remains primary bottleneck
- Amodei's 10x estimate is ~85th percentile of our distribution

## License

MIT
