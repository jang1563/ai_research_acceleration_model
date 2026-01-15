# AI-Accelerated Biological Discovery Model

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated biological research.

## Project Goal

Create a rigorous bioRxiv paper modeling how AI accelerates biological science, with:
- End-to-end pipeline modeling
- Bottleneck identification and transition forecasting
- Scenario-based analysis
- Policy intervention ROI rankings

## Quick Start

```bash
cd v0.1
pip install -r requirements.txt
python run_model.py
```

## Project Structure

```
ai_bio_acceleration_model/
├── HANDOFF_GUIDE.md           # How to continue with Claude Desktop
├── CLAUDE_DESKTOP_CONTEXT.md  # Copy-paste context for Claude sessions
├── README.md                  # This file
└── v0.1/                      # Current version
    ├── run_model.py           # Main execution script
    ├── README.md              # Version documentation
    ├── CHANGELOG.md           # Version history
    ├── requirements.txt       # Python dependencies
    ├── src/
    │   ├── model.py          # Core model
    │   └── visualize.py      # Visualization
    ├── docs/
    │   ├── TECHNICAL_SPECIFICATION.md
    │   └── FINDINGS_v0.1.md
    └── outputs/               # Generated results
```

## Iteration Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | Core pipeline model | ✅ Complete |
| v0.2 | Parameter calibration | Next |
| v0.3 | Scenario analysis | Planned |
| v0.4 | AI feedback loop | Planned |
| v0.5 | Multi-type AI | Planned |
| v0.6 | Data quality module | Planned |
| v0.7 | Pipeline iteration | Planned |
| v0.8 | Disease models | Planned |
| v0.9 | Policy analysis | Planned |
| v1.0 | Uncertainty quantification | Planned |

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

## Key Results (v0.1)

- Baseline scenario: ~57 equivalent years by 2050 (2.2x acceleration)
- Clinical Trials (S6) identified as persistent bottleneck
- Limited scenario differentiation due to bottleneck constraints

## License

MIT
