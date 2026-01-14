# AI-Accelerated Scientific Research Model

**Version 0.4** | January 2026

A quantitative model analyzing how AI accelerates scientific research, from hypothesis generation to publication. Validated against historical paradigm shifts and real-world AI breakthroughs (AlphaFold, GNoME, ESM-3).

## Version History

| Version | Focus | Key Features |
|---------|-------|--------------|
| **v0.4** | Refined model | Improved architecture, UQ foundation |
| **v0.3** | Case study validation | AlphaFold, GNoME, ESM-3 validation |
| **v0.2** | Historical calibration | 6 paradigm shifts, R² = 0.89 |
| **v0.1** | Initial model | 8-stage pipeline, Monte Carlo |

## Key Findings (v0.3 Calibrated)

| Year | Acceleration | 80% CI | Bottleneck |
|------|-------------|--------|------------|
| 2025 | 1.9x | [1.7, 2.1] | Wet Lab Data Generation |
| 2030 | 8x | [5, 12] | Wet Lab Data Generation |
| 2050 | 21x | [15, 30] | Wet Lab Data Generation |

## Quick Start

```bash
cd v0.3  # or v0.1, v0.2, v0.4

# Quick deterministic forecast
python run_model.py

# Run validation against case studies (v0.3)
python run_validation.py

# Run historical calibration (v0.2)
python run_calibration.py
```

## Model Structure

### 8-Stage Research Pipeline

1. **S1 - Literature Synthesis** (3 mo): High AI potential (100x speed)
2. **S2 - Hypothesis Generation** (6 mo): Cognitive task (100x speed)
3. **S3 - Experimental Design** (2 mo): Design optimization (30x speed)
4. **S4 - Data Generation (Wet Lab)** (12 mo): **Physical constraint** (2.5x max)
5. **S5 - Data Analysis** (3.5 mo): Highest AI potential (100x speed)
6. **S6 - Validation & Replication** (8 mo): **Social + physical** (2.5x max)
7. **S7 - Writing & Peer Review** (6 mo): Social process (10x speed)
8. **S8 - Publication & Dissemination** (3 mo): Social process (20x speed)

### Key Insight

**Wet lab data generation (S4) and validation (S6) remain the binding constraints** on research acceleration. Even with transformative AI capabilities, biological timescales and replication requirements limit ultimate acceleration to ~20x.

### Case Study Validation (v0.3)

| AI System | Domain | Acceleration | Model Prediction |
|-----------|--------|--------------|------------------|
| AlphaFold | Protein structure | 50+ years compressed | Validated |
| GNoME | Materials discovery | 800 years of work | Validated |
| ESM-3 | Protein design | Democratized | Validated |

## Project Structure

```
ai_research_acceleration_model/
├── PROJECT_BIBLE.md          # Master reference document
├── README.md                 # This file
├── v0.1/                     # Initial model
│   ├── src/                  # Core modules
│   ├── figures/              # 5 publication figures
│   └── outputs/              # MC simulation results
├── v0.2/                     # Historical calibration
│   ├── src/                  # Calibration module
│   ├── figures/              # 6 calibration figures
│   └── EXPERT_REVIEW_V0.2.md # Expert review
├── v0.3/                     # Case study validation
│   ├── src/                  # AlphaFold, GNoME, ESM-3
│   ├── figures/              # 6 validation figures
│   └── V0.3_TECHNICAL_REPORT.md
├── v0.4/                     # Refined model
│   └── src/                  # Improved architecture
├── config/                   # Configuration files
├── data/                     # Input data
└── tests/                    # Unit tests
```

## Documentation

- `PROJECT_BIBLE.md` - Complete documentation
- `v0.2/EXPERT_REVIEW_V0.2.md` - 10-expert simulated review
- `v0.3/V0.3_TECHNICAL_REPORT.md` - Case study validation report

## Scenarios

| Scenario | g_ai | 2050 Acceleration | Description |
|----------|------|-------------------|-------------|
| AI Winter | 0.15 | 15x | Major disruption |
| Conservative | 0.30 | 19x | Steady progress |
| **Baseline** | 0.40 | 21x | Expected trajectory |
| Ambitious | 0.55 | 22x | Accelerated progress |

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

## Related Projects

- [AI Bio Acceleration Model](https://github.com/jang1563/ai_bio_acceleration_model) - Drug discovery focus
- [AI Bio Acceleration Website](https://jang1563.github.io/ai-bio-acceleration/) - Interactive visualization

## Citation

```bibtex
@software{ai_research_acceleration_model,
  title={AI-Accelerated Scientific Research Model},
  version={0.4},
  year={2026},
  url={https://github.com/jang1563/ai_research_acceleration_model}
}
```

## License

MIT License - Research use permitted with attribution.

---

*Model development assisted by Claude (Anthropic)*
