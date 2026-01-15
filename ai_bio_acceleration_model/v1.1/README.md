# AI-Accelerated Biological Discovery Model - v1.1

## Overview

Version 1.1 implements fixes from a **15-expert simulated review panel** that identified critical and important issues in the statistical model. This version includes 10 P1 (Critical) fixes and 8 P2 (Important) fixes.

> **Important Disclosure:** Expert review was AI-simulated using Claude (Anthropic). See REPRODUCIBILITY.md for methodology details.

## Key Changes from v1.0

### P1 Critical Fixes

| ID | Fix | Impact |
|----|-----|--------|
| P1-1 | Sobol indices labeled as APPROXIMATE | Transparency |
| P1-2 | Calibrated g_ai distribution (σ doubled to 0.50) | 2x wider CIs |
| P1-3 | Historical validation against FDA 2015-2023 | Credibility |
| P1-4 | Reduced wet lab M_max (5.0 → 2.5) | Lower acceleration |
| P1-5 | Regulatory floor (6-month minimum) | Realistic timelines |
| P1-6 | Logistic AI growth model | Saturation dynamics |
| P1-7 | AI winter scenario (15% probability) | Tail risk |
| P1-8 | Global access factors (LMIC populations) | Realistic impact |
| P1-9 | Methodology disclosure (AI-simulated review) | Transparency |
| P1-10 | Reproducibility artifacts | Exact reproducibility |

### P2 Important Fixes

| ID | Fix | Impact |
|----|-----|--------|
| P2-11 | Bootstrap CIs on Sobol (1000 samples, 90% CI) | Uncertainty on sensitivity |
| P2-12 | Disease-specific Phase II M_max overrides | Nuanced predictions |
| P2-13 | Manufacturing constraints (cell/gene therapy) | Realistic deployment |
| P2-14 | Compute constraints on AI types | Slightly lower growth |
| P2-15 | Policy implementation curves | Realistic policy analysis |
| P2-16 | Expanded QALY range ($50K-$200K) | Wider economic uncertainty |
| P2-17 | Vaccine pipeline pathway | Pandemic preparedness |
| P2-18 | Reduced S1 p_success (0.95 → 0.40) | More realistic |

## Numerical Changes Summary

| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| S3 (Wet Lab) M_max | 5.0 | 2.5 | -50% |
| S1 p_success | 0.95 | 0.40 | -58% |
| g_ai σ | 0.25 | 0.50 | +100% |
| AI growth model | Exponential | Logistic | Saturation |
| Regulatory floor | None | 6 months | New |
| AI winter probability | 0% | 15% | New |
| Global access | 100% | 20-80% | New |

## Installation

```bash
# Clone repository
git clone https://github.com/jang1563/ai-bio-model.git
cd ai-bio-model/ai_bio_acceleration_model/v1.1

# Install dependencies
pip install -r requirements.txt

# Run model
python run_model.py
```

## Usage

### Quick Start

```python
from src.model import AIBioAccelerationModel, ModelConfig, DEFAULT_SCENARIOS

# Initialize model
config = ModelConfig()
model = AIBioAccelerationModel(config)

# Run baseline scenario
results = model.run_scenario(DEFAULT_SCENARIOS[1])  # Baseline

print(f"Progress by 2050: {results.iloc[-1]['cumulative_progress']:.1f} equivalent years")
```

### Generate Figures

```bash
python generate_publication_figures.py
```

### Full Analysis

```bash
python run_model.py --full
```

## File Structure

```
v1.1/
├── src/
│   ├── model.py                    # Core model with P1/P2 fixes
│   ├── uncertainty_quantification.py  # Calibrated distributions
│   ├── historical_validation.py    # P1-3: FDA validation
│   ├── data_quality.py            # P1-8: Global access factors
│   ├── disease_models.py          # P2-17: Vaccine pathway
│   ├── pipeline_iteration.py      # P2-13: Manufacturing constraints
│   ├── policy_analysis.py         # P2-15: Implementation curves
│   ├── sobol_analysis.py          # P2-11: Bootstrap CIs
│   └── visualize*.py              # Visualization modules
├── outputs/
│   └── figures/                   # Generated figures
├── docs/
│   └── TECHNICAL_SPECIFICATION.md
├── run_model.py                   # Main runner script
├── generate_publication_figures.py # Figure generation
├── requirements.txt               # P1-10: Pinned dependencies
├── REPRODUCIBILITY.md             # P1-10: Seeds and verification
├── CHANGELOG.md                   # Complete change documentation
└── README.md                      # This file
```

## Key Results

### Scenario Comparison (2050 Progress)

| Scenario | Progress | Acceleration |
|----------|----------|--------------|
| Pessimistic | ~51 yr | 2.0x |
| Baseline | ~74 yr | 2.8x |
| Optimistic | ~106 yr | 4.1x |
| AI Winter (P1-7) | ~59 yr | 2.3x |
| Upper Bound (Amodei) | ~110 yr | 4.2x |

**Note:** v1.1 results are ~20-30% lower than v1.0 due to more conservative parameters.

### Key Findings

1. **AI growth rate dominates uncertainty** (Sobol S_i ≈ 0.85)
2. **Logistic saturation** limits long-term acceleration
3. **Regulatory floors** are binding constraints
4. **Global access** significantly reduces beneficiary estimates
5. **AI winter scenario** captures tail risk

## References

### Data Sources
- FDA CDER Annual Reports (2015-2023)
- Wong et al. (2019) Phase success rates
- Tufts CSDD Development times
- Epoch AI Compute Trends

### Methodology
- Saltelli et al. (2010) Sobol sensitivity analysis
- Iman & Conover (1982) Correlated sampling
- NICE/ICER QALY guidelines

## Citation

```bibtex
@software{ai_bio_acceleration_model_v11,
  title={AI-Accelerated Biological Discovery Model},
  version={1.1},
  year={2026},
  url={https://github.com/jang1563/ai-bio-model}
}
```

## License

MIT License - See LICENSE file for details.

---

*Version 1.1 - January 2026*
*Expert review was AI-simulated using Claude (Anthropic)*
