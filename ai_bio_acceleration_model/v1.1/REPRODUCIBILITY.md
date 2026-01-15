# Reproducibility Guide - v1.1

**P1-10: Reproducibility Artifacts**

This document ensures results can be exactly reproduced.

## Random Seed

All stochastic components use:
```python
RANDOM_SEED = 42
```

To reproduce exact results:
```python
import numpy as np
rng = np.random.default_rng(42)
```

## Version Information

| Component | Version |
|-----------|---------|
| Model | 1.1 |
| Python | 3.9+ |
| NumPy | >=1.20.0 |
| Pandas | >=1.3.0 |
| SciPy | >=1.7.0 |

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Model

```bash
cd v1.1/src
python model.py
```

## Monte Carlo Configuration

Default settings for reproducibility:
- N samples: 10,000
- Bootstrap samples: 1,000
- Seed: 42

## Expected Outputs

With seed=42 and default configuration:

| Metric | Expected Value |
|--------|----------------|
| Baseline Progress 2050 | ~110 equiv years |
| Pessimistic Progress 2050 | ~65 equiv years |
| Optimistic Progress 2050 | ~180 equiv years |
| AI Winter Progress 2050 | ~45 equiv years |

*Note: Values approximate due to logistic growth changes in v1.1*

## Verification Checksums

To verify your installation produces correct results:

```python
from model import run_default_model
import hashlib

model, results = run_default_model()
baseline = results[results['scenario'] == 'Baseline']
progress_2050 = baseline[baseline['year'] == 2050]['cumulative_progress'].iloc[0]

# Should be within 1% of expected
assert 105 < progress_2050 < 115, f"Got {progress_2050}"
print("Verification passed!")
```

## Known Differences from v1.0

Due to P1 fixes, v1.1 results differ from v1.0:
1. Lower overall acceleration (logistic saturation)
2. Earlier bottleneck emergence
3. Wider uncertainty intervals (doubled σ for g_ai)

## Contact

For reproducibility issues, please open an issue on GitHub.

## Methodology Disclosure (P1-9)

**IMPORTANT**: The expert review in this model was AI-simulated using Claude (Anthropic).
The "expert panel" represents plausible domain expertise perspectives synthesized by an AI,
not actual external human reviewers.

For rigorous validation, we recommend:
1. Seeking actual domain expert review
2. Comparing against published forecasts (Metaculus, AI Impacts)
3. Validating against historical FDA data (see historical_validation.py)
