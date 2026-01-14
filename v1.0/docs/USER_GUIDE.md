# AI Research Acceleration Model v1.0

## User Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Interpreting Results](#interpreting-results)
8. [FAQ](#faq)

---

## Overview

The AI Research Acceleration Model predicts how AI will accelerate scientific research across five key domains:

| Domain | Description | Current AI Impact |
|--------|-------------|-------------------|
| **Structural Biology** | Protein structure prediction | AlphaFold revolution |
| **Drug Discovery** | End-to-end drug development | AI-accelerated discovery |
| **Materials Science** | Novel materials discovery | GNoME predictions |
| **Protein Design** | De novo protein engineering | ESM-3, RFdiffusion |
| **Clinical Genomics** | Variant interpretation | AlphaMissense |

### Key Features

- **Probabilistic forecasts** with confidence intervals
- **Cross-domain spillover** effects (how one domain accelerates another)
- **Scenario analysis** (pessimistic to breakthrough)
- **Workforce impact** projections
- **Policy recommendations**

---

## Installation

### Requirements

- Python 3.8+
- NumPy

### Setup

```bash
# Clone or download the model
cd ai_research_acceleration_model/v1.0

# Run the model
python src/ai_acceleration_model.py
```

No external dependencies required beyond NumPy.

---

## Quick Start

### Basic Usage

```python
from ai_acceleration_model import AIAccelerationModel

# Initialize the model
model = AIAccelerationModel()

# Get a quick forecast
forecast = model.forecast("drug_discovery", 2030)
print(f"Drug Discovery 2030: {forecast.acceleration:.1f}x")
print(f"90% CI: [{forecast.ci_90[0]:.1f}x - {forecast.ci_90[1]:.1f}x]")

# Get system-wide summary
print(model.executive_summary(2030))
```

### Convenience Functions

```python
from ai_acceleration_model import quick_forecast, quick_summary

# One-liner forecast
forecast = quick_forecast("protein_design", 2030)

# One-liner summary
print(quick_summary(2030))
```

---

## Core Concepts

### Acceleration

**Acceleration** measures how much faster research proceeds with AI compared to traditional methods:

- **1.0x** = No acceleration (same as before)
- **2.0x** = Research proceeds twice as fast
- **10.0x** = Research proceeds ten times as fast

### Confidence Intervals

All forecasts include uncertainty quantification:

- **50% CI**: True value likely falls in this range half the time
- **90% CI**: True value falls in this range 90% of the time

**Wider CI = more uncertainty**

### Scenarios

Five scenarios capture different futures:

| Scenario | Probability | Description |
|----------|-------------|-------------|
| Pessimistic | 10% | AI winter, regulatory backlash |
| Conservative | 20% | Slower progress than expected |
| Baseline | 40% | Expected trajectory continues |
| Optimistic | 20% | Faster progress, favorable conditions |
| Breakthrough | 10% | Transformative advances |

### Cross-Domain Spillovers

Acceleration in one domain enables acceleration in others:

```
Structural Biology ──(+33%)──> Drug Discovery
Structural Biology ──(+37%)──> Protein Design
Protein Design ──────(+16%)──> Drug Discovery
```

---

## API Reference

### AIAccelerationModel

The main class for all model operations.

```python
model = AIAccelerationModel(seed=42)
```

**Parameters:**
- `seed` (int): Random seed for reproducibility (default: 42)

### Methods

#### `forecast(domain, year, scenario="baseline")`

Generate a forecast for a single domain.

```python
forecast = model.forecast("drug_discovery", 2030, "optimistic")
```

**Parameters:**
- `domain` (str): Domain name
- `year` (int): Target year (2024-2050)
- `scenario` (str): "pessimistic", "conservative", "baseline", "optimistic", "breakthrough"

**Returns:** `DomainForecast` object

#### `system_snapshot(year, scenario="baseline")`

Generate a system-wide snapshot.

```python
snapshot = model.system_snapshot(2030)
```

**Returns:** `SystemSnapshot` object

#### `compare_scenarios(domain, year)`

Compare all scenarios for a domain.

```python
scenarios = model.compare_scenarios("drug_discovery", 2030)
for name, forecast in scenarios.items():
    print(f"{name}: {forecast.acceleration:.1f}x")
```

**Returns:** Dict of scenario name → DomainForecast

#### `trajectory(domain=None, start_year=2025, end_year=2035)`

Generate trajectory over time.

```python
# Single domain trajectory
traj = model.trajectory("drug_discovery", 2025, 2035)

# System-wide trajectory
traj = model.trajectory(None, 2025, 2035)
```

**Returns:** List of forecasts/snapshots

#### `get_policy_recommendations(year=2030)`

Get policy recommendations.

```python
recs = model.get_policy_recommendations(2030)
for rec in recs:
    print(f"[{rec.priority.upper()}] {rec.title}")
```

**Returns:** List of `PolicyRecommendation` objects

#### `executive_summary(year=2030)`

Generate formatted executive summary.

```python
print(model.executive_summary(2030))
```

**Returns:** Formatted string

### Data Classes

#### DomainForecast

```python
@dataclass
class DomainForecast:
    domain: str
    year: int
    acceleration: float           # Point estimate
    ci_50: Tuple[float, float]    # 50% confidence interval
    ci_90: Tuple[float, float]    # 90% confidence interval
    standalone_acceleration: float
    cross_domain_boost: float
    primary_bottleneck: str
    bottleneck_fraction: float
    jobs_displaced: float         # Millions
    jobs_created: float           # Millions
    net_jobs: float               # Millions
    confidence_level: str         # "high", "medium", "low"
    key_assumptions: List[str]
```

#### SystemSnapshot

```python
@dataclass
class SystemSnapshot:
    year: int
    total_acceleration: float
    acceleration_ci_90: Tuple[float, float]
    domain_forecasts: Dict[str, DomainForecast]
    total_displaced: float
    total_created: float
    workforce_change: float
    investment_needed: str
    critical_actions: int
    fastest_domain: str
    slowest_domain: str
    highest_spillover: Tuple[str, str, float]
```

---

## Examples

### Example 1: Compare Domains

```python
model = AIAccelerationModel()

print("Domain Comparison (2030 Baseline)")
print("-" * 50)

for domain in model.domains:
    f = model.forecast(domain, 2030)
    name = model.DOMAIN_NAMES[domain]
    print(f"{name:<22} {f.acceleration:>6.1f}x  [{f.ci_90[0]:.1f}-{f.ci_90[1]:.1f}]")
```

### Example 2: Track Progress Over Time

```python
model = AIAccelerationModel()

print("Drug Discovery Trajectory")
print("-" * 40)

for year in range(2025, 2036):
    f = model.forecast("drug_discovery", year)
    bar = "█" * int(f.acceleration * 5)
    print(f"{year}: {f.acceleration:>5.1f}x {bar}")
```

### Example 3: Scenario Planning

```python
model = AIAccelerationModel()

domain = "structural_biology"
year = 2030

print(f"\n{model.DOMAIN_NAMES[domain]} Scenario Analysis ({year})")
print("=" * 60)

scenarios = model.compare_scenarios(domain, year)
for name, forecast in scenarios.items():
    print(f"\n{name.upper()} Scenario:")
    print(f"  Acceleration: {forecast.acceleration:.1f}x")
    print(f"  90% CI: [{forecast.ci_90[0]:.1f}x - {forecast.ci_90[1]:.1f}x]")
    print(f"  Net jobs: {forecast.net_jobs:+.2f}M")
```

### Example 4: Workforce Analysis

```python
model = AIAccelerationModel()
snapshot = model.system_snapshot(2030)

print("\nWorkforce Impact Analysis (2030)")
print("=" * 60)
print(f"{'Domain':<22} {'Displaced':>12} {'Created':>12} {'Net':>12}")
print("-" * 60)

for domain in model.domains:
    f = snapshot.domain_forecasts[domain]
    name = model.DOMAIN_NAMES[domain]
    print(f"{name:<22} {f.jobs_displaced:>11.2f}M {f.jobs_created:>11.2f}M {f.net_jobs:>+11.2f}M")

print("-" * 60)
print(f"{'TOTAL':<22} {snapshot.total_displaced:>11.2f}M {snapshot.total_created:>11.2f}M {snapshot.workforce_change:>+11.2f}M")
```

### Example 5: Policy Briefing

```python
model = AIAccelerationModel()

print("\nPolicy Recommendations")
print("=" * 60)

recs = model.get_policy_recommendations(2030)

for priority in ["critical", "high", "medium"]:
    priority_recs = [r for r in recs if r.priority == priority]
    if priority_recs:
        print(f"\n{priority.upper()} PRIORITY:")
        for rec in priority_recs:
            print(f"  [{rec.id}] {rec.title}")
            print(f"        Stakeholders: {', '.join(rec.stakeholders)}")
            print(f"        Timeline: {rec.timeline} | Investment: {rec.investment}")
```

---

## Interpreting Results

### Reading Acceleration Values

| Acceleration | Interpretation |
|--------------|----------------|
| < 1.5x | Modest impact - AI augments but doesn't transform |
| 1.5x - 3x | Significant impact - clear time/cost savings |
| 3x - 10x | Transformative - fundamentally changes process |
| > 10x | Revolutionary - enables previously impossible research |

### Understanding Uncertainty

**Narrow CI (e.g., [1.2-1.6])**: High confidence in prediction
- Well-validated by historical data
- Mature technology with predictable trajectory

**Wide CI (e.g., [1.0-5.0])**: High uncertainty
- Limited historical validation
- Depends on breakthrough discoveries
- Policy/regulatory uncertainty

### Bottleneck Analysis

The model identifies the **primary bottleneck** constraining acceleration:

| Bottleneck | Domain | Implication |
|------------|--------|-------------|
| Clinical trials | Drug Discovery | Regulatory reform needed |
| Synthesis | Materials Science | Automation investment |
| Expression validation | Protein Design | Scale-up facilities |
| Experimental validation | Structural Biology | Cryo-EM capacity |
| Clinical adoption | Clinical Genomics | Training & standards |

### Workforce Implications

- **Displaced**: Jobs likely automated or significantly changed
- **Created**: New roles enabled by AI acceleration
- **Net**: Overall employment effect

**Key insight**: Net impact is typically positive, but **transition support** is essential for displaced workers.

---

## FAQ

### Q: How accurate are these predictions?

The model has been validated against 9 historical case studies (2021-2024) with a mean log error of 0.17. Future predictions carry more uncertainty, captured in the confidence intervals.

### Q: Why is drug discovery acceleration limited?

Clinical trials remain the dominant bottleneck. AI can accelerate discovery phases but cannot fully replace human trials under current regulations. Regulatory reform could unlock higher acceleration.

### Q: How do spillovers work?

When one domain accelerates, it enables faster progress in others. For example, AlphaFold (structural biology) provides structures that accelerate structure-based drug design.

### Q: Can I customize the model parameters?

The model uses calibrated parameters from literature review and historical validation. For research purposes, you can modify the class attributes directly, but this may affect validity.

### Q: How should I cite this model?

```
AI Research Acceleration Model v1.0 (January 2026)
Available at: [repository URL]
```

---

## Support

For questions, issues, or contributions, please refer to the technical report and source code documentation.

---

*User Guide v1.0 - January 2026*
