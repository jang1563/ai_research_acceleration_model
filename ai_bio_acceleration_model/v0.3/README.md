# AI-Accelerated Biological Discovery Model

**Version 0.1 (Pilot Model)**

A quantitative pipeline model for analyzing bottlenecks in AI-accelerated biological research and drug development.

---

## Overview

This model quantifies how AI capabilities affect the end-to-end biological discovery pipeline, from hypothesis generation through drug deployment. It identifies rate-limiting bottlenecks and computes cumulative scientific progress under different scenarios.

### Key Features (v0.1)

- **8-stage scientific pipeline** modeling the full discovery-to-deployment process
- **AI acceleration curves** with saturation dynamics
- **Bottleneck identification** at each time point
- **Scenario analysis** (pessimistic, baseline, optimistic)
- **Publication-quality visualizations**

### Model Outputs

1. **Progress Rate** - How much faster science advances relative to 2024
2. **Cumulative Progress** - Equivalent years of scientific advancement
3. **Bottleneck Timeline** - Which stage limits progress over time

---

## Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib
```

### Run the Model

```bash
cd v0.1
python run_model.py
```

This will:
1. Run the model for all three scenarios
2. Generate output data in `outputs/results.csv`
3. Create publication-quality figures in `outputs/`
4. Print summary statistics to console

### Expected Output

```
Key Results (Baseline Scenario):
  By 2030: 12.8 equiv. years (6 calendar → 2.1x)
  By 2040: 38.2 equiv. years (16 calendar → 2.4x)
  By 2050: 87.5 equiv. years (26 calendar → 3.4x)
```

---

## Model Structure

### Pipeline Stages

| Stage | Description | Baseline Duration | Max AI Speedup |
|-------|-------------|-------------------|----------------|
| S1 | Hypothesis Generation | 6 months | 50x |
| S2 | Experiment Design | 3 months | 20x |
| S3 | Wet Lab Execution | 12 months | 5x |
| S4 | Data Analysis | 2 months | 100x |
| S5 | Validation & Replication | 8 months | 5x |
| S6 | Clinical Trials | 72 months | 2.5x |
| S7 | Regulatory Approval | 12 months | 2x |
| S8 | Deployment | 12 months | 4x |

### Scenarios

| Scenario | AI Growth Rate | Description |
|----------|---------------|-------------|
| Pessimistic | 30%/year | Slower AI progress, institutional resistance |
| Baseline | 50%/year | Current trends continue |
| Optimistic | 70%/year | Breakthroughs, regulatory reform |

---

## File Structure

```
v0.1/
├── run_model.py              # Main execution script
├── README.md                 # This file
├── src/
│   ├── model.py             # Core model implementation
│   └── visualize.py         # Visualization module
├── docs/
│   └── TECHNICAL_SPECIFICATION.md  # Mathematical specification
├── outputs/                  # Generated outputs (after running)
│   ├── results.csv          # Complete results data
│   ├── parameters.json      # Model parameters
│   ├── summary.txt          # Summary statistics
│   ├── fig1_ai_capability.png/pdf
│   ├── fig2_ai_multipliers.png/pdf
│   ├── fig3_service_rates.png/pdf
│   ├── fig4_bottleneck_timeline.png/pdf
│   ├── fig5_progress_rate.png/pdf
│   ├── fig6_cumulative_progress.png/pdf
│   └── summary_dashboard.png/pdf
└── data/                     # Input data (if any)
```

---

## Mathematical Framework

### Core Equations

**AI Capability:**
$$A(t) = \exp(g \cdot (t - t_0))$$

**AI Multiplier (with saturation):**
$$M_i(t) = 1 + (M_i^{\max} - 1) \cdot \left(1 - A(t)^{-k_i}\right)$$

**System Throughput:**
$$\Theta(t) = \min_i \left\{ \mu_i(t) \cdot p_i \right\}$$

**Cumulative Progress:**
$$Y(T) = \sum_{t=t_0}^{T} R(t)$$

See `docs/TECHNICAL_SPECIFICATION.md` for complete mathematical details.

---

## API Usage

```python
from src.model import AIBioAccelerationModel, ModelConfig

# Initialize with default parameters
model = AIBioAccelerationModel()

# Run all scenarios
results = model.run_all_scenarios()

# Get summary statistics
summary = model.get_summary_statistics()

# Identify bottleneck transitions
transitions = model.get_bottleneck_transitions('Baseline')

# Export results
model.export_results('my_results.csv')
model.export_parameters('my_params.json')
```

### Custom Configuration

```python
from src.model import AIBioAccelerationModel, ModelConfig, Stage, Scenario

# Create custom configuration
config = ModelConfig(
    t0=2024,
    T=2060,  # Extended horizon
    stages=[
        Stage(index=1, name="Custom Stage", tau_baseline=6.0, 
              M_max=30.0, p_success=0.9, k_saturation=1.0),
        # ... more stages
    ],
    scenarios=[
        Scenario(name="Custom", g_ai=0.60, description="My scenario"),
    ]
)

model = AIBioAccelerationModel(config)
results = model.run_all_scenarios()
```

---

## Visualization

```python
from src.model import run_default_model
from src.visualize import ModelVisualizer

# Run model
model, results = run_default_model()

# Create visualizer
viz = ModelVisualizer(results, model.config.stages, output_dir='my_figs')

# Generate individual figures
viz.plot_ai_capability()
viz.plot_cumulative_progress()
viz.plot_bottleneck_timeline()

# Generate all figures
viz.generate_all_figures()
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v0.1 | Jan 2026 | Pilot model: 8-stage pipeline, 3 scenarios, basic visualizations |

### Planned Versions

- **v0.2**: Parameter calibration with literature sources
- **v0.3**: Full scenario analysis
- **v0.4**: AI feedback loop (recursive improvement)
- **v0.5**: Multi-type AI (cognitive/robotic/scientific)
- **v0.6**: Data quality module
- **v0.7**: Pipeline iteration/failure dynamics
- **v0.8**: Disease-specific time-to-cure
- **v0.9**: Policy intervention ROI analysis
- **v1.0**: Monte Carlo uncertainty quantification

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{ai_bio_acceleration_model,
  title = {AI-Accelerated Biological Discovery Model},
  version = {0.1},
  year = {2026},
  url = {https://github.com/[TBD]}
}
```

---

## License

MIT License

---

## Contact

For questions or feedback, please open an issue on GitHub or contact [TBD].
