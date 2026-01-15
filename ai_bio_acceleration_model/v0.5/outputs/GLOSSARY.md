# Glossary of Key Terms
## AI-Accelerated Biological Discovery Model

This glossary explains technical terms used in the model in plain English.

---

## Core Concepts

### Equivalent Years
A measure of how much scientific progress is made compared to the 2024 baseline pace.

**Example:** "93.5 equivalent years by 2050" means that in the 26 calendar years from 2024-2050, we make as much progress as would take 93.5 years at the 2024 pace. This represents ~3.6x acceleration.

**What it means for you:** Think of it like a time machine for science. At 3.6x acceleration, discoveries that would normally take 10 years happen in under 3 years.

---

### Acceleration Factor
The ratio of equivalent years to calendar years.

| Acceleration | Meaning | Example |
|--------------|---------|---------|
| 1x | Same as today | 10 years of work takes 10 years |
| 2x | Twice as fast | 10 years of work takes 5 years |
| 3x | Three times as fast | 10 years of work takes 3.3 years |
| 4x | Four times as fast | 10 years of work takes 2.5 years |

---

### Bottleneck
The slowest step in the discovery pipeline that limits overall speed.

**Analogy:** Like a narrow section of pipe that restricts water flow, the bottleneck determines how fast the entire system can operate—no matter how fast other parts are.

**Current bottleneck:** Phase II clinical trials (testing if drugs actually work)

**Why it matters:** Speeding up the bottleneck has the biggest impact. Making other stages faster won't help much until the bottleneck is addressed.

---

### Progress Rate
How fast scientific discovery is happening relative to 2024.

| Progress Rate | Interpretation |
|---------------|----------------|
| 1.0 | Same speed as 2024 |
| 2.0 | Twice as fast as 2024 |
| 3.0 | Three times as fast |

---

## Model Parameters

### M_max (Maximum AI Multiplier)
The theoretical maximum speedup AI can provide for each stage.

**Key insight:** Physical reality limits AI's impact. Even perfect AI can't make cells divide faster or human bodies metabolize drugs quicker.

| Stage | M_max | Why |
|-------|-------|-----|
| Data Analysis | 100x | Pure computation, no physical limits |
| Hypothesis Generation | 50x | Cognitive task, AI excels |
| Wet Lab Experiments | 5x | Cell biology sets the pace |
| Clinical Trials | 2-4x | Human biology is irreducible |

---

### Service Rate
How quickly each stage of the pipeline processes scientific projects, measured in projects per year.

**Higher service rate = faster stage**

The *effective* service rate also accounts for success probability:
- A stage that processes 10 projects/year but only 30% succeed
- Has an effective rate of 3 successful projects/year

---

### g (Growth Rate)
How fast AI capability grows, measured per year.

| g value | Meaning | AI capability by 2050 |
|---------|---------|----------------------|
| 0.30 | Slow (pessimistic) | ~1,000x today |
| 0.50 | Moderate (baseline) | ~500,000x today |
| 0.70 | Fast (optimistic) | ~200 million x today |

---

## Statistical Concepts

### Confidence Interval (CI)
A range that captures uncertainty in our estimates.

**Example:** "90% CI of [70, 115]" means:
- We're 90% confident the true value is between 70 and 115
- There's only a 5% chance it's below 70
- There's only a 5% chance it's above 115

**Plain English version:**
> "We're 90% confident acceleration will be between 2.7x and 4.4x"

---

### Monte Carlo Simulation
A technique that runs the model thousands of times with slightly different inputs to understand how uncertainty in our assumptions affects conclusions.

**Analogy:** Like playing out thousands of possible futures to see the range of outcomes, rather than betting on a single prediction.

---

## Therapeutic Areas

### Oncology (Cancer)
Cancer treatments. **Best AI potential** because:
- Biomarkers allow targeted trial designs
- Large datasets for AI training
- Clear endpoints (tumor shrinkage)

### CNS (Central Nervous System)
Brain and nervous system diseases like Alzheimer's and Parkinson's. **Hardest for AI** because:
- Complex biology not fully understood
- Difficult to measure outcomes
- Blood-brain barrier limits drug delivery

### Infectious Disease
Bacterial and viral diseases. **Moderate AI potential**:
- Clear targets (kill pathogen)
- Established success patterns
- Recent AI wins (mRNA vaccines)

### Rare Disease
Conditions affecting fewer than 200,000 people. **Good AI potential**:
- Genetic causes often known
- Targeted development possible
- Regulatory incentives exist

---

## AI Types (v0.5)

### Cognitive AI
Language, reasoning, and synthesis capabilities. Examples: GPT-4, Claude.
- **Growth rate:** g = 0.60 (fastest)
- **Best at:** Hypothesis generation, literature review, data analysis

### Robotic AI
Physical manipulation and lab automation. Examples: Lab robots, automated microscopes.
- **Growth rate:** g = 0.30 (slowest)
- **Best at:** Wet lab execution, sample handling

### Scientific AI
Hypothesis generation and pattern recognition. Examples: AlphaFold, drug design AI.
- **Growth rate:** g = 0.55 (intermediate)
- **Best at:** Protein structure prediction, molecular design

---

## Quick Reference

| Term | One-Sentence Definition |
|------|------------------------|
| Equivalent years | Progress made vs. 2024 pace |
| Acceleration factor | How many times faster than 2024 |
| Bottleneck | The slowest step limiting everything |
| M_max | Maximum possible AI speedup per stage |
| Service rate | Projects processed per year |
| Progress rate | Speed relative to 2024 baseline |
| Monte Carlo | Running model many times to capture uncertainty |

---

*Last updated: January 13, 2026*
*Version: 0.5.1*
