# Key Findings - v0.5 (Multi-Type AI + Therapeutic Areas)

## Executive Summary

Version 0.5 introduces multi-type AI differentiation (Cognitive, Robotic, Scientific) and therapeutic area-specific modeling. This enables more nuanced analysis of which AI types matter most for which stages, and how acceleration varies by disease area.

**Bottom Line:** By 2050, Infectious Disease research advances fastest (72.3 equiv. years) while CNS remains most challenging (48.7 equiv. years). Robotic AI is the critical bottleneck due to slower improvement rate (g=0.30).

---

## Key Results

### Multi-Type AI Capabilities (2050, Baseline)

| AI Type | Growth Rate | 2050 Capability | Primary Role |
|---------|-------------|-----------------|--------------|
| Cognitive | 60%/year | 3,814x | Hypothesis, design, analysis |
| Robotic | 30%/year | 62x | Wet lab, validation, manufacturing |
| Scientific | 55%/year | 1,808x | Specialized scientific models |

### Therapeutic Area Comparison (Baseline, 2050)

| Therapeutic Area | Equiv. Years | Relative to Base |
|------------------|--------------|------------------|
| Infectious Disease | 72.3 | +15% |
| Oncology | 68.5 | +9% |
| Rare Disease | 61.2 | -3% |
| CNS | 48.7 | -23% |

### Scenario Summary

| Scenario | By 2030 | By 2040 | By 2050 | 90% CI (2050) |
|----------|---------|---------|---------|---------------|
| Pessimistic | 8.5 yr | 24.6 yr | 43.4 yr | [34.7, 51.1] |
| Baseline | 11.2 yr | 35.7 yr | 62.9 yr | [48.6, 76.7] |
| Optimistic | 16.9 yr | 55.0 yr | 98.4 yr | [71.6, 120.1] |

---

## Critical Insights

### 1. Robotic AI is the Critical Bottleneck

Despite Cognitive AI reaching 3,814x capability:
- Wet lab execution (S3) depends 90% on Robotic AI
- Robotic AI only reaches 62x by 2050
- Physical processes limit overall acceleration

**Implication:** Lab automation research is as important as AI foundation model research.

### 2. Therapeutic Areas Diverge Significantly

The 23 percentage point spread (Infectious: +15% vs CNS: -23%) reflects:
- Infectious diseases have clear targets and endpoints
- CNS faces blood-brain barrier, complex mechanisms
- Oncology benefits from biomarker-driven precision medicine

### 3. Stage-AI Type Matching Matters

Different stages need different AI types:

| Stage | Primary AI Type | Weight |
|-------|-----------------|--------|
| S1 Hypothesis | Cognitive | 80% |
| S3 Wet Lab | Robotic | 90% |
| S4 Analysis | Mixed | 50%/50% |
| S7 Phase II | Cognitive | 50% |

### 4. Scientific AI Bridges Cognitive and Robotic

Scientific AI (AlphaFold-type systems) provides:
- Structure prediction (reducing wet lab needs)
- Mechanism understanding (improving clinical design)
- 1,808x capability partially compensates for slow Robotic AI

---

## Mathematical Framework

### Multi-Type AI Capability

$$A_c(t) = \exp(g_c \cdot (t - t_0))$$ (Cognitive)
$$A_r(t) = \exp(g_r \cdot (t - t_0))$$ (Robotic)
$$A_s(t) = \exp(g_s \cdot (t - t_0))$$ (Scientific)

### Stage-Specific Effective AI

$$A_i^{eff}(t) = w_i^c \cdot A_c(t) + w_i^r \cdot A_r(t) + w_i^s \cdot A_s(t)$$

### Therapeutic Area Modification

$$p_i^{area} = p_i^{base} \cdot f_{area}$$

Where $f_{area}$ is the area-specific success rate modifier.

---

## Stage AI Type Weights

| Stage | Cognitive | Robotic | Scientific |
|-------|-----------|---------|------------|
| S1 Hypothesis | 0.80 | 0.00 | 0.20 |
| S2 Design | 0.60 | 0.00 | 0.40 |
| S3 Wet Lab | 0.10 | 0.90 | 0.00 |
| S4 Analysis | 0.50 | 0.00 | 0.50 |
| S5 Validation | 0.30 | 0.70 | 0.00 |
| S6 Phase I | 0.40 | 0.40 | 0.20 |
| S7 Phase II | 0.50 | 0.30 | 0.20 |
| S8 Phase III | 0.30 | 0.50 | 0.20 |
| S9 Regulatory | 0.90 | 0.10 | 0.00 |
| S10 Deployment | 0.40 | 0.60 | 0.00 |

---

## Therapeutic Area Parameters

### Success Rate Modifiers

| Area | Phase II | Phase III | Overall |
|------|----------|-----------|---------|
| Oncology | 1.18x | 1.12x | 1.25x |
| CNS | 0.45x | 0.86x | 0.65x |
| Infectious | 1.36x | 1.29x | 1.40x |
| Rare Disease | 0.91x | 1.03x | 0.95x |

### Area-Specific M_max Modifiers

Some therapeutic areas may have different AI acceleration potential:
- Oncology: Higher due to biomarker-driven approaches
- CNS: Lower due to mechanism complexity
- Infectious: Higher due to clear targets

---

## Policy Implications

### 1. Invest in Lab Automation

Robotic AI is the limiting factor:
- Lab robotics R&D
- Cloud labs and automation platforms
- Integration of AI planning with robotic execution

### 2. Therapeutic Area Prioritization

Resource allocation should consider area-specific dynamics:
- Infectious diseases: High acceleration potential, prioritize
- CNS: Lower acceleration, may need different approaches
- Oncology: Good balance of opportunity and impact

### 3. Scientific AI as Bridge Technology

AlphaFold-type systems can partially substitute for slow wet lab:
- In silico screening reduces wet lab burden
- Structure prediction accelerates mechanism understanding
- Continue investment in scientific AI

---

## Limitations & Next Steps

### Current Limitations

1. Fixed AI type weights (could evolve over time)
2. Simplified therapeutic area model
3. No interaction between AI types

### Planned for v0.6

- Data Quality Module
- Quality-driven acceleration multipliers
- Stage-specific data quality elasticities

---

## References

- Wong CH, Siah KW, Lo AW. (2019) "Estimation of clinical trial success rates and related parameters" *Biostatistics* 20(2):273-286. [DOI: 10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069) - Clinical trial success by therapeutic area
- Epoch AI. (2024) "AI Trends" [https://epoch.ai/trends](https://epoch.ai/trends) - AI capability growth rates
- Harrer S, Shah P, Antber B, Hu J. (2019) "Artificial Intelligence for Clinical Trial Design" *Trends Pharmacol Sci* 40(8):577-591. [DOI: 10.1016/j.tips.2019.05.005](https://doi.org/10.1016/j.tips.2019.05.005)
- Amodei D. (2024) "Machines of Loving Grace" *Anthropic Blog*. [Link](https://www.anthropic.com/news/machines-of-loving-grace)
