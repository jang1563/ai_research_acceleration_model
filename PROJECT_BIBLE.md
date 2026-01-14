# PROJECT BIBLE: AI-Accelerated Scientific Research Model

> **Purpose:** Master reference document for the AI-Accelerated Scientific Research Model project. This is a spin-off from the AI-Accelerated Biological Discovery Model, focusing on **basic and translational research** (hypothesis to publication) rather than drug development (clinical trials to approval).

> **Version:** v0.2 (Historical Calibration Complete)
> **Last Updated:** January 14, 2026
> **Status:** Core Framework + Historical Calibration Complete

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Theoretical Foundation: Amodei Framework](#2-theoretical-foundation-amodei-framework)
3. [Historical Paradigm Shifts](#3-historical-paradigm-shifts)
4. [Research Pipeline Definition](#4-research-pipeline-definition)
5. [Paradigm Shift Module (PSM)](#5-paradigm-shift-module-psm)
6. [Mathematical Framework](#6-mathematical-framework)
7. [AI Failure Modes & Risks](#7-ai-failure-modes--risks)
8. [Infrastructure Constraints](#8-infrastructure-constraints)
9. [Research System Transformation](#9-research-system-transformation)
10. [International Coordination (Pillar 5)](#10-international-coordination-pillar-5)
11. [Researcher Education Reform](#11-researcher-education-reform)
12. [Case Studies (2022-2025)](#12-case-studies-2022-2025)
13. [Scenario Definitions](#13-scenario-definitions)
14. [Expert Reviewer Panel Structure](#14-expert-reviewer-panel-structure)
15. [Iteration Roadmap](#15-iteration-roadmap)
16. [Paper Outline](#16-paper-outline)
17. [Key Decisions Log](#17-key-decisions-log)
18. [Data Sources & References](#18-data-sources--references)
19. [Open Questions](#19-open-questions)
20. [Expert Review Summary](#20-expert-review-summary)

---

## 1. PROJECT OVERVIEW

### 1.1 What We're Building

A **quantitative model** analyzing how AI accelerates scientific research, covering:

- The complete research pipeline from hypothesis generation to publication
- **Paradigm shift dynamics** (qualitative breakthroughs, not just speedups)
- **AI failure modes and risks** (hallucination propagation, monoculture risks)
- Multi-stakeholder **system transformation** recommendations (institutions, funders, researchers)
- **International coordination** mechanisms for global AI research governance
- **Historical calibration** using microscope, telescope, genome project, CRISPR, sequencing
- **Phased implementation roadmap** (2025-2035, 2035-2045, 2045-2055) [Extended per expert review]

### 1.2 Relationship to Drug Development Model

| Aspect | Drug Development Model (Existing) | Research Model (This Project) |
|--------|-----------------------------------|-------------------------------|
| **Pipeline End** | FDA Approval | Publication / Validated Discovery |
| **Stages** | 10 stages (incl. Phase I/II/III) | 8 stages (hypothesis → publication) |
| **Bottleneck** | Phase II Clinical Trials | Validation & Replication + Infrastructure |
| **Success Metric** | Approved Therapies | **Validated Discoveries** (operationally defined) |
| **Novel Feature** | Regulatory constraints | **Paradigm Shift Module + AI Failure Modes** |
| **Physical Limits** | Human biology | Experimental time, data generation |
| **Scope** | Drug development only | Basic + Translational research |

### 1.3 Core Thesis

> "AI is not merely accelerating research tasks incrementally—it is enabling **capability extensions** that make previously intractable questions answerable. However, this acceleration is bounded by physical constraints, requires system adaptation, and carries risks of failure modes that must be actively managed. Understanding both opportunities and risks enables evidence-based research policy."

**Note:** Per expert review (Panel C2), we distinguish between:
- **Capability extensions**: Expanding what can be done within existing conceptual frameworks (AlphaFold, sequencing)
- **Methodological shifts**: Changing how research is conducted (CRISPR, PCR)
- **Conceptual paradigm shifts**: Changing fundamental assumptions (germ theory, central dogma)—these remain unpredictable

### 1.4 Success Metric: Validated Discoveries (Operational Definition)

**[P1 Fix: Operationalized per expert review]**

"Validated Discovery" is defined as a published finding meeting ANY of the following criteria:

| Tier | Criterion | Measurement |
|------|-----------|-------------|
| **Tier 1 (Gold)** | Independently replicated | ≥1 successful replication in OSF/CurateScience registry |
| **Tier 2 (Silver)** | High citation impact | ≥50 citations within 5 years (field-normalized) |
| **Tier 3 (Bronze)** | Downstream incorporation | Referenced in ≥10 subsequent publications with methods use |

**Baseline Discovery Rate (D_baseline):**
- Estimated from 2020-2024 Web of Science data
- Biology/Life Sciences: ~15,000 Tier 1+2 validated discoveries/year
- To be refined with bibliometric analysis in v0.2

### 1.5 Project Scope

**In Scope:**
- Basic research (hypothesis → discovery)
- Translational research (pre-clinical, bridges to drug development model)
- Life sciences / biology focus
- System transformation recommendations
- Researcher education reform
- International coordination mechanisms
- AI failure mode modeling

**Out of Scope:**
- Clinical trials (covered by existing model)
- Drug regulatory approval
- Non-biology domains (physics, chemistry—future extension)

---

## 2. THEORETICAL FOUNDATION: AMODEI FRAMEWORK

### 2.1 Source: "Machines of Loving Grace" (October 2024)

Dario Amodei (CEO, Anthropic) published a 15,000-word essay outlining AI's potential to transform science.

**Key Predictions:**
- **"Compressed 21st Century"**: 50-100 years of biological progress in 5-10 years
- **10x acceleration** target (not 100x due to physical world constraints)
- **Biology as highest-potential domain** for AI impact
- **AGI by 2026** (informs our AI capability growth rates)

### 2.2 The Five Limiting Factors

Amodei introduces "Marginal Returns to Intelligence"—the idea that adding more AI/intelligence hits constraints:

| Factor | Description | Implication for Research | Our Model Treatment |
|--------|-------------|--------------------------|---------------------|
| **1. Speed of Physical World** | Cells divide, animals grow at fixed speeds | Wet lab stages: M_max ≤ 2.5x | S4 hard ceiling |
| **2. Need for Data** | AI effectiveness limited by data quality/availability | Data generation remains bottleneck | Infrastructure module |
| **3. Intrinsic Complexity** | Some systems are chaotic/unpredictable | Not all problems yield to more intelligence | PSM Type III uncertainty |
| **4. Human Constraints** | Regulations, ethics, social factors | Peer review, publication, validation | S6, S7 social ceilings |
| **5. Physical Laws** | Absolute constraints | Cannot be circumvented | Model boundary |

### 2.3 Why 10x and Not 100x?

Amodei's reasoning:
- **Serial dependencies**: Many experiments must be done sequentially
- **Iteration requirements**: Animal experiments, microscope design need multiple rounds
- **Irreducible latency**: Some biological processes simply take time
- **Hardware constraints**: Lab equipment design has inherent delays

**Our model interpretation (revised per expert review):**
- Cognitive stages (hypothesis, analysis): **Speed** 50-100x, **Quality** 15-50x (distinct multipliers)
- Physical stages (wet lab, validation): 2.5x maximum (biological timescales irreducible)
- Social stages (peer review, publication): 2.5-5x (institutional processes)
- **System bottleneck**: Overall 5-8x achievable, consistent with Amodei's conservative estimate

### 2.4 The Unlock: AI-Invented Simulation Tools

The diagram of "Marginal Returns to Intelligence" shows raw intelligence passing through successive bottlenecks (Speed of Physical World → Data Scarcity → Intrinsic Complexity → Human Constraints). However, there exists a potential **bypass pathway**:

> **The Unlock:** AI invents simulation tools that replace physical trials.

This represents a qualitatively different acceleration mechanism—not just speeding up existing processes, but **circumventing physical-world bottlenecks entirely**.

#### 2.4.1 Current Examples of Physical-to-Simulation Substitution

| Domain | Physical Process | AI Simulation Substitute | Status |
|--------|-----------------|-------------------------|--------|
| **Protein Structure** | X-ray crystallography (months-years) | AlphaFold prediction (minutes) | ✅ Deployed |
| **Drug Binding** | Wet lab binding assays | Molecular dynamics + ML | 🔄 Emerging |
| **Cell Behavior** | Cell culture experiments | Virtual cell models | 🔄 Early |
| **Toxicology** | Animal testing | In-silico ADMET prediction | 🔄 Partial |
| **Clinical Trials** | Human trials | Digital twin simulation | ⬜ Speculative |

#### 2.4.2 Model Treatment: Simulation Unlock Module

We model this as a **potential regime change** that could dramatically increase M_max for physical stages:

```
M_max^{unlock}(t) = M_max^{physical} + P_unlock(t) × (M_max^{cognitive} - M_max^{physical})
```

Where:
- `P_unlock(t)` = Probability that simulation tools achieve physical-trial equivalence by time t
- For wet lab (S4): P_unlock currently estimated at 0.05 (2025) → 0.30 (2035) → 0.60 (2050)
- **High uncertainty**: This is the most speculative component of our model

#### 2.4.3 Implications for Acceleration Ceiling

| Scenario | 2050 M_max (Wet Lab) | System Acceleration | Probability |
|----------|---------------------|---------------------|-------------|
| **No Unlock** | 2.5x | ~20x | 40% |
| **Partial Unlock** | 10x | ~50x | 45% |
| **Full Unlock** | 50x | ~100x+ | 15% |

**Key Insight:** The "Unlock" pathway represents the primary source of **upside uncertainty** in our model. If AI succeeds in creating validated simulation substitutes for physical experiments, the 10x ceiling could be shattered.

#### 2.4.4 What Would Enable the Unlock?

1. **Sufficient training data** from physical experiments to validate simulations
2. **Regulatory acceptance** of in-silico evidence (FDA, EMA)
3. **Multi-scale modeling** connecting molecular to organismal predictions
4. **Uncertainty quantification** for simulation predictions (knowing when to trust them)

**Paradox:** Physical experiments are needed to validate simulations that would replace physical experiments. This creates a **bootstrap problem** that may limit how quickly the Unlock can occur

---

## 3. HISTORICAL PARADIGM SHIFTS

### 3.1 Framework: Technology-Enabled Scientific Revolutions

**[P1 Fix: Distinguish capability vs. conceptual shifts per expert review]**

Each major scientific instrument/technology created shifts by changing **what questions could be asked**. However, we must distinguish:

| Shift Type | Definition | AI Potential | Predictability |
|------------|------------|--------------|----------------|
| **Capability Extension** | Expand what can be done within existing frameworks | High | Moderate |
| **Methodological Shift** | Change how research is conducted | High | Moderate |
| **Conceptual Paradigm Shift** | Change fundamental assumptions about nature | Uncertain | Low |

**Important Caveat:** The claim that "paradigm shift timelines are accelerating" (150yr → 25yr → 15yr) may reflect measurement artifacts (recency compression, survivorship bias) rather than genuine pattern. We treat historical calibration as indicative, not definitive.

### 3.2 Five Historical Case Studies

#### 3.2.1 Microscope (1600s-1700s)

| Aspect | Pre-Microscope | Post-Microscope |
|--------|----------------|-----------------|
| **Visible World** | Macro only (>1mm) | Micro world revealed |
| **Biology Paradigm** | Humoral theory | Cell theory, germ theory |
| **Time to Full Impact** | ~150 years | Slow manufacturing adoption |
| **Shift Type** | **Capability Extension** → enabled Conceptual Shift | Created microbiology |

**Key Insight:** Microscope enabled new observations, but the conceptual revolution (germ theory) required additional human intellectual work by Pasteur, Koch, etc.

#### 3.2.2 Telescope (1600s-1700s)

| Aspect | Pre-Telescope | Post-Telescope |
|--------|---------------|----------------|
| **Observable Universe** | Few thousand stars | Millions of objects |
| **Astronomy Paradigm** | Geocentric debates | Heliocentric confirmation |
| **Time to Paradigm Shift** | Galileo (1609) → Newton (1687) = ~80 years |
| **Data Multiplication** | 100x-1000x |

**Key Insight:** Provided overwhelming evidence but conceptual frameworks (Kepler, Newton) still required human genius.

#### 3.2.3 Human Genome Project (1990-2003)

| Metric | Pre-HGP (1990) | Post-HGP (2003) | Now (2024) | Acceleration |
|--------|----------------|-----------------|------------|--------------|
| Time per genome | 13 years | 3 months | 1 day | **4,700x** |
| Cost per genome | $3B | $10M | $200 | **15,000,000x** |
| Publications/year | ~500 | ~15,000 | ~80,000 | **160x** |

**Key Insight:** Front-loaded infrastructure investment (13 years) unlocked exponential returns. **Capability extension**, not conceptual paradigm shift.

#### 3.2.4 DNA Sequencing Revolution (2005-2015)

| Generation | Technology | Cost/Genome | Time | Acceleration |
|------------|------------|-------------|------|--------------|
| 1st (1977) | Sanger | $3B | 13 years | Baseline |
| 2nd (2005) | Illumina NGS | $10,000 | 1 week | 300,000x cost |
| 3rd (2015) | PacBio/Nanopore | $1,000 | 1 day | 3,000,000x cost |
| 4th (2024) | ONT portable | $200 | 6 hours | 15,000,000x cost |

**Pattern:**
```
Initial Breakthrough → Technology Democratization → New Questions Tractable → Field Explosion → Secondary Breakthroughs
```

#### 3.2.5 CRISPR (2012-present)

| Metric | Pre-CRISPR | Post-CRISPR | Acceleration |
|--------|------------|-------------|--------------|
| Gene knockout cost | $50,000 | $50 | **1,000x** |
| Time per edit | 6-12 months | 2-4 weeks | **10-20x** |
| Publications/year | ~200 (ZFN/TALEN) | ~15,000 | **75x** |
| Clinical trials | 0 | 70+ | ∞ |

**Key Insight:** CRISPR's impact came from **democratization**—making expert-only techniques accessible to all labs. **Methodological shift**, not conceptual paradigm shift.

### 3.3 Historical Calibration Summary

| Technology | Shift Type | Time to 10x Impact | Time to Field Transformation |
|------------|------------|--------------------|-----------------------------|
| Microscope | Capability → Conceptual | ~50 years | ~150 years |
| Telescope | Capability | ~30 years | ~150 years |
| HGP/Sequencing | Capability | ~5 years | ~25 years |
| CRISPR | Methodological | ~3 years | ~15 years (ongoing) |
| **AI (projected)** | **Capability + Methodological** | **~2-5 years** | **~15-25 years** |

**Uncertainty Note:** The projected AI timeline has wide uncertainty bounds. Historical analogy provides rough guidance, not precision.

---

## 4. RESEARCH PIPELINE DEFINITION

### 4.1 8-Stage Research Pipeline (Revised per Expert Review)

**[P1/P2 Fixes: Adjusted M_max, durations, added speed/quality split]**

| Stage | Name | τ₀ (mo) | M_max_speed | M_max_quality | p_success | k | Primary AI Impact |
|-------|------|---------|-------------|---------------|-----------|---|-------------------|
| S1 | **Literature Synthesis** | 3 | 100 | 30 | 0.95 | 1.2 | LLMs, knowledge graphs |
| S2 | **Hypothesis Generation** | 6 | 100 | 20 | 0.85* | 1.0 | AI-generated hypotheses |
| S3 | **Experimental Design** | 2 | 30 | 20 | 0.90 | 1.0 | Optimal design, simulation |
| S4 | **Data Generation (Wet Lab)** | 12 | 2.5 | 2.5 | 0.30 | 0.5 | Automation (physical limits) |
| S5 | **Data Analysis** | 3.5** | 100 | 50 | 0.95 | 1.2 | ML analysis, pattern detection |
| S6 | **Validation & Replication** | 8 | 2.5 | 2.5 | 0.50 | 0.5 | Social process limits |
| S7 | **Writing & Peer Review** | 6 | 10 | 5 | 0.70 | 0.6 | AI writing, but social bottleneck |
| S8 | **Publication & Dissemination** | 3 | 20 | 10 | 0.95 | 0.8 | Preprints, AI summarization |

*S2 p_success = 0.85 for hypothesis generation; translation failure captured at S4 (wet lab)
**S5 increased from 2mo to 3.5mo per bioinformatics expert review

**Total baseline pipeline:** ~43.5 months (~3.6 years) from hypothesis to publication

### 4.2 Speed vs. Quality Acceleration (P2 Fix)

**[New per expert review - Panel A2]**

For cognitive stages, we distinguish:
- **M_max_speed**: How much faster AI can process (e.g., read papers, run analyses)
- **M_max_quality**: How much better AI output quality matches expert human work

**Effective M_max** uses the **geometric mean** or **minimum** depending on task:

$$M_i^{eff}(t) = \sqrt{M_i^{speed}(t) \times M_i^{quality}(t)}$$

Or for quality-critical stages:

$$M_i^{eff}(t) = \min(M_i^{speed}(t), M_i^{quality}(t))$$

### 4.3 AI Reliability Factor r(t) (P2 Fix)

**[New per expert review - Panel A2]**

AI reasoning reliability grows over time as models improve:

$$r_i(t) = r_{i,0} + (r_{i,max} - r_{i,0}) \cdot (1 - e^{-\lambda_r \cdot (t - t_0)})$$

| Stage | r₀ (2024) | r_max | λ_r | Rationale |
|-------|-----------|-------|-----|-----------|
| S1 Literature | 0.7 | 0.95 | 0.3 | Citation verification needed |
| S2 Hypothesis | 0.3 | 0.80 | 0.2 | Novel reasoning unreliable |
| S3 Design | 0.5 | 0.90 | 0.25 | Domain expertise gaps |
| S5 Analysis | 0.6 | 0.95 | 0.3 | Interpretation limits |

**Modified Service Rate:**

$$\mu_i(t) = \mu_i^0 \cdot M_i^{eff}(t) \cdot r_i(t)$$

### 4.4 Bottleneck Identification

**Primary Bottlenecks (time-varying):**

| Period | Primary Bottleneck | Secondary Bottleneck | Reason |
|--------|-------------------|---------------------|--------|
| 2024-2030 | S4 Wet Lab | S6 Validation | Biological timescales |
| 2030-2040 | S6 Validation | S4 Wet Lab | Social process limits |
| 2040-2050 | S6 Validation | Infrastructure | Institutional change slow |

**Cross-cutting Bottleneck:** Infrastructure (compute, data access) - see Section 8

---

## 5. PARADIGM SHIFT MODULE (PSM)

### 5.1 Concept

The PSM captures **capability extensions and methodological shifts** that go beyond incremental acceleration.

**[P1 Fix: Clarified terminology per expert review - Panel C2]**

### 5.2 Three Types of AI-Enabled Shifts

| Type | Description | Weight Range* | Examples |
|------|-------------|---------------|----------|
| **Type I: Scale** | 10x-1000x more data/throughput | 2-5x (μ=3, σ=1) | GNoME, LLM synthesis |
| **Type II: Accessibility** | Expert techniques → Everyone | 3-10x (μ=6, σ=2) | AlphaFold democratization |
| **Type III: Capability Extension** | Previously intractable → Tractable | 5-30x (μ=15, σ=8) | Protein folding, de novo design |

*Weights now specified as distributions, not point estimates (P1 fix)

### 5.3 Mathematical Formulation

**[P1 Fix: Added uncertainty bounds per expert review]**

**Paradigm Shift Multiplier (PSM):**

$$\text{PSM}(t) = 1 + \sum_{j \in \text{active shifts}} w_j \cdot (1 - e^{-\lambda_j \cdot (t - t_j)})$$

Where:
- $w_j \sim \text{LogNormal}(\mu_j, \sigma_j)$ — magnitude weight with uncertainty
- $\lambda_j$ = adoption rate of shift $j$
- $t_j$ = onset time of shift $j$

**Calibrated Parameters (with uncertainty):**

| Shift Type | w (mean) | w (95% CI) | λ (historical) | λ (AI-era) |
|------------|----------|------------|----------------|------------|
| Type I (Scale) | 3 | [1.5, 6] | 0.10 year⁻¹ | 0.30 year⁻¹ |
| Type II (Accessibility) | 6 | [3, 12] | 0.15 year⁻¹ | 0.40 year⁻¹ |
| Type III (Capability) | 15 | [5, 45] | 0.05 year⁻¹ | 0.20 year⁻¹ |

### 5.4 PSM Architecture (P2 Fix - Clarification)

**[Clarified per expert review - Panel E2]**

To avoid double-counting:

1. **Stage-specific PSM_i(t)**: Affects service rate μ_i(t) for stages where the shift applies
2. **System PSM_sys(t)**: NOT used separately—captured through stage-specific effects

**Correct formulation (no double-counting):**

$$\mu_i(t) = \mu_i^0 \cdot M_i^{eff}(t) \cdot r_i(t) \cdot \text{PSM}_i(t)$$

$$\Theta(t) = \min_i \mu_i^{eff}(t)$$

$$R(t) = \frac{\Theta(t)}{\Theta(t_0)}$$

**PSM does NOT multiply R(t) again**—this was an error in the original formulation.

### 5.5 Stage-Specific PSM Sensitivity

| Stage | Type I | Type II | Type III |
|-------|--------|---------|----------|
| S1 Literature | 0.9 | 0.5 | 0.2 |
| S2 Hypothesis | 0.5 | 0.5 | **1.0** |
| S3 Design | 0.6 | 0.8 | 0.5 |
| S4 Wet Lab | 0.2 | 0.3 | 0.2 |
| S5 Analysis | **1.0** | 0.7 | 0.6 |
| S6 Validation | 0.3 | 0.3 | 0.2 |
| S7 Writing | 0.6 | 0.5 | 0.1 |
| S8 Publication | 0.8 | 0.6 | 0.1 |

---

## 6. MATHEMATICAL FRAMEWORK

### 6.1 Core Variables

| Symbol | Description | Units | Distribution (for UQ) |
|--------|-------------|-------|----------------------|
| $t$ | Time | years | - |
| $t_0$ | Baseline year (2024) | years | - |
| $T$ | Horizon year (2050) | years | - |
| $i$ | Stage index | 1...8 | - |
| $A(t)$ | AI capability | dimensionless | - |
| $g$ | AI growth rate | year⁻¹ | LogNormal(μ, σ) |
| $M_i(t)$ | AI multiplier | dimensionless | - |
| $M_i^{\max}$ | Maximum multiplier | dimensionless | LogNormal |
| $r_i(t)$ | AI reliability factor | [0, 1] | - |
| $p_i$ | Success probability | probability | Beta(α, β) |
| $C(t)$ | Infrastructure capacity | dimensionless | - |
| $\Theta(t)$ | System throughput | projects/year | - |
| $R(t)$ | Progress rate | dimensionless | - |
| $Y(t)$ | Cumulative progress | equivalent years | - |

### 6.2 Core Equations

**Equation 1: AI Capability Growth (Logistic)**
$$A(t) = \frac{A_{\text{ceiling}}}{1 + (A_{\text{ceiling}} - 1) \cdot e^{-g \cdot (t - t_0)}}$$

Where $A_{\text{ceiling}} \sim \text{Uniform}(50, 500)$ for uncertainty quantification.

**Equation 2: AI Acceleration Multiplier**
$$M_i(t) = 1 + (M_i^{\max} - 1) \cdot (1 - A(t)^{-k_i})$$

**Equation 3: Effective Multiplier (with speed/quality and reliability)**
$$M_i^{eff}(t) = \sqrt{M_i^{speed}(t) \cdot M_i^{quality}(t)} \cdot r_i(t)$$

**Equation 4: Service Rate with PSM**
$$\mu_i(t) = \mu_i^0 \cdot M_i^{eff}(t) \cdot \text{PSM}_i(t)$$

**Equation 5: Infrastructure-Constrained Service Rate**
$$\mu_i^{infra}(t) = \min(\mu_i(t), C_i(t))$$

Where $C_i(t)$ is infrastructure capacity for stage $i$ (see Section 8).

**Equation 6: Effective Service Rate**
$$\mu_i^{\text{eff}}(t) = \mu_i^{infra}(t) \cdot p_i(t)$$

**Equation 7: System Throughput**
$$\Theta(t) = \min_{i} \mu_i^{\text{eff}}(t)$$

**[P1 Note: This min formula is a simplification. In v0.3+, consider queuing-theoretic formulation to capture variance effects.]**

**Equation 8: Progress Rate**
$$R(t) = \frac{\Theta(t)}{\Theta(t_0)}$$

**Equation 9: Cumulative Progress**
$$Y(T) = \int_{t_0}^{T} R(t) \, dt$$

### 6.3 Parameter Distributions for Uncertainty Quantification

**[P1 Fix: Explicit distributions per expert review - Panel E1]**

| Parameter | Distribution | Parameters | Bounds | Rationale |
|-----------|--------------|------------|--------|-----------|
| g_ai | LogNormal | μ=log(0.4), σ=0.35 | [0.15, 0.80] | Calibrated to compute trends |
| A_ceiling | Uniform | - | [50, 500] | True ceiling unknown |
| M_max_cognitive | LogNormal | μ=log(25), σ=0.5 | [5, 100] | Wide uncertainty |
| M_max_physical | LogNormal | μ=log(2), σ=0.3 | [1.3, 4] | Biological limits |
| p_S4 (wet lab) | Beta | α=3, β=7 | [0, 1] | Mean ~0.30 |
| p_S6 (validation) | Beta | α=5, β=5 | [0, 1] | Mean ~0.50 |
| PSM_w_TypeIII | LogNormal | μ=log(15), σ=0.6 | [5, 45] | High uncertainty |

### 6.4 Correlated Parameter Structure

**[P1 Fix: Acknowledge correlations per expert review - Panel E1]**

Parameters are not independent. Key correlations to model:

| Parameter Pair | Correlation | Rationale |
|----------------|-------------|-----------|
| g_ai ↔ M_max_cognitive | +0.6 | Faster AI → higher ceilings |
| PSM_TypeII ↔ PSM_TypeIII | +0.4 | Shifts often co-occur |
| p_S4 ↔ p_S6 | +0.3 | Failed experiments reduce validation success |
| M_max_physical ↔ Infrastructure | +0.5 | Automation requires infrastructure |

**Implementation:** Use Iman-Conover method for correlated sampling in Monte Carlo.

---

## 7. AI FAILURE MODES & RISKS

### 7.1 Overview

**[P1 Fix: New section per expert review - Panel A3]**

AI acceleration carries risks that could reduce or reverse gains. The Conservative scenario explicitly models these.

### 7.2 Failure Mode Taxonomy

| Failure Mode | Description | Affected Stages | Impact |
|--------------|-------------|-----------------|--------|
| **Hallucination Propagation** | AI generates false citations/claims that propagate through pipeline | S1, S2 | Wasted wet lab resources |
| **Monoculture Risk** | All researchers use same AI → hypothesis space narrows | S2 | Reduced discovery diversity |
| **Automation Bias** | Over-trust in AI outputs reduces human error-checking | S1-S5 | Increased false positives |
| **Quality Degradation** | Immature AI deployment degrades success rates | S1-S3 | Temporary p_success reduction |
| **Infrastructure Overwhelm** | AI generates more hypotheses than validation can handle | S6 | Queue explosion |

### 7.3 Failure Mode Parameters

**Quality Degradation Factor (Conservative Scenario, 2025-2032):**

$$q_i(t) = 1 - \delta_i \cdot e^{-\gamma \cdot (t - t_0)}$$

Where:
- $\delta_i$ = maximum degradation (0.10-0.20 for cognitive stages)
- $\gamma$ = recovery rate (0.15 year⁻¹)

**Modified p_success during degradation phase:**

$$p_i^{adj}(t) = p_i \cdot q_i(t)$$

### 7.4 Validation Burden Effect

**[P1 Fix: Inter-stage feedback per expert review - Panel A3]**

When S2 throughput increases faster than S6 capacity:

$$p_{S6}(t) = p_{S6}^0 \cdot \frac{1}{1 + \beta \cdot (\Theta_{S2}(t) / \Theta_{S2}^0 - 1)}$$

Where $\beta$ = 0.1 (validation burden elasticity)

This creates negative feedback: faster hypothesis generation can temporarily reduce validation success.

---

## 8. INFRASTRUCTURE CONSTRAINTS

### 8.1 Overview

**[P2 Fix: New section per expert review - Panel B3]**

AI acceleration is bounded by infrastructure availability, not just AI capability.

### 8.2 Infrastructure Components

| Component | Current Bottleneck | Growth Rate | Ceiling |
|-----------|-------------------|-------------|---------|
| **Compute (GPU/TPU)** | Queue times, cost | g_compute = 0.25 | 100x current |
| **Data Access** | Federated data agreements | g_data = 0.15 | Regulatory dependent |
| **Personnel** | Bioinformatics expertise | g_talent = 0.10 | Training pipeline |

### 8.3 Infrastructure Capacity Function

$$C_i(t) = C_i^0 \cdot (1 + (C_i^{max} - 1) \cdot (1 - e^{-g_{infra} \cdot (t - t_0)}))$$

| Stage | C₀ | C_max | g_infra | Constraint Type |
|-------|----|----|---------|-----------------|
| S1 Literature | 1.0 | 50 | 0.25 | Compute |
| S5 Analysis | 1.0 | 30 | 0.20 | Compute + Personnel |
| S4 Wet Lab | 1.0 | 5 | 0.10 | Lab capacity |
| S6 Validation | 1.0 | 3 | 0.08 | Social + Personnel |

### 8.4 Effective Throughput

$$\mu_i^{infra}(t) = \min(\mu_i(t), C_i(t) \cdot \mu_i^0)$$

When AI capability exceeds infrastructure capacity, throughput is infrastructure-limited.

---

## 9. RESEARCH SYSTEM TRANSFORMATION

### 9.1 Five Pillars of Transformation

**[Updated: Added Pillar 5 per expert review]**

| Pillar | Focus | Key Stakeholder |
|--------|-------|-----------------|
| **1. Institutional Redesign** | Organization structure | Universities, Institutes |
| **2. Workforce & Training** | Skills, education | Researchers, Students |
| **3. Funding Restructuring** | Grant mechanisms | Funders |
| **4. Quality Assurance** | Integrity, reproducibility | All |
| **5. International Coordination** | Global governance | Governments, Intl. Bodies |

### 9.2 Pillar 1: Institutional Redesign

| Current State | Transformation | Timeline* |
|---------------|----------------|-----------|
| Siloed departments | **Convergence hubs** (AI + domain) | 2028-2035 |
| Individual PI model | **Team science with AI specialists** | 2027-2033 |
| Fixed lab structures | **Flexible, project-based teams** | 2033-2042 |
| Sequential workflows | **Parallel hypothesis testing** | 2035-2045 |
| Publication as endpoint | **Continuous knowledge streams** | 2045-2055 |

*Timelines extended by 3-5 years per expert review (Panel D1)

### 9.3 Pillar 2: Workforce & Training Evolution

See Section 11 for detailed treatment.

### 9.4 Pillar 3: Funding & Incentive Restructuring

**[P1 Fix: Extended timelines per expert review]**

| Current Model | Transformation | Timeline | Mechanism |
|---------------|----------------|----------|-----------|
| 3-5 year grants | **Shorter, iterative cycles** | 2032-2042 | Pilot programs first |
| Conservative funding | **Paradigm shift funds** | 2028-2035 | 5-10% of portfolio |
| Single-PI grants | **Team & infrastructure grants** | 2030-2040 | Supplement programs |
| Publication metrics | **Multi-dimensional impact** | 2035-2045 | DORA alignment |
| National competition | **International consortia** | 2030-2045 | Bilateral first |

**Pilot Program Pathway (per expert review):**

| Pilot | Agency | Timeline | Scale |
|-------|--------|----------|-------|
| Rapid AI Hypothesis Grants | NIGMS | 2027-2030 | $50M |
| Compute Allocation for Biology | NSF BIO + ACCESS | 2026-2029 | $100M |
| Team Science PhD Supplement | NIH Training | 2028-2032 | 500 awards |

### 9.5 Pillar 4: Quality Assurance & Integrity

| Challenge | Solution | Timeline | Reference Framework |
|-----------|----------|----------|---------------------|
| Replication crisis amplified | **Automated replication pipelines** | 2028-2035 | OSF, CurateScience |
| AI hallucinations | **Verification requirements** | 2026-2030 | To be developed |
| Black-box models | **Explainability standards** | 2030-2038 | EU AI Act, DARPA XAI |
| Data quality issues | **Provenance tracking** | 2028-2035 | FAIR principles |
| Gaming of metrics | **Multi-dimensional impact** | 2035-2045 | DORA, Leiden |

---

## 10. INTERNATIONAL COORDINATION (PILLAR 5)

### 10.1 Overview

**[P1 Fix: New pillar per expert review - Panel D3]**

AI research acceleration is inherently global. National-only policy is insufficient.

### 10.2 International Coordination Roadmap

| Component | 2025-2032 | 2032-2042 | 2042-2055 |
|-----------|-----------|-----------|-----------|
| **Regulatory Alignment** | Monitor EU AI Act, OECD participation | Harmonize research exemptions | Global research AI governance |
| **Data Sharing** | Bilateral agreements (US-EU, US-UK) | Regional data commons | Global research data infrastructure |
| **Compute Access** | National compute strategies | Regional consortia | Global allocation system |
| **Talent Mobility** | Preserve visa programs | AI researcher visa category | Harmonized credentials |
| **Standards** | ISO/IEC AI standards contribution | Common research AI standards | Unified integrity protocols |

### 10.3 Key International Bodies

| Body | Role | Engagement Priority |
|------|------|---------------------|
| **OECD STI** | Science policy coordination | High |
| **G7/G20 Science** | Political commitment | High |
| **ISO/IEC JTC 1/SC 42** | AI standards | Medium |
| **UNESCO** | Ethics frameworks | Medium |
| **Bilateral S&T Agreements** | Data/talent flows | High |

### 10.4 Regulatory Landscape Impact

| Jurisdiction | Key Regulation | Research Impact | Model Treatment |
|--------------|----------------|-----------------|-----------------|
| EU | AI Act (2024) | Transparency requirements | Compliance cost factor |
| US | Executive Orders | Sector-specific | Moderate constraints |
| UK | Pro-innovation | Regulatory sandbox | Lower barriers |
| China | DSL, AI regulations | Data localization | Separate track |

---

## 11. RESEARCHER EDUCATION REFORM

### 11.1 The Problem: Training Model Mismatch

| Current Training (1960s Design) | AI-Era Research Reality |
|---------------------------------|-------------------------|
| 5-7 year PhD, single discipline | Rapid, interdisciplinary problems |
| Apprenticeship under one PI | Team science, multiple mentors |
| Learn techniques by repetition | Techniques automated in 2-3 years |
| Methods: manual, artisanal | Methods: computational, scaled |
| Success = publications in niche | Success = impact across domains |
| Career path: postdoc → PI | Multiple paths: industry, AI labs |

### 11.2 Three Phases of Education Transformation

#### Phase 1: Immediate Adaptations (2025-2032)

| Component | Current | Transformation |
|-----------|---------|----------------|
| **Coursework** | Domain-only | + AI/ML foundations required |
| **Tools** | Manual analysis | AI tools as standard lab skills |
| **Data skills** | Basic statistics | Large-scale data engineering |
| **Collaboration** | Optional | Required team projects |
| **Timeline** | Fixed 5-7 years | Flexible, competency-based |

#### Phase 2: Structural Reforms (2032-2042)

**Modular PhD Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│  MODULAR PHD (Total: 4-6 years flexible)                │
├─────────────────────────────────────────────────────────┤
│  Module 1: Foundations (1 year)                         │
│  - Domain fundamentals + AI/computational core          │
│  - Qualifying exam: demonstrate dual competency         │
├─────────────────────────────────────────────────────────┤
│  Module 2: Research Immersion (1.5-2 years)             │
│  - Primary research (can be team-based)                 │
│  - Required: one AI and one domain project              │
├─────────────────────────────────────────────────────────┤
│  Module 3: Integration & Impact (1-2 years)             │
│  - Translation, dissemination, application              │
│  - Options: startup, policy, academic, industry track   │
├─────────────────────────────────────────────────────────┤
│  Module 4 (Optional): Specialization (1 year)           │
│  - Deep expertise for academic track                    │
└─────────────────────────────────────────────────────────┘
```

### 11.3 Global Mobility Considerations

**[P2 Fix: Added per expert review - Panel D3]**

| Issue | Current State | Needed Reform |
|-------|---------------|---------------|
| Credential recognition | National only | Bologna-style international framework |
| Researcher mobility | Visa restrictions | AI researcher visa category |
| Training standards | Varies by institution | Core competency certification |
| Brain drain risk | Uncoordinated | Coordinated with immigration policy |

### 11.4 Education Transformation Metrics

| Metric | Current | 2032 Target | 2042 Target |
|--------|---------|-------------|-------------|
| PhDs with AI competency | ~10% | 50% | 90% |
| Time to PhD (median) | 6.3 years | 5.5 years | 4.5 years |
| Team dissertations | <5% | 20% | 40% |
| Industry collaboration | ~15% | 40% | 60% |
| International mobility | ~25% | 35% | 45% |

---

## 12. CASE STUDIES (2022-2025)

### 12.1 Overview: Five Deep-Dive Case Studies

| Case Study | Domain | Year | Shift Type | Impact |
|------------|--------|------|------------|--------|
| **AlphaFold 2/3** | Structural Biology | 2021-2024 | Type III (Capability) | 50-year problem solved |
| **ESM-3** | Protein Design | 2024 | Type III (Capability) | De novo protein generation |
| **GNoME** | Materials Science | 2023 | Type I (Scale) | 2.2M new materials |
| **AlphaGeometry** | Mathematics | 2024 | Type III (Capability) | IMO-level proofs |
| **LLM Synthesis** | All Fields | 2022-2025 | Type I+II | Literature 30-50x faster |

### 12.2 Case Study 1: AlphaFold (2021-2024)

**Impact Metrics:**

| Metric | Pre-AlphaFold | Post-AlphaFold | Acceleration |
|--------|---------------|----------------|--------------|
| Time per structure | Months-years | Seconds | **10,000x+** |
| Cost per structure | $100K+ | ~$0 | **∞** |
| Structures available | ~180,000 | 200M+ | **1,000x** |
| Publications citing | 0 | 20,000+ | N/A |

**2024 Nobel Prize:** Demis Hassabis and John Jumper awarded Chemistry Nobel

**Shift Type:** Type III (Capability Extension) — made intractable problem tractable

**Key Lessons:**
- Type III shifts can happen suddenly (2-3 years from breakthrough to field transformation)
- Democratization (free access) multiplies impact
- **This was capability extension, not conceptual paradigm shift** (protein folding was a known problem)

### 12.3 Case Study 2: ESM-3 (Meta, 2024)

**Key Capability:** De novo protein design from text prompts

**Shift Type:** Type III (Capability Extension)

**Implications:**
- Biology becomes "programmable"
- Design space expands exponentially
- **Verification remains bottleneck** (consistent with our model)

### 12.4 Case Study 3: GNoME (DeepMind, 2023)

**Impact:** 2.2 million new stable materials predicted (800x expansion)

**Shift Type:** Type I (Scale)

**Key Lesson:** Hypothesis generation at unprecedented scale, but validation becomes bottleneck (consistent with our S6 analysis)

### 12.5 Case Study 4: AlphaGeometry (DeepMind, 2024)

**Achievement:** Solved 25/30 IMO geometry problems (silver medal level)

**Shift Type:** Type III (demonstrates AI reasoning capability)

**Relevance:** Shows AI can perform creative scientific reasoning, not just pattern matching

### 12.6 Case Study 5: LLM Synthesis (2022-2025)

| Task | Traditional | With LLMs | Acceleration |
|------|-------------|-----------|--------------|
| Literature review | 2-3 months | 2-7 days | **15-45x** |
| Paper summarization | 30 min/paper | 2 min/paper | **15x** |
| Cross-domain synthesis | Very difficult | Feasible | High |

**Note:** LLM synthesis claim (30-50x) revised to 15-45x with range reflecting quality verification needs.

---

## 13. SCENARIO DEFINITIONS

### 13.1 Four Core Scenarios

**[Updated: Added AI Winter scenario per expert review]**

| Scenario | g (year⁻¹) | Description | PSM Types Active | P(Scenario) |
|----------|------------|-------------|------------------|-------------|
| **AI Winter** | 0.15 | Stagnation, safety pause, regulation | Type I only | 10% |
| **Conservative** | 0.30 | Slow progress, limited adoption, failure modes | Type I only | 25% |
| **Baseline** | 0.40* | Current trends, moderate adoption | Type I + II | 45% |
| **Ambitious** | 0.55* | Rapid progress, system adaptation | Type I + II + III | 20% |

*g values reduced per expert review calibration

### 13.2 Scenario Parameters

#### AI Winter Scenario (New)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| g_ai | 0.15 | Scaling laws plateau, safety concerns |
| M_max_cognitive | 15 | Limited capability growth |
| Quality degradation | Active 2025-2035 | Immature deployment |
| Infrastructure | Slow growth | Investment pullback |
| System adaptation | Stalled | Institutional resistance |

#### Conservative Scenario

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| g_ai | 0.30 | AI progress slows, deployment lag |
| M_max_cognitive | 30 | Moderate capability |
| Quality degradation | Active 2025-2030 | Early deployment problems |
| Failure modes | Modeled | Hallucination, monoculture risks |
| Infrastructure | g_infra = 0.15 | Slow capacity growth |

#### Baseline Scenario

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| g_ai | 0.40 | Current trends, calibrated |
| M_max_cognitive | 50 | Strong reasoning capability |
| Quality degradation | Minimal | Mature deployment |
| Infrastructure | g_infra = 0.20 | Moderate growth |
| System adaptation | Moderate | Gradual change |

#### Ambitious Scenario

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| g_ai | 0.55 | Strong progress, rapid adoption |
| M_max_cognitive | 80 | Near-human reasoning |
| PSM_Type_III | Active | Capability extensions |
| Infrastructure | g_infra = 0.30 | Strong investment |
| System adaptation | Rapid | Institutional reform |

---

## 14. EXPERT REVIEWER PANEL STRUCTURE

### 14.1 Five Panels (3 Reviewers Each = 15 Experts)

| Panel | Focus | Expertise |
|-------|-------|-----------|
| **A: AI/ML Experts** | AI capability projections | ML researchers, AI forecasters |
| **B: Domain Scientists** | Biology pipeline accuracy | Lab biologists, PIs |
| **C: Science of Science** | Scientometrics | Bibliometricians, historians |
| **D: Research Policy** | Funding, institutional | Program officers, policy experts |
| **E: Methodological Critics** | Model validity | Statisticians, systems modelers |

### 14.2 Review Completed

Initial expert review completed January 14, 2026. See Section 20 for summary.

---

## 15. ITERATION ROADMAP

### 15.1 Version Overview

| Version | Focus | Key Addition | Status |
|---------|-------|--------------|--------|
| **v0.1** | Core Framework | 8-stage pipeline, basic model, MC simulation | ✅ Complete |
| **v0.2** | Historical Calibration | Formal statistical calibration against 5 paradigm shifts | ✅ Complete |
| **v0.3** | Infrastructure Module | Compute, data, talent constraints | ✅ Integrated in v0.1 |
| **v0.4** | AI Failure Modes | Quality degradation, feedback effects | ✅ Integrated in v0.1 |
| **v0.5** | Case Study Validation | AlphaFold, GNoME calibration | ⬜ Planned |
| **v0.6** | Multi-Domain | Biology subfields comparison | ⬜ Planned |
| **v0.7** | System Transformation | 5 Pillars with international | ⬜ Planned |
| **v0.8** | Education Reform | Training pipeline model | ⬜ Planned |
| **v0.9** | Policy Analysis | Intervention ROI | ⬜ Planned |
| **v1.0** | Uncertainty Quantification | Full UQ, Sobol | ⬜ Planned |
| **v1.1** | Expert Review Integration | Panel feedback | ⬜ Planned |

---

## 16. PAPER OUTLINE

### 16.1 Proposed Title

**"Accelerating Science in the AI Era: A Quantitative Framework for Research Transformation"**

### 16.2 Structure (~24,000 words)

1. **Abstract** (~300 words)
2. **Introduction** (~2,500 words)
3. **Background** (~3,000 words)
   - Historical capability extensions (not paradigm shifts)
   - Amodei's framework
   - Current research system limitations
4. **Model Framework** (~4,000 words)
   - Pipeline, PSM, failure modes
   - Infrastructure constraints
5. **Case Studies** (~3,000 words)
6. **Results: Acceleration Forecasts** (~3,500 words)
7. **Research System Transformation** (~4,500 words)
   - 5 pillars including international coordination
8. **Policy Recommendations** (~2,500 words)
9. **Discussion** (~2,500 words)
   - Limitations, risks, caveats
10. **Conclusion** (~500 words)

---

## 17. KEY DECISIONS LOG

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-14 | Spin-off from drug development model | Focus on basic/translational research |
| 2026-01-14 | 8-stage pipeline | Research ends at publication |
| 2026-01-14 | Include PSM | Capture capability extensions |
| 2026-01-14 | Three shift types (I/II/III) | Scale, Accessibility, Capability |
| 2026-01-14 | Success metric: Validated Discoveries | Operationally defined |
| 2026-01-14 | Include system transformation | Beyond forecasting to prescription |
| 2026-01-14 | Three scenarios | Conservative/Baseline/Ambitious |
| 2026-01-14 | 15-expert panel review | Comprehensive validation |
| 2026-01-14 | **P1: Wet Lab M_max → 2.5x** | Biological timescales irreducible |
| 2026-01-14 | **P1: Add Pillar 5 (International)** | Global coordination required |
| 2026-01-14 | **P1: Add AI failure modes** | Risks must be modeled |
| 2026-01-14 | **P1: Operationalize Validated Discoveries** | Measurable metric |
| 2026-01-14 | **P1: Add parameter distributions** | Enable proper UQ |
| 2026-01-14 | **P1: Distinguish capability vs. paradigm shifts** | Precision in terminology |
| 2026-01-14 | **P2: Split speed/quality M_max** | Cognitive stages need distinction |
| 2026-01-14 | **P2: Add reliability factor r(t)** | AI quality improves over time |
| 2026-01-14 | **P2: Extend policy timelines** | Federal reform takes longer |
| 2026-01-14 | **P2: Add infrastructure constraints** | Compute/data/talent limits |
| 2026-01-14 | **P2: Validation M_max → 2.5x** | Social process limits |
| 2026-01-14 | **P2: Data Analysis → 3.5mo** | Real bioinformatics workflows |
| 2026-01-14 | **P2: Clarify PSM architecture** | Avoid double-counting |
| 2026-01-14 | **P2: Add AI Winter scenario** | Tail risk |

---

## 18. DATA SOURCES & REFERENCES

### 18.1 Primary References

**Theoretical Framework:**
- Amodei, D. (2024). "Machines of Loving Grace." [darioamodei.com](https://www.darioamodei.com/essay/machines-of-loving-grace)
- DeepMind (2024). "A New Golden Age of Discovery."

**AI Case Studies:**
- Jumper et al. (2021). Nature (AlphaFold 2)
- Abramson et al. (2024). Nature (AlphaFold 3)
- Merchant et al. (2023). Nature (GNoME)
- Trinh et al. (2024). Nature (AlphaGeometry)

### 18.2 Methodological References

- Saltelli et al. (2010). Global Sensitivity Analysis
- Sobol (2001). Sensitivity analysis methods
- Rogers (1962). Diffusion of Innovations
- Kuhn (1962). Structure of Scientific Revolutions

---

## 19. OPEN QUESTIONS

### 19.1 For v0.2+

- [ ] How to formally calibrate PSM weights from historical data?
- [ ] What queuing model best captures pipeline variance effects?
- [ ] How to validate model prospectively (milestones)?

### 19.2 For Discussion

- [ ] Should we add discipline-specific sub-models?
- [ ] How to handle rapidly evolving AI capabilities?
- [ ] What publication venue is most appropriate?

---

## 20. EXPERT REVIEW SUMMARY

### 20.1 Review Completed: January 14, 2026

15-expert simulated panel review identified 10 P1 (Critical) and 8 P2 (Important) issues.

### 20.2 P1 Issues Addressed

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Wet Lab M_max too optimistic | Revised 3x → 2.5x |
| 2 | "Validated Discoveries" undefined | Added operational definition (Section 1.4) |
| 3 | g_ai lacks empirical grounding | Revised baseline 0.50 → 0.40 with distribution |
| 4 | Paradigm shift claim conflates categories | Distinguished capability vs. conceptual (Section 3.1) |
| 5 | No international coordination | Added Pillar 5 (Section 10) |
| 6 | PSM weights false precision | Added uncertainty ranges (Section 5.3) |
| 7 | Throughput formula simplified | Noted limitation, planned enhancement (Section 6.2) |
| 8 | No AI failure modes | Added Section 7 |
| 9 | Federal timeline unrealistic | Extended by 3-5 years (Section 9) |
| 10 | Undefined distributions | Added explicit distributions (Section 6.3) |

### 20.3 P2 Issues Addressed

| # | Issue | Resolution |
|---|-------|------------|
| 11 | Speed vs. quality M_max | Added split (Section 4.1, 4.2) |
| 12 | Add reliability factor r(t) | Added (Section 4.3) |
| 13 | Validation M_max too optimistic | Revised 5x → 2.5x |
| 14 | Data Analysis duration too short | Revised 2mo → 3.5mo |
| 15 | Infrastructure constraints missing | Added Section 8 |
| 16 | PSM double-counting risk | Clarified architecture (Section 5.4) |
| 17 | Global mobility in education | Added (Section 11.3) |
| 18 | Production-function critique | Acknowledged; multi-dimensional impact noted |

---

*Last updated: January 14, 2026*
*Version: v0.1-draft (Post Expert Review)*
*Status: P1 + P2 Issues Addressed, Ready for Implementation*
