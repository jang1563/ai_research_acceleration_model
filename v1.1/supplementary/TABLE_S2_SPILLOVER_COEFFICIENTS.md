# Supplementary Table S2: Cross-Domain Spillover Coefficients

## AI Research Acceleration Model v1.1

**Purpose**: Document the rationale, sources, and uncertainty for spillover effects between domains.

---

## S2.1 Spillover Matrix

| Source Domain | Target Domain | Coefficient | Range | Source | Notes |
|--------------|---------------|-------------|-------|--------|-------|
| Structural Biology | Drug Discovery | 0.25 | 0.15-0.35 | Sledz & Caflisch (2018); SBDD literature | AlphaFold structures enable structure-based drug design |
| Structural Biology | Protein Design | 0.30 | 0.20-0.40 | Protein engineering case studies | Structural understanding enables better designs |
| Protein Design | Drug Discovery | 0.12 | 0.08-0.18 | Biologics development literature | Designed proteins as therapeutics |
| Clinical Genomics | Drug Discovery | 0.08 | 0.04-0.12 | Pharmacogenomics literature | Variant interpretation guides drug targeting |
| Drug Discovery | Clinical Genomics | 0.04 | 0.02-0.08 | Drug-gene interaction databases | Weak reverse effect |
| Materials Science | Structural Biology | 0.03 | 0.01-0.05 | Cryo-EM methodology papers | Material advances for sample prep |
| Protein Design | Materials Science | 0.04 | 0.02-0.08 | Protein-based materials literature | Bio-inspired materials |
| Clinical Genomics | Protein Design | 0.04 | 0.02-0.08 | Variant-informed design | Understanding natural variation aids design |

---

## S2.2 Spillover Methodology

### Theoretical Framework

Based on R&D spillover literature (Griliches, 1992; Jaffe, 1986; Jaffe et al., 1993):

**Spillover Effect Formula**:
```
effect_ij = log(1 + accel_i - 1) × coefficient_ij × lag_factor(t)
```

Where:
- `effect_ij`: Spillover effect from domain i to domain j
- `accel_i`: Acceleration in source domain i
- `coefficient_ij`: Spillover coefficient (from Table S2.1)
- `lag_factor(t)`: Time-dependent lag function

### Logarithmic Transformation

**Rationale**: Marginal benefit decreases as source acceleration grows
- Prevents explosive compounding of spillovers
- Consistent with diminishing returns in R&D
- Source: Griliches (1992) R&D spillover methodology

### Lag Function

**Formula**: `lag_factor(t) = 1 - exp(-t / lag_years)`

Where `lag_years = 2.0` (average time for spillover to materialize)

**Rationale**:
- Spillovers require knowledge diffusion (publications, hiring)
- Technology transfer takes 1-3 years in biotech (Zucker et al., 2002)
- 2-year average based on patent citation lag studies

### Caps and Bounds

| Cap Type | Value | Rationale |
|----------|-------|-----------|
| Individual spillover cap | 0.4 | Prevents single source from dominating |
| Total spillover cap | 0.6 | Diminishing returns to multiple sources |
| Minimum | 0.0 | No negative spillovers in model |

---

## S2.3 Spillover Pathways by Domain Pair

### Structural Biology → Drug Discovery (0.25)

**Mechanism**: Structure-based drug design (SBDD)
- AlphaFold structures used for virtual screening
- Binding site identification for lead optimization
- Target druggability assessment

**Evidence**:
- Sledz & Caflisch (2018): "Protein structure-based drug design" review
- ~30% of FDA-approved drugs 2020-2024 used structure-guided design
- AlphaFold structures cited in 500+ drug discovery papers (2022-2024)

**Uncertainty**: Range 0.15-0.35 reflects variability across drug programs

### Structural Biology → Protein Design (0.30)

**Mechanism**: Structure-guided protein engineering
- Understanding folding enables better backbone design
- Contact prediction informs stability optimization
- Evolution-structure relationships guide mutation selection

**Evidence**:
- RFdiffusion explicitly uses structural knowledge
- ProteinMPNN trained on structural data
- Design success rate correlates with structural accuracy

**Uncertainty**: Range 0.20-0.40 reflects method-dependent variation

### Protein Design → Drug Discovery (0.12)

**Mechanism**: Designed proteins as therapeutics
- Enzyme therapeutics for metabolic diseases
- Designed binders as imaging agents
- Protein scaffolds for drug delivery

**Evidence**:
- ~20% of biotech pipeline involves designed proteins
- De novo protein therapeutics in clinical trials (2024)
- Antibody engineering efficiency improvements

**Uncertainty**: Range 0.08-0.18 reflects clinical success uncertainty

### Clinical Genomics → Drug Discovery (0.08)

**Mechanism**: Pharmacogenomics and target selection
- Variant classification informs dosing decisions
- Genetic associations guide target selection
- Patient stratification for clinical trials

**Evidence**:
- FDA pharmacogenomic labels on 200+ drugs
- GWAS-informed target selection success rate higher (Nelson et al., 2015)

**Uncertainty**: Range 0.04-0.12 reflects adoption variability

### Drug Discovery → Clinical Genomics (0.04)

**Mechanism**: Drug-gene interaction knowledge
- Drug development reveals gene function
- Pharmacogenomic studies generate variant data
- Clinical trials provide functional validation

**Evidence**:
- Weak effect; primarily one-directional
- PharmGKB database growth from drug studies

### Materials Science → Structural Biology (0.03)

**Mechanism**: Instrumentation and sample preparation
- Advanced grids for cryo-EM
- Better sample support materials
- Improved detectors

**Evidence**:
- Small but measurable effect
- Cryo-EM sample preparation innovations

### Protein Design → Materials Science (0.04)

**Mechanism**: Bio-inspired and protein-based materials
- Designed protein fibers and scaffolds
- Enzyme catalysts for green chemistry
- Self-assembling protein materials

**Evidence**:
- Growing field but still limited impact
- Spider silk and collagen design programs

### Clinical Genomics → Protein Design (0.04)

**Mechanism**: Natural variation informs design
- Pathogenic variants identify critical residues
- Polymorphism data guides stability engineering
- Evolutionary constraint patterns

**Evidence**:
- Variant effect predictors used in design
- CADD/REVEL integration with design tools

---

## S2.4 Missing Spillovers

The following potential spillovers are **not modeled** due to insufficient evidence or magnitude:

| Source | Target | Rationale for Exclusion |
|--------|--------|------------------------|
| Drug Discovery | Structural Biology | Minimal reverse flow |
| Drug Discovery | Protein Design | Limited direct effect |
| Drug Discovery | Materials Science | Different research communities |
| Materials Science | Drug Discovery | Indirect (via delivery) |
| Materials Science | Protein Design | Limited overlap |
| Materials Science | Clinical Genomics | No significant pathway |
| Clinical Genomics | Materials Science | No significant pathway |
| Structural Biology | Materials Science | Limited (some cryo-EM) |
| Structural Biology | Clinical Genomics | Indirect only |
| Protein Design | Clinical Genomics | Emerging but small |

---

## S2.5 Spillover Network Visualization

```
                    0.30
    Structural ───────────────► Protein
    Biology                      Design
        │                          │
        │ 0.25                     │ 0.12
        │                          │
        ▼         0.08             ▼
    Drug ◄─────────────────── Clinical
    Discovery ─────────────► Genomics
                  0.04

    Materials Science: Weakly connected (≤0.04 to all)
```

---

## S2.6 Total Spillover Effects by Domain (2030)

| Target Domain | Total Spillover Boost | Main Source |
|--------------|----------------------|-------------|
| Drug Discovery | +22% | Structural Biology (via SBDD) |
| Protein Design | +18% | Structural Biology |
| Clinical Genomics | +5% | Drug Discovery |
| Structural Biology | +2% | Materials Science |
| Materials Science | +4% | Protein Design |

---

## References

1. Griliches, Z. (1992). The search for R&D spillovers. *Scandinavian Journal of Economics*, 94, S29-S47.
2. Jaffe, A. B. (1986). Technological opportunity and spillovers of R&D. *American Economic Review*, 76(5), 984-1001.
3. Jaffe, A. B., Trajtenberg, M., & Henderson, R. (1993). Geographic localization of knowledge spillovers as evidenced by patent citations. *Quarterly Journal of Economics*, 108(3), 577-598.
4. Sledz, P., & Caflisch, A. (2018). Protein structure-based drug design: from docking to molecular dynamics. *Current Opinion in Structural Biology*, 48, 93-102.
5. Zucker, L. G., Darby, M. R., & Armstrong, J. S. (2002). Commercializing knowledge: University science, knowledge capture, and firm performance in biotechnology. *Management Science*, 48(1), 138-153.
6. Nelson, M. R., et al. (2015). The support of human genetic evidence for approved drug indications. *Nature Genetics*, 47(8), 856-860.

---

*Table S2 completed: January 2026*
*AI Research Acceleration Model v1.1*
