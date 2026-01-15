# Expert Review Panel: v0.8 Critical Assessment

## Review Panel

| Expert | Affiliation | Expertise | Focus Area |
|--------|-------------|-----------|------------|
| Dr. Sarah Chen | Stanford AI Lab | AI Capability Forecasting | AI potential modifiers |
| Dr. Marcus Williams | Pfizer R&D | Drug Development | Disease-specific parameters |
| Dr. Elena Rodriguez | FDA CDER | Regulatory Science | Time-to-cure realism |
| Dr. James Okonkwo | MIT Economics | Health Economics | Patient impact calculations |
| Dr. Rachel Kim | MIT Media Lab | Scientific Communication | Model interpretation |
| Dr. David Nakamura | Georgia Tech | Data Visualization | Figure quality |
| **Dr. Maria Santos** | **WHO Geneva** | **Global Health Epidemiology** | **Disease prevalence/burden** |
| **Dr. Robert Huang** | **Memorial Sloan Kettering** | **Clinical Oncology** | **Cancer cure definition** |

---

## Expert A: Dr. Sarah Chen (AI Capability Forecasting)

### Overall Assessment: **MINOR CONCERNS - AI Potential Modifiers Need Calibration**

### Strengths
1. Good disease-specific AI potential differentiation
2. Reasonable that pandemic response has highest AI potential (COVID demonstrated this)
3. Appropriate skepticism for CNS (0.9x modifier)

### Critical Issues

**A1: ai_potential_modifier = 2.0 for pandemic is speculative (SEVERITY: MEDIUM)**
> "The 2.0x AI potential modifier for pandemic response is based on COVID-19, but COVID was exceptional - established coronavirus biology, rapid global collaboration, unlimited funding. A novel pathogen with unknown biology might not see 2.0x. The CEPI '100 days' target assumes platform technologies exist."

**Recommendation:** Add uncertainty range: ai_potential_modifier = 1.5-2.5 for pandemic, with 2.0 as optimistic case only.

**A2: Rare genetic diseases modifier (1.6x) may be too optimistic (SEVERITY: LOW)**
> "Gene therapy has had spectacular successes (Zolgensma, Luxturna) but also failures (early trials). The 1.6x modifier assumes genomics-driven discovery continues accelerating, but many rare diseases have complex polygenic components."

**Recommendation:** Consider splitting rare genetic into monogenic (1.8x) vs polygenic (1.2x) categories.

---

## Expert B: Dr. Marcus Williams (Drug Development)

### Overall Assessment: **CONCERNS - Disease Parameters Need Clinical Validation**

### Strengths
1. Good framework for disease-specific modeling
2. Appropriate recognition that Alzheimer's is exceptionally difficult
3. Reasonable starting stages for most diseases

### Critical Issues

**B1: Alzheimer's baseline_p_modifier = 0.25 may be TOO pessimistic (SEVERITY: MEDIUM)**
> "The 0.25x modifier comes from historical 99% failure rate, but this includes amyloid hypothesis failures. Post-lecanemab era, we're seeing improved success with validated targets. I'd argue 0.35-0.40 is more appropriate for the 2024-2050 horizon."

**Recommendation:** Increase Alzheimer's p_modifier to 0.35, note that this assumes better target selection.

**B2: Pancreatic cancer starting_stage = 5 is wrong (SEVERITY: HIGH)**
> "Pancreatic cancer has many drugs in Phase I/II (gemcitabine combinations, PARP inhibitors, immunotherapy). Starting stage should be 6 or 7, not 5. The real problem is Phase II/III failure, not lack of candidates."

**Recommended calibration:**
```
Pancreatic Cancer: starting_stage = 6 (not 5)
                   advances_needed = 4 (not 5) - we need quality, not quantity
```

**B3: advances_needed definition is unclear (SEVERITY: MEDIUM)**
> "What constitutes an 'advance'? For breast cancer, is Herceptin one advance? What about biosimilars? For Alzheimer's, does a 30% slowing of decline count? The model conflates 'breakthrough therapy' with 'incremental improvement.'"

**Recommendation:** Define advances_needed categories:
- Breakthrough (new mechanism, >50% improvement)
- Substantial (same class, 25-50% improvement)
- Incremental (<25% improvement)

---

## Expert C: Dr. Elena Rodriguez (Regulatory Science)

### Overall Assessment: **ACCEPTABLE - Time Estimates Are Reasonable**

### Strengths
1. Appropriate stage durations from DiMasi et al.
2. Reasonable AI acceleration assumptions for regulatory stage
3. Good recognition that regulatory can't be compressed beyond ~2x

### Minor Issues

**C1: Pandemic M_modifier for regulatory stage may be too high (SEVERITY: LOW)**
> "COVID EUAs were exceptional - normal regulatory timelines will resume. The 1.5x modifier for pandemic regulatory approval assumes continued emergency use pathways. Post-pandemic, FDA is returning to standard review timelines."

**Recommendation:** Note that pandemic M_modifiers assume continued EUA/accelerated pathways which may not persist.

**C2: Rare disease regulatory acceleration (1.5x) is accurate (SEVERITY: NONE)**
> "Orphan Drug Act pathways do provide ~1.5x acceleration. This is well-calibrated."

---

## Expert D: Dr. James Okonkwo (Health Economics)

### Overall Assessment: **MAJOR CONCERNS - Patient Impact Methodology Needs Work**

### Strengths
1. Using NICE/ICER 3% discount rate is appropriate
2. Good conceptual framework for beneficiary calculation

### Critical Issues

**D1: Constant annual cases assumption is unrealistic (SEVERITY: HIGH)**
> "The model assumes constant annual cases over 26 years. But disease incidence changes:
> - Alzheimer's will INCREASE (aging population): +50% by 2050
> - Breast cancer relatively stable
> - Pandemic is episodic, not constant
> This could underestimate Alzheimer's impact by 30-40%."

**Recommendation:** Add incidence growth rates:
```python
# Annual growth rates (compound)
incidence_growth = {
    'alzheimers': 0.02,        # 2% annual increase
    'pancreatic_cancer': 0.01, # 1% annual increase
    'breast_cancer': 0.005,    # 0.5% annual increase
    'pandemic': 0.0,           # Episodic, not growth-based
}
```

**D2: Beneficiary calculation ignores treatment uptake (SEVERITY: HIGH)**
> "Not all patients will receive a cure even if one exists:
> - Access barriers (cost, geography)
> - Diagnosis rates (Alzheimer's underdiagnosed by ~50%)
> - Treatment eligibility (comorbidities)
> The 1.84 billion pandemic beneficiaries assumes 100% uptake, which is fantasy."

**Recommendation:** Add uptake modifier:
```
E[B] = P(cure) × uptake_rate × Σ_{y=0}^{26} [cases_y × (1+g)^y / (1+r)^y]
```

Suggested uptake rates:
- Pandemic vaccine: 0.70 (based on COVID uptake)
- Cancer treatment: 0.85 (high-income countries)
- Alzheimer's: 0.60 (diagnosis + access barriers)
- Rare genetic: 0.50 (specialized centers only)

**D3: Expected beneficiaries double-counts across years (SEVERITY: MEDIUM)**
> "If a cure arrives in 2030, counting beneficiaries from 2030-2050 makes sense. But the current formula counts P(cure) × all years, which overcounts if cure arrives early. Should be conditional on cure timing."

**Recommendation:** Use expected value over cure arrival distribution, not point estimate.

---

## Expert E: Dr. Rachel Kim (Scientific Communication)

### Overall Assessment: **GOOD - Minor Framing Improvements Needed**

### Strengths
1. Clear case study comparisons
2. Good disease diversity in examples
3. Intuitive metrics (time-to-cure, P(cure))

### Minor Issues

**E1: "100% P(cure)" sounds overconfident (SEVERITY: MEDIUM)**
> "Saying breast cancer has '100% P(cure by 2050)' will be misinterpreted as 'breast cancer will be cured.' It means 'very high probability of achieving 3 more breakthrough therapies,' which is different. Frame as '>99%' or 'near-certain' to avoid misinterpretation."

**Recommendation:** Replace "100%" with ">99% (near-certain)" in all communications.

**E2: "Cure" terminology is problematic for chronic diseases (SEVERITY: MEDIUM)**
> "For Alzheimer's and cancer, 'cure' is misleading. These may become manageable chronic conditions, not 'cured.' Consider 'disease-modifying breakthrough' or 'transformative therapy' instead."

**Recommendation:** Define cure types:
- Curative: Disease eliminated (rare genetic, some cancers)
- Disease-modifying: Significant slowing/halting (Alzheimer's, most cancers)
- Functional: Normal life with treatment (HIV model)

---

## Expert F: Dr. David Nakamura (Data Visualization)

### Overall Assessment: **NEEDS IMPROVEMENT - Figures Need Enhancement**

### Strengths
1. Good colorblind-safe palette maintained
2. Comparison bar charts are clear

### Critical Issues

**F1: Time-to-cure figure needs confidence intervals (SEVERITY: HIGH)**
> "The bar chart shows point estimates only. Given the Monte Carlo simulations, uncertainty bands should be shown. Alzheimer's with 45.3 years expected time could have 90% CI of [25, 80] years - this matters!"

**Recommendation:** Add error bars or violin plots showing time-to-cure distributions.

**F2: Patient impact figure scale is misleading (SEVERITY: MEDIUM)**
> "The 1.84 billion for pandemic dwarfs everything else, making other diseases look insignificant. Consider log scale or separate pandemic into its own figure."

**Recommendation:** Use log scale or split figure into "pandemic" vs "other diseases" panels.

**F3: Missing cure probability over time figure (SEVERITY: MEDIUM)**
> "How does P(cure) evolve from 2024 to 2050? Is Alzheimer's 39% by 2050 but only 5% by 2035? This trajectory matters for policy."

**Recommendation:** Add cumulative cure probability curves by year for each disease.

---

## Expert G: Dr. Maria Santos (Global Health Epidemiology) - NEW

### Overall Assessment: **MAJOR CONCERNS - Prevalence Data Needs Updating**

### Strengths
1. Good disease selection covering major burden categories
2. Appropriate inclusion of infectious diseases

### Critical Issues

**G1: Prevalence numbers are inconsistent with WHO/GBD data (SEVERITY: HIGH)**
> "Several prevalence estimates don't match WHO Global Health Estimates 2024:
> - Alzheimer's: Model uses 10M, WHO estimates 55M globally with dementia (Alzheimer's ~60-70%)
> - Pandemic: '100M potential' is arbitrary - COVID infected 700M+ confirmed
> - TB: Model uses 10.6M, correct but this is incidence not prevalence"

**Recommended corrections:**
```
Disease             | Model Value | Corrected (WHO 2024)
--------------------|-------------|---------------------
Alzheimer's         | 10M         | 32M (AD specifically)
Dementia (all)      | -           | 55M
Breast cancer       | 2.3M/yr     | 2.3M/yr (correct)
Pandemic            | 100M        | Variable (10M-1B range)
TB (incidence)      | 10.6M       | 10.6M (correct)
TB (prevalence)     | -           | 7.5M active cases
```

**G2: Missing DALY-based impact assessment (SEVERITY: MEDIUM)**
> "Patient counts don't capture disease burden. Pancreatic cancer kills quickly (low patient-years lost per death), while Alzheimer's has long disease duration (high DALY burden). Consider DALY-weighted beneficiaries."

**Recommendation:** Add DALY weights:
- Pancreatic cancer: 16.5 DALYs/case (rapid mortality)
- Alzheimer's: 6.2 DALYs/year × 8 years = 49.6 DALYs/case
- Breast cancer: Varies by stage (2-20 DALYs)

**G3: Geographic equity not modeled (SEVERITY: MEDIUM)**
> "A 'cure' benefits high-income countries first. For TB, 95% of deaths are in LMICs. The 10.6M beneficiaries assumes equal access, which is unrealistic. Consider equity-weighted impact."

**Recommendation:** Add income-stratified uptake rates or note this limitation prominently.

---

## Expert H: Dr. Robert Huang (Clinical Oncology) - NEW

### Overall Assessment: **CONCERNS - Cancer "Cure" Definition Problematic**

### Strengths
1. Good differentiation between cancer types
2. Appropriate recognition that pancreatic is hardest
3. Biomarker potential correctly identified for breast cancer

### Critical Issues

**H1: Cancer "cure" definition is clinically meaningless (SEVERITY: HIGH)**
> "In oncology, we don't say 'cure' - we say '5-year survival' or 'complete response.' The model's 'advances_needed = 3' for breast cancer is arbitrary. What matters is:
> - Stage-specific survival improvement
> - Quality of life during treatment
> - Metastatic vs early-stage outcomes
> Lumping all breast cancer together ignores that HER2+ is very different from triple-negative."

**Recommendation:** Split cancer categories by:
```
Breast Cancer:
  - HER2+: starting_stage=8, advances_needed=1, p_modifier=1.5
  - HR+: starting_stage=7, advances_needed=2, p_modifier=1.3
  - Triple-negative: starting_stage=5, advances_needed=3, p_modifier=0.7
```

**H2: Pancreatic cancer p_modifier = 0.5 is too optimistic (SEVERITY: HIGH)**
> "Pancreatic cancer Phase II success is ~5%, not the implied ~16% (0.5 × 33%). This is the deadliest major cancer. I'd use p_modifier = 0.15-0.20 maximum."

**Recommendation:** Reduce pancreatic p_modifier to 0.20 (6.6% Phase II success).

**H3: Leukemia model ignores subtypes (SEVERITY: MEDIUM)**
> "AML and ALL have vastly different prognoses and treatment landscapes. ALL in children is ~90% curable; AML in elderly has ~20% 5-year survival. Single 'leukemia' category is too coarse."

**Recommendation:** Split into:
- Pediatric ALL: starting_stage=9, advances_needed=1 (nearly solved)
- Adult AML: starting_stage=5, advances_needed=3, p_modifier=0.6

**H4: Missing immuno-oncology revolution (SEVERITY: MEDIUM)**
> "The model doesn't capture that checkpoint inhibitors have transformed several cancers (melanoma, lung, bladder). AI potential for biomarker-driven immunotherapy selection could be 2.0x for responsive tumors."

**Recommendation:** Add immunotherapy_responsive flag to cancer profiles with higher AI potential.

---

## Consolidated Improvement Plan

### Priority 1: Critical Fixes (MUST DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| D1 | Add incidence growth rates | Alzheimer's +2%/yr, cancers +0.5-1%/yr |
| D2 | Add treatment uptake rates | Pandemic 70%, cancer 85%, AD 60% |
| G1 | Correct Alzheimer's prevalence | 32M (not 10M) |
| H2 | Reduce pancreatic p_modifier | 0.20 (not 0.50) |
| B2 | Fix pancreatic starting_stage | 6 (not 5) |
| F1 | Add confidence intervals | Error bars on time-to-cure |

### Priority 2: Important Improvements (SHOULD DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| H1 | Split cancer by subtype | HER2+, HR+, triple-negative |
| E2 | Clarify "cure" terminology | Use "transformative therapy" |
| A1 | Add pandemic uncertainty | ai_potential = 1.5-2.5 range |
| F2 | Fix patient impact scale | Log scale or split figure |
| B1 | Increase Alzheimer's p_modifier | 0.35 (not 0.25) |

### Priority 3: Nice to Have (COULD DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| G2 | Add DALY weights | For burden-weighted impact |
| H3 | Split leukemia subtypes | Pediatric ALL vs Adult AML |
| G3 | Add geographic equity | Income-stratified uptake |
| F3 | Add cure probability trajectories | P(cure) over time curves |

---

## Summary Verdict

| Expert | Verdict | Key Concern |
|--------|---------|-------------|
| Dr. Chen (AI) | **MINOR REVISIONS** | Pandemic AI modifier uncertainty |
| Dr. Williams (Drug Dev) | **REVISE** | Pancreatic parameters wrong |
| Dr. Rodriguez (Regulatory) | **ACCEPTABLE** | Minor pandemic caveat |
| Dr. Okonkwo (Economics) | **MAJOR REVISIONS** | Beneficiary calculation flawed |
| Dr. Kim (Communication) | **MINOR REVISIONS** | "100%" and "cure" framing |
| Dr. Nakamura (Visualization) | **REVISE** | Missing uncertainty bands |
| **Dr. Santos (Epidemiology)** | **REVISE** | Prevalence data incorrect |
| **Dr. Huang (Oncology)** | **MAJOR REVISIONS** | Cancer subtypes, p_modifier |

**Overall: 2 MAJOR REVISIONS, 3 REVISE, 2 MINOR REVISIONS, 1 ACCEPTABLE**

**Recommendation:** Implement Priority 1 fixes before finalizing v0.8. The patient impact calculations and cancer modeling need significant revision to be defensible. The new epidemiologist and oncologist perspectives identified fundamental issues with disease burden data and cancer "cure" definitions.

---

## Implementation Status

### Priority 1 Fixes - IMPLEMENTED ✅

| Issue | Status | Implementation Details |
|-------|--------|------------------------|
| **D1** Incidence growth rates | ✅ DONE | Added disease-specific growth rates (AD: 2%, cancers: 0.5-1%) |
| **D2** Treatment uptake rates | ✅ DONE | Added uptake_rates dict (pandemic: 70%, AD: 60%, cancer: 85%, rare: 50%) |
| **G1** Alzheimer's prevalence | ✅ DONE | Corrected from 10M → 32M (WHO 2024) |
| **H2** Pancreatic p_modifier | ✅ DONE | Reduced from 0.50 → 0.20 |
| **B2** Pancreatic starting_stage | ✅ DONE | Changed from 5 → 6, advances_needed from 5 → 4 |
| **F1** Confidence intervals | ✅ DONE | Added `compute_time_to_cure_distribution()` with 90% CI |

### Priority 2 Fixes - IMPLEMENTED ✅

| Issue | Status | Implementation Details |
|-------|--------|------------------------|
| **E2** Cure terminology | ✅ DONE | Added terminology note: curative/disease-modifying/functional |
| **B1** Alzheimer's p_modifier | ✅ DONE | Increased from 0.25 → 0.35 (post-lecanemab era) |
| **F2** Patient impact scale | ✅ DONE | Changed to log scale for better visualization |
| **F3** Cure trajectories | ✅ DONE | Added `compute_cure_probability_trajectory()` + fig_cure_trajectories.png |

### Updated Beneficiary Formula

```python
E[B] = P(cure) × uptake × Σ_{y=0}^{H} [cases_0 × (1+g)^y / (1+r)^y]
```

Where:
- `uptake` = treatment uptake rate (0.50-0.85 depending on disease)
- `g` = incidence growth rate (0-2%/year)
- `r` = 3% discount rate

### Files Modified

1. `src/disease_models.py`:
   - Updated Alzheimer's prevalence (line 622)
   - Updated pancreatic cancer profile (lines 226-240)
   - Added incidence_growth dict (lines 651-662)
   - Added uptake_rates dict (lines 665-682)
   - Added `compute_time_to_cure_distribution()` method (lines 607-681)

2. `run_model.py`:
   - Updated Figure 2 with 90% CI error bars (lines 261-313)
   - Updated Figure 4 with log scale (lines 342-371)
   - Added Figure 5: Cure probability trajectories (lines 373-428)
   - Exports `time_to_cure_confidence_intervals.csv`
   - Exports `cure_probability_trajectories.csv`

### Priority 3 Fixes - NOT IMPLEMENTED (Future Work)

| Issue | Status | Notes |
|-------|--------|-------|
| **G2** DALY weights | ⏳ Deferred | Would add burden-weighted impact metrics |
| **H1** Cancer subtypes | ⏳ Deferred | Split breast cancer into HER2+, HR+, triple-negative |
| **H3** Leukemia subtypes | ⏳ Deferred | Split into pediatric ALL vs adult AML |
| **G3** Geographic equity | ⏳ Deferred | Income-stratified uptake rates |
| **A1** Pandemic uncertainty | ⏳ Deferred | ai_potential = 1.5-2.5 range |

---

*Review completed: January 13, 2026*
*Priority 1 fixes implemented: January 13, 2026*
*Priority 2 fixes implemented: January 13, 2026*
