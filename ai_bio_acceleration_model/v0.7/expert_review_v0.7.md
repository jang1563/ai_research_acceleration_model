# Expert Review Panel: v0.7 Critical Assessment

## Review Panel

| Expert | Affiliation | Expertise | Focus Area |
|--------|-------------|-----------|------------|
| Dr. Sarah Chen | Stanford AI Lab | AI Capability Forecasting | Amodei scenario assumptions |
| Dr. Marcus Williams | Pfizer R&D | Drug Development | Pipeline iteration realism |
| Dr. Elena Rodriguez | FDA CDER | Regulatory Science | Regulatory reform assumptions |
| Dr. James Okonkwo | MIT Economics | Health Economics | Parallelization claims |
| Dr. Rachel Kim | MIT Media Lab | Scientific Communication | Model interpretation |
| Dr. David Nakamura | Georgia Tech | Data Visualization | Figure quality |

---

## Expert A: Dr. Sarah Chen (AI Capability Forecasting)

### Overall Assessment: **CONCERNS - Amodei Scenario Too Aggressive**

### Strengths
1. Good separation of AI types (cognitive/robotic/scientific)
2. Proper citation of Amodei's essay
3. Reasonable baseline scenario (5.8x)

### Critical Issues

**A1: g_cognitive = 0.90 is unrealistic (SEVERITY: HIGH)**
> "The Amodei scenario assumes g_cognitive = 0.90, implying ~500 million x capability growth by 2050. This far exceeds any historical AI trend. Even the most aggressive Epoch AI projections don't support this."

**Recommendation:** Cap g_cognitive at 0.75 maximum, add sensitivity analysis showing dependence on this parameter.

**A2: Parallelization factor = 2.0 is unsubstantiated (SEVERITY: MEDIUM)**
> "Where does the 2.0x parallelization factor come from? This assumes perfect doubling of R&D capacity, ignoring coordination costs, talent constraints, and diminishing returns to parallel effort."

**Recommendation:** Model parallelization with diminishing returns: `P_eff = 1 + (P-1) * 0.7` (30% efficiency loss).

**A3: No AI capability saturation in Amodei scenario (SEVERITY: MEDIUM)**
> "The Amodei scenario assumes continued exponential growth through 2050, but even Amodei's essay acknowledges physical limits and 'irreducible latency.' Where is the saturation?"

**Recommendation:** Add A_max parameter to Amodei scenario to capture eventual saturation.

---

## Expert B: Dr. Marcus Williams (Drug Development)

### Overall Assessment: **MIXED - Pipeline Iteration Good, Rework Values Need Calibration**

### Strengths
1. Excellent conceptual framework for rework dynamics
2. Stage-specific return destinations are realistic
3. Phase II as "graveyard" correctly identified

### Critical Issues

**B1: Rework fractions not calibrated to literature (SEVERITY: HIGH)**
> "The rework_fraction values (0.2-0.9) appear to be estimated, not empirically derived. Paul et al. (2010) provides actual reformulation rates: Phase II failures have ~15% reformulation rate, not 20%. Phase III is closer to 30%, not 40%."

**Recommended calibration:**
```
Stage 6 (Phase I): rework_fraction = 0.25 (not 0.30)
Stage 7 (Phase II): rework_fraction = 0.15 (not 0.20) - CRITICAL
Stage 8 (Phase III): rework_fraction = 0.30 (not 0.40)
Stage 9 (Regulatory): rework_fraction = 0.80 (not 0.70) - FDA gives CRLs
```

**B2: max_attempts too high for clinical stages (SEVERITY: MEDIUM)**
> "max_attempts = 2 for Phase II/III suggests drugs can fail twice and retry. In reality, most drugs get ONE shot. After Phase III failure, reformulation is rare (<10%)."

**Recommendation:** Set max_attempts = 1 for Phase II/III except for rare diseases.

**B3: Missing compound-level vs. indication-level distinction (SEVERITY: LOW)**
> "The model treats all failures equally, but compound failures (toxicity) vs. indication failures (wrong patient population) have very different rework paths."

**Recommendation:** Consider for v0.8 - not critical for current version.

---

## Expert C: Dr. Elena Rodriguez (Regulatory Science)

### Overall Assessment: **SKEPTICAL - Regulatory Reform Assumptions Too Optimistic**

### Strengths
1. Acknowledges regulatory constraints exist
2. Separates regulatory from deployment

### Critical Issues

**C1: Phase II M_max = 5.0 assumes impossible regulatory reform (SEVERITY: HIGH)**
> "The Amodei scenario assumes Phase II trials can be accelerated 5x. This would require abandoning human safety standards. The FDA's Accelerated Approval pathway already represents near-maximum acceleration (~2x). Beyond that requires fundamental changes to drug safety law."

**Reality check:**
- Current FDA Accelerated Approval: ~1.5-2x acceleration
- Adaptive trials: ~1.3x acceleration
- Maximum realistic with reform: ~2.5-3x

**Recommendation:** Cap Phase II M_max at 3.5 for Amodei scenario, or clearly label as "regulatory revolution" requiring Congressional action.

**C2: p_success_max = 0.65 for Phase II assumes better drugs exist (SEVERITY: MEDIUM)**
> "A 65% Phase II success rate (vs 33% baseline) assumes drugs will be twice as effective. This is a biological assumption, not a regulatory one. The model conflates AI improving drug design with AI improving trial efficiency."

**Recommendation:** Separate two effects:
1. AI improving trial efficiency (regulatory) - M_max
2. AI improving drug quality (scientific) - p_success_max

**C3: Regulatory stage M_max = 4.0 is unrealistic (SEVERITY: HIGH)**
> "FDA review cannot be accelerated 4x. Even with AI assistance, minimum review times are mandated by law. PDUFA requires 6-10 months minimum for standard and priority reviews."

**Recommendation:** Cap regulatory M_max at 2.0 even in Amodei scenario.

---

## Expert D: Dr. James Okonkwo (Health Economics)

### Overall Assessment: **CONCERNED - Economic Constraints Missing**

### Strengths
1. Good throughput modeling
2. Cumulative progress metric is useful

### Critical Issues

**D1: No cost constraints in parallelization (SEVERITY: HIGH)**
> "The model assumes unlimited R&D funding. The 2.0x parallelization factor would require ~$200B annual global pharma R&D (vs ~$100B today). Who pays? Without cost modeling, parallelization is fantasy."

**Recommendation:** Add cost constraint:
```
P_affordable = min(P_target, Budget / Unit_Cost)
```

**D2: Talent/capacity constraints ignored (SEVERITY: HIGH)**
> "Even with unlimited funding, there are ~500,000 clinical researchers globally. 2x parallelization would require 1 million. Where do they come from? Training takes 10+ years."

**Recommendation:** Add human capital constraint that limits parallelization growth rate to ~5% annually.

**D3: No market dynamics for drug pricing (SEVERITY: MEDIUM)**
> "More therapies doesn't mean more patient benefit if pricing makes them inaccessible. The 'equivalent years' metric ignores economic accessibility."

**Recommendation:** Add accessibility discount factor based on therapeutic area pricing.

---

## Expert E: Dr. Rachel Kim (Scientific Communication)

### Overall Assessment: **GOOD PROGRESS - Minor Clarity Issues**

### Strengths
1. Clear scenario comparison table
2. Amodei target comparison is valuable
3. Progress metrics are intuitive

### Critical Issues

**E1: "Meets Amodei Target" is misleading (SEVERITY: MEDIUM)**
> "The table shows 'meets_amodei_low = True' for Optimistic scenario at 50.2 years, but Amodei's prediction is 50-100 years in 5-10 years. The Optimistic scenario BARELY meets the low end at EXACTLY 10 years. This should be noted."

**Recommendation:** Add confidence language: "marginally meets" vs "clearly meets" vs "exceeds"

**E2: 13.1x acceleration sounds implausible to policymakers (SEVERITY: MEDIUM)**
> "Claiming 13x acceleration will be dismissed as hype. Frame it as 'theoretical upper bound under ideal conditions' not 'prediction.'"

**Recommendation:** Rename "Amodei" scenario to "Upper Bound (Amodei)" or "Theoretical Maximum"

**E3: Missing plain-English summary of pipeline iteration (SEVERITY: LOW)**
> "What does 'rework overhead 1.21x' mean to a non-expert? Add interpretation like 'projects take 21% longer due to failures and retries.'"

---

## Expert F: Dr. David Nakamura (Data Visualization)

### Overall Assessment: **NEEDS IMPROVEMENT - Figures Need Polish**

### Strengths
1. Good color palette (colorblind-safe)
2. Amodei target zones are helpful

### Critical Issues

**F1: Figure legends need more context (SEVERITY: MEDIUM)**
> "The acceleration timeline figure shows lines crossing but doesn't explain why. Add annotations: 'Amodei scenario accelerates faster due to regulatory reform assumptions.'"

**F2: Missing uncertainty bands (SEVERITY: HIGH)**
> "All figures show point estimates. Where are the confidence intervals? Without uncertainty, the figures overstate precision. The Amodei scenario should have WIDE uncertainty bands."

**Recommendation:** Add Monte Carlo uncertainty bands to all scenario projections.

**F3: 10-year comparison bar chart needs context (SEVERITY: MEDIUM)**
> "The bar chart shows progress values but doesn't show what 50 or 100 means. Add reference lines with labels like '50yr = Amodei minimum target.'"

---

## Consolidated Improvement Plan

### Priority 1: Critical Fixes (MUST DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| C1, C3 | Cap M_max for regulatory stages | Phase II: 3.5, Regulatory: 2.0 |
| B1 | Calibrate rework fractions to literature | Phase II: 0.15, Phase III: 0.30 |
| A1 | Reduce g_cognitive in Amodei | Cap at 0.75 |
| F2 | Add uncertainty bands | Monte Carlo on all scenarios |

### Priority 2: Important Improvements (SHOULD DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| A2 | Model parallelization with diminishing returns | P_eff = 1 + (P-1)*0.7 |
| D1 | Add cost constraint note | Document assumption in BIBLE |
| E2 | Rename Amodei scenario | "Upper Bound (Amodei conditions)" |
| B2 | Reduce max_attempts for clinical stages | Phase II/III: max_attempts = 1 |

### Priority 3: Nice to Have (COULD DO)

| Issue | Fix | Implementation |
|-------|-----|----------------|
| D2 | Add talent constraint | For v0.8 |
| C2 | Separate trial efficiency from drug quality | Conceptual clarification |
| E3 | Add plain-English glossary entry | Update GLOSSARY.md |

---

## Summary Verdict

| Expert | Verdict | Key Concern |
|--------|---------|-------------|
| Dr. Chen (AI) | **REVISE** | g_cognitive too high |
| Dr. Williams (Drug Dev) | **MINOR REVISIONS** | Rework fractions need calibration |
| Dr. Rodriguez (Regulatory) | **REVISE** | M_max values unrealistic |
| Dr. Okonkwo (Economics) | **NOTE LIMITATIONS** | Cost/talent constraints missing |
| Dr. Kim (Communication) | **ACCEPTABLE** | Minor framing improvements |
| Dr. Nakamura (Visualization) | **MINOR REVISIONS** | Add uncertainty bands |

**Overall: 2 REVISE, 2 MINOR REVISIONS, 1 NOTE LIMITATIONS, 1 ACCEPTABLE**

**Recommendation:** Implement Priority 1 fixes before finalizing v0.7. The Amodei scenario needs recalibration to be defensible in peer review.

---

*Review completed: January 13, 2026*
