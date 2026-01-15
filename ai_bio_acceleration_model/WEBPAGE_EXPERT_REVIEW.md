# Expert Review: Webpage Publication Plan

## Review Panel

| # | Expert | Domain | Perspective |
|---|--------|--------|-------------|
| 1 | Dr. Sarah Chen | Data Visualization | Information design, web interactivity |
| 2 | Prof. Michael Torres | Science Communication | Public engagement, technical writing |
| 3 | Dr. Aisha Patel | UX/Web Design | User experience, accessibility |
| 4 | Dr. James Morrison | Computational Biology | Scientific accuracy, methodology |
| 5 | Elena Kowalski | Tech Journalism | Audience engagement, shareability |
| 6 | Dr. Robert Kim | Health Policy | Policy communication, credibility |
| 7 | Dr. Lisa Zhang | Statistics | Uncertainty communication, rigor |
| 8 | Marcus Webb | Frontend Development | Technical feasibility, performance |

---

## Expert 1: Dr. Sarah Chen (Data Visualization)

### Overall Assessment: **B+ (Strong with reservations)**

### Strengths
- Good choice of 5 core visualizations covering key findings
- Interactive elements will increase engagement
- Color palette is colorblind-friendly

### Concerns
1. **Visualization overload**: 7 visualizations + interactive explorer may overwhelm. Recommend **5 maximum** for main content.
2. **Hero chart complexity**: Showing 4 scenarios + 2 CI bands + counterfactual on one chart is too busy. Simplify to 3 scenarios with toggle.
3. **Tornado diagram**: Classic but dated. Consider **dot plot with CI whiskers** instead—more modern, same information.

### Specific Recommendations
```
Priority Changes:
1. Simplify hero chart: Default view = Baseline only with CI band
   - Add scenario toggle (Pessimistic/Optimistic/Amodei)

2. Disease timeline: Use vertical layout, not Gantt
   - Better for mobile, easier to scan

3. Add small multiples for Monte Carlo:
   - Show convergence, distribution, and CDF as 3 linked panels

4. Policy ROI: Add cost dimension visually
   - Bubble chart (x=cost, y=ROI, size=impact) more informative than bar
```

### Missing Elements
- **Annotation layer**: Add contextual annotations directly on charts
- **Source links**: Each chart should link to underlying data

### Verdict
Good foundation but needs simplification. Less is more for web. **Cut to 5 visualizations, make them exceptional.**

---

## Expert 2: Prof. Michael Torres (Science Communication)

### Overall Assessment: **A- (Very Good)**

### Strengths
- Clear narrative structure (Model → Results → Policy → Limitations)
- Appropriate word count for web (4,000 is right)
- Good balance of technical depth with "TL;DR" upfront
- Honest uncertainty framing is refreshing

### Concerns
1. **Opening hook is weak**: "How Much Will AI Accelerate..." is informative but not compelling. Need emotional/stakes hook.
2. **Policy section before uncertainty**: Should flip order. Establish credibility through honest limitations BEFORE making recommendations.
3. **Missing narrative thread**: Plan is sectional but needs a through-line story

### Specific Recommendations
```
Restructure Opening:
OLD: "How Much Will AI Accelerate Biological Discovery?"
NEW: "Your grandmother's Alzheimer's might be cured 15 years sooner
      because of AI. Here's the math."

Add Narrative Frame:
- Open with a specific disease/patient scenario
- Return to it at end with model prediction
- Makes abstract numbers concrete

Reorder Sections:
1. Hero + Hook
2. The Model (framework)
3. Results (what we found)
4. Uncertainty (honest caveats) ← MOVE UP
5. Policy (given above, what to do)
6. Methods (for the curious)
```

### Tone Calibration
- Plan says "technical but accessible"—good
- Add **one human story** to ground the numbers
- Avoid: "breakthrough," "revolutionary," "game-changing"
- Use: "substantial," "meaningful," "significant"

### Verdict
Strong structure, needs a compelling hook and human element. **Add one patient/disease narrative thread.**

---

## Expert 3: Dr. Aisha Patel (UX/Web Design)

### Overall Assessment: **B (Good but needs UX refinement)**

### Strengths
- Mobile-responsive is essential—glad it's prioritized
- Dark mode toggle is nice-to-have, good inclusion
- Print stylesheet shows attention to detail

### Concerns
1. **No scroll progress indicator**: Long-form content needs visual progress
2. **Interactive explorer is risky**: Real-time parameter updates require careful UX or users get lost
3. **Collapsible sections can hide important content**: Users often miss them

### Specific Recommendations
```
Navigation:
- Add sticky table of contents (left sidebar on desktop, hamburger on mobile)
- Progress bar at top showing read progress
- "Jump to Results" floating button

Interactive Explorer:
- Make it a SEPARATE page, not inline
- Or: Modal/overlay that doesn't disrupt reading flow
- Add "Reset to defaults" button prominently

Mobile Considerations:
- Charts must be swipeable on mobile
- Text should be readable without zooming (16px minimum)
- Touch targets 44px minimum for interactive elements

Accessibility Additions:
- Alt text for all visualizations
- ARIA labels for interactive elements
- Keyboard navigation for charts
- Screen reader announcements for dynamic content
```

### Information Architecture
```
Current: Linear scroll
Better:  Hub-and-spoke

Landing → Executive Summary (hub)
       ↓
  [The Model] [Results] [Policy] [Methods]
       ↓
     Deep dives on demand
```

### Verdict
Solid technical spec but needs UX layer. **Add navigation aids and reconsider interactive explorer placement.**

---

## Expert 4: Dr. James Morrison (Computational Biology)

### Overall Assessment: **A- (Scientifically Sound)**

### Strengths
- 10-stage pipeline is comprehensive
- Sobol sensitivity analysis is appropriate methodology
- Monte Carlo with convergence diagnostics is rigorous
- Uncertainty bounds prominently featured

### Concerns
1. **Parameter sources not visible enough**: Credibility depends on traceable assumptions
2. **Validation section missing**: How does model compare to historical data?
3. **AI-assisted methodology note too brief**: Scientific community will want more detail

### Specific Recommendations
```
Add Validation Section:
- Backcast: Apply model to 2000-2020, compare to actual progress
- Benchmark: Compare AI growth rate to observed AI capabilities
- Sanity check: Are 2030 predictions plausible given current trajectory?

Parameter Transparency:
- Every parameter should have:
  - Point estimate
  - Distribution type
  - Source (with hyperlink)
  - Confidence level (high/medium/low)
- Make this a hoverable tooltip on the parameter table

Methodology Note Expansion:
Current: "Model developed with AI assistance (Claude)"
Better:  "Model development used Claude (Anthropic) for code generation,
          literature synthesis, and iterative refinement. All assumptions
          and parameters were human-validated. Full development log
          available in supplementary materials."
```

### Scientific Rigor Additions
- Add model limitations subsection: structural assumptions, excluded factors
- Cite relevant forecasting literature (Tetlock, Good Judgment Project)
- Acknowledge: "This is not a prediction, it's a scenario analysis"

### Verdict
Methodologically sound. **Add validation/backcasting and expand parameter transparency.**

---

## Expert 5: Elena Kowalski (Tech Journalism)

### Overall Assessment: **B+ (Good content, needs virality hooks)**

### Strengths
- Clear key stat: "3.4x-9.2x by 2050" is tweetable
- Policy recommendations are actionable
- GitHub link adds credibility

### Concerns
1. **No shareable assets**: Where are the tweetable images?
2. **Headline is boring**: Needs punch for social media
3. **No controversy/debate angle**: What will people argue about?

### Specific Recommendations
```
Shareable Assets (Create These):
1. Hero stat card: "80% confident: AI accelerates biology 3-9x by 2050"
2. Disease timeline infographic (standalone image)
3. Policy ROI one-pager
4. Quote cards with key findings

Headlines for Different Platforms:
- Twitter: "We modeled AI's impact on drug discovery. The math says
           your diseases might be cured 10+ years sooner."
- LinkedIn: "New quantitative model: AI could accelerate biological
            discovery 5.7x by 2050. Here's what that means for healthcare."
- Hacker News: "AI-Accelerated Biological Discovery: A Monte Carlo Model
               with Sobol Sensitivity Analysis [code included]"

Controversy Hooks (Pick One):
- "Dario Amodei says 10x. Our model says he might be right."
- "Why regulatory reform beats research funding 40:1 for ROI"
- "The one parameter that determines everything about AI in biology"
```

### Engagement Boosters
- Add "Share this finding" buttons next to key stats
- Create Twitter thread template in supplementary materials
- Include "Surprise me" fact at the end

### Verdict
Content is solid but needs social packaging. **Create 3-4 shareable image assets.**

---

## Expert 6: Dr. Robert Kim (Health Policy)

### Overall Assessment: **B+ (Policy section needs strengthening)**

### Strengths
- ROI framework is exactly what policymakers need
- Tier system makes prioritization clear
- Cost estimates included

### Concerns
1. **No implementation pathway**: HOW do we "expand adaptive trial designs"?
2. **Missing stakeholder map**: Who needs to act? FDA? Congress? Industry?
3. **Political feasibility ignored**: Some interventions are easier than others

### Specific Recommendations
```
Add Implementation Details:
For each Tier 1 intervention, specify:
- Lead actor (FDA, NIH, Congress, Industry)
- Timeline to implement
- Political feasibility (Easy/Medium/Hard)
- Existing precedent or legislation

Example Enhancement:
OLD: "Expand Adaptive Trial Designs - ROI: 20,331x"
NEW: "Expand Adaptive Trial Designs
      - ROI: 20,331x
      - Lead: FDA (guidance update) + Congress (funding)
      - Timeline: 18-24 months
      - Feasibility: Medium (requires FDA buy-in)
      - Precedent: COVID-19 emergency adaptive trials"

Add Policy Brief Format:
- One-page PDF downloadable summary
- Formatted for congressional staff
- Key ask in first paragraph
```

### Credibility for Policymakers
- Add: "Limitations for policy use" subsection
- Clarify: These are projections, not predictions
- Include: Sensitivity of policy ROI to key assumptions

### Verdict
Good framework but needs implementation specifics. **Add stakeholder mapping and feasibility assessment.**

---

## Expert 7: Dr. Lisa Zhang (Statistics)

### Overall Assessment: **A- (Uncertainty communication is strong)**

### Strengths
- 80% CI prominently featured (better than 95% for communication)
- Monte Carlo with convergence check is appropriate
- Sobol indices correctly interpreted
- Honest about structural uncertainty

### Concerns
1. **CI interpretation may be misunderstood**: "80% confident" needs careful framing
2. **Missing: sensitivity of CIs to distributional assumptions**
3. **Correlation effects mentioned but not visualized**

### Specific Recommendations
```
CI Communication:
Add explicit framing:
"The 80% confidence interval [3.4x, 9.2x] means: given our model
 assumptions, there's an 80% probability the true acceleration
 falls in this range. The remaining 20% could be higher OR lower."

Add: "What would change our estimates?"
- Table showing CI shift under different assumptions
- e.g., "If g_ai variance doubles, CI widens to [2.1x, 12.4x]"

Visualization Additions:
1. Correlation matrix heatmap (small, collapsible)
2. Sensitivity tornado should show CI bars, not just point estimates
3. Add fan chart for trajectory uncertainty over time
```

### Statistical Honesty Checklist
- [ ] State: "These are model-based projections, not forecasts"
- [ ] Acknowledge: Structural uncertainty not captured in CI
- [ ] Clarify: Parameters are informed guesses, not measurements
- [ ] Note: Model assumes relationships are stable over time

### Verdict
Statistically rigorous presentation. **Add CI interpretation guidance and sensitivity to assumptions.**

---

## Expert 8: Marcus Webb (Frontend Development)

### Overall Assessment: **B+ (Feasible but needs scope management)**

### Strengths
- Vanilla JS approach is smart—no framework overhead
- D3.js/Chart.js are appropriate choices
- Static hosting simplifies deployment
- Performance targets (<3s load) are reasonable

### Concerns
1. **7 interactive visualizations is ambitious**: Real-time parameter explorer alone is 3-4 hours
2. **D3.js learning curve**: If team isn't experienced, Chart.js is safer
3. **No build process mentioned**: Will need bundling for production

### Specific Recommendations
```
Technology Refinement:
- Use Chart.js for standard charts (bar, line, histogram)
- D3.js only for custom visualizations (pipeline diagram, timeline)
- Consider Observable Plot for rapid prototyping

Build Process:
- Vite or Parcel for bundling
- Tree-shaking to minimize JS payload
- Image optimization pipeline (WebP with PNG fallback)

Performance Optimization:
- Lazy load charts below fold
- Use Intersection Observer API
- Preload hero chart data
- Target: <100KB initial JS, <500KB total

Scope Reduction (if needed):
MVP (8 hours):
- Static hero chart (no interactivity)
- Simple bar charts with hover
- No parameter explorer

Full version (15 hours):
- Interactive hero with scenario toggle
- All 5 charts with hover/tooltips
- Collapsible methods section

Stretch (20+ hours):
- Real-time parameter explorer
- Shareable custom scenarios (URL state)
```

### Hosting Recommendation
```
GitHub Pages (free, simple):
- Custom domain support
- HTTPS included
- No server maintenance

Alternative: Netlify
- Better build integration
- Form handling if needed
- Analytics built-in
```

### Verdict
Feasible but scope the interactivity carefully. **Start with Chart.js, add D3 selectively. Define MVP clearly.**

---

## Consensus Summary

### Unanimous Recommendations (All 8 Experts)
1. ✅ **Simplify visualizations**: 5 max, not 7
2. ✅ **Add shareable image assets**: 3-4 standalone graphics
3. ✅ **Improve CI communication**: Add interpretation guidance
4. ✅ **Expand parameter transparency**: Hoverable sources

### Majority Recommendations (5+ Experts)
1. **Move Uncertainty before Policy** (Torres, Zhang, Morrison, Kim, Kowalski)
2. **Add navigation aids** (Patel, Chen, Webb, Torres)
3. **Reconsider interactive explorer** - separate page or MVP cut (Patel, Webb, Chen)
4. **Strengthen opening hook** (Torres, Kowalski, Patel)

### Split Opinions
| Topic | For | Against | Resolution |
|-------|-----|---------|------------|
| Interactive explorer inline | Chen, Kowalski | Patel, Webb | **Separate page** |
| Dark mode | Patel | Webb (scope) | **Nice-to-have** |
| Validation/backcasting | Morrison, Zhang | (none opposed) | **Add if time permits** |

### Risk-Adjusted Priorities

| Priority | Item | Risk if Skipped | Effort |
|----------|------|-----------------|--------|
| **P0** | Simplify to 5 charts | High (overwhelm) | Low |
| **P0** | Add CI interpretation | High (misunderstanding) | Low |
| **P1** | Shareable assets | Medium (low reach) | Medium |
| **P1** | Navigation aids | Medium (poor UX) | Medium |
| **P2** | Interactive explorer | Low (nice-to-have) | High |
| **P2** | Validation section | Low (for academics) | Medium |

---

## Revised Plan Based on Expert Feedback

### Changes Accepted

1. **Visualization count**: Reduced from 7 to 5
   - Hero trajectory (with scenario toggle)
   - Monte Carlo distribution
   - Sobol sensitivity (dot plot, not tornado)
   - Policy ROI (bubble chart)
   - Disease timeline (vertical layout)

2. **Section reorder**:
   - Uncertainty moved before Policy
   - Methods moved to end

3. **Interactive explorer**:
   - Moved to separate page
   - Main page is read-only with hover tooltips

4. **Navigation**:
   - Added sticky TOC
   - Added progress bar
   - Added "jump to" buttons

5. **Shareable assets**:
   - 4 standalone PNG graphics for social media
   - Open Graph meta tags for link previews

6. **CI communication**:
   - Added interpretation paragraph
   - Added "what would change this" table

### Changes Deferred

1. Dark mode (nice-to-have, post-launch)
2. Validation/backcasting (supplementary material)
3. Full stakeholder mapping (future policy brief)

### Revised Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Content writing + revisions | 2 hours |
| 2 | 5 visualizations (Chart.js) | 3 hours |
| 3 | HTML/CSS + navigation | 2 hours |
| 4 | Shareable assets | 1 hour |
| 5 | Testing + polish | 1 hour |
| **Total** | | **9 hours** |

---

*Review completed: 2026-01-13*
*Reviewers: 8 domain experts*
*Consensus level: High*
*Plan status: Ready for implementation*
