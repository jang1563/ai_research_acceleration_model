# Webpage Publication Plan: AI-Accelerated Biological Discovery Model

## Overview

**Goal:** Create a single, self-contained webpage that presents the AI-Accelerated Biological Discovery Model with technical rigor but accessible presentation.

**URL Structure:** Single HTML page (e.g., `ai-bio-acceleration.html` or hosted on GitHub Pages)

**Target Audience:**
- Scientists and researchers interested in AI/biology intersection
- Policy makers considering AI investment
- Tech-savvy general audience interested in AI progress
- Investors and funders in biotech/AI

---

## Page Structure

### 1. Hero Section
- **Headline:** "How Much Will AI Accelerate Biological Discovery?"
- **Subheadline:** "A quantitative model projecting 3.4x-9.2x acceleration by 2050"
- **Hero Visualization:** Interactive acceleration trajectory (2024-2050)
- **Key stat callouts:**
  - "80% confident"
  - "5.7x mean acceleration"
  - "$47T potential value"

### 2. Executive Summary (TL;DR)
- 3-4 bullet points with key findings
- One-paragraph abstract
- "Jump to" navigation links

### 3. The Model
#### 3.1 Framework Overview
- 10-stage drug discovery pipeline diagram (interactive)
- How AI affects each stage
- Key mechanisms: target identification → clinical trials → approval

#### 3.2 Key Parameters
- Collapsible table with parameters and sources
- Highlight: g_ai (AI growth rate) dominates at 91.5% sensitivity

#### 3.3 Scenarios
- Interactive chart: Pessimistic / Baseline / Optimistic / Amodei
- Slider to explore different g_ai values

### 4. Results
#### 4.1 Acceleration Projections
- Main finding: 80% CI [3.4x, 9.2x] by 2050
- Monte Carlo distribution visualization
- Comparison with historical rates

#### 4.2 Sensitivity Analysis
- Tornado diagram (Sobol indices)
- Key insight: AI growth rate is everything

#### 4.3 Disease-Specific Projections
- Interactive timeline: When will we cure X?
- 12 diseases with probability estimates
- Hover for details

### 5. Policy Implications
#### 5.1 ROI Rankings
- Bar chart: Top 8 interventions by ROI
- Tier system (Tier 1 > 10,000x ROI)

#### 5.2 Recommendations
- 3-tier action framework
- Cost-benefit summary table

### 6. Uncertainty & Limitations
- Honest assessment of model limitations
- What could make us wrong (upside and downside)
- Known unknowns vs unknown unknowns

### 7. Methods
- Collapsible technical details
- Monte Carlo methodology
- Sobol sensitivity analysis
- Brief mention: "Model developed with AI assistance (Claude)"
- Link to full GitHub repository

### 8. Interactive Explorer (Optional)
- Parameter sliders to run custom scenarios
- Real-time output updates
- "What if" exploration

### 9. Footer
- Download links: PDF, data, code
- Citation format
- Author/contact info
- GitHub repository link

---

## Technical Implementation

### Technology Stack
```
- HTML5 + CSS3 (responsive design)
- JavaScript (vanilla or lightweight framework)
- D3.js or Chart.js for visualizations
- No backend required (static page)
- Optional: Observable/Jupyter widgets for interactivity
```

### Visualizations (5 Main + 2 Interactive)

| # | Visualization | Type | Interactivity |
|---|---------------|------|---------------|
| 1 | Hero: Acceleration Trajectories | Line chart | Hover tooltips, scenario toggle |
| 2 | Pipeline Diagram | Flow diagram | Click to expand stages |
| 3 | Monte Carlo Distribution | Histogram + CDF | Hover for percentiles |
| 4 | Sobol Tornado | Horizontal bar | Hover for details |
| 5 | Disease Timeline | Gantt-style | Hover for probability |
| 6 | Policy ROI | Bar chart | Sort/filter options |
| 7 | Parameter Explorer | Sliders + output | Real-time updates |

### Design Principles
- **Mobile-responsive:** Works on all devices
- **Fast loading:** < 3 seconds initial load
- **Accessible:** WCAG 2.1 AA compliant
- **Print-friendly:** CSS print stylesheet
- **Dark mode:** Optional toggle

### Color Palette
```css
--primary: #2E86AB;      /* Blue - main accent */
--secondary: #A23B72;    /* Magenta - secondary */
--accent: #F18F01;       /* Orange - highlights */
--success: #28A745;      /* Green - positive */
--warning: #DC3545;      /* Red - caution */
--neutral: #3B3B3B;      /* Dark gray - text */
--background: #FAFAFA;   /* Light gray - bg */
```

---

## Content Specifications

### Word Count Targets
| Section | Words | % of Total |
|---------|-------|------------|
| Hero + Executive | 200 | 5% |
| The Model | 800 | 20% |
| Results | 1,200 | 30% |
| Policy | 600 | 15% |
| Uncertainty | 400 | 10% |
| Methods | 600 | 15% |
| Interactive | 200 | 5% |
| **Total** | **~4,000** | 100% |

### Tone Guidelines
- Technical but accessible
- Data-driven claims with uncertainty bounds
- Avoid hype; honest about limitations
- Active voice, clear structure
- Define jargon on first use

### Key Messages (Priority Order)
1. AI will likely accelerate biological discovery 3-9x by 2050
2. The AI growth rate (g_ai) is the dominant uncertainty
3. Regulatory reforms offer highest ROI policy interventions
4. Uncertainty is large but skewed toward optimism
5. Model is open-source and reproducible

---

## Deliverables

### Primary
1. `index.html` - Main webpage
2. `styles.css` - Stylesheet
3. `app.js` - Interactivity
4. `/assets/` - Images, fonts
5. `/data/` - JSON data files for charts

### Supporting
6. `README.md` - GitHub repository readme
7. `og-image.png` - Social sharing image
8. `favicon.ico` - Browser icon

---

## Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1 | Content writing | 2-3 hours |
| 2 | Visualization code | 3-4 hours |
| 3 | HTML/CSS layout | 2-3 hours |
| 4 | Interactivity | 2-3 hours |
| 5 | Testing & polish | 1-2 hours |
| **Total** | | **10-15 hours** |

---

## Success Metrics

- **Engagement:** Time on page > 3 minutes
- **Sharing:** Social shares, backlinks
- **Technical:** Lighthouse score > 90
- **Reach:** Views from target audiences
- **Impact:** Citations, policy discussions

---

## Open Questions for Expert Review

1. Is the balance between technical depth and accessibility right?
2. Which visualizations are most essential vs. nice-to-have?
3. Should the interactive parameter explorer be included or is it scope creep?
4. How prominent should the "AI-assisted" methodology note be?
5. What's missing that would strengthen credibility?
6. Is the 4,000-word target appropriate for web consumption?
7. Should we include a "How to cite" section?
8. What would make this more shareable/viral?

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overcomplicated interactivity | Medium | High | Start simple, add features iteratively |
| Too technical for general audience | Low | Medium | Beta test with non-experts |
| Performance issues on mobile | Medium | Medium | Lazy load visualizations |
| Misinterpretation of uncertainty | Medium | High | Clear CI labeling, honest caveats |
| Scope creep | High | Medium | Strict MVP definition |

---

*Plan Version: 1.0*
*Created: 2026-01-13*
*Status: Ready for Expert Review*
