# AI Bio Acceleration Model - Brand Design System

## Overview

This design system creates a cohesive visual theme for the AI Bio Acceleration Model blog, conveying the core concept: **AI accelerates biological research by 2.8x, but physical bottlenecks limit gains**.

---

## 1. Color Palette

### Primary Colors - "Velocity Blue"
Represents digital acceleration, AI, and computational speed.

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-primary-500` | `#0967d2` | Primary buttons, links, accents |
| `--color-primary-400` | `#2186eb` | Hover states, highlights |
| `--color-primary-600` | `#0552b5` | Active states, borders |
| `--color-primary-900` | `#002159` | Dark backgrounds with blue tint |

### Secondary Colors - "Bio Green"
Represents biological systems, life sciences, and organic processes.

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-secondary-500` | `#0ca750` | Success states, bio-related elements |
| `--color-secondary-400` | `#1db954` | Highlights, positive indicators |
| `--color-secondary-600` | `#058a42` | Active states |

### Accent Colors - "Acceleration Gold"
Represents breakthroughs, key insights, and acceleration highlights.

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-accent-500` | `#f5a623` | Call-to-action highlights, key stats |
| `--color-accent-400` | `#ffc21a` | Hover states on accent elements |

### Constraint Colors - "Bottleneck Red"
Represents physical constraints, limitations, and pipeline discounts.

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-constraint-500` | `#e53e3e` | Constraint indicators, warnings |
| `--color-constraint-400` | `#ff2626` | Highlighted constraints |

### Dark Mode Neutrals - "Deep Space"

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-neutral-900` | `#111118` | Main background |
| `--color-neutral-850` | `#16161f` | Elevated backgrounds |
| `--color-neutral-800` | `#1c1c27` | Card backgrounds |
| `--color-neutral-700` | `#2a2a3a` | Borders, dividers |
| `--color-neutral-200` | `#d1d1e0` | Body text |
| `--color-neutral-50` | `#f5f5fa` | Headings, bright text |

---

## 2. Typography

### Font Stack

```css
/* Display headings - modern, geometric */
--font-display: 'Space Grotesk', -apple-system, sans-serif;

/* Body text - highly readable */
--font-body: 'Inter', -apple-system, 'Segoe UI', sans-serif;

/* Code and data */
--font-mono: 'JetBrains Mono', 'SF Mono', monospace;
```

### Font Loading (add to HTML head)

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700;800&display=swap" rel="stylesheet">
```

### Type Scale

| Size Token | Range | Usage |
|------------|-------|-------|
| `--text-5xl` | 48-72px | Hero titles |
| `--text-4xl` | 36-52px | Section headings (h2) |
| `--text-3xl` | 30-40px | Subsection headings (h3) |
| `--text-2xl` | 24-30px | Card titles (h4) |
| `--text-xl` | 20-24px | Lead paragraphs |
| `--text-base` | 16-17px | Body text |
| `--text-sm` | 12-14px | Captions, meta |

---

## 3. Visual Motifs

### Gradients

**Acceleration Gradient** - Primary brand gradient
```css
background: var(--gradient-acceleration);
/* linear-gradient(135deg, #0552b5, #0967d2, #0ca750, #1db954) */
```

**Speed Lines Gradient** - Motion effect
```css
background: var(--gradient-speed-lines);
/* linear-gradient(90deg, transparent, #0967d2, #2186eb, transparent) */
```

**Constraint Gradient** - Bottleneck visualization
```css
background: var(--gradient-constraint);
/* linear-gradient(90deg, #c53030, #e53e3e, #f5a623) */
```

### Glow Effects

```css
/* Primary glow for key elements */
box-shadow: var(--glow-primary);
/* 0 0 20px rgba(9, 103, 210, 0.4) */

/* Accent glow for highlights */
box-shadow: var(--glow-accent);
/* 0 0 20px rgba(245, 166, 35, 0.4) */
```

### Grid Pattern

The hero sections feature an animated background grid that moves diagonally, creating a sense of forward momentum:

```css
background-image:
  linear-gradient(var(--color-neutral-700) 1px, transparent 1px),
  linear-gradient(90deg, var(--color-neutral-700) 1px, transparent 1px);
background-size: 50px 50px;
animation: grid-move 20s linear infinite;
```

---

## 4. Animation Patterns

### Acceleration Easings

```css
/* Standard acceleration */
--ease-accelerate: cubic-bezier(0.4, 0, 1, 1);

/* Deceleration (arrival) */
--ease-decelerate: cubic-bezier(0, 0, 0.2, 1);

/* Spring bounce */
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
```

### Key Animations

**Speed Lines**
Horizontal lines that streak across elements, conveying motion.

```html
<div class="speed-lines">
  <!-- Content with speed line effect -->
</div>
```

**Pulse Acceleration**
Pulsing glow effect for key statistics.

```html
<div class="pulse-acceleration">2.8x</div>
```

**Gradient Shift**
Subtle background animation for hero sections.

```html
<div class="gradient-animate">
  <!-- Content with shifting gradient bg -->
</div>
```

**Progress Shimmer**
Shimmer effect on progress bars indicating ongoing process.

```html
<div class="progress">
  <div class="progress__bar" style="width: 70%"></div>
</div>
```

---

## 5. Component Styling

### Cards

```html
<!-- Standard card -->
<div class="card">
  <div class="card__header">
    <h3 class="card__title">Title</h3>
    <p class="card__subtitle">Subtitle</p>
  </div>
  <div class="card__body">
    Content goes here
  </div>
  <div class="card__footer">
    Footer actions
  </div>
</div>

<!-- Highlighted card -->
<div class="card card--highlight">...</div>

<!-- Constraint indicator card -->
<div class="card card--constraint">...</div>

<!-- Success indicator card -->
<div class="card card--success">...</div>
```

### Buttons

```html
<!-- Primary button -->
<button class="btn btn--primary">Get Started</button>

<!-- Secondary button -->
<button class="btn btn--secondary">Learn More</button>

<!-- Ghost button -->
<button class="btn btn--ghost">Cancel</button>

<!-- Accent button -->
<button class="btn btn--accent">Key Action</button>

<!-- Sizes -->
<button class="btn btn--primary btn--sm">Small</button>
<button class="btn btn--primary btn--lg">Large</button>
```

### Tables

```html
<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Stage</th>
        <th class="numeric">Duration</th>
        <th class="numeric">Acceleration</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Data Analysis</td>
        <td class="numeric">2 weeks</td>
        <td class="numeric">10x</td>
      </tr>
      <tr class="constraint">
        <td>Clinical Trials</td>
        <td class="numeric">3 years</td>
        <td class="numeric">1.0x</td>
      </tr>
    </tbody>
  </table>
</div>
```

### Callouts

```html
<!-- Info callout -->
<div class="callout callout--info">
  <div class="callout__title">Key Finding</div>
  <div class="callout__content">Important information here.</div>
</div>

<!-- Constraint callout -->
<div class="callout callout--constraint">
  <div class="callout__title">Physical Bottleneck</div>
  <div class="callout__content">This stage cannot be accelerated.</div>
</div>

<!-- Key insight (special treatment) -->
<div class="callout callout--key-insight">
  <div class="callout__title">Core Insight</div>
  <div class="callout__content">2.8x acceleration, not 100x</div>
</div>
```

### Stats Display

```html
<div class="stat">
  <div class="stat__value stat__value--acceleration">2.8x</div>
  <div class="stat__label">Overall Acceleration</div>
  <div class="stat__change stat__change--up">+0.3 from 2023</div>
</div>
```

### Pipeline Stages

```html
<div class="pipeline-stage pipeline-stage--fast">
  <div class="pipeline-stage__icon">AI</div>
  <div class="pipeline-stage__info">
    <div class="pipeline-stage__name">Data Analysis</div>
    <div class="pipeline-stage__duration">2 weeks (was 6 months)</div>
  </div>
</div>

<div class="pipeline-stage pipeline-stage--slow">
  <div class="pipeline-stage__icon">Lab</div>
  <div class="pipeline-stage__info">
    <div class="pipeline-stage__name">Clinical Trials</div>
    <div class="pipeline-stage__duration">3 years (unchanged)</div>
  </div>
</div>
```

---

## 6. Audience-Specific Themes

Apply these classes to the `<body>` or main container:

### Technical/Research Post
```html
<body class="theme-technical">
```
- Blue-focused glow
- More subdued animations
- Emphasis on data precision

### Student Guide
```html
<body class="theme-student">
```
- Green/blue balanced glow
- Slightly more vibrant
- Engaging visual elements

### Academic Manuscript
```html
<body class="theme-academic">
```
- Reduced animations
- More formal spacing
- Subdued effects

---

## 7. Page Templates

### Hero Section

```html
<section class="hero">
  <div class="hero__bg-grid"></div>
  <div class="hero__glow"></div>
  <div class="container hero__content">
    <h1 class="hero__title">
      AI Accelerates Biology by
      <span class="text-gradient">2.8x</span>
    </h1>
    <p class="hero__subtitle">
      Understanding the gap between digital speed and physical constraints
    </p>
    <div class="flex flex--center flex--gap">
      <a href="#" class="btn btn--primary btn--lg">Read Research</a>
      <a href="#" class="btn btn--ghost btn--lg">View Data</a>
    </div>
  </div>
</section>
```

### Navigation

```html
<nav class="nav">
  <div class="container nav__inner">
    <a href="/" class="nav__logo">AI Bio Model</a>
    <ul class="nav__links">
      <li><a href="#" class="nav__link nav__link--active">Technical</a></li>
      <li><a href="#" class="nav__link">Student Guide</a></li>
      <li><a href="#" class="nav__link">Manuscript</a></li>
    </ul>
  </div>
</nav>
```

### Content Section

```html
<section class="section">
  <div class="container container--prose">
    <h2>The Pipeline Problem</h2>
    <p class="lead">
      While AI can accelerate computational tasks by orders of magnitude,
      physical processes remain stubbornly resistant to speedup.
    </p>
    <!-- Content -->
  </div>
</section>
```

---

## 8. Implementation Checklist

### For All Posts

1. Link the CSS file:
   ```html
   <link rel="stylesheet" href="design-system/ai-bio-theme.css">
   ```

2. Add Google Fonts to `<head>`:
   ```html
   <link rel="preconnect" href="https://fonts.googleapis.com">
   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
   <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700;800&display=swap" rel="stylesheet">
   ```

3. Add appropriate theme class to body

4. Use consistent component classes

### Post-Specific Settings

| Post | Theme Class | Notes |
|------|-------------|-------|
| Technical Analysis | `theme-technical` | Use more data tables, code blocks |
| Student Guide | `theme-student` | More callouts, simpler cards |
| Manuscript | `theme-academic` | Minimal animation, formal layout |

---

## 9. Accessibility Notes

- All color combinations meet WCAG AA contrast requirements
- Animations respect `prefers-reduced-motion`
- Focus states are clearly visible
- Print styles included for academic use

---

## 10. File Structure

```
design-system/
  ai-bio-theme.css      # Complete CSS file
  DESIGN-GUIDE.md       # This documentation
```

Apply the theme by linking `ai-bio-theme.css` in your HTML files.
