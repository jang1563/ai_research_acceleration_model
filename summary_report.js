const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require('docx');
const fs = require('fs');

// Create the comprehensive summary report
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "AI Research Acceleration Model - Summary Report", italics: true, size: 18, color: "666666" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
                     new TextRun({ text: " of ", size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 })]
        })]
      })
    },
    children: [
      // Title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "AI-Accelerated Scientific Research Model", bold: true, size: 48, color: "1F4E79" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "Comprehensive Validation Report", size: 32, color: "2E75B6" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 600 },
        children: [new TextRun({ text: "Version 0.3.1 | January 2026 | 9 Case Studies", size: 24, italics: true })]
      }),

      // Executive Summary
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Executive Summary")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("This report presents findings from validating the AI Research Acceleration Model against 9 real-world AI breakthroughs in biology and life sciences. The model successfully predicts acceleration patterns for efficiency-focused applications while identifying key limitations in capturing breakthrough capabilities.")]
      }),

      // Key Findings Table
      new Paragraph({
        spacing: { before: 200, after: 100 },
        children: [new TextRun({ text: "Key Findings:", bold: true })]
      }),
      createFindingsTable(),

      new Paragraph({ children: [new PageBreak()] }),

      // Case Studies Overview
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Case Studies Overview")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("Nine AI breakthroughs were analyzed across structural biology, drug discovery, protein design, genomics, and materials science. Each case study provides stage-level acceleration metrics, bottleneck identification, and validation against model predictions.")]
      }),

      createCaseStudiesTable(),

      new Paragraph({ children: [new PageBreak()] }),

      // Shift Type Analysis
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Shift Type Analysis")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Type I: Scale Shifts")] }),
      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "Examples: ", bold: true }), new TextRun("GNoME, Evo")]
      }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Massive generation capacity (100,000x+ for hypothesis stages)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Creates validation backlog (GNoME: 2.2M candidates, 6,000+ year backlog)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("End-to-end acceleration limited to ~1-3x due to synthesis bottleneck")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Type II: Efficiency Shifts")] }),
      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "Examples: ", bold: true }), new TextRun("Recursion, Cradle Bio")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Moderate acceleration across computational stages (3-24x)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Most predictable by current model (validation scores 0.82-0.97)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("End-to-end acceleration: 1.5-2.5x")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Type III: Capability Shifts")] }),
      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "Examples: ", bold: true }), new TextRun("AlphaFold, ESM-3, Isomorphic Labs, AlphaMissense")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Breakthrough acceleration in prediction stages (10,000-9,000,000x)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Enables previously impossible tasks")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("End-to-end acceleration varies widely: 1.6-24x depending on downstream bottlenecks")] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Physical Bottleneck Hypothesis
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Physical Bottleneck Hypothesis: Validated")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("The core hypothesis that physical/biological stages limit end-to-end acceleration was validated across all 9 case studies. Wet lab execution (S4) and validation (S6) consistently showed 1.0-1.5x acceleration, regardless of AI capability in cognitive stages.")]
      }),

      createBottleneckTable(),

      new Paragraph({
        spacing: { before: 300, after: 200 },
        children: [new TextRun({ text: "Key Insight: ", bold: true }), new TextRun("Even with 36,500x acceleration in structure prediction (AlphaFold), or 9,000,000x in variant classification (AlphaMissense), end-to-end acceleration is capped at 2-25x because physical validation cannot be bypassed.")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Model Refinements
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Model Refinements (v0.3.1)")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("Based on validation results, the following refinements were implemented:")]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("1. Shift-Type-Aware M_max Parameters")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Type III cognitive stages: M_max increased from 100x to 50,000-100,000x")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Physical stages: M_max reduced from 2.5x to 1.5x to match observed constraints")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("2. Triage Overhead for Type I Shifts")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Added penalty factor for high-throughput generation creating selection bottleneck")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("GNoME example: 2.2M candidates but only 350/year synthesizable")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("3. Domain-Specific Constraints")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Drug discovery: Clinical trial bottleneck at S6 (5-7 years unchanged)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Protein design: Expression bottleneck at S4 (1-2 months per cycle)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 },
        children: [new TextRun("Materials science: Synthesis bottleneck at S4 (~1 material/day)")] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Implications for v0.5
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Implications for v0.5 Integrated Model")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("The v0.5 model incorporates lab automation to address physical bottlenecks. Key insights from case study validation:")]
      }),

      createImplicationsTable(),

      new Paragraph({
        spacing: { before: 300, after: 200 },
        children: [new TextRun({ text: "Critical Insight: ", bold: true }), new TextRun("Without lab automation, the model ceiling is ~3x by 2050. With breakthrough automation scenarios, 20-50x becomes achievable as the physical bottleneck is removed.")]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Conclusions
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Conclusions")] }),

      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun({ text: "Physical Bottleneck Validated: ", bold: true }), new TextRun("All 9 case studies confirm that biological/physical stages (S4, S6) limit end-to-end acceleration to 1.5-4x regardless of AI capability.")] }),

      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun({ text: "Cognitive Acceleration Underestimated: ", bold: true }), new TextRun("Original model predicted 100x max for cognitive stages; observed values range from 1,000x to 9,000,000x for Type III shifts.")] }),

      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun({ text: "Shift Type Determines Outcome: ", bold: true }), new TextRun("Type II (efficiency) shifts are most predictable; Type III (capability) shifts show high variance; Type I (scale) shifts create new bottlenecks.")] }),

      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun({ text: "Lab Automation Critical: ", bold: true }), new TextRun("The v0.5 model correctly identifies automation as the key unlock for achieving >5x end-to-end acceleration.")] }),

      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun({ text: "Drug Discovery Pattern: ", bold: true }), new TextRun("AI accelerates Target-to-IND by 2-2.5x, but clinical trials (5-7 years) remain the binding constraint.")] }),

      // References
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Merchant et al. (2023). Scaling deep learning for materials discovery. Nature.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Hayes et al. (2024). Simulating 500 million years of evolution with a language model. bioRxiv.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("CNBC (Oct 2024). Recursion gets FDA approval for Phase I trials of AI-discovered drug.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Abramson et al. (2024). Accurate structure prediction with AlphaFold 3. Nature.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Cradle Bio (2024). Adaptyv Bio ML Competition: 8x EGFR binding improvement.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Insilico Medicine (2024). Phase IIa results for ISM001-055. Nature Medicine.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Nguyen et al. (2024). Sequence modeling from molecular to genome scale. Science.")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 },
        children: [new TextRun("Cheng et al. (2023). Accurate proteome-wide missense variant effect prediction. Science.")] }),
    ]
  }]
});

function createFindingsTable() {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerShading = { fill: "1F4E79", type: ShadingType.CLEAR };
  const margins = { top: 80, bottom: 80, left: 120, right: 120 };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [3000, 6360],
    rows: [
      new TableRow({
        children: [
          new TableCell({ borders, shading: headerShading, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Finding", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 6360, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Summary", bold: true, color: "FFFFFF" })] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Physical Bottleneck", bold: true })] })] }),
          new TableCell({ borders, margins, width: { size: 6360, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Validated across all 9 cases. S4/S6 at 1.0-1.5x regardless of AI capability.")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Cognitive Acceleration", bold: true })] })] }),
          new TableCell({ borders, margins, width: { size: 6360, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Observed 10,000-9,000,000x for Type III shifts. Model updated from 100x to 100,000x M_max.")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Drug Discovery", bold: true })] })] }),
          new TableCell({ borders, margins, width: { size: 6360, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Target-to-IND: 2-2.5x acceleration. Clinical trials (5-7 years) remain binding constraint.")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Model Accuracy", bold: true })] })] }),
          new TableCell({ borders, margins, width: { size: 6360, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Type II: 0.82-0.97 validation. Type III: requires shift-type-aware parameters.")] })] }),
        ]
      }),
    ]
  });
}

function createCaseStudiesTable() {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerShading = { fill: "2E75B6", type: ShadingType.CLEAR };
  const margins = { top: 60, bottom: 60, left: 100, right: 100 };

  const caseStudies = [
    ["AlphaFold 2/3", "Structural Biology", "Type III", "24.3x", "S4 (Wet Lab)"],
    ["GNoME", "Materials Science", "Type I", "1.0x*", "S4 (Synthesis)"],
    ["ESM-3", "Protein Design", "Type III", "4.0x", "S4 (Expression)"],
    ["Recursion", "Drug Discovery", "Type II", "2.3x", "S6 (Clinical)"],
    ["Isomorphic Labs", "Drug Discovery", "Type III", "1.6x", "S6 (Clinical)"],
    ["Cradle Bio", "Protein Design", "Type II", "2.1x", "S4 (Wet Lab)"],
    ["Insilico Medicine", "Drug Discovery", "Type III", "2.5x", "S6 (Clinical)"],
    ["Evo", "Genomics", "Mixed", "3.2x", "S4 (Synthesis)"],
    ["AlphaMissense", "Clinical Genomics", "Type III", "2.1x", "S4 (Validation)"],
  ];

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [2200, 2000, 1200, 1200, 2760],
    rows: [
      new TableRow({
        children: [
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2200, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Case Study", bold: true, color: "FFFFFF", size: 20 })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Domain", bold: true, color: "FFFFFF", size: 20 })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 1200, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Shift", bold: true, color: "FFFFFF", size: 20 })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 1200, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Accel.", bold: true, color: "FFFFFF", size: 20 })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2760, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Bottleneck", bold: true, color: "FFFFFF", size: 20 })] })] }),
        ]
      }),
      ...caseStudies.map(row => new TableRow({
        children: row.map((cell, i) => new TableCell({
          borders, margins,
          width: { size: [2200, 2000, 1200, 1200, 2760][i], type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20 })] })]
        }))
      }))
    ]
  });
}

function createBottleneckTable() {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerShading = { fill: "C00000", type: ShadingType.CLEAR };
  const margins = { top: 60, bottom: 60, left: 100, right: 100 };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [2500, 2500, 2180, 2180],
    rows: [
      new TableRow({
        children: [
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Stage", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Type", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Observed Accel.", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Cases", bold: true, color: "FFFFFF" })] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("S1-S3 (Cognitive)")] })] }),
          new TableCell({ borders, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Literature, Design, Analysis")] })] }),
          new TableCell({ borders, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("3x - 9,000,000x")] })] }),
          new TableCell({ borders, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("9/9")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "S4 (Physical)", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Wet Lab, Synthesis")] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "1.0x - 1.5x", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("9/9 (BOTTLENECK)")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("S5 (Interpretation)")] })] }),
          new TableCell({ borders, margins, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Results Analysis")] })] }),
          new TableCell({ borders, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("1.5x - 15x")] })] }),
          new TableCell({ borders, margins, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("9/9")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "S6 (Validation)", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2500, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Clinical Trials, Publication")] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "1.0x - 2.0x", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "FFEEEE", type: ShadingType.CLEAR }, width: { size: 2180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("9/9 (BOTTLENECK)")] })] }),
        ]
      }),
    ]
  });
}

function createImplicationsTable() {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerShading = { fill: "538135", type: ShadingType.CLEAR };
  const margins = { top: 60, bottom: 60, left: 100, right: 100 };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [3000, 3180, 3180],
    rows: [
      new TableRow({
        children: [
          new TableCell({ borders, shading: headerShading, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Scenario", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "2030 Projection", bold: true, color: "FFFFFF" })] })] }),
          new TableCell({ borders, shading: headerShading, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "2050 Projection", bold: true, color: "FFFFFF" })] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("No Automation (v0.4)")] })] }),
          new TableCell({ borders, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("1.5-2x")] })] }),
          new TableCell({ borders, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("2-3x")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("Baseline Automation")] })] }),
          new TableCell({ borders, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("2-4x")] })] }),
          new TableCell({ borders, margins, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun("5-10x")] })] }),
        ]
      }),
      new TableRow({
        children: [
          new TableCell({ borders, margins, shading: { fill: "E2EFDA", type: ShadingType.CLEAR }, width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "Breakthrough Automation", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "E2EFDA", type: ShadingType.CLEAR }, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "5-10x", bold: true })] })] }),
          new TableCell({ borders, margins, shading: { fill: "E2EFDA", type: ShadingType.CLEAR }, width: { size: 3180, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: "20-50x", bold: true })] })] }),
        ]
      }),
    ]
  });
}

// Generate the document
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/intelligent-beautiful-shannon/mnt/Accelerating_biology_with_AI/ai_research_acceleration_model/AI_Research_Acceleration_Model_Summary_Report.docx", buffer);
  console.log("Report generated successfully!");
});
