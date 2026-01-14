# Supplementary Table S4: Domain Boundary Definitions

## AI Research Acceleration Model v1.1

**Purpose**: Provide explicit scope statements for each scientific domain, including inclusions, exclusions, and boundary cases.

---

## S4.1 Domain Overview

| Domain | Scope Summary | Primary Output |
|--------|---------------|----------------|
| Structural Biology | Determination and analysis of 3D molecular structures | Atomic coordinates, structural models |
| Drug Discovery | Development of small molecule and biologic therapeutics | Drug candidates, clinical compounds |
| Materials Science | Discovery and characterization of functional materials | Novel materials, property predictions |
| Protein Design | Engineering of proteins with novel or enhanced function | Designed protein sequences |
| Clinical Genomics | Genetic analysis for clinical decision-making | Variant interpretations, diagnoses |

---

## S4.2 Structural Biology

### Definition
Research focused on determining, predicting, and analyzing the three-dimensional structures of biological macromolecules at atomic or near-atomic resolution.

### Inclusions

| Category | Examples |
|----------|----------|
| **Experimental methods** | X-ray crystallography, cryo-EM, NMR spectroscopy |
| **Computational prediction** | AlphaFold, ESMFold, RoseTTAFold |
| **Structure analysis** | Fold recognition, structural alignment, dynamics simulation |
| **Target molecules** | Proteins, nucleic acids, protein-ligand complexes |
| **Technique development** | New structure determination methods, sample preparation |

### Exclusions

| Category | Reason | Alternative Domain |
|----------|--------|-------------------|
| Protein engineering | Design focus | Protein Design |
| Structure-based drug design | Application focus | Drug Discovery |
| Cell/tissue imaging | Non-atomic resolution | (Not modeled) |
| Genomic structural variants | Sequence-level | Clinical Genomics |

### Boundary Cases

| Case | Classification | Rationale |
|------|---------------|-----------|
| AlphaFold for drug design | Split: Structural Biology (prediction) + Drug Discovery (application) | Track separately where prediction ends and application begins |
| Molecular dynamics | Structural Biology | Structure-focused even if simulating dynamics |
| Cryo-ET | Structural Biology | Atomic-level structural information |
| Integrative modeling | Structural Biology | Primary output is structural model |

---

## S4.3 Drug Discovery

### Definition
Research and development activities aimed at identifying, optimizing, and advancing therapeutic compounds from initial concept through clinical development.

### Inclusions

| Category | Examples |
|----------|----------|
| **Target identification** | GWAS analysis, computational target finding |
| **Hit finding** | High-throughput screening, virtual screening |
| **Lead optimization** | ADMET prediction, medicinal chemistry |
| **Preclinical development** | Toxicology, pharmacokinetics |
| **Clinical trials** | Phase 1-3 studies, biomarker development |
| **Modalities** | Small molecules, antibodies, gene therapies |

### Exclusions

| Category | Reason | Alternative Domain |
|----------|--------|-------------------|
| Basic structural biology | Upstream of application | Structural Biology |
| De novo protein therapeutics | Design focus | Protein Design |
| Companion diagnostics | Diagnostic focus | Clinical Genomics |
| Drug manufacturing | Post-discovery | (Not modeled) |

### Sub-Domain Breakdown

| Sub-domain | Time Fraction | AI Acceleration Potential |
|------------|---------------|--------------------------|
| Target identification | 5% | 3.0x |
| Hit identification | 8% | 4.0x |
| Lead optimization | 12% | 2.5x |
| Preclinical | 15% | 1.5x |
| Phase 1 | 8% | 1.2x |
| Phase 2 | 12% | 1.15x |
| Phase 3 | 20% | 1.1x |
| Regulatory | 10% | 1.3x |
| Manufacturing | 10% | 1.5x |

### Boundary Cases

| Case | Classification | Rationale |
|------|---------------|-----------|
| Designed protein therapeutics | Drug Discovery for development; Protein Design for design | Split based on workflow stage |
| CRISPR therapies | Drug Discovery | Therapeutic focus, clinical development pathway |
| mRNA therapeutics | Drug Discovery | Drug modality, regulatory pathway |
| Biosimilars | Drug Discovery | Same development pathway as originators |

---

## S4.4 Materials Science

### Definition
Research focused on discovering, characterizing, and optimizing functional materials for technological applications.

### Inclusions

| Category | Examples |
|----------|----------|
| **Computational discovery** | GNoME, property prediction |
| **Synthesis** | A-Lab, automated synthesis |
| **Characterization** | Property measurement, structure analysis |
| **Material classes** | Inorganics, polymers, composites, batteries |
| **Applications** | Energy storage, catalysis, electronics |

### Exclusions

| Category | Reason | Alternative Domain |
|----------|--------|-------------------|
| Protein-based materials | Biological focus | Protein Design |
| Drug delivery materials | Application focus | Drug Discovery |
| Biomaterials for implants | Medical device, not material discovery | (Not modeled) |

### Boundary Cases

| Case | Classification | Rationale |
|------|---------------|-----------|
| Bio-inspired materials | Materials Science unless protein engineering central | Based on primary methodology |
| Organic electronics | Materials Science | Materials focus despite organic chemistry |
| Catalysts | Materials Science | Material optimization focus |
| Metal-organic frameworks (MOFs) | Materials Science | Property-driven material class |

### Key Bottleneck: Synthesis

The materials science domain has a unique characteristic:
- **Computational discovery**: Can generate millions of candidates (GNoME: 2.2M)
- **Synthesis capacity**: ~1000 new materials/year globally
- **Gap**: >2000-year "backlog" of untested predictions
- **Model implication**: Near-1x acceleration despite massive computational gains

---

## S4.5 Protein Design

### Definition
Engineering of proteins with novel or enhanced functions through computational design, directed evolution, or hybrid approaches.

### Inclusions

| Category | Examples |
|----------|----------|
| **De novo design** | RFdiffusion, backbone generation |
| **Sequence design** | ProteinMPNN, inverse folding |
| **Enzyme engineering** | Activity optimization, stability improvement |
| **Antibody design** | CDR optimization, humanization |
| **Biosensor design** | Fluorescent proteins, binding sensors |

### Exclusions

| Category | Reason | Alternative Domain |
|----------|--------|-------------------|
| Structure prediction | Prediction, not design | Structural Biology |
| Natural protein characterization | Discovery, not design | Structural Biology |
| Protein therapeutic development | Clinical development | Drug Discovery |
| Variant effect prediction | Diagnostic focus | Clinical Genomics |

### Sub-Type Breakdown

| Sub-type | AI Potential | Validation Fraction | Notes |
|----------|--------------|---------------------|-------|
| Enzyme engineering | 4.5x | 50% | Function can be assayed rapidly |
| De novo design | 6.0x | 70% | Higher validation burden |
| Antibody design | 3.0x | 60% | More constrained design space |
| Scaffold design | 5.0x | 40% | Structural validation only |

### Boundary Cases

| Case | Classification | Rationale |
|------|---------------|-----------|
| Directed evolution | Protein Design | Optimization methodology |
| AlphaFold for design | Structural Biology (structure prediction) but enables Protein Design | Track spillover effect |
| Designed protein therapeutics | Protein Design (design phase) → Drug Discovery (development) | Handoff at IND-enabling |
| Protein purification optimization | Not modeled | Process engineering, not design |

---

## S4.6 Clinical Genomics

### Definition
Application of genomic analysis to clinical diagnosis, prognosis, and treatment decisions for patients.

### Inclusions

| Category | Examples |
|----------|----------|
| **Variant interpretation** | AlphaMissense, ClinVar curation |
| **Variant calling** | DeepVariant, Illumina analysis |
| **Splicing prediction** | SpliceAI, splice variant analysis |
| **Cancer genomics** | Tumor profiling, ctDNA analysis |
| **Pharmacogenomics** | Drug-gene interactions |
| **Rare disease diagnosis** | Exome/genome interpretation |

### Exclusions

| Category | Reason | Alternative Domain |
|----------|--------|-------------------|
| Basic genomics research | Not clinical application | (Not modeled) |
| Evolutionary genomics | Research focus | (Not modeled) |
| Population genetics | Epidemiology focus | (Not modeled) |
| Drug target discovery | Drug development | Drug Discovery |

### Boundary Cases

| Case | Classification | Rationale |
|------|---------------|-----------|
| CRISPR diagnostics | Clinical Genomics | Diagnostic application |
| Companion diagnostics | Clinical Genomics (development) + Drug Discovery (use) | Split based on context |
| Prenatal screening | Clinical Genomics | Clinical genetic testing |
| Ancestry testing | Not modeled | Consumer, not clinical |
| Research variant databases | Clinical Genomics | Enables clinical interpretation |

---

## S4.7 Cross-Domain Overlap Matrix

| | Structural Bio | Drug Discovery | Materials | Protein Design | Clinical Genomics |
|-|----------------|----------------|-----------|----------------|-------------------|
| **Structural Bio** | - | Structure-based drug design | Sample prep advances | Enables design | - |
| **Drug Discovery** | Uses structures | - | Drug delivery | Protein therapeutics | Pharmacogenomics |
| **Materials** | - | Delivery systems | - | Bio-materials | - |
| **Protein Design** | Uses predictions | Biologics | Protein materials | - | Variant-informed |
| **Clinical Genomics** | - | Companion Dx | - | - | - |

---

## S4.8 Scope Clarifications

### What is "Research Acceleration"?

For each domain, acceleration is measured as:

| Domain | Acceleration Metric |
|--------|-------------------|
| Structural Biology | Time from sequence to validated structure |
| Drug Discovery | Time from target to IND/Phase 1 |
| Materials Science | Time from hypothesis to characterized material |
| Protein Design | Time from specification to validated design |
| Clinical Genomics | Time from sample to actionable interpretation |

### What is NOT Included

1. **Basic research**: Fundamental discovery without application focus
2. **Manufacturing**: Scale-up and production (post-discovery)
3. **Healthcare delivery**: Clinical implementation beyond interpretation
4. **Regulatory science**: Policy and regulatory development
5. **Research infrastructure**: Databases, repositories, standards

---

## S4.9 Emerging Technologies Mapping

| Technology | Primary Domain | Secondary Domain | Notes |
|------------|---------------|------------------|-------|
| AlphaFold 3 | Structural Biology | Drug Discovery (via SBDD) | Multi-molecule prediction |
| ESM-3 | Protein Design | Structural Biology | Design + structure |
| GNoME | Materials Science | - | Discovery only |
| AlphaMissense | Clinical Genomics | Protein Design (variant effects) | Classification focus |
| RFdiffusion | Protein Design | - | Backbone generation |
| Autonomous labs | Materials Science | Protein Design | Automation |

---

*Table S4 completed: January 2026*
*AI Research Acceleration Model v1.1*
