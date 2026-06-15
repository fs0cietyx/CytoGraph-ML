# Framework & Model Limitations

While CytoGraph-ML demonstrates robust statistical validation, zero-leakage cross-validation, and explainable attributions, several biological, clinical, and methodological limitations must be acknowledged before interpreting its outputs.

---

## 1. Biological Limitations

### Bulk Transcriptomics vs. Single-Cell Resolution
* **The Issue:** CytoGraph-ML analyzes bulk transcriptomic microarrays and RNA-Seq counts. Bulk sequencing measures the average gene expression across millions of cells in a tumor biopsy sample.
* **The Limitation:** Bulk expression averages out stromal contamination, vascular tissues, and infiltrating immune cell profiles. A high expression of *SLIT3* or *EPAS1* may originate from surrounding non-malignant vasculature or tumor-associated macrophages rather than the cancer cells themselves. Single-cell RNA-Seq (scRNA-Seq) or cell deconvolution (e.g., CIBERSORT) is required to resolve cellular origin.

### Single-Omic Constraint
* **The Issue:** Cancer is driven by multi-faceted genomic alterations: DNA mutations, copy number variations (CNV), methylation shifts, chromatin accessibility, and post-translational protein alterations.
* **The Limitation:** CytoGraph-ML operates strictly on transcriptomic (expression) data. It cannot identify transcriptionally silent oncogenic mutations (e.g., specific driver point mutations in *KRAS* or *EGFR*) that govern pathway activation without altering absolute mRNA levels.

---

## 2. Clinical & Diagnostic Limitations

### Diagnostic vs. Prognostic Scope
* **The Issue:** CytoGraph-ML is configured as a binary and tissue-of-origin diagnostic classifier (e.g., separating Lung Adenocarcinoma from Adjacent Non-Malignant controls).
* **The Limitation:** The framework does not predict clinical stage progression, therapeutic recurrence, or overall patient survival time. It is a diagnostic support aid rather than a prognostic forecasting tool.

### Platform and Technical Batch Limits
* **The Issue:** Sequencing platform scales vary widely (e.g., Affymetrix GPL96 vs. GPL570, or Illumina RNA-Seq vs. microarrays).
* **The Limitation:** Quantile Normalization aligns global distributions but does not eliminate gene-level batch biases. Directly running single-sample inference across vastly different technical platforms without reference-based alignment (R-QN) will degrade specificity due to scale drift.

---

## 3. Methodological Limitations

### Post-Hoc Local Interpretability
* **The Issue:** TreeSHAP calculates feature attributions based on cooperative game theory.
* **The Limitation:** SHAP values explain the model's internal decision boundaries; they do not establish biological causality. A high SHAP attribution for *LDB2* or *SLIT3* denotes statistical significance inside the random forest splits, which must be validated in vitro via gene knock-out assays (e.g., CRISPR screens) to verify genuine oncogenic pathways.
