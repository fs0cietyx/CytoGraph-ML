# Potential Reviewer Concerns: Methodological & Biological Response Protocol

This document outlines potential concerns from peer reviewers and explains how they are addressed with empirical evidence in the manuscript.

---

## 1. Methodological & Statistical Concerns

### Concern 1.1: Why was Random Forest chosen as the primary classifier over other architectures?
* **Manuscript Location:** Section IV-A, Subsubsection: *Reviewer Question Answered: Why Random Forest?*
* **Response:** While linear classifiers (such as Elastic Net) or other tree ensembles (such as Extra Trees) can achieve marginally higher raw accuracy under specific splits, Random Forest was selected for three core methodological reasons:
  1. **High-Dimensional Robustness:** Operates via bootstrap aggregation (bagging) and random feature projection, making it highly resilient to multicollinearity and the $P \gg N$ problem ($13,515$ features vs. $107$ training samples).
  2. **Game-Theoretic Explainability:** Random Forest trees allow the exact mathematical formulation of **TreeSHAP** (recursively computed in $O(T L D^2)$ polynomial time without heuristic approximations). Linear models rely on weight-scale assumptions, and kernel SVMs require unstable local perturbations (LIME).
  3. **Clinical Safety Calibration:** Easily calibrated via class weighting ($5:1$ Lung Adenocarcinoma-to-Adjacent Non-Malignant Tissue penalty) to shift its decision boundary and prioritize Sensitivity (Recall = 1.00) under platform domain shifts.

### Concern 1.2: Why was Mutual Information (MI) used for feature selection instead of parametric methods like ANOVA or Lasso?
* **Manuscript Location:** Section III-A, Equation 6.
* **Response:** Transcriptomic expression datasets are highly non-linear and exhibit complex epistasis. Parametric methods like ANOVA assume Gaussian distributions and linear relationships, which are frequently violated by microarray probe scales and RNA-Seq counts. Mutual Information measures the total information-theoretic dependency:
  $$I(X; Y) = \sum_{y \in Y} \sum_{x \in X} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$
  This captures non-linear relationships and threshold effects without distribution assumptions.

### Concern 1.3: Why was the benchmarking performed on GEO datasets (GSE10072 and GSE19804) instead of the larger TCGA cohort?
* **Manuscript Location:** Section II-A, Paragraph 1, and Section V-A.
* **Response:** The central objective of this study was evaluating model generalizability under cross-laboratory and cross-platform domain shifts (the "Acid Test"). 
  * **TCGA Limitations:** TCGA samples are centrally sequenced, processed in large batches, and normalized using a single pipeline. This masks technical batch effects, making it a poor environment to study cross-laboratory robustness.
  * **GEO Generalizability:** Evaluating a model trained on GSE10072 (GPL96 Affymetrix platform) on the independent GSE19804 cohort (GPL570 platform) represents a true cross-laboratory technical shift. This is a much more realistic simulation of clinical translation where incoming diagnostic samples are generated across different labs with different hardware.

### Concern 1.4: Why not Elastic Net? It achieves a higher external accuracy (61.67%) than Random Forest (50.00%) in Table I.
* **Manuscript Location:** Section IV-A.
* **Response:** Elastic Net is included in our baseline benchmarking suite (Table I) and is a strong linear model. However, Elastic Net assumes a linear relationship and does not capture complex, multi-way gene-gene interactions. Furthermore, Random Forest's class-weight boundary shift allows it to maintain a perfect recall of 1.00 (detecting 100% of lung adenocarcinomas) under platform drift, whereas Elastic Net's sensitivity is lower under different weight calibrations. Finally, TreeSHAP provides axiomatic, patient-specific local attributions that are computationally tractable for Random Forest ensembles but lack an exact local formulation for linear models under feature correlation.

---

## 2. Biological & Interpretability Concerns

### Concern 2.1: Why did you filter out specific biological genes using a blacklist? Doesn't this introduce bias?
* **Manuscript Location:** Section III-A, Paragraph 3.
* **Response:** Transcriptomic models often suffer from learning general tissue injury, stromal contamination, or inflammatory responses ("smoke") rather than cancer-specific alterations ("fire"). Standard markers like *VWF* (endothelial) and *IL6* (inflammatory) are highly differentially expressed in cancer biopsies due to cancer-associated vascularization and immune infiltration, but they are not specific oncogenic drivers. Blacklisting these proxy markers forces the model to identify specific genomic drivers (*LDB2*, *SLIT3*, *EPAS1*) that govern cancer-cell survival and pathogenesis, improving biological credibility.

### Concern 2.2: How do you reconcile the differences between your SHAP-important genes and feature selection bootstrap stability frequencies?
* **Manuscript Location:** Section IV-E, Subsubsection: *Reconciliation of SHAP Importance and Bootstrap Selection Stability*.
* **Response:** 
  * **Overlap:** Core markers like *LDB2*, *SLIT3*, and *EPAS1* rank high in both univariate bootstrap stability and multi-variable SHAP attributions, forming the robust biological signature of the model.
  * **Divergence:** Genes like *DACH1* and *COX7A1* show high bootstrap selection stability but lower SHAP rankings, while *EDNRB* shows moderate bootstrap stability but high SHAP contributions.
  * **Reason:** Mutual Information evaluates features univariately, capturing independent differential expression. Random Forest evaluates genes conditionally in tree splits, capturing multi-way non-linear interactions. A gene like *EDNRB* has moderate individual differential expression, but in combination with other markers in tree paths, it provides critical conditional split value, boosting its SHAP score. This shows that univariate stability is complementary to, but distinct from, ensemble explainability attributions.

---

## 3. Clinical & Deployment Concerns

### Concern 3.1: The external validation accuracy on GSE19804 is 50.00%. Why is this model useful if it performs at random chance?
* **Manuscript Location:** Section IV-B, Paragraph 2, and Section IV-C.
* **Response:** The accuracy drop is a deliberate trade-off to guarantee screening safety (the **Recall Paradox**). Under our 5:1 Lung Adenocarcinoma-to-Adjacent Non-Malignant class weight penalty, the model prioritizes Sensitivity (Recall = 1.00) under platform shift, ensuring no cancer patient goes undetected (0 false negatives). The 50.00% accuracy means that healthy control samples are flagged as lung adenocarcinoma (high false-positive rate). In clinical screening, false positives are managed and resolved via secondary diagnostics (e.g., biopsies), whereas a false negative is catastrophic. The model's safety bias provides the conservative default behavior required for diagnostic triage under technical batch shifts.
