# Lessons Learned for Genomic Machine Learning

From the development and validation of the CytoGraph-ML framework, we synthesize five fundamental insights for applying machine learning to high-dimensional transcriptomic datasets:

---

## 1. Leakage Prevention is Necessary
Standard cross-validation and preprocessing protocols in bioinformatics frequently leak data. Performing variance filtering, imputation, or scaling globally prior to train-validation splitting leaks class boundaries, resulting in artificial performance inflation ($\sim 100\%$). Encapsulating all steps in strict fit-transform pipelines (e.g., scikit-learn Pipelines) is a non-negotiable requirement for scientific validity.

---

## 2. Leakage Prevention is Insufficient
A model trained with zero data leakage can still fail catastrophically when evaluated on independent cohorts. Standard cross-validation folds share the same center-specific technical batch effects. Ensuring a zero-leakage training boundary provides an honest baseline but does not insulate the pipeline from platform drift.

---

## 3. Domain Shift Dominates Deployment Risk
Systematic differences in sequencing platforms (e.g., Affymetrix GPL96 vs. GPL570 microarrays) or sample preparation methods introduce severe technical shifts (batch effects) that alter absolute expression ranges. Normalization techniques (like Quantile Normalization) help align global distributions, but real-world clinical translation requires reference-based alignment (R-QN) or explicit domain adaptation.

---

## 4. Explainability Does Not Guarantee Transportability
Post-hoc explainability (such as TreeSHAP) is highly valuable for confirming that model decisions align with known cancer hallmarks (e.g., *LDB2* suppressor loss, *SLIT3* angiogenesis role). However, even if the model locks onto genuine biology, a technical batch shift that compresses the scale of these biological drivers on a foreign platform will still cause prediction failures. Explainability verifies biological sanity but does not solve feature drift.

---

## 5. External Validation Must Be Standard Practice
Bioinformatics publications must shift from reporting marginal performance improvements on single cohorts (e.g., in-study TCGA subsets) to validating models on independent, cross-laboratory external cohorts. Subjecting models to domain shifts is the only way to evaluate clinical generalizability and ensure diagnostic transparency.
