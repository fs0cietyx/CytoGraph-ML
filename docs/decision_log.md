# Architectural Decision Log

This document records the design choices, statistical justifications, and mathematical rationales for the core components of the CytoGraph-ML framework.

---

## 1. Classifier Selection: Random Forest
* **Status:** Accepted (Production Classifier)
* **Alternative Considered:** Multi-Layer Perceptron (MLP) Neural Networks, Gradient Boosting (XGBoost/LightGBM), Elastic Net.
* **Justification:**
  1. **High-Dimensional Stability ($P \gg N$):** With only 107 lung samples ($N$) and 13,515 features ($P$), deep learning architectures are highly susceptible to overfitting and parameter instability. Random Forest constructs bootstrap-aggregated (bagged) decision tree ensembles with random subspace projections, making it exceptionally resilient to high-dimensional multicollinearity.
  2. **Cooperative Game-Theoretic Attributions:** Random Forest models support TreeSHAP, which computes exact, local attributions in polynomial time without heuristic approximations. Neural networks and support vector machines require perturbation-based approximations (KernelSHAP or LIME) that are computationally expensive and statistically unstable on genomic scales.
  3. **Decision Boundary Shift Capability:** Through simple class-weight calibration ($5:1$), we can shift the posterior probability threshold for lung adenocarcinoma classification, securing a perfect sensitivity (Recall = 1.00) under platform domain shifts.

---

## 2. Feature Selector: Mutual Information (MI)
* **Status:** Accepted (Primary Feature Reduction)
* **Alternative Considered:** ANOVA F-test, LASSO (L1 regularization), Principal Component Analysis (PCA).
* **Justification:**
  1. **Non-parametric and Non-linear Support:** Transcriptomic expression violate Gaussian assumptions, displaying highly non-linear, multi-modal distributions. Parametric methods like ANOVA assume normal distributions and linear variances, making them poor indicators of transcriptomic relationships.
  2. **Information-Theoretic Coverage:** Mutual Information evaluates the total statistical dependency between variables:
     $$I(X; Y) = \sum_{y \in Y} \sum_{x \in X} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$
     capturing threshold differences, multi-way interactions, and non-linear boundaries.
  3. **Preservation of HGNC Symbols:** Unlike PCA, which projects expression values into uninterpretable orthogonal latent spaces, MI ranks and selects genuine HUGO Gene Nomenclature Committee (HGNC) symbols. This preserves direct mapping to downstream biological pathways (Reactome/KEGG).

---

## 3. Split Protocol: GroupKFold (Patient Grouping)
* **Status:** Accepted (Cross-Validation Splitting)
* **Alternative Considered:** Standard K-Fold Cross-Validation, Stratified K-Fold.
* **Justification:**
  1. **Patient-Level Leakage Vulnerability:** Clinical microarrays often contain multiple tissue biopsies (e.g., tumor core, adjacent normal, tumor margin) derived from the same patient ID. In standard K-Fold splits, adjacent normal and tumor samples from the same patient can span the train/test folds. The classifier would easily memorize patient-specific genetic structures (e.g., patient-specific mutation background, ethnic SNPs) rather than cancer-specific markers, inflating performance metrics.
  2. **Group Blindness Guarantee:** GroupKFold enforces that all samples belonging to a patient ID are allocated strictly to a single fold, ensuring that validation folds contain only completely unseen patient genotypes.

---

## 4. Interpretability Framework: TreeSHAP
* **Status:** Accepted (Explainability Layer)
* **Alternative Considered:** LIME, Saliency Maps, Random Forest Impurity (Gini Importance).
* **Justification:**
  1. **Axiomatic Properties:** SHAP (Shapley Additive exPlanations) is the only local feature attribution framework derived from cooperative game theory that satisfies the mathematical axioms of:
     * **Local Accuracy (Efficiency):** Attributions sum to the difference between prediction and expected base value.
     * **Consistency:** If a feature's marginal contribution increases or stays the same, its attribution cannot decrease.
     * **Missingness:** Features with zero contribution receive zero attribution.
  2. **Exact Attributions:** By exploiting the tree ensemble structure, TreeSHAP computes attributions recursively through node partitions in $O(TLD^2)$ time (where $T$ is trees, $L$ is max leaves, $D$ is max depth) without resorting to sampling approximations.
