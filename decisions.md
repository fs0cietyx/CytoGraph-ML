# Project Decisions

## 2026-06-06
- **Model Selection: Random Forest**: We decided to use a Random Forest model for the initial rebuild.
  - **Why**: It performed well in the original project, handles structured data effectively, and provides clear feature importance scores for identifying key transcription factors.
  - **Revisit**: Once the baseline model is stable, evaluate if more complex models (e.g., Gradient Boosting) offer significant improvements.

- **Dataset Selection: UCI Gene Expression Cancer RNA-Seq (PANCAN)**:
  - **Why**: This is the best choice for a high-impact portfolio. It uses data from **The Cancer Genome Atlas (TCGA)**, contains over 20,000 features (genes), and provides a robust sample size (801). It allows for a powerful "finding the signal in the noise" narrative.
  - **Trade-off**: While technically gene expression (RNA-Seq), it serves as a high-quality proxy for protein activity and is much cleaner/larger than existing direct protein expression datasets.
