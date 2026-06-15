# Paper Reproducibility Guide

CytoGraph-ML is designed to make paper reproduction simple, automated, and mathematically verifiable.

## One-Click Reproduction

To run all data loaders, fit pipelines, execute audits, run sweeps, and output final metrics:

```bash
python scripts/reproduce_paper.py
```

## Generated Outputs

Executing `reproduce_paper.py` generates the following files:

### Data Tables (`results/`)
- `cv/lung_cv_scores.csv`: Cross-validation accuracy scores.
- `holdout/lung_holdout_scores.csv`: Holdout validation scores.
- `external/external_validation_scores.csv`: Cross-study external accuracy metrics.
- `shap/gene_importance.csv`: Top genomic biomarkers ranked by SHAP value.
- `pathway/pathway_shap_rollup.csv`: Pathway contribution scores.
- `ablations/ablation_results.csv`: Results of parameter sweeps.

### Figures (`figures/`)
- `figure1_pipeline.png`: Technical system architecture diagram.
- `figure2_shap.png`: Biomarker importance bar plot.
- `figure3_roc.png`: ROC curve showing external cohort classification performance.
- `figure4_drift.png`: Distribution shift density plot for top biomarkers.
