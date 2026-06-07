# Project Progress

## 2026-06-06
- **Initial Setup**: Created dedicated project folder in the Obsidian vault.
- **Documentation**: Initialized `README.md`, `STATUS.md`, `progress.md`, and `decisions.md` with the project's background and goals.
- **Dataset Selected**: Chose **UCI Gene Expression Cancer RNA-Seq (PANCAN)** for its high portfolio impact and alignment with TCGA standards.
- **Environment Setup**: Created a virtual environment and installed `pandas`, `scikit-learn`, and `ucimlrepo`.
- **Data Acquisition**: Manually downloaded and extracted the TCGA PANCAN dataset (801 samples, 20,531 genes).
- **APEX Phase 1**: Refactored into Modular OOP. Implemented Stratified 5-Fold CV (99.53% Acc) and SHAP Interpretability.
- **APEX Phase 2**: Deployed FastAPI REST service. Conducted Deep Learning (MLP) benchmarking (99.38% Acc).
- **APEX Phase 3**: Implemented Dockerization (`Dockerfile`, `docker-compose.yml`). Added Pytest suite and Bio-Pathway mapping report.
