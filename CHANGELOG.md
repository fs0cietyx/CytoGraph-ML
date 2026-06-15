# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-paper-submission] - 2026-06-16

### Added
- Created complete modular bioinformatics folder structure (`configs/`, `docs/`, `results/`, `figures/`, `models/`, `scripts/`, `src/`, `tests/`).
- Added scikit-learn `Pipeline` encapsulating `VarianceThreshold`, `SimpleImputer`, `RobustScaler`, and `MutualInformationSelector` for zero-leakage cross-validation.
- Implemented `test_no_leakage` unit test to verify fold-level boundary isolation.
- Implemented label-shuffling `PermutationTester` for statistical p-value significance bounds.
- Added TreeSHAP explainer module for cooperative game-theoretic feature attributions.
- Added `BioMapper` to aggregate individual SHAP attributions into pathway rollup scores (Reactome/KEGG).
- Added `reproduce_paper.py` script for single-command end-to-end replication.
- Added `CHANGELOG.md`, `CONTRIBUTING.md`, and `decision_log.md` documentation.
- Generated publication-ready figures for architecture flow, SHAP summaries, ROC validation curves, and expression drift.

### Changed
- Refactored layperson README statements to precise scientific descriptions.
- Renamed baseline classifier calibrations from "Hardened Model" to "Domain-Aligned Model" to align with domain adaptation terminology.
- Extracted REST API/FastAPI schemas and mock endpoints into supplemental files, keeping the main manuscript focused strictly on core scientific findings.

### Fixed
- Fixed an `UnboundLocalError` in `publication_figures.py` by relocating the `numpy` import to the top of the module scope.
- Fixed missing serialized models check by saving the standalone fitted `feature_selector.pkl` alongside `rf_lung.pkl`.
- Expanded acronyms (`GEO`, `SHAP`, `FDA`, `CLIA`, `CAP`) on their first occurrence in the LaTeX manuscript body.
