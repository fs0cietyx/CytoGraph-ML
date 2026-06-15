# Contributing to CytoGraph-ML

Thank you for your interest in contributing to CytoGraph-ML! This document outlines the guidelines for proposing modifications, report issues, and improving the framework code.

---

## Code of Conduct

By participating in this project, you agree to maintain a professional, constructive, and respectful environment. Please treat all contributors and researchers with courtesy.

---

## How to Contribute

### 1. Reporting Bugs & Issues
Before opening a new issue, please search the active issues to see if the bug has already been reported. When reporting an issue, include:
* Your OS version and Python environment specifications.
* A minimal, self-contained reproducible code example.
* The complete traceback log of the error.

### 2. Submitting Pull Requests (PRs)
To propose changes:
1. **Fork the Repository** and create a feature branch named after your contribution (e.g., `feature/custom-normalizer`).
2. **Implement Tests**: Any new preprocessing step, model architecture, or validation metric must have associated unit tests in the `tests/` directory.
3. **Run Linting & Formatter**: Ensure your code conforms to standard PEP 8 styling conventions.
4. **Run Pytest**: Verify that all unit and integration tests pass successfully:
   ```bash
   pytest
   ```
5. **Verify Zero Leakage**: If you modify the pipeline or validation steps, you must run `pytest tests/test_pipeline.py` to ensure that `test_no_leakage` passes.
6. **Submit the PR**: Provide a detailed description of the changes, their rationale, and how they affect the expected CV or external accuracies.

---

## Coding Standards & Integrity
* **Modularity**: Every transformer step must do **one thing** and be isolated under `src/preprocessing/` or `src/models/`.
* **Zero Leakage**: Never fit any preprocessing steps (e.g., imputation, scaling, feature selection) globally on the dataset. All transformations must be encapsulated inside a scikit-learn `Pipeline` or custom fit-transform boundary.
* **Seeds**: Always use the global random state `seed = 42` for reproducibility.
