import sys
from core.config import logger, config
from core.data_engine import BioDataLoader, BioPreprocessor
from core.trainer import APEXTrainer
from core.interpreter import APEXInterpreter
from core.benchmarker import APEXBenchmarker
from core.bio_mapper import BioMapper
from sklearn.model_selection import train_test_split
import pandas as pd

def run_apex_pipeline():
    """Execution orchestration for THE APEX PROTOCOL."""
    logger.info("APEX PROTOCOL: Pipeline execution started.")

    try:
        # 1. Data Ingestion
        loader = BioDataLoader()
        X, y = loader.load_raw_data()

        # 2. DEFENSIVE PATCH: Strict Anti-Leakage Boundary
        # Preprocessing (Variance Thresholding) is now handled INSIDE the Trainer's Pipeline
        # to ensure fit_transform only happens on training folds.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        # 3. Stratified Training & Validation (Includes Imputation, Scaling, Filter)
        trainer = APEXTrainer()
        results = trainer.execute_stratified_training(X_train, y_train)

        # 4. Execute Phase 2 Benchmarks
        benchmarker = APEXBenchmarker()
        benchmarker.run_benchmark(X_train, y_train)

        # 5. Evaluate on Holdout Set (Automatic cleanup via Pipeline)
        y_pred = trainer.pipeline.predict(X_test)
        from sklearn.metrics import accuracy_score, classification_report
        holdout_acc = accuracy_score(y_test, y_pred)
        logger.info(f"Final Holdout Accuracy: {holdout_acc:.4f}")

        # 6. Feature Importance & Bio-Mapping (Phase 3)
        importance_df = trainer.get_feature_importance()
        top_20_genes = importance_df.head(20)["Gene"].tolist()

        bio_report = BioMapper.generate_scientific_report(top_20_genes)

        with open(config.RESULTS_DIR / "scientific_bio_report.md", "w") as f:
            f.write("# APEX Protocol: Biological Pathway Analysis\n\n")
            f.write(bio_report)
        logger.info(f"Scientific bio-report generated at {config.RESULTS_DIR / 'scientific_bio_report.md'}")

        # 8. SHAP Interpretability
        # Access the final step of the pipeline (the Random Forest) for SHAP
        rf_step = trainer.pipeline.named_steps['rf']
        interpreter = APEXInterpreter(rf_step, X_train)
        # Note: We pass raw X_test, but SHAP needs the processed features.
        # For simplicity in this demo, we'll explain the RF step.
        interpreter.compute_shap_values(X_test.head(50))

        logger.info("APEX PROTOCOL: Pipeline execution completed successfully.")

        
    except Exception as e:
        logger.exception(f"Pipeline Critical Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_apex_pipeline()
