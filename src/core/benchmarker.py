import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .config import logger, config

class APEXBenchmarker:
    """Enterprise Benchmarking Suite: Neural Network vs. Random Forest."""
    
    def __init__(self):
        # Deep Learning benchmark using MLP (Multi-Layer Perceptron)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=config.RANDOM_STATE
        )

    def run_benchmark(self, X: pd.DataFrame, y: pd.Series):
        """
        Executes a comparative study between the production RF and a Neural Network.
        """
        logger.info("APEX BENCHMARK: Initiating Neural Network (MLP) validation...")
        
        skf = StratifiedKFold(n_splits=config.N_FOLD, shuffle=True, random_state=config.RANDOM_STATE)
        
        logger.info("Training MLP baseline (this may take a moment due to high dimensionality)...")
        scores = cross_val_score(self.mlp, X, y, cv=skf, scoring='accuracy', n_jobs=config.RF_N_JOBS)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        
        logger.info(f"MLP Benchmark Result: {mean_acc:.4f} (+/- {std_acc:.4f})")
        
        # Save results to markdown for documentation
        report = f"""# APEX Protocol: Model Benchmark Report
- **Production Model (Random Forest)**: 99.53% Accuracy (from Phase 1)
- **Benchmark Model (Deep MLP)**: {mean_acc*100:.2f}% Accuracy

## Summary
{"The Random Forest remains the superior choice for this genomic profile." if mean_acc < 0.9953 else "The Neural Architecture shows marginal gains but higher complexity."}
"""
        with open(config.RESULTS_DIR / "benchmark_report.md", "w") as f:
            f.write(report)
            
        return mean_acc
