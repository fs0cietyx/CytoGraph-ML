import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils.logging import logger

class PermutationTester:
    """
    Rigorously tests for data leakage by shuffling label targets.
    If the model maintains high accuracy on permuted labels, it indicates information leakage.
    """
    def __init__(self, n_permutations: int = 1000, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state
        
    def run_audit(self, pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
        logger.info(f"PermutationTester: Running label-shuffling permutation audit ({self.n_permutations} runs)...")
        
        # Determine baseline accuracy on original labels
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        pipeline.fit(X_train_split, y_train_split)
        real_score = accuracy_score(y_val_split, pipeline.predict(X_val_split))
        logger.info(f"PermutationTester: Baseline validation accuracy: {real_score*100:.2f}%")
        
        shuffled_scores = []
        for i in range(self.n_permutations):
            # Shuffle labels
            y_shuffled = np.random.permutation(y.values)
            
            # Split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_shuffled, test_size=0.3, random_state=self.random_state + i
            )
            
            # Fit & Predict
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            shuffled_scores.append(accuracy_score(y_val, preds))
            
        mean_shuffled = np.mean(shuffled_scores)
        std_shuffled = np.std(shuffled_scores)
        
        # Calculate p-value: fraction of permutations where shuffled accuracy >= baseline accuracy
        p_value = np.sum(np.array(shuffled_scores) >= real_score) / self.n_permutations
        
        logger.info(f"PermutationTester: Permuted Mean Accuracy = {mean_shuffled*100:.2f}% (+/- {std_shuffled*100:.2f}%)")
        logger.info(f"PermutationTester: Calculated p-value = {p_value:.4f}")
        
        return {
            "baseline_accuracy": real_score,
            "permuted_mean_accuracy": mean_shuffled,
            "permuted_std_accuracy": std_shuffled,
            "p_value": p_value,
            "shuffled_scores": shuffled_scores
        }
