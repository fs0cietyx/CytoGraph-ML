import os
import sys
import pandas as pd

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, RESULTS_DIR
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_lung_pipeline
from src.validation import PermutationTester

def main():
    set_seed(42)
    logger.info("================== RUNNING PERMUTATION AUDIT ==================")
    
    # Load config
    config = load_config("lung")
    
    # Ingest data
    X_raw, y, _ = GEODataLoader.load_gse10072()
    
    # Map probes
    mapper = GeneMapper(platform_id="GPL96")
    X = mapper.fit_transform(X_raw)
    
    # Build pipeline
    pipeline = create_lung_pipeline(config)
    
    # Permutation Tester
    n_perms = 100  # We run 100 permutations for fast audit execution
    tester = PermutationTester(n_permutations=n_perms, random_state=42)
    audit_res = tester.run_audit(pipeline, X, y)
    
    # Save results
    os.makedirs(RESULTS_DIR / "cv", exist_ok=True)
    audit_df = pd.DataFrame([{
        "Metric": "Permutation Audit",
        "Baseline Accuracy": f"{audit_res['baseline_accuracy']*100:.2f}%",
        "Permuted Mean Accuracy": f"{audit_res['permuted_mean_accuracy']*100:.2f}%",
        "Permuted Std Accuracy": f"{audit_res['permuted_std_accuracy']*100:.2f}%",
        "Empirical p-value": f"{audit_res['p_value']:.4f}"
    }])
    
    audit_df.to_csv(RESULTS_DIR / "cv" / "permutation_audit.csv", index=False)
    
    logger.info("Permutation audit complete.")
    print(f"Permutation Audit Results (n={n_perms}):")
    print(audit_df.to_string(index=False))
    
    if audit_res['p_value'] < 0.05:
        print("AUDIT PASS: Empirical p-value < 0.05 (No significant leakage detected).")
    else:
        print("AUDIT WARNING: Empirical p-value >= 0.05 (Verify leakage boundaries).")

if __name__ == "__main__":
    main()
