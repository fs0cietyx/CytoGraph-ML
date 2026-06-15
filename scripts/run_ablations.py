import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, RESULTS_DIR
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_lung_pipeline

def main():
    set_seed(42)
    logger.info("================== RUNNING ABLATION STUDIES ==================")
    
    # Ingest datasets
    X_train_raw, y_train, _ = GEODataLoader.load_gse10072()
    X_test_raw, y_test, _ = GEODataLoader.load_gse19804()
    
    # Map probes
    mapper_gpl96 = GeneMapper(platform_id="GPL96")
    mapper_gpl570 = GeneMapper(platform_id="GPL570")
    
    X_train = mapper_gpl96.fit_transform(X_train_raw)
    X_test = mapper_gpl570.fit_transform(X_test_raw)
    
    common_genes = list(set(X_train.columns).intersection(set(X_test.columns)))
    X_train_aligned = X_train.reindex(columns=common_genes).fillna(0)
    X_test_aligned = X_test.reindex(columns=common_genes).fillna(0)
    
    # 1. Feature Count Ablation (K = 50, 100, 150, 250, 500)
    logger.info("Ablation: Feature count K sweep...")
    config = load_config("lung")
    k_results = []
    
    for k in [50, 100, 150, 250, 500]:
        config["feature_count"] = k
        pipeline = create_lung_pipeline(config)
        pipeline.fit(X_train_aligned, y_train)
        preds = pipeline.predict(X_test_aligned)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        k_results.append({"Setting": f"K={k}", "Accuracy": f"{acc*100:.2f}%", "Recall": f"{rec:.4f}"})
        
    # 2. Class Weight Ablation (1:1, 2:1, 3:1, 5:1)
    logger.info("Ablation: Class weight ratio sweep...")
    config = load_config("lung")
    weight_results = []
    
    for w in [1, 2, 3, 5]:
        config["classifier"]["class_weight"] = {0: 1, 1: w}
        pipeline = create_lung_pipeline(config)
        pipeline.fit(X_train_aligned, y_train)
        preds = pipeline.predict(X_test_aligned)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        weight_results.append({"Setting": f"Weight {w}:1", "Accuracy": f"{acc*100:.2f}%", "Recall": f"{rec:.4f}"})
        
    # 3. Biological Blacklist Ablation
    logger.info("Ablation: Blacklist activation toggle...")
    config = load_config("lung")
    blacklist_results = []
    
    for name, active in [("Blacklist On", True), ("Blacklist Off", False)]:
        pipeline = create_lung_pipeline(config)
        if not active:
            # Disable blacklist step by removing it from the pipeline steps
            pipeline.steps.pop(0)
            
        pipeline.fit(X_train_aligned, y_train)
        preds = pipeline.predict(X_test_aligned)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        blacklist_results.append({"Setting": name, "Accuracy": f"{acc*100:.2f}%", "Recall": f"{rec:.4f}"})
        
    # Combine and save results
    os.makedirs(RESULTS_DIR / "ablations", exist_ok=True)
    all_ablations = k_results + weight_results + blacklist_results
    ablation_df = pd.DataFrame(all_ablations)
    ablation_df.to_csv(RESULTS_DIR / "ablations" / "ablation_results.csv", index=False)
    
    logger.info("Ablation studies complete.")
    print("Ablation Experiments Table:")
    print(ablation_df.to_string(index=False))

if __name__ == "__main__":
    main()
