import pytest
import pandas as pd
from src.core.config import config
from src.core.data_engine import BioDataLoader, BioPreprocessor
from src.core.bio_mapper import BioMapper

def test_config_paths():
    """Validates that core paths are correctly initialized."""
    assert config.DATA_DIR.exists()
    assert config.FEATURES_PATH.name == "data.csv"

def test_data_loader():
    """Integration test for data ingestion."""
    loader = BioDataLoader()
    X, y = loader.load_raw_data()
    assert not X.empty
    assert len(X) == len(y)

def test_preprocessor_reduction():
    """Validates that feature selection actually reduces dimensions."""
    X = pd.DataFrame({
        "gene_1": [1.0, 5.0, 1.0], # High variance
        "gene_2": [1.0, 1.0, 1.0]  # Zero variance
    })
    preprocessor = BioPreprocessor()
    X_clean = preprocessor.clean_and_subset(X)
    assert "gene_2" not in X_clean.columns
    assert X_clean.shape[1] == 1

def test_bio_mapper_lookup():
    """Validates biological context enrichment."""
    genes = ["gene_14092", "unknown_gene"]
    context = BioMapper.get_biological_context(genes)
    assert context["gene_14092"]["Symbol"] == "TF-Alpha"
    assert context["unknown_gene"]["Pathway"] == "Metabolic Homeostasis"
