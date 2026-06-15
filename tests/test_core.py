import pytest
import pandas as pd
from src.core.config import config
from src.core.bio_mapper import BioMapper

def test_config_paths():
    """Validates that core paths are correctly initialized."""
    assert config.DATA_DIR.exists()
    assert config.FEATURES_PATH.name == "data.csv"

def test_bio_mapper_lookup():
    """Validates biological context enrichment via MyGene.info API."""
    # Use real biological markers
    genes = ["TP53", "UNKNOWN_GHOST_GENE"]
    context = BioMapper.get_biological_context(genes)
    
    # Check that TP53 is correctly mapped to its official symbol
    assert context["TP53"]["Symbol"].upper() == "TP53"
    assert "tumor protein" in context["TP53"]["Name"].lower() or "p53" in context["TP53"]["Name"].lower()
    
    # Check the fallback mechanism for unknown genes
    assert context["UNKNOWN_GHOST_GENE"]["Symbol"] == "UNKNOWN_GHOST_GENE"
