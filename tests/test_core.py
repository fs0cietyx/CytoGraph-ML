import pytest
from src.utils.io import load_config, CONFIG_DIR
from src.utils.bio_mapper import BioMapper

def test_config_paths():
    """Validates that configs are loaded correctly and directories exist."""
    assert CONFIG_DIR.exists()
    lung_config = load_config("lung")
    assert lung_config["feature_count"] == 250
    assert lung_config["random_seed"] == 42

def test_bio_mapper_lookup():
    """Validates biological context enrichment via MyGene.info API."""
    genes = ["TP53", "UNKNOWN_GHOST_GENE"]
    context = BioMapper.get_biological_context(genes)
    
    # Check that TP53 is correctly mapped to its official symbol
    assert context["TP53"]["Symbol"].upper() == "TP53"
    
    # Check the fallback mechanism for unknown genes
    assert "UNKNOWN_GHOST_GENE" in context["UNKNOWN_GHOST_GENE"]["Symbol"]
