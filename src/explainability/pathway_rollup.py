import pandas as pd
from src.utils.logging import logger

class PathwayRollup:
    """
    Aggregates gene-level SHAP values to biological pathway-level contribution scores.
    """
    def __init__(self, gene_metadata_provider=None):
        self.provider = gene_metadata_provider
        
    def rollup_shap_values(self, gene_importance_df: pd.DataFrame) -> pd.DataFrame:
        """Rolls up gene-level SHAP values to their corresponding pathways."""
        logger.info("PathwayRollup: Rolling up gene SHAP importances to pathway levels...")
        genes = gene_importance_df["Gene"].tolist()
        
        if self.provider is not None:
            try:
                metadata = self.provider.get_biological_context(genes)
            except Exception as e:
                logger.error(f"PathwayRollup: Metadata provider failed: {e}. Falling back to offline mapping.")
                metadata = self._get_offline_pathway_mapping(genes)
        else:
            metadata = self._get_offline_pathway_mapping(genes)
            
        gene_to_pathway = {g: info.get("Pathway", "Metabolic Homeostasis") for g, info in metadata.items()}
        
        df = gene_importance_df.copy()
        df["Pathway"] = df["Gene"].map(gene_to_pathway)
        
        pathway_df = df.groupby("Pathway")["Mean_Abs_SHAP"].sum().reset_index()
        pathway_df = pathway_df.rename(columns={"Mean_Abs_SHAP": "Pathway_SHAP_Contribution"})
        pathway_df = pathway_df.sort_values(by="Pathway_SHAP_Contribution", ascending=False)
        
        return pathway_df
        
    def _get_offline_pathway_mapping(self, genes: list[str]) -> dict:
        fallback_pathways = {
            "TCF21": "Tumor Suppressor Signaling",
            "CRYAB": "Heat Shock Response / Protein Folding",
            "SLIT3": "Axon Guidance & Angiogenesis",
            "LDB2": "Cell Junction Adhesion",
            "EPAS1": "Hypoxia Response Pathway",
            "EDNRB": "G-Protein Coupled Receptor Signaling",
            "KIAA1462": "Cell Junction Adhesion",
        }
        
        metadata = {}
        for gene in genes:
            metadata[gene] = {
                "Pathway": fallback_pathways.get(gene.upper(), "Metabolic Homeostasis")
            }
        return metadata
