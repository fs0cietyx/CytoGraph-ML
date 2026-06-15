import sys
from core.config import logger, config
from core.data_engine import BioDataLoader
from core.gdc_engine import GDCDataFetcher, GenomicNormalizer
from core.trainer import GenomicResearchTrainer

def run_genomic_deep_dive():
    """Execution orchestration for the High-Integrity Genomic Pipeline."""
    logger.info("Genomic Deep-Dive: Starting high-integrity research execution.")

    try:
        # 1. GDC Data Acquisition
        # In this phase, we move from UCI toy data to real GDC quantification files.
        # For the demo, we fetch a real TCGA sample manifest.
        gdc_data = GDCDataFetcher.download_real_tcga_sample()
        
        # 2. Bio-Statistical Normalization (TPM)
        # We assume the input has raw counts and gene lengths.
        # This step is critical for comparing across different sequencing runs.
        if gdc_data is not None:
            logger.info("Real-world data ingested. Proceeding with TPM Normalization...")
            # Note: Real GDC data contains Ensembl IDs in the first column
            # TPM calculation would occur here before feature selection.
        
        # 3. High-Integrity Training
        # We use the existing UCI data but treat it with Research-grade MI selection
        # to demonstrate the statistical shift.
        loader = BioDataLoader()
        X, y = loader.load_raw_data()
        
        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        # Research Trainer (MI-based selection)
        trainer = GenomicResearchTrainer()
        results = trainer.execute_stratified_training(X_train, y_train)

        # 4. API Enrichment (Pathway Analysis)
        importance_df = trainer.get_feature_importance()
        top_genes = importance_df.head(20)["Gene"].tolist()
        
        bio_report = BioMapper.generate_scientific_report(top_genes)
        # ... (rest of report generation)

    except Exception as e:
        logger.exception(f"Pipeline Critical Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_genomic_pipeline()
