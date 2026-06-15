import pandas as pd
import numpy as np
import requests
import io
import gzip
from core.config import logger

class GDCDataFetcher:
    """
    Genuine GDC Data Ingestion Engine.
    Fetches harmonized RNA-seq data from the Genomic Data Commons API.
    """
    
    BASE_URL = "https://api.gdc.cancer.gov/data/"
    
    # Example UUID for a TCGA-LUAD (Lung Adenocarcinoma) gene expression quantification file
    # In a full-scale app, this would iterate through a manifest of multiple samples.
    SAMPLE_UUID = "80f14652-32a5-4f40-9a3d-495e268f7423" 

    @classmethod
    def download_real_tcga_sample(cls):
        """
        Downloads a real GDC gene expression file (STAR-Counts) with genuine Ensembl IDs.
        """
        logger.info(f"GDC: Initiating download for sample {cls.SAMPLE_UUID}...")
        
        response = requests.get(f"{cls.BASE_URL}{cls.SAMPLE_UUID}")
        if response.status_code == 200:
            # GDC files are typically TSV or compressed
            try:
                # Decoding the raw response (simulating the handling of GDC .tsv.gz)
                data = pd.read_csv(io.BytesIO(response.content), sep='\t', compression='gzip', comment='#')
                logger.info("GDC: Successfully downloaded and decompressed real genomic data.")
                return data
            except Exception as e:
                logger.error(f"GDC: Data decompression failure: {e}")
                return None
        else:
            logger.error(f"GDC: API connection failed with status {response.status_code}")
            return None

class GenomicNormalizer:
    """
    Bio-Statistical Normalization Layer.
    Implements TPM (Transcripts Per Million) to normalize for gene length and library depth.
    """
    
    @staticmethod
    def calculate_tpm(counts_df: pd.DataFrame, lengths_kb: pd.Series) -> pd.DataFrame:
        """
        Standardizes raw counts to TPM.
        1. Divide counts by gene length (RPK).
        2. Divide RPK by total RPK in sample / 1 million.
        """
        logger.info("Normalizing raw counts to TPM (Transcripts Per Million)...")
        # RPK: Reads Per Kilobase
        rpk = counts_df.divide(lengths_kb, axis=0)
        # Scale to 1 million
        tpm = rpk.divide(rpk.sum(axis=0), axis=1) * 1e6
        return tpm
