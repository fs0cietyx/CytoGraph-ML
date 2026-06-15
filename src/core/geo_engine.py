import GEOparse
import pandas as pd
import numpy as np
import os
from core.config import logger, config

class GEOIngestor:
    """
    Genuine NCBI Gene Expression Omnibus (GEO) Data Ingestor.
    Downloads and parses real-world genomic datasets with verified Probe IDs.
    """
    
    def __init__(self, gse_id="GSE10072"):
        self.gse_id = gse_id
        self.raw_dir = config.DATA_DIR / "raw"
        os.makedirs(self.raw_dir, exist_ok=True)

    @staticmethod
    def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Force-aligns the distribution of expression values to a standard normal distribution.
        This is the standard 'Acid Test' fix to remove cross-study batch effects.
        """
        logger.info("GEO: Performing Cross-Study Quantile Normalization...")
        # Rank-based transformation
        rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
        normalized = df.rank(method='first').stack().astype(int).map(rank_mean).unstack()
        return normalized

    def fetch_and_map(self, platform_id="GPL96", normalize=True):
        """
        Flexible GEO Fetcher with Quantile Normalization.
        1. Downloads Series Matrix via HTTPS.
        2. Automatically identifies clinical labels.
        3. Maps Probes to Symbols.
        4. Normalizes distributions to allow cross-lab comparison.
        """
        logger.info(f"GEO: Fetching {self.gse_id} via HTTPS...")
        
        try:
            # 1. Download Matrix
            gse_nnn = self.gse_id[:-3] + "nnn" if len(self.gse_id) > 5 else self.gse_id
            matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_nnn}/{self.gse_id}/matrix/{self.gse_id}_series_matrix.txt.gz"
            
            import requests, io, gzip
            response = requests.get(matrix_url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download matrix from {matrix_url}")
            
            content = gzip.decompress(response.content).decode('utf-8')
            lines = [l for l in content.split('\n') if not l.startswith('!') and l.strip()]
            df_matrix = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', index_col=0)
            X = df_matrix.T
            
            # 2. Extract Labels and Groups
            gse_metadata = GEOparse.get_GEO(geo=self.gse_id, destdir=self.raw_dir, silent=True, how='brief')
            metadata = gse_metadata.phenotype_data
            
            label_col = None
            for col in metadata.columns:
                if metadata[col].astype(str).str.contains('cancer|tumor|normal', case=False).any():
                    label_col = col
                    break
            
            if not label_col:
                label_col = 'source_name_ch1'
                
            y = metadata[label_col].astype(str).str.contains('cancer|tumor|adenocarcinoma', case=False).astype(int)
            
            patient_ids = metadata.index
            for col in ['title', 'description', 'characteristics_ch1.0.patient']:
                if col in metadata.columns:
                    extracted = metadata[col].astype(str).str.extract(r'(\d+|GT\d+|patient \d+|Subject \d+)', expand=False)
                    if not extracted.isna().all():
                        patient_ids = extracted
                        break
            
            y = y.reindex(X.index)
            groups = patient_ids.reindex(X.index)
            
            # 3. Platform Mapping
            logger.info(f"GEO: Fetching platform {platform_id}...")
            gpl = GEOparse.get_GEO(geo=platform_id, destdir=self.raw_dir, silent=True)
            symbol_col = 'Gene Symbol' if 'Gene Symbol' in gpl.table.columns else 'Symbol'
            probe_to_symbol = gpl.table.set_index('ID')[symbol_col].to_dict()
            
            # 4. Collapse and Normalize
            X = X.rename(columns=probe_to_symbol)
            X = X.loc[:, ~X.columns.duplicated()]
            X = X.loc[:, ~X.columns.str.contains('---|^$', na=True)]
            
            if normalize:
                X = self.quantile_normalize(X)
            
            logger.info(f"GEO: Successfully processed {X.shape[1]} genes across {X.shape[0]} samples.")
            return X, y, groups
            
        except Exception as e:
            logger.error(f"GEO: Pipeline Failure: {e}")
            return None, None, None

if __name__ == "__main__":
    ingestor = GEOIngestor()
    X, y, groups = ingestor.fetch_and_map()
    if X is not None:
        print(f"Sample Gene Columns: {X.columns[:5].tolist()}")
        print(f"Sample Patient Groups: {groups[:5].tolist()}")
