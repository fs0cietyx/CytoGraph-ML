import os
import io
import gzip
import requests
import pandas as pd
import GEOparse
from src.utils.logging import logger
from src.utils.io import RAW_DATA_DIR, INTERIM_DATA_DIR

class GEODataLoader:
    """
    Loads and caches NCBI Gene Expression Omnibus (GEO) series matrices and metadata.
    """

    @staticmethod
    def _load_gse(gse_id: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Generic loader for a GSE dataset matrix and metadata."""
        logger.info(f"GEO: Loading dataset {gse_id}...")
        
        # Determine paths
        gse_dir = RAW_DATA_DIR / gse_id
        os.makedirs(gse_dir, exist_ok=True)
        
        # Cache file path for the series matrix
        matrix_cache_path = INTERIM_DATA_DIR / f"{gse_id}_series_matrix.csv"
        
        # 1. Load or Download Series Matrix
        if matrix_cache_path.exists():
            logger.info(f"GEO: Loading cached series matrix for {gse_id} from {matrix_cache_path}")
            df_matrix = pd.read_csv(matrix_cache_path, index_col=0)
            X = df_matrix.T
        else:
            logger.info(f"GEO: Cached matrix not found. Fetching {gse_id} series matrix via HTTPS...")
            gse_nnn = gse_id[:-3] + "nnn" if len(gse_id) > 5 else gse_id
            matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_nnn}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
            
            response = requests.get(matrix_url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download matrix from {matrix_url}")
            
            content = gzip.decompress(response.content).decode('utf-8')
            lines = [l for l in content.split('\n') if not l.startswith('!') and l.strip()]
            df_matrix = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', index_col=0)
            
            # Save to interim cache
            os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
            df_matrix.to_csv(matrix_cache_path)
            X = df_matrix.T

        # 2. Extract Labels and Groups using GEOparse (loads from raw directory local file if available)
        logger.info(f"GEO: Loading metadata for {gse_id}...")
        gse_metadata = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir, silent=True, how='brief')
        metadata = gse_metadata.phenotype_data
        
        # Identify label column automatically
        label_col = None
        for col in metadata.columns:
            if metadata[col].astype(str).str.contains('cancer|tumor|normal', case=False).any():
                label_col = col
                break
        
        if not label_col:
            label_col = 'source_name_ch1'
            
        # 1 = Tumor/Cancer, 0 = Normal
        y = metadata[label_col].astype(str).str.contains('cancer|tumor|adenocarcinoma', case=False).astype(int)
        
        # Extract patient group IDs for group-blind validation
        patient_ids = metadata.index
        for col in ['title', 'description', 'characteristics_ch1.0.patient']:
            if col in metadata.columns:
                extracted = metadata[col].astype(str).str.extract(r'(\d+|GT\d+|patient \d+|Subject \d+)', expand=False)
                if not extracted.isna().all():
                    patient_ids = extracted
                    break
        
        y = y.reindex(X.index)
        groups = patient_ids.reindex(X.index)
        
        logger.info(f"GEO: Loaded {X.shape[0]} samples and {X.shape[1]} probes for {gse_id}.")
        return X, y, groups

    @classmethod
    def load_gse10072(cls) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Loads dataset GSE10072."""
        return cls._load_gse("GSE10072")

    @classmethod
    def load_gse19804(cls) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Loads dataset GSE19804."""
        return cls._load_gse("GSE19804")

    @classmethod
    def load_gse21510(cls) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Loads dataset GSE21510."""
        return cls._load_gse("GSE21510")
