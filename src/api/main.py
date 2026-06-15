from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from core.config import config, logger

app = FastAPI(
    title="Genomic Growth Prediction API",
    description="Inference service for genomic-based cancer classification using TCGA-anchored biomarkers.",
    version="2.1.0"
)

# Load the production pipeline once at startup
try:
    PIPELINE_PATH = config.MODEL_DIR / "final_pipeline.joblib"
    FEATURE_LIST_PATH = config.MODEL_DIR / "research_features.joblib"
    
    pipeline = joblib.load(PIPELINE_PATH)
    research_features = joblib.load(FEATURE_LIST_PATH)
    
    logger.info(f"API: Research pipeline loaded from {PIPELINE_PATH}")
    logger.info(f"API: Validating against {len(research_features)} genomic symbols.")
except Exception as e:
    logger.error(f"API: Failed to load research artifacts: {e}")
    raise

class InferenceRequest(BaseModel):
    """Schema for a vectorized batch genomic inference request."""
    # List of samples, where each sample is a Dict[gene_symbol, value]
    samples: List[Dict[str, float]] 

from core.bio_mapper import BioMapper

class SinglePrediction(BaseModel):
    prediction: str
    confidence: float
    biological_insight: Dict[str, Any] = {}

class PredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    results: List[SinglePrediction]
    status: str = "success"

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "pipeline_loaded": pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: InferenceRequest):
    """
    Vectorized Inference Engine with Biological Enrichment.
    """
    try:
        df_input = pd.DataFrame(request.samples)
        
        # Align features to the mapped symbols used during training
        df_aligned = df_input.reindex(columns=research_features)
        
        predictions = pipeline.predict(df_aligned)
        probabilities = pipeline.predict_proba(df_aligned)
        
        # Identify top genes for context
        top_genes = research_features[:5] 
        bio_context = BioMapper.get_biological_context(top_genes)
        
        results = []
        for i in range(len(predictions)):
            results.append(SinglePrediction(
                prediction=predictions[i],
                confidence=float(np.max(probabilities[i])),
                biological_insight=bio_context
            ))
            
        return PredictionResponse(results=results)
        
    except Exception as e:
        logger.error(f"API: Inference failure: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during batch inference.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
