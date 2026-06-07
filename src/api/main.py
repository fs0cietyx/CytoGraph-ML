from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
from core.config import config, logger

app = FastAPI(
    title="APEX Cancer Growth Prediction API",
    description="Enterprise-grade inference service for genomic-based cancer classification.",
    version="2.0.0"
)

# Load the production pipeline and base feature list once at startup
try:
    PIPELINE_PATH = config.MODEL_DIR / "final_pipeline.joblib"
    BASE_FEATURE_LIST_PATH = config.MODEL_DIR / "base_features.joblib"
    
    pipeline = joblib.load(PIPELINE_PATH)
    base_features = joblib.load(BASE_FEATURE_LIST_PATH)
    
    logger.info(f"API: Production pipeline loaded from {PIPELINE_PATH}")
    logger.info(f"API: Validating against {len(base_features)} base genomic features.")
except Exception as e:
    logger.error(f"API: Failed to load production artifacts: {e}")
    raise

class InferenceRequest(BaseModel):
    """Schema for a vectorized batch inference request."""
    # List of samples, where each sample is a Dict[gene_id, value]
    samples: List[Dict[str, float]] 

class SinglePrediction(BaseModel):
    prediction: str
    confidence: float

class PredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    results: List[SinglePrediction]
    status: str = "success"

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "pipeline_loaded": pipeline is not None,
        "base_feature_count": len(base_features)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: InferenceRequest):
    """
    Vectorized Inference Engine.
    1. Validates schema against full TCGA-PANCAN profile.
    2. Implements mathematical imputation for missing genes via Pipeline.
    3. Executes vectorized batch prediction for high-throughput hospitals.
    """
    try:
        # 1. Efficiently convert batch JSON to Pandas DataFrame
        df_input = pd.DataFrame(request.samples)
        
        # 2. Re-index to ensure correct gene order and handle missing columns
        # Genes missing from input will be filled with NaN, then handled by the Pipeline's Imputer.
        df_aligned = df_input.reindex(columns=base_features)
        
        # 3. Vectorized Prediction
        predictions = pipeline.predict(df_aligned)
        probabilities = pipeline.predict_proba(df_aligned)
        
        # 4. Format results
        results = []
        for i in range(len(predictions)):
            results.append(SinglePrediction(
                prediction=predictions[i],
                confidence=float(np.max(probabilities[i]))
            ))
            
        return PredictionResponse(results=results)
        
    except Exception as e:
        logger.error(f"API: Inference failure: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during batch inference.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
