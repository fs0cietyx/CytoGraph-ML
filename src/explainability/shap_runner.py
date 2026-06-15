import shap
import pandas as pd
import numpy as np
from src.utils.logging import logger

class SHAPRunner:
    """
    Computes SHAP (SHapley Additive exPlanations) values for the ensemble classifier in the pipeline.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.explainer = None
        self.feature_names = []
        
    def fit_explainer(self, X: pd.DataFrame):
        """Prepares the SHAP TreeExplainer by running data through the preprocessing steps."""
        logger.info("SHAPRunner: Transforming data and fitting TreeExplainer...")
        
        # Extract the preprocessor steps (all steps except the final estimator)
        if len(self.pipeline.steps) > 1:
            preprocessor = self.pipeline[:-1]
            X_trans = preprocessor.transform(X)
        else:
            X_trans = X
            
        self.feature_names = X_trans.columns.tolist()
        
        # Fit TreeExplainer on the classifier
        clf = self.pipeline.named_steps['classifier'] if 'classifier' in self.pipeline.named_steps else self.pipeline.steps[-1][1]
        self.explainer = shap.TreeExplainer(clf)
        return self
        
    def compute_shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Computes SHAP values on transformed features."""
        if self.explainer is None:
            self.fit_explainer(X)
            
        if len(self.pipeline.steps) > 1:
            preprocessor = self.pipeline[:-1]
            X_trans = preprocessor.transform(X)
        else:
            X_trans = X
        
        shap_values = self.explainer.shap_values(X_trans)
        
        # For Random Forest in scikit-learn, shap_values returns a list of shape [samples, features] for each class
        if isinstance(shap_values, list):
            # Take SHAP values for class 1 (Tumor/Cancer)
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            # Multi-class or binary layout in some SHAP versions
            shap_values = shap_values[:, :, 1]
            
        return shap_values, X_trans
