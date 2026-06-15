from sklearn.pipeline import Pipeline
from src.preprocessing.blacklist import BlacklistFilter
from src.preprocessing.quantile_norm import QuantileNormalizer
from src.preprocessing.imputation import MedianImputer
from src.preprocessing.scaling import GenomicScaler
from src.preprocessing.feature_selection import MutualInformationSelector
from src.models.factory import ModelFactory

def create_lung_pipeline(config: dict) -> Pipeline:
    """
    Creates the scikit-learn pipeline for lung cancer classification.
    """
    feature_count = config.get("feature_count", 250)
    random_seed = config.get("random_seed", 42)
    classifier_config = config.get("classifier", {})
    
    # Inject seed into classifier configuration
    clf_params = dict(classifier_config)
    clf_params["random_seed"] = random_seed
    
    clf = ModelFactory.create(clf_params.get("type", "rf"), clf_params)
    
    return Pipeline([
        ('blacklist', BlacklistFilter()),
        ('quantile_norm', QuantileNormalizer()),
        ('imputer', MedianImputer()),
        ('scaler', GenomicScaler()),
        ('feature_selector', MutualInformationSelector(k=feature_count, random_state=random_seed)),
        ('classifier', clf)
    ])
