from typing import Any, Dict
from src.models.random_forest import get_random_forest
from src.models.logistic_regression import get_logistic_regression
from src.models.svm import get_svm
from src.models.xgboost import get_xgboost
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

class ModelFactory:
    """
    Factory for creating and initializing machine learning classifiers.
    """
    @staticmethod
    def create(model_name: str, params: Dict[str, Any] = None) -> Any:
        if params is None:
            params = {}
            
        # Extract common settings
        random_seed = params.get("random_seed", 42)
        n_estimators = params.get("n_estimators", 100)
        n_jobs = params.get("n_jobs", -1)
        class_weight = params.get("class_weight", {0: 1, 1: 5})
        
        name_lower = model_name.lower().strip()
        
        if name_lower in ["rf", "randomforest", "random_forest"]:
            return get_random_forest(
                n_estimators=n_estimators,
                class_weight=class_weight,
                random_state=random_seed,
                n_jobs=n_jobs
            )
        elif name_lower in ["lr", "logisticregression", "logistic_regression"]:
            return get_logistic_regression(
                penalty='l2',
                C=params.get("C", 1.0),
                random_state=random_seed,
                max_iter=params.get("max_iter", 1000)
            )
        elif name_lower in ["elasticnet", "elastic_net"]:
            return get_logistic_regression(
                penalty='elasticnet',
                C=params.get("C", 1.0),
                random_state=random_seed,
                max_iter=params.get("max_iter", 1000)
            )
        elif name_lower in ["svm", "svm_linear", "svm (linear)"]:
            return get_svm(
                kernel='linear',
                C=params.get("C", 1.0),
                random_state=random_seed
            )
        elif name_lower in ["svm_rbf", "svm (rbf)"]:
            return get_svm(
                kernel='rbf',
                C=params.get("C", 1.0),
                random_state=random_seed
            )
        elif name_lower in ["xgb", "xgboost"]:
            return get_xgboost(
                n_estimators=n_estimators,
                random_state=random_seed,
                n_jobs=n_jobs
            )
        elif name_lower in ["lgb", "lightgbm"]:
            return LGBMClassifier(
                n_estimators=n_estimators,
                random_state=random_seed,
                n_jobs=n_jobs,
                verbose=-1
            )
        elif name_lower in ["extra_trees", "extratrees", "et"]:
            return ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight=class_weight,
                random_state=random_seed,
                n_jobs=n_jobs
            )
        else:
            raise ValueError(f"Unknown classifier type: {model_name}")
