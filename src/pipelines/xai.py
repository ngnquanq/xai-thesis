import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
from sklearn.base import BaseEstimator
from typing import Tuple, List

def calculate_shap_lime_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    feature_names: List[str],
    num_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate SHAP and LIME scores for a given model and dataset.

    Args:
        model (BaseEstimator): Trained model
        X (pd.DataFrame): Input features
        feature_names (List[str]): List of feature names
        num_samples (int): Number of samples to use for LIME explanation

    Returns:
        Tuple[np.ndarray, np.ndarray]: SHAP and LIME scores
    """
    # Calculate SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_scores = np.abs(shap_values.values).mean(axis=0)

    # Calculate LIME scores
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=feature_names,
        class_names=['Not Churn', 'Churn'],
        mode='classification'
    )
    
    lime_scores = np.zeros(len(feature_names))
    for _ in range(num_samples):
        idx = np.random.randint(0, len(X))
        exp = lime_explainer.explain_instance(X.iloc[idx].values, model.predict_proba)
        for feature, importance in exp.as_list():
            feature_idx = feature_names.index(feature)
            lime_scores[feature_idx] += abs(importance)
    
    lime_scores /= num_samples

    return shap_scores, lime_scores
