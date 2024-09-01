
from typing import Any, Dict
from typing_extensions import Annotated

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from utils import get_model_from_config
from logging import getLogger

logger = getLogger(__name__)

def hp_tuning_single_search(
    model_package: str, 
    model_class: str,
    search_grid: Dict[str, Any],
    dataset_trn: pd.DataFrame, 
    dataset_tst: pd.DataFrame,
    target: str
) -> Annotated[ClassifierMixin, "hyper_tuning_result"]:
    """Later
    """
    # Get the model from config
    model_class = get_model_from_config(model_package=model_package, model_class=model_class)
    
    for search_key in search_grid:
        if "range" in search_grid[search_key]:
            search_grid[search_key] = range(search_grid[search_key]["range"]["start"],
                                            search_grid[search_key]["range"]["end"],
                                            search_grid[search_key]["range"].get("step",1))
    
    # Get the data and apply it
    X_train = dataset_trn.drop(columns=[target])
    y_train = dataset_trn[target]
    X_test = dataset_tst.drop(columns=[target])
    y_test = dataset_tst[target]
    
    logger.info("Running hyperparameter tuning ...")
    
    cv = RandomizedSearchCV(
        estimator=model_class(),
        param_distributions=search_grid, 
        cv=3,
        n_jobs=-1,
        n_iter=10,
        random_state=42
        scoring="accuracy",
        refit=True
    )
    
    cv.fit(X=X_train, y=y_train)
    y_pred  = cv.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    # Need a file to log scoring as metadata including metric 
    
    return cv.best_estimator_