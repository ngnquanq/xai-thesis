from typing import List, Dict
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from logging import getLogger

logger = getLogger(__name__)

def hp_tuning_select_best_model(
    hp_results: List[Dict[str, Annotated[ClassifierMixin, "hp_result"]]],
) -> Annotated[ClassifierMixin, "best_model"]:
    """Find the best model across all HP tuning attempts.

    This function takes the results from multiple hyperparameter tuning steps,
    each represented as a dictionary, and selects the model with the best accuracy score.

    Args:
        hp_results: A list of dictionaries, where each dictionary contains a model 
                    and associated metadata (including the accuracy score).

    Returns:
        The best possible model across all provided models.
    """
    # Initialize variables to keep track of the best model and its score
    best_model = None
    best_score = -1

    # Iterate over each result dictionary in the list
    for result in hp_results:
        # Assuming each result is a dictionary with keys 'model' and 'score'
        current_model = result.get("model")
        current_score = result.get("best_score", -1)  # Default to -1 if not found
   
        # Log the model and its score
        logger.info(f"Model: {current_model}, Score: {current_score}")

        # Check if this model has the best score so far
        if current_score > best_score:
            best_model = current_model
            best_score = current_score

    # Log the best model and its score
    logger.info(f"Best Model: {best_model}, Best Score: {best_score}")

    return best_model
