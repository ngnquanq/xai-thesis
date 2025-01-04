
from typing import Tuple, Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger



logger = get_logger(__name__)


def data_loader(
    random_state: int, 
    is_inference: bool = False,
    inference_data: Optional[str] = None
):
    """Dataset reader step.

    This is an example of a dataset reader step that load Breast Cancer dataset.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured with number of rows and logic
    to drop target column or not. See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        is_inference: If `True` subset will be returned and target column
            will be removed from dataset.
        random_state: Random state for sampling

    Returns:
        The dataset artifact as Pandas DataFrame and name of target column.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    dataset = pd.read_csv("data/telecom_churn.csv")
    inference_size = int(len(dataset.Churn) * 0.05)
    target = "Churn"
    # dataset: pd.DataFrame = dataset.frame
    inference_subset = dataset.sample(inference_size, random_state=random_state)
    if is_inference:
        dataset = pd.read_csv("data/synthetic_data.csv")
        #dataset.drop(columns=target, inplace=True)
    else:
        #dataset.drop(inference_subset.index, inplace=True)
        pass
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    ### YOUR CODE ENDS HERE ###
    return dataset, target, random_state

if __name__=='__main__':
    dataset, target, random_state = data_loader(random_state=42)
    print(dataset)
