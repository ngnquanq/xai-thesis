# 


from typing import List, Optional, Tuple, Dict
from typing_extensions import Annotated

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.preprocess import ColumnsDropper, DataFrameCaster, NADropper, Mapper, HandlingNull, EncodeCategorical, ConvertDataType, NormalizeData, DataPreProcessStrategy, DataCleaning
from zenml import step

# These following modules is for testing purpose
from .data_loader import *
from .inference_data_preprocessor import *
from .train_data_splitter import *


@step
def train_data_preprocessor(
    dataset_trn: Optional[pd.DataFrame] = None,
    dataset_tst: Optional[pd.DataFrame] = None,
    mapping: Dict[str, float] = {'False': 0.0, 'True': 1.0},
    convert_data: Optional[bool] = True,
    handle_na: Optional[bool] = True, 
    drop_na: Optional[bool] = False,
    normalize: Optional[bool] = True,
    drop_columns = ["Account length", "State", "Area code"],
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model training. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps like cleaning,
    feature engineering, feature selection, etc. It then returns the processed
    dataset as an step output artifact.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to drop NA values, drop some
    columns and normalize numerical columns. See the documentation for more
    information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        drop_na: If `True` all NA rows will be dropped.
        normalize: If `True` all numeric fields will be normalized.
        drop_columns: List of column names to drop.

    Returns:
        The processed datasets (dataset_trn, dataset_tst) and fitted `Pipeline` object.
    """
    # ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
    
    # # Dropping columns
    # if drop_columns:
    #     # Drop columns
    #     preprocess_pipeline.steps.append(("drop_columns", ColumnsDropper(columns=drop_columns)))
        
    # # Dropping NaN values
    # if drop_na:
    #     preprocess_pipeline.steps.append(("drop_na", NADropper()))
        
    # # Convert data type
    # if convert_data: 
    #     preprocess_pipeline.steps.append(("convert_data_type", ConvertDataType()))
        
    # # Handling NaN values
    # if handle_na:
    #     preprocess_pipeline.steps.append(("handle_na", HandlingNull()))
        
    # # Mapping data 
    # preprocess_pipeline.steps.append(("transform_target", Mapper(mapping=mapping)))
    
    # # Encode categorical columns
    # preprocess_pipeline.steps.append(("encode_cat_cols", EncodeCategorical()))
    
        
    # # Normalize data 
    # if normalize:
    #     # Normalize the data
    #     preprocess_pipeline.steps.append(("normalize", NormalizeData()))
        
    # preprocess_pipeline.steps.append(("cast", DataFrameCaster(dataset_trn.columns)))
    # dataset_trn = preprocess_pipeline.transform(dataset_trn)
    # dataset_tst = preprocess_pipeline.transform(dataset_tst)
    # ### YOUR CODE ENDS HERE ###

    # Initialize the data preprocessing strategy
    preprocess_strategy = DataPreProcessStrategy()

    # Create the pipeline with the DataCleaning step
    pipeline = Pipeline([
        ('preprocess_pipeline', DataCleaning(strategy=preprocess_strategy))
    ])

    # Example usage with your training and test datasets
    # dataset_trn = pd.DataFrame(...)  # Your training dataset
    # dataset_tst = pd.DataFrame(...)  # Your test dataset
    dataset_trn = None
    dataset_tst = None
    if dataset_trn is not None:
        # Fit the pipeline on the training data and transform it
        dataset_trn = pipeline.fit_transform(dataset_trn)
    if dataset_tst is not None:
        # Transform the test data using the same pipeline
        dataset_tst = pipeline.transform(dataset_tst)
    return dataset_trn, dataset_tst, pipeline

if __name__ == '__main__':
    dataset, target, random_state = data_loader(random_state=42)
    dataset_trn, dataset_tst = train_data_splitter(dataset)
    dataset_trn, dataset_tst, preprocess_pipeline = train_data_preprocessor(dataset_trn = dataset_trn, dataset_tst = dataset_tst)
    print(preprocess_pipeline)
                                                                            