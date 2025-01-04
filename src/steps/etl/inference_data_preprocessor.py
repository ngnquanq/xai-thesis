import pandas as pd 
import numpy as np 

import logging
from typing import List
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def inference_data_preprocessor(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe for inference.
    
    Args:
        data (pd.DataFrame): DataFrame that needs to be preprocessed.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info("Begin preprocessing the dataframe for inference...")
    
    try:
        logger.info("1. Dropping unnecessary columns")
        data = data.drop(columns=["Account length", "State", "Area code"])
    except Exception as e:
        logger.exception("Error when dropping columns")
        raise e
    
    try:
        logger.info("2. Converting 'Churn' to 0 and 1 (float)")
        data['Churn'] = data['Churn'].astype(float).astype(int)
    except Exception as e:
        logger.exception("Error when converting 'Churn' column")
        raise e
    
    try:
        logger.info("3. Converting object columns to categorical")
        for col in data.select_dtypes(include='object').columns:
            data[col] = data[col].astype('category')
    except Exception as e:
        logger.exception("Error when converting object columns")
        raise e
    
    logger.info("4. Scaling and encoding the values")
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = data.select_dtypes(include='category').columns.tolist()
    
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    try:
        encoded_data = preprocessor.fit_transform(data)
        new_cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
        columns = num_cols + list(new_cat_cols) + ['Churn']
        
        encoded_df = pd.DataFrame(encoded_data, columns=num_cols + list(new_cat_cols))
        encoded_df['Churn'] = data['Churn'].values
        
        return encoded_df
    
    except Exception as e:
        logger.exception("Error during scaling and encoding")
        raise e
