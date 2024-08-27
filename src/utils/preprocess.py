# 


from typing import Union

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin



class NADropper:
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.dropna()
    

class ConvertDataType:
    """Support class to convert object columns to categorical in sklearn Pipeline."""
    
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        # Convert object columns to category
        for col in X.select_dtypes(include='object').columns.to_list():
            X[col] = X[col].astype('category')
        return X    

from typing import Union
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class EncodeCategorical:
    def __init__(self):
        self.encoder = None
        self.categorical_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        # Select only categorical columns
        self.categorical_columns = X.select_dtypes(include=['category']).columns.to_list()
        
        # Initialize and fit the OneHotEncoder on the selected categorical columns
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[self.categorical_columns])
        return self

    def transform(self, X: pd.DataFrame):
        # Apply the encoder to the categorical columns
        transformed_data = self.encoder.transform(X[self.categorical_columns])
        columns = self.encoder.get_feature_names_out(self.categorical_columns)
        
        # Create a DataFrame for the transformed data
        transformed_df = pd.DataFrame(transformed_data, columns=columns, index=X.index)
        
        # Drop the original categorical columns and concatenate the transformed data
        X_dropped = X.drop(columns=self.categorical_columns)
        X_transformed = pd.concat([X_dropped, transformed_df], axis=1)
        
        return X_transformed

    
class HandlingNull:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        try:
            if X.isnull().sum().any():
                for column in X.select_dtypes(include=['int64', 'float64']).columns.to_list():
                    X[column].fillna(X[column].mean(), inplace=True)
                X = X.dropna()
            else:
                print("The data had no missing values")
        except Exception as e:
            print(f"Encountering an exception when handling null value: {e}")
        return X

class ColumnsDropper:
    """Support class to drop specific columns in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.drop(columns=self.columns)


class DataFrameCaster:
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

class Mapper:
    """Support class for mapping values in sklearn Pipeline."""
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, target_col: str = 'Churn'):
        # Check if the target column exists in the DataFrame
        if target_col in X.columns:
            # Apply the mapping only to the specified target column
            X[target_col] = X[target_col].map(self.mapping)
        else:
            raise ValueError(f"Column '{target_col}' does not exist in the DataFrame.")
        return X
    

class NormalizeData:
    """Support class to normalize data except the 'Churn' column using StandardScaler."""
    
    def __init__(self):
        self.scaler = None
        self.columns_to_scale = None

    def fit(self, X: pd.DataFrame, y=None):
        # Identify columns to scale (excluding 'Churn')
        self.columns_to_scale = [col for col in X.columns if col != 'Churn']
        
        # Initialize and fit the StandardScaler on the selected columns
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X: pd.DataFrame):
        # Scale the selected columns
        X_scaled = X.copy()
        X_scaled[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X_scaled

import logging
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd 
from typing import Union, Tuple, Annotated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


logger = logging.getLogger(__name__)


# Abstract class is a strategy for handling data
class DataStrategy(ABC):
    """Abstract class defining strategy for handling data

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass 
    
    
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class DataPreProcessStrategy(DataStrategy):
    """Inherit the DataStrategy and overwrite the handle_data method provided by the DataStrategy above"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: 
        """Preprocess the dataframe
        
        Args:
            data (pd.DataFrame): DataFrame that needs to be preprocessed.
        """
        # Drop unnecessary columns
        logger.info("Begin preprocessing the dataframe ...")
        try: 
            logger.info("1. Start dropping unnecessary columns")
            data = data.drop(columns=["Account length", "State", "Area code"])
            logger.info("Dropped unnecessary columns successfully")
        except Exception as e:
            logger.exception("Encountered an exception when dropping columns")
            raise e
        
        # Convert 'Churn' column to 0 and 1 (float)
        try: 
            logger.info("2. Converting 'Churn' to 0 and 1 (float)")
            data['Churn'] = data['Churn'].astype(float).astype(int)
            logger.info("'Churn' conversion complete")
        except Exception as e:
            logger.exception("Encountered an exception when converting 'Churn' column")
            raise e

        # Convert other object columns to category
        try: 
            logger.info("3. Converting other object columns to categorical data type")
            for col in data.select_dtypes(include='object').columns.to_list():
                data[col] = data[col].astype('category')
            logger.info("Conversion of object columns complete")
        except Exception as e:
            logger.exception("Encountered an exception when converting object columns")
            raise e
        
        # Handle null values
        try: 
            if data.isnull().sum().any():
                logger.info("4. Handling null values")
                for col in data.select_dtypes(include=['int64', 'float64']).columns.to_list():
                    data[col].fillna(data[col].mean(), inplace=True)
                data = data.dropna()
            else: 
                logger.info("4. The data has no missing values")
        except Exception as e: 
            logger.exception("Encountered an exception when handling null values")
            raise e

        # Identify columns for scaling and encoding
        logger.info("5. Scaling and encoding the values")
        num_col = data.select_dtypes(include=['int64', 'float64']).columns.values.tolist()
        cat_col = data.select_dtypes(include='category').columns.values.tolist()
        
        # Ensure 'Churn' is excluded from categorical processing
        if 'Churn' in num_col:
            num_col.remove('Churn')

        # Define pipelines for numeric and categorical transformations
        numeric_transformer = Pipeline(
            steps=[("Scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_col),
                ('cat', categorical_transformer, cat_col)
            ]
        )

        # Apply transformations
        encoded_data = preprocessor.fit_transform(data)
        
        # Create new column names
        new_num_col = num_col
        new_cat_col = preprocessor.named_transformers_['cat'].named_steps['OneHotEncoder'].get_feature_names_out(cat_col)
        columns = list(new_num_col) + list(new_cat_col) + ['Churn']  # Add 'Churn' back to the column list

        # Combine transformed data with the original 'Churn' column
        encoded_data = pd.DataFrame(encoded_data, columns=list(new_num_col) + list(new_cat_col))
        encoded_data['Churn'] = data['Churn'].values  # Add the 'Churn' column back as 0/1
        
        return encoded_data

class DataCleaning(BaseEstimator, TransformerMixin):
    """
    Data cleaning class which preprocesses the data and can be used within a sklearn pipeline.
    """

    def __init__(self, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method, required by sklearn but not used here."""
        return self

    def transform(self, X: pd.DataFrame):
        """Applies the strategy's handle_data method to transform the data."""
        return self.strategy.handle_data(X)
