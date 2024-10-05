from typing import Union
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
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

# class DataPreProcessStrategy(DataStrategy):
#     """Inherit the DataStrategy and overwrite the handle_data method provided by the DataStrategy above"""
    
#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: 
#         """Preprocess the dataframe
        
#         Args:
#             data (pd.DataFrame): DataFrame that needs to be preprocessed.
#         """
#         # Drop unnecessary columns
#         logger.info("Begin preprocessing the dataframe ...")
#         try: 
#             logger.info("1. Start dropping unnecessary columns")
#             data = data.drop(columns=["Account length", "State", "Area code"])
#             logger.info("Dropped unnecessary columns successfully")
#         except Exception as e:
#             logger.exception("Encountered an exception when dropping columns")
#             raise e
        
#         # Convert 'Churn' column to 0 and 1 (float)
#         try: 
#             logger.info("2. Converting 'Churn' to 0 and 1 (float)")
#             data['Churn'] = data['Churn'].astype(float).astype(int)
#             logger.info("'Churn' conversion complete")
#         except Exception as e:
#             logger.exception("Encountered an exception when converting 'Churn' column")
#             raise e

#         # Convert other object columns to category
#         try: 
#             logger.info("3. Converting other object columns to categorical data type")
#             for col in data.select_dtypes(include='object').columns.to_list():
#                 data[col] = data[col].astype('category')
#             logger.info("Conversion of object columns complete")
#         except Exception as e:
#             logger.exception("Encountered an exception when converting object columns")
#             raise e
        
#         # Handle null values
#         try: 
#             if data.isnull().sum().any():
#                 logger.info("4. Handling null values")
#                 for col in data.select_dtypes(include=['int64', 'float64']).columns.to_list():
#                     data[col].fillna(data[col].mean(), inplace=True)
#                 data = data.dropna()
#             else: 
#                 logger.info("4. The data has no missing values")
#         except Exception as e: 
#             logger.exception("Encountered an exception when handling null values")
#             raise e

#         # Identify columns for scaling and encoding
#         logger.info("5. Scaling and encoding the values")
#         num_col = data.select_dtypes(include=['int64', 'float64']).columns.values.tolist()
#         cat_col = data.select_dtypes(include='category').columns.values.tolist()
        
#         # Ensure 'Churn' is excluded from categorical processing
#         if 'Churn' in num_col:
#             num_col.remove('Churn')

#         # Define pipelines for numeric and categorical transformations
#         numeric_transformer = Pipeline(
#             steps=[("Scaler", StandardScaler())]
#         )

#         categorical_transformer = Pipeline(
#             steps=[('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))]
#         )

#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', numeric_transformer, num_col),
#                 ('cat', categorical_transformer, cat_col)
#             ]
#         )

#         # Apply transformations
#         encoded_data = preprocessor.fit_transform(data)
        
#         # Create new column names
#         new_num_col = num_col
#         new_cat_col = preprocessor.named_transformers_['cat'].named_steps['OneHotEncoder'].get_feature_names_out(cat_col)
#         columns = list(new_num_col) + list(new_cat_col) + ['Churn']  # Add 'Churn' back to the column list

#         # Combine transformed data with the original 'Churn' column
#         encoded_data = pd.DataFrame(encoded_data, columns=list(new_num_col) + list(new_cat_col))
#         encoded_data['Churn'] = data['Churn'].values  # Add the 'Churn' column back as 0/1
        
#         return encoded_data
class DataPreProcessStrategy(DataStrategy):
    """Preprocess the dataframe while retaining column names for XAI purposes."""

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Begin preprocessing the dataframe ...")
        
        # Create a copy of the data to avoid modifying the original DataFrame
        data = data.copy()
        
        # Map 'Yes'/'No' and 'True'/'False' to 1/0 for binary categorical variables
        try:
            logger.info("1. Encoding binary categorical variables")
            binary_cols = ['International plan', 'Churn']
            for col in binary_cols:
                data[col] = data[col].map({'No': 0, 'Yes': 1, 'False': 0, 'True': 1})
            logger.info("Binary categorical variables encoded successfully")
        except Exception as e:
            logger.exception("Error encoding binary categorical variables")
            raise e

        # Decide between 'Voice mail plan' and 'Number vmail messages'
        # Based on the analysis, we will keep 'Number vmail messages' and drop 'Voice mail plan'
        try:
            logger.info("2. Dropping 'Voice mail plan' due to multicollinearity with 'Number vmail messages'")
            data = data.drop(columns=['Voice mail plan'])
            logger.info("'Voice mail plan' dropped successfully")
        except Exception as e:
            logger.exception("Error dropping 'Voice mail plan'")
            raise e

        # Drop features due to multicollinearity and lack of statistical significance
        try:
            logger.info("3. Dropping features due to multicollinearity and insignificance")
            data = data.drop(columns=[
                # Multicollinearity
                'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
                # Not statistically significant
                'Total day calls', 'Account length', 'Total eve calls', 'State', 'Area code', 'Total night calls'
            ])
            logger.info("Features dropped successfully")
        except Exception as e:
            logger.exception("Error dropping features")
            raise e

        # Handle missing values
        try:
            logger.info("4. Handling missing values")
            if data.isnull().sum().any():
                # Fill numerical columns with mean
                num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                num_cols.remove('Churn')  # Exclude target variable
                for col in num_cols:
                    data[col].fillna(data[col].mean(), inplace=True)
                # Fill categorical columns with mode
                cat_cols = data.select_dtypes(include='object').columns.tolist()
                for col in cat_cols:
                    data[col].fillna(data[col].mode()[0], inplace=True)
                logger.info("Missing values handled successfully")
            else:
                logger.info("No missing values found")
        except Exception as e:
            logger.exception("Error handling missing values")
            raise e

        # Scale numerical features while retaining column names
        try:
            logger.info("5. Scaling numerical features")
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols.remove('Churn')  # Exclude the target variable

            scaler = StandardScaler()
            # Fit the scaler and transform the numerical columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

            logger.info("Numerical features scaled successfully")
        except Exception as e:
            logger.exception("Error scaling numerical features")
            raise e

        # Encode remaining categorical variables if any
        try:
            logger.info("6. Encoding remaining categorical variables")
            cat_cols = data.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                # Use get_dummies to one-hot encode categorical variables while retaining column names
                data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
                logger.info("Categorical variables encoded successfully")
            else:
                logger.info("No categorical variables to encode")
        except Exception as e:
            logger.exception("Error encoding categorical variables")
            raise e

        # Ensure column names are retained
        data.columns = data.columns.astype(str)
        logger.info(f"Final columns after preprocessing: {data.columns.tolist()}")

        logger.info("Data preprocessing complete.")
        return data
    
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
