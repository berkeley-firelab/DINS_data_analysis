import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder 
from utils.directory_structure import OUTPUT_DIR


"""a class for multi-column encoding"""

class categorical_to_numerical():
    """does multi-column imputation for each categorical column
    """

    def __init__(self):
        pass
    
    def separate_to_cat_num(self, df):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
        return df[cat_cols], df[num_cols]    

    def cat_encoder(self, cat_df, drop_cols=None):
        cat_df = cat_df.copy()
        cat_cols = cat_df.columns.to_list()
        if drop_cols:
            cat_df = cat_df.drop(drop_cols, axis=1)

        encoder = OneHotEncoder(dtype=np.float64, sparse_output=False)
        
        for i, col in enumerate(cat_cols):
            encoded_data = encoder.fit_transform(cat_df[[col]])
            col_names = [f"{col}_{category}" for category in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_data, columns=col_names, index=cat_df.index)
            if i == 0:                
                new_df = encoded_df
            else:
                new_df = pd.concat([new_df, encoded_df], axis=1)

        return new_df


class custom_categorical_encoder(BaseEstimator, TransformerMixin, categorical_to_numerical):
    """
    A custom transformer for the categorical data
    Input:
        reads in the categorical dataframe
    Transforms:
        transforms it into a dataframe consisting of a set of added features and a set of encoded features.
    """

    def __init__(self):
        super().__init__()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cat_df, num_df = self.separate_to_cat_num(X)
        cat_encoded = self.cat_encoder(cat_df)
        return num_df.join(cat_encoded)