import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 


# ML related imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import haversine_distances


# AT utility imports
from utils.directory_structure import DATA_DIR



##-----------------------------------------------
## Helper functions for abstracting the steps
##-----------------------------------------------

def data_spliter(dataframe, test_size=0.2, target_label="DAMAGE"):
    """separates the train and test dataset using stratified
    shuffle sampling.

    :param dataframe: pandas dataframe of the entire data
    :param test_size: portion allocated to test set, defaults to 0.2
    :param target_label: feature label to use for stratification, defaults to "DAMAGE"
    :return: panda dataframes of X_train, X_test, y_train, y_test
    """    

    df = dataframe.copy()
    y = df.pop(target_label)
    X = df

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=1980891, 
                                                        shuffle=True, 
                                                        stratify=y.values)
    return X_train, X_test, y_train, y_test


def read_split(data_file_name):
    """reads the data from CSV file, does minor edits, and
    separates based on class balances in the dataset.

    :param data_file_name: string of the file name
    :return: panda dataframes of X_train_full, X_test, y_train_full, y_test, 
    and tuple of (num_cols, cat_cols) names in string format.
    """    

    file_path = os.path.join(DATA_DIR, data_file_name)
    df = pd.read_csv(file_path, delimiter=",")

    # dropping distance_meter
    df.drop("distance_meter", axis=1, inplace=True)

    # converting YEARBUILT to a numeric column in the main dataset 
    df["YEARBUILT"] = pd.to_numeric(df["YEARBUILT"].values, errors="coerce")

    # identify the numerical and categorical data types
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns


    X_train_full, X_test, y_train_full, y_test = data_spliter(df)

    return X_train_full, X_test, y_train_full, y_test, (num_cols, cat_cols)


def haversine_dist_imputer(df, var="YEARBUILT"):
    """
    Imputes the missing values in var feature of the dataframe (df)
    using the closest Haversine distance based on the location data

    :param df: pandas dataframe of the features 
    :param var: string of the feature (variable) name, defaults to "YEARBUILT"
    :return: pandas dataframes of the reference and imputed data
    """

    df = df[["LATITUDE", "LONGITUDE", var]].copy()
    df.replace('NaN', np.nan, inplace=True)
    
    df_nan = df[df[var].isna()]
    df_ref = df[~df[var].isna()]
    del df
    
    for i in range(df_nan.shape[0]):
        dist_matrix = haversine_distances(df_ref.iloc[:, :2].values, df_nan.iloc[i, :2].values.reshape(1, 2))
        idx = np.argpartition(dist_matrix.reshape(-1), 2)[1]
        df_nan.iloc[i, 2] = df_ref.iloc[idx, 2]

    return df_ref, df_nan


def haversine_nearest_dist_imputer(data, var="YEARBUILT"):
    """A faster version of the location based imputer
    using nearest neighbor distance

    :param data: pandas dataframe of the data that includes lat, lon, var in exactly this order
    :param var: variable that has NaN or missing values, defaults to "YEARBUILT"
    :return: pandas dataframes of the reference and imputed data 
    """    
    
    df = data.copy()
    df = df[["LATITUDE", "LONGITUDE", var]]
    df.replace('NaN', np.nan, inplace=True)
        
    df_nan = df[df[var].isna()]
    df_ref = df[~df[var].isna()]

    idx = pairwise_distances_argmin(df_nan.iloc[:, :2].values, df_ref.iloc[:, :2].values, metric='haversine')
    df_nan.iloc[:, 2] = df_ref.iloc[idx, 2].values
    
    return df_ref, df_nan
