import os
import numpy as np
import pandas as pd


# Geospatial imports
import pyproj
from pyproj import Transformer


# ML related imports
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import haversine_distances, pairwise_distances


# AT utility imports
from utils.directory_structure import DATA_DIR


from IPython import embed

##-----------------------------------------------
## Helper functions for abstracting the steps
##-----------------------------------------------


def latlon_to_utm(row):
    """converts lat, lon to UTM

    :param lat: lat in degree
    :param lon: lon in degree
    :return: easting, northing, and utm zone
    """    
    
    lat, lon = row["LATITUDE"], row["LONGITUDE"]

    # determine the UTM zone number from longitude and 
    # whether zone is in the northern or southern hemisphere
    utm_zone = int(1 + (lon + 180.0) / 6.0)
    is_northern = lat >= 0

    # UTM projection and transformer from WGS84 to UTM
    utm_crs = pyproj.CRS.from_user_input(f"+proj=utm +zone={utm_zone} +{'+north' if is_northern else '+south'} +datum=WGS84")
    transformer = Transformer.from_crs(pyproj.CRS("EPSG:4326"), utm_crs, always_xy=True)

    easting, northing = transformer.transform(lon, lat)
    
    return pd.Series([easting, northing, utm_zone])


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


def read_split(data_file_name, utm=True):
    """reads the data from CSV file, does minor edits, and
    separates based on class balances in the dataset.

    :param data_file_name: string of the file name
    :return: panda dataframes of X_train_full, X_test, y_train_full, y_test, 
    and tuple of (num_cols, cat_cols) names in string format.
    """    

    file_path = os.path.join(DATA_DIR, data_file_name)
    missing_values = ["", "NA", "na", "n/a", "N/A", "--", "nan"]
    df = pd.read_csv(file_path, delimiter=",", na_values=missing_values)

    # dropping distance_meter
    df.drop("distance_meter", axis=1, inplace=True)

    # converting YEARBUILT to a numeric column in the main dataset 
    df["YEARBUILT"] = pd.to_numeric(df["YEARBUILT"].values, errors="coerce")

    # adding UTM coordinate and zone for potential imputation methods
    if utm:
        df[["utm_easting", "utm_northing", "utm_zone"]] = df.apply(latlon_to_utm, axis=1)

    # identify the numerical and categorical data types
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    X_train_full, X_test, y_train_full, y_test = data_spliter(df)

    return X_train_full, X_test, y_train_full, y_test, (num_cols, cat_cols)


def haversine_imputer_catnum(df, var="YEARBUILT", n_neighbors=5, is_categorical=False):
    """
    Imputes the missing values in var feature of the dataframe (df)
    using the N closest Haversine distances based on the location data

    :param df: pandas dataframe of the features 
    :param var: string of the feature (variable) name, defaults to "YEARBUILT"
    :param n_neighbors: number of nearest neighbors to consider for imputation
    :param is_categorical: boolean indicating if the variable is categorical
    :return: pandas dataframe with imputed data
    """


    df = df[["LATITUDE", "LONGITUDE", var]].copy()
    df.replace('NaN', np.nan, inplace=True)

    # convert lat and lon values to radians
    for c in ["LATITUDE", "LONGITUDE"]:
        df.iloc[:, df.columns.get_loc(c)] = np.radians(df[c].values)
    
    # split into NaN and non-NaN dataframes
    df_nan = df[df[var].isna()]
    df_ref = df[~df[var].isna()]

    # calculate distance matrix (takes some time to calculate)
    dist_matrix = haversine_distances(df_nan.iloc[:, :2], df_ref.iloc[:, :2]) * 6371  # Earth radius in kilometers
    nearest_idxs = np.argpartition(dist_matrix, n_neighbors, axis=1)[:, :n_neighbors]

    # impute missing values
    for i, idxs in enumerate(nearest_idxs):
        if is_categorical:
            # mode (most frequent value) for categorical data
            imputed_value = df_ref.iloc[idxs][var].mode()[0]
        else:
            # mean for numerical data
            imputed_value = df_ref.iloc[idxs][var].mean()
        
        df_nan.iloc[i, df_nan.columns.get_loc(var)] = imputed_value

    df_imputed = pd.concat([df_ref, df_nan])

    return df_imputed


def pairwise_dist_imputer_catnum(data, nan_cols, n_neighbors=5, metric="euclidean"):
    """
    

    :param df: pandas dataframe of the features 
    :param nan_cols: a list of feature (variable) names (in string) that have nan or missing values
    :param n_neighbors: number of nearest neighbors to consider for imputation
    :param metric: method for calculating the matrix distance based on UTM projections
    :return: pandas dataframe with imputed data
    """


    df = data.copy()

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()

    for c in nan_cols:
        df_nan = df[df[c].isna()]
        df_ref = df[~df[c].isna()]
        df_nan_idx = df_nan.index
        
        dist_matrix = pairwise_distances(df_nan.loc[:, ("utm_easting", "utm_northing")].values, 
                                         df_ref.loc[:, ("utm_easting", "utm_northing")].values, 
                                         metric=metric, n_jobs=-1)
        nearest_idxs = np.argpartition(dist_matrix, n_neighbors, axis=1)[:, :n_neighbors]

        if c in cat_cols:
            for i, idx in enumerate(nearest_idxs):
                agg_value = df_ref.iloc[idx][c].mode()[0]
                df_nan.iloc[i, df_nan.columns.get_loc(c)] = agg_value

        elif c in num_cols:
            for i, idx in enumerate(nearest_idxs): 
                agg_value = df_ref.iloc[idx][c].mean()
                df_nan.iloc[i, df_nan.columns.get_loc(c)] = agg_value

        else:
            print(c, " ---> Data type is not valid!")

        df.loc[df_nan_idx, c] = df_nan.loc[df_nan_idx, c]

    return df