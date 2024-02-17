import os
import pickle
import numpy as np
import pandas as pd
import swifter

# Geospatial imports
import pyproj
from pyproj import Transformer


# ML related imports
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import haversine_distances, pairwise_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTEN


# AT utility imports
from utils.directory_structure import DATA_DIR, OUTPUT_DIR
from utils.custom_encoder import categorical_to_numerical

from IPython import embed

"""functions for abstracting the steps"""


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


def read_feature_engineer_split(data_file_name, x_eng=True):
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

    # dropping samples with YEARBUILT less than 1800!
    df = df[df["YEARBUILT"] >= 1800]

    # adding UTM coordinate and zone for potential imputation methods
    if x_eng:
        df[["utm_easting", "utm_northing", "utm_zone"]] = df.swifter.apply(latlon_to_utm, axis=1)
        df = feature_engineering(df, cols_drop=["DISTANCE"])

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
    conducts imputation based on pairwise distance clustering in UTM projection 

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
                                         metric=metric, n_jobs=10)
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


def latlon_to_xyz(df):
    """converting lat, lon, alt to x, y, z coordinates
    this is a minor feature engineering to enable the
    use of data for a variety of ML models, in particular 
    the ones that need a Standard feature set.

    Method:
        x = math.cos(phi) * math.cos(theta) * rho
        y = math.cos(phi) * math.sin(theta) * rho
        z = math.sin(phi) * rho # z is 'up'
    
        where phi = lat, theta = lon

    Since there is no elevation and the numerics will be 
    normalized, rho = 1.

    :param data: pandas dataframe of the entire dataset
    """

    data = df.copy()
    data["X"] = np.cos(data["LATITUDE"].values) * np.cos(data["LONGITUDE"].values)
    data["Y"] = np.cos(data["LATITUDE"].values) * np.sin(data["LONGITUDE"].values)
    data["Z"] = np.sin(data["LATITUDE"].values)

    return data


def feature_engineering(X, cols_drop=None):
    """minor feature engineering on the dataset

    :param X: pandas dataframe of the features 
    :return: pandas dataframe of the features + processed/added ones
    """


    df = X.copy()

    # convert (lat, lon) to xyz
    df = latlon_to_xyz(df)    

    # nonlinear transformation of distance (provides a better PDF)
    df["SSD"] = np.log10(df["DISTANCE"] * 0.3048 + 1)

    if cols_drop:
        df.drop(cols_drop, axis=1, inplace=True)

    return df




def data_preprocessing_pipeline(case_name, renew_data=False, encode_data=True, scale_data=True, weighted_classes=False):
    """_summary_

    :param case_name: _description_
    :param renew_data: _description_, defaults to False
    :param encode_data: _description_, defaults to True
    :param scale_data: _description_, defaults to True
    :return: _description_
    """
    if weighted_classes is True:
        postfix_name = "lbe_targets"
    else:
        postfix_name = "ohe_targets"

    if encode_data:
        fname = os.path.join(OUTPUT_DIR, f"{case_name}_{postfix_name}_train_test_ml_ready_data.pkl")
    else:
        fname = os.path.join(OUTPUT_DIR, f"{case_name}_{postfix_name}_train_test_catboost_ready_data.pkl")

    col_processor = categorical_to_numerical()
    if renew_data:
        # step 1. 
        print("Read, feature engineer, and split between train and test")
        X_train_full, X_test, y_train_full, y_test, _ = read_feature_engineer_split(f"{case_name}.csv")

        # step 2. 
        print("Imputation based on location information")
        X_train_nan_cols = X_train_full.columns[X_train_full.isna().any()].tolist()
        X_train_full = pairwise_dist_imputer_catnum(X_train_full, nan_cols=X_train_nan_cols)
        
        X_test_nan_cols = X_test.columns[X_test.isna().any()].tolist()
        X_test = pairwise_dist_imputer_catnum(X_test, nan_cols=X_test_nan_cols)      

        if encode_data:
            # step 3. 
            print("Encoding")
            X_encoder = OneHotEncoder(dtype=np.float64, sparse_output=False)
            X_train_cat, X_train_num = col_processor.separate_to_cat_num(X_train_full)        
            X_test_cat, X_test_num = col_processor.separate_to_cat_num(X_test)

            X_train_cat_encoded = X_encoder.fit_transform(X_train_cat)
            X_test_cat_encoded  = X_encoder.transform(X_test_cat)
            X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded, columns=X_encoder.get_feature_names_out().tolist(), index=X_train_full.index)
            X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded, columns=X_encoder.get_feature_names_out().tolist(), index=X_test.index)
            X_train_encoded = X_train_num.join(X_train_cat_encoded)
            X_test_encoded = X_test_num.join(X_test_cat_encoded)

        else:
            print("No encoding for the features!")
            X_train_encoded = X_train_full
            X_test_encoded = X_test

        # step 4. encoding vs labeling (depending on whether classes have weights)
        if weighted_classes is True:
            y_encoder = LabelEncoder()
            y_train_encoded = y_encoder.fit_transform(y_train_full.values.ravel())
            y_test_encoded = y_encoder.transform(y_test.values.ravel())
            y_train_encoded = pd.DataFrame(y_train_encoded, columns=["DAMAGE"], index=y_train_full.index)
            y_test_encoded = pd.DataFrame(y_test_encoded, columns=["DAMAGE"], index=y_test.index)

        else:
            y_encoder = OneHotEncoder(dtype=np.float64, sparse_output=False)
            y_train_encoded = y_encoder.fit_transform(y_train_full.values.reshape(-1, 1))
            y_test_encoded  = y_encoder.transform(y_test.values.reshape(-1, 1))
            y_col_name = [y_train_full.name+"_"+c.split("_")[1] for c in y_encoder.get_feature_names_out().tolist()]
            y_train_encoded = pd.DataFrame(y_train_encoded, columns=y_col_name, index=y_train_full.index)
            y_test_encoded = pd.DataFrame(y_test_encoded, columns=y_col_name, index=y_test.index)

        # step 5.
        cols_drop  = ["LATITUDE", "LONGITUDE", "utm_easting", "utm_northing", "utm_zone", "YEARBUILT", "SSD"]
        cols_scale = ["YEARBUILT", "SSD"]
        if scale_data:
            print("Normalize the required features and drop extra information!")
            scaler = StandardScaler()
            X_train_encoded[["year_scaled", "ssd_scaled"]] = scaler.fit_transform(X_train_encoded[cols_scale])
            X_test_encoded[["year_scaled", "ssd_scaled"]] = scaler.transform(X_test_encoded[cols_scale])
            X_train_encoded.drop(cols_drop, axis=1, inplace=True)
            X_test_encoded.drop(cols_drop, axis=1, inplace=True)
        else:
            pass
        
        if encode_data:
            data_dict = {"X_train": X_train_encoded, "X_test": X_test_encoded, 
                         "y_train": y_train_encoded, "y_test": y_test_encoded,
                         "X_encoder": X_encoder, "y_encoder": y_encoder}
        else:
            data_dict = {"X_train": X_train_encoded, "X_test": X_test_encoded, 
                         "y_train": y_train_encoded, "y_test": y_test_encoded, 
                         "y_encoder": y_encoder}
        
        with open(fname, 'wb') as file:
            pickle.dump(data_dict, file)
    else:
        with open(fname, 'rb') as file:
            data_dict = pickle.load(file)

    return data_dict


def balance_classes(X, y, strategy="auto", k_neighbors=10, mixed_features=False):
    """resample to balance class representation in dataset

    :param X: a dataframe of features
    :param y: a dataframe of target classes which can be encoded 
    :param strategy: sampling strategy, defaults to "auto"
    :param k_neighbors: number of neighboring points, defaults to 10
    :param mixed_features: boolean determining whether the features X
    are only numiercal or mix of numerical and categorical, defaults to False
    :return: _description_
    """

    X = X.copy()
    y = y.copy()
    y_cols = y.columns.tolist()

    if mixed_features is True:
        crs = SMOTEN(sampling_strategy=strategy, random_state=2001, k_neighbors=k_neighbors, n_jobs=-1)
    else:
        crs = SMOTETomek(sampling_strategy=strategy, random_state=1991, n_jobs=-1)

    X_train, y_train = crs.fit_resample(X, y.values)
    y_train = pd.DataFrame(y_train, columns=y_cols, index=X_train.index)

    return X_train, y_train



