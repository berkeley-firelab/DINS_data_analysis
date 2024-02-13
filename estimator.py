import pandas as pd

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTEN




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


