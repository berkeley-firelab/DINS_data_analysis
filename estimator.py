import os
import json 
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.metrics import (fbeta_score,
                             accuracy_score,
                             matthews_corrcoef,
                             balanced_accuracy_score,
                             make_scorer)

from utils.directory_structure import DATA_DIR, OUTPUT_DIR
from utils.helpers import save_python_objects

from IPython import embed


with open(os.path.join(DATA_DIR, "dins_estimator_params.json")) as f:
    est_dict = json.load(f)

if est_dict["MIXED_TYPE"] is True:
    FEATURE_TYPE = "mixed"
else:
    FEATURE_TYPE = "numeric"

MODEL_TYPE = est_dict["MODEL_TYPE"]
MODEL_VERSION = est_dict["MODEL_VERSION"]
MODEL_DIR = os.path.join(OUTPUT_DIR, MODEL_TYPE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)


ML_MODEL_NAME = "{}_{}_using_{}_features.pkl".format(MODEL_TYPE, MODEL_VERSION, FEATURE_TYPE)
SAVE_NAME = os.path.join(MODEL_DIR, ML_MODEL_NAME)


def logit_model(X_train, y_train,
                cv_k_folds=10,
                refit_J="accu",
                do_grid_search=False,
                save_path=None):
    """
    Trains a logistic regression classifier for the given
    X_train and y_train. The training is done based on either
    the grid search or the K-fold cross validation.

    :param X_train: a numpy array of the training features
    :param y_train: a numpy array of the training labels
    :param cv_k_folds: number of folds for cross-validation
    :param refit_J: the cost surface based on which the
    best estimator is determined
    :param do_grid_search: flag to do grid search or not
    :param save_path: a path to save the trained model(s)
    :returns: either the classifier or the grid search results
    :rtype: sklearn objects

    """

    # the estimator with parameters optimized for global minima/maxima

    if do_grid_search:
        clf = LogisticRegression(solver="saga",
                                 warm_start=False,
                                 max_iter=10**3,
                                 multi_class="multinomial",
                                 penalty="elasticnet",
                                 fit_intercept=True)

        # custom cost functions
        J_COST_SURFACE = {"fbeta": make_scorer(fbeta_score, beta=2, average="micro"),
                          "accu": make_scorer(accuracy_score),
                          "bl": make_scorer(balanced_accuracy_score),
                          "mcc": make_scorer(matthews_corrcoef)}

        # grid search
        param_grid = [{"l1_ratio": np.logspace(-4, 0, 5, endpoint=True),
                       "C": np.logspace(-5, 1, 5, endpoint=True)}]

        grid_search = GridSearchCV(estimator=clf,
                                   param_grid=param_grid,
                                   cv=cv_k_folds,
                                   scoring=J_COST_SURFACE,
                                   refit=refit_J,
                                   n_jobs=-1,
                                   pre_dispatch="2*n_jobs")

        grid_search.fit(X_train, y_train)

        # save results
        est_dict["grid_search"] = grid_search
        est_dict["estimator"] = grid_search.best_estimator_

        if save_path:
            save_python_objects(est_dict, save_path)

        return grid_search
    
    else:
        clf = LogisticRegression(solver="saga",
                                 warm_start=False,
                                 max_iter=1000,
                                 multi_class="multinomial",
                                 penalty="elasticnet",
                                 n_jobs=-1,
                                 C=69.0522,
                                 fit_intercept=True,
                                 l1_ratio=0.95477)

        clf.fit(X_train, y_train)

        # save results
        est_dict["estimator"] = clf

        if save_path:
            save_python_objects(est_dict, save_path)

        return clf


def rf_model(X_train, y_train,
             cv_k_folds=10,
             refit_J="accu",
             do_grid_search=False,
             save_path=None):

    if do_grid_search:

        rfc = RandomForestClassifier(bootstrap=True,
                                     oob_score=True,
                                     random_state=42,
                                     n_jobs=-1)

        J_COST_SURFACE = {"fbeta" : make_scorer(fbeta_score, beta=2, average="micro"),
                          "accu" : make_scorer(accuracy_score),
                          "bl" : make_scorer(balanced_accuracy_score),
                          "mcc" : make_scorer(matthews_corrcoef)}
        param_dist = {
            "n_estimators": np.logspace(2, 3, 20, endpoint=True).astype(int).tolist(),
            "max_depth": [None]+np.linspace(10, 1000, num=10, endpoint=True, dtype=np.int).tolist(),
            "criterion": ["gini", "entropy"],
            "max_features": [None, "sqrt", "log2"],
            "max_samples": [None, 0.9],
            "min_impurity_decrease": [0]+np.logspace(-4, 0, 4, endpoint=False).tolist(),
            "class_weight": [None, "balanced", "balanced_subsample"]+[{0 : 1, 1 : i} for i in [3, 6, 9]],
            "ccp_alpha": np.logspace(-3, -1, 3, endpoint=False).tolist()
            }

        rnd_gs = RandomizedSearchCV(estimator=rfc,
                                    param_distributions=param_dist,
                                    n_iter=20,
                                    scoring=J_COST_SURFACE,
                                    cv=cv_k_folds,
                                    refit=refit_J,
                                    random_state=312,
                                    verbose=2)

        rnd_gs.fit(X_train, y_train)

        if save_path:
            save_python_objects(rnd_gs, save_path)

        return rnd_gs

    else:
        rfc = RandomForestClassifier(criterion="entropy")
        rfc.fit(X_train, y_train.ravel())

        save_python_objects(rfc, save_path)

        return rfc
