import os
import json 
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools import add_constant
from statsmodels.genmod import families
import statsmodels.api as sm
from sklearn.metrics import (fbeta_score,
                             accuracy_score,
                             matthews_corrcoef,
                             balanced_accuracy_score,
                             make_scorer)

from utils.directory_structure import DATA_DIR, OUTPUT_DIR
from utils.helpers import save_python_objects

from IPython import embed

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True, family=None):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.family = family

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        if self.family is None:
            self.family = sm.families.Gaussian()  # Default to Gaussian if family is not provided
        self.model_ = self.model_class(y, X, family=self.family)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

with open(os.path.join(DATA_DIR, "estimator_params.json")) as f:
    est_dict = json.load(f)

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
                                 max_iter=int(1.e4),
                                 multi_class="auto",
                                 penalty="elasticnet",
                                 tol=1.e-4,
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
                                 max_iter=int(1.e4),
                                 multi_class="auto",
                                 penalty="elasticnet",
                                 n_jobs=-1,
                                 C=1,
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
            "max_depth": [None]+np.linspace(10, 1000, num=10, endpoint=True, dtype=np.int64).tolist(),
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

def xgb_model(X_train, y_train,
              cv_k_folds=10,
              refit_J="accu",
              do_grid_search=False,
              save_path=None):

    xgb_ = xgb.XGBClassifier

    """
    Trains an XGBoost classifier model.

    :param X_train: Training data features (DataFrame).
    :param y_train: Training data labels (Series).
    :param cv_k_folds: Number of cross-validation folds (default=10).
    :param refit_J: Metric to use for refitting the model (default="accu").
    :param do_grid_search: Whether to perform grid search for hyperparameter tuning (default=False).
    :param save_path: Path to save the trained model (default=None).
    :return: Trained XGBoost classifier model or RandomizedSearchCV object if do_grid_search=True.
    :rtype: object
    """

    if do_grid_search:
        xgbc = xgb_(random_state=42)

        J_COST_SURFACE = {"fbeta" : make_scorer(fbeta_score, beta=2, average="micro"),
                          "accu" : make_scorer(accuracy_score),
                          "bl" : make_scorer(balanced_accuracy_score),
                          "mcc" : make_scorer(matthews_corrcoef)}
        
        if isinstance(xgbc, xgb.XGBClassifier):
            # XGB requires the column names to be strings
            X_train.columns = [str(i) for i in X_train.columns]
            # And the column names should not contain any of the following characters: '[', ']', '<'
            X_train.columns = X_train.columns.str.replace('[', '')
            X_train.columns = X_train.columns.str.replace(']', '')
            X_train.columns = X_train.columns.str.replace('<', 'lt')
            # XGB uses other parameters than the sklearn's GradientBoostingClassifier
            param_dist = {
                "n_estimators": np.logspace(2, 3, 20, endpoint=True).astype(int).tolist(),
                "learning_rate": np.logspace(-3, 0, 10, endpoint=True).tolist(),
                "max_depth": np.linspace(3, 10, 8, endpoint=True, dtype=np.int64).tolist(),
                "gamma": [0, 0.1, 0.2, 0.3, 0.4],
                "lambda": np.logspace(-3, 0, 10, endpoint=True).tolist()
                # It has more parameters, but for the sake of simplicity, we will only consider these
                # https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
            }
        else:
            param_dist = {
                "n_estimators": np.logspace(2, 3, 20, endpoint=True).astype(int).tolist(),
                "learning_rate": np.logspace(-3, 0, 10, endpoint=True).tolist(),
                "max_depth": np.linspace(3, 10, 8, endpoint=True, dtype=np.int64).tolist(),
                "subsample": np.logspace(-3, 0, 10, endpoint=True).tolist(),
                "max_features": [None, "sqrt", "log2"],
                "min_impurity_decrease": [0]+np.logspace(-4, 0, 4, endpoint=False).tolist(),
                "ccp_alpha": np.logspace(-3, -1, 3, endpoint=False).tolist()
                }

        rnd_gs = RandomizedSearchCV(estimator=xgbc,
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
        xgbc = xgb_(n_estimators=1000,
                                         learning_rate=0.1,
                                         max_depth=3,
                                         subsample=0.9,
                                         max_features="sqrt",
                                         min_impurity_decrease=0,
                                         ccp_alpha=0.01,
                                         random_state=42)

        xgbc.fit(X_train, y_train.ravel())

        save_python_objects(xgbc, save_path)

        return xgbc
    
def glm_model(X_train, y_train, cv_k_folds=10, refit_J="accu", do_grid_search=False, save_path=None):
    glm = SMWrapper(sm.GLM)

    J_COST_SURFACE = {
        "fbeta": make_scorer(fbeta_score, beta=2, average="micro"),
        "accu": make_scorer(accuracy_score),
        "bl": make_scorer(balanced_accuracy_score),
        "mcc": make_scorer(matthews_corrcoef)
    }

    if do_grid_search:
        param_dist = {
            'family': [sm.families.Binomial(), sm.families.Gaussian(), sm.families.Gamma(), sm.families.Poisson()],
            'fit_intercept': [True, False],
        }

        grid_search = RandomizedSearchCV(estimator=glm,
                                         param_distributions=param_dist,
                                         n_iter=20,
                                         scoring=J_COST_SURFACE,
                                         cv=cv_k_folds,
                                         refit=refit_J,
                                         random_state=312,
                                         verbose=2)
        grid_search.fit(X_train, y_train)

        if save_path:
            save_python_objects(grid_search, save_path)

        return grid_search

    else:
        glm.fit(X_train, y_train)

        if save_path:
            save_python_objects(glm, save_path)

        return glm
    
