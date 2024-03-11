import os
import json 

from utils.directory_structure import DATA_DIR, OUTPUT_DIR
from utils.preprocessing import data_preprocessing_pipeline, balance_classes
from utils.estimator import logit_model

with open(os.path.join(DATA_DIR, "estimator_params.json")) as f:
    est_dict = json.load(f)

MODEL_TYPE = est_dict["MODEL_TYPE"]
MODEL_VERSION = est_dict["MODEL_VERSION"]
MODEL_DIR = os.path.join(OUTPUT_DIR, MODEL_TYPE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

ML_MODEL_NAME = "{}_{}_using_{}_features.pkl".format(MODEL_TYPE, MODEL_VERSION, est_dict["FEATURE_TYPE"])
SAVE_PATH = os.path.join(MODEL_DIR, ML_MODEL_NAME)

if __name__ == "__main__":

    print("Data creation!")

    data_dict = data_preprocessing_pipeline(case_name=est_dict["DATA_CASE"], 
                                            renew_data=est_dict["RENEW_DATA"],
                                            encode_data=est_dict["ENCODE_DATA"],
                                            scale_data=est_dict["SCALE_DATA"],
                                            task_type=est_dict["TASK_TYPE"],
                                            weighted_classes=est_dict["WEIGHTED_CLASSES"])

    print("Data dictionary is created!")

    X = data_dict["X_train"]
    y = data_dict["y_train"]

    if est_dict["REBALANCE"]:
        X, y = balance_classes(
            X,
            y,
            strategy=est_dict["BALANCE_STRATEGY"],
            k_neighbors=est_dict["K_NS"],
            feature_type=est_dict["FEATURE_TYPE"],
        )
        print("Classes are balanced and start of training!")

    else:
        print("No class balancing is considered!")

    if y.shape[1] > 1:
        clf = logit_model(X, y, do_grid_search=est_dict["GRID_SEARCH"], save_path=SAVE_PATH)
    else:
        clf = logit_model(X, y.values.ravel(), do_grid_search=est_dict["GRID_SEARCH"], save_path=SAVE_PATH)
