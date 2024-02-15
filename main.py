import os
import json 

from utils.directory_structure import DATA_DIR
from utils.preprocessing import data_preprocessing_pipeline, balance_classes
from estimator import logit_model

from IPython import embed

with open(os.path.join(DATA_DIR, "dins_estimator_params.json")) as f:
    est_dict = json.load(f)



if __name__ == "__main__":

    print("Data creation!")
    os.environ['OMP_NUM_THREADS'] = '60'

    data_dict = data_preprocessing_pipeline(case_name=est_dict["DATA_CASE"], 
                                            new_features=est_dict["GENERATE_PKL_DATA"],
                                            encode_data=est_dict["ENCODE_DATA"],
                                            scale_data=est_dict["SCALE_DATA"],
                                            weighted_classes=est_dict["WEIGHTED_CLASSES"])
    
    print("Data dictionary is created!")

    X = data_dict["X_train"]
    y = data_dict["y_train"]

    # TODO: implement an if statment to whether balance the dataset
    X, y = balance_classes(X, y, 
                           strategy=est_dict["BALANCE_STRATEGY"],
                           k_neighbors=est_dict["K_NS"], 
                           mixed_features=est_dict["MIXED_TYPE"])
    
    clf = logit_model(X, y)
    