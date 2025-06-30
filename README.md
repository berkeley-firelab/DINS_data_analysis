# Damage Inspection (DINS) Data Analysis

This repository contains scripts for data preprocessing and analysis for Damage Inspection Data (DINS).

## Usage 

### Requirements

The implementation of this repository utilizes [conda](https://docs.conda.io/projects/conda/en/stable/) for environment management and [mamba](https://mamba.readthedocs.io/en/latest/index.html) for efficient package installation. 

To streamline the setup process, a configuration file is provided to create a virtual environment with all the necessary dependencies. 

If you already have conda and mamba installed, execute the following command to set up the required environment:

```
mamba env create -f resources/mamba_env.yml
```

### Models training

The `main.py` script facilitates model training by performing a grid search to optimize the parameters of the selected machine learning model. 

Before starting the training process, ensure that the `estimator_params.json` configuration file is available. An example of this file is provided in the `data` directory.

To initiate the training process, run the following command:

```
python main.py
```

## Utilities (`utils/`)

All reusable workflows and helper functions live in the `utils/` folder.  These modules simplify data loading, model retrieval, SHAP-value post-processing, and directory management.


| Module                          | Description                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------|
| `directory_structure.py`        | Defines standard project directories:  
                                    - `DATA_DIR`: where raw and intermediate data live  
                                    - `OUTPUT_DIR`: where model outputs, figures, and results are written                          |
| `preprocessing.py`              | Pipeline steps for data cleaning, imputation, scaling, encoding, and train/test splitting.      |
| `estimators.py`                 | Factory functions for instantiating and loading ML models from disk (e.g. `get_model()`).       |
| `custom_encoder.py`             | Wraps scikit-learn encoders to handle mixed categorical/numerical data, and invert transformations. |
| `helpers.py`                    | Miscellaneous utility functions (e.g. logging setup, JSON/YAML readers, environment checks).     |


### Quick start: loading data, model, and interpreting SHAP

```python
import os, json, pickle
import numpy as np, pandas as pd
import shap
from utils.directory_structure import DATA_DIR, OUTPUT_DIR
from utils.estimators       import get_model, get_data
from utils.helpers          import treat_encoded_shap_vals

# 1) Load your experiment configuration
with open(os.path.join(DATA_DIR, "estimator_params.json")) as f:
    est_dict = json.load(f)

# 2) Load pre-processed data
data_dict = get_data(est_dict)
X_train, y_train = data_dict["X_train"], data_dict["y_train"]
X_test,  y_test  = data_dict["X_test"],  data_dict["y_test"]

# 3) Load fitted model (from OUTPUT_DIR/<<MODEL_TYPE>>/)
lr_gs = get_model(est_dict)["grid_search"]
best_model = lr_gs.best_estimator_

# 4) Define prediction functions for SHAP
def model_log_odds(x):
    proba = best_model.predict_log_proba(x)
    return proba[:, 1] - proba[:, 0]

# 5) Explain predictions on test set
explainer        = shap.Explainer(model_log_odds, masker=X_test)
shap_values_exp  = explainer(X_test)

# 6) Adjust SHAP outputs to original categorical feature levels
shap_values, X_test_orig = treat_encoded_shap_vals(
    shap_values_exp, X_test, data_dict
)

# 7) Now you can plot or analyze:
shap.plots.bar(shap_values)

```
Details about the pre-processing and data in the sections below.

### Analysis

The analysis of the results is performed using Jupyter Notebooks, which require a Jupyter server to run. Jupyter is an open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text. For more information, visit the [Jupyter website](https://jupyter.org/).

To start the Jupyter server and run the notebooks, use the following command:

```
jupyter notebook
```

This will open the Jupyter interface in your default web browser, where you can navigate to the desired notebook and execute it.

All of the result‐generation and manuscript‐figure workflows are contained in Jupyter notebooks.  The primary notebook for the paper is:

- **Paper_Analysis.ipynb**  
  Contains all of the figures, tables, and narrative analyses referenced in the manuscript.
  
## data

* `dins-2017-2022`: This is the original dataset containing string values.
* `WUI_fires`: This directory contains pre-processed and original datasets for five WUI fires with NaN values. 

## dins and WUI datasets dictionary

### Categorical Variables

`DAMAGE`, `ROOFCONSTRUCTION`, `EXTERIORSIDING`, `WINDOWPANE`, `EAVES`, `VENTSCREEN`, `DECK/PORCH ON GRADE`, `DECKPORCHELEVATED`, `PATIOCOVER`, `FENCE`, `VSD`

## Numerical Variables:

`Distance`, `YEARBUILT`, `ZIPCODE`, `EMBER`, `FLAME`

## `dins`

* `DAMAGE`: ['Inaccessible', 'No Damage', 'Affected (1-9%)', 'Minor (10-25%)', 'Major (26-50%)', 'Destroyed (>50%)']
* `ROOFCONSTRUCTION`: ['Unknown', 'Fire Resistant', 'Metal', 'Concrete', 'Tile', 'Asphalt', 'Other', 'Wood', 'Combustible']
* `EXTERIORSIDING`: ['Unknown', 'Metal', 'Stucco Brick Cement', 'Stucco/Brick/Cement', 'Ignition Resistant', 'Fire Resistant', 'Vinyl', 'Other', 'Combustible', 'Wood']
* `WINDOWPANE`: ['Unknown', No Windows', 'Single Pane', 'Multi Pane']
* `EAVES`: ['Unknown', 'Not Applicable', 'No Eaves', 'Unenclosed', ‘'Enclosed']
* `VENTSCREEN`: ['Unknown', 'Screened', 'Mesh Screen <= 1/8"', 'Mesh Screen > 1/8"', 'Unscreened', 'No Vents']
* `DECK/PORCH ON GRADE`: ['Unknown', 'No Deck/Porch', 'Masonry/Concrete', 'Composite', 'Wood']
* `DECKPORCHELEVATED`: ['Unknown', 'No Deck/Porch', 'Masonry/Concrete', 'Composite', 'Wood']
* `PATIOCOVER (PATIOCOVERCARPORT)`: ['Unknown', 'No Patio Cover/Carport', 'Non Combustible', 'Combustible']
* `FENCE (FENCEATTACHEDTOSTRUCTURE)`:['Unknown', 'No Fence', 'Non Combustible', 'Combustible']
* `Distance`: Structure Separation Distance (SSD) in *feet*, Using the building footprints data set, the distance from one structure to its nearest neighboring structure is measured with the QGIS tools.
* `YEARBUILT`: Year that primary structure in parcel was constructed
* `ZIPCODE`: Zip Code

## `WUI fires`

In addition to the `dins` variables, we also have `VSD`, `EMBER`, and `FLAME` for our WUI fires:

* `VSD`: Vegetation Separation Distance (VSD), or Defensible Space, refers to the minimum distance between a structure and the surrounding vegetation, established through a buffer zone (Through an airborn LiDAR data for Sonoma County as well as Aerial/Street View Imagery). Zone 0= (0-5ft), Zone 1= (5-30ft), Zone 2= (30-100ft), and +100ft
  * 0 = Zone 0
  * 1 = Zone 1
  * 2 = Zone 2
  * 3 = +100ft
  * 4 = Not Applicable


* `EMBER and FLAME`= Generated by the WUI model

## Adopted data processing

The steps for data preprocessing are as follows:

1. Separate the data into train and test cases with 20% going to the test set.
2. Design imputation strategies, train and apply them to the train set, and fit to the test set.
To enable the use of a variety of models:
    - Normalize the numerical variables
    - Conduct `OneHotEncoding` on categorical variables
4. Resample to make the representation of all classes equal to in the train set.
5. If necessary do a `PCA` conversion
6. Put all steps into a pipeline under one function

### Imputation strategies

The strategy differs for each type of `categorical` and `numerical` feature and even within each category

*DINS*: Adopted strategy for features with missing values in samples,

- `ROOFCONSTRUCTION`  has `82817` non-null objects: Nearest neighbor imputation.
- `EAVES` has `82741` non-null objects: Nearest neighbor imputation.
- `VENTSCREEN` has `82692` non-null objects: Nearest neighbor imputation.
- `EXTERIORSIDING` has`82800` non-null objects: Nearest neighbor imputation.
- `WINDOWPANE` has `82732` non-null objects: Nearest neighbor imputation.
- `DECKPORCHONGRADE` has `70291` non-null objects: Nearest neighbor imputation.
- `DECKPORCHELEVATED` has `70290` non-null objects: Nearest neighbor imputation.
- `PATIOCOVER` has `70286` non-null objects: Nearest neighbor imputation.
- `FENCE` has `70289` non-null objects: Nearest neighbor imputation.
- `YEARBUILT` has `53075` non-null objects: Nearest neighbor imputation.

*Wildfire cases*: Adopted strategy for features with missing values in samples,

- `ZIPCODE` has `15` non-null floats: Reverse geo-encoding can be used if this is useful. Potentially for future studies. 
- `ROOFCONSTR` has `19318`  non-null samples: Nearest neighbor imputation
- `EAVES` has `19318`  non-null samples: Nearest neighbor imputation
- `VENTSCREEN` has `19318`  non-null samples: Nearest neighbor imputation
- `EXTERIORSI` has `19318`  non-null samples: Nearest neighbor imputation
- `WINDOWPANE` has `19318`  non-null samples: Nearest neighbor imputation
- `DECKPORCHO` has `19318`  non-null samples: Nearest neighbor imputation
- `DECKPORCHE` has `19318`  non-null samples: Nearest neighbor imputation
- `PATIOCOVER` has `19318`  non-null samples: Nearest neighbor imputation
- `FENCEATTAC` has `19317`  non-null samples: Nearest neighbor imputation
- `YEARBUILT ` has `22501`  non-null samples: Nearest neighbor imputation or median
- `VSD` has `3504 ` non-null  samples: Aggregate (mean, median, etc) potentially with KNN
- `EMBER` has `11549`  non-null samples: Aggregate (mean, median, etc) potentially with KNN
- `FLAME` has `14578`  non-null samples: Aggregate (mean, median, etc) potentially with KNN

## Workflow Types

Two types of workflows are provided:

* `ipyn`: Jupyter Notebook-based workflow.
* `py`: Python script-based workflow.
