import os

# directory structure
CU_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.join(CU_DIR, "..")
PROJ_DIR = os.path.abspath(PROJ_DIR)

# normalized paths
DATA_DIR = os.path.join(PROJ_DIR, "data")
OUTPUT_DIR = os.path.join(PROJ_DIR, "outputs")
ASSETS = os.path.join(PROJ_DIR, "assets")
RESOURCES = os.path.join(PROJ_DIR, "resources")

dir_list = [DATA_DIR, OUTPUT_DIR, ASSETS, RESOURCES]

for d in dir_list:
    if not os.path.exists(os.path.join(PROJ_DIR, d)):
        os.makedirs(d, exist_ok=True)