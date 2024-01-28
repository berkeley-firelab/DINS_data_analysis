"""
Explanation of the code to be wirtten here
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd

from utils.directory_structure import DATA_DIR

# reading all files and creating access through a global dictionary
files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
DATA_DICT = {}

for f in files:
    file_name = f.split("/")[-1].split(".")[0]
    DATA_DICT[file_name] = pd.read_csv(f)

# case or cases to be studied or included in the training and testing
CASE_STUDY = ["dins_2017_2022"]
df = DATA_DICT[CASE_STUDY]

"""# Missing values

For all DINS
"""

pd.DataFrame(df.isna().sum(), columns=['Missing values']).sort_values(by='Missing values', ascending=False)

"""## Start processing

Backup to avoid load the data again...
"""

df_filter = df.copy() # A copy to work with

df_filter.drop('distance_meter', inplace=True, axis=1)

"""### Column analysis

#### Damage

Convert values to classes.
"""

df_filter['DAMAGE'].unique()

damage_labels = ['Inaccessible', 'No Damage', 'Affected (1-9%)', 'Minor (10-25%)', 'Major (26-50%)', 'Destroyed (>50%)']
damage_values = [np.nan, 0, 1, 1, 1, 2]
df_filter['DAMAGE'].replace(
    damage_labels,
    damage_values,
    inplace=True
)

"""#### Latitude and Longitude

Nothing to do with them.
"""

df_filter['LATITUDE'].describe()

df_filter['LONGITUDE'].describe()

"""#### DECKPORCHONGRADE

Text to categorical values.
"""

df_filter['DECKPORCHONGRADE'].unique()

"""There are some `' '` values. We will replace them by NaN.

"""

deck_porch_labels = ['Unknown', 'No Deck/Porch', 'Masonry/Concrete', 'Composite', 'Wood', ' ']
deck_porch_values = [np.nan, 0, 1, 2, 3, np.nan]
df_filter['DECKPORCHONGRADE'].replace(
    deck_porch_labels,
    deck_porch_values,
    inplace=True
)

print(df_filter['DECKPORCHONGRADE'].unique())

"""#### DECKPORCHELEVATED

Text to categorical values.
"""

df_filter['DECKPORCHELEVATED'].unique()

"""There are some `' '` values. We will replace them by NaN."""

deck_porch_elev_labels = ['Unknown', 'No Deck/Porch', 'Composite', 'Masonry/Concrete', 'Wood', ' ']
deck_porch_elev_values = [np.nan, 0, 1, 2, 3, np.nan]
df_filter['DECKPORCHELEVATED'].replace(
    deck_porch_elev_labels,
    deck_porch_elev_values,
    inplace=True
)

"""#### PATIOCOVER

Text to categorical values.
"""

df_filter['PATIOCOVER'].unique()

"""There are some `' '` values. We will replace them by NaN."""

patio_labels = ['Unknown', 'No Patio Cover/Carport', 'Non Combustible', 'Combustible', ' ']
patio_values = [np.nan, 0, 1, 2, np.nan]
df_filter['PATIOCOVER'].replace(
    patio_labels,
    patio_values,
    inplace=True
)

"""#### FENCE

Text to categorical values.
"""

df_filter['FENCE'].unique()

"""There are some `''` values. We will replace them by NaN."""

fence_labels = ['Unknown', 'No Fence', 'Non Combustible', 'Combustible', '']
fence_values = [np.nan, 0, 1, 2, np.nan]
df_filter['FENCE'].replace(
    fence_labels,
    fence_values,
    inplace=True
)

"""### ROOFCONSTRUCTION

Text to integer
"""

df_filter['ROOFCONSTRUCTION'].unique()

roof_construction_labels = ['Unknown', 'Fire Resistant', 'Metal', 'Concrete', 'Tile', 'Asphalt','Other','Wood', 'Combustible', ' ']
roof_construction_values = [np.nan, 0, 0, 0, 1, 2, 3 , 4, 4, np.nan]
df_filter['ROOFCONSTRUCTION'].replace(
    roof_construction_labels,
    roof_construction_values,
    inplace=True
)

np.sort(df_filter['ROOFCONSTRUCTION'].unique())

"""### EXTERIORSIDING

Text to int
"""

df_filter['EXTERIORSIDING'].unique()

exterior_siding_labels = ['Unknown', 'Metal', 'Stucco Brick Cement', 'Stucco/Brick/Cement','Ignition Resistant','Fire Resistant', 'Vinyl', 'Other','Combustible','Wood', ' ']
exterior_siding_values = [ np.nan, 0, 1, 1, 0, 0, 2, 3, 4, 4, np.nan,]
df_filter['EXTERIORSIDING'].replace(
    exterior_siding_labels,
    exterior_siding_values,
    inplace=True
)

np.sort(df_filter['EXTERIORSIDING'].unique())

"""### WINDOWPANE

Text to int
"""

df_filter['WINDOWPANE'].unique()

windows_pane_labels = ['Unknown', 'No Windows', 'Single Pane', 'Multi Pane', ' ']
windows_pane_values = [np.nan, 0, 1, 2, np.nan]
df_filter['WINDOWPANE'].replace(
    windows_pane_labels,
    windows_pane_values,
    inplace=True
)

np.sort(df_filter['WINDOWPANE'].unique())

"""### EAVES

Text to int
"""

df_filter['EAVES'].unique()

eaves_labels = ['Unknown', 'Not Applicable', 'No Eaves', 'Unenclosed', 'Enclosed', ' ']
eaves_values = [np.nan, 0, 1, 2, 3, np.nan]
df_filter['EAVES'].replace(
    eaves_labels,
    eaves_values,
    inplace=True
)

np.sort(df_filter['EAVES'].unique())

"""### VENTSCREEN

Text to integer
"""

df_filter['VENTSCREEN'].unique()

vent_screen_labels = ['Unknown', 'Screened', 'Mesh Screen <= 1/8"', 'Mesh Screen > 1/8"', 'Unscreened', 'No Vents', ' ']
vent_screen_values = [np.nan, 0, 1, 2, 3, 4, np.nan]
df_filter['VENTSCREEN'].replace(
    vent_screen_labels,
    vent_screen_values,
    inplace=True
)

np.sort(df_filter['VENTSCREEN'].unique())

"""### YEARBUILT

It's a number, nothing to do.
"""

df_filter['YEARBUILT'].unique()

"""Replacing value $0$ to NaN."""

df_filter['YEARBUILT'].replace(
    [0],
    [np.nan],
    inplace=True
)

df_filter.dtypes

df_filter.sample(10)

df_filter.to_pickle(DIR + 'dins_2017_2022.pkl')

