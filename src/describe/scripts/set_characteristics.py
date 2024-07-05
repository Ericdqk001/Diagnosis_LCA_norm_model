from pathlib import Path

import pandas as pd

all_brain_features_path = Path(
    "data",
    "processed_data",
    "all_brain_features_resid_exc_sex.csv",
)

all_brain_features = pd.read_csv(
    all_brain_features_path,
    index_col=0,
)
