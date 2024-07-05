import json
from pathlib import Path

processed_data_path = Path(
    "data",
    "processed_data",
)

data_splits_path = Path(
    processed_data_path,
    "data_splits.json",
)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

t1w_data_splits = data_splits["structural"]

rsfmri_data_splits = data_splits["functional"]


for key, value in t1w_data_splits.items():
    print(f"Length of {key} in t1w_data_splits: {len(value)}")

print("\n")

for key, value in rsfmri_data_splits.items():
    print(f"Length of {key} in rsfmri_data_splits: {len(value)}")
