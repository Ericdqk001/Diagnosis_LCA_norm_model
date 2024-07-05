from pathlib import Path

import pandas as pd

features_set = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "rsfmri",
]

tune_resuls_path = Path(
    "src",
    "modelling",
    "tune",
    "ucl_cluster_tune",
    "cVAE",
    "tune_results_v2",
)

for feature in features_set:

    cVAE_volume_results = Path(
        tune_resuls_path,
        f"cVAE_{feature}_UCL_hyper_tune_results.csv",
    )

    cVAE_volume_results = pd.read_csv(
        cVAE_volume_results,
        index_col=0,
        low_memory=False,
    )

    # Find the index of the row with the maximum average_separation
    max_average_separation_index = cVAE_volume_results["average_val_loss"].idxmin()

    # Retrieve the corresponding config value
    average_val_loss = cVAE_volume_results.loc[
        max_average_separation_index, "average_val_loss"
    ]

    best_config = cVAE_volume_results.loc[max_average_separation_index, "config"]

    print(f"Results for {feature} features:")

    print(
        f"The best config is {best_config} with average validation loss of {average_val_loss}"
    )
