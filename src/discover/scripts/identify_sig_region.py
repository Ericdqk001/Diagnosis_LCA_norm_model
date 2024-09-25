import json
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets

# %%
### To get the roi names
destrieux_2009 = datasets.fetch_atlas_destrieux_2009(legacy_format=False)

labels = destrieux_2009.labels

### NOTE index 42 and 116 are not present in the ABCD dextrieux atlas so are removed ('L Medial_wall' and 'R Medial_wall')
labels_dropped = labels.drop(index=42).reset_index(drop=True)
labels_dropped = labels_dropped.drop(index=116).reset_index(drop=True)


brain_features_of_interest_path = Path(
    "data",
    "processed_data",
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as file:
    brain_features_of_interest = json.load(file)

low_entropy = False

if low_entropy:

    bootstrap_effect_size_path = Path(
        "src",
        "discover",
        "bootstrap_effect_size",
        "ind_recon_dev_effect_size_CIs.json",
    )

else:

    bootstrap_effect_size_path = Path(
        "src",
        "discover",
        "bootstrap_effect_size",
        "ind_recon_dev_effect_size_CIs_original.json",
    )

with open(bootstrap_effect_size_path, "r") as f:
    bootstrap_effect_size = json.load(f)

modality_map = {
    "cortical_thickness": "t1w_cortical_thickness_rois",
    "cortical_volume": "t1w_cortical_volume_rois",
    "cortical_surface_area": "t1w_cortical_surface_area_rois",
    # "rsfmri": "gordon_net_subcor_limbic_no_dup",
}


# Function to calculate the 99% confidence interval using percentiles
def calculate_confidence_interval(data, confidence=1):
    lower_percentile = (1.0 - confidence) / 2.0 * 100
    upper_percentile = (1.0 + confidence) / 2.0 * 100
    mean_value = np.mean(data)
    ci_lower = np.percentile(data, lower_percentile)
    ci_upper = np.percentile(data, upper_percentile)
    return ci_lower, ci_upper, mean_value


# Iterate through the JSON structure
for modality, metrics in bootstrap_effect_size.items():

    if modality == "rsfmri":
        continue

    input_features = brain_features_of_interest[modality_map[modality]]

    sig_metric_lists = []
    ci_low_effect_sizes = []
    ci_high_effect_sizes = []
    mean_effect_sizes = []
    group_names = []
    modality_names = []
    var_names = []
    brain_regions = []
    brain_features = []

    var_dict_path = Path(
        "data",
        "var_dict",
        f"{modality}_var_table.csv",
    )

    var_dict = pd.read_csv(var_dict_path)

    for metric, groups in metrics.items():

        if metric.split("_")[-1] == "deviation":
            continue

        for group, values in groups.items():
            ci_low, ci_high, mean_value = calculate_confidence_interval(values)
            # Check if the CI does not cover 0
            if ci_low > 0:
                sig_metric_lists.append(metric)
                ci_low_effect_sizes.append(ci_low)
                ci_high_effect_sizes.append(ci_high)
                mean_effect_sizes.append(mean_value)
                group_names.append(group)
                modality_names.append(modality)

                sig_brain_feature = input_features[int(metric.split("_")[-1])]

                brain_features.append(sig_brain_feature)

                sig_brain_region = labels_dropped.iloc[int(metric.split("_")[-1]) + 1][
                    "name"
                ]

                brain_regions.append(sig_brain_region)

                var_name = var_dict[var_dict["var_name"] == sig_brain_feature][
                    "var_label"
                ].values[0]

                print(var_name)

                var_names.append(var_name)

    sig_metric_df = pd.DataFrame(
        {
            "Modality": modality_names,
            "Metric": sig_metric_lists,
            "CI Lower": ci_low_effect_sizes,
            "CI Upper": ci_high_effect_sizes,
            "Mean Effect Size": mean_effect_sizes,
            "Group": group_names,
            "Brain Feature": brain_features,
            "Brain Region": brain_regions,
            "Variable Name": var_names,
        }
    )

    if low_entropy:

        sig_metric_df.to_csv(
            Path(
                "src",
                "discover",
                "results",
                "low_entropy",
                "sig_ind_regions",
                f"{modality}_significant_regions.csv",
            ),
            index=False,
        )

    else:

        sig_metric_df.to_csv(
            Path(
                "src",
                "discover",
                "results",
                "sig_ind_regions",
                f"{modality}_significant_regions.csv",
            ),
            index=False,
        )

# Print the results
# for modality, metrics in significant_results.items():
#     for metric, groups in metrics.items():
#         print(
#             f"For modality {modality}, metric {metric} is statistically significant for groups: {groups}"
#         )
