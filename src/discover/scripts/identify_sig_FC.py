import json
from pathlib import Path

import numpy as np
import pandas as pd

brain_features_of_interest_path = Path(
    "data",
    "processed_data",
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as file:
    brain_features_of_interest = json.load(file)


bootstrap_effect_size_path = Path(
    "src",
    "discover",
    "results",
    "bootstrap",
    "ind_recon_dev_effect_size_CIs.json",
)

with open(bootstrap_effect_size_path, "r") as f:
    bootstrap_effect_size = json.load(f)

rsfmri_bootstrap_effect_size = {
    "rsfmri": bootstrap_effect_size["rsfmri"],
}

modality_map = {
    "rsfmri": "gordon_net_subcor_limbic_no_dup",
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
for modality, metrics in rsfmri_bootstrap_effect_size.items():

    input_features = brain_features_of_interest[modality_map[modality]]

    sig_metric_lists = []
    ci_low_effect_sizes = []
    ci_high_effect_sizes = []
    mean_effect_sizes = []
    group_names = []
    modality_names = []
    var_names = []

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

                var_name = var_dict[var_dict["var_name"] == sig_brain_feature][
                    "var_label"
                ].values[0]

                var_names.append(var_name)

    print(
        len(modality_names),
        len(sig_metric_lists),
        len(mean_effect_sizes),
        len(group_names),
        len(brain_features),
        len(var_names),
    )

    sig_metric_df = pd.DataFrame(
        {
            "Modality": modality_names,
            "Metric": sig_metric_lists,
            "CI Lower": ci_low_effect_sizes,
            "CI Upper": ci_high_effect_sizes,
            "Mean Effect Size": mean_effect_sizes,
            "Group": group_names,
            # Named brain region here for consistency with cortical features
            "Brain Region": brain_features,
            "Variable Name": var_names,
        }
    )

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


# Print the results
# for modality, metrics in significant_results.items():
#     for metric, groups in metrics.items():
#         print(
#             f"For modality {modality}, metric {metric} is statistically significant for groups: {groups}"
#         )
