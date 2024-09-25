import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

bootstrap_result_path = Path(
    "src",
    "discover",
    "results",
    "bootstrap",
)

features = [
    "cortical_thickness",
    "cortical_volume",
    "cortical_surface_area",
    "rsfmri",
]


def cliffs_delta(x, y):
    """Compute Cliff's Delta.
    x: Test group deviations.
    y: Control group deviations.
    """
    n, m = len(x), len(y)
    more = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    delta = (more - less) / (n * m)
    return delta


bootstrap_num = 1000
low_entropy = False
metric = "reconstruction_deviation"

metric_lists = [metric] + [f"{metric}_{i}" for i in range(148)]

feature_effect_sizes_CIs = {}

for feature in features:

    feature_bootstrap_result_path = Path(
        bootstrap_result_path,
        f"{feature}_bootstrap_results.csv",
    )
    print(f"Checking if file exists: {feature_bootstrap_result_path}")

    if feature_bootstrap_result_path.exists():
        print("File exists.")
    else:
        print("File does not exist.")

    # Re-initialize metric_lists at the beginning of each iteration
    if feature == "rsfmri":
        metric = "reconstruction_deviation"
        metric_lists = [metric] + [f"{metric}_{i}" for i in range(143)]

    effect_size_CIs = {metric: defaultdict(list) for metric in metric_lists}
    bootstrap_result = pd.read_csv(
        feature_bootstrap_result_path,
        index_col=0,
    )

    print(bootstrap_result.head())

    for i in tqdm(range(bootstrap_num), desc=f"Processing {feature}"):

        output_data = bootstrap_result[bootstrap_result["bootstrap_num"] == i]

        if low_entropy == True:
            # Remove subjects with high entropy
            high_entropy_subs_path = Path(
                bootstrap_result_path,
                "processed_data",
                "subjects_with_high_entropy.csv",
            )

            high_entropy_subs = pd.read_csv(
                high_entropy_subs_path,
                low_memory=False,
            )["subject"].tolist()

            output_data = output_data[~output_data.index.isin(high_entropy_subs)]

        metric_groups_effect_sizes = {}

        for metric in metric_lists:

            control_deviation = output_data[metric][
                output_data["low_symp_test_subs"] == 1
            ].values
            inter_test_deviation = output_data[metric][
                output_data["inter_test_subs"] == 1
            ].values
            exter_test_deviation = output_data[metric][
                output_data["exter_test_subs"] == 1
            ].values
            high_test_deviation = output_data[metric][
                output_data["high_test_subs"] == 1
            ].values

            # Define the groups to test against control
            test_groups = {
                "inter_test": inter_test_deviation,
                "exter_test": exter_test_deviation,
                "high_test": high_test_deviation,
            }

            for group, deviations in test_groups.items():
                effect_size = cliffs_delta(deviations, control_deviation)

                effect_size_CIs[metric][group].append(effect_size)

    feature_effect_sizes_CIs[feature] = effect_size_CIs

if low_entropy == True:
    output_path = Path(
        bootstrap_result_path,
        "ind_recon_dev_effect_size_CIs.json",
    )

else:
    output_path = Path(
        bootstrap_result_path,
        "ind_recon_dev_effect_size_CIs_original.json",
    )

with open(output_path, "w") as f:
    json.dump(
        feature_effect_sizes_CIs,
        f,
        indent=4,
    )
