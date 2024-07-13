from collections import defaultdict
from pathlib import Path

import pandas as pd
from discover.scripts.test_utils import cliffs_delta
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

bootstrap_num = 2
low_entropy = True
metric = "reconstruction_deviation"

feature_effect_sizes_CIs = {}

metric_lists = [metric] + [f"{metric}_{i}" for i in range(148)]

for feature in features:

    feature_bootstrap_result_path = Path(
        bootstrap_result_path,
        f"{feature}_bootstrap_results.csv",
    )

    # Re-initialize metric_lists at the beginning of each iteration
    if feature == "rsfmri":
        metric = "reconstruction_deviation"
        metric_lists = [metric] + [f"{metric}_{i}" for i in range(143)]

    # Debug print to check the metric_lists initialization
    print(f"Initialized metric_lists for {feature}: {metric_lists}")

    effect_size_CIs = {metric: defaultdict(list) for metric in metric_lists}

    print(effect_size_CIs)

    bootstrap_result = pd.read_csv(
        feature_bootstrap_result_path,
        index_col=0,
    )

    for i in tqdm(range(bootstrap_num), desc=f"Processing {feature}"):

        output_data = bootstrap_result[bootstrap_result["bootstrap_num"] == i]

        if low_entropy:
            # Remove subjects with high entropy
            high_entropy_subs_path = Path(
                "data",
                "LCA",
                "subjects_with_high_entropy.csv",
            )

            high_entropy_subs = pd.read_csv(
                high_entropy_subs_path,
                low_memory=False,
            )["subject"].tolist()

            output_data = output_data[~output_data.index.isin(high_entropy_subs)]

        metric_groups_effect_sizes = {}

        for metric in metric_lists:

            print(metric)

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

output_path = Path(
    bootstrap_result_path,
    "ind_recon_dev_effect_size_CIs.json",
)