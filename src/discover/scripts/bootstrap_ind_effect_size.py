from collections import defaultdict
from pathlib import Path

import pandas as pd
from discover.scripts.test_utils import cliffs_delta

bootstrap_result_paths = Path(
    "src",
    "discover",
    "results",
    "bootstrap",
).glob("*.csv")


bootstrap_num = 10
low_entropy = True
metric = "reconstruction_deviation"
metric_lists = [metric] + [f"{metric}_{i}" for i in range(148)]

effect_size_CIs = {metric: defaultdict(list) for metric in metric_lists}

for bootstrap_result_path in bootstrap_result_paths:

    bootstrap_result = pd.read_csv(
        bootstrap_result_path,
        index_col=0,
    )

    for i in range(bootstrap_num):

        output_data = bootstrap_result[bootstrap_result["bootstrap_num"] == i]

        if low_entropy == True:
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

            # Extract mahalanobis distances for each group
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
