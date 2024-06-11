import json
from pathlib import Path

import pandas as pd
from nilearn import datasets
from statsmodels.stats.multitest import multipletests

# %%
### To get the roi names
destrieux_2009 = datasets.fetch_atlas_destrieux_2009(legacy_format=False)

labels = destrieux_2009.labels

### NOTE index 42 and 116 are not present in the ABCD dextrieux atlas so are removed ('L Medial_wall' and 'R Medial_wall')
labels_dropped = labels.drop(index=42).reset_index(drop=True)
labels_dropped = labels_dropped.drop(index=116).reset_index(drop=True)


# Get brain features of interest

brain_features_of_interest_path = Path(
    "data",
    "processed_data",
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as file:
    brain_features_of_interest = json.load(file)


# Get brain features variable label

brain_features_variable_label_path = Path(
    "data",
    "var_dict",
    "cortical_thickness_var_table.csv",
)

brain_features_variable_label = pd.read_csv(
    brain_features_variable_label_path,
)

low_entropy = True

# Get effect size
ind_brain_region_effect_size_path = Path(
    "src",
    "discover",
    "results",
    "ind_brain_region_U_test_results",
    "ind_brain_region_U_test_results.csv",
)

if low_entropy:
    ind_brain_region_effect_size_path = Path(
        "src",
        "discover",
        "results",
        "low_entropy",
        "ind_brain_region_U_test_results",
        "ind_brain_region_U_test_results.csv",
    )

ind_brain_region_effect_size = pd.read_csv(
    ind_brain_region_effect_size_path,
    index_col=0,
)

clinical_cohorts = [
    "inter_test",
    "exter_test",
    "high_test",
]

cortical_feature = "t1w_cortical_thickness_rois"

brain_features = brain_features_of_interest[cortical_feature]

for cohort in clinical_cohorts:
    cohort_ind_brain_region_effect_size = ind_brain_region_effect_size[
        ind_brain_region_effect_size["Cohort"] == cohort
    ]

    # Identify the regions with significant effect sizes
    # Apply FDR correction

    p_values = cohort_ind_brain_region_effect_size["P_value"]
    _, fdr_corrected_p_values, _, _ = multipletests(p_values, method="fdr_bh")

    cohort_ind_brain_region_effect_size["FDR_P_value"] = fdr_corrected_p_values

    significant_effect_size_metrics = cohort_ind_brain_region_effect_size[
        (cohort_ind_brain_region_effect_size["Effect_size"] > 0)
        & (cohort_ind_brain_region_effect_size["FDR_P_value"] < 0.05)
    ]["metric"].to_list()

    significant_effect_sizes = cohort_ind_brain_region_effect_size[
        (cohort_ind_brain_region_effect_size["Effect_size"] > 0)
        & (cohort_ind_brain_region_effect_size["FDR_P_value"] < 0.05)
    ]["Effect_size"].to_list()

    positive_brain_features = [
        brain_features[int(metric.split("_")[-1])]
        for metric in significant_effect_size_metrics
    ]

    positive_brain_region_indices = [
        int(positive_brain_features.split("_")[-1])
        for positive_brain_features in positive_brain_features
    ]

    brain_var_names = brain_features_variable_label[
        brain_features_variable_label["var_name"].isin(positive_brain_features)
    ]["var_label"].tolist()

    positive_brain_regions = labels_dropped.iloc[positive_brain_region_indices][
        "name"
    ].tolist()

    brain_regions = pd.DataFrame(
        {
            "brain_features": positive_brain_features,
            "brain_regions": positive_brain_regions,
            "variable_label": brain_var_names,
            "effect_size": significant_effect_sizes,
        }
    )

    save_path = Path(
        "src",
        "interpret",
        "results",
        f"{cortical_feature}",
    )

    if low_entropy:
        save_path = Path(
            save_path,
            "low_entropy",
        )

    if not save_path.exists():
        save_path.mkdir(parents=True)

    brain_regions.to_csv(
        Path(
            save_path,
            f"{cohort}_positive_brain_regions.csv",
        ),
        index=False,
    )
