import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

processed_data_path = Path(
    "data",
    "processed_data",
)

data_path = Path(
    processed_data_path,
    "all_brain_features_resid_exc_sex.csv",
)

data = pd.read_csv(
    data_path,
    index_col=0,
    low_memory=False,
)

data_splits_path = Path(
    processed_data_path,
    "data_splits.json",
)

features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

with open(features_of_interest_path, "r") as f:
    brain_features_of_interest = json.load(f)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

    modality_data_splits = data_splits["structural"]

low_symp_subs = modality_data_splits["low_symptom_test"]
inter_test_subs = modality_data_splits["internalising_test"]
exter_test_subs = modality_data_splits["externalising_test"]
high_test_subs = modality_data_splits["high_symptom_test"]

feature_sets = {
    "t1w_cortical_thickness_rois": "cortical_thickness",
    "t1w_cortical_volume_rois": "cortical_volume",
    "t1w_cortical_surface_area_rois": "cortical_surface_area",
}

symptom_cohorts = {
    "inter": inter_test_subs,
    "exter": exter_test_subs,
    "high": high_test_subs,
}

low_entropy = True

if low_entropy:

    high_entropy_subs_path = Path(
        "data",
        "LCA",
        "subjects_with_high_entropy.csv",
    )

    high_entropy_subs = pd.read_csv(
        high_entropy_subs_path,
        low_memory=False,
    )["subject"].tolist()

    low_symp_subs = [sub for sub in low_symp_subs if sub not in high_entropy_subs]

    inter_test_subs = [sub for sub in inter_test_subs if sub not in high_entropy_subs]

    exter_test_subs = [sub for sub in exter_test_subs if sub not in high_entropy_subs]

    high_test_subs = [sub for sub in high_test_subs if sub not in high_entropy_subs]


# Loop over each cortical modality
for feature in feature_sets:

    features = brain_features_of_interest[feature]

    # Extract low symptom data
    low_symp_data = data.loc[low_symp_subs, features]

    # Standardize features
    scaler = StandardScaler()
    low_symp_data_scaled = pd.DataFrame(
        scaler.fit_transform(low_symp_data),
        index=low_symp_data.index,
        columns=low_symp_data.columns,
    )

    # Loop over each symptom cohort
    for cohort_name, cohort_subs in symptom_cohorts.items():

        # Extract symptom cohort data
        cohort_data = data.loc[cohort_subs, features]

        # Apply the same scaling
        cohort_data_scaled = pd.DataFrame(
            scaler.transform(cohort_data),
            index=cohort_data.index,
            columns=cohort_data.columns,
        )

        # Combine low_symp and symptom cohort data
        combined_data = pd.concat([low_symp_data_scaled, cohort_data_scaled])

        # Create target variable: 0 for low_symp, 1 for symptom cohort
        y = pd.Series(
            [0] * len(low_symp_data_scaled) + [1] * len(cohort_data_scaled),
            index=combined_data.index,
        )

        # Convert to numpy arrays
        X = combined_data.values
        y = y.values

        # Elastic Net Regression
        elastic_net_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=5,
            random_state=42,
            max_iter=10000,
        )

        elastic_net_cv.fit(X, y)

        # Get coefficients
        coefs = pd.Series(elastic_net_cv.coef_, index=features)

        # Identify significant features (non-zero coefficients)
        significant_features = coefs[coefs != 0]

        print(
            f"\nElastic Net Regression for {feature_sets[feature]} - {cohort_name} vs low_symp"
        )
        print(f"Optimal alpha: {elastic_net_cv.alpha_}")
        print(f"Optimal l1_ratio: {elastic_net_cv.l1_ratio_}")

        print(f"Number of significant features: {len(significant_features)}")
        print("Significant features and their coefficients:")
        print(significant_features.sort_values(key=abs, ascending=False))
