# TODO: Check if the site effect on different strata of covariate values is identical
# TODO: Check the sample size of each site by the covariates
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import StratifiedKFold


def check_Combat():
    processed_data_path = Path(
        "data",
        "processed_data",
    )

    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "r") as f:
        brain_features_of_interest = json.load(f)

    all_brain_features_precon_path = Path(
        processed_data_path,
        "all_brain_features_precon.csv",
    )

    all_brain_features_precon = pd.read_csv(
        all_brain_features_precon_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_features_resid_path = Path(
        processed_data_path,
        "t1w_cortical_features_resid_exc_sex.csv",
    )

    t1w_cortical_features_resid = pd.read_csv(
        t1w_cortical_features_resid_path,
        index_col=0,
        low_memory=False,
    )

    gordon_cor_subcortical_resid_path = Path(
        "data",
        "processed_data",
        "gordon_cor_subcortical_resid_exc_sex.csv",
    )

    gordon_cor_subcortical_resid = pd.read_csv(
        gordon_cor_subcortical_resid_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_thickness_rois = brain_features_of_interest[
        "t1w_cortical_thickness_rois"
    ]
    t1w_cortical_volume_rois = brain_features_of_interest["t1w_cortical_volume_rois"]

    t1w_cortical_surface_area_rois = brain_features_of_interest[
        "t1w_cortical_surface_area_rois"
    ]

    gordon_net_subcor_no_dup = brain_features_of_interest["gordon_net_subcor_no_dup"]

    # Statistically estimate the effect of site on intracranial volume
    intracranialV_on_site(all_brain_features_precon)

    # Statistically test the effect of site on predicted class membership
    site_on_latent_class(all_brain_features_precon)

    # Check the sample size of each site by the covariates
    check_sample_size_per_site(all_brain_features_precon)

    # Test site effect after NeuroCombat

    print(
        "Cortical Thickness Mean MCC is ",
        test_site_effect(t1w_cortical_features_resid, t1w_cortical_thickness_rois),
    )
    print(
        "Cortical Volume Mean MCC is ",
        test_site_effect(t1w_cortical_features_resid, t1w_cortical_volume_rois),
    )

    print(
        "Cortical Surface Area Mean MCC is ",
        test_site_effect(t1w_cortical_features_resid, t1w_cortical_surface_area_rois),
    )

    print(
        "Rsfmri Mean MCC is ",
        test_site_effect(gordon_cor_subcortical_resid, gordon_net_subcor_no_dup),
    )


def intracranialV_on_site(df):
    # Convert categorical variable 'label_site' to dummy variables
    df = pd.get_dummies(df, columns=["label_site"], drop_first=True)

    # Rename columns to ensure they are valid variable names
    df.columns = [col.replace(" ", "_").replace(".", "_") for col in df.columns]

    # Define the formula for the linear regression
    formula = "smri_vol_scs_intracranialv ~" + " + ".join(
        [col for col in df.columns if col.startswith("label_site_")]
    )

    # formula = (
    #     "smri_vol_scs_intracranialv ~ interview_age + demo_sex_v2 + "
    #     + " + ".join([col for col in df.columns if col.startswith("label_site_")])
    # )

    # Fit the linear regression model
    model = smf.ols(formula, data=df).fit()

    # Print the model summary
    print(model.summary())


def site_on_latent_class(df):

    predicted_class = df["predicted_class"]

    label_site = df["label_site"]

    # Drop the sites with small number of subjects and see if there are still significant relationship
    selected_rows = df[df["label_site"].isin([16.0, 4.0, 6.0, 27.0])]

    predicted_class = predicted_class.drop(selected_rows.index)
    label_site = label_site.drop(selected_rows.index)

    # Create a contingency table
    contingency_table = pd.crosstab(predicted_class, label_site)

    # Perform the Chi-Square test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results
    print("Chi-Square Test Statistic:", chi2)
    print("P-Value:", p)
    print("Degrees of Freedom:", dof)
    print("Expected Frequencies:\n", expected)

    # Interpret the result
    if p < 0.05:
        print(
            "There is a significant relationship between predicted_class and label_site (reject H0)"
        )
    else:
        print(
            "There is no significant relationship between predicted_class and label_site (fail to reject H0)"
        )


def check_sample_size_per_site(df):
    """Checks the sample size of each site by the covariates, sex, predicted_class,
    household income."""
    # Ensure that the necessary columns are present in the dataframe
    required_columns = [
        "label_site",
        "demo_sex_v2",
        "predicted_class",
        "demo_comb_income_v2",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataframe")

    # Calculate sample size per site by sex
    sex_counts = df.groupby(["label_site", "demo_sex_v2"]).size().unstack(fill_value=0)
    print("Sample size per site by sex:")
    print(sex_counts)
    print("\n")

    # Calculate sample size per site by predicted class
    class_counts = (
        df.groupby(["label_site", "predicted_class"]).size().unstack(fill_value=0)
    )
    print("Sample size per site by predicted class:")
    print(class_counts)
    print("\n")

    # Calculate sample size per site by household income
    income_counts = (
        df.groupby(["label_site", "demo_comb_income_v2"]).size().unstack(fill_value=0)
    )
    print("Sample size per site by household income:")
    print(income_counts)
    print("\n")


def test_site_effect(df, feature):

    X = np.asarray(df[feature])
    y = np.asarray(df["label_site"])
    skf = StratifiedKFold(n_splits=5)
    gen_split = skf.split(X, y)
    list_mcc = []
    for i, (train_index, test_index) in enumerate(gen_split):
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        RF = RFC()
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)
        print(mcc(y_test, y_pred))
        list_mcc.append(mcc(y_test, y_pred))

    # print("Mean MCC is ", np.mean(np.asarray(list_mcc)))

    return np.mean(np.asarray(list_mcc))


if __name__ == "__main__":
    check_Combat()
