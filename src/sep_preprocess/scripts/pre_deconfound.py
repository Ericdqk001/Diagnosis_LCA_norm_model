# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import LabelEncoder


def pre_deconfound_image_exc_sex():

    # %%
    processed_data_path = Path(
        "data",
        "processed_data",
    )

    LCA_path = Path(
        "data",
        "LCA",
    )

    core_data_path = Path(
        "data",
        "raw_data",
        "core",
    )

    general_info_path = Path(
        core_data_path,
        "abcd-general",
    )

    demographics_path = Path(
        general_info_path,
        "abcd_p_demo.csv",
    )

    demographics = pd.read_csv(
        demographics_path,
        index_col=0,
        low_memory=False,
    )

    demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

    family_income = demographics_bl["demo_comb_income_v2"].copy()

    family_income = family_income.replace(777, 999)

    imaging_path = Path(
        core_data_path,
        "imaging",
    )

    # Load cbcl-LCA data
    cbcl_LCA_path = Path(
        LCA_path,
        "cbcl_class_member_prob.csv",
    )

    cbcl_LCA = pd.read_csv(
        cbcl_LCA_path,
        index_col=0,
        low_memory=False,
    )

    # %%
    ### Perform neuroCombat to harmonize the imaging data

    # from neuroCombat import neuroCombat

    # For interview_age (in months)
    abcd_y_lt_path = Path(
        general_info_path,
        "abcd_y_lt.csv",
    )

    abcd_y_lt = pd.read_csv(
        abcd_y_lt_path,
        index_col=0,
        low_memory=False,
    )

    abcd_y_lt_bl = abcd_y_lt[abcd_y_lt.eventname == "baseline_year_1_arm_1"]

    # For biological sex (demo_sex_v2)

    demographics_bl.demo_sex_v2.value_counts()

    # For site information
    mri_y_adm_info_path = Path(
        imaging_path,
        "mri_y_adm_info.csv",
    )

    mri_y_adm_info = pd.read_csv(
        mri_y_adm_info_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_adm_info_bl = mri_y_adm_info[
        mri_y_adm_info.eventname == "baseline_year_1_arm_1"
    ]

    # mri_y_adm_info_bl.mri_info_deviceserialnumber.unique()

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info_bl["mri_info_deviceserialnumber"])
    mri_y_adm_info_bl["label_site"] = label

    # For smri_vol_scs_intracranialv (intracranial volume)
    mri_y_smr_vol_aseg_path = Path(
        imaging_path,
        "mri_y_smr_vol_aseg.csv",
    )

    mri_y_smr_vol_aseg = pd.read_csv(
        mri_y_smr_vol_aseg_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_aseg_bl = mri_y_smr_vol_aseg[
        mri_y_smr_vol_aseg.eventname == "baseline_year_1_arm_1"
    ]

    # mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv.describe()

    # Combine covariates together (interview_age, intra cranial volume, and site)
    # List of covariates to combine
    series_list = [
        demographics_bl.demo_sex_v2,
        cbcl_LCA.predicted_class,
        mri_y_adm_info_bl.label_site,
        mri_y_smr_vol_aseg_bl.smri_vol_scs_intracranialv,
        abcd_y_lt_bl.interview_age,
        family_income,
    ]

    covariates = pd.concat(series_list, axis=1)

    # %%
    ### Add the covariates to imaging data

    # Cortical thickness
    t1w_cortical_thickness_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_thickness_bl_pass.csv",
    )

    t1w_cortical_thickness_bl_pass = pd.read_csv(
        t1w_cortical_thickness_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    # Cortical volume
    t1w_cortical_volume_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_volume_bl_pass.csv",
    )

    t1w_cortical_volume_bl_pass = pd.read_csv(
        t1w_cortical_volume_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_volume_bl_pass = t1w_cortical_volume_bl_pass.drop(
        columns=["eventname"]
    )

    # Cortical surface area

    t1w_cortical_surface_area_bl_pass_path = Path(
        processed_data_path,
        "t1w_cortical_surface_area_bl_pass.csv",
    )

    t1w_cortical_surface_area_bl_pass = pd.read_csv(
        t1w_cortical_surface_area_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    t1w_cortical_surface_area_bl_pass = t1w_cortical_surface_area_bl_pass.drop(
        columns=["eventname"]
    )

    # Join the covariates to the cortical features (no missing data here)

    t1w_cortical_features_bl_pass = (
        t1w_cortical_thickness_bl_pass.join(
            t1w_cortical_volume_bl_pass,
            how="left",
        ).join(
            t1w_cortical_surface_area_bl_pass,
            how="left",
        )
    ).dropna()
    # %%

    gordon_cor_subcortical_bl_pass_path = Path(
        processed_data_path,
        "gordon_cor_subcortical_bl_pass.csv",
    )

    gordon_cor_subcortical_bl_pass = pd.read_csv(
        gordon_cor_subcortical_bl_pass_path,
        index_col=0,
        low_memory=False,
    )

    # %%
    ### Merge all features here before deconfounding for descriptive analysis

    all_features_precon = t1w_cortical_features_bl_pass.merge(
        gordon_cor_subcortical_bl_pass.dropna(),
        left_index=True,
        right_index=True,
        how="outer",
    ).join(
        covariates,
        how="inner",
    )

    all_features_precon.to_csv(
        Path(
            processed_data_path,
            "all_brain_features_precon.csv",
        ),
        index=True,
    )

    # %%
    ### Apply NeuroCombat to the imaging data

    # NeuroCombat for cortical thickness
    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "r") as f:
        brain_features_of_interest = json.load(f)

    t1w_cortical_thickness_rois = brain_features_of_interest[
        "t1w_cortical_thickness_rois"
    ]

    t1w_cortical_volume_rois = brain_features_of_interest["t1w_cortical_volume_rois"]

    t1w_cortical_surface_area_rois = brain_features_of_interest[
        "t1w_cortical_surface_area_rois"
    ]

    # Add intracranial volume here to be combated as well as other brain features
    t1w_brain_features_list = (
        t1w_cortical_thickness_rois
        + t1w_cortical_volume_rois
        + t1w_cortical_surface_area_rois
        + ["smri_vol_scs_intracranialv"]
    )

    t1w_cortical_features_list = (
        t1w_cortical_thickness_rois
        + t1w_cortical_volume_rois
        + t1w_cortical_surface_area_rois
    )

    t1w_cortical_features_bl_pass = t1w_cortical_features_bl_pass.join(
        covariates,
        how="left",
    )

    t1w_cortical_features_covariates = t1w_cortical_features_bl_pass[
        covariates.columns
    ].drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    cortical_features_combat = neuroCombat(
        dat=np.array(t1w_cortical_features_bl_pass[t1w_brain_features_list]).T,
        covars=t1w_cortical_features_covariates,
        batch_col="label_site",
        categorical_cols=["demo_sex_v2"],
        continuous_cols=["interview_age"],
    )["data"]

    cortical_features_post_combat = pd.DataFrame(
        data=cortical_features_combat.T, columns=t1w_brain_features_list
    ).set_index(t1w_cortical_features_bl_pass.index)

    # Drop intracranial volume because we have the post-combat ones now
    covariates_no_intracranialv = covariates.drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    cortical_features_post_combat_covar = cortical_features_post_combat.join(
        covariates_no_intracranialv,
        how="left",
    )

    # NeuroCombat for rs-fMRI
    gordon_net_subcor_no_dup = brain_features_of_interest["gordon_net_subcor_no_dup"]

    rsfmri_brain_features = gordon_net_subcor_no_dup + ["smri_vol_scs_intracranialv"]

    gordon_cor_subcortical_bl_pass = gordon_cor_subcortical_bl_pass.join(
        covariates, how="left"
    ).dropna()

    rsfmri_covariates_no_intracranialv = gordon_cor_subcortical_bl_pass[
        covariates.columns
    ].drop(
        "smri_vol_scs_intracranialv",
        axis=1,
    )

    rsfmri_combat = neuroCombat(
        dat=np.array(gordon_cor_subcortical_bl_pass[rsfmri_brain_features]).T,
        covars=rsfmri_covariates_no_intracranialv,
        batch_col="label_site",
        categorical_cols=["demo_sex_v2"],
        continuous_cols=["interview_age"],
    )["data"]

    rsfmri_post_combat = pd.DataFrame(
        data=rsfmri_combat.T, columns=rsfmri_brain_features
    ).set_index(gordon_cor_subcortical_bl_pass.index)

    rsfmri_post_combat_covars = rsfmri_post_combat.join(
        covariates_no_intracranialv,
        how="left",
    )

    cortical_features_post_combat_covar.to_csv(
        Path(processed_data_path, "t1w_cortical_features_post_combat.csv"),
        index=True,
    )

    rsfmri_post_combat_covars.to_csv(
        Path(processed_data_path, "gordon_cor_subcortical_post_combat.csv"),
        index=True,
    )


if __name__ == "__main__":
    pre_deconfound_image_exc_sex()
