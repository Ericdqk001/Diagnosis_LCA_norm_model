import json
from pathlib import Path

import pandas as pd


def prepare_image():
    """Prepare the imaging data for the analysis.

    Familial members were removed (randomly kept one). Subjects with intersex were
    removed. Subjects with data quality issues were removed. The data was saved in csv
    files. Brain features of interest were selected and saved in a JSON file.

    """
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

    imaging_path = Path(
        core_data_path,
        "imaging",
    )

    general_info_path = Path(
        core_data_path,
        "abcd-general",
    )

    # For biological sex (demo_sex_v2)
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

    demographics_bl.demo_sex_v2.value_counts()

    inter_sex_subs = demographics_bl[demographics_bl.demo_sex_v2 == 3].index

    genetics_path = Path(
        core_data_path,
        "genetics",
    )

    # Recommended image inclusion (NDA 4.0 abcd_imgincl01)
    mri_y_qc_incl_path = Path(
        imaging_path,
        "mri_y_qc_incl.csv",
    )

    mri_y_qc_incl = pd.read_csv(
        mri_y_qc_incl_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_qc_incl_bl = mri_y_qc_incl[mri_y_qc_incl.eventname == "baseline_year_1_arm_1"]

    # Remove subjects with intersex from the imaging data
    mri_y_qc_incl_bl = mri_y_qc_incl_bl[~mri_y_qc_incl_bl.index.isin(inter_sex_subs)]

    # %%
    ### Remove imaging data with data quality issues, overall MRI clinical report is used
    # here as well.

    # First, we apply quality control to T1 weighted images (for structural features).
    # Conditions for inclusion:
    # 1. T1w data recommended for inclusion (YES)
    # 2. Overall MRI clinical report score < 3, which excludes subjects with neurological issues.

    mri_clin_report_path = Path(
        imaging_path,
        "mri_y_qc_clfind.csv",
    )

    mri_clin_report = pd.read_csv(
        mri_clin_report_path,
        index_col=0,
        low_memory=False,
    )

    mri_clin_report_bl = mri_clin_report[
        mri_clin_report.eventname == "baseline_year_1_arm_1"
    ]

    t1w_qc_passed_indices = list(
        mri_y_qc_incl_bl[(mri_y_qc_incl_bl.imgincl_t1w_include == 1)].index
    )

    t1w_qc_passed_mask = mri_clin_report_bl.index.isin(t1w_qc_passed_indices)

    score_mask = mri_clin_report_bl.mrif_score < 3

    # No missing values here
    subs_t1w_pass = mri_clin_report_bl[t1w_qc_passed_mask & score_mask]

    # Second, we apply quality control to rs-fMRI images.
    # Conditions for inclusion:
    # 1. rs-fMRI data recommended for inclusion (YES)
    # 2. Overall MRI clinical report score < 3, which means normal
    # 3. Remove subjects without cbcl data
    # 4. Remove familial members

    rsfmri_qc_passed_indices = list(
        mri_y_qc_incl_bl[(mri_y_qc_incl_bl.imgincl_rsfmri_include == 1)].index
    )

    rsfmri_qc_passed_mask = mri_clin_report_bl.index.isin(rsfmri_qc_passed_indices)

    # No missing values here
    subs_rsfmri_pass = mri_clin_report_bl[rsfmri_qc_passed_mask & score_mask]

    # Remove subjects with no cbcl data here
    cbcl_LCA_path = Path(
        LCA_path,
        "cbcl_class_member_prob.csv",
    )

    cbcl_LCA = pd.read_csv(
        cbcl_LCA_path,
        index_col=0,
        low_memory=False,
    )

    predicted_class = cbcl_LCA["predicted_class"]

    subs_t1w_pass = subs_t1w_pass[subs_t1w_pass.index.isin(predicted_class.index)]

    # 6 removed
    subs_rsfmri_pass = subs_rsfmri_pass[
        subs_rsfmri_pass.index.isin(predicted_class.index)
    ]

    # Remove subjects without diagnosis

    psych_dx_path = Path(
        "data",
        "liza_data",
        "all_psych_dx_r5.csv",
    )

    psych_dx = pd.read_csv(
        psych_dx_path,
        index_col=0,
        low_memory=False,
    )["psych_dx"]

    subs_t1w_pass = subs_t1w_pass[subs_t1w_pass.index.isin(psych_dx.index)]

    # 113 removed
    subs_rsfmri_pass = subs_rsfmri_pass[subs_rsfmri_pass.index.isin(psych_dx.index)]

    # Genetics and relatedness (NDA 4.0 acspsw03), used to remove familial members.
    # Randomly select one subject from each family to include in the analysis.
    genetics_relatedness_path = Path(
        genetics_path,
        "gen_y_pihat.csv",
    )

    genetics_relatedness = pd.read_csv(
        genetics_relatedness_path,
        index_col=0,
        low_memory=False,
    )

    family_id = genetics_relatedness["rel_family_id"]

    # No missing value of family_id after joining
    subs_t1w_pass_fam_id = subs_t1w_pass.join(
        family_id,
        how="inner",
    )

    subs_rsfmri_pass_fam_id = subs_rsfmri_pass.join(
        family_id,
        how="inner",
    )

    seed = 42

    # Before removing familial members, a total of 11771 - 10733 = 1038 subjects were removed
    # 10733 - 9031 = 1702
    unrelated_subs_t1w = subs_t1w_pass_fam_id.loc[
        subs_t1w_pass_fam_id.groupby(["rel_family_id"]).apply(
            lambda x: x.sample(n=1, random_state=seed).index[0]
        ),
    ]

    unrelated_subs_rsfmri = subs_rsfmri_pass_fam_id.loc[
        subs_rsfmri_pass_fam_id.groupby(["rel_family_id"]).apply(
            lambda x: x.sample(n=1, random_state=seed).index[0]
        ),
    ]

    ###

    # %%
    ### Now prepare the smri (cortical features) data
    # and rsfmri (Gordon Network correlations plus subcortical) data

    mri_y_smr_thk_dst_path = Path(
        imaging_path,
        "mri_y_smr_thk_dst.csv",
    )

    mri_y_smr_thk_dst = pd.read_csv(
        mri_y_smr_thk_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_dst_path = Path(
        imaging_path,
        "mri_y_smr_vol_dst.csv",
    )

    mri_y_smr_vol_dst = pd.read_csv(
        mri_y_smr_vol_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_area_dst_path = Path(
        imaging_path,
        "mri_y_smr_area_dst.csv",
    )

    mri_y_smr_area_dst = pd.read_csv(
        mri_y_smr_area_dst_path,
        index_col=0,
        low_memory=False,
    )

    # With subcortical areas
    mri_y_rsfmr_cor_gp_aseg_path = Path(
        imaging_path,
        "mri_y_rsfmr_cor_gp_aseg.csv",
    )

    mri_y_rsfmr_cor_gp_aseg = pd.read_csv(
        mri_y_rsfmr_cor_gp_aseg_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_rsfmr_cor_gp_gp_path = Path(
        imaging_path,
        "mri_y_rsfmr_cor_gp_gp.csv",
    )

    # Gordon Network correlations
    mri_y_rsfmr_cor_gp_gp = pd.read_csv(
        mri_y_rsfmr_cor_gp_gp_path,
        index_col=0,
        low_memory=False,
    )

    # Select the baseline data for the subjects who passed the quality control and drop
    # subjects with missing data and save the data in csv files

    # Cortical thickness data
    mri_y_smr_thk_dst_bl = mri_y_smr_thk_dst[
        mri_y_smr_thk_dst.eventname == "baseline_year_1_arm_1"
    ]

    # 10 with missing values are dropped here for t1w
    t1w_cortical_thickness_bl_pass = mri_y_smr_thk_dst_bl[
        mri_y_smr_thk_dst_bl.index.isin(unrelated_subs_t1w.index)
    ].dropna()

    t1w_cortical_thickness_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_thickness_bl_pass.csv",
        ),
        index=True,
    )

    # Cortical volume data
    mri_y_smr_vol_dst_bl = mri_y_smr_vol_dst[
        mri_y_smr_vol_dst.eventname == "baseline_year_1_arm_1"
    ]

    t1w_cortical_volume_bl_pass = mri_y_smr_vol_dst_bl[
        mri_y_smr_vol_dst_bl.index.isin(unrelated_subs_t1w.index)
    ].dropna()

    t1w_cortical_volume_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_volume_bl_pass.csv",
        ),
        index=True,
    )

    # Cortical surface area data

    mri_y_smr_area_dst_bl = mri_y_smr_area_dst[
        mri_y_smr_area_dst.eventname == "baseline_year_1_arm_1"
    ]

    t1w_cortical_surface_area_bl_pass = mri_y_smr_area_dst_bl[
        mri_y_smr_area_dst_bl.index.isin(unrelated_subs_t1w.index)
    ].dropna()

    t1w_cortical_surface_area_bl_pass.to_csv(
        Path(
            processed_data_path,
            "t1w_cortical_surface_area_bl_pass.csv",
        ),
        index=True,
    )

    # rs-fMRI data
    # First combine the Gordon Network correlations and their correlations with subcortical
    # areas
    mri_y_rsfmr_cor_gp_aseg_bl = mri_y_rsfmr_cor_gp_aseg[
        mri_y_rsfmr_cor_gp_aseg.eventname == "baseline_year_1_arm_1"
    ]

    mri_y_rsfmr_cor_gp_gp_bl = mri_y_rsfmr_cor_gp_gp[
        mri_y_rsfmr_cor_gp_gp.eventname == "baseline_year_1_arm_1"
    ]

    # Combine the dataframes
    gordon_cor_subcortical = pd.concat(
        [mri_y_rsfmr_cor_gp_gp_bl, mri_y_rsfmr_cor_gp_aseg_bl], axis=1
    )

    gordon_cor_subcortical_bl_pass = gordon_cor_subcortical[
        gordon_cor_subcortical.index.isin(unrelated_subs_rsfmri.index)
    ].dropna()

    gordon_cor_subcortical_bl_pass.to_csv(
        Path(
            processed_data_path,
            "gordon_cor_subcortical_bl_pass.csv",
        ),
        index=True,
    )

    # %%
    ### Now select the columns that are the phenotypes of interest for each modality

    # For cortical thickness ('mrisdp_1' to 'mrisdp_148')
    t1w_cortical_thickness_rois = list(t1w_cortical_thickness_bl_pass.columns[1:-3])

    # For cortical volume
    t1w_cortical_volume_rois = list(t1w_cortical_volume_bl_pass.columns[1:-3])

    # For surface area

    t1w_cortical_surface_area_rois = list(
        t1w_cortical_surface_area_bl_pass.columns[1:-3]
    )

    # For rs-fMRI, we remove duplicate corrections between Gordon Network

    def remove_duplicates(column_names):
        """Remove duplicates between networks in a list of column names.

        Args:
            column_names (list of str): List of column names representing correlations
            between brain functional networks.

        Returns:
            list of str: List of unique column names without duplicates between networks.
        """
        unique_names = set()
        unique_columns = []

        for column_name in column_names:
            # Split the column name by underscores
            parts = column_name.split("_")

            # Sort the parts to ensure consistency
            parts.sort()

            # Reconstruct the sorted column name
            sorted_name = "_".join(parts)

            # Check if the sorted column name is already in the set
            if sorted_name not in unique_names:
                # If not, add it to the set and append the original column name to
                # the unique columns list
                unique_names.add(sorted_name)
                unique_columns.append(column_name)

        return unique_columns

    gordon_net_cor = list(mri_y_rsfmr_cor_gp_gp_bl.columns)[1:]

    # Remove duplicates between networks
    gordon_net_cor_no_dup = remove_duplicates(gordon_net_cor)

    # Add the subcortical areas to the list of unique columns (+ 247 items in total)
    gordon_net_subcor_no_dup = gordon_net_cor_no_dup.copy()

    gordon_net_subcor_no_dup.extend(list(mri_y_rsfmr_cor_gp_aseg_bl.columns)[1:])

    # Add only amygdala and accumbens

    gordon_net_subcor_limbic_no_dup = gordon_net_cor_no_dup.copy()

    # TODO remove hippocampus if model fit is too low (removed, loss 300+)
    gordon_net_subcor_limbic_no_dup.extend(
        [
            col
            for col in mri_y_rsfmr_cor_gp_aseg_bl.columns
            if "aglh" in col or "agrh" in col or "aalh" in col or "aarh" in col
            # or "_hplh" in col
            # or "_hprh" in col
        ]
    )

    # Save the selected features to a dictionary as a JSON file
    brain_features_of_interest = {
        "t1w_cortical_thickness_rois": t1w_cortical_thickness_rois,
        "t1w_cortical_volume_rois": t1w_cortical_volume_rois,
        "t1w_cortical_surface_area_rois": t1w_cortical_surface_area_rois,
        "gordon_net_cor_no_dup": gordon_net_cor_no_dup,
        "gordon_net_subcor_no_dup": gordon_net_subcor_no_dup,
        "gordon_net_subcor_limbic_no_dup": gordon_net_subcor_limbic_no_dup,
    }

    brain_features_of_interest_path = Path(
        processed_data_path,
        "brain_features_of_interest.json",
    )

    with open(brain_features_of_interest_path, "w") as f:
        json.dump(brain_features_of_interest, f)


if __name__ == "__main__":
    prepare_image()
