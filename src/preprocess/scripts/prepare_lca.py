from pathlib import Path

import pandas as pd


def prepare_lca():
    """Prepare the CBCL data for LCA analysis by selecting the baseline data and
    removing missing values."""
    data_path = Path(
        "data",
        "raw_data",
        "core",
        "mental-health",
        "mh_p_cbcl.csv",
    )

    cbcl_t_vars_path = Path(
        "data",
        "var_dict",
        "cbcl_8_dim_t.csv",
    )

    cbcl = pd.read_csv(
        data_path,
        index_col=0,
        low_memory=False,
    )

    cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)

    cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

    # Add the internalising and externalising syndromes and total problem scales to the list for analysis
    sum_syndrome = [
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
        "cbcl_scr_syn_totprob_t",
    ]

    cbcl_t_vars.extend(sum_syndrome)

    # Select the baseline data
    baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

    # Filter columns with t variables
    filtered_cbcl = baseline_cbcl[cbcl_t_vars]

    # Count the number of missing data
    num_missing = filtered_cbcl.isnull().sum().sum()
    print("Number of missing data:", num_missing)

    # Record the subject ids with missing data

    missing_subject_ids = filtered_cbcl[
        filtered_cbcl.isnull().any(axis=1)
    ].index.tolist()
    print("Subject IDs with missing data:", missing_subject_ids)

    # Remove missing values

    filtered_cbcl = filtered_cbcl.dropna()

    # Create dummy variables using a threshold of 65 for the t scores
    filtered_cbcl = (filtered_cbcl >= 65).astype(int)

    # Add one here because LCA expects 1/2 rather than 0/1
    filtered_cbcl += 1

    # Save the filtered data
    filtered_cbcl_save_path = Path(
        "data",
        "LCA",
    )

    if not filtered_cbcl_save_path.exists():
        filtered_cbcl_save_path.mkdir(parents=True)

    filtered_cbcl.to_csv(
        Path(
            filtered_cbcl_save_path,
            "cbcl_t_no_mis_dummy.csv",
        ),
    )


if __name__ == "__main__":
    prepare_lca()
