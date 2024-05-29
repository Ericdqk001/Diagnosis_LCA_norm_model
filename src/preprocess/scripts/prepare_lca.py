from pathlib import Path

import pandas as pd


def prepare_lca():

    data_path = Path("data")

    all_psych_dx_path = Path(
        data_path,
        "liza_data",
        "all_psych_dx_r5.csv",
    )

    all_psych_dx = pd.read_csv(
        all_psych_dx_path,
        index_col=0,
        low_memory=False,
    )

    # Exclude all rows with 'control' in the 'psych_dx' column
    # all_psych_dx = all_psych_dx[all_psych_dx["psych_dx"] != "control"]

    # Get the diagnosis columns
    dx_columns = [col for col in all_psych_dx.columns if "Has" in col]

    # Convert True/False to 1/0 for those columns
    all_psych_dx[dx_columns] = all_psych_dx[dx_columns].applymap(
        lambda x: 2 if x == True else 1
    )

    # Select only the dx_columns
    dx_columns_df = all_psych_dx[dx_columns]

    # Join with cbcl dummy

    cbcl_dummy_path = Path(
        data_path,
        "LCA",
        "cbcl_t_no_mis_dummy.csv",
    )

    cbcl_dummy = pd.read_csv(
        cbcl_dummy_path,
        index_col=0,
        low_memory=False,
    )

    cbcl_psych_dx_dummy = cbcl_dummy.join(
        dx_columns_df,
        how="inner",
    )

    # Save it as a csv file, keeping the index
    cbcl_psych_dx_dummy.to_csv(
        Path(
            "data",
            "LCA",
            "cbcl_psych_dx_dummy.csv",
        )
    )


if __name__ == "__main__":
    prepare_lca()
