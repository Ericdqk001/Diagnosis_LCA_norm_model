from pathlib import Path

import pandas


def prepare_final_sample():
    """Prepare the CBCL data for the final sample."""
    original_cbcl_dummy_path = Path(
        "data",
        "LCA",
        "cbcl_t_no_mis_dummy.csv",
    )

    cbcl = pandas.read_csv(
        original_cbcl_dummy_path,
        index_col=0,
        low_memory=False,
    )

    final_sample_mri_path = Path(
        "data",
        "processed_data",
        "t1w_cortical_thickness_bl_pass.csv",
    )

    final_sample_mri = pandas.read_csv(
        final_sample_mri_path,
        index_col=0,
        low_memory=False,
    )

    # Filter the cbcl sample with the index in the final sample
    final_sample_cbcl = cbcl.loc[final_sample_mri.index]

    # Save the final sample
    final_sample_cbcl.to_csv(
        Path(
            "data",
            "LCA",
            "final_sample_cbcl.csv",
        ),
    )


if __name__ == "__main__":
    prepare_final_sample()
