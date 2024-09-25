from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

psych_dx_path = Path(
    "data",
    "liza_data",
    "all_psych_dx_r5.csv",
)

psych_dx = pd.read_csv(
    psych_dx_path,
    index_col=0,
    low_memory=False,
)

lca_path = Path(
    "data",
    "LCA",
)

lca_class_memberships_path = Path(
    lca_path,
    "cbcl_class_member_prob.csv",
)

lca_class_memberships = pd.read_csv(
    lca_class_memberships_path,
    index_col=0,
    low_memory=False,
)

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

# Add the internalising and externalising syndromes and total problem scales to the
# list for analysis
# sum_syndrome = [
#     "cbcl_scr_syn_internal_t",
#     "cbcl_scr_syn_external_t",
#     "cbcl_scr_syn_totprob_t",
# ]

# cbcl_t_vars.extend(sum_syndrome)

# Select the baseline data
baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

# Filter columns with t variables
filtered_cbcl = baseline_cbcl[cbcl_t_vars]

# Drop the dummy variables from lca first with the same cbcl variable names

lca_class_memberships = lca_class_memberships.drop(
    cbcl_t_vars,
    axis=1,
)

# Join the LCA class memberships with the CBCL data

cbcl_lca_memberships = lca_class_memberships.join(
    filtered_cbcl,
    how="inner",
)


def plot_mean_variance_cbcl_patterns_with_comparison(df, cbcl_t_vars):
    cbcl_scales = [
        "cbcl_scr_syn_anxdep_t",
        "cbcl_scr_syn_withdep_t",
        "cbcl_scr_syn_somatic_t",
        "cbcl_scr_syn_social_t",
        "cbcl_scr_syn_thought_t",
        "cbcl_scr_syn_attention_t",
        "cbcl_scr_syn_rulebreak_t",
        "cbcl_scr_syn_aggressive_t",
    ]

    syndrome_map = {
        "cbcl_scr_syn_anxdep_t": "Anxiety/Depress",
        "cbcl_scr_syn_withdep_t": "Withdraw/Depress",
        "cbcl_scr_syn_somatic_t": "Somatic",
        "cbcl_scr_syn_social_t": "Social",
        "cbcl_scr_syn_thought_t": "Thought",
        "cbcl_scr_syn_attention_t": "Attention",
        "cbcl_scr_syn_rulebreak_t": "RuleBreak",
        "cbcl_scr_syn_aggressive_t": "Aggressive",
    }

    axis_labels = [syndrome_map[scale] for scale in cbcl_scales]

    high_entropy_subs_path = Path("data", "LCA", "subjects_with_high_entropy.csv")
    high_entropy_subs = pd.read_csv(high_entropy_subs_path, low_memory=False)[
        "subject"
    ].tolist()
    df_low_entropy = df[~df.index.isin(high_entropy_subs)]

    # class_names = [
    #     "Low Symptom",
    #     "Predominantly Internalising",
    #     "Predominantly Externalising",
    #     "Highly Dysregulated",
    # ]

    class_names = [
        "class 1",
        "class 2",
        "class 3",
        "class 4",
    ]

    colors = ["blue", "green", "red", "purple"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)

    for i, class_id in enumerate(sorted(df["predicted_class"].unique())):
        ax = axes[i // 2, i % 2]

        class_df = df[df["predicted_class"] == class_id]
        class_df_low_entropy = df_low_entropy[
            df_low_entropy["predicted_class"] == class_id
        ]

        means_before = class_df[cbcl_scales].mean()
        stds_before = class_df[cbcl_scales].std()

        means_after = class_df_low_entropy[cbcl_scales].mean()
        stds_after = class_df_low_entropy[cbcl_scales].std()

        ax.errorbar(
            axis_labels,
            means_before,
            yerr=stds_before,
            fmt="o--",
            color=colors[i],
            ecolor="gray",
            elinewidth=2,
            capsize=5,
            label="Before Exclusion (Mean ± 1 SD)",
        )

        ax.errorbar(
            axis_labels,
            means_after,
            yerr=stds_after,
            fmt="o-",
            color=colors[i],
            ecolor="black",
            elinewidth=3,
            capsize=5,
            label="After Exclusion (Mean ± 1 SD)",
        )

        ax.set_title(f"{class_names[class_id - 1]}", fontweight="bold")
        ax.set_xticks(np.arange(len(cbcl_scales)))
        ax.set_xticklabels(axis_labels, rotation=45)
        ax.set_ylabel("Scores")
        ax.set_ylim([50, 85])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# Example usage, assuming 'cbcl_lca_memberships' is already defined and loaded as before
plot_mean_variance_cbcl_patterns_with_comparison(cbcl_lca_memberships, cbcl_t_vars)
