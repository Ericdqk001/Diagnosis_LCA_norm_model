from pathlib import Path

import matplotlib.pyplot as plt
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
## list for analysis
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


def plot_individual_cbcl_patterns_separate(df, low_entropy=False):
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

    if low_entropy:
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
        df = df[~df.index.isin(high_entropy_subs)]

    class_names = [
        "Low Symptom",
        "Predominantly Internalising",
        "Predominantly Externalising",
        "Highly Dysregulated",
    ]

    bright_colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
        "#FF33A1",
    ]

    for class_id in sorted(df["predicted_class"].unique()):
        class_df = df[df["predicted_class"] == class_id]

        plt.figure(figsize=(10, 6.7))
        # Plot each individual's pattern as a line
        for _, row in class_df.iterrows():
            plt.plot(
                cbcl_scales,
                row[cbcl_scales],
                color=bright_colors[class_id - 1],
                alpha=0.1,  # Make lines semi-transparent
                linewidth=1,  # Make lines thinner
            )

        # Set plot title and labels
        plt.title(
            f"Individual CBCL Score Patterns - {class_names[class_id - 1]}",
            fontweight="bold",
        )
        plt.xlabel("CBCL Scales", color="black", fontweight="bold")
        plt.ylabel("Scores", color="black", fontweight="bold")
        plt.xticks(rotation=45, color="black", fontweight="bold")
        plt.yticks(color="black", fontweight="bold")
        plt.grid(False)
        plt.gca().set_facecolor("white")
        plt.tight_layout()
        plt.show()


plot_individual_cbcl_patterns_separate(cbcl_lca_memberships, low_entropy=True)


def plot_cbcl_patterns_by_syndrome(df, low_entropy=False):
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

    sum_syndrome = [
        "cbcl_scr_syn_internal_t",
        "cbcl_scr_syn_external_t",
        "cbcl_scr_syn_totprob_t",
    ]

    if low_entropy:
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
        df = df[~df.index.isin(high_entropy_subs)]

    bright_colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
    ]

    syndrome_names = [
        "Internalizing Problems",
        "Externalizing Problems",
        "Total Problems",
    ]

    for i, syndrome in enumerate(sum_syndrome):
        syndrome_df = df[df[syndrome] == 2]

        plt.figure(figsize=(10, 6.7))
        # Plot each individual's pattern as a line
        for _, row in syndrome_df.iterrows():
            plt.plot(
                cbcl_scales,
                row[cbcl_scales],
                color=bright_colors[i],
                alpha=0.1,  # Make lines semi-transparent
                linewidth=1,  # Make lines thinner
            )

        # Set plot title and labels
        plt.title(f"Individual CBCL Score Patterns - {syndrome_names[i]}", fontweight='bold')
        plt.xlabel("CBCL Scales", color="black", fontweight="bold")
        plt.ylabel("Scores", color="black", fontweight="bold")
        plt.xticks(rotation=45, color="black", fontweight="bold")
        plt.yticks(color="black", fontweight="bold")
        plt.grid(False)
        plt.gca().set_facecolor("white")
        plt.tight_layout()
        plt.show()

# Example usage
# plot_cbcl_patterns_by_syndrome(cbcl_lca_memberships, low_entropy=False)