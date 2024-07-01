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

    # Invert the syndrome_map to use as labels
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

    # Map the cbcl_scales to their respective names for the x-axis labels
    axis_labels = [syndrome_map[scale] for scale in cbcl_scales]

    if low_entropy:
        # Load subjects with high entropy
        high_entropy_subs_path = Path("data", "LCA", "subjects_with_high_entropy.csv")
        high_entropy_subs = pd.read_csv(high_entropy_subs_path, low_memory=False)[
            "subject"
        ].tolist()
        # Separate the dataframe into high entropy and non-high entropy subjects
        high_entropy_df = df[df.index.isin(high_entropy_subs)]
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

    # Set up a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, class_id in enumerate(sorted(df["predicted_class"].unique())):
        ax = axes[i // 2, i % 2]  # Determine subplot position
        class_df = df[df["predicted_class"] == class_id]

        for _, row in class_df.iterrows():
            ax.plot(
                axis_labels,
                row[cbcl_scales],
                color=bright_colors[class_id - 1],
                alpha=0.1,  # Make lines semi-transparent
                linewidth=1,  # Make lines thinner
            )

        ax.set_title(f"{class_names[class_id - 1]}", fontweight="bold")
        ax.set_xticks(range(len(cbcl_scales)))
        ax.set_xticklabels(axis_labels, rotation=45)
        ax.set_ylabel("Scores")
        ax.grid(True)

    plt.tight_layout()

    if low_entropy:
        # Plot high entropy subjects separately
        plt.figure(figsize=(10, 6.7))
        for _, row in high_entropy_df.iterrows():
            plt.plot(
                axis_labels,
                row[cbcl_scales],
                color="grey",
                alpha=0.1,  # Make lines semi-transparent
                linewidth=1,  # Make lines thinner
            )

        plt.title("High Entropy Subjects", fontweight="bold")
        plt.xlabel("CBCL Scales")
        plt.ylabel("Scores")
        plt.xticks(range(len(cbcl_scales)), labels=axis_labels, rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Assuming cbcl_lca_memberships is already defined and loaded as before
plot_individual_cbcl_patterns_separate(cbcl_lca_memberships, low_entropy=True)
