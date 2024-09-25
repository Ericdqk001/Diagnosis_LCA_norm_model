from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the datasets
psych_dx_path = Path("data", "liza_data", "all_psych_dx_r5.csv")
psych_dx = pd.read_csv(psych_dx_path, index_col=0, low_memory=False)

lca_path = Path("data", "LCA")
lca_class_memberships_path = Path(lca_path, "cbcl_class_member_prob.csv")
lca_class_memberships = pd.read_csv(
    lca_class_memberships_path, index_col=0, low_memory=False
)

data_path = Path("data", "raw_data", "core", "mental-health", "mh_p_cbcl.csv")
cbcl_t_vars_path = Path("data", "var_dict", "cbcl_8_dim_t.csv")
cbcl = pd.read_csv(data_path, index_col=0, low_memory=False)
cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)

cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

# Select the baseline data
baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

# Filter columns with t variables
filtered_cbcl = baseline_cbcl[cbcl_t_vars]

# Drop the dummy variables from LCA first with the same CBCL variable names
lca_class_memberships = lca_class_memberships.drop(cbcl_t_vars, axis=1)

# Join the LCA class memberships with the CBCL data
cbcl_lca_memberships = lca_class_memberships.join(filtered_cbcl, how="inner")


def plot_radar_chart_after_exclusion(df, threshold=65, low_entropy=False):
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
        df = df[~df.index.isin(high_entropy_subs)]

    class_names = [
        "Predominantly Internalizing",
        "Predominantly Externalizing",
        "Highly Dysregulated",
    ]

    colors = ["green", "red", "purple"]

    num_vars = len(cbcl_scales)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Ensure the plot is closed

    fig, axes = plt.subplots(
        1, 3, subplot_kw=dict(polar=True), figsize=(18, 6), dpi=100
    )

    class_ids_to_plot = [2, 3, 4]

    for i, class_id in enumerate(class_ids_to_plot):
        ax = axes[i]
        class_df = df[df["predicted_class"] == class_id]

        proportions = (class_df[cbcl_scales] >= threshold).mean().tolist()
        proportions += proportions[:1]

        ax.plot(
            angles,
            proportions,
            "o-",
            color=colors[i],
            alpha=0.8,
            linewidth=3,
        )
        ax.fill(angles, proportions, color=colors[i], alpha=0.3)

        ax.set_title(f"{class_names[i]}", fontweight="bold", fontsize=16)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, rotation=45, fontsize=12)

        # Set radial limits explicitly to ensure consistency
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="black", size=10)

        ax.grid(True)

    fig.suptitle("", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Example usage
plot_radar_chart_after_exclusion(
    cbcl_lca_memberships,
    threshold=65,
    low_entropy=True,
)
