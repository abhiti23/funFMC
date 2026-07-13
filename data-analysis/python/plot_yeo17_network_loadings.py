import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nilearn import datasets

# ── 1. Load data ────────────────────────────────────────────────────────────
DIR = "../results"
A = np.loadtxt(f"{DIR}/est_A.csv", delimiter=",")
A = A[1:]
p, K = A.shape

THRESHOLD = 0.10
A[np.abs(A) <= THRESHOLD] = 0.0

schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
labels = schaefer.labels
if len(labels) == p + 1:
    labels = labels[1:]


def parse_network_fine(label):
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    parts = label.split("_")
    return parts[2] if len(parts) > 2 else "Unknown"


def parse_hemi(label):
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    parts = label.split("_")
    return parts[1] if len(parts) > 1 else "Unknown"


network_labels = np.array([parse_network_fine(l) for l in labels])
hemi_labels = np.array([parse_hemi(l) for l in labels])
unique_networks = sorted(set(network_labels))

# ── 2. Compute hemisphere-separated mean loadings ───────────────────────────
hemi_records = []
for net in unique_networks:
    for hemi in ["LH", "RH"]:
        mask = (network_labels == net) & (hemi_labels == hemi)
        if mask.sum() == 0:
            continue
        for k in range(K):
            vals = A[mask, k]
            hemi_records.append({
                "network": net,
                "hemi": hemi,
                "cluster": k,
                "mean": vals.mean(),
                "sem": vals.std() / np.sqrt(len(vals)),
                "n": int(mask.sum()),
            })

df_hemi = pd.DataFrame(hemi_records)
pivot_hemi = df_hemi.pivot_table(
    index=["network", "hemi"], columns="cluster", values="mean").round(3)
print("\nHemisphere-separated mean loadings (fine networks)::")
print(pivot_hemi.to_string())
pivot_hemi.to_csv(f"{DIR}/yeo17_network_hemi_mean_loadings.csv")
print("Saved hemisphere table to yeo17_network_hemi_mean_loadings.csv")

# Sort networks by mean loading of Cluster 1 (averaged across hemispheres),
# descending — so most important networks appear at top (for lollipop) or left (for bars)
cluster1_order = (
    df_hemi[df_hemi["cluster"] == 0]
    .groupby("network")["mean"].mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_titles = ["Cluster 1", "Cluster 2", "Cluster 3"]
hemi_colors = {"LH": "#2c7bb6", "RH": "#d7191c"}  # blue / red
hemi_labels_display = {"LH": "Left hemisphere", "RH": "Right hemisphere"}

# ════════════════════════════════════════════════════════════════════════════
# OPTION 1 — Three panels (one per cluster), bar chart, LH vs RH side by side
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 12), facecolor="white",
                         sharex=False)

x = np.arange(len(unique_networks))
width = 0.35

for k, ax in enumerate(axes):
    sub = df_hemi[df_hemi["cluster"] == k]

    for i, hemi in enumerate(["LH", "RH"]):
        h_sub = sub[sub["hemi"] == hemi].set_index("network").loc[
            cluster1_order]
        ax.bar(
            x + (i - 0.5) * width,
            h_sub["mean"],
            width,
            yerr=h_sub["sem"],
            capsize=3,
            color=hemi_colors[hemi],
            alpha=0.80,
            label=hemi_labels_display[hemi],
            error_kw={"elinewidth": 0.8, "ecolor": "black"},
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster1_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean loading", fontsize=10)
    ax.set_title(cluster_titles[k], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, len(unique_networks) - 0.4)

    if k == 0:
        ax.legend(fontsize=9, loc="upper right")

plt.suptitle(
    "Mean cluster loadings by Yeo-17 network and hemisphere",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig(f"{DIR}/yeo17_three_panels_bar.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("Saved three-panel bar chart")

# ════════════════════════════════════════════════════════════════════════════
# OPTION 3 — Three panels (one per cluster), lollipop / dot plot
#             Networks on y-axis (horizontal labels), LH and RH as separate dots
#             connected by a thin line to show asymmetry at a glance
# ════════════════════════════════════════════════════════════════════════════

# For lollipop, sort ascending so highest-loading network is at the TOP
lollipop_order = list(reversed(cluster1_order))

fig, axes = plt.subplots(1, 3, figsize=(14, 7), facecolor="white",
                         sharey=True)

for k, ax in enumerate(axes):
    sub = df_hemi[df_hemi["cluster"] == k]
    y = np.arange(len(unique_networks))

    for j, net in enumerate(lollipop_order):
        net_sub = sub[sub["network"] == net]
        lh_row = net_sub[net_sub["hemi"] == "LH"]
        rh_row = net_sub[net_sub["hemi"] == "RH"]

        lh_val = lh_row["mean"].values[0] if len(lh_row) else 0.0
        rh_val = rh_row["mean"].values[0] if len(rh_row) else 0.0

        # Thin line connecting LH and RH dots
        ax.plot([lh_val, rh_val], [j, j],
                color="gray", linewidth=0.9, zorder=1)

        # Vertical reference line at zero
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--", zorder=0)

        # Dots
        ax.scatter(lh_val, j, color=hemi_colors["LH"], s=55, zorder=2,
                   label=hemi_labels_display["LH"] if j == 0 else "")
        ax.scatter(rh_val, j, color=hemi_colors["RH"], s=55, zorder=2,
                   label=hemi_labels_display["RH"] if j == 0 else "",
                   marker="D")  # diamond for RH to aid B&W printing

    ax.set_yticks(y)
    ax.set_yticklabels(lollipop_order, fontsize=10)
    ax.set_xlabel("Mean loading", fontsize=10)
    ax.set_title(cluster_titles[k], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.8, len(unique_networks) - 0.2)

    if k == 2:
        # Build legend manually to avoid duplicates
        lh_patch = plt.Line2D([0], [0], marker="o",
                              color="w", markerfacecolor=hemi_colors["LH"],
                              markersize=8, label="Left hemisphere")
        rh_patch = plt.Line2D([0], [0], marker="D",
                              color="w", markerfacecolor=hemi_colors["RH"],
                              markersize=8, label="Right hemisphere")
        ax.legend(handles=[lh_patch, rh_patch], fontsize=8, loc="upper right")

#plt.suptitle(
#    "Mean cluster loadings by Yeo-17 network and hemisphere",
#    fontsize=11,
#)
plt.tight_layout()
plt.savefig(f"{DIR}/yeo17_lollipop.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lollipop / dot plot")
