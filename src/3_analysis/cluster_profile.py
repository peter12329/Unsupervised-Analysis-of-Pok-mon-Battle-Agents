from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"

df = pd.read_csv(PROCESSED / "clustered_battles.csv")

features = df.drop(columns=["episode_id", "winner_p1", "winner_p2", "cluster"])
X_scaled = pd.DataFrame(
    RobustScaler().fit_transform(features), columns=features.columns
)
X_scaled["cluster"] = df["cluster"].values

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)

# 1) PCA scatter
coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled[features.columns])
fig, ax = plt.subplots(figsize=(9, 7))
for c in sorted(df["cluster"].unique()):
    m = (df["cluster"] == c).values
    ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.7,
               label=f"Cluster {c} (n={m.sum()})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Battle Clusters (PCA 2D View)")
ax.legend()
plt.savefig(out_dir / "cluster_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# 2) Cluster profile heatmap (mean robust-scaled value per feature)
profile = X_scaled.groupby("cluster").mean()
sizes = df["cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(profile.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(profile.columns)))
ax.set_xticklabels(profile.columns, rotation=60, ha="right")
ax.set_yticks(range(len(profile.index)))
ax.set_yticklabels([f"Cluster {c} (n={sizes[c]})" for c in profile.index])
for i in range(profile.shape[0]):
    for j in range(profile.shape[1]):
        ax.text(j, i, f"{profile.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="Mean Robust-Scaled Feature Value")
ax.set_title("Cluster Profiles")
plt.savefig(out_dir / "cluster_profile.png", dpi=150, bbox_inches="tight")
plt.show()

# 3) Winner leakage check
print(pd.crosstab(df["cluster"], df["winner_p1"], normalize="index").round(2))
plt.show()