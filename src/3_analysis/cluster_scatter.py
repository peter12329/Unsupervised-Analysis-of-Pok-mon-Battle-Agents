from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "dataset" / "processed" / "clustered_battles.csv")

X = df.drop(columns=["episode_id", "winner_p1", "winner_p2", "cluster"])
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))

for c in sorted(df["cluster"].unique()):
    mask = df["cluster"] == c
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=18,
        alpha=0.7,
        label=f"Cluster {c} (n={mask.sum()})"
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Battle Clusters (PCA 2D View)")
ax.legend()

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "cluster_scatter.png", dpi=150, bbox_inches="tight")

plt.show()