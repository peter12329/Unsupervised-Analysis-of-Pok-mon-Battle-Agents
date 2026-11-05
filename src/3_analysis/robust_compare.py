from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "dataset" / "processed" / "final_battles.csv")
X = df.drop(columns=["episode_id", "winner_p1", "winner_p2"])

variants = {
    "standard": StandardScaler().fit_transform(X),
    "robust": RobustScaler().fit_transform(X),
    "log1p+standard": StandardScaler().fit_transform(np.log1p(X)),
}

# Sanity check: robust scaling can blow up columns with a tiny IQR
print("Max |value| per column after RobustScaler:")
print(pd.DataFrame(variants["robust"], columns=X.columns).abs().max().round(1))

seeds = [0, 1, 2, 3, 4]
for name, Xs in variants.items():
    print(f"\n{name}")
    for k in range(2, 9):
        sils, smallest = [], []
        for s in seeds:
            labels = KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(Xs)
            sils.append(silhouette_score(Xs, labels))
            smallest.append(np.bincount(labels).min())
        print(f"k={k}  silhouette={np.mean(sils):.3f} (±{np.std(sils):.3f})  "
              f"smallest cluster≈{int(np.mean(smallest))}")