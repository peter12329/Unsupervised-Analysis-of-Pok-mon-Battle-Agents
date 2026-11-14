import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

DROP = ["p1_active_switches", "p2_active_switches",
        "p1_energy_mean", "p2_energy_mean",
        "p1_bench_mean", "p2_bench_mean"]

PROCESSED = Path(__file__).resolve().parents[2] / "dataset" / "processed"
beh = pd.read_csv(PROCESSED / "behavior_features.csv")

X = beh.drop(columns=["episode_id"] + DROP)
X["n_steps"] = np.log1p(X["n_steps"])
Xs = RobustScaler().fit_transform(X)

print("k   inertia   silhouette (mean of 5 seeds)")
for k in range(2, 10):
    sils, inertia = [], []
    for seed in range(5):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
        sils.append(silhouette_score(Xs, km.labels_))
        inertia.append(km.inertia_)
    print(f"{k}   {np.mean(inertia):8.0f}   {np.mean(sils):.3f}  (±{np.std(sils):.3f})")

for k in (2, 3, 4):
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
    print(f"\nk={k} sizes:", np.bincount(labels))
    print(X.groupby(labels).mean().round(2).T)