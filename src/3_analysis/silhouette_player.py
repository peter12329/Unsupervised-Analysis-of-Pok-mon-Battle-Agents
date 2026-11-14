import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

PROCESSED = Path(__file__).resolve().parents[2] / "dataset" / "processed"
beh = pd.read_csv(PROCESSED / "behavior_features.csv")

parts = []
for tag in ("p1", "p2"):
    d = pd.DataFrame({
        "episode_id": beh["episode_id"],
        "seat": tag,
        "n_steps": np.log1p(beh["n_steps"]),
        "action_rate": beh[f"{tag}_action_rate"],
        "bench_fill_speed": beh[f"{tag}_bench_fill_speed"],
        "evolutions_max": beh[f"{tag}_evolutions_max"],
    })
    parts.append(d)
P = pd.concat(parts, ignore_index=True)

never = P["bench_fill_speed"] >= 0.99
print("Players:", len(P), "| never filled bench:", never.sum())

feats = ["n_steps", "action_rate", "bench_fill_speed", "evolutions_max"]
P_main = P[~never].reset_index(drop=True)
Xs = RobustScaler().fit_transform(P_main[feats])

print("\nk   inertia   silhouette (excluding never-filled-bench)")
for k in range(2, 10):
    sils, inertia = [], []
    for seed in range(5):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
        sils.append(silhouette_score(Xs, km.labels_))
        inertia.append(km.inertia_)
    print(f"{k}   {np.mean(inertia):8.0f}   {np.mean(sils):.3f}  (±{np.std(sils):.3f})")