import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

N_CLUSTERS = 6
DROP = ["p1_active_switches", "p2_active_switches",
        "p1_energy_mean", "p2_energy_mean",
        "p1_bench_mean", "p2_bench_mean"]

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"

final = pd.read_csv(PROCESSED / "final_battles.csv")
beh = pd.read_csv(PROCESSED / "behavior_features.csv")

final["episode_id"] = final["episode_id"].astype(str)
beh["episode_id"] = beh["episode_id"].astype(str)

df = final[["episode_id", "round", "winner_p1"]].merge(beh, on="episode_id")
print("Merged rows:", len(df), "(expected 1713)")

# round-2 only: winner_p1 is the true match winner here
df = df[df["round"] == 2].reset_index(drop=True)
print("Round-2 games:", len(df))

feature_cols = [c for c in beh.columns if c != "episode_id" and c not in DROP]
X = df[feature_cols].copy()
if "n_steps" in X:
    X["n_steps"] = np.log1p(X["n_steps"])

X_scaled = RobustScaler().fit_transform(X)
df["cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(X_scaled)

print("\nWinner split per cluster:")
print(pd.crosstab(df["cluster"], df["winner_p1"]))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
score = cross_val_score(rf, X, df["winner_p1"], cv=5).mean()
print(f"\nRandom forest accuracy: {score:.3f}  (chance is about 0.51)")

rf.fit(X, df["winner_p1"])
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False))