import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from pathlib import Path

N_CLUSTERS = 6

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"

df = pd.read_csv(PROCESSED / "final_battles.csv")
X = df.drop(columns=["episode_id", "winner_p1", "winner_p2",
                      "p1_deckCount", "p2_deckCount",
                      "p1_prizes", "p2_prizes"])

X_scaled = RobustScaler().fit_transform(X)


kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"Scaler: RobustScaler   k = {N_CLUSTERS}")
print(df["cluster"].value_counts().sort_index())
print(f"Total battles: {len(df)}")


df.to_csv(PROCESSED / "clustered_battles.csv", index=False)
df.to_csv(PROCESSED / f"clustered_battles_k{N_CLUSTERS}.csv", index=False)

print(f"\nSaved to {PROCESSED / 'clustered_battles.csv'}")
print(f"Also saved clustered_battles_k{N_CLUSTERS}.csv")
