import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final_battles.csv")

X = df.drop(
    columns=[
        "episode_id",
        "winner_p1",
        "winner_p2"
    ]
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print(df["cluster"].value_counts())

df.to_csv("clustered_battles.csv", index=False)

print("\nSaved to clustered_battles.csv")