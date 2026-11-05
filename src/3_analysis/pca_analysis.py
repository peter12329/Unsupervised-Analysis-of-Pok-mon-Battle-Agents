import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("final_battles.csv")

# Remove identifiers and outcome columns
X = df.drop(
    columns=[
        "episode_id",
        "winner_p1",
        "winner_p2"
    ]
)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nCumulative Variance:")
print(pca.explained_variance_ratio_.cumsum())