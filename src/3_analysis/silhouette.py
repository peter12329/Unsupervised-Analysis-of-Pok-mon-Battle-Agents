from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "dataset" / "processed" / "final_battles.csv")

X = df.drop(columns=["episode_id", "winner_p1", "winner_p2"])
X_scaled = StandardScaler().fit_transform(X)

ks = range(2, 9)
scores = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print(f"k={k}  silhouette={score:.3f}")

plt.plot(list(ks), scores, marker="o")
plt.title("Silhouette Score by k")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "silhouette.png", dpi=150, bbox_inches="tight")

plt.show()