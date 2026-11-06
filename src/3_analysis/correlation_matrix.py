from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"
OUT = ROOT / "outputs" / "correlation"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED / "final_battles.csv")
X = df.drop(columns=["episode_id", "winner_p1", "winner_p2"])

corr = X.corr()

THRESHOLD = 0.80

print(f"Feature pairs with |correlation| >= {THRESHOLD}:")
print("-" * 50)

found = False
pairs = set()
for col_a in corr.columns:
    for col_b in corr.columns:
        if col_a >= col_b:          # skip self and duplicates
            continue
        val = corr.loc[col_a, col_b]
        if abs(val) >= THRESHOLD:
            pairs.add((col_a, col_b, round(val, 3)))
            found = True

if not found:
    print(f"None found above {THRESHOLD}.")
else:
    for col_a, col_b, val in sorted(pairs, key=lambda x: -abs(x[2])):
        flag = " <-- consider dropping one" if abs(val) >= 0.90 else ""
        print(f"  {col_a:25s}  {col_b:25s}  r = {val:+.3f}{flag}")

print()

print("Full correlation matrix:")
print(corr.round(2).to_string())
print()

if pairs:
    pair_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])
    pair_df = pair_df.sort_values("correlation", key=abs, ascending=False)
    pair_df.to_csv(OUT / "high_correlation_pairs.csv", index=False)
    print(f"High-correlation pairs saved to {OUT / 'high_correlation_pairs.csv'}")


fig, ax = plt.subplots(figsize=(12, 10))
mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        pass                        

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 7},
)
ax.set_title("Feature Correlation Matrix", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.savefig(OUT / "correlation_matrix.png", dpi=150, bbox_inches="tight")
print(f"Heatmap saved to {OUT / 'correlation_matrix.png'}")

plt.show()