"""Validate the battle clusters: are they stable, and are they just the winner in disguise?

Two checks on the clustering made by clustering.py:

1. STABILITY  Re-run k-means with other random seeds and on random 80% samples of the
              battles. If the clusters are real structure, the labels should mostly agree
              with the reference clustering (Adjusted Rand Index, ARI: 1 = identical,
              about 0 = no better than chance).

2. LEAKAGE    Some end-of-game columns can give the outcome away on their own (for example
              prize cards left). Such columns are found from the data (how well each one
              alone separates winner from loser, as an AUC), the clustering is repeated
              without them, and the two clusterings are compared. If the clusters only
              exist because of those columns, they describe who won, not how the battle went.

Run from anywhere:
    python cluster_validation.py
    python cluster_validation.py --k 6 --scaler robust --seeds 10 --boot 20

Reads   dataset/processed/final_battles.csv   (from extract_final_features.py)
Writes  outputs/validation/leakage_auc.csv, stability.csv, cluster_vs_winner.csv,
        validation.png, validation_report.md

The thresholds used for the verdict are rules of thumb, not statistical tests. Read the
numbers, not only the verdict line.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # save the figure, do not open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"
OUT = ROOT / "outputs" / "validation"

ID_AND_OUTCOME = ["episode_id", "winner_p1", "winner_p2"]
SCALERS = {"robust": RobustScaler, "standard": StandardScaler}
REFERENCE_SEED = 42                        # same as clustering.py


def load(path):
    """Return (features, winner_p1 as 0/1). Rows with missing values are dropped."""
    df = pd.read_csv(path)
    missing = [c for c in ID_AND_OUTCOME if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing values.")
    y = df["winner_p1"].astype(int)
    if y.nunique() < 2:
        raise SystemExit("winner_p1 has only one value, so leakage cannot be measured.")
    X = df.drop(columns=ID_AND_OUTCOME)
    constant = [c for c in X.columns if X[c].nunique() < 2]
    if constant:
        print(f"Ignoring constant columns: {constant}")
        X = X.drop(columns=constant)
    return X, y


def scale(X, name):
    return SCALERS[name]().fit_transform(X)


def kmeans(Xs, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)


def leakage_table(X, y):
    """How well each feature ALONE separates winner from loser.

    strength = |AUC - 0.5| * 2, from 0 (no information) to 1 (perfect). The direction
    does not matter, so a column that predicts a loss counts as much as one that
    predicts a win.
    """
    rows = []
    for col in X.columns:
        auc = roc_auc_score(y, X[col])
        rows.append({"feature": col, "auc": round(auc, 3), "strength": round(abs(auc - 0.5) * 2, 3)})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def stability(Xs, k, reference, seeds, n_boot, frac, rng_seed=0):
    """ARI of other seeds and of bootstrap samples against the reference labels."""
    rows = []
    for s in range(seeds):
        labels = kmeans(Xs, k, s).labels_
        rows.append({"check": "other seed", "ari": adjusted_rand_score(reference, labels)})
    rng = np.random.default_rng(rng_seed)
    n = len(Xs)
    for b in range(n_boot):
        idx = rng.choice(n, size=int(n * frac), replace=False)
        model = kmeans(Xs[idx], k, b)
        rows.append({"check": f"{int(frac * 100)}% sample", "ari": adjusted_rand_score(reference, model.predict(Xs))})
    return pd.DataFrame(rows)


def cluster_vs_winner(labels, y):
    """Size and player-1 win rate of each cluster, and how much the labels say about the winner."""
    table = (pd.DataFrame({"cluster": labels, "p1_won": y})
             .groupby("cluster")["p1_won"].agg(battles="size", p1_win_rate="mean").round(3))
    return table, normalized_mutual_info_score(y, labels)


def silhouette(Xs, labels):
    n = len(Xs)
    return silhouette_score(Xs, labels, sample_size=min(5000, n), random_state=0)


def verdict(nmi_full, nmi_clean, ari_ablation, ari_stab):
    lines = []
    if ari_stab >= 0.8:
        lines.append("Stable: other seeds and samples reproduce the clusters well.")
    elif ari_stab >= 0.5:
        lines.append("Moderately stable: the clusters are recognisable but the boundaries move.")
    else:
        lines.append("Unstable: the clusters change a lot between runs, so do not over-interpret them.")
    if nmi_full >= 0.3 and nmi_clean < 0.5 * nmi_full:
        lines.append("Outcome-driven: the clusters carry a lot of information about the winner, and most of "
                     "it disappears without the leaking columns.")
    elif nmi_full >= 0.3:
        lines.append("The clusters carry a lot of information about the winner even without the leaking columns.")
    else:
        lines.append("The clusters say little about the winner.")
    if ari_ablation < 0.5:
        lines.append("Dropping the leaking columns changes the clustering substantially (ARI below 0.5).")
    else:
        lines.append("The clustering survives dropping the leaking columns (ARI 0.5 or more).")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=PROCESSED / "final_battles.csv")
    ap.add_argument("--k", type=int, default=6, help="number of clusters (clustering.py uses 6)")
    ap.add_argument("--scaler", choices=SCALERS, default="robust", help="clustering.py uses robust")
    ap.add_argument("--seeds", type=int, default=10, help="other random seeds to try")
    ap.add_argument("--boot", type=int, default=20, help="random subsamples to try")
    ap.add_argument("--frac", type=float, default=0.8, help="share of battles in each subsample")
    ap.add_argument("--auc-strength", type=float, default=0.5,
                    help="a column whose |AUC-0.5|*2 is at least this is treated as leaking the outcome")
    ap.add_argument("--drop", nargs="*", default=None,
                    help="columns to treat as leaking instead of the automatic choice")
    args = ap.parse_args()
    if not 0 < args.frac < 1:
        raise SystemExit("--frac must be between 0 and 1")

    OUT.mkdir(parents=True, exist_ok=True)
    X, y = load(args.data)
    if args.k >= len(X):
        raise SystemExit("k must be smaller than the number of battles")
    print(f"{len(X)} battles, {X.shape[1]} features, scaler={args.scaler}, k={args.k}\n")

    # ---- leakage candidates, found from the data
    leak = leakage_table(X, y)
    leak.to_csv(OUT / "leakage_auc.csv", index=False)
    flagged = args.drop if args.drop is not None else leak.loc[leak["strength"] >= args.auc_strength, "feature"].tolist()
    unknown = [c for c in flagged if c not in X.columns]
    if unknown:
        raise SystemExit(f"--drop names columns that do not exist: {unknown}")
    print("Features that alone separate the winner (strength = |AUC-0.5|*2):")
    print(leak.head(10).to_string(index=False))
    print(f"\nTreated as leaking (strength >= {args.auc_strength}): {flagged or 'none'}\n")

    # ---- reference clustering (same recipe as clustering.py)
    Xs = scale(X, args.scaler)
    ref = kmeans(Xs, args.k, REFERENCE_SEED)
    ref_labels = ref.labels_

    # ---- 1. stability
    stab = stability(Xs, args.k, ref_labels, args.seeds, args.boot, args.frac)
    stab.to_csv(OUT / "stability.csv", index=False)
    summary = stab.groupby("check")["ari"].agg(["mean", "std", "min"]).round(3)
    print("Stability (ARI against the reference clustering):")
    print(summary.to_string(), "\n")
    ari_stab = float(stab["ari"].mean())

    # ---- 2. leakage ablation
    table_full, nmi_full = cluster_vs_winner(ref_labels, y)
    table_full.to_csv(OUT / "cluster_vs_winner.csv")
    print("Clusters vs winner (reference clustering):")
    print(table_full.to_string())
    print(f"Information the clusters carry about the winner (NMI): {nmi_full:.3f}\n")

    if flagged and len(flagged) < X.shape[1]:
        X_clean = X.drop(columns=flagged)
        Xs_clean = scale(X_clean, args.scaler)
        clean_labels = kmeans(Xs_clean, args.k, REFERENCE_SEED).labels_
        ari_ablation = adjusted_rand_score(ref_labels, clean_labels)
        nmi_clean = normalized_mutual_info_score(y, clean_labels)
        print(f"Without {len(flagged)} leaking column(s):")
        print(f"  agreement with the full clustering (ARI): {ari_ablation:.3f}")
        print(f"  information about the winner (NMI):       {nmi_clean:.3f}  (was {nmi_full:.3f})")
        print(f"  silhouette: {silhouette(Xs, ref_labels):.3f} -> {silhouette(Xs_clean, clean_labels):.3f} "
              "(computed in different feature spaces, so only roughly comparable)\n")
    else:
        ari_ablation, nmi_clean = 1.0, nmi_full
        print("No leaking columns to remove, so no ablation was run.\n")

    lines = verdict(nmi_full, nmi_clean, ari_ablation, ari_stab)
    print("Verdict (rules of thumb):")
    for line in lines:
        print("  -", line)

    # ---- figure
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    top = leak.head(12)[::-1]
    a.barh(top["feature"], top["strength"], color=["#d9534f" if f in flagged else "#6c8ebf" for f in top["feature"]])
    a.axvline(args.auc_strength, color="black", linestyle="--", linewidth=1)
    a.set_xlabel("How well the column alone separates the winner (0 to 1)")
    a.set_title("Leakage candidates (red = removed in the check)")
    groups = [stab.loc[stab["check"] == c, "ari"].to_numpy() for c in stab["check"].unique()]
    labels_ = list(stab["check"].unique())
    if flagged and len(flagged) < X.shape[1]:
        groups.append(np.array([ari_ablation])); labels_.append("without leaking columns")
    b.boxplot(groups, tick_labels=labels_)
    b.set_ylim(0, 1.02)
    b.set_ylabel("ARI vs reference clustering")
    b.set_title(f"Stability of the k={args.k} clustering")
    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=150)
    plt.close(fig)

    # ---- written report
    report = [f"# Cluster validation (k={args.k}, scaler={args.scaler}, {len(X)} battles)", "",
              "## Verdict (rules of thumb)", *[f"- {line}" for line in lines], "",
              "## Stability (ARI against the reference clustering)", "```", summary.to_string(), "```", "",
              "## Leakage candidates", "```", leak.head(10).to_string(index=False), "```",
              f"Treated as leaking: {flagged or 'none'}", "",
              "## Clusters vs winner", "```", table_full.to_string(), "```",
              f"NMI with the winner: {nmi_full:.3f} (without leaking columns: {nmi_clean:.3f})"]
    (OUT / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
