import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "processed" / "behavior_features.csv"


def snapshot(step):
    for entry in step or []:
        if not entry:
            continue
        cur = (entry.get("observation") or {}).get("current")
        if cur and len(cur.get("players") or []) == 2:
            return cur
    return None


def extract(path):
    data = json.load(open(path, encoding="utf-8"))
    steps = data.get("steps") or []
    n = len(steps)
    if n == 0:
        raise ValueError("no steps")
    row = {"episode_id": path.stem, "n_steps": n}

    snaps = [(i, snapshot(s)) for i, s in enumerate(steps)]
    snaps = [(i, c) for i, c in snaps if c]

    for p in (0, 1):
        tag = f"p{p + 1}"

        acts = sum(
            1 for s in steps
            if s and len(s) > p and s[p] and s[p].get("action")
        )
        row[f"{tag}_action_rate"] = acts / n

        bench_sizes, energy_totals, evo_max = [], [], 0
        bench_fill = 1.0
        switches, prev_active = 0, None

        for i, cur in snaps:
            pl = cur["players"][p] or {}
            bench = [m for m in (pl.get("bench") or []) if m]
            active = [m for m in (pl.get("active") or []) if m]
            mons = active + bench

            bench_sizes.append(len(bench))
            energy_totals.append(sum(len(m.get("energies") or []) for m in mons))
            evo_max = max(evo_max, sum(1 for m in mons if m.get("preEvolution")))

            if len(bench) >= 3 and bench_fill == 1.0:
                bench_fill = i / n

            serial = active[0].get("serial") if active else None
            if prev_active is not None and serial != prev_active:
                switches += 1
            prev_active = serial

        k = max(len(bench_sizes), 1)
        row[f"{tag}_bench_mean"] = sum(bench_sizes) / k
        row[f"{tag}_bench_fill_speed"] = bench_fill
        row[f"{tag}_energy_mean"] = sum(energy_totals) / k
        row[f"{tag}_evolutions_max"] = evo_max
        row[f"{tag}_active_switches"] = switches

    return row


files = sorted(RAW.glob("*.json"))
rows, skipped = [], 0
for f in files:
    try:
        rows.append(extract(f))
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            import traceback
            print("skip", f.name)
            traceback.print_exc()

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.shape)
print(df.describe().T)
print(f"Rows kept: {len(rows)} of {len(files)}  (skipped {skipped})")
print("Saved to", OUT)