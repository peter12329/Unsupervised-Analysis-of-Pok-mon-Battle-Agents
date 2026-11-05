import json
import pandas as pd
from pathlib import Path

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

rows = []

for file in folder.glob("*.json"):

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        current = None

        # Find the LAST valid state
        for step in reversed(data["steps"]):

            obs = step[0]["observation"]

            if obs["current"] is not None:
                current = obs["current"]
                break

        if current is None:
            continue

        p1 = current["players"][0]
        p2 = current["players"][1]

        rows.append({
            "episode_id": data["info"]["EpisodeId"],

            "battle_length": len(data["steps"]),

            "round": current["round"],
            "turn": current["turn"],
            "turnActionCount": current["turnActionCount"],

            "p1_deckCount": p1["deckCount"],
            "p2_deckCount": p2["deckCount"],

            "p1_handCount": p1["handCount"],
            "p2_handCount": p2["handCount"],

            "p1_bench": len(p1["bench"]),
            "p2_bench": len(p2["bench"]),

            "p1_discard": len(p1["discard"]),
            "p2_discard": len(p2["discard"]),

            "p1_active": len(p1["active"]),
            "p2_active": len(p2["active"]),

            "p1_prizes": len(p1["prize"]),
            "p2_prizes": len(p2["prize"]),

            "winner_p1": p1["win"],
            "winner_p2": p2["win"]
        })

    except Exception as e:
        print(f"Error in {file.name}: {e}")

df = pd.DataFrame(rows)

print(df.head())
print(df.shape)

df.to_csv("final_battles.csv", index=False)