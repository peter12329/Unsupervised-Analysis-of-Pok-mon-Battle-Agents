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

        rows.append({
            "episode_id": data["info"]["EpisodeId"],
            "battle_length": len(data["steps"]),
            "player1_reward": data["rewards"][0],
            "player2_reward": data["rewards"][1]
        })

    except Exception as e:
        print(f"Error processing {file.name}: {e}")

df = pd.DataFrame(rows)

print(df.head())
print(df.shape)

df.to_csv("battles.csv", index=False)