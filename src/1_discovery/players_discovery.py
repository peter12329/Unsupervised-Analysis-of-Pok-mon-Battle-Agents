import json
from pathlib import Path
from pprint import pprint

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

json_file = next(folder.glob("*.json"))

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for step in data["steps"]:

    current = step[0]["observation"]["current"]

    if current is not None:

        players = current["players"]

        print("Number of players:", len(players))

        print("\nPlayer 0 keys:")
        pprint(list(players[0].keys()))

        break