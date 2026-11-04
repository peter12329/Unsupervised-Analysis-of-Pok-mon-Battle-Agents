import json
from pathlib import Path
from pprint import pprint

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

json_file = next(folder.glob("*.json"))

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

current = None

for step in reversed(data["steps"]):

    obs = step[0]["observation"]

    if obs["current"] is not None:
        current = obs["current"]
        break

print("Current keys:")
pprint(list(current.keys()))

print("\nPlayer 0:")
pprint(current["players"][0])
