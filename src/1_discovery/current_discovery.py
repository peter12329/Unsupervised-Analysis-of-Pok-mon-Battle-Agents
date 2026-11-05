import json
from pathlib import Path
from pprint import pprint

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

json_file = next(folder.glob("*.json"))

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

last_step = data["steps"][-1][0]

current = last_step["observation"]["current"]

print(type(current))

if current is not None:
    print("\nCURRENT KEYS:")
    pprint(list(current.keys()))
else:
    print("Current is still None.")