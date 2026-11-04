import json
from pathlib import Path
from pprint import pprint

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

json_file = next(folder.glob("*.json"))

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total steps:", len(data["steps"]))

print("\nSTEP KEYS:")
print(data["steps"][0][0].keys())

print("\nOBSERVATION KEYS:")
pprint(list(data["steps"][0][0]["observation"].keys()))