import json
from pathlib import Path
from pprint import pprint

folder = Path(
    r"C:\Users\Test\Desktop\Unsupervised Analysis of Pokémon Battle Agents\dataset\archive"
)

# Open first JSON file
json_file = next(folder.glob("*.json"))

print("FILE:", json_file.name)

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTYPE:")
print(type(data))

print("\nKEYS:")
if isinstance(data, dict):
    print(list(data.keys()))

print("\nFIRST 3000 CHARS:")
pprint(str(data)[:3000])