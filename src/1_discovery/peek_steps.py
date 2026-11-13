import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
files = [f for f in (ROOT / "dataset").rglob("*.json")]
print("JSON files found:", len(files))
f = files[0]
print("Using:", f)

data = json.load(open(f, encoding="utf-8"))
print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))

steps = data["steps"]
print("Number of steps:", len(steps))

def short(x, n=1500):
    return json.dumps(x, indent=1)[:n]

print("\n--- step 0 ---\n", short(steps[0]))
print("\n--- step 50 ---\n", short(steps[min(50, len(steps) - 1)]))
print("\n--- last step ---\n", short(steps[-1]))