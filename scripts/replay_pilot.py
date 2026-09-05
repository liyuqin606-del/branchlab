"""Recompute search decisions from released tables without retraining models."""
import argparse
import json
from pathlib import Path
from branchlab.experiments import analyze
from branchlab.training import json_write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pilot", default="artifacts/pilot")
parser.add_argument("--output", default="artifacts/replay")
args = parser.parse_args()
source, out = Path(args.pilot), Path(args.output)
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Choose an empty replay directory")
summary = json.loads((source / "summary.json").read_text())
methods, gate = analyze(json.loads((source / "episodes.json").read_text()),
                        json.loads((source / "curves.json").read_text()),
                        summary["config"], out, summary["failures"])
assert methods == summary["methods"], "Replayed decisions differ from released results"
assert gate == summary["gate"], "Replayed gate differs from released gate"
json_write(out / "verification.json", {"decisions_match": True, "gate_match": True,
                                     "scope": "Analysis replay only; no physical collection savings"})
print("All released decisions and the gate reproduced exactly.")
