"""Verify the explicit local handoff before opening the independent dataset."""

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL = PROJECT_ROOT / "data/EE4D_MISMATCH_READY.json"


def read_handoff(signal=DEFAULT_SIGNAL):
    signal = Path(signal)
    value = json.loads(signal.read_text())
    if value.get("status") != "complete" or not value.get("datasets"):
        raise ValueError("Dataset handoff is not complete.")
    if value.get("tests", {}).get("failed") != 0:
        raise ValueError("Dataset handoff does not declare passing tests.")
    for item in value["datasets"]:
        root = Path(item["path"])
        audit = json.loads((root / "audit/validation.json").read_text())
        if audit.get("status") != "passed" or audit.get("source_hashes_verified") is not True:
            raise ValueError(f"Dataset audit is not complete: {root}")
        for file, key in ((root / "spec.json", "spec_sha256"), (Path(item["manifest"]), "manifest_sha256")):
            if hashlib.sha256(file.read_bytes()).hexdigest() != item[key]:
                raise ValueError(f"Handoff hash mismatch: {file}")
    return value


def open_dataset(*, profile="pilot", signal=DEFAULT_SIGNAL):
    handoff = read_handoff(signal)
    entries = [entry for entry in handoff["datasets"] if entry["profile"] == profile]
    if len(entries) != 1:
        raise ValueError(f"Expected one dataset for profile {profile!r}.")
    # The dataset producer is a separate sibling package, not part of UEM.
    project = Path(handoff["code_root"]).parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from data_pipeline.ee4d_mismatch.dataset import MismatchDataset

    return MismatchDataset(entries[0]["path"]), handoff
