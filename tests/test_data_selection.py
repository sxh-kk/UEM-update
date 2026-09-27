"""A growing handoff must not silently change a frozen experiment dataset."""

import pytest

from egorecover.data import select_handoff_dataset


def test_duplicate_pilot_profiles_require_the_exact_frozen_spec():
    handoff = {
        "datasets": [
            {"profile": "pilot", "spec_sha256": "frozen", "path": "old"},
            {"profile": "pilot", "spec_sha256": "new", "path": "new"},
        ]
    }
    assert select_handoff_dataset(handoff, profile="pilot", spec_sha256="frozen")["path"] == "old"
    with pytest.raises(ValueError, match="Expected one dataset"):
        select_handoff_dataset(handoff, profile="pilot")
    with pytest.raises(ValueError, match="Expected one dataset"):
        select_handoff_dataset(handoff, profile="pilot", spec_sha256="missing")
