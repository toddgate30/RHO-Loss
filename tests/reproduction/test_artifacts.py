import json
from pathlib import Path

import pytest

from reproduction.data import sha256
from reproduction.execute import artifact_for


def test_wrong_seed_or_changed_artifact_is_rejected(tmp_path):
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"fixture checkpoint")
    artifact = {
        "dataset": "cifar10",
        "irr_seed": 2,
        "recipe_sha256": "recipe",
        "checkpoint": str(model),
        "checkpoint_sha256": sha256(model),
    }
    (tmp_path / "artifact.json").write_text(json.dumps(artifact))
    manifest = {
        "dataset": "cifar10",
        "irr_seed": 1,
        "recipe_sha256": "recipe",
        "irr_dir": str(tmp_path),
        "loss_source": "checkpoint",
    }
    with pytest.raises(RuntimeError, match="irr_seed"):
        artifact_for(manifest)
    manifest["irr_seed"] = 2
    assert artifact_for(manifest)[1] == model
    model.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="corrupt"):
        artifact_for(manifest)
