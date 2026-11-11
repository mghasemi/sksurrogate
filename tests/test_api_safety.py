"""Path-traversal guards for caller-supplied storage segments.

Before :mod:`api.safety`, values such as ``model_version`` and ``partition``
were joined onto the storage root verbatim: ``{"model_version": "../../evil"}``
resolved outside the bundles directory, and ``joblib.load`` would execute any
``*.bundle`` reachable on the host; a traversal ``partition`` read arbitrary
``*.csv`` files. These tests pin the guards at both layers (helper + HTTP).
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from api.config import settings
from api.main import app
from api.routers.bundles import _bundle_file
from api.safety import UnsafeName, contained_path, safe_segment
from SKSurrogate import ModelBundle, save_bundle


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Patch the storage root before the tests touch any path."""
    root = tmp_path_factory.mktemp("api-safety")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "root", root)
    settings.ensure()
    yield TestClient(app)
    patcher.undo()


@pytest.fixture()
def demo_bundle():
    """Register a real bundle for task ``demo`` under the patched root."""
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    model = LinearRegression().fit(frame, [0.0, 1.0, 2.0, 3.0])
    bundle = ModelBundle(model, task_name="demo", schema={"a": {"dtype": "float64"}})
    path = settings.task_bundles_dir("demo") / (bundle.model_version + ".bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, path)
    return bundle


def test_safe_segment_rejects_traversal_and_controls():
    for bad in ("..", "../x", "a/b", "a\\b", "", "   ", "a\x00b", "a\r\nb"):
        with pytest.raises(UnsafeName):
            safe_segment(bad, "field")
    assert safe_segment("model-1.2_ok", "field") == "model-1.2_ok"


def test_contained_path_blocks_escape_and_allows_plain_names(tmp_path):
    with pytest.raises(UnsafeName):
        contained_path(tmp_path, "../../etc/passwd", ".csv")
    assert contained_path(tmp_path, "train", ".csv") == tmp_path / "train.csv"


def test_settings_task_paths_reject_traversal():
    for bad in ("..", "../x", "a/b"):
        with pytest.raises(UnsafeName):
            settings.task_dataset_dir(bad)
        with pytest.raises(UnsafeName):
            settings.task_bundles_dir(bad)


def test_predict_rejects_traversing_model_version(client):
    res = client.post(
        "/api/inference/demo/predict",
        json={"model_version": "../../evil", "rows": [{"a": 1.0}]},
    )
    assert res.status_code == 400
    assert "model version" in res.json()["detail"]


def test_predict_rejects_traversing_partition(client, demo_bundle):
    res = client.post(
        "/api/inference/demo/predict",
        json={
            "model_version": demo_bundle.model_version,
            "partition": "../../../etc/passwd",
        },
    )
    assert res.status_code == 400
    assert "partition" in res.json()["detail"]


def test_bundle_lookup_rejects_traversing_version(client):
    with pytest.raises(UnsafeName):
        _bundle_file("demo", "../../evil")


def test_bundle_outside_task_directory_cannot_be_loaded(client):
    """A real bundle one level above the task directory stays unreachable."""
    frame = pd.DataFrame({"a": [0.0, 1.0]})
    model = LinearRegression().fit(frame, [0.0, 1.0])
    outside = ModelBundle(model, task_name="demo")
    save_bundle(outside, settings.root / "evil.bundle")

    res = client.post(
        "/api/inference/demo/predict",
        json={"model_version": "../../evil", "rows": [{"a": 1.0}]},
    )
    assert res.status_code == 400
    # The rejection happens on the name, before any joblib load of that file.
    assert "model version" in res.json()["detail"]
