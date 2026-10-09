import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from greek_nlp.core import artifacts


@pytest.mark.parametrize(
    ("name", "repo_env", "repo_type"),
    [
        ("greek-bert-news-classifier", "HF_MODEL_REPO", "model"),
        ("rag_index", "HF_INDEX_REPO", "dataset"),
    ],
)
def test_download_returns_path_and_reuses_local_folder(
    tmp_path, monkeypatch, name, repo_env, repo_type
):
    monkeypatch.setattr(artifacts, "STORE", tmp_path)
    monkeypatch.setenv(repo_env, "example/test-repo")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    local = tmp_path / name

    def fake_download(**kwargs):
        kwargs["local_dir"].mkdir(parents=True)
        return str(kwargs["local_dir"])

    download = Mock(side_effect=fake_download)
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert artifacts.ensure_artifact(name, repo_env) == local
    download.assert_called_once_with(
        repo_id="example/test-repo",
        repo_type=repo_type,
        local_dir=local,
        token=None,
    )

    assert artifacts.ensure_artifact(name, repo_env) == local
    assert download.call_count == 1
