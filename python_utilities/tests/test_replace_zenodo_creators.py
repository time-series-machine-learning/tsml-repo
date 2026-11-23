"""Tests for the Zenodo creator replacement utility."""

from unittest import mock

import pytest

from python_utilities import replace_zenodo_creators
from python_utilities import zenodo_maintenance as zm


def make_deposition(creators, state="done", submitted=True, title="Example"):
    """Create the deposition fields used by creator replacement tests."""
    return {
        "metadata": {
            "title": title,
            "creators": creators,
            "keywords": ["time series"],
        },
        "state": state,
        "submitted": submitted,
    }


def test_cleans_and_extracts_creator_names_from_both_metadata_formats():
    """Normalize names and support both legacy and InvenioRDM creator dicts."""
    names = replace_zenodo_creators.clean_creator_names(
        [" TSML Archive Maintainers ", "Second Creator"]
    )
    assert names == ("TSML Archive Maintainers", "Second Creator")

    with pytest.raises(ValueError, match="Duplicate creator"):
        replace_zenodo_creators.clean_creator_names(["Same", "Same"])

    invenio = {
        "creators": [
            {"person_or_org": {"name": "TSML Archive Maintainers", "type": "organizational"}}
        ]
    }
    legacy = {"creators": [{"name": "TSML Archive Maintainers"}]}

    assert replace_zenodo_creators.extract_creator_names(invenio, "123") == (
        "TSML Archive Maintainers",
    )
    assert replace_zenodo_creators.extract_creator_names(legacy, "123") == (
        "TSML Archive Maintainers",
    )


def test_cli_replaces_creators_and_preserves_other_metadata(tmp_path, monkeypatch):
    """Unlock, replace only creators, update, and republish cleanly."""
    targets_path = tmp_path / "targets"
    targets_path.write_text("Example: 123\n", encoding="utf-8")
    dummy_token = tmp_path / "sandbox-token"
    dummy_token.write_text("sandbox-dummy-token\n", encoding="utf-8")

    old_creators = [{"name": "Original", "orcid": "0000-0001-2345-6789"}]
    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "Example",
        {"title": "Example", "creators": old_creators, "keywords": ["time series"]},
    )
    write_client = mock.Mock()
    before = make_deposition(old_creators, state="done")
    draft = make_deposition(old_creators, state="inprogress")
    published = make_deposition([{"name": "TSML Archive Maintainers"}], state="done")
    write_client.request_json.side_effect = [before, draft, draft, draft, published]

    monkeypatch.setattr(
        zm,
        "ZenodoClient",
        lambda token=None, api_root=zm.PRODUCTION_API_ROOT: (
            write_client if token else read_client
        ),
    )

    result = replace_zenodo_creators.main(
        [
            str(targets_path),
            "--creator",
            "TSML Archive Maintainers",
            "--sandbox",
            "--token-file",
            str(dummy_token),
            "--apply",
        ],
        input_fn=lambda _: "APPLY",
    )

    assert result == 0
    path = "deposit/depositions/123"
    assert write_client.request_json.call_args_list == [
        mock.call("GET", path),
        mock.call("POST", f"{path}/actions/edit"),
        mock.call("GET", path),
        mock.call(
            "PUT",
            path,
            {
                "metadata": {
                    "title": "Example",
                    "creators": [{"name": "TSML Archive Maintainers"}],
                    "keywords": ["time series"],
                }
            },
        ),
        mock.call("POST", f"{path}/actions/publish"),
    ]
    assert before["metadata"]["creators"] == old_creators
