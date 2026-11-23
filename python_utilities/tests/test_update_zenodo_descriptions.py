"""Tests for the Zenodo description and related-identifier updater."""

from unittest import mock

import pytest

from python_utilities import update_zenodo_descriptions as desc
from python_utilities import zenodo_maintenance as zm


def make_deposition(description, related=None, state="done", title="Example"):
    """Build a synthetic Zenodo deposition payload."""
    metadata = {
        "title": title,
        "description": description,
        "creators": [{"name": "TSML Archive Maintainers"}],
        "keywords": ["time series"],
    }
    if related is not None:
        metadata["related_identifiers"] = related
    return {"metadata": metadata, "state": state, "submitted": True}


def test_validates_operations_and_ignores_synthetic_version_relations():
    """Normalize relations, block version relations, and make appends idempotent."""
    assert desc.normalize_relation("issupplementto") == "isSupplementTo"
    with pytest.raises(ValueError, match="managed automatically"):
        desc.normalize_relation("isVersionOf")
    with pytest.raises(ValueError, match="Unsupported relation"):
        desc.normalize_relation("invalidRelation")

    # Public /api/records/{id} includes synthetic isVersionOf; extract_user_related_identifiers ignores it
    public_md = {
        "related_identifiers": [
            {"identifier": "10.5281/zenodo.100", "relation": "isVersionOf", "scheme": "doi"},
            {"identifier": "10.1000/paper", "relation": "isSupplementTo", "scheme": "doi"},
        ]
    }
    assert desc.extract_user_related_identifiers(public_md, "101") == (
        ("10.1000/paper", "isSupplementTo"),
    )

    # Idempotent append
    once = desc.compute_updated_description(
        "<p>Base.</p>", append_blocks=("<p>Variant note.</p>",)
    )
    twice = desc.compute_updated_description(
        once, append_blocks=("<p>Variant note.</p>",)
    )
    assert once == twice == "<p>Base.</p>\n<p>Variant note.</p>"


def test_cli_replaces_description_and_adds_related_identifier(tmp_path, monkeypatch):
    """Preview and apply a description substring fix + related paper DOI."""
    index_path = tmp_path / "current-versions"
    index_path.write_text("SyntheticArchive: 18300287\n", encoding="utf-8")
    dummy_token = tmp_path / "sandbox-token"
    dummy_token.write_text("sandbox-dummy-token\n", encoding="utf-8")

    old_desc = "<p>224 datasets processed to equal length.</p>"
    new_desc = "<p>224 datasets truncated to unequal lengths.</p>"

    # Public API includes synthetic isVersionOf, while deposition draft does not
    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "SyntheticArchive",
        {
            "title": "SyntheticArchive",
            "description": old_desc,
            "related_identifiers": [
                {"identifier": "10.5281/zenodo.18300286", "relation": "isVersionOf"}
            ],
        },
    )

    write_client = mock.Mock()
    before = make_deposition(old_desc, state="done", title="SyntheticArchive")
    draft = make_deposition(old_desc, state="inprogress", title="SyntheticArchive")
    published = make_deposition(
        new_desc,
        related=[{"identifier": "10.1007/s10618-024-01022-1", "relation": "cites"}],
        state="done",
        title="SyntheticArchive",
    )
    write_client.request_json.side_effect = [before, draft, draft, draft, published]

    monkeypatch.setattr(
        zm,
        "ZenodoClient",
        lambda token=None, api_root=zm.PRODUCTION_API_ROOT: (
            write_client if token else read_client
        ),
    )

    result = desc.main(
        [
            "--record",
            "SyntheticArchive",
            "--record-index",
            str(index_path),
            "--replace",
            "processed to equal length",
            "truncated to unequal lengths",
            "--add-related",
            "10.1007/s10618-024-01022-1",
            "cites",
            "--sandbox",
            "--token-file",
            str(dummy_token),
            "--apply",
        ],
        input_fn=lambda _: "APPLY",
    )

    assert result == 0
    path = "deposit/depositions/18300287"
    assert write_client.request_json.call_args_list[3] == mock.call(
        "PUT",
        path,
        {
            "metadata": {
                "title": "SyntheticArchive",
                "description": new_desc,
                "creators": [{"name": "TSML Archive Maintainers"}],
                "keywords": ["time series"],
                "related_identifiers": [
                    {"identifier": "10.1007/s10618-024-01022-1", "relation": "cites"}
                ],
            }
        },
    )
