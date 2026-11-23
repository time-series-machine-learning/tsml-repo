"""Tests for the Zenodo tag removal utility."""

from unittest import mock

from python_utilities import remove_zenodo_tags
from python_utilities import zenodo_maintenance as zm


def make_deposition(tags, state="done", submitted=True, title="Example"):
    """Create the deposition fields used by tag removal tests."""
    return {
        "metadata": {
            "title": title,
            "creators": [{"name": "TSML Archive Maintainers"}],
            "keywords": list(tags),
        },
        "state": state,
        "submitted": submitted,
    }


def test_removes_only_requested_tags_preserving_remaining_order():
    """Keep remaining tags in their original order."""
    assert remove_zenodo_tags.compute_removed_tags(
        ("time series", "multiverse-core", "classification"),
        ("multiverse-core",),
    ) == ("time series", "classification")


def test_cli_apply_removes_tags_and_preserves_other_metadata(tmp_path, monkeypatch):
    """Unlock, remove requested tag, update, and republish cleanly."""
    index_path = tmp_path / "current-versions"
    index_path.write_text("BenzeneConcentration_disc: 23060100\n", encoding="utf-8")
    dummy_token = tmp_path / "sandbox-token"
    dummy_token.write_text("sandbox-dummy-token\n", encoding="utf-8")

    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "BenzeneConcentration_disc",
        {
            "title": "BenzeneConcentration_disc",
            "keywords": ["multiverse", "multiverse-core", "time series"],
        },
    )
    write_client = mock.Mock()
    before = make_deposition(
        ["multiverse", "multiverse-core", "time series"],
        state="done",
        title="BenzeneConcentration_disc",
    )
    draft = make_deposition(
        ["multiverse", "multiverse-core", "time series"],
        state="inprogress",
        title="BenzeneConcentration_disc",
    )
    published = make_deposition(
        ["multiverse", "time series"],
        state="done",
        title="BenzeneConcentration_disc",
    )
    write_client.request_json.side_effect = [before, draft, draft, draft, published]

    monkeypatch.setattr(
        zm,
        "ZenodoClient",
        lambda token=None, api_root=zm.PRODUCTION_API_ROOT: (
            write_client if token else read_client
        ),
    )

    result = remove_zenodo_tags.main(
        [
            "--record",
            "BenzeneConcentration_disc",
            "--record-index",
            str(index_path),
            "--tag",
            "multiverse-core",
            "--sandbox",
            "--token-file",
            str(dummy_token),
            "--apply",
        ],
        input_fn=lambda _: "APPLY",
    )

    assert result == 0
    path = "deposit/depositions/23060100"
    assert write_client.request_json.call_args_list[3] == mock.call(
        "PUT",
        path,
        {
            "metadata": {
                "title": "BenzeneConcentration_disc",
                "creators": [{"name": "TSML Archive Maintainers"}],
                "keywords": ["multiverse", "time series"],
            }
        },
    )
