"""Tests for the Zenodo tag addition utility."""

from unittest import mock

import pytest

from python_utilities import add_zenodo_tags
from python_utilities import zenodo_maintenance as zm


def make_deposition(tags, state="done", submitted=True, title="Example"):
    """Create the deposition fields used by tag update tests."""
    return {
        "metadata": {
            "title": title,
            "creators": [{"name": "TSML Archive Maintainers"}],
            "keywords": list(tags),
        },
        "state": state,
        "submitted": submitted,
    }


def test_validates_and_computes_added_tags():
    """Preserve existing order when already present; sort when adding new tags."""
    assert zm.clean_tags([" rehab ", "univariate"]) == ("rehab", "univariate")
    with pytest.raises(ValueError, match="Duplicate configured tag"):
        zm.clean_tags(["rehab", "rehab"])
    with pytest.raises(ValueError, match="non-empty string"):
        zm.clean_tags([" "])

    assert add_zenodo_tags.compute_added_tags(
        ("time series", "rehab"), ("rehab",)
    ) == ("time series", "rehab")

    assert add_zenodo_tags.compute_added_tags(
        ("time series",), ("rehab", "time series")
    ) == ("rehab", "time series")


def test_cli_preview_and_apply_add_tags(tmp_path, monkeypatch, capsys):
    """Run both read-only preview and confirmed apply via CLI arguments."""
    index_path = tmp_path / "current-versions"
    index_path.write_text("Sleep: 15526565\n", encoding="utf-8")
    dummy_token = tmp_path / "sandbox-token"
    dummy_token.write_text("sandbox-dummy-token\n", encoding="utf-8")

    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "Sleep",
        {"title": "Sleep", "keywords": ["classification", "time series"]},
    )
    write_client = mock.Mock()
    before = make_deposition(
        ["classification", "time series"], state="done", title="Sleep"
    )
    draft = make_deposition(
        ["classification", "time series"], state="inprogress", title="Sleep"
    )
    published = make_deposition(
        ["classification", "time series", "univariate"], state="done", title="Sleep"
    )
    write_client.request_json.side_effect = [before, draft, draft, draft, published]

    monkeypatch.setattr(
        zm,
        "ZenodoClient",
        lambda token=None, api_root=zm.PRODUCTION_API_ROOT: (
            write_client if token else read_client
        ),
    )

    assert (
        add_zenodo_tags.main(
            [
                "--record",
                "Sleep",
                "--record-index",
                str(index_path),
                "--tag",
                "univariate",
            ]
        )
        == 0
    )
    write_client.request_json.assert_not_called()
    assert '-> ["classification", "time series", "univariate"]' in capsys.readouterr().out

    assert (
        add_zenodo_tags.main(
            [
                "--record",
                "Sleep",
                "--record-index",
                str(index_path),
                "--tag",
                "univariate",
                "--sandbox",
                "--token-file",
                str(dummy_token),
                "--apply",
            ],
            input_fn=lambda _: "APPLY",
        )
        == 0
    )

    path = "deposit/depositions/15526565"
    assert write_client.request_json.call_args_list == [
        mock.call("GET", path),
        mock.call("POST", f"{path}/actions/edit"),
        mock.call("GET", path),
        mock.call(
            "PUT",
            path,
            {
                "metadata": {
                    "title": "Sleep",
                    "creators": [{"name": "TSML Archive Maintainers"}],
                    "keywords": ["classification", "time series", "univariate"],
                }
            },
        ),
        mock.call("POST", f"{path}/actions/publish"),
    ]
